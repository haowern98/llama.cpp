#include "mtmd-video.h"

#include "ggml.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifdef LLAMA_MTMD_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
}
#endif

namespace {

constexpr double   MTMD_VIDEO_DEFAULT_FPS        = 2.0;
constexpr uint32_t MTMD_VIDEO_FRAME_FACTOR       = 2;
constexpr uint32_t MTMD_VIDEO_DEFAULT_MIN_FRAMES = 4;
constexpr uint32_t MTMD_VIDEO_DEFAULT_MAX_FRAMES = 768;
constexpr int      MTMD_VIDEO_AVIO_BUFFER_SIZE   = 4096;

static mtmd_helper_media_options get_options_or_default(const mtmd_helper_media_options * options) {
    return options ? *options : mtmd_helper_media_options_default();
}

static uint32_t round_by_factor(uint32_t value, uint32_t factor) {
    return (uint32_t) std::llround((double) value / factor) * factor;
}

static uint32_t ceil_by_factor(uint32_t value, uint32_t factor) {
    return ((value + factor - 1) / factor) * factor;
}

static uint32_t floor_by_factor(uint32_t value, uint32_t factor) {
    return (value / factor) * factor;
}

static bool calculate_frame_range(
        int total_frames,
        double video_fps,
        const mtmd_helper_media_options & options,
        int & start_frame,
        int & end_frame,
        int & range_frames) {
    if (total_frames < 2 || video_fps <= 0.0) {
        return false;
    }

    start_frame = 0;
    end_frame = total_frames - 1;

    if (options.video_start >= 0.0 || options.video_end >= 0.0) {
        const double max_duration = total_frames / video_fps;
        if (options.video_start >= 0.0) {
            const double clamped = std::max(0.0, std::min(options.video_start, max_duration));
            start_frame = (int) std::ceil(clamped * video_fps);
        }
        if (options.video_end >= 0.0) {
            const double clamped = std::max(0.0, std::min(options.video_end, max_duration));
            end_frame = std::min((int) std::floor(clamped * video_fps), total_frames - 1);
        }
    }

    if (start_frame >= end_frame) {
        return false;
    }

    range_frames = end_frame - start_frame + 1;
    return range_frames >= 2;
}

static bool smart_nframes(
        int total_frames,
        double video_fps,
        const mtmd_helper_media_options & options,
        uint32_t & out_nframes) {
    if (total_frames < (int) MTMD_VIDEO_FRAME_FACTOR || video_fps <= 0.0) {
        return false;
    }

    if (options.video_fps > 0.0 && options.video_nframes > 0) {
        return false;
    }

    uint32_t nframes = 0;
    if (options.video_nframes > 0) {
        nframes = round_by_factor((uint32_t) options.video_nframes, MTMD_VIDEO_FRAME_FACTOR);
    } else {
        const double fps = options.video_fps > 0.0 ? options.video_fps : MTMD_VIDEO_DEFAULT_FPS;
        const uint32_t min_frames = ceil_by_factor(
            options.video_min_frames > 0 ? (uint32_t) options.video_min_frames : MTMD_VIDEO_DEFAULT_MIN_FRAMES,
            MTMD_VIDEO_FRAME_FACTOR);
        const uint32_t requested_max = options.video_max_frames > 0
            ? (uint32_t) options.video_max_frames
            : std::min<uint32_t>(MTMD_VIDEO_DEFAULT_MAX_FRAMES, (uint32_t) total_frames);
        const uint32_t max_frames = floor_by_factor(
            std::min<uint32_t>(requested_max, (uint32_t) total_frames),
            MTMD_VIDEO_FRAME_FACTOR);
        double approx = total_frames / video_fps * fps;
        approx = std::min<double>(std::max<double>(approx, min_frames), max_frames);
        approx = std::min<double>(approx, total_frames);
        nframes = floor_by_factor((uint32_t) std::floor(approx), MTMD_VIDEO_FRAME_FACTOR);
    }

    if (nframes < MTMD_VIDEO_FRAME_FACTOR) {
        nframes = MTMD_VIDEO_FRAME_FACTOR;
    }
    if (nframes > (uint32_t) total_frames) {
        nframes = floor_by_factor((uint32_t) total_frames, MTMD_VIDEO_FRAME_FACTOR);
    }

    if (!(MTMD_VIDEO_FRAME_FACTOR <= nframes && nframes <= (uint32_t) total_frames)) {
        return false;
    }

    out_nframes = nframes;
    return true;
}

static std::vector<int> make_sample_indices(int start_frame, int end_frame, uint32_t nframes) {
    std::vector<int> indices;
    indices.reserve(nframes);
    if (nframes == 0) {
        return indices;
    }
    if (nframes == 1) {
        indices.push_back(start_frame);
        return indices;
    }

    const double span = (double) (end_frame - start_frame);
    for (uint32_t i = 0; i < nframes; ++i) {
        const double alpha = (double) i / (double) (nframes - 1);
        indices.push_back((int) std::llround(start_frame + span * alpha));
    }
    return indices;
}

#ifdef LLAMA_MTMD_FFMPEG

struct av_format_deleter {
    void operator()(AVFormatContext * fmt) const {
        if (fmt) {
            avformat_close_input(&fmt);
        }
    }
};

struct av_codec_deleter {
    void operator()(AVCodecContext * ctx) const {
        if (ctx) {
            avcodec_free_context(&ctx);
        }
    }
};

struct av_frame_deleter {
    void operator()(AVFrame * frame) const {
        if (frame) {
            av_frame_free(&frame);
        }
    }
};

struct av_packet_deleter {
    void operator()(AVPacket * pkt) const {
        if (pkt) {
            av_packet_free(&pkt);
        }
    }
};

struct sws_deleter {
    void operator()(SwsContext * sws) const {
        if (sws) {
            sws_freeContext(sws);
        }
    }
};

static double measure_luma_delta(
        const AVFrame * frame,
        const std::vector<uint8_t> & prev_luma,
        int step_y,
        int step_x) {
    if (!frame || frame->width <= 0 || frame->height <= 0 || !frame->data[0] || prev_luma.empty()) {
        return 0.0;
    }

    const int width = frame->width;
    const int height = frame->height;
    const int linesize = frame->linesize[0];
    double total = 0.0;
    size_t count = 0;

    for (int y = 0; y < height; y += step_y) {
        const uint8_t * row = frame->data[0] + (size_t) y * linesize;
        const uint8_t * prev_row = prev_luma.data() + (size_t) y * width;
        for (int x = 0; x < width; x += step_x) {
            total += std::abs((int) row[x] - (int) prev_row[x]);
            ++count;
        }
    }

    return count > 0 ? total / (double) count : 0.0;
}

static bool make_motion_aware_sample_indices(
        AVFormatContext * fmt,
        int stream_index,
        int start_frame,
        int end_frame,
        uint32_t nframes,
        std::vector<int> & out_indices) {
    out_indices.clear();
    if (!fmt || nframes == 0 || start_frame >= end_frame) {
        return false;
    }

    const AVCodec * codec = avcodec_find_decoder(fmt->streams[stream_index]->codecpar->codec_id);
    if (!codec) {
        return false;
    }

    std::unique_ptr<AVCodecContext, av_codec_deleter> codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) {
        return false;
    }
    if (avcodec_parameters_to_context(codec_ctx.get(), fmt->streams[stream_index]->codecpar) < 0) {
        return false;
    }
    if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
        return false;
    }

    std::unique_ptr<AVFrame, av_frame_deleter> frame(av_frame_alloc());
    std::unique_ptr<AVPacket, av_packet_deleter> packet(av_packet_alloc());
    if (!frame || !packet) {
        return false;
    }

    std::vector<double> cumulative_motion;
    cumulative_motion.reserve((size_t) (end_frame - start_frame + 1));

    std::vector<uint8_t> prev_luma;
    prev_luma.reserve(4096);
    int current_frame = 0;
    bool have_prev = false;
    int width = 0;
    int height = 0;
    int step_x = 4;
    int step_y = 4;

    auto receive_frames = [&](bool flush_only) -> bool {
        while (true) {
            const int ret = avcodec_receive_frame(codec_ctx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return true;
            }
            if (ret < 0) {
                return false;
            }

            if (current_frame >= start_frame && current_frame <= end_frame) {
                if (width == 0 || height == 0) {
                    width = frame->width;
                    height = frame->height;
                    step_x = std::max(1, width / 160);
                    step_y = std::max(1, height / 160);
                    prev_luma.resize((size_t) width * height);
                }

                double motion = 1.0;
                if (have_prev) {
                    motion += measure_luma_delta(frame.get(), prev_luma, step_y, step_x);
                }

                cumulative_motion.push_back(
                    (cumulative_motion.empty() ? 0.0 : cumulative_motion.back()) + motion);

                for (int y = 0; y < height; ++y) {
                    std::memcpy(
                        prev_luma.data() + (size_t) y * width,
                        frame->data[0] + (size_t) y * frame->linesize[0],
                        (size_t) width);
                }
                have_prev = true;
            }

            ++current_frame;
            av_frame_unref(frame.get());

            if (!flush_only && current_frame > end_frame) {
                return true;
            }
        }
    };

    while (av_read_frame(fmt, packet.get()) >= 0) {
        if (packet->stream_index == stream_index) {
            if (avcodec_send_packet(codec_ctx.get(), packet.get()) < 0) {
                av_packet_unref(packet.get());
                return false;
            }
            if (!receive_frames(false)) {
                av_packet_unref(packet.get());
                return false;
            }
        }
        av_packet_unref(packet.get());
        if (current_frame > end_frame) {
            break;
        }
    }

    if (current_frame <= end_frame) {
        if (avcodec_send_packet(codec_ctx.get(), nullptr) < 0) {
            return false;
        }
        if (!receive_frames(true)) {
            return false;
        }
    }

    const int range_frames = end_frame - start_frame + 1;
    if ((int) cumulative_motion.size() != range_frames) {
        return false;
    }

    out_indices = make_sample_indices(start_frame, end_frame, nframes);
    if (nframes <= 2) {
        return true;
    }

    const double total_motion = cumulative_motion.back();
    if (total_motion <= 0.0) {
        return true;
    }

    out_indices.front() = start_frame;
    out_indices.back() = end_frame;
    for (uint32_t i = 1; i + 1 < nframes; ++i) {
        const double target = total_motion * ((double) i / (double) (nframes - 1));
        const auto it = std::lower_bound(cumulative_motion.begin(), cumulative_motion.end(), target);
        const int idx = it == cumulative_motion.end()
            ? range_frames - 1
            : (int) std::distance(cumulative_motion.begin(), it);
        out_indices[i] = start_frame + idx;
    }

    for (size_t i = 1; i < out_indices.size(); ++i) {
        const int min_allowed = out_indices[i - 1] + 1;
        const int max_allowed = end_frame - (int) (out_indices.size() - 1 - i);
        out_indices[i] = std::max(min_allowed, std::min(out_indices[i], max_allowed));
    }

    return true;
}

struct buffer_data {
    const unsigned char * data = nullptr;
    size_t size = 0;
    size_t pos = 0;
};

static int avio_read_packet(void * opaque, uint8_t * buf, int buf_size) {
    auto * bd = static_cast<buffer_data *>(opaque);
    if (!bd || !bd->data) {
        return AVERROR(EIO);
    }
    if (bd->pos >= bd->size) {
        return AVERROR_EOF;
    }
    const size_t remaining = bd->size - bd->pos;
    const int to_copy = (int) std::min<size_t>(remaining, (size_t) buf_size);
    std::memcpy(buf, bd->data + bd->pos, to_copy);
    bd->pos += (size_t) to_copy;
    return to_copy;
}

static int64_t avio_seek_packet(void * opaque, int64_t offset, int whence) {
    auto * bd = static_cast<buffer_data *>(opaque);
    if (!bd) {
        return -1;
    }
    if (whence == AVSEEK_SIZE) {
        return (int64_t) bd->size;
    }

    size_t new_pos = bd->pos;
    switch (whence) {
        case SEEK_SET:
            if (offset < 0 || (uint64_t) offset > bd->size) {
                return -1;
            }
            new_pos = (size_t) offset;
            break;
        case SEEK_CUR:
            if (offset < 0 && (uint64_t) (-offset) > bd->pos) {
                return -1;
            }
            new_pos = (size_t) ((int64_t) bd->pos + offset);
            if (new_pos > bd->size) {
                return -1;
            }
            break;
        case SEEK_END:
            if (offset > 0 || (uint64_t) (-offset) > bd->size) {
                return -1;
            }
            new_pos = (size_t) ((int64_t) bd->size + offset);
            break;
        default:
            return -1;
    }

    bd->pos = new_pos;
    return (int64_t) bd->pos;
}

static bool open_input_from_buffer(
        const unsigned char * buf,
        size_t len,
        AVFormatContext * & fmt_out,
        AVIOContext * & avio_out) {
    AVFormatContext * raw_fmt = avformat_alloc_context();
    if (!raw_fmt) {
        return false;
    }
    fmt_out = raw_fmt;

    auto * bd = new buffer_data();
    bd->data = buf;
    bd->size = len;

    unsigned char * avio_buf = (unsigned char *) av_malloc(MTMD_VIDEO_AVIO_BUFFER_SIZE);
    if (!avio_buf) {
        delete bd;
        avformat_free_context(raw_fmt);
        fmt_out = nullptr;
        return false;
    }

    avio_out = avio_alloc_context(
        avio_buf,
        MTMD_VIDEO_AVIO_BUFFER_SIZE,
        0,
        bd,
        avio_read_packet,
        nullptr,
        avio_seek_packet);
    if (!avio_out) {
        av_free(avio_buf);
        delete bd;
        avformat_free_context(raw_fmt);
        fmt_out = nullptr;
        return false;
    }

    fmt_out->pb = avio_out;
    fmt_out->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&fmt_out, nullptr, nullptr, nullptr) < 0) {
        delete bd;
        avio_context_free(&avio_out);
        avformat_free_context(raw_fmt);
        fmt_out = nullptr;
        return false;
    }
    return true;
}

static void close_input_from_buffer(AVFormatContext * & fmt, AVIOContext * & avio) {
    if (fmt) {
        avformat_close_input(&fmt);
    }
    if (avio) {
        if (avio->opaque) {
            delete static_cast<buffer_data *>(avio->opaque);
            avio->opaque = nullptr;
        }
        avio_context_free(&avio);
    }
}

static bool should_reject_demuxer(const char * demuxer_name) {
    if (!demuxer_name) {
        return false;
    }
    static const char * image_demuxers[] = {
        "image2",
        "image2pipe",
        "jpeg_pipe",
        "png_pipe",
        "bmp_pipe",
        "gif_pipe",
        "webp_pipe",
        "tiff_pipe",
        "mjpeg",
    };
    for (const char * name : image_demuxers) {
        if (std::strstr(demuxer_name, name) != nullptr) {
            return true;
        }
    }
    return false;
}

static bool inspect_video_stream(AVFormatContext * fmt, int & stream_index, double & fps, int & total_frames) {
    if (!fmt) {
        return false;
    }
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        return false;
    }
    if (fmt->iformat && should_reject_demuxer(fmt->iformat->name)) {
        return false;
    }

    stream_index = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (stream_index < 0) {
        return false;
    }

    AVStream * st = fmt->streams[stream_index];
    if (st->avg_frame_rate.num > 0 && st->avg_frame_rate.den > 0) {
        fps = av_q2d(st->avg_frame_rate);
    } else if (st->r_frame_rate.num > 0 && st->r_frame_rate.den > 0) {
        fps = av_q2d(st->r_frame_rate);
    } else {
        fps = 0.0;
    }

    if (st->nb_frames > 0) {
        total_frames = (int) st->nb_frames;
    } else if (fmt->duration > 0 && fps > 0.0) {
        total_frames = (int) std::llround((fmt->duration / (double) AV_TIME_BASE) * fps);
    } else {
        total_frames = 0;
    }

    return fps > 0.0 && total_frames >= 2;
}

static bool decode_selected_frames(
        AVFormatContext * fmt,
        int stream_index,
        const std::vector<int> & sample_indices,
        std::vector<unsigned char> & out_rgb,
        uint32_t & out_w,
        uint32_t & out_h,
        uint32_t & out_nframes) {
    if (sample_indices.empty()) {
        return false;
    }
    const AVCodec * codec = avcodec_find_decoder(fmt->streams[stream_index]->codecpar->codec_id);
    if (!codec) {
        return false;
    }

    std::unique_ptr<AVCodecContext, av_codec_deleter> codec_ctx(avcodec_alloc_context3(codec));
    if (!codec_ctx) {
        return false;
    }
    if (avcodec_parameters_to_context(codec_ctx.get(), fmt->streams[stream_index]->codecpar) < 0) {
        return false;
    }
    if (avcodec_open2(codec_ctx.get(), codec, nullptr) < 0) {
        return false;
    }

    std::unique_ptr<AVFrame, av_frame_deleter> frame(av_frame_alloc());
    std::unique_ptr<AVPacket, av_packet_deleter> packet(av_packet_alloc());
    if (!frame || !packet) {
        return false;
    }

    std::unique_ptr<SwsContext, sws_deleter> sws;
    size_t next_sample = 0;
    int current_frame = 0;
    std::vector<unsigned char> frame_rgb;

    auto receive_frames = [&](bool flush_only) -> bool {
        while (true) {
            const int ret = avcodec_receive_frame(codec_ctx.get(), frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                return true;
            }
            if (ret < 0) {
                return false;
            }

            if (!sws) {
                out_w = (uint32_t) frame->width;
                out_h = (uint32_t) frame->height;
                frame_rgb.resize((size_t) out_w * out_h * 3);
                sws.reset(sws_getContext(
                    frame->width,
                    frame->height,
                    (AVPixelFormat) frame->format,
                    frame->width,
                    frame->height,
                    AV_PIX_FMT_RGB24,
                    SWS_BILINEAR,
                    nullptr,
                    nullptr,
                    nullptr));
                if (!sws) {
                    return false;
                }
            }

            if (next_sample < sample_indices.size() && current_frame == sample_indices[next_sample]) {
                uint8_t * dst_data[4] = { frame_rgb.data(), nullptr, nullptr, nullptr };
                int dst_linesize[4] = { (int) out_w * 3, 0, 0, 0 };
                sws_scale(sws.get(), frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);

                while (next_sample < sample_indices.size() && current_frame == sample_indices[next_sample]) {
                    out_rgb.insert(out_rgb.end(), frame_rgb.begin(), frame_rgb.end());
                    ++next_sample;
                }
            }

            ++current_frame;
            av_frame_unref(frame.get());

            if (next_sample >= sample_indices.size()) {
                return true;
            }
            if (!flush_only && current_frame > sample_indices.back()) {
                return true;
            }
        }
    };

    while (next_sample < sample_indices.size() && av_read_frame(fmt, packet.get()) >= 0) {
        if (packet->stream_index == stream_index) {
            if (avcodec_send_packet(codec_ctx.get(), packet.get()) < 0) {
                av_packet_unref(packet.get());
                return false;
            }
            if (!receive_frames(false)) {
                av_packet_unref(packet.get());
                return false;
            }
        }
        av_packet_unref(packet.get());
    }

    if (next_sample < sample_indices.size()) {
        if (avcodec_send_packet(codec_ctx.get(), nullptr) < 0) {
            return false;
        }
        if (!receive_frames(true)) {
            return false;
        }
    }

    const size_t frame_bytes = (size_t) out_w * out_h * 3;
    out_nframes = frame_bytes == 0 ? 0 : (uint32_t) (out_rgb.size() / frame_bytes);
    if (out_nframes >= 1 && (out_nframes % MTMD_VIDEO_FRAME_FACTOR) != 0) {
        out_rgb.insert(out_rgb.end(), out_rgb.end() - frame_bytes, out_rgb.end());
        ++out_nframes;
    }
    return out_nframes >= MTMD_VIDEO_FRAME_FACTOR;
}

#endif

} // namespace

mtmd_helper_media_options mtmd_helper_media_options_default(void) {
    return mtmd_helper_media_options {
        /* media_type       */ MTMD_HELPER_MEDIA_TYPE_AUTO,
        /* video_fps        */ -1.0,
        /* video_nframes    */ -1,
        /* video_min_frames */ -1,
        /* video_max_frames */ -1,
        /* video_start      */ -1.0,
        /* video_end        */ -1.0,
    };
}

bool mtmd_video_is_video_buffer(const unsigned char * buf, size_t len) {
#ifdef LLAMA_MTMD_FFMPEG
    AVFormatContext * fmt = nullptr;
    AVIOContext * avio = nullptr;
    if (!open_input_from_buffer(buf, len, fmt, avio)) {
        return false;
    }

    int stream_index = -1;
    double fps = 0.0;
    int total_frames = 0;
    const bool ok = inspect_video_stream(fmt, stream_index, fps, total_frames);
    close_input_from_buffer(fmt, avio);
    return ok;
#else
    GGML_UNUSED(buf);
    GGML_UNUSED(len);
    return false;
#endif
}

mtmd_bitmap * mtmd_video_bitmap_init_from_buf(
        mtmd_context * ctx,
        const unsigned char * buf,
        size_t len,
        const mtmd_helper_media_options * options) {
#ifdef LLAMA_MTMD_FFMPEG
    GGML_UNUSED(ctx);
    const auto resolved = get_options_or_default(options);

    AVFormatContext * fmt = nullptr;
    AVIOContext * avio = nullptr;
    if (!open_input_from_buffer(buf, len, fmt, avio)) {
        return nullptr;
    }

    int stream_index = -1;
    double fps = 0.0;
    int total_frames = 0;
    if (!inspect_video_stream(fmt, stream_index, fps, total_frames)) {
        close_input_from_buffer(fmt, avio);
        return nullptr;
    }

    int start_frame = 0;
    int end_frame = 0;
    int range_frames = 0;
    uint32_t nframes = 0;
    if (!calculate_frame_range(total_frames, fps, resolved, start_frame, end_frame, range_frames)
            || !smart_nframes(range_frames, fps, resolved, nframes)) {
        close_input_from_buffer(fmt, avio);
        return nullptr;
    }

    std::vector<int> sample_indices;
    if (!make_motion_aware_sample_indices(fmt, stream_index, start_frame, end_frame, nframes, sample_indices)) {
        sample_indices = make_sample_indices(start_frame, end_frame, nframes);
    }
    close_input_from_buffer(fmt, avio);

    fmt = nullptr;
    avio = nullptr;
    if (!open_input_from_buffer(buf, len, fmt, avio)) {
        return nullptr;
    }
    if (!inspect_video_stream(fmt, stream_index, fps, total_frames)) {
        close_input_from_buffer(fmt, avio);
        return nullptr;
    }

    std::vector<unsigned char> rgb_frames;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t out_frames = 0;
    const bool ok = decode_selected_frames(fmt, stream_index, sample_indices, rgb_frames, width, height, out_frames);
    close_input_from_buffer(fmt, avio);
    if (!ok) {
        return nullptr;
    }

    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_video(width, height, out_frames, rgb_frames.data());
    if (bitmap) {
        std::vector<int32_t> sample_indices32(sample_indices.begin(), sample_indices.end());
        mtmd_bitmap_set_video_metadata(bitmap, fps, sample_indices32.data(), sample_indices32.size());
    }
    return bitmap;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(buf);
    GGML_UNUSED(len);
    GGML_UNUSED(options);
    return nullptr;
#endif
}

mtmd_bitmap * mtmd_video_bitmap_init_from_file(
        mtmd_context * ctx,
        const char * fname,
        const mtmd_helper_media_options * options) {
#ifdef LLAMA_MTMD_FFMPEG
    GGML_UNUSED(ctx);
    const auto resolved = get_options_or_default(options);

    std::unique_ptr<AVFormatContext, av_format_deleter> fmt;
    AVFormatContext * raw_fmt = nullptr;
    if (avformat_open_input(&raw_fmt, fname, nullptr, nullptr) < 0) {
        return nullptr;
    }
    fmt.reset(raw_fmt);

    int stream_index = -1;
    double fps = 0.0;
    int total_frames = 0;
    if (!inspect_video_stream(fmt.get(), stream_index, fps, total_frames)) {
        return nullptr;
    }

    int start_frame = 0;
    int end_frame = 0;
    int range_frames = 0;
    uint32_t nframes = 0;
    if (!calculate_frame_range(total_frames, fps, resolved, start_frame, end_frame, range_frames)
            || !smart_nframes(range_frames, fps, resolved, nframes)) {
        return nullptr;
    }

    std::vector<int> sample_indices;
    if (!make_motion_aware_sample_indices(fmt.get(), stream_index, start_frame, end_frame, nframes, sample_indices)) {
        sample_indices = make_sample_indices(start_frame, end_frame, nframes);
    }

    fmt.reset();
    AVFormatContext * raw_fmt2 = nullptr;
    if (avformat_open_input(&raw_fmt2, fname, nullptr, nullptr) < 0) {
        return nullptr;
    }
    fmt.reset(raw_fmt2);
    if (!inspect_video_stream(fmt.get(), stream_index, fps, total_frames)) {
        return nullptr;
    }

    std::vector<unsigned char> rgb_frames;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t out_frames = 0;
    if (!decode_selected_frames(fmt.get(), stream_index, sample_indices, rgb_frames, width, height, out_frames)) {
        return nullptr;
    }

    mtmd_bitmap * bitmap = mtmd_bitmap_init_from_video(width, height, out_frames, rgb_frames.data());
    if (bitmap) {
        std::vector<int32_t> sample_indices32(sample_indices.begin(), sample_indices.end());
        mtmd_bitmap_set_video_metadata(bitmap, fps, sample_indices32.data(), sample_indices32.size());
    }
    return bitmap;
#else
    GGML_UNUSED(ctx);
    GGML_UNUSED(fname);
    GGML_UNUSED(options);
    return nullptr;
#endif
}
