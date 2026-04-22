#include "clip.h"
#include "clip-impl.h"
#include "mtmd.h"
#include "mtmd-audio.h"
#include "mtmd-image.h"
#include "mtmd-video.h"
#include "debug/mtmd-debug.h"

#include "llama.h"

// fix problem with std::min and std::max
#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// represents raw image data, layout is RGBRGBRGB...
// length of data must be nx * ny * 3
struct mtmd_bitmap {
    uint32_t nx;
    uint32_t ny;
    uint32_t n_frames = 0; // 0 for image/audio, >= 2 for video
    std::vector<unsigned char> data;
    std::string id; // optional user-defined id, for ex: can be set to image hash, useful for KV cache tracking
    bool is_audio = false; // true if the bitmap is audio
    double video_fps = 0.0; // original video fps used to derive timestamps
    std::vector<int32_t> video_frame_indices; // sampled frame indices before temporal merge
};

struct mtmd_image_tokens {
    uint32_t nx; // number of tokens in x direction
    uint32_t ny; // number of tokens in y direction
    uint32_t nt = 1; // number of tokens in temporal direction
    bool use_mrope_pos = false; // use M-RoPE position counting (the whole image is 1 temporal position)
    bool split_video_temporal_pos = false; // qwen3.5-vl splits video into repeated 2D vision segments
    uint32_t video_slice_index = 0; // zero-based slice index inside the parent video
    uint32_t video_slice_count = 0; // total number of slices in the parent video
    int32_t video_frame_start = -1; // source frame index of the first frame in the slice
    int32_t video_frame_end   = -1; // source frame index of the last frame in the slice
    double video_timestamp_seconds = -1.0; // midpoint timestamp of the slice
    std::string video_parent_id; // parent logical video id for batched encode/cache
    uint32_t n_tokens() const { return nt * nx * ny; }
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_image_tokens clone() {
        return mtmd_image_tokens{
            nx,
            ny,
            nt,
            use_mrope_pos,
            split_video_temporal_pos,
            video_slice_index,
            video_slice_count,
            video_frame_start,
            video_frame_end,
            video_timestamp_seconds,
            video_parent_id,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_image_tokens_ptr = std::unique_ptr<mtmd_image_tokens>;

struct mtmd_audio_tokens {
    uint32_t n_tokens; // number of tokens
    clip_image_f32_batch batch_f32; // preprocessed image patches
    std::string id; // optional user-defined ID, useful for KV cache tracking

    mtmd_audio_tokens clone() {
        return mtmd_audio_tokens{
            n_tokens,
            batch_f32.clone(),
            id
        };
    }
};
using mtmd_audio_tokens_ptr = std::unique_ptr<mtmd_audio_tokens>;

struct mtmd_input_chunk {
    mtmd_input_chunk_type type;
    std::vector<llama_token> tokens_text;
    mtmd_image_tokens_ptr tokens_image;
    mtmd_audio_tokens_ptr tokens_audio;
};

struct mtmd_input_chunks {
    std::vector<mtmd_input_chunk> entries;
};

// slice template, used by some llava-uhd models to correctly place the special tokens around image embeddings
// models not having it (llava-1.6) will process embeddings without any special tokens in-between
enum mtmd_slice_tmpl {
    MTMD_SLICE_TMPL_NONE,
    MTMD_SLICE_TMPL_MINICPMV_2_5,
    MTMD_SLICE_TMPL_MINICPMV_2_6,
    MTMD_SLICE_TMPL_LLAMA4,
    MTMD_SLICE_TMPL_IDEFICS3,
    MTMD_SLICE_TMPL_LFM2,
    MTMD_SLICE_TMPL_STEP3VL,
};

const char * mtmd_default_marker() {
    return "<__media__>";
}

static clip_flash_attn_type mtmd_get_clip_flash_attn_type(enum llama_flash_attn_type flash_attn_type) {
    switch (flash_attn_type) {
        case LLAMA_FLASH_ATTN_TYPE_AUTO:     return CLIP_FLASH_ATTN_TYPE_AUTO;
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return CLIP_FLASH_ATTN_TYPE_DISABLED;
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return CLIP_FLASH_ATTN_TYPE_ENABLED;
    }
    return CLIP_FLASH_ATTN_TYPE_AUTO;
}

mtmd_context_params mtmd_context_params_default() {
    mtmd_context_params params {
        /* use_gpu           */ true,
        /* print_timings     */ true,
        /* n_threads         */ 4,
        /* image_marker      */ nullptr,
        /* media_marker      */ mtmd_default_marker(),
        /* flash_attn_type   */ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /* warmup            */ true,
        /* image_min_tokens  */ -1,
        /* image_max_tokens  */ -1,
        /* cb_eval           */ nullptr,
        /* cb_eval_user_data */ nullptr,
    };
    return params;
}

struct mtmd_context {
    struct clip_ctx * ctx_v; // vision
    struct clip_ctx * ctx_a; // audio
    const struct llama_model * text_model;
    std::vector<float> image_embd_v; // image embedding vector

    bool print_timings;
    int n_threads;
    std::string media_marker;
    const int n_embd_text;

    // these are not token, but strings used to mark the beginning and end of image/audio embeddings
    std::string img_beg;
    std::string img_end;
    std::string aud_beg;
    std::string aud_end;
    std::string img_placeholder;
    std::string video_placeholder;

    // for llava-uhd style models, we need special tokens in-between slices
    // minicpmv calls them "slices", llama 4 calls them "tiles"
    mtmd_slice_tmpl slice_tmpl = MTMD_SLICE_TMPL_NONE;
    std::vector<llama_token> tok_ov_img_start;  // overview image
    std::vector<llama_token> tok_ov_img_end;    // overview image
    std::vector<llama_token> tok_slices_start;  // start of all slices
    std::vector<llama_token> tok_slices_end;    // end of all slices
    std::vector<llama_token> tok_sli_img_start; // single slice start
    std::vector<llama_token> tok_sli_img_end;   // single slice end
    std::vector<llama_token> tok_sli_img_mid;   // between 2 slices
    std::vector<llama_token> tok_row_end;       // end of row
    bool tok_row_end_trail = false;
    bool ov_img_first      = false;

    // string template for slice image delimiters with row/col (idefics3)
    std::string sli_img_start_tmpl;

    std::unique_ptr<mtmd_audio_preprocessor> audio_preproc;
    std::unique_ptr<mtmd_image_preprocessor> image_preproc;

    struct video_encode_cache_entry {
        std::string parent_id;
        size_t n_slices = 0;
        size_t n_tokens_per_slice = 0;
        std::vector<float> embd;

        void clear() {
            parent_id.clear();
            n_slices = 0;
            n_tokens_per_slice = 0;
            embd.clear();
        }

        bool matches(const mtmd_image_tokens * image_tokens) const {
            return !parent_id.empty()
                && image_tokens != nullptr
                && image_tokens->video_slice_count > 0
                && image_tokens->video_parent_id == parent_id
                && image_tokens->batch_f32.entries.size() == n_slices
                && image_tokens->n_tokens() == n_tokens_per_slice;
        }
    };

    video_encode_cache_entry video_encode_cache;

    // TODO @ngxson : add timings

    mtmd_context(const char * mmproj_fname,
                   const llama_model * text_model,
                   const mtmd_context_params & ctx_params) :
        text_model   (text_model),
        print_timings(ctx_params.print_timings),
        n_threads    (ctx_params.n_threads),
        media_marker (ctx_params.media_marker),
        n_embd_text  (llama_model_n_embd_inp(text_model))
    {
        if (ctx_params.image_marker != nullptr) {
            throw std::runtime_error("custom image_marker is not supported anymore, use media_marker instead");
        }

        if (media_marker.empty()) {
            throw std::runtime_error("media_marker must not be empty");
        }

        clip_context_params ctx_clip_params {
            /* use_gpu           */ ctx_params.use_gpu,
            /* flash_attn_type   */ mtmd_get_clip_flash_attn_type(ctx_params.flash_attn_type),
            /* image_min_tokens  */ ctx_params.image_min_tokens,
            /* image_max_tokens  */ ctx_params.image_max_tokens,
            /* warmup            */ ctx_params.warmup,
            /* cb_eval           */ ctx_params.cb_eval,
            /* cb_eval_user_data */ ctx_params.cb_eval_user_data,
        };

        auto res = clip_init(mmproj_fname, ctx_clip_params);
        ctx_v = res.ctx_v;
        ctx_a = res.ctx_a;
        if (!ctx_v && !ctx_a) {
            throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
        }

        // if both vision and audio mmproj are present, we need to validate their n_embd
        if (ctx_v && ctx_a) {
            int n_embd_v = clip_n_mmproj_embd(ctx_v);
            int n_embd_a = clip_n_mmproj_embd(ctx_a);
            if (n_embd_v != n_embd_a) {
                throw std::runtime_error(string_format(
                    "mismatch between vision and audio mmproj (n_embd_v = %d, n_embd_a = %d)\n",
                    n_embd_v, n_embd_a));
            }
        }

        // since we already validate n_embd of vision and audio mmproj,
        // we can safely assume that they are the same
        int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
        if (n_embd_text != n_embd_clip) {
            throw std::runtime_error(string_format(
                "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
                "hint: you may be using wrong mmproj\n",
                n_embd_text, n_embd_clip));
        }
        if (ctx_v) {
            init_vision();
        }
        if (ctx_a) {
            init_audio();
        }
    }

    void init_vision() {
        GGML_ASSERT(ctx_v != nullptr);
        image_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_v);

        switch (proj) {
            case PROJECTOR_TYPE_MLP:
            case PROJECTOR_TYPE_MLP_NORM:
            case PROJECTOR_TYPE_LDP:
            case PROJECTOR_TYPE_LDPV2:
            case PROJECTOR_TYPE_COGVLM:
            case PROJECTOR_TYPE_JANUS_PRO:
            case PROJECTOR_TYPE_GLM_EDGE:
                {
                    bool has_pinpoints = !clip_get_hparams(ctx_v)->image_res_candidates.empty();
                    if (has_pinpoints) {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                    } else {
                        image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                    }
                } break;
            case PROJECTOR_TYPE_MINICPMV:
                {
                    int minicpmv_version = clip_is_minicpmv(ctx_v);
                    if (minicpmv_version == 2) {
                        // minicpmv 2.5 format:
                        // <image> (overview) </image><slice><image> (slice) </image><image> (slice) </image>\n ... </slice>
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_5;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_slices_start  = {lookup_token("<slice>")};
                        tok_slices_end    = {lookup_token("</slice>")};
                        tok_sli_img_start = tok_ov_img_start;
                        tok_sli_img_end   = tok_ov_img_end;
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;
                    } else if (minicpmv_version == 3 || minicpmv_version == 4 || minicpmv_version == 5 || minicpmv_version == 6 || minicpmv_version == 100045) {
                        // minicpmv 2.6 format:
                        // <image> (overview) </image><slice> (slice) </slice><slice> (slice) </slice>\n ...
                        slice_tmpl        = MTMD_SLICE_TMPL_MINICPMV_2_6;
                        tok_ov_img_start  = {lookup_token("<image>")};
                        tok_ov_img_end    = {lookup_token("</image>")};
                        tok_sli_img_start = {lookup_token("<slice>")};
                        tok_sli_img_end   = {lookup_token("</slice>")};
                        tok_row_end       = {lookup_token("\n")};
                        tok_row_end_trail = false; // no trailing end-of-row token
                        ov_img_first      = true;

                    } else if (minicpmv_version != 0) {
                        throw std::runtime_error(string_format("unsupported minicpmv version: %d\n", minicpmv_version));
                    }
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_QWEN2VL:
            case PROJECTOR_TYPE_QWEN25VL:
            case PROJECTOR_TYPE_QWEN3VL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    img_placeholder = img_beg + "<|image_pad|>" + img_end;
                    video_placeholder = img_beg + "<|video_pad|>" + img_end;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_YOUTUVL:
                {
                    // <|vision_start|> ... (image embeddings) ... <|vision_end|>
                    img_beg = "<|vision_start|>";
                    img_end = "<|vision_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_youtuvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA3:
            case PROJECTOR_TYPE_GEMMA3NV:
                {
                    // <start_of_image> ... (image embeddings) ... <end_of_image>
                    img_beg = "<start_of_image>";
                    img_end = "<end_of_image>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_IDEFICS3:
                {
                    // https://github.com/huggingface/transformers/blob/a42ba80fa520c784c8f11a973ca9034e5f859b79/src/transformers/models/idefics3/processing_idefics3.py#L192-L215
                    slice_tmpl         = MTMD_SLICE_TMPL_IDEFICS3;
                    tok_ov_img_start   = {lookup_token("\n\n"), lookup_token("<fake_token_around_image>"), lookup_token("<global-img>")};
                    tok_ov_img_end     = {lookup_token("<fake_token_around_image>")};
                    tok_row_end        = {lookup_token("\n")};
                    sli_img_start_tmpl = "<fake_token_around_image><row_%d_col_%d>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_idefics3>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PIXTRAL:
                {
                    // https://github.com/huggingface/transformers/blob/1cd110c6cb6a6237614130c470e9a902dbc1a4bd/docs/source/en/model_doc/pixtral.md
                    img_end = "[IMG_END]";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PHI4:
                {
                    // Phi-4 uses media marker insertion only. Keep image boundary text empty.
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LLAMA4:
                {
                    // (more details in mtmd_context constructor)
                    img_beg = "<|image_start|>";
                    img_end = "<|image_end|>";
                    LOG_WRN("%s: llama 4 vision is known to have degraded quality:\n"
                            "    https://github.com/ggml-org/llama.cpp/pull/13282\n", __func__);
                    image_preproc = std::make_unique<mtmd_image_preprocessor_llava_uhd>(ctx_v);
                } break;
            case PROJECTOR_TYPE_STEP3VL:
                {
                    // Step3 format:
                    //   <patch_start> (patch) <patch_end> [<patch_newline>]
                    //   ... (all patch rows)
                    //   <im_start> (overview) <im_end>
                    slice_tmpl        = MTMD_SLICE_TMPL_STEP3VL;
                    tok_ov_img_start  = {lookup_token("<im_start>")};
                    tok_ov_img_end    = {lookup_token("<im_end>")};
                    tok_sli_img_start = {lookup_token("<patch_start>")};
                    tok_sli_img_end   = {lookup_token("<patch_end>")};
                    tok_row_end       = {lookup_token("<patch_newline>")};
                    tok_row_end_trail = false;
                    ov_img_first      = false; // patches first, overview last
                    image_preproc = std::make_unique<mtmd_image_preprocessor_step3vl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_INTERNVL:
                {
                    // <img> ... (image embeddings) ... </img>
                    img_beg = "<img>";
                    img_end = "</img>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_internvl>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIVL:
                {
                    // <|media_start|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_start|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_KIMIK25:
                {
                    // <|media_begin|> ... (image embeddings) ... <|media_end|>
                    img_beg = "<|media_begin|>";
                    img_end = "<|media_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LIGHTONOCR:
                {
                    // <|im_start|> ... (image embeddings) ... <|im_end|>
                    img_beg = "<|im_start|>";
                    img_end = "<|im_end|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_longest_edge>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DOTS_OCR:
                {
                    // <|img|> ... (image embeddings) ... <|endofimg|>
                    img_beg = "<|img|>";
                    img_end = "<|endofimg|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_NEMOTRON_V2_VL:
                {
                    image_preproc = std::make_unique<mtmd_image_preprocessor_fixed_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_LFM2:
                {
                    // multi-tile:
                    //   <|image_start|>
                    //     <|img_row_1_col_1|> (tile) <|img_row_1_col_2|> (tile) ...
                    //     <|img_thumbnail|> (thumbnail)
                    //   <|image_end|>
                    // single-tile:
                    //   <|image_start|> (image) <|image_end|>
                    img_beg            = "<|image_start|>";
                    img_end            = "<|image_end|>";
                    slice_tmpl         = MTMD_SLICE_TMPL_LFM2;
                    sli_img_start_tmpl = "<|img_row_%d_col_%d|>";
                    tok_ov_img_start   = {lookup_token("<|img_thumbnail|>")};
                    ov_img_first       = false;
                    image_preproc = std::make_unique<mtmd_image_preprocessor_lfm2>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GLM4V:
                {
                    // <|begin_of_image|> ... (image embeddings) ... <|end_of_image|>
                    img_beg = "<|begin_of_image|>";
                    img_end = "<|end_of_image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_PADDLEOCR:
                {
                    // <|IMAGE_START|> ... (image embeddings) ... <|IMAGE_END|>
                    img_beg = "<|IMAGE_START|>";
                    img_end = "<|IMAGE_END|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_GEMMA4V:
                {
                    // <|image> ... (image embeddings) ... <image|>
                    img_beg = "<|image>";
                    img_end = "<image|>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            case PROJECTOR_TYPE_DEEPSEEKOCR:
                {
                    img_end = "\n"; // prevent empty batch on llama-server
                    image_preproc = std::make_unique<mtmd_image_preprocessor_deepseekocr>(ctx_v);
                } break;
            case PROJECTOR_TYPE_HUNYUANOCR:
                {
                    // note: these use fullwidth ｜ (U+FF5C) and ▁ (U+2581) to match the tokenizer vocabulary
                    img_beg = "<｜hy_place▁holder▁no▁100｜>";
                    img_end = "<｜hy_place▁holder▁no▁101｜>";
                    image_preproc = std::make_unique<mtmd_image_preprocessor_dyn_size>(ctx_v);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected vision projector type %d\n", __func__, proj));
        }

        GGML_ASSERT(image_preproc != nullptr);
    }

    void init_audio() {
        GGML_ASSERT(ctx_a != nullptr);
        audio_preproc.reset();

        projector_type proj = clip_get_projector_type(ctx_a);

        LOG_WRN("%s: audio input is in experimental stage and may have reduced quality:\n"
                "    https://github.com/ggml-org/llama.cpp/discussions/13759\n", __func__);

        // set preprocessor
        switch (proj) {
            case PROJECTOR_TYPE_QWEN2A:
            case PROJECTOR_TYPE_QWEN3A:
            case PROJECTOR_TYPE_QWEN25O:
                {
                    // <|audio_bos|> ... (embeddings) ... <|audio_eos|>
                    aud_beg = "<|audio_bos|>";
                    aud_end = "<|audio_eos|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_VOXTRAL:
                {
                    // [BEGIN_AUDIO] ... (embeddings) ...
                    aud_beg = "[BEGIN_AUDIO]";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_MUSIC_FLAMINGO:
                {
                    // <sound> ... (embeddings) ...
                    aud_beg = "<sound>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_ULTRAVOX:
            case PROJECTOR_TYPE_GLMA:
            case PROJECTOR_TYPE_MERALION:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_whisper>(ctx_a);
                } break;
            case PROJECTOR_TYPE_LFM2A:
                {
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_conformer>(ctx_a);
                } break;
            case PROJECTOR_TYPE_GEMMA4A:
                {
                    aud_beg = "<|audio>";
                    aud_end = "<audio|>";
                    audio_preproc = std::make_unique<mtmd_audio_preprocessor_gemma4a>(ctx_a);
                } break;
            default:
                throw std::runtime_error(string_format("%s: unexpected audio projector type %d\n", __func__, proj));
        }

        // initialize audio preprocessor
        GGML_ASSERT(audio_preproc != nullptr);
        audio_preproc->initialize();
    }

    // get clip ctx based on chunk type
    clip_ctx * get_clip_ctx(const mtmd_input_chunk * chunk) const {
        if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            return ctx_v;
        } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            return ctx_a;
        }
        GGML_ABORT("unknown chunk type");
    }

    projector_type proj_type_v() const {
        return ctx_v ? clip_get_projector_type(ctx_v) : PROJECTOR_TYPE_UNKNOWN;
    }

    projector_type proj_type_a() const {
        return ctx_a ? clip_get_projector_type(ctx_a) : PROJECTOR_TYPE_UNKNOWN;
    }

    ~mtmd_context() {
        clip_free(ctx_a);
        clip_free(ctx_v);
    }

private:
    llama_token lookup_token(const std::string & token_text) {
        const llama_vocab * vocab = llama_model_get_vocab(text_model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < n_vocab; i++) {
            if (token_to_piece(vocab, i, true) == token_text) {
                return i;
            }
        }
        return LLAMA_TOKEN_NULL;
    }

    std::string token_to_piece(const llama_vocab * vocab, llama_token token, bool special) {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }
        return piece;
    }
};

mtmd_context * mtmd_init_from_file(const char * mmproj_fname,
        const struct llama_model * text_model,
        const struct mtmd_context_params ctx_params) {
    try {
        return new mtmd_context(mmproj_fname, text_model, ctx_params);
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
        return nullptr;
    }
}

void mtmd_free(mtmd_context * ctx) {
    delete ctx;
}

struct mtmd_tokenizer {
    enum marker_kind {
        MARKER_KIND_NONE,
        MARKER_KIND_GENERIC,
        MARKER_KIND_IMAGE,
        MARKER_KIND_VIDEO,
    };

    struct marker_match {
        marker_kind kind = MARKER_KIND_NONE;
        size_t pos = std::string::npos;
        size_t len = 0;
    };

    mtmd_context * ctx;
    std::vector<const mtmd_bitmap *> bitmaps;

    std::string input_text;
    bool add_special;
    bool parse_special;
    const llama_vocab * vocab;

    mtmd_input_chunks cur;

    mtmd_tokenizer(mtmd_context * ctx,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) : ctx(ctx), bitmaps(bitmaps, bitmaps + n_bitmaps) {
        add_special   = text->add_special;
        parse_special = text->parse_special;
        input_text    = text->text;
        vocab         = llama_model_get_vocab(ctx->text_model);
    }

    int32_t tokenize(mtmd_input_chunks * output) {
        cur.entries.clear();
        size_t i_bm = 0; // index of the current bitmap
        size_t n_markers = 0;
        size_t start = 0;

        while (start < input_text.size()) {
            marker_match match = find_next_marker(start);
            if (match.kind == MARKER_KIND_NONE) {
                add_text(input_text.substr(start), parse_special);
                break;
            }

            if (match.pos > start) {
                add_text(input_text.substr(start, match.pos - start), parse_special);
            }

            if (i_bm >= bitmaps.size()) {
                LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                        __func__, bitmaps.size(), n_markers + 1);
                return 1;
            }

            const mtmd_bitmap * bitmap = bitmaps[i_bm++];
            int32_t res = add_media(bitmap, match.kind);
            if (res != 0) {
                return res;
            }

            n_markers++;
            start = match.pos + match.len;
        }

        if (add_special && llama_vocab_get_add_bos(vocab)) {
            // if first chunk is text, we add BOS token to first text chunk
            // otherwise, create a new text chunk with BOS token
            if (!cur.entries.empty() && cur.entries[0].type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                // add BOS token to the beginning of first text chunk
                cur.entries[0].tokens_text.insert(cur.entries[0].tokens_text.begin(), llama_vocab_bos(vocab));
            } else {
                // create a new text chunk with BOS token at the beginning
                mtmd_input_chunk bos_chunk{
                    MTMD_INPUT_CHUNK_TYPE_TEXT,
                    {llama_vocab_bos(vocab)},
                    nullptr, // image tokens
                    nullptr, // audio tokens
                };
                cur.entries.insert(cur.entries.begin(), std::move(bos_chunk));
            }
        }

        if (add_special && llama_vocab_get_add_eos(vocab)) {
            // if last chunk is text, we add EOS token to it
            add_text({llama_vocab_eos(vocab)});
        }

        if (i_bm != bitmaps.size()) {
            LOG_ERR("%s: error: number of bitmaps (%zu) does not match number of markers (%zu)\n",
                    __func__, bitmaps.size(), n_markers);
            return 1;
        }

        *output = std::move(cur);

        return 0;
    }

    void add_text(const std::string & txt, bool parse_special) {
        LOG_DBG("%s: %s\n", __func__, txt.c_str());
        auto tokens = mtmd_tokenize_text_internal(vocab, txt, /* add_special */ false, parse_special);
        add_text(tokens);
    }

    void add_text(const std::vector<llama_token> & tokens) {
        if (tokens.empty()) {
            return;
        }
        // if last entry is also a text chunk, add tokens to it instead of creating new chunk
        if (!cur.entries.empty() && cur.entries.back().type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            cur.entries.back().tokens_text.insert(
                                            cur.entries.back().tokens_text.end(),
                                            tokens.begin(),
                                            tokens.end());
        } else {
            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_TEXT,
                tokens,
                nullptr, // image tokens
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));
        }
    }

    int32_t add_media(const mtmd_bitmap * bitmap, marker_kind kind) {
        if (kind == MARKER_KIND_IMAGE && (bitmap->is_audio || bitmap->n_frames >= 2)) {
            LOG_ERR("%s: image placeholder was matched with non-image media input\n", __func__);
            return 2;
        }
        if (kind == MARKER_KIND_VIDEO && (bitmap->is_audio || bitmap->n_frames < 2)) {
            LOG_ERR("%s: video placeholder was matched with non-video media input\n", __func__);
            return 2;
        }

        if (bitmap->n_frames >= 2) {
            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            GGML_ASSERT(bitmap->nx > 0 && bitmap->ny > 0);
            GGML_ASSERT(bitmap->n_frames % 2 == 0);
            GGML_ASSERT(ctx->image_preproc != nullptr);

            const size_t frame_bytes = (size_t) bitmap->nx * bitmap->ny * 3;
            GGML_ASSERT(bitmap->data.size() == frame_bytes * bitmap->n_frames);

            const bool use_qwen3_timestamp_segments = ctx->proj_type_v() == PROJECTOR_TYPE_QWEN3VL;
            double video_fps = 0.0;
            const int32_t * video_frame_indices = nullptr;
            size_t n_video_frame_indices = 0;
            const bool has_video_metadata = mtmd_bitmap_get_video_metadata(
                bitmap,
                &video_fps,
                &video_frame_indices,
                &n_video_frame_indices);
            std::vector<int32_t> synthetic_frame_indices;
            double effective_video_fps = video_fps;
            const int32_t * effective_video_frame_indices = video_frame_indices;
            size_t n_effective_video_frame_indices = n_video_frame_indices;

            if (use_qwen3_timestamp_segments && !has_video_metadata) {
                effective_video_fps = 24.0;
                synthetic_frame_indices.resize(bitmap->n_frames);
                for (uint32_t frame_idx = 0; frame_idx < bitmap->n_frames; ++frame_idx) {
                    synthetic_frame_indices[frame_idx] = (int32_t) frame_idx;
                }
                effective_video_frame_indices = synthetic_frame_indices.data();
                n_effective_video_frame_indices = synthetic_frame_indices.size();
                LOG_WRN("%s: qwen3 video metadata missing; defaulting timestamps to fps=24 for pre-sampled frames\n", __func__);
            }

            auto create_video_pair_tokens = [&](uint32_t pair_idx, clip_image_f32_ptr & pair_img) -> mtmd_image_tokens_ptr {
                auto make_frame = [&](uint32_t frame_idx) {
                    clip_image_u8_ptr img_u8(clip_image_u8_init());
                    img_u8->nx = bitmap->nx;
                    img_u8->ny = bitmap->ny;
                    img_u8->buf.resize(frame_bytes);
                    std::memcpy(img_u8->buf.data(), bitmap->data.data() + frame_idx * frame_bytes, frame_bytes);
                    return img_u8;
                };

                clip_image_f32_batch even_batch;
                clip_image_f32_batch odd_batch;
                if (!ctx->image_preproc->preprocess(*make_frame(pair_idx), even_batch)) {
                    LOG_ERR("Unable to preprocess video frame %u\n", pair_idx);
                    return nullptr;
                }
                if (!ctx->image_preproc->preprocess(*make_frame(pair_idx + 1), odd_batch)) {
                    LOG_ERR("Unable to preprocess video frame %u\n", pair_idx + 1);
                    return nullptr;
                }

                if (even_batch.entries.size() != 1 || odd_batch.entries.size() != 1) {
                    LOG_ERR("%s: only non-tiled video preprocessing is supported in v1\n", __func__);
                    return nullptr;
                }

                const auto & even = even_batch.entries[0];
                const auto & odd  = odd_batch.entries[0];
                if (even->nx != odd->nx || even->ny != odd->ny) {
                    LOG_ERR("%s: mismatched preprocessed video frame sizes\n", __func__);
                    return nullptr;
                }

                pair_img.reset(clip_image_f32_init());
                pair_img->nx = even->nx;
                pair_img->ny = even->ny;
                pair_img->buf.resize((size_t) pair_img->nx * pair_img->ny * 6);

                const size_t n_pixels = (size_t) pair_img->nx * pair_img->ny;
                for (size_t i = 0; i < n_pixels; ++i) {
                    const size_t src = i * 3;
                    const size_t dst = i * 6;
                    pair_img->buf[dst + 0] = even->buf[src + 0];
                    pair_img->buf[dst + 1] = even->buf[src + 1];
                    pair_img->buf[dst + 2] = even->buf[src + 2];
                    pair_img->buf[dst + 3] = odd ->buf[src + 0];
                    pair_img->buf[dst + 4] = odd ->buf[src + 1];
                    pair_img->buf[dst + 5] = odd ->buf[src + 2];
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
                if (mtmd_decode_use_mrope(ctx)) {
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, pair_img.get());
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, pair_img.get());
                    image_tokens->nt = 1;
                    image_tokens->use_mrope_pos = true;
                } else {
                    image_tokens->nx = clip_n_output_tokens(ctx->ctx_v, pair_img.get());
                    image_tokens->ny = 1;
                    image_tokens->nt = 1;
                }
                image_tokens->video_slice_index = pair_idx / 2;
                image_tokens->video_slice_count = bitmap->n_frames / 2;
                image_tokens->video_frame_start = (int32_t) pair_idx;
                image_tokens->video_frame_end   = (int32_t) pair_idx + 1;

                image_tokens->batch_f32.entries.push_back(std::move(pair_img));
                image_tokens->id = bitmap->id;
                return image_tokens;
            };

            auto format_qwen3_timestamp = [&](uint32_t pair_idx) {
                GGML_ASSERT(effective_video_fps > 0.0);
                GGML_ASSERT(pair_idx + 1 < n_effective_video_frame_indices);
                const double start = (double) effective_video_frame_indices[pair_idx] / effective_video_fps;
                const double end   = (double) effective_video_frame_indices[pair_idx + 1] / effective_video_fps;
                const double midpoint = (start + end) / 2.0;
                char buffer[64];
                std::snprintf(buffer, sizeof(buffer), "<%.1f seconds>", midpoint);
                return std::string(buffer);
            };

            if (use_qwen3_timestamp_segments && effective_video_fps > 0.0) {
                clip_image_f32_batch parent_video_batch;
                parent_video_batch.entries.reserve(bitmap->n_frames / 2);
                std::vector<int32_t> slice_frame_starts;
                std::vector<int32_t> slice_frame_ends;
                std::vector<double> slice_timestamps;
                slice_frame_starts.reserve(bitmap->n_frames / 2);
                slice_frame_ends.reserve(bitmap->n_frames / 2);
                slice_timestamps.reserve(bitmap->n_frames / 2);

                uint32_t slice_nx = 0;
                uint32_t slice_ny = 0;
                bool slice_use_mrope = false;

                for (uint32_t pair_idx = 0; pair_idx < bitmap->n_frames; pair_idx += 2) {
                    clip_image_f32_ptr pair_img;
                    auto image_tokens = create_video_pair_tokens(pair_idx, pair_img);
                    if (!image_tokens || image_tokens->batch_f32.entries.empty()) {
                        return 2;
                    }

                    if (parent_video_batch.entries.empty()) {
                        slice_nx = image_tokens->nx;
                        slice_ny = image_tokens->ny;
                        slice_use_mrope = image_tokens->use_mrope_pos;
                    }

                    parent_video_batch.entries.push_back(std::move(image_tokens->batch_f32.entries[0]));
                    slice_frame_starts.push_back(effective_video_frame_indices[pair_idx]);
                    slice_frame_ends.push_back(effective_video_frame_indices[pair_idx + 1]);
                    slice_timestamps.push_back(
                        ((double) effective_video_frame_indices[pair_idx] / effective_video_fps +
                         (double) effective_video_frame_indices[pair_idx + 1] / effective_video_fps) / 2.0);
                }

                for (size_t slice_idx = 0; slice_idx < parent_video_batch.entries.size(); ++slice_idx) {
                    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
                    image_tokens->nx = slice_nx;
                    image_tokens->ny = slice_ny;
                    image_tokens->nt = 1;
                    image_tokens->use_mrope_pos = slice_use_mrope;
                    image_tokens->split_video_temporal_pos = true;
                    image_tokens->video_slice_index = (uint32_t) slice_idx;
                    image_tokens->video_slice_count = (uint32_t) parent_video_batch.entries.size();
                    image_tokens->video_frame_start = slice_frame_starts[slice_idx];
                    image_tokens->video_frame_end = slice_frame_ends[slice_idx];
                    image_tokens->video_timestamp_seconds = slice_timestamps[slice_idx];
                    image_tokens->video_parent_id = bitmap->id.empty() ? string_format("__video_%p", (const void *) bitmap) : bitmap->id;
                    image_tokens->batch_f32 = parent_video_batch.clone();
                    image_tokens->id = bitmap->id;
                    if (!bitmap->id.empty()) {
                        image_tokens->id = string_format("%s#t%zu", bitmap->id.c_str(), slice_idx);
                    }

                    add_text(format_qwen3_timestamp((uint32_t) slice_idx * 2), false);
                    if (!ctx->img_beg.empty()) {
                        add_text(ctx->img_beg, true);
                    }

                    mtmd_input_chunk chunk{
                        MTMD_INPUT_CHUNK_TYPE_VIDEO,
                        {}, // text tokens
                        std::move(image_tokens),
                        nullptr, // audio tokens
                    };
                    cur.entries.emplace_back(std::move(chunk));

                    if (!ctx->img_end.empty()) {
                        add_text(ctx->img_end, true);
                    }
                }

                return 0;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true);
            }

            clip_image_f32_batch video_batch;
            video_batch.entries.reserve(bitmap->n_frames / 2);

            for (uint32_t pair_idx = 0; pair_idx < bitmap->n_frames; pair_idx += 2) {
                clip_image_f32_ptr pair_img;
                auto image_tokens = create_video_pair_tokens(pair_idx, pair_img);
                if (!image_tokens || image_tokens->batch_f32.entries.empty()) {
                    return 2;
                }
                video_batch.entries.push_back(std::move(image_tokens->batch_f32.entries[0]));
            }

            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            if (mtmd_decode_use_mrope(ctx)) {
                image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, video_batch.entries[0].get());
                image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, video_batch.entries[0].get());
                image_tokens->nt = (uint32_t) video_batch.entries.size();
                image_tokens->use_mrope_pos = true;
                image_tokens->split_video_temporal_pos = ctx->proj_type_v() == PROJECTOR_TYPE_QWEN3VL;
                image_tokens->video_slice_count = image_tokens->nt;
                image_tokens->video_parent_id = bitmap->id.empty() ? string_format("__video_%p", (const void *) bitmap) : bitmap->id;
            } else {
                size_t n_tokens = 0;
                for (const auto & entry : video_batch.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_v, entry.get());
                }
                image_tokens->nx = n_tokens;
                image_tokens->ny = 1;
                image_tokens->nt = 1;
            }
            image_tokens->batch_f32 = std::move(video_batch);
            image_tokens->id = bitmap->id;

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_VIDEO,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            cur.entries.emplace_back(std::move(chunk));

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true);
            }

            return 0;
        }

        if (!bitmap->is_audio) {
            // handle image

            if (!ctx->ctx_v) {
                LOG_ERR("%s: error: model does not support vision input\n", __func__);
                return 2;
            }

            if (!ctx->img_beg.empty()) {
                add_text(ctx->img_beg, true); // add image begin token
            }

            // sanity check
            GGML_ASSERT(bitmap->nx > 0 && bitmap->ny > 0);
            GGML_ASSERT(bitmap->data.size() == (size_t)bitmap->nx * bitmap->ny * 3);
            GGML_ASSERT(ctx->image_preproc != nullptr);

            // convert mtmd_bitmap to clip_image_u8
            clip_image_u8_ptr img_u8(clip_image_u8_init());
            img_u8->nx = bitmap->nx;
            img_u8->ny = bitmap->ny;
            img_u8->buf.resize(bitmap->data.size());
            std::memcpy(img_u8->buf.data(), bitmap->data.data(), img_u8->nx * img_u8->ny * 3);

            // preprocess image
            clip_image_f32_batch batch_f32;
            bool ok = ctx->image_preproc->preprocess(*img_u8, batch_f32);
            if (!ok) {
                LOG_ERR("Unable to preprocess image\n");
                return 2;
            }

            // handle llava-uhd style preprocessing
            const bool has_tiling_grid = batch_f32.grid_x > 0 && batch_f32.grid_y > 0;
            if (
                ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_5
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_MINICPMV_2_6
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_LLAMA4
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_IDEFICS3
                || ctx->slice_tmpl == MTMD_SLICE_TMPL_STEP3VL
                || (ctx->slice_tmpl == MTMD_SLICE_TMPL_LFM2 && has_tiling_grid)
            ) {
                const int n_col = batch_f32.grid_x;
                const int n_row = batch_f32.grid_y;
                // split batch into chunks of single images
                // NOTE: batch_f32 will be invalidated after this call
                auto chunks = split_batch_to_chunk(std::move(batch_f32), bitmap->id);
                GGML_ASSERT(chunks.size() > 0);

                auto ov_chunk = std::move(chunks.front());
                chunks.erase(chunks.begin());

                // add overview image (first)
                if (ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

                // add slices (or tiles)
                if (!chunks.empty()) {
                    GGML_ASSERT((int)chunks.size() == n_row * n_col);
                    add_text(ctx->tok_slices_start);
                    for (int y = 0; y < n_row; y++) {
                        for (int x = 0; x < n_col; x++) {
                            const bool is_last_in_row = (x == n_col - 1);
                            if (!ctx->tok_sli_img_start.empty()) {
                                add_text(ctx->tok_sli_img_start);
                            } else if (!ctx->sli_img_start_tmpl.empty()) {
                                // If using a template to preceed a slice image
                                const size_t sz = std::snprintf(nullptr, 0, ctx->sli_img_start_tmpl.c_str(), y+1, x+1) + 1;
                                std::unique_ptr<char[]> buf(new char[sz]);
                                std::snprintf(buf.get(), sz, ctx->sli_img_start_tmpl.c_str(), y+1, x+1);
                                add_text(std::string(buf.get(), buf.get() + sz - 1), true);
                            }
                            cur.entries.emplace_back(std::move(chunks[y * n_col + x]));
                            add_text(ctx->tok_sli_img_end);
                            if (!is_last_in_row) {
                                add_text(ctx->tok_sli_img_mid);
                            }
                        }
                        if ((y != n_row - 1 || ctx->tok_row_end_trail)) {
                            add_text(ctx->tok_row_end);
                        }
                    }
                    add_text(ctx->tok_slices_end);
                }

                // add overview image (last)
                if (!ctx->ov_img_first) {
                    add_text(ctx->tok_ov_img_start);
                    cur.entries.emplace_back(std::move(ov_chunk));
                    add_text(ctx->tok_ov_img_end);
                }

            } else {
                size_t n_tokens = 0;
                for (const auto & entry : batch_f32.entries) {
                    n_tokens += clip_n_output_tokens(ctx->ctx_v, entry.get());
                }

                mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
                if (mtmd_decode_use_mrope(ctx)) {
                    // for Qwen2VL, we need this information for M-RoPE decoding positions
                    image_tokens->nx = clip_n_output_tokens_x(ctx->ctx_v, batch_f32.entries[0].get());
                    image_tokens->ny = clip_n_output_tokens_y(ctx->ctx_v, batch_f32.entries[0].get());
                    image_tokens->nt = 1;
                    image_tokens->use_mrope_pos = true;
                    image_tokens->split_video_temporal_pos = false;
                } else {
                    // other models, we only need the total number of tokens
                    image_tokens->nx = n_tokens;
                    image_tokens->ny = 1;
                    image_tokens->nt = 1;
                }
                image_tokens->batch_f32 = std::move(batch_f32);
                image_tokens->id = bitmap->id; // optional

                LOG_DBG("image_tokens->nx = %d\n", image_tokens->nx);
                LOG_DBG("image_tokens->ny = %d\n", image_tokens->ny);
                LOG_DBG("batch_f32 size = %d\n", (int)image_tokens->batch_f32.entries.size());

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_IMAGE,
                    {}, // text tokens
                    std::move(image_tokens),
                    nullptr, // audio tokens
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->img_end.empty()) {
                add_text(ctx->img_end, true); // add image end token
            }

        } else {
            // handle audio

            if (!ctx->ctx_a) {
                LOG_ERR("%s: error: model does not support audio input\n", __func__);
                return 2;
            }

            if (bitmap->data.size() == 0) {
                LOG_ERR("%s: error: empty audio data\n", __func__);
                return 2;
            }

            if (!ctx->aud_beg.empty()) {
                add_text(ctx->aud_beg, true); // add audio begin token
            }

            // sanity check
            GGML_ASSERT(ctx->audio_preproc != nullptr);
            GGML_ASSERT(bitmap->data.size() > sizeof(float));
            GGML_ASSERT(bitmap->data.size() % sizeof(float) == 0);

            // preprocess audio
            std::vector<mtmd_audio_mel> mel_spec_chunks;
            const float * samples = (const float *)bitmap->data.data();
            size_t n_samples = bitmap->data.size() / sizeof(float);
            bool ok = ctx->audio_preproc->preprocess(samples, n_samples, mel_spec_chunks);
            if (!ok) {
                LOG_ERR("Unable to preprocess audio\n");
                return 2;
            }

            // consider each mel_spec as a separate audio chunk
            // TODO: maybe support batching, but this may come with memory cost
            for (auto & mel_spec : mel_spec_chunks) {
                clip_image_f32_ptr mel_f32(clip_image_f32_init());
                mel_f32->nx  = mel_spec.n_len;
                mel_f32->ny  = mel_spec.n_mel;
                mel_f32->buf = std::move(mel_spec.data);
                size_t n_tokens = clip_n_output_tokens(ctx->ctx_a, mel_f32.get());

                clip_image_f32_batch batch_f32;
                batch_f32.is_audio = true;
                batch_f32.entries.push_back(std::move(mel_f32));

                mtmd_audio_tokens_ptr audio_tokens(new mtmd_audio_tokens);
                audio_tokens->n_tokens = n_tokens;
                audio_tokens->batch_f32 = std::move(batch_f32);
                audio_tokens->id = bitmap->id; // optional

                LOG_DBG("audio_tokens->n_tokens = %d\n", audio_tokens->n_tokens);

                mtmd_input_chunk chunk{
                    MTMD_INPUT_CHUNK_TYPE_AUDIO,
                    {}, // text tokens
                    nullptr, // image tokens
                    std::move(audio_tokens),
                };
                cur.entries.emplace_back(std::move(chunk));
            }

            if (!ctx->aud_end.empty()) {
                add_text(ctx->aud_end, true); // add audio end token
            }
        }

        return 0;
    }

    std::vector<mtmd_input_chunk> split_batch_to_chunk(clip_image_f32_batch && batch_f32, const std::string & id) {
        std::vector<mtmd_input_chunk> chunks;

        for (auto & entry : batch_f32.entries) {
            mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
            image_tokens->nx = clip_n_output_tokens(ctx->ctx_v, entry.get());
            image_tokens->ny = 1;
            image_tokens->nt = 1;
            image_tokens->batch_f32.entries.push_back(std::move(entry));
            image_tokens->id = id;

            mtmd_input_chunk chunk{
                MTMD_INPUT_CHUNK_TYPE_IMAGE,
                {}, // text tokens
                std::move(image_tokens),
                nullptr, // audio tokens
            };
            chunks.emplace_back(std::move(chunk));
        }

        return chunks;
    }

    marker_match find_next_marker(size_t start) const {
        marker_match best;

        auto try_marker = [&](const std::string & marker, marker_kind kind) {
            if (marker.empty()) {
                return;
            }

            size_t pos = input_text.find(marker, start);
            if (pos == std::string::npos) {
                return;
            }

            if (best.kind == MARKER_KIND_NONE || pos < best.pos) {
                best.kind = kind;
                best.pos = pos;
                best.len = marker.size();
            }
        };

        try_marker(ctx->media_marker, MARKER_KIND_GENERIC);
        try_marker(ctx->img_placeholder, MARKER_KIND_IMAGE);
        try_marker(ctx->video_placeholder, MARKER_KIND_VIDEO);

        return best;
    }

    static const char * marker_kind_name(marker_kind kind) {
        switch (kind) {
            case MARKER_KIND_GENERIC: return "media";
            case MARKER_KIND_IMAGE:   return "image";
            case MARKER_KIND_VIDEO:   return "video";
            case MARKER_KIND_NONE:    return "none";
        }
        return "unknown";
    }

    // copied from common_tokenize
    static std::vector<llama_token> mtmd_tokenize_text_internal(
        const struct llama_vocab * vocab,
               const std::string & text,
                            bool   add_special,
                            bool   parse_special) {
        // upper limit for the number of tokens
        int n_tokens = text.length() + 2 * add_special;
        std::vector<llama_token> result(n_tokens);
        n_tokens = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
        if (n_tokens < 0) {
            result.resize(-n_tokens);
            int check = llama_tokenize(vocab, text.data(), text.length(), result.data(), result.size(), add_special, parse_special);
            GGML_ASSERT(check == -n_tokens);
        } else {
            result.resize(n_tokens);
        }
        return result;
    }
};

int32_t mtmd_tokenize(mtmd_context * ctx,
            mtmd_input_chunks * output,
            const mtmd_input_text * text,
            const mtmd_bitmap ** bitmaps,
            size_t n_bitmaps) {
    mtmd_tokenizer tokenizer(ctx, text, bitmaps, n_bitmaps);
    return tokenizer.tokenize(output);
}

int32_t mtmd_encode_chunk(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        LOG_WRN("mtmd_encode_chunk has no effect for text chunks\n");
        return 0;
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk->type == MTMD_INPUT_CHUNK_TYPE_VIDEO) {
        if (!ctx->ctx_v) {
            LOG_ERR("%s: model does not support vision input\n", __func__);
            return 1;
        }
        return mtmd_encode(ctx, chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        if (!ctx->ctx_a) {
            LOG_ERR("%s: model does not support audio input\n", __func__);
            return 1;
        }
        int n_mmproj_embd = ctx->n_embd_text;
        ctx->image_embd_v.resize(chunk->tokens_audio->n_tokens * n_mmproj_embd);
        bool ok = clip_image_batch_encode(
            ctx->ctx_a,
            ctx->n_threads,
            &chunk->tokens_audio->batch_f32,
            ctx->image_embd_v.data());
        return ok ? 0 : 1;
    }

    LOG_ERR("%s: unknown chunk type %d\n", __func__, (int)chunk->type);
    return 1;
}

int32_t mtmd_encode(mtmd_context * ctx, const mtmd_image_tokens * image_tokens) {
    clip_ctx * ctx_clip = ctx->ctx_v;
    if (!ctx_clip) {
        LOG_ERR("%s: this API does not support non-vision input, please use mtmd_encode_chunk instead\n", __func__);
        return 1;
    }
    auto proj_type = clip_get_projector_type(ctx_clip);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    ctx->image_embd_v.resize(image_tokens->n_tokens() * n_mmproj_embd);
    bool ok = false;
    const bool use_qwen3vl_parent_video_batch =
        proj_type == PROJECTOR_TYPE_QWEN3VL &&
        image_tokens->batch_f32.entries.size() > 1;

    if (use_qwen3vl_parent_video_batch) {
        const int n_tokens_per_slice = clip_n_output_tokens(ctx_clip, image_tokens->batch_f32.entries[0].get());
        const size_t n_slices = image_tokens->batch_f32.entries.size();
        const size_t slice_width = (size_t) n_tokens_per_slice * n_mmproj_embd;

        if (!ctx->video_encode_cache.matches(image_tokens)) {
            std::vector<float> full_video_embd(n_slices * slice_width);
            ok = clip_image_batch_encode(
                ctx_clip,
                ctx->n_threads,
                &image_tokens->batch_f32,
                full_video_embd.data());
            if (!ok) {
                return 1;
            }

            ctx->video_encode_cache.parent_id = image_tokens->video_parent_id;
            ctx->video_encode_cache.n_slices = n_slices;
            ctx->video_encode_cache.n_tokens_per_slice = n_tokens_per_slice;
            ctx->video_encode_cache.embd = std::move(full_video_embd);
        }

        if (image_tokens->nt == 1 && image_tokens->video_slice_count > 1) {
            const size_t slice_idx = std::min<size_t>(image_tokens->video_slice_index, image_tokens->batch_f32.entries.size() - 1);
            const size_t slice_bytes = slice_width;
            std::copy_n(
                ctx->video_encode_cache.embd.data() + slice_idx * slice_bytes,
                slice_bytes,
                ctx->image_embd_v.data());
        } else {
            ctx->image_embd_v = ctx->video_encode_cache.embd;
        }

        return 0;
    }

    ctx->video_encode_cache.clear();

    if (image_tokens->nt > 1
        || clip_is_llava(ctx_clip)
        || clip_is_minicpmv(ctx_clip)
        || clip_is_glm(ctx_clip)
        || proj_type == PROJECTOR_TYPE_INTERNVL) {
        // Videos and some multimodal models need one entry at a time.
        const auto & entries = image_tokens->batch_f32.entries;
        for (size_t i = 0; i < entries.size(); i++) {
            int n_tokens_per_image = clip_n_output_tokens(ctx_clip, entries[i].get());
            ok = clip_image_encode(
                ctx_clip,
                ctx->n_threads,
                entries[i].get(),
                ctx->image_embd_v.data() + i*n_mmproj_embd*n_tokens_per_image);
        }
    } else {
        ok = clip_image_batch_encode(
            ctx_clip,
            ctx->n_threads,
            &image_tokens->batch_f32,
            ctx->image_embd_v.data());
    }

    return ok ? 0 : 1;
}

float * mtmd_get_output_embd(mtmd_context * ctx) {
    return ctx->image_embd_v.data();
}

bool mtmd_decode_use_non_causal(mtmd_context * ctx, const mtmd_input_chunk * chunk) {
    auto proj_type = ctx->proj_type_v();
    if (chunk && chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        proj_type = ctx->proj_type_a();
    }
    switch (proj_type) {
        case PROJECTOR_TYPE_GEMMA3:
        case PROJECTOR_TYPE_GEMMA4V:
            return true;
        default:
            return false;
    }
}

bool mtmd_decode_use_mrope(mtmd_context * ctx) {
    if (ctx->ctx_v == nullptr && ctx->proj_type_a() == PROJECTOR_TYPE_QWEN3A) {
        // qwen3-asr
        return true;
    }
    switch (ctx->proj_type_v()) {
        case PROJECTOR_TYPE_QWEN2VL:
        case PROJECTOR_TYPE_QWEN25VL:
        case PROJECTOR_TYPE_QWEN3VL:
        case PROJECTOR_TYPE_GLM4V:
        case PROJECTOR_TYPE_PADDLEOCR:
            return true;
        default:
            return false;
    }
}

bool mtmd_support_vision(mtmd_context * ctx) {
    return ctx->ctx_v != nullptr;
}

bool mtmd_support_audio(mtmd_context * ctx) {
    return ctx->ctx_a != nullptr;
}

int mtmd_get_audio_sample_rate(mtmd_context * ctx) {
    if (!ctx->ctx_a) {
        return -1;
    }
    return clip_get_hparams(ctx->ctx_a)->audio_sample_rate;
}

//
// public API functions
//

// mtmd_bitmap

mtmd_bitmap * mtmd_bitmap_init(uint32_t nx,
                               uint32_t ny,
                               const unsigned char * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = nx;
    bitmap->ny = ny;
    size_t data_size = (size_t)nx * ny * 3;
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_video(uint32_t nx,
                                          uint32_t ny,
                                          uint32_t n_frames,
                                          const unsigned char * data) {
    GGML_ASSERT(n_frames >= 2 && n_frames % 2 == 0);
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = nx;
    bitmap->ny = ny;
    bitmap->n_frames = n_frames;
    size_t data_size = (size_t) nx * ny * 3 * n_frames;
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

mtmd_bitmap * mtmd_bitmap_init_from_audio(size_t n_samples,
                                          const float * data) {
    mtmd_bitmap * bitmap = new mtmd_bitmap;
    bitmap->nx = n_samples;
    bitmap->ny = 1;
    bitmap->is_audio = true;
    size_t data_size = n_samples * sizeof(float);
    bitmap->data.resize(data_size);
    std::memcpy(bitmap->data.data(), data, data_size);
    return bitmap;
}

uint32_t mtmd_bitmap_get_nx(const mtmd_bitmap * bitmap) {
    return bitmap->nx;
}

uint32_t mtmd_bitmap_get_ny(const mtmd_bitmap * bitmap) {
    return bitmap->ny;
}

const unsigned char * mtmd_bitmap_get_data(const mtmd_bitmap * bitmap) {
    return bitmap->data.data();
}

size_t mtmd_bitmap_get_n_bytes(const mtmd_bitmap * bitmap) {
    return bitmap->data.size();
}

bool mtmd_bitmap_is_audio(const mtmd_bitmap * bitmap) {
    return bitmap->is_audio;
}

bool mtmd_bitmap_is_video(const mtmd_bitmap * bitmap) {
    return bitmap->n_frames >= 2;
}

uint32_t mtmd_bitmap_get_n_frames(const mtmd_bitmap * bitmap) {
    return bitmap->n_frames;
}

void mtmd_bitmap_set_video_metadata(
        mtmd_bitmap * bitmap,
        double fps,
        const int32_t * frame_indices,
        size_t n_frame_indices) {
    if (!bitmap || bitmap->n_frames < 2) {
        return;
    }

    bitmap->video_fps = fps;
    bitmap->video_frame_indices.clear();
    if (frame_indices && n_frame_indices > 0) {
        bitmap->video_frame_indices.assign(frame_indices, frame_indices + n_frame_indices);
    }
}

bool mtmd_bitmap_get_video_metadata(
        const mtmd_bitmap * bitmap,
        double * fps,
        const int32_t ** frame_indices,
        size_t * n_frame_indices) {
    if (!bitmap || bitmap->n_frames < 2) {
        return false;
    }

    if (fps) {
        *fps = bitmap->video_fps;
    }
    if (frame_indices) {
        *frame_indices = bitmap->video_frame_indices.empty() ? nullptr : bitmap->video_frame_indices.data();
    }
    if (n_frame_indices) {
        *n_frame_indices = bitmap->video_frame_indices.size();
    }
    return bitmap->video_fps > 0.0 && bitmap->video_frame_indices.size() == bitmap->n_frames;
}

const char * mtmd_bitmap_get_id(const mtmd_bitmap * bitmap) {
    return bitmap->id.c_str();
}

void mtmd_bitmap_set_id(mtmd_bitmap * bitmap, const char * id) {
    if (id) {
        bitmap->id = std::string(id);
    } else {
        bitmap->id.clear();
    }
}

void mtmd_bitmap_free(mtmd_bitmap * bitmap) {
    if (bitmap) {
        delete bitmap;
    }
}

// mtmd_input_chunks

mtmd_input_chunks * mtmd_input_chunks_init() {
    return new mtmd_input_chunks;
}

size_t mtmd_input_chunks_size(const mtmd_input_chunks * chunks) {
    return chunks->entries.size();
}

const mtmd_input_chunk * mtmd_input_chunks_get(const mtmd_input_chunks * chunks, size_t idx) {
    if (idx >= chunks->entries.size()) {
        return nullptr;
    }
    return &chunks->entries[idx];
}

void mtmd_input_chunks_free(mtmd_input_chunks * chunks) {
    if (chunks) {
        delete chunks;
    }
}

// mtmd_input_chunk

enum mtmd_input_chunk_type mtmd_input_chunk_get_type(const mtmd_input_chunk * chunk) {
    return chunk->type;
}

const llama_token * mtmd_input_chunk_get_tokens_text(const mtmd_input_chunk * chunk, size_t * n_tokens_output) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        *n_tokens_output = chunk->tokens_text.size();
        return chunk->tokens_text.data();
    }
    *n_tokens_output = 0;
    return nullptr;
}

const mtmd_image_tokens * mtmd_input_chunk_get_tokens_image(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk->type == MTMD_INPUT_CHUNK_TYPE_VIDEO) {
        return chunk->tokens_image.get();
    }
    return nullptr;
}

size_t mtmd_input_chunk_get_n_tokens(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk->type == MTMD_INPUT_CHUNK_TYPE_VIDEO) {
        return mtmd_image_tokens_get_n_tokens(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

llama_pos mtmd_input_chunk_get_n_pos(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
        return chunk->tokens_text.size();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk->type == MTMD_INPUT_CHUNK_TYPE_VIDEO) {
        return mtmd_image_tokens_get_n_pos(chunk->tokens_image.get());
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->n_tokens;
    } else {
        GGML_ABORT("invalid chunk type");
    }
}

const char * mtmd_input_chunk_get_id(const mtmd_input_chunk * chunk) {
    if (chunk->type == MTMD_INPUT_CHUNK_TYPE_IMAGE || chunk->type == MTMD_INPUT_CHUNK_TYPE_VIDEO) {
        return chunk->tokens_image->id.c_str();
    } else if (chunk->type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
        return chunk->tokens_audio->id.c_str();
    }
    return nullptr;
}

mtmd_input_chunk * mtmd_input_chunk_copy(const mtmd_input_chunk * chunk) {
    mtmd_input_chunk * copy = new mtmd_input_chunk{
        chunk->type,
        chunk->tokens_text,
        nullptr,
        nullptr,
    };
    if (chunk->tokens_image) {
        // copy the image tokens
        copy->tokens_image = mtmd_image_tokens_ptr(new mtmd_image_tokens());
        *copy->tokens_image = chunk->tokens_image->clone();
    }
    if (chunk->tokens_audio) {
        // copy the audio tokens
        copy->tokens_audio = mtmd_audio_tokens_ptr(new mtmd_audio_tokens());
        *copy->tokens_audio = chunk->tokens_audio->clone();
    }
    return copy;
}

void mtmd_input_chunk_free(mtmd_input_chunk * chunk) {
    if (chunk) {
        delete chunk;
    }
}

// mtmd_image_tokens

size_t mtmd_image_tokens_get_n_tokens(const mtmd_image_tokens * image_tokens) {
    return image_tokens->n_tokens();
}

size_t mtmd_image_tokens_get_nx(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nx;
}

size_t mtmd_image_tokens_get_ny(const mtmd_image_tokens * image_tokens) {
    return image_tokens->ny;
}

size_t mtmd_image_tokens_get_nt(const mtmd_image_tokens * image_tokens) {
    return image_tokens->nt;
}

size_t mtmd_image_tokens_get_video_slice_index(const mtmd_image_tokens * image_tokens) {
    return image_tokens->video_slice_index;
}

size_t mtmd_image_tokens_get_video_slice_count(const mtmd_image_tokens * image_tokens) {
    return image_tokens->video_slice_count;
}

int32_t mtmd_image_tokens_get_video_frame_start(const mtmd_image_tokens * image_tokens) {
    return image_tokens->video_frame_start;
}

int32_t mtmd_image_tokens_get_video_frame_end(const mtmd_image_tokens * image_tokens) {
    return image_tokens->video_frame_end;
}

double mtmd_image_tokens_get_video_timestamp_seconds(const mtmd_image_tokens * image_tokens) {
    return image_tokens->video_timestamp_seconds;
}

mtmd_decoder_pos mtmd_image_tokens_get_decoder_pos(const mtmd_image_tokens * image_tokens, llama_pos pos_0, size_t i) {
    mtmd_decoder_pos pos;
    const size_t n_xy = (size_t) image_tokens->nx * image_tokens->ny;
    const size_t rem = i % n_xy;
    llama_pos t;
    llama_pos y;
    llama_pos x;

    if (image_tokens->split_video_temporal_pos && image_tokens->nt > 1) {
        // Qwen3.5-VL separates video into timestamp-delimited 2D vision segments.
        // Each temporal slice consumes a multimodal position span of max(h, w).
        const llama_pos span = (llama_pos) std::max(image_tokens->nx, image_tokens->ny);
        const llama_pos start = pos_0 + (llama_pos) (i / n_xy) * span;
        t = start;
        y = start + (llama_pos) (rem / image_tokens->nx);
        x = start + (llama_pos) (rem % image_tokens->nx);
    } else {
        t = pos_0 + (llama_pos) (i / n_xy);
        y = pos_0 + (llama_pos) (rem / image_tokens->nx);
        x = pos_0 + (llama_pos) (rem % image_tokens->nx);
    }

    pos.t = t;
    pos.x = x;
    pos.y = y;
    pos.z = 0; // reserved for future use
    return pos;
}

const char * mtmd_image_tokens_get_id(const mtmd_image_tokens * image_tokens) {
    return image_tokens->id.c_str();
}

llama_pos mtmd_image_tokens_get_n_pos(const mtmd_image_tokens * image_tokens) {
    if (image_tokens->use_mrope_pos) {
        if (image_tokens->split_video_temporal_pos && image_tokens->nt > 1) {
            return (llama_pos) image_tokens->nt * (llama_pos) std::max(image_tokens->nx, image_tokens->ny);
        }
        // for M-RoPE, temporal dimension = max(t, h, w)
        return (llama_pos) std::max({image_tokens->nt, image_tokens->nx, image_tokens->ny});
    }
    return image_tokens->n_tokens();
}

// test function

mtmd_input_chunks * mtmd_test_create_input_chunks() {
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    if (!chunks) {
        return nullptr;
    }

    // create a text chunk
    std::vector<llama_token> tokens_text = { 1, 2, 3, 4, 5 };
    mtmd_input_chunk chunk_text{
        MTMD_INPUT_CHUNK_TYPE_TEXT,
        std::move(tokens_text),
        nullptr, // image tokens
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_text));

    // create an image chunk
    mtmd_image_tokens_ptr image_tokens(new mtmd_image_tokens);
    image_tokens->nx = 4;
    image_tokens->ny = 4;
    image_tokens->batch_f32.entries.resize(16);
    image_tokens->id = "image_1";
    mtmd_input_chunk chunk_image{
        MTMD_INPUT_CHUNK_TYPE_IMAGE,
        {}, // text tokens
        std::move(image_tokens),
        nullptr, // audio tokens
    };
    chunks->entries.emplace_back(std::move(chunk_image));

    return chunks;
}

void mtmd_log_set(ggml_log_callback log_callback, void * user_data) {
    g_logger_state.log_callback = log_callback ? log_callback : clip_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

//
// Debugging API (NOT intended for public use)
//

static void mtmd_debug_encode_impl(mtmd_context * ctx, clip_ctx * ctx_clip, clip_image_f32 & image) {
    clip_set_debug_output_embeddings(ctx_clip, true);
    int n_mmproj_embd = clip_n_mmproj_embd(ctx_clip);
    int n_tokens = clip_n_output_tokens(ctx_clip, &image);
    std::vector<float> embd_output(n_tokens * n_mmproj_embd, 0.0f);
    bool ok = clip_image_encode(
        ctx_clip,
        ctx->n_threads,
        &image,
        embd_output.data());
    if (!ok) {
        LOG_ERR("%s: failed to encode image\n", __func__);
    }
}

void mtmd_debug_encode_image(mtmd_context * ctx, const std::vector<std::vector<float>> & image) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_f32 inp_image;
    inp_image.nx = image.size();
    inp_image.ny = inp_image.nx;
    inp_image.buf.reserve(inp_image.nx * inp_image.ny);
    for (const auto & row : image) {
        inp_image.buf.insert(inp_image.buf.end(), row.begin(), row.end());
    }
    LOG_INF("%s: created input image with nx=%d, ny=%d\n", __func__, inp_image.nx, inp_image.ny);
    mtmd_debug_encode_impl(ctx, ctx->ctx_v, inp_image);
}

void mtmd_debug_encode_audio(mtmd_context * ctx, const std::vector<float> & input) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    int n_mel = clip_get_hparams(ctx->ctx_a)->n_mel_bins;
    clip_image_f32 inp_audio;
    inp_audio.nx = input.size();
    inp_audio.ny = n_mel;
    inp_audio.buf.resize(input.size() * n_mel);
    for (size_t i = 0; i < input.size(); i++) {
        for (int j = 0; j < n_mel; j++) {
            inp_audio.buf[j * inp_audio.nx + i] = input[i];
        }
    }
    LOG_INF("%s: created input audio with nx=%d, ny=%d\n", __func__, inp_audio.nx, inp_audio.ny);
    mtmd_debug_encode_impl(ctx, ctx->ctx_a, inp_audio);
}

void mtmd_debug_preprocess_image(mtmd_context * ctx, const std::vector<uint8_t> & rgb_values, int nx, int ny) {
    if (!ctx->ctx_v) {
        LOG_ERR("%s: model does not support vision input\n", __func__);
        return;
    }
    clip_image_u8 img_u8;
    img_u8.nx = nx;
    img_u8.ny = ny;
    img_u8.buf = rgb_values;
    clip_image_f32_batch batch_f32;
    GGML_ASSERT(ctx->image_preproc != nullptr);
    bool ok = ctx->image_preproc->preprocess(img_u8, batch_f32);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess image\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed image to batch_f32 with %d entries\n", __func__, (int)batch_f32.entries.size());
    for (size_t i = 0; i < batch_f32.entries.size(); i++) {
        LOG_INF("%s: entry %zu has nx=%d, ny=%d\n", __func__, i, batch_f32.entries[i]->nx, batch_f32.entries[i]->ny);
        // TODO: better way to dump entry content?
    }
}

void mtmd_debug_preprocess_audio(mtmd_context * ctx, const std::vector<float> & samples) {
    if (!ctx->ctx_a) {
        LOG_ERR("%s: model does not support audio input\n", __func__);
        return;
    }
    std::vector<mtmd_audio_mel> mel_spec_chunks;
    bool ok = ctx->audio_preproc->preprocess(samples.data(), samples.size(), mel_spec_chunks);
    if (!ok) {
        LOG_ERR("%s: failed to preprocess audio\n", __func__);
        return;
    }
    LOG_INF("%s: preprocessed audio to %zu mel spec chunks\n", __func__, mel_spec_chunks.size());
    for (size_t i = 0; i < mel_spec_chunks.size(); i++) {
        LOG_INF("%s: mel spec chunk %zu has n_len=%d, n_mel=%d\n", __func__, i, mel_spec_chunks[i].n_len, mel_spec_chunks[i].n_mel);

        // dump mel entries: data is stored as [n_mel][n_len] (mel-major)
        const auto & mel = mel_spec_chunks[i];
        for (int m = 0; m < mel.n_mel; m++) {
            for (int t = 0; t < mel.n_len; t++) {
                LOG_INF("mel[%zu][m=%d][t=%d] = %f\n", i, m, t, mel.data[m * mel.n_len + t]);
            }
        }
    }
}
