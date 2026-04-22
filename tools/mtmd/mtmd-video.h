#pragma once

#include "mtmd-helper.h"

#include <stddef.h>
#include <stdint.h>

void mtmd_bitmap_set_video_metadata(
        mtmd_bitmap * bitmap,
        double fps,
        const int32_t * frame_indices,
        size_t n_frame_indices);

bool mtmd_bitmap_get_video_metadata(
        const mtmd_bitmap * bitmap,
        double * fps,
        const int32_t ** frame_indices,
        size_t * n_frame_indices);

bool mtmd_video_is_video_buffer(const unsigned char * buf, size_t len);

mtmd_bitmap * mtmd_video_bitmap_init_from_buf(
        mtmd_context * ctx,
        const unsigned char * buf,
        size_t len,
        const mtmd_helper_media_options * options);

mtmd_bitmap * mtmd_video_bitmap_init_from_file(
        mtmd_context * ctx,
        const char * fname,
        const mtmd_helper_media_options * options);
