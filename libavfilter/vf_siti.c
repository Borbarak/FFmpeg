/*
 * Copyright (c) 2002 A'rpi
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

/**
 * @file
 * Calculate Spatial Info (SI) and Temporal Info (TI) scores
 */

#include <math.h>

#include "libavutil/imgutils.h"
#include "libavutil/internal.h"
#include "libavutil/opt.h"

#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

static const int X_FILTER[9] = {
    1, 0, -1,
    2, 0, -2,
    1, 0, -1
};

static const int Y_FILTER[9] = {
    1, 2, 1,
    0, 0, 0,
    -1, -2, -1
};

typedef struct SiTiContext {
    const AVClass *class;
    int pixel_depth;
    int width, height;
    int nb_frames;
    unsigned char *prev_frame;
    double max_si;
    double max_ti;
    double min_si;
    double min_ti;
    double sum_si;
    double sum_ti;
    int full_range;
} SiTiContext;

static int query_formats(AVFilterContext *ctx) {
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P,
        AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV422P10,
        AV_PIX_FMT_NONE
    };

    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

static av_cold int init(AVFilterContext *ctx) {
    // User options but no input data
    SiTiContext *s = ctx->priv;
    s->max_si = 0;
    return 0;
}

static av_cold void uninit(AVFilterContext *ctx) {
    SiTiContext *s = ctx->priv;

    double avg_si = s->sum_si / s->nb_frames;
    double avg_ti = s->sum_ti / s->nb_frames;
    av_log(ctx, AV_LOG_INFO,
           "Summary:\nTotal frames: %d\n\n"
           "Spatial Information:\nAverage: %f\nMax: %f\nMin: %f\n\n"
           "Temporal Information:\nAverage: %f\nMax: %f\nMin: %f\n",
           s->nb_frames, avg_si, s->max_si, s->min_si, avg_ti, s->max_ti, s->min_ti
    );
}

static int config_input(AVFilterLink *inlink) {
    // Video input data avilable
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;
    int max_pixsteps[4];

    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);
    av_image_fill_max_pixsteps(max_pixsteps, NULL, desc);

    s->pixel_depth = max_pixsteps[0];
    s->width = inlink->w;
    s->height = inlink->h;
    size_t pixel_sz = s->pixel_depth==1? (size_t) sizeof(uint8_t) : (size_t) sizeof(uint16_t);
    size_t data_sz = (size_t) s->width * pixel_sz * s->height;
    s->prev_frame = av_malloc(data_sz);

    return 0;
}

// Get frame data handling 8 and 10 bit formats
static uint16_t get_frame_data(const unsigned char* src, int pixel_depth, int index) {
    const uint16_t *src16 = (const uint16_t *)src;
    if (pixel_depth == 2)
        return src16[index];
    return (uint16_t) src[index];
}

// Set frame data handling 8 and 10 bit formats
static void set_frame_data(unsigned char* dst, int pixel_depth, int index, uint16_t data) {
    uint16_t *dst16 = (uint16_t *)dst;
    if (pixel_depth == 2)
        dst16[index] = data;
    else
        dst[index] = (uint8_t) data;
}

// Determine whether the video is in full or limited range. If not defined, assume limited.
static int is_full_range(AVFrame* frame) {
    // If color range not specified, fallback to pixel format
    if (frame->color_range == AVCOL_RANGE_UNSPECIFIED || frame->color_range == AVCOL_RANGE_NB)
        return frame->format == AV_PIX_FMT_YUVJ420P || frame->format == AV_PIX_FMT_YUVJ422P;
    return frame->color_range == AVCOL_RANGE_JPEG;
}

// Check frame's color range and convert to full range if needed
static uint16_t convert_full_range(uint16_t y, SiTiContext *s) {
    if (s->full_range == 1)
        return y;

    // For 8 bits, limited range goes from 16 to 235, for 10 bits the range is multiplied by 4
    double factor = s->pixel_depth == 1? 1 : 4;
    double shift = 16 * factor;
    double limit_upper = 235 * factor - shift;
    double full_upper = 256 * factor - 1;
    double limit_y = fmin(fmax(y - shift, 0), limit_upper);
    return (uint16_t) (full_upper * limit_y / limit_upper);
}

// Applies sobel convolution
static void convolve_sobel(const unsigned char* src, double* dst, int linesize, SiTiContext *s) {
    int filter_width = 3;
    int filter_size = filter_width * filter_width;
    for (int j=1; j<s->height-1; j++) {
        for (int i=1; i<s->width-1; i++) {
            double x_conv_sum = 0, y_conv_sum = 0;
            for (int k=0; k<filter_size; k++) {
                int ki = k % filter_width - 1;
                int kj = floor(k / filter_width) - 1;
                int index = (j + kj) * (linesize / s->pixel_depth) + (i + ki);
                uint16_t data = convert_full_range(get_frame_data(src, s->pixel_depth, index), s);
                x_conv_sum += data * X_FILTER[k];
                y_conv_sum += data * Y_FILTER[k];
            }
            double gradient = sqrt(x_conv_sum * x_conv_sum + y_conv_sum * y_conv_sum);
            // Dst matrix is smaller than src since we ignore edges that can't be convolved
            dst[(j - 1) * (s->width - 2) + (i - 1)] = gradient;
        }
    }
}

// Calculate pixel difference between current and previous frame, and update previous
static void calculate_motion(const unsigned char* curr, double* motion_matrix,
                             int linesize, SiTiContext *s) {
    for (int j=0; j<s->height; j++) {
        for (int i=0; i<s->width; i++) {
            double motion = 0;
            int curr_index = j * (linesize / s->pixel_depth) + i;
            int prev_index = j * s->width + i;
            uint16_t curr_data = convert_full_range(get_frame_data(curr, s->pixel_depth, curr_index), s);

            // Previous frame is already converted to full range
            if (s->nb_frames > 1)
                motion = curr_data - get_frame_data(s->prev_frame, s->pixel_depth, prev_index);
            set_frame_data(s->prev_frame, s->pixel_depth, prev_index, curr_data);
            motion_matrix[j * s->width + i] = motion;
        }
    }
}

static double std_deviation(double* img_metrics, int width, int height) {
    double size = height * width;
    double mean_sum = 0;
    for (int j=0; j<height; j++)
        for (int i=0; i<width; i++)
            mean_sum += img_metrics[j * width + i];

    double mean = mean_sum / size;

    double sqr_diff_sum = 0;
    for (int j=0; j<height; j++) {
        for (int i=0; i<width; i++) {
            double mean_diff = img_metrics[j * width + i] - mean;
            sqr_diff_sum += (mean_diff * mean_diff);
        }
    }
    double variance = sqr_diff_sum / size;
    return sqrt(variance);
}

static void set_meta(AVDictionary **metadata, const char *key, float d) {
    char value[128];
    snprintf(value, sizeof(value), "%0.2f", d);
    av_dict_set(metadata, key, value, 0);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *frame) {
    AVFilterContext *ctx = inlink->dst;
    SiTiContext *s = ctx->priv;

    // Gradient matrix will not include the input frame's edges
    size_t gradient_data_sz = (size_t) (s->width - 2) * sizeof(double) * (s->height - 2);
    double *gradient_matrix = av_malloc(gradient_data_sz);
    size_t motion_data_sz = (size_t) s->width * sizeof(double) * s->height;
    double *motion_matrix = av_malloc(motion_data_sz);
    if (!gradient_matrix || !motion_matrix) {
        av_frame_free(&frame);
        return AVERROR(ENOMEM);
    }

    s->full_range = is_full_range(frame);
    s->nb_frames++;

    // Calculate si and ti
    convolve_sobel(frame->data[0], gradient_matrix, frame->linesize[0], s);
    calculate_motion(frame->data[0], motion_matrix, frame->linesize[0], s);
    double si = std_deviation(gradient_matrix, s->width - 2, s->height - 2);
    double ti = std_deviation(motion_matrix, s->width, s->height);

    // Calculate statistics
    s->max_si = fmax(si, s->max_si);
    s->max_ti = fmax(ti, s->max_ti);
    s->sum_si += si;
    s->sum_ti += ti;
    s->min_si = s->nb_frames == 1? si : fmin(si, s->min_si);
    s->min_ti = s->nb_frames == 1? ti : fmin(ti, s->min_ti);

    // Set si ti information in frame metadata
    set_meta(&frame->metadata, "lavfi.siti.si", si);
    set_meta(&frame->metadata, "lavfi.siti.ti", ti);

    av_free(gradient_matrix);
    return ff_filter_frame(inlink->dst->outputs[0], frame);
}

#define OFFSET(x) offsetof(SiTiContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM|AV_OPT_FLAG_FILTERING_PARAM

static const AVOption siti_options[] = {
    { NULL }
};

AVFILTER_DEFINE_CLASS(siti);

static const AVFilterPad avfilter_vf_siti_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_siti_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO
    },
    { NULL }
};

AVFilter ff_vf_siti = {
    .name          = "siti",
    .description   = NULL_IF_CONFIG_SMALL("Calculate spatial info (SI)."),
    .priv_size     = sizeof(SiTiContext),
    .priv_class    = &siti_class,
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = avfilter_vf_siti_inputs,
    .outputs       = avfilter_vf_siti_outputs,
};
