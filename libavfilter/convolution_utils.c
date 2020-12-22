/*
 * Copyright (c) 2012-2013 Oka Motofumi (chikuzen.mo at gmail dot com)
 * Copyright (c) 2015 Paul B Mahol
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "libavutil/avstring.h"
#include "libavutil/imgutils.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avfilter.h"
#include "convolution_utils.h"
#include "formats.h"
#include "internal.h"
#include "video.h"


static void filter16_prewitt(uint8_t *dstp, int width,
                             float scale, float delta, const int *const matrix,
                             const uint8_t *c[], int peak, int radius,
                             int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        float suma = AV_RN16A(&c[0][2 * x]) * -1 + AV_RN16A(&c[1][2 * x]) * -1 + AV_RN16A(&c[2][2 * x]) * -1 +
                     AV_RN16A(&c[6][2 * x]) *  1 + AV_RN16A(&c[7][2 * x]) *  1 + AV_RN16A(&c[8][2 * x]) *  1;
        float sumb = AV_RN16A(&c[0][2 * x]) * -1 + AV_RN16A(&c[2][2 * x]) *  1 + AV_RN16A(&c[3][2 * x]) * -1 +
                     AV_RN16A(&c[5][2 * x]) *  1 + AV_RN16A(&c[6][2 * x]) * -1 + AV_RN16A(&c[8][2 * x]) *  1;

        dst[x] = av_clip(sqrtf(suma*suma + sumb*sumb) * scale + delta, 0, peak);
    }
}

static void filter16_roberts(uint8_t *dstp, int width,
                             float scale, float delta, const int *const matrix,
                             const uint8_t *c[], int peak, int radius,
                             int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        float suma = AV_RN16A(&c[0][2 * x]) *  1 + AV_RN16A(&c[1][2 * x]) * -1;
        float sumb = AV_RN16A(&c[4][2 * x]) *  1 + AV_RN16A(&c[3][2 * x]) * -1;

        dst[x] = av_clip(sqrtf(suma*suma + sumb*sumb) * scale + delta, 0, peak);
    }
}

static void filter16_sobel(uint8_t *dstp, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        float suma = AV_RN16A(&c[0][2 * x]) * -1 + AV_RN16A(&c[1][2 * x]) * -2 + AV_RN16A(&c[2][2 * x]) * -1 +
                     AV_RN16A(&c[6][2 * x]) *  1 + AV_RN16A(&c[7][2 * x]) *  2 + AV_RN16A(&c[8][2 * x]) *  1;
        float sumb = AV_RN16A(&c[0][2 * x]) * -1 + AV_RN16A(&c[2][2 * x]) *  1 + AV_RN16A(&c[3][2 * x]) * -2 +
                     AV_RN16A(&c[5][2 * x]) *  2 + AV_RN16A(&c[6][2 * x]) * -1 + AV_RN16A(&c[8][2 * x]) *  1;

        dst[x] = av_clip(sqrtf(suma*suma + sumb*sumb) * scale + delta, 0, peak);
    }
}

static void filter16_kirsch(uint8_t *dstp, int width,
                            float scale, float delta, const int *const matrix,
                            const uint8_t *c[], int peak, int radius,
                            int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    const uint16_t *c0 = (const uint16_t *)c[0], *c1 = (const uint16_t *)c[1], *c2 = (const uint16_t *)c[2];
    const uint16_t *c3 = (const uint16_t *)c[3], *c5 = (const uint16_t *)c[5];
    const uint16_t *c6 = (const uint16_t *)c[6], *c7 = (const uint16_t *)c[7], *c8 = (const uint16_t *)c[8];
    int x;

    for (x = 0; x < width; x++) {
        int sum0 = c0[x] *  5 + c1[x] *  5 + c2[x] *  5 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum1 = c0[x] * -3 + c1[x] *  5 + c2[x] *  5 +
                   c3[x] *  5 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum2 = c0[x] * -3 + c1[x] * -3 + c2[x] *  5 +
                   c3[x] *  5 + c5[x] *  5 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum3 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] *  5 + c5[x] *  5 +
                   c6[x] *  5 + c7[x] * -3 + c8[x] * -3;
        int sum4 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] *  5 +
                   c6[x] *  5 + c7[x] *  5 + c8[x] * -3;
        int sum5 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] *  5 + c7[x] *  5 + c8[x] *  5;
        int sum6 = c0[x] *  5 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] *  5 + c8[x] *  5;
        int sum7 = c0[x] *  5 + c1[x] *  5 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] *  5;

        sum0 = FFMAX(sum0, sum1);
        sum2 = FFMAX(sum2, sum3);
        sum4 = FFMAX(sum4, sum5);
        sum6 = FFMAX(sum6, sum7);
        sum0 = FFMAX(sum0, sum2);
        sum4 = FFMAX(sum4, sum6);
        sum0 = FFMAX(sum0, sum4);

        dst[x] = av_clip(FFABS(sum0) * scale + delta, 0, peak);
    }
}

static void filter_prewitt(uint8_t *dst, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size)
{
    const uint8_t *c0 = c[0], *c1 = c[1], *c2 = c[2];
    const uint8_t *c3 = c[3], *c5 = c[5];
    const uint8_t *c6 = c[6], *c7 = c[7], *c8 = c[8];
    int x;

    for (x = 0; x < width; x++) {
        float suma = c0[x] * -1 + c1[x] * -1 + c2[x] * -1 +
                     c6[x] *  1 + c7[x] *  1 + c8[x] *  1;
        float sumb = c0[x] * -1 + c2[x] *  1 + c3[x] * -1 +
                     c5[x] *  1 + c6[x] * -1 + c8[x] *  1;

        dst[x] = av_clip_uint8(sqrtf(suma*suma + sumb*sumb) * scale + delta);
    }
}

static void filter_roberts(uint8_t *dst, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size)
{
    int x;

    for (x = 0; x < width; x++) {
        float suma = c[0][x] *  1 + c[1][x] * -1;
        float sumb = c[4][x] *  1 + c[3][x] * -1;

        dst[x] = av_clip_uint8(sqrtf(suma*suma + sumb*sumb) * scale + delta);
    }
}

static void filter_sobel(uint8_t *dst, int width,
                         float scale, float delta, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size)
{
    const uint8_t *c0 = c[0], *c1 = c[1], *c2 = c[2];
    const uint8_t *c3 = c[3], *c5 = c[5];
    const uint8_t *c6 = c[6], *c7 = c[7], *c8 = c[8];
    int x;

    for (x = 0; x < width; x++) {
        float suma = c0[x] * -1 + c1[x] * -2 + c2[x] * -1 +
                     c6[x] *  1 + c7[x] *  2 + c8[x] *  1;
        float sumb = c0[x] * -1 + c2[x] *  1 + c3[x] * -2 +
                     c5[x] *  2 + c6[x] * -1 + c8[x] *  1;

        dst[x] = av_clip_uint8(sqrtf(suma*suma + sumb*sumb) * scale + delta);
    }
}

static void filter_kirsch(uint8_t *dst, int width,
                          float scale, float delta, const int *const matrix,
                          const uint8_t *c[], int peak, int radius,
                          int dstride, int stride, int size)
{
    const uint8_t *c0 = c[0], *c1 = c[1], *c2 = c[2];
    const uint8_t *c3 = c[3], *c5 = c[5];
    const uint8_t *c6 = c[6], *c7 = c[7], *c8 = c[8];
    int x;

    for (x = 0; x < width; x++) {
        int sum0 = c0[x] *  5 + c1[x] *  5 + c2[x] *  5 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum1 = c0[x] * -3 + c1[x] *  5 + c2[x] *  5 +
                   c3[x] *  5 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum2 = c0[x] * -3 + c1[x] * -3 + c2[x] *  5 +
                   c3[x] *  5 + c5[x] *  5 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] * -3;
        int sum3 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] *  5 + c5[x] *  5 +
                   c6[x] *  5 + c7[x] * -3 + c8[x] * -3;
        int sum4 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] *  5 +
                   c6[x] *  5 + c7[x] *  5 + c8[x] * -3;
        int sum5 = c0[x] * -3 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] *  5 + c7[x] *  5 + c8[x] *  5;
        int sum6 = c0[x] *  5 + c1[x] * -3 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] *  5 + c8[x] *  5;
        int sum7 = c0[x] *  5 + c1[x] *  5 + c2[x] * -3 +
                   c3[x] * -3 + c5[x] * -3 +
                   c6[x] * -3 + c7[x] * -3 + c8[x] *  5;

        sum0 = FFMAX(sum0, sum1);
        sum2 = FFMAX(sum2, sum3);
        sum4 = FFMAX(sum4, sum5);
        sum6 = FFMAX(sum6, sum7);
        sum0 = FFMAX(sum0, sum2);
        sum4 = FFMAX(sum4, sum6);
        sum0 = FFMAX(sum0, sum4);

        dst[x] = av_clip_uint8(FFABS(sum0) * scale + delta);
    }
}

static void filter16_3x3(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        int sum = AV_RN16A(&c[0][2 * x]) * matrix[0] +
                  AV_RN16A(&c[1][2 * x]) * matrix[1] +
                  AV_RN16A(&c[2][2 * x]) * matrix[2] +
                  AV_RN16A(&c[3][2 * x]) * matrix[3] +
                  AV_RN16A(&c[4][2 * x]) * matrix[4] +
                  AV_RN16A(&c[5][2 * x]) * matrix[5] +
                  AV_RN16A(&c[6][2 * x]) * matrix[6] +
                  AV_RN16A(&c[7][2 * x]) * matrix[7] +
                  AV_RN16A(&c[8][2 * x]) * matrix[8];
        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip(sum, 0, peak);
    }
}

static void filter16_5x5(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 25; i++)
            sum += AV_RN16A(&c[i][2 * x]) * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip(sum, 0, peak);
    }
}

static void filter16_7x7(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 49; i++)
            sum += AV_RN16A(&c[i][2 * x]) * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip(sum, 0, peak);
    }
}

static void filter16_row(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size)
{
    uint16_t *dst = (uint16_t *)dstp;
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 2 * radius + 1; i++)
            sum += AV_RN16A(&c[i][2 * x]) * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip(sum, 0, peak);
    }
}

static void filter16_column(uint8_t *dstp, int height,
                            float rdiv, float bias, const int *const matrix,
                            const uint8_t *c[], int peak, int radius,
                            int dstride, int stride, int size)
{
    DECLARE_ALIGNED(64, int, sum)[16];
    uint16_t *dst = (uint16_t *)dstp;
    const int width = FFMIN(16, size);

    for (int y = 0; y < height; y++) {

        memset(sum, 0, sizeof(sum));
        for (int i = 0; i < 2 * radius + 1; i++) {
            for (int off16 = 0; off16 < width; off16++)
                sum[off16] += AV_RN16A(&c[i][0 + y * stride + off16 * 2]) * matrix[i];
        }

        for (int off16 = 0; off16 < width; off16++) {
            sum[off16] = (int)(sum[off16] * rdiv + bias + 0.5f);
            dst[off16] = av_clip(sum[off16], 0, peak);
        }
        dst += dstride / 2;
    }
}

static void filter_7x7(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size)
{
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 49; i++)
            sum += c[i][x] * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip_uint8(sum);
    }
}

static void filter_5x5(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size)
{
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 25; i++)
            sum += c[i][x] * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip_uint8(sum);
    }
}

static void filter_3x3(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size)
{
    const uint8_t *c0 = c[0], *c1 = c[1], *c2 = c[2];
    const uint8_t *c3 = c[3], *c4 = c[4], *c5 = c[5];
    const uint8_t *c6 = c[6], *c7 = c[7], *c8 = c[8];
    int x;

    for (x = 0; x < width; x++) {
        int sum = c0[x] * matrix[0] + c1[x] * matrix[1] + c2[x] * matrix[2] +
                  c3[x] * matrix[3] + c4[x] * matrix[4] + c5[x] * matrix[5] +
                  c6[x] * matrix[6] + c7[x] * matrix[7] + c8[x] * matrix[8];
        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip_uint8(sum);
    }
}

static void filter_row(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size)
{
    int x;

    for (x = 0; x < width; x++) {
        int i, sum = 0;

        for (i = 0; i < 2 * radius + 1; i++)
            sum += c[i][x] * matrix[i];

        sum = (int)(sum * rdiv + bias + 0.5f);
        dst[x] = av_clip_uint8(sum);
    }
}

static void filter_column(uint8_t *dst, int height,
                          float rdiv, float bias, const int *const matrix,
                          const uint8_t *c[], int peak, int radius,
                          int dstride, int stride, int size)
{
    DECLARE_ALIGNED(64, int, sum)[16];

    for (int y = 0; y < height; y++) {
        memset(sum, 0, sizeof(sum));

        for (int i = 0; i < 2 * radius + 1; i++) {
            for (int off16 = 0; off16 < 16; off16++)
                sum[off16] += c[i][0 + y * stride + off16] * matrix[i];
        }

        for (int off16 = 0; off16 < 16; off16++) {
            sum[off16] = (int)(sum[off16] * rdiv + bias + 0.5f);
            dst[off16] = av_clip_uint8(sum[off16]);
        }
        dst += dstride;
    }
}

static void setup_3x3(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc)
{
    int i;

    for (i = 0; i < 9; i++) {
        int xoff = FFABS(x + ((i % 3) - 1));
        int yoff = FFABS(y + (i / 3) - 1);

        xoff = xoff >= w ? 2 * w - 1 - xoff : xoff;
        yoff = yoff >= h ? 2 * h - 1 - yoff : yoff;

        c[i] = src + xoff * bpc + yoff * stride;
    }
}

static void setup_5x5(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc)
{
    int i;

    for (i = 0; i < 25; i++) {
        int xoff = FFABS(x + ((i % 5) - 2));
        int yoff = FFABS(y + (i / 5) - 2);

        xoff = xoff >= w ? 2 * w - 1 - xoff : xoff;
        yoff = yoff >= h ? 2 * h - 1 - yoff : yoff;

        c[i] = src + xoff * bpc + yoff * stride;
    }
}

static void setup_7x7(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc)
{
    int i;

    for (i = 0; i < 49; i++) {
        int xoff = FFABS(x + ((i % 7) - 3));
        int yoff = FFABS(y + (i / 7) - 3);

        xoff = xoff >= w ? 2 * w - 1 - xoff : xoff;
        yoff = yoff >= h ? 2 * h - 1 - yoff : yoff;

        c[i] = src + xoff * bpc + yoff * stride;
    }
}

static void setup_row(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc)
{
    int i;

    for (i = 0; i < radius * 2 + 1; i++) {
        int xoff = FFABS(x + i - radius);

        xoff = xoff >= w ? 2 * w - 1 - xoff : xoff;

        c[i] = src + xoff * bpc + y * stride;
    }
}

static void setup_column(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                         int x, int w, int y, int h, int bpc)
{
    int i;

    for (i = 0; i < radius * 2 + 1; i++) {
        int xoff = FFABS(x + i - radius);

        xoff = xoff >= h ? 2 * h - 1 - xoff : xoff;

        c[i] = src + y * bpc + xoff * stride;
    }
}
