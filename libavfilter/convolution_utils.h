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

#include "avfilter.h"

static void filter16_prewitt(uint8_t *dstp, int width,
                             float scale, float delta, const int *const matrix,
                             const uint8_t *c[], int peak, int radius,
                             int dstride, int stride, int size);

static void filter16_roberts(uint8_t *dstp, int width,
                             float scale, float delta, const int *const matrix,
                             const uint8_t *c[], int peak, int radius,
                             int dstride, int stride, int size);

static void filter16_sobel(uint8_t *dstp, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size);

static void filter16_kirsch(uint8_t *dstp, int width,
                            float scale, float delta, const int *const matrix,
                            const uint8_t *c[], int peak, int radius,
                            int dstride, int stride, int size);

static void filter_prewitt(uint8_t *dst, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size);

static void filter_roberts(uint8_t *dst, int width,
                           float scale, float delta, const int *const matrix,
                           const uint8_t *c[], int peak, int radius,
                           int dstride, int stride, int size);

static void filter_sobel(uint8_t *dst, int width,
                         float scale, float delta, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size);

static void filter_kirsch(uint8_t *dst, int width,
                          float scale, float delta, const int *const matrix,
                          const uint8_t *c[], int peak, int radius,
                          int dstride, int stride, int size);

static void filter16_3x3(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size);

static void filter16_5x5(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size);

static void filter16_7x7(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size);

static void filter16_row(uint8_t *dstp, int width,
                         float rdiv, float bias, const int *const matrix,
                         const uint8_t *c[], int peak, int radius,
                         int dstride, int stride, int size);

static void filter16_column(uint8_t *dstp, int height,
                            float rdiv, float bias, const int *const matrix,
                            const uint8_t *c[], int peak, int radius,
                            int dstride, int stride, int size);

static void filter_7x7(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size);

static void filter_5x5(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size);

static void filter_3x3(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size);

static void filter_row(uint8_t *dst, int width,
                       float rdiv, float bias, const int *const matrix,
                       const uint8_t *c[], int peak, int radius,
                       int dstride, int stride, int size);

static void filter_column(uint8_t *dst, int height,
                          float rdiv, float bias, const int *const matrix,
                          const uint8_t *c[], int peak, int radius,
                          int dstride, int stride, int size);

static void setup_3x3(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc);

static void setup_5x5(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc);

static void setup_7x7(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc);

static void setup_row(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                      int x, int w, int y, int h, int bpc);

static void setup_column(int radius, const uint8_t *c[], const uint8_t *src, int stride,
                         int x, int w, int y, int h, int bpc);
