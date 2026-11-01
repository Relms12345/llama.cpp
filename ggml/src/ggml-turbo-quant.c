// TurboQuant KV cache types: reference (scalar) codec
//
// A block holds one group of 128 values. Quantization:
//   1. norm = |x|, x /= norm
//   2. rotate: x = s2 * (H * (s1 * x)) / sqrt(128), H the Walsh-Hadamard matrix, s1/s2 fixed random signs
//   3. code of each value = number of codebook midpoints <= value (nearest Lloyd-Max centroid for N(0, 1/128))
//   4. store norm / |centroids| so that the decoded block keeps the original L2 norm
// Decoding is centroid * norm and stays in the rotated domain.
//
// The sums of squares are accumulated in 16 lanes (element i in lane i % 16, fused multiply-add) and
// reduced pairwise, so the SIMD implementations in ggml-cpu produce bit-identical blocks.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-quants.h"
#include "ggml-impl.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#define TURBO_INV_SQRT_128 0.08838834764831845f

static float turbo_reduce16(float * acc) {
    for (int w = 8; w >= 1; w /= 2) {
        for (int j = 0; j < w; ++j) {
            acc[j] += acc[j + w];
        }
    }
    return acc[0];
}

static void turbo_butterfly(float * x) {
    for (int h = 1; h < QK_TURBO; h *= 2) {
        for (int i = 0; i < QK_TURBO; i += 2*h) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

void ggml_turbo_rotate_ref(float * x, int direction) {
    const float * sa = direction == 0 ? turbo_s1 : turbo_s2;
    const float * sb = direction == 0 ? turbo_s2 : turbo_s1;

    for (int i = 0; i < QK_TURBO; ++i) {
        x[i] *= sa[i];
    }
    turbo_butterfly(x);
    for (int i = 0; i < QK_TURBO; ++i) {
        x[i] *= TURBO_INV_SQRT_128*sb[i];
    }
}

// steps 1 and 2: r = rotated unit vector, returns the norm of x
static float turbo_prepare(const float * GGML_RESTRICT x, float * GGML_RESTRICT r) {
    float acc[16] = { 0.0f };
    for (int i = 0; i < QK_TURBO; ++i) {
        acc[i % 16] = fmaf(x[i], x[i], acc[i % 16]);
    }
    const float norm = sqrtf(turbo_reduce16(acc));
    const float inv  = norm > 1e-10f ? 1.0f/norm : 0.0f;

    for (int i = 0; i < QK_TURBO; ++i) {
        r[i] = x[i]*inv;
    }
    ggml_turbo_rotate_ref(r, 0);

    return norm;
}

// count of midpoints <= v, by binary search over n midpoints (n + 1 codes)
static int turbo_code(const float * mid, int n, float v) {
    int lo = 0;
    int hi = n;
    while (lo < hi) {
        const int m = (lo + hi)/2;
        if (v < mid[m]) {
            hi = m;
        } else {
            lo = m + 1;
        }
    }
    return lo;
}

// step 4: the stored norm, from the original norm and the codes of the block
static ggml_half turbo_norm(float norm, const float * c, const uint8_t * codes) {
    float acc[16] = { 0.0f };
    for (int i = 0; i < QK_TURBO; ++i) {
        acc[i % 16] = fmaf(c[codes[i]], c[codes[i]], acc[i % 16]);
    }
    const float recon = sqrtf(turbo_reduce16(acc));
    return GGML_FP32_TO_FP16(recon > 1e-10f ? norm/recon : norm);
}

// shared by all types: codes of one block, returns the stored norm
static ggml_half turbo_quantize_block(const float * GGML_RESTRICT x, const float * c, const float * mid, int n_mid, uint8_t * codes) {
    float r[QK_TURBO];
    const float norm = turbo_prepare(x, r);
    for (int i = 0; i < QK_TURBO; ++i) {
        codes[i] = (uint8_t) turbo_code(mid, n_mid, r[i]);
    }
    return turbo_norm(norm, c, codes);
}

//
// packing
//

void ggml_turbo_pack_turbo2_0(const uint8_t * codes, block_turbo2_0 * b) {
    memset(b->qs, 0, sizeof(b->qs));
    for (int j = 0; j < QK_TURBO; ++j) {
        b->qs[j/4] |= (uint8_t) ((codes[j] & 3) << (2*(j%4)));
    }
}

void ggml_turbo_pack_turbo3_0(const uint8_t * codes, block_turbo3_0 * b) {
    memset(b->qs,    0, sizeof(b->qs));
    memset(b->signs, 0, sizeof(b->signs));
    for (int j = 0; j < QK_TURBO; ++j) {
        b->qs[j/4]    |= (uint8_t) ((codes[j] & 3) << (2*(j%4)));
        b->signs[j/8] |= (uint8_t) (((codes[j] >> 2) & 1) << (j%8));
    }
}

void ggml_turbo_pack_turbo4_0(const uint8_t * codes, block_turbo4_0 * b) {
    for (int j = 0; j < QK_TURBO/2; ++j) {
        b->qs[j] = (uint8_t) ((codes[2*j] & 0xF) | ((codes[2*j + 1] & 0xF) << 4));
    }
}

void ggml_turbo_pack_turbo5_0(const uint8_t * codes, block_turbo5_0 * b) {
    memset(b->qs, 0, sizeof(b->qs));
    memset(b->qh, 0, sizeof(b->qh));
    for (int j = 0; j < QK_TURBO; ++j) {
        const uint8_t neg = codes[j] < 16;
        const uint8_t mag = neg ? (uint8_t) (15 - codes[j]) : (uint8_t) (codes[j] - 16);
        b->qs[j/2] |= (uint8_t) (mag << (4*(j%2)));
        b->qh[j/8] |= (uint8_t) (neg << (j%8));
    }
}

void ggml_turbo_pack_turbo6_0(const uint8_t * codes, block_turbo6_0 * b) {
    memset(b->qs, 0, sizeof(b->qs));
    memset(b->qh, 0, sizeof(b->qh));
    for (int j = 0; j < QK_TURBO; ++j) {
        b->qs[j/2] |= (uint8_t) ((codes[j] & 0xF) << (4*(j%2)));
        b->qh[j/4] |= (uint8_t) (((codes[j] >> 4) & 3) << (2*(j%4)));
    }
}

//
// quantize / dequantize
//

#define TURBO_QUANTIZE_IMPL(NAME, C, M, NM)                                                                    \
void quantize_row_##NAME##_ref(const float * GGML_RESTRICT x, block_##NAME * GGML_RESTRICT y, int64_t k) {     \
    assert(k % QK_TURBO == 0);                                                                                  \
    uint8_t codes[QK_TURBO];                                                                                    \
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib) {                                                               \
        y[ib].norm = turbo_quantize_block(x + ib*QK_TURBO, C, M, NM, codes);                                    \
        ggml_turbo_pack_##NAME(codes, &y[ib]);                                                                  \
    }                                                                                                           \
}                                                                                                               \
size_t quantize_##NAME(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows,               \
        int64_t n_per_row, const float * imatrix) {                                                             \
    GGML_UNUSED(imatrix);                                                                                       \
    assert(n_per_row % QK_TURBO == 0);                                                                          \
    const size_t row_size = (n_per_row/QK_TURBO)*sizeof(block_##NAME);                                          \
    for (int64_t row = 0; row < nrows; ++row) {                                                                 \
        quantize_row_##NAME##_ref(src + row*n_per_row, (block_##NAME *) ((char *) dst + row*row_size), n_per_row); \
    }                                                                                                           \
    return nrows*row_size;                                                                                      \
}

TURBO_QUANTIZE_IMPL(turbo2_0, turbo_c2, turbo_m2,  3)
TURBO_QUANTIZE_IMPL(turbo3_0, turbo_c3, turbo_m3,  7)
TURBO_QUANTIZE_IMPL(turbo4_0, turbo_c4, turbo_m4, 15)
TURBO_QUANTIZE_IMPL(turbo5_0, turbo_c5, turbo_m5, 31)
TURBO_QUANTIZE_IMPL(turbo6_0, turbo_c6, turbo_m6, 63)

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO; ++j) {
            y[j] = turbo_c2[(x[ib].qs[j/4] >> (2*(j%4))) & 3]*norm;
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO; ++j) {
            const int code = ((x[ib].qs[j/4] >> (2*(j%4))) & 3) | (((x[ib].signs[j/8] >> (j%8)) & 1) << 2);
            y[j] = turbo_c3[code]*norm;
        }
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO; ++j) {
            y[j] = turbo_c4[(x[ib].qs[j/2] >> (4*(j%2))) & 0xF]*norm;
        }
    }
}

void dequantize_row_turbo5_0(const block_turbo5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO; ++j) {
            const int mag = (x[ib].qs[j/2] >> (4*(j%2))) & 0xF;
            const int neg = (x[ib].qh[j/8] >> (j%8)) & 1;
            y[j] = turbo_c5[neg ? 15 - mag : 16 + mag]*norm;
        }
    }
}

void dequantize_row_turbo6_0(const block_turbo6_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const float norm = GGML_FP16_TO_FP32(x[ib].norm);
        for (int j = 0; j < QK_TURBO; ++j) {
            const int code = ((x[ib].qs[j/2] >> (4*(j%2))) & 0xF) | (((x[ib].qh[j/4] >> (2*(j%4))) & 3) << 4);
            y[j] = turbo_c6[code]*norm;
        }
    }
}
