// TurboQuant KV cache types: CPU kernels
//
// quantize_row_turboN_0 : cache writes (SET_ROWS), bit-identical to quantize_row_turboN_0_ref
// ggml_vec_dot_turboN_0 : K * Q dot products in the attention (Q already rotated by the graph)
// ggml_cpu_get_to_float : V and K tiles of the attention
// ggml_cpu_turbo_rotate : GGML_OP_TURBO_WHT
//
// The SIMD paths (AArch64 NEON, x86 AVX2+FMA) follow the operation order of the reference codec in
// ggml-turbo-quant.c: the sums of squares in 16 lanes (element i in lane i % 16) with fused multiply-add
// reduced pairwise, the butterflies in the same stage order, so a block quantizes to the same bytes on
// every path. Decoding is a table lookup and a multiply, exact on every path.
//
// SVE/SME: on current cores the SVE vector length is 128 bits, the same as NEON, so an SVE version of these
// table lookups would not be wider; and streaming SME pays a mode switch that one 128-value block cannot
// amortize. NEON is used on all AArch64 targets.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"
#include "turbo.h"

#include <assert.h>
#include <math.h>
#include <string.h>

#if defined(__ARM_NEON) && defined(__aarch64__)
#define TURBO_NEON 1
#include <arm_neon.h>
#elif defined(__AVX2__) && defined(__FMA__)
#define TURBO_AVX2 1
#include <immintrin.h>
#endif

#define TURBO_INV_SQRT_128 0.08838834764831845f

// values looked up by the decoder: the codebook, for turbo5 the 16 magnitudes c[16..31] (the sign is a separate bit)
static const float * turbo_lut(int t) {
    switch (t) {
        case 2: return turbo_c2;
        case 3: return turbo_c3;
        case 4: return turbo_c4;
        case 5: return turbo_c5 + 16;
        case 6: return turbo_c6;
        default: GGML_ABORT("invalid turbo type");
    }
}

static int turbo_lut_size(int t) {
    return t == 6 ? 64 : t == 5 ? 16 : 1 << t;
}

static int turbo_n_mid(int t) {
    return (1 << t) - 1;
}

static const float * turbo_mid(int t) {
    switch (t) {
        case 2: return turbo_m2;
        case 3: return turbo_m3;
        case 4: return turbo_m4;
        case 5: return turbo_m5;
        case 6: return turbo_m6;
        default: GGML_ABORT("invalid turbo type");
    }
}

static size_t turbo_block_size(int t) {
    switch (t) {
        case 2: return sizeof(block_turbo2_0);
        case 3: return sizeof(block_turbo3_0);
        case 4: return sizeof(block_turbo4_0);
        case 5: return sizeof(block_turbo5_0);
        case 6: return sizeof(block_turbo6_0);
        default: GGML_ABORT("invalid turbo type");
    }
}

static void turbo_pack(int t, const uint8_t * codes, void * b) {
    switch (t) {
        case 2: ggml_turbo_pack_turbo2_0(codes, (block_turbo2_0 *) b); break;
        case 3: ggml_turbo_pack_turbo3_0(codes, (block_turbo3_0 *) b); break;
        case 4: ggml_turbo_pack_turbo4_0(codes, (block_turbo4_0 *) b); break;
        case 5: ggml_turbo_pack_turbo5_0(codes, (block_turbo5_0 *) b); break;
        case 6: ggml_turbo_pack_turbo6_0(codes, (block_turbo6_0 *) b); break;
        default: GGML_ABORT("invalid turbo type");
    }
}

// all block types start with the fp16 norm
static float turbo_block_norm(const void * b) {
    return GGML_CPU_FP16_TO_FP32(*(const ggml_half *) b);
}

//
// unpack: block -> lookup indices (one byte per value) and, for turbo5, a 0xFF/0x00 sign byte per value
//

static void turbo_unpack_scalar(int t, const void * vb, uint8_t * GGML_RESTRICT idx, uint8_t * GGML_RESTRICT neg) {
    switch (t) {
        case 2: {
            const block_turbo2_0 * b = (const block_turbo2_0 *) vb;
            for (int j = 0; j < QK_TURBO; ++j) {
                idx[j] = (b->qs[j/4] >> (2*(j%4))) & 3;
            }
        } break;
        case 3: {
            const block_turbo3_0 * b = (const block_turbo3_0 *) vb;
            for (int j = 0; j < QK_TURBO; ++j) {
                idx[j] = ((b->qs[j/4] >> (2*(j%4))) & 3) | (((b->signs[j/8] >> (j%8)) & 1) << 2);
            }
        } break;
        case 4: {
            const block_turbo4_0 * b = (const block_turbo4_0 *) vb;
            for (int j = 0; j < QK_TURBO; ++j) {
                idx[j] = (b->qs[j/2] >> (4*(j%2))) & 0xF;
            }
        } break;
        case 5: {
            const block_turbo5_0 * b = (const block_turbo5_0 *) vb;
            for (int j = 0; j < QK_TURBO; ++j) {
                idx[j] = (b->qs[j/2] >> (4*(j%2))) & 0xF;
                neg[j] = ((b->qh[j/8] >> (j%8)) & 1) ? 0xFF : 0x00;
            }
        } break;
        case 6: {
            const block_turbo6_0 * b = (const block_turbo6_0 *) vb;
            for (int j = 0; j < QK_TURBO; ++j) {
                idx[j] = ((b->qs[j/2] >> (4*(j%2))) & 0xF) | (((b->qh[j/4] >> (2*(j%4))) & 3) << 4);
            }
        } break;
        default: GGML_ABORT("invalid turbo type");
    }
}

// turbo5 quantizer codes 0..31 -> magnitude index and sign byte
static void turbo5_split_codes(const uint8_t * GGML_RESTRICT codes, uint8_t * GGML_RESTRICT idx, uint8_t * GGML_RESTRICT neg) {
    for (int j = 0; j < QK_TURBO; ++j) {
        const int is_neg = codes[j] < 16;
        idx[j] = is_neg ? (uint8_t) (15 - codes[j]) : (uint8_t) (codes[j] - 16);
        neg[j] = is_neg ? 0xFF : 0x00;
    }
}

#if defined(TURBO_NEON)

// byte planes of the lookup tables: plane p holds byte p of each float, so vqtbl gathers 16 floats at once
static uint8_t turbo_planes[7][4][64];

static void turbo_planes_init(void) {
    for (int t = 2; t <= 6; ++t) {
        const float * lut = turbo_lut(t);
        for (int i = 0; i < turbo_lut_size(t); ++i) {
            uint8_t bytes[4];
            memcpy(bytes, &lut[i], 4);
            for (int p = 0; p < 4; ++p) {
                turbo_planes[t][p][i] = bytes[p];
            }
        }
    }
}

static inline void turbo_unpack_2bit_neon(const uint8_t * GGML_RESTRICT qs, int nbytes, uint8_t * GGML_RESTRICT out) {
    const uint8x16_t m3 = vdupq_n_u8(3);
    for (int r = 0; r < nbytes/16; ++r) {
        const uint8x16_t q = vld1q_u8(qs + 16*r);
        uint8x16x4_t c;
        c.val[0] = vandq_u8(q, m3);
        c.val[1] = vandq_u8(vshrq_n_u8(q, 2), m3);
        c.val[2] = vandq_u8(vshrq_n_u8(q, 4), m3);
        c.val[3] = vshrq_n_u8(q, 6);
        vst4q_u8(out + 64*r, c);
    }
}

static inline void turbo_unpack_4bit_neon(const uint8_t * GGML_RESTRICT qs, uint8_t * GGML_RESTRICT out) {
    const uint8x16_t m4 = vdupq_n_u8(0xF);
    for (int r = 0; r < 4; ++r) {
        const uint8x16_t q = vld1q_u8(qs + 16*r);
        uint8x16x2_t c;
        c.val[0] = vandq_u8(q, m4);
        c.val[1] = vshrq_n_u8(q, 4);
        vst2q_u8(out + 32*r, c);
    }
}

// one bit per value (bit j%8 of bits[j/8]) -> 0xFF/0x00 per value
static inline void turbo_unpack_1bit_neon(const uint8_t * GGML_RESTRICT bits, uint8_t * GGML_RESTRICT out) {
    static const uint8_t k_bit[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
    const uint8x16_t s    = vld1q_u8(bits);
    const uint8x16_t mask = vld1q_u8(k_bit);
    uint8x16_t sel = vcombine_u8(vdup_n_u8(0), vdup_n_u8(1));
    for (int k = 0; k < 8; ++k) {
        vst1q_u8(out + 16*k, vtstq_u8(vqtbl1q_u8(s, sel), mask));
        sel = vaddq_u8(sel, vdupq_n_u8(2));
    }
}

static void turbo_unpack_neon(int t, const void * vb, uint8_t * GGML_RESTRICT idx, uint8_t * GGML_RESTRICT neg) {
    switch (t) {
        case 2: {
            const block_turbo2_0 * b = (const block_turbo2_0 *) vb;
            turbo_unpack_2bit_neon(b->qs, sizeof(b->qs), idx);
        } break;
        case 3: {
            const block_turbo3_0 * b = (const block_turbo3_0 *) vb;
            uint8_t hi[QK_TURBO];
            turbo_unpack_2bit_neon(b->qs, sizeof(b->qs), idx);
            turbo_unpack_1bit_neon(b->signs, hi);
            for (int j = 0; j < QK_TURBO; j += 16) {
                vst1q_u8(idx + j, vorrq_u8(vld1q_u8(idx + j), vandq_u8(vld1q_u8(hi + j), vdupq_n_u8(4))));
            }
        } break;
        case 4: {
            const block_turbo4_0 * b = (const block_turbo4_0 *) vb;
            turbo_unpack_4bit_neon(b->qs, idx);
        } break;
        case 5: {
            const block_turbo5_0 * b = (const block_turbo5_0 *) vb;
            turbo_unpack_4bit_neon(b->qs, idx);
            turbo_unpack_1bit_neon(b->qh, neg);
        } break;
        case 6: {
            const block_turbo6_0 * b = (const block_turbo6_0 *) vb;
            uint8_t hi[QK_TURBO];
            turbo_unpack_4bit_neon(b->qs, idx);
            turbo_unpack_2bit_neon(b->qh, sizeof(b->qh), hi);
            for (int j = 0; j < QK_TURBO; j += 16) {
                vst1q_u8(idx + j, vorrq_u8(vld1q_u8(idx + j), vshlq_n_u8(vld1q_u8(hi + j), 4)));
            }
        } break;
        default: GGML_ABORT("invalid turbo type");
    }
}

// 16 lookups: v[0..3] = lut[idx[0..15]], negated where neg is 0xFF (neg may be NULL)
static inline void turbo_lookup16_neon(int t, const uint8_t * GGML_RESTRICT idx, const uint8_t * GGML_RESTRICT neg, float32x4_t v[4]) {
    const uint8x16_t c = vld1q_u8(idx);
    uint8x16_t b[4];
    if (t == 6) {
        for (int p = 0; p < 4; ++p) {
            b[p] = vqtbl4q_u8(vld1q_u8_x4(turbo_planes[6][p]), c);
        }
    } else {
        for (int p = 0; p < 4; ++p) {
            b[p] = vqtbl1q_u8(vld1q_u8(turbo_planes[t][p]), c);
        }
    }
    if (neg) {
        b[3] = vorrq_u8(b[3], vandq_u8(vld1q_u8(neg), vdupq_n_u8(0x80)));
    }
    const uint8x16x2_t z01 = vzipq_u8(b[0], b[1]);
    const uint8x16x2_t z23 = vzipq_u8(b[2], b[3]);
    const uint16x8x2_t lo  = vzipq_u16(vreinterpretq_u16_u8(z01.val[0]), vreinterpretq_u16_u8(z23.val[0]));
    const uint16x8x2_t hi  = vzipq_u16(vreinterpretq_u16_u8(z01.val[1]), vreinterpretq_u16_u8(z23.val[1]));
    v[0] = vreinterpretq_f32_u16(lo.val[0]);
    v[1] = vreinterpretq_f32_u16(lo.val[1]);
    v[2] = vreinterpretq_f32_u16(hi.val[0]);
    v[3] = vreinterpretq_f32_u16(hi.val[1]);
}

static void turbo_rotate_neon(float * GGML_RESTRICT x, const float * GGML_RESTRICT sa, const float * GGML_RESTRICT sbn) {
    float32x4_t v[32];
    for (int r = 0; r < 32; ++r) {
        v[r] = vmulq_f32(vld1q_f32(x + 4*r), vld1q_f32(sa + 4*r));
    }
    for (int r = 0; r < 32; ++r) {
        // h = 1
        const float32x4_t rev = vrev64q_f32(v[r]);
        const float32x4_t a   = vtrn1q_f32(vaddq_f32(v[r], rev), vsubq_f32(v[r], rev));
        // h = 2
        const float32x2_t lo = vget_low_f32(a);
        const float32x2_t hi = vget_high_f32(a);
        v[r] = vcombine_f32(vadd_f32(lo, hi), vsub_f32(lo, hi));
    }
    for (int d = 1; d < 32; d *= 2) {
        for (int i = 0; i < 32; i += 2*d) {
            for (int j = i; j < i + d; ++j) {
                const float32x4_t a = v[j];
                const float32x4_t b = v[j + d];
                v[j]     = vaddq_f32(a, b);
                v[j + d] = vsubq_f32(a, b);
            }
        }
    }
    for (int r = 0; r < 32; ++r) {
        vst1q_f32(x + 4*r, vmulq_f32(v[r], vld1q_f32(sbn + 4*r)));
    }
}

// sum of 16 lanes in the order of the reference (acc[j] += acc[j + w] for w = 8, 4, 2, 1)
static inline float turbo_reduce16_neon(float32x4_t a0, float32x4_t a1, float32x4_t a2, float32x4_t a3) {
    const float32x4_t t = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
    const float32x2_t u = vadd_f32(vget_low_f32(t), vget_high_f32(t));
    return vpadds_f32(u);
}

static inline float turbo_sumsq_neon(const float * GGML_RESTRICT x) {
    float32x4_t a[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
    for (int i = 0; i < QK_TURBO; i += 16) {
        for (int r = 0; r < 4; ++r) {
            const float32x4_t v = vld1q_f32(x + i + 4*r);
            a[r] = vfmaq_f32(a[r], v, v);
        }
    }
    return turbo_reduce16_neon(a[0], a[1], a[2], a[3]);
}

// codes = number of midpoints m with !(x < m), 16 values at a time
static void turbo_codes_neon(const float * GGML_RESTRICT x, const float * GGML_RESTRICT mid, int n_mid, uint8_t * GGML_RESTRICT codes) {
    for (int i = 0; i < QK_TURBO; i += 16) {
        float32x4_t v[4];
        uint32x4_t  c[4];
        for (int r = 0; r < 4; ++r) {
            v[r] = vld1q_f32(x + i + 4*r);
            c[r] = vdupq_n_u32((uint32_t) n_mid);
        }
        for (int m = 0; m < n_mid; ++m) {
            const float32x4_t vm = vdupq_n_f32(mid[m]);
            for (int r = 0; r < 4; ++r) {
                c[r] = vaddq_u32(c[r], vcltq_f32(v[r], vm));
            }
        }
        const uint16x8_t c01 = vcombine_u16(vmovn_u32(c[0]), vmovn_u32(c[1]));
        const uint16x8_t c23 = vcombine_u16(vmovn_u32(c[2]), vmovn_u32(c[3]));
        vst1q_u8(codes + i, vcombine_u8(vmovn_u16(c01), vmovn_u16(c23)));
    }
}

#elif defined(TURBO_AVX2)

static inline void turbo_unpack_2bit_sse(const uint8_t * GGML_RESTRICT qs, int nbytes, uint8_t * GGML_RESTRICT out) {
    const __m128i m3 = _mm_set1_epi8(3);
    for (int r = 0; r < nbytes/16; ++r) {
        const __m128i q  = _mm_loadu_si128((const __m128i *) (qs + 16*r));
        const __m128i c0 = _mm_and_si128(q, m3);
        const __m128i c1 = _mm_and_si128(_mm_srli_epi16(q, 2), m3);
        const __m128i c2 = _mm_and_si128(_mm_srli_epi16(q, 4), m3);
        const __m128i c3 = _mm_and_si128(_mm_srli_epi16(q, 6), m3);
        const __m128i a_lo = _mm_unpacklo_epi8(c0, c1);
        const __m128i a_hi = _mm_unpackhi_epi8(c0, c1);
        const __m128i b_lo = _mm_unpacklo_epi8(c2, c3);
        const __m128i b_hi = _mm_unpackhi_epi8(c2, c3);
        _mm_storeu_si128((__m128i *) (out + 64*r +  0), _mm_unpacklo_epi16(a_lo, b_lo));
        _mm_storeu_si128((__m128i *) (out + 64*r + 16), _mm_unpackhi_epi16(a_lo, b_lo));
        _mm_storeu_si128((__m128i *) (out + 64*r + 32), _mm_unpacklo_epi16(a_hi, b_hi));
        _mm_storeu_si128((__m128i *) (out + 64*r + 48), _mm_unpackhi_epi16(a_hi, b_hi));
    }
}

static inline void turbo_unpack_4bit_sse(const uint8_t * GGML_RESTRICT qs, uint8_t * GGML_RESTRICT out) {
    const __m128i m4 = _mm_set1_epi8(0xF);
    for (int r = 0; r < 4; ++r) {
        const __m128i q  = _mm_loadu_si128((const __m128i *) (qs + 16*r));
        const __m128i lo = _mm_and_si128(q, m4);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 4), m4);
        _mm_storeu_si128((__m128i *) (out + 32*r +  0), _mm_unpacklo_epi8(lo, hi));
        _mm_storeu_si128((__m128i *) (out + 32*r + 16), _mm_unpackhi_epi8(lo, hi));
    }
}

static inline void turbo_unpack_1bit_sse(const uint8_t * GGML_RESTRICT bits, uint8_t * GGML_RESTRICT out) {
    const __m128i s    = _mm_loadu_si128((const __m128i *) bits);
    const __m128i mask = _mm_setr_epi8(1, 2, 4, 8, 16, 32, 64, (char) 128, 1, 2, 4, 8, 16, 32, 64, (char) 128);
    __m128i sel = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
    for (int k = 0; k < 8; ++k) {
        const __m128i v = _mm_and_si128(_mm_shuffle_epi8(s, sel), mask);
        _mm_storeu_si128((__m128i *) (out + 16*k), _mm_cmpeq_epi8(v, mask));
        sel = _mm_add_epi8(sel, _mm_set1_epi8(2));
    }
}

static void turbo_unpack_avx2(int t, const void * vb, uint8_t * GGML_RESTRICT idx, uint8_t * GGML_RESTRICT neg) {
    switch (t) {
        case 2: {
            const block_turbo2_0 * b = (const block_turbo2_0 *) vb;
            turbo_unpack_2bit_sse(b->qs, sizeof(b->qs), idx);
        } break;
        case 3: {
            const block_turbo3_0 * b = (const block_turbo3_0 *) vb;
            uint8_t hi[QK_TURBO];
            turbo_unpack_2bit_sse(b->qs, sizeof(b->qs), idx);
            turbo_unpack_1bit_sse(b->signs, hi);
            for (int j = 0; j < QK_TURBO; j += 16) {
                const __m128i a = _mm_loadu_si128((const __m128i *) (idx + j));
                const __m128i h = _mm_and_si128(_mm_loadu_si128((const __m128i *) (hi + j)), _mm_set1_epi8(4));
                _mm_storeu_si128((__m128i *) (idx + j), _mm_or_si128(a, h));
            }
        } break;
        case 4: {
            const block_turbo4_0 * b = (const block_turbo4_0 *) vb;
            turbo_unpack_4bit_sse(b->qs, idx);
        } break;
        case 5: {
            const block_turbo5_0 * b = (const block_turbo5_0 *) vb;
            turbo_unpack_4bit_sse(b->qs, idx);
            turbo_unpack_1bit_sse(b->qh, neg);
        } break;
        case 6: {
            const block_turbo6_0 * b = (const block_turbo6_0 *) vb;
            uint8_t hi[QK_TURBO];
            turbo_unpack_4bit_sse(b->qs, idx);
            turbo_unpack_2bit_sse(b->qh, sizeof(b->qh), hi);
            for (int j = 0; j < QK_TURBO; j += 16) {
                const __m128i a = _mm_loadu_si128((const __m128i *) (idx + j));
                const __m128i h = _mm_slli_epi16(_mm_loadu_si128((const __m128i *) (hi + j)), 4); // hi < 4: no carry between bytes
                _mm_storeu_si128((__m128i *) (idx + j), _mm_or_si128(a, h));
            }
        } break;
        default: GGML_ABORT("invalid turbo type");
    }
}

// 8 lookups: lut[idx[0..7]], negated where neg is 0xFF (neg may be NULL)
static inline __m256 turbo_lookup8_avx2(int t, const float * GGML_RESTRICT lut, const uint8_t * GGML_RESTRICT idx, const uint8_t * GGML_RESTRICT neg) {
    const __m256i i = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *) idx));
    __m256 v;
    if (t == 2 || t == 3) {
        v = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lut), i); // lut padded below to 8 entries
    } else if (t == 4 || t == 5) {
        const __m256 lo = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lut + 0), i);
        const __m256 hi = _mm256_permutevar8x32_ps(_mm256_loadu_ps(lut + 8), i);
        v = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_slli_epi32(i, 28)));
    } else {
        v = _mm256_i32gather_ps(lut, i, 4);
    }
    if (neg) {
        const __m256i n = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *) neg));
        v = _mm256_xor_ps(v, _mm256_and_ps(_mm256_castsi256_ps(n), _mm256_set1_ps(-0.0f)));
    }
    return v;
}

// lookup tables padded to 8 entries for the permutes (turbo2)
static float turbo_lut_pad[7][64];

static void turbo_planes_init(void) {
    for (int t = 2; t <= 6; ++t) {
        memcpy(turbo_lut_pad[t], turbo_lut(t), turbo_lut_size(t)*sizeof(float));
    }
}

// partner lane j ^ h, lanes with bit h set get partner - x, the others x + partner
static inline __m256 turbo_bfly_intra_avx2(__m256 x, int h) {
    const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    const __m256i hv   = _mm256_set1_epi32(h);
    const __m256  p    = _mm256_permutevar8x32_ps(x, _mm256_xor_si256(lane, hv));
    const __m256  sel  = _mm256_castsi256_ps(_mm256_cmpeq_epi32(_mm256_and_si256(lane, hv), hv));
    return _mm256_blendv_ps(_mm256_add_ps(x, p), _mm256_sub_ps(p, x), sel);
}

static void turbo_rotate_avx2(float * GGML_RESTRICT x, const float * GGML_RESTRICT sa, const float * GGML_RESTRICT sbn) {
    __m256 v[16];
    for (int r = 0; r < 16; ++r) {
        v[r] = _mm256_mul_ps(_mm256_loadu_ps(x + 8*r), _mm256_loadu_ps(sa + 8*r));
    }
    for (int r = 0; r < 16; ++r) {
        v[r] = turbo_bfly_intra_avx2(v[r], 1);
        v[r] = turbo_bfly_intra_avx2(v[r], 2);
        v[r] = turbo_bfly_intra_avx2(v[r], 4);
    }
    for (int d = 1; d < 16; d *= 2) {
        for (int i = 0; i < 16; i += 2*d) {
            for (int j = i; j < i + d; ++j) {
                const __m256 a = v[j];
                const __m256 b = v[j + d];
                v[j]     = _mm256_add_ps(a, b);
                v[j + d] = _mm256_sub_ps(a, b);
            }
        }
    }
    for (int r = 0; r < 16; ++r) {
        _mm256_storeu_ps(x + 8*r, _mm256_mul_ps(v[r], _mm256_loadu_ps(sbn + 8*r)));
    }
}

static inline float turbo_reduce16_avx2(__m256 a, __m256 b) {
    const __m256 t = _mm256_add_ps(a, b);
    const __m128 u = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    const __m128 w = _mm_add_ps(u, _mm_movehl_ps(u, u));
    return _mm_cvtss_f32(_mm_add_ss(w, _mm_shuffle_ps(w, w, 1)));
}

static inline float turbo_sumsq_avx2(const float * GGML_RESTRICT x) {
    __m256 a = _mm256_setzero_ps();
    __m256 b = _mm256_setzero_ps();
    for (int i = 0; i < QK_TURBO; i += 16) {
        const __m256 v0 = _mm256_loadu_ps(x + i);
        const __m256 v1 = _mm256_loadu_ps(x + i + 8);
        a = _mm256_fmadd_ps(v0, v0, a);
        b = _mm256_fmadd_ps(v1, v1, b);
    }
    return turbo_reduce16_avx2(a, b);
}

static void turbo_codes_avx2(const float * GGML_RESTRICT x, const float * GGML_RESTRICT mid, int n_mid, uint8_t * GGML_RESTRICT codes) {
    for (int i = 0; i < QK_TURBO; i += 32) {
        __m256  v[4];
        __m256i c[4];
        for (int r = 0; r < 4; ++r) {
            v[r] = _mm256_loadu_ps(x + i + 8*r);
            c[r] = _mm256_set1_epi32(n_mid);
        }
        for (int m = 0; m < n_mid; ++m) {
            const __m256 vm = _mm256_set1_ps(mid[m]);
            for (int r = 0; r < 4; ++r) {
                c[r] = _mm256_add_epi32(c[r], _mm256_castps_si256(_mm256_cmp_ps(v[r], vm, _CMP_LT_OQ)));
            }
        }
        int32_t tmp[32];
        for (int r = 0; r < 4; ++r) {
            _mm256_storeu_si256((__m256i *) (tmp + 8*r), c[r]);
        }
        for (int j = 0; j < 32; ++j) {
            codes[i + j] = (uint8_t) tmp[j];
        }
    }
}

#else

static void turbo_planes_init(void) {
}

#endif

static float turbo_s1n[QK_TURBO]; // turbo_s1 / sqrt(128)
static float turbo_s2n[QK_TURBO]; // turbo_s2 / sqrt(128)

void ggml_cpu_turbo_init(void) {
    for (int i = 0; i < QK_TURBO; ++i) {
        turbo_s1n[i] = TURBO_INV_SQRT_128*turbo_s1[i];
        turbo_s2n[i] = TURBO_INV_SQRT_128*turbo_s2[i];
    }
    turbo_planes_init();
}

static void turbo_unpack(int t, const void * vb, uint8_t * GGML_RESTRICT idx, uint8_t * GGML_RESTRICT neg) {
#if defined(TURBO_NEON)
    turbo_unpack_neon(t, vb, idx, neg);
#elif defined(TURBO_AVX2)
    turbo_unpack_avx2(t, vb, idx, neg);
#else
    turbo_unpack_scalar(t, vb, idx, neg);
#endif
}

static void turbo_rotate128(float * GGML_RESTRICT x, int direction) {
#if defined(TURBO_NEON)
    turbo_rotate_neon(x, direction == 0 ? turbo_s1 : turbo_s2, direction == 0 ? turbo_s2n : turbo_s1n);
#elif defined(TURBO_AVX2)
    turbo_rotate_avx2(x, direction == 0 ? turbo_s1 : turbo_s2, direction == 0 ? turbo_s2n : turbo_s1n);
#else
    ggml_turbo_rotate_ref(x, direction);
#endif
}

void ggml_cpu_turbo_rotate(float * GGML_RESTRICT x, int64_t n, int direction) {
    assert(n % QK_TURBO == 0);
    for (int64_t i = 0; i < n; i += QK_TURBO) {
        turbo_rotate128(x + i, direction);
    }
}

//
// quantize (cache writes)
//

static void turbo_quantize_row(int t, const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    assert(k % QK_TURBO == 0);
#if defined(TURBO_NEON) || defined(TURBO_AVX2)
    const float * mid   = turbo_mid(t);
    const int     n_mid = turbo_n_mid(t);
    const size_t  bs    = turbo_block_size(t);

    float   r[QK_TURBO];
    uint8_t codes[QK_TURBO];
    uint8_t idx[QK_TURBO];
    uint8_t neg[QK_TURBO];

    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib) {
        const float * xb = x + ib*QK_TURBO;
        void        * yb = (char *) vy + ib*bs;

#if defined(TURBO_NEON)
        const float norm = sqrtf(turbo_sumsq_neon(xb));
#else
        const float norm = sqrtf(turbo_sumsq_avx2(xb));
#endif
        const float inv = norm > 1e-10f ? 1.0f/norm : 0.0f;
        for (int i = 0; i < QK_TURBO; ++i) {
            r[i] = xb[i]*inv;
        }
        turbo_rotate128(r, 0);

#if defined(TURBO_NEON)
        turbo_codes_neon(r, mid, n_mid, codes);
#else
        turbo_codes_avx2(r, mid, n_mid, codes);
#endif

        // norm correction: sum of the squared centroids, same lane order as the reference
        const uint8_t * pi = codes;
        const uint8_t * pn = NULL;
        if (t == 5) {
            turbo5_split_codes(codes, idx, neg);
            pi = idx;
            pn = neg;
        }
        float recon_sq;
#if defined(TURBO_NEON)
        {
            float32x4_t a[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
            for (int i = 0; i < QK_TURBO; i += 16) {
                float32x4_t v[4];
                turbo_lookup16_neon(t, pi + i, pn ? pn + i : NULL, v);
                for (int q = 0; q < 4; ++q) {
                    a[q] = vfmaq_f32(a[q], v[q], v[q]);
                }
            }
            recon_sq = turbo_reduce16_neon(a[0], a[1], a[2], a[3]);
        }
#else
        {
            const float * lut = turbo_lut_pad[t];
            __m256 a = _mm256_setzero_ps();
            __m256 b = _mm256_setzero_ps();
            for (int i = 0; i < QK_TURBO; i += 16) {
                const __m256 v0 = turbo_lookup8_avx2(t, lut, pi + i,     pn ? pn + i     : NULL);
                const __m256 v1 = turbo_lookup8_avx2(t, lut, pi + i + 8, pn ? pn + i + 8 : NULL);
                a = _mm256_fmadd_ps(v0, v0, a);
                b = _mm256_fmadd_ps(v1, v1, b);
            }
            recon_sq = turbo_reduce16_avx2(a, b);
        }
#endif
        const float recon = sqrtf(recon_sq);
        *(ggml_half *) yb = GGML_CPU_FP32_TO_FP16(recon > 1e-10f ? norm/recon : norm);
        turbo_pack(t, codes, yb);
    }
#else
    switch (t) {
        case 2: quantize_row_turbo2_0_ref(x, (block_turbo2_0 *) vy, k); break;
        case 3: quantize_row_turbo3_0_ref(x, (block_turbo3_0 *) vy, k); break;
        case 4: quantize_row_turbo4_0_ref(x, (block_turbo4_0 *) vy, k); break;
        case 5: quantize_row_turbo5_0_ref(x, (block_turbo5_0 *) vy, k); break;
        case 6: quantize_row_turbo6_0_ref(x, (block_turbo6_0 *) vy, k); break;
        default: GGML_ABORT("invalid turbo type");
    }
#endif
}

void quantize_row_turbo2_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { turbo_quantize_row(2, x, y, k); }
void quantize_row_turbo3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { turbo_quantize_row(3, x, y, k); }
void quantize_row_turbo4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { turbo_quantize_row(4, x, y, k); }
void quantize_row_turbo5_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { turbo_quantize_row(5, x, y, k); }
void quantize_row_turbo6_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) { turbo_quantize_row(6, x, y, k); }

//
// decode
//

static void turbo_dequantize_row(int t, const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO == 0);
    const size_t bs = turbo_block_size(t);

    uint8_t idx[QK_TURBO];
    uint8_t neg[QK_TURBO];

    for (int64_t ib = 0; ib < k/QK_TURBO; ++ib, y += QK_TURBO) {
        const void * b    = (const char *) vx + ib*bs;
        const float  norm = turbo_block_norm(b);
        turbo_unpack(t, b, idx, neg);
        const uint8_t * pn = t == 5 ? neg : NULL;
#if defined(TURBO_NEON)
        const float32x4_t vn = vdupq_n_f32(norm);
        for (int i = 0; i < QK_TURBO; i += 16) {
            float32x4_t v[4];
            turbo_lookup16_neon(t, idx + i, pn ? pn + i : NULL, v);
            for (int q = 0; q < 4; ++q) {
                vst1q_f32(y + i + 4*q, vmulq_f32(v[q], vn));
            }
        }
#elif defined(TURBO_AVX2)
        const __m256 vn  = _mm256_set1_ps(norm);
        const float * lut = turbo_lut_pad[t];
        for (int i = 0; i < QK_TURBO; i += 8) {
            _mm256_storeu_ps(y + i, _mm256_mul_ps(turbo_lookup8_avx2(t, lut, idx + i, pn ? pn + i : NULL), vn));
        }
#else
        const float * lut = turbo_lut(t);
        for (int i = 0; i < QK_TURBO; ++i) {
            const float v = lut[idx[i]];
            y[i] = (pn && pn[i] ? -v : v)*norm;
        }
#endif
    }
}

static void dequantize_row_turbo2_0_cpu(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { turbo_dequantize_row(2, x, y, k); }
static void dequantize_row_turbo3_0_cpu(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { turbo_dequantize_row(3, x, y, k); }
static void dequantize_row_turbo4_0_cpu(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { turbo_dequantize_row(4, x, y, k); }
static void dequantize_row_turbo5_0_cpu(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { turbo_dequantize_row(5, x, y, k); }
static void dequantize_row_turbo6_0_cpu(const void * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) { turbo_dequantize_row(6, x, y, k); }

ggml_to_float_t ggml_cpu_get_to_float(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0: return dequantize_row_turbo2_0_cpu;
        case GGML_TYPE_TURBO3_0: return dequantize_row_turbo3_0_cpu;
        case GGML_TYPE_TURBO4_0: return dequantize_row_turbo4_0_cpu;
        case GGML_TYPE_TURBO5_0: return dequantize_row_turbo5_0_cpu;
        case GGML_TYPE_TURBO6_0: return dequantize_row_turbo6_0_cpu;
        default:                 return ggml_get_type_traits(type)->to_float;
    }
}

//
// dot products with an F32 vector (the rotated Q)
//

static float turbo_vec_dot(int t, int n, const void * GGML_RESTRICT vx, const float * GGML_RESTRICT y) {
    assert(n % QK_TURBO == 0);
    const size_t bs = turbo_block_size(t);

    uint8_t idx[QK_TURBO];
    uint8_t neg[QK_TURBO];

    float sum = 0.0f;
    for (int ib = 0; ib < n/QK_TURBO; ++ib, y += QK_TURBO) {
        const void * b = (const char *) vx + ib*bs;
        turbo_unpack(t, b, idx, neg);
        const uint8_t * pn = t == 5 ? neg : NULL;
        float sb;
#if defined(TURBO_NEON)
        float32x4_t a[4] = { vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f), vdupq_n_f32(0.0f) };
        for (int i = 0; i < QK_TURBO; i += 16) {
            float32x4_t v[4];
            turbo_lookup16_neon(t, idx + i, pn ? pn + i : NULL, v);
            for (int q = 0; q < 4; ++q) {
                a[q] = vfmaq_f32(a[q], v[q], vld1q_f32(y + i + 4*q));
            }
        }
        sb = vaddvq_f32(vaddq_f32(vaddq_f32(a[0], a[1]), vaddq_f32(a[2], a[3])));
#elif defined(TURBO_AVX2)
        const float * lut = turbo_lut_pad[t];
        __m256 a0 = _mm256_setzero_ps();
        __m256 a1 = _mm256_setzero_ps();
        for (int i = 0; i < QK_TURBO; i += 16) {
            a0 = _mm256_fmadd_ps(turbo_lookup8_avx2(t, lut, idx + i,     pn ? pn + i     : NULL), _mm256_loadu_ps(y + i),     a0);
            a1 = _mm256_fmadd_ps(turbo_lookup8_avx2(t, lut, idx + i + 8, pn ? pn + i + 8 : NULL), _mm256_loadu_ps(y + i + 8), a1);
        }
        const __m256 s = _mm256_add_ps(a0, a1);
        __m128 u = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        u = _mm_add_ps(u, _mm_movehl_ps(u, u));
        sb = _mm_cvtss_f32(_mm_add_ss(u, _mm_shuffle_ps(u, u, 1)));
#else
        const float * lut = turbo_lut(t);
        sb = 0.0f;
        for (int i = 0; i < QK_TURBO; ++i) {
            const float v = lut[idx[i]];
            sb += (pn && pn[i] ? -v : v)*y[i];
        }
#endif
        sum += sb*turbo_block_norm(b);
    }
    return sum;
}

#define TURBO_VEC_DOT(N)                                                                                      \
void ggml_vec_dot_turbo##N##_0_f32(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, \
        size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc) {                                      \
    assert(nrc == 1);                                                                                         \
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);                                      \
    *s = turbo_vec_dot(N, n, vx, (const float *) vy);                                                         \
}

TURBO_VEC_DOT(2)
TURBO_VEC_DOT(3)
TURBO_VEC_DOT(4)
TURBO_VEC_DOT(5)
TURBO_VEC_DOT(6)
