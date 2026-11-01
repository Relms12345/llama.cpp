// TurboQuant KV cache types (turbo2..turbo6) on the CPU backend:
//  - the CPU quantizer (SIMD where available) writes the same bytes as the reference codec
//  - round trip quality (decode + inverse rotation) per type
//  - the CPU dot product matches the decoded values
//  - GGML_OP_TURBO_WHT: inverse(forward(x)) == x, norms preserved
//  - attention with rotated Q, turbo K/V and the inverse rotation of the output matches F32 attention

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static const ggml_type k_types[]   = { GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO5_0, GGML_TYPE_TURBO6_0 };
static const double    k_min_cos[] = { 0.92, 0.975, 0.99, 0.997, 0.999 }; // per-vector round trip
static const double    k_min_att[] = { 0.80, 0.93, 0.975, 0.993, 0.997 }; // attention output, random data (loose)

static std::vector<float> make_data(std::mt19937 & rng, int64_t n, bool odd_blocks = true, bool outliers = true) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    std::vector<float> x(n);
    for (int64_t i = 0; i < n; ++i) {
        x[i] = nd(rng);
        if (outliers && ud(rng) < 0.01f) {
            x[i] *= 20.0f; // outliers, as in real K caches
        }
    }
    // a few blocks with unusual scales: zero, tiny, huge
    if (odd_blocks && n >= 4*128) {
        std::fill(x.begin(), x.begin() + 128, 0.0f);
        for (int i = 128; i < 256; ++i) x[i] *= 1e-6f;
        for (int i = 256; i < 384; ++i) x[i] *= 1e3f;
    }
    return x;
}

static double cosine(const float * a, const float * b, int64_t n) {
    double ab = 0, aa = 0, bb = 0;
    for (int64_t i = 0; i < n; ++i) {
        ab += (double) a[i]*b[i];
        aa += (double) a[i]*a[i];
        bb += (double) b[i]*b[i];
    }
    if (aa == 0 && bb == 0) return 1.0;
    return ab/std::sqrt(aa*bb + 1e-300);
}

static ggml_context * new_ctx(size_t mem) {
    ggml_init_params params = { mem, nullptr, false };
    return ggml_init(params);
}

// y = rotate(x) for rows of ne0 values, direction 0 = forward, 1 = inverse
static std::vector<float> rotate(const std::vector<float> & x, int64_t ne0, int direction, int n_threads) {
    ggml_context * ctx = new_ctx(64*1024*1024);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne0, (int64_t) x.size()/ne0);
    memcpy(a->data, x.data(), ggml_nbytes(a));
    ggml_tensor * r = ggml_turbo_wht(ctx, a, direction);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, r);
    ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    std::vector<float> y(x.size());
    memcpy(y.data(), r->data, ggml_nbytes(r));
    ggml_free(ctx);
    return y;
}

static int test_codec(ggml_type type, double min_cos, std::mt19937 & rng) {
    const char * name = ggml_type_name(type);
    const int64_t n = 128*64;

    const auto * tr  = ggml_get_type_traits(type);
    const auto * trc = ggml_get_type_traits_cpu(type);

    int fails = 0;

    const std::vector<float> x = make_data(rng, n);
    const size_t row_size = ggml_row_size(type, n);

    // 1. CPU quantizer == reference quantizer, byte for byte
    std::vector<uint8_t> q_ref(row_size), q_cpu(row_size);
    tr->from_float_ref(x.data(), q_ref.data(), n);
    trc->from_float(x.data(), q_cpu.data(), n);
    size_t n_diff = 0;
    for (size_t i = 0; i < row_size; ++i) {
        n_diff += q_ref[i] != q_cpu[i];
    }
    if (n_diff) {
        printf("  %-7s FAIL: CPU quantizer differs from the reference in %zu of %zu bytes\n", name, n_diff, row_size);
        fails++;
    }

    // 2. round trip: decode, inverse rotation, compare per 128-value block
    std::vector<float> d(n);
    tr->to_float(q_cpu.data(), d.data(), n);
    const std::vector<float> back = rotate(d, 128, 1, 2);
    double worst = 1.0;
    double mean  = 0.0;
    for (int64_t ib = 0; ib < n/128; ++ib) {
        const double c = cosine(x.data() + 128*ib, back.data() + 128*ib, 128);
        worst = std::min(worst, c);
        mean += c;
    }
    mean /= n/128;
    if (worst < min_cos || !std::isfinite(worst)) {
        printf("  %-7s FAIL: round trip cosine worst %.5f mean %.5f (min %.3f)\n", name, worst, mean, min_cos);
        fails++;
    }

    // 3. CPU dot product == dot product of the decoded values
    std::vector<float> y = make_data(rng, n);
    double max_rel = 0.0;
    for (int64_t len : { (int64_t) 128, (int64_t) 256, (int64_t) 640, n }) {
        for (int64_t off = 0; off + len <= n; off += 3*len) {
            float s = 0.0f;
            trc->vec_dot((int) len, &s, 0, q_cpu.data() + (off/128)*ggml_type_size(type), 0, y.data() + off, 0, 1);
            double ref = 0.0, mag = 0.0;
            for (int64_t i = 0; i < len; ++i) {
                ref += (double) d[off + i]*y[off + i];
                mag += std::fabs((double) d[off + i]*y[off + i]);
            }
            max_rel = std::max(max_rel, std::fabs(s - ref)/(mag + 1e-30));
        }
    }
    if (max_rel > 1e-5) {
        printf("  %-7s FAIL: dot product relative error %.3g\n", name, max_rel);
        fails++;
    }

    printf("  %-7s quantizer bytes differ: %zu, round trip cosine worst %.5f mean %.5f, dot rel err %.2g -> %s\n",
            name, n_diff, worst, mean, max_rel, fails ? "FAIL" : "OK");
    return fails;
}

static int test_rotation(std::mt19937 & rng) {
    int fails = 0;
    const int64_t ne0 = 384;
    const std::vector<float> x = make_data(rng, ne0*7);
    const std::vector<float> f = rotate(x, ne0, 0, 3);
    const std::vector<float> b = rotate(f, ne0, 1, 3);
    double max_err = 0.0, max_norm_err = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_err = std::max(max_err, (double) std::fabs(x[i] - b[i])/(1.0 + std::fabs(x[i])));
    }
    for (size_t g = 0; g < x.size()/128; ++g) {
        double nx = 0, nf = 0;
        for (int i = 0; i < 128; ++i) {
            nx += (double) x[128*g + i]*x[128*g + i];
            nf += (double) f[128*g + i]*f[128*g + i];
        }
        max_norm_err = std::max(max_norm_err, std::fabs(std::sqrt(nx) - std::sqrt(nf))/(std::sqrt(nx) + 1e-30));
    }
    if (max_err > 1e-5 || max_norm_err > 1e-5) {
        fails++;
    }
    printf("  rotation: inverse(forward(x)) max err %.2g, norm err %.2g -> %s\n", max_err, max_norm_err, fails ? "FAIL" : "OK");
    return fails;
}

// attention of one sequence: reference with F32 K/V vs turbo K/V with the rotations the llama graph adds
static int test_attention(ggml_type type_k, ggml_type type_v, double min_cos, int64_t n_kv, int64_t n_q) {
    std::mt19937 rng(42 + 1000*type_k + 10*type_v + n_kv + n_q); // independent of the order of the tests
    const int64_t D = 128, n_head = 8, n_head_kv = 2;

    ggml_context * ctx = new_ctx(256*1024*1024);

    std::vector<float> qd = make_data(rng, D*n_q*n_head,     false, false);
    std::vector<float> kd = make_data(rng, D*n_kv*n_head_kv, false, false);
    std::vector<float> vd = make_data(rng, D*n_kv*n_head_kv, false, false);

    // layouts as in build_attn_mha after the permutes: q [D, n_q, n_head], k/v [D, n_kv, n_head_kv]
    ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_q, n_head);
    ggml_tensor * kf = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, n_head_kv);
    ggml_tensor * vf = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D, n_kv, n_head_kv);
    memcpy(q->data, qd.data(), ggml_nbytes(q));
    ggml_fp32_to_fp16_row(kd.data(), (ggml_fp16_t *) kf->data, ggml_nelements(kf));
    ggml_fp32_to_fp16_row(vd.data(), (ggml_fp16_t *) vf->data, ggml_nelements(vf));

    ggml_tensor * kt = ggml_new_tensor_3d(ctx, type_k, D, n_kv, n_head_kv);
    ggml_tensor * vt = ggml_new_tensor_3d(ctx, type_v, D, n_kv, n_head_kv);
    // the cache holds K/V as written by SET_ROWS (the CPU quantizer)
    for (auto [t, src] : { std::pair<ggml_tensor *, const std::vector<float> *>{ kt, &kd }, { vt, &vd } }) {
        ggml_get_type_traits_cpu(t->type)->from_float(src->data(), t->data, ggml_nelements(t));
    }

    const float scale = 1.0f/std::sqrt((float) D);

    ggml_tensor * ref = ggml_flash_attn_ext(ctx, q, kf, vf, nullptr, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(ref, GGML_PREC_F32);

    ggml_tensor * qt = ggml_is_turbo(type_k) ? ggml_turbo_wht(ctx, q, 0) : q;
    ggml_tensor * out = ggml_flash_attn_ext(ctx, qt, kt, vt, nullptr, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    if (ggml_is_turbo(type_v)) {
        out = ggml_turbo_wht(ctx, out, 1);
    }

    // the same attention over the decoded (F32) K/V: checks the kernels, without the quantization error
    ggml_tensor * kd32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_kv, n_head_kv);
    ggml_tensor * vd32 = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, n_kv, n_head_kv);
    for (auto [dst, src] : { std::pair<ggml_tensor *, ggml_tensor *>{ kd32, kt }, { vd32, vt } }) {
        ggml_get_type_traits(src->type)->to_float(src->data, (float *) dst->data, ggml_nelements(src));
    }
    ggml_tensor * exact = ggml_flash_attn_ext(ctx, qt, kd32, vd32, nullptr, scale, 0.0f, 0.0f);
    ggml_prec_set_acc(exact, GGML_PREC_F32);
    if (ggml_is_turbo(type_v)) {
        exact = ggml_turbo_wht(ctx, exact, 1);
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, ref);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, exact);
    ggml_graph_compute_with_ctx(ctx, gf, 4);

    double max_err = 0.0, max_ref = 0.0;
    for (int64_t i = 0; i < ggml_nelements(out); ++i) {
        max_err = std::max(max_err, (double) std::fabs(((const float *) out->data)[i] - ((const float *) exact->data)[i]));
        max_ref = std::max(max_ref, (double) std::fabs(((const float *) exact->data)[i]));
    }
    const double kern_err = max_err/(max_ref + 1e-30);

    const double c = cosine((const float *) ref->data, (const float *) out->data, ggml_nelements(ref));
    // with F16/Q8_0 K the kernel converts Q to that type, the decoded reference does not
    const double max_kern_err = ggml_is_turbo(type_k) && ggml_is_turbo(type_v) ? 1e-4 :
                                type_k == GGML_TYPE_Q8_0                         ? 1e-1 : 1e-2;
    const bool ok = c >= min_cos && std::isfinite(c) && kern_err < max_kern_err;
    printf("  attention K %-7s V %-7s cosine vs F16 %.6f (min %.3f), kernel err vs decoded K/V %.2g -> %s\n",
            ggml_type_name(type_k), ggml_type_name(type_v), c, min_cos, kern_err, ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok ? 0 : 1;
}

// the non flash attention path of the llama graph casts a permuted view of the turbo V cache to F32
static int test_cast_view(ggml_type type, std::mt19937 & rng) {
    const int64_t D = 128, n_kv = 37, n_head_kv = 3;
    ggml_context * ctx = new_ctx(64*1024*1024);

    ggml_tensor * v = ggml_new_tensor_3d(ctx, type, D*n_head_kv, n_kv, 1); // [n_embd_v_gqa, kv_size] like the cache
    const std::vector<float> vd = make_data(rng, D*n_head_kv*n_kv, false);
    ggml_get_type_traits_cpu(type)->from_float(vd.data(), v->data, ggml_nelements(v));

    // the view of get_v, permuted as in build_attn_mha: [D, n_kv, n_head_kv]
    ggml_tensor * view = ggml_view_3d(ctx, v, D, n_head_kv, n_kv, ggml_row_size(type, D), v->nb[1], 0);
    ggml_tensor * perm = ggml_permute(ctx, view, 0, 2, 1, 3);
    ggml_tensor * f    = ggml_cast(ctx, perm, GGML_TYPE_F32);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);
    ggml_graph_compute_with_ctx(ctx, gf, 2);

    std::vector<float> dec(ggml_nelements(v));
    ggml_get_type_traits(type)->to_float(v->data, dec.data(), ggml_nelements(v));

    int bad = 0;
    for (int64_t h = 0; h < n_head_kv; ++h) {
        for (int64_t c = 0; c < n_kv; ++c) {
            for (int64_t d = 0; d < D; ++d) {
                const float a = ((const float *) f->data)[(h*n_kv + c)*D + d];
                const float b = dec[c*D*n_head_kv + h*D + d];
                bad += a != b;
            }
        }
    }
    printf("  cast of a permuted %-7s view to F32: %d mismatches -> %s\n", ggml_type_name(type), bad, bad ? "FAIL" : "OK");
    ggml_free(ctx);
    return bad ? 1 : 0;
}

int main(void) {
    ggml_cpu_init();

    std::mt19937 rng(1234);
    int fails = 0;

    printf("codec:\n");
    for (size_t i = 0; i < sizeof(k_types)/sizeof(k_types[0]); ++i) {
        fails += test_codec(k_types[i], k_min_cos[i], rng);
    }

    printf("cast:\n");
    for (ggml_type t : k_types) {
        fails += test_cast_view(t, rng);
    }

    printf("rotation:\n");
    fails += test_rotation(rng);

    printf("attention:\n");
    // n_q = 1: decode kernels (per head, grouped, split KV for n_kv >= 512), n_q > 1: tiled kernel
    for (auto [n_kv, n_q] : { std::pair<int64_t, int64_t>{ 256, 1 }, { 1024, 1 }, { 300, 5 } }) {
        printf(" n_kv = %lld, n_q = %lld\n", (long long) n_kv, (long long) n_q);
        for (size_t i = 0; i < sizeof(k_types)/sizeof(k_types[0]); ++i) {
            fails += test_attention(k_types[i], k_types[i], k_min_att[i], n_kv, n_q);
            fails += test_attention(GGML_TYPE_F16, k_types[i], k_min_att[i], n_kv, n_q);
            fails += test_attention(k_types[i], GGML_TYPE_F16, k_min_att[i], n_kv, n_q);
        }
        fails += test_attention(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0, k_min_att[0], n_kv, n_q);
        fails += test_attention(GGML_TYPE_TURBO6_0, GGML_TYPE_TURBO3_0, k_min_att[1], n_kv, n_q);
        fails += test_attention(GGML_TYPE_Q8_0,     GGML_TYPE_TURBO3_0, k_min_att[1], n_kv, n_q);
    }

    printf("%s\n", fails ? "FAILED" : "all tests passed");
    return fails ? 1 : 0;
}
