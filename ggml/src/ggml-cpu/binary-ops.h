#pragma once

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

void ggml_compute_forward_add_non_quantized(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_sub(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_mul(const struct ggml_compute_params * params, struct ggml_tensor * dst);
void ggml_compute_forward_div(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// fused chain of n - 1 F32 additions: dst = ((srcs[0] + srcs[1]) + srcs[2]) + ... + srcs[n - 1]
// all tensors have the shape of dst and contiguous rows; the summation order matches the unfused chain
void ggml_compute_forward_add_n_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst, const struct ggml_tensor ** srcs, int n);

#ifdef __cplusplus
}
#endif
