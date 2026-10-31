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
// MoE weighted expert sum: dst[:, t] = sum_i experts[:, i, t]*weights[0, i, t], with the rounding of the unfused
// MUL followed by the ADD chain ((m_0 + m_1) + m_2) + ...
void ggml_compute_forward_moe_weighted_sum_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst,
        const struct ggml_tensor * experts, const struct ggml_tensor * weights, int n);

void ggml_compute_forward_add_n_f32(const struct ggml_compute_params * params, struct ggml_tensor * dst, const struct ggml_tensor ** srcs, int n);

#ifdef __cplusplus
}
#endif
