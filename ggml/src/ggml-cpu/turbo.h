#pragma once

#include "ggml.h"

// TurboQuant KV cache types: CPU helpers (turbo.c)

#ifdef __cplusplus
extern "C" {
#endif

// in-place rotation of n values (n % 128 == 0), direction 0 = forward, 1 = inverse
void ggml_cpu_turbo_rotate(float * GGML_RESTRICT x, int64_t n, int direction);

// to_float for the CPU backend: the SIMD decoder of a turbo type, the generic to_float of any other type
ggml_to_float_t ggml_cpu_get_to_float(enum ggml_type type);

void ggml_cpu_turbo_init(void);

#ifdef __cplusplus
}
#endif
