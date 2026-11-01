# TurboQuant KV cache types

`turbo2` ... `turbo6` are KV cache types that rotate each group of 128 K/V values before quantizing them to 2-6 bits.
The rotation (random signs, Walsh-Hadamard transform, random signs) makes the values close to Gaussian, so a fixed
Lloyd-Max codebook fits every block, with one fp16 norm per block of 128 values.
Ported from [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (KV cache types only).

| type     | bits/value | vs f16 | vs q8_0 | codebook               |
|----------|-----------:|-------:|--------:|------------------------|
| `turbo2` | 2.125      | 7.5x   | 4.0x    | 4 centroids            |
| `turbo3` | 3.125      | 5.1x   | 2.7x    | 8 centroids            |
| `turbo4` | 4.125      | 3.9x   | 2.1x    | 16 centroids           |
| `turbo5` | 5.125      | 3.1x   | 1.7x    | 32 centroids (sign + 16 magnitudes) |
| `turbo6` | 6.125      | 2.6x   | 1.4x    | 64 centroids           |

They are runtime-only: they are never stored in model files.

## Usage

```bash
llama-cli    -m model.gguf -ctk turbo4 -ctv turbo3
llama-server -m model.gguf -ctk q8_0   -ctv turbo4
llama-bench  -m model.gguf -ctk turbo4 -ctv turbo4,turbo3,q8_0
```

Any K/V combination of the turbo types with `f16`, `q8_0`, `q4_0`, ... is allowed. Flash attention is required and is
enabled automatically when `-fa` is `auto` (the default); `-fa off` with a turbo cache is an error.

Backends: CPU (scalar, AArch64 NEON, x86 AVX2) and Vulkan. Other GPU backends do not support the types: keep the KV
cache on the CPU there (`-nkvo`), or use another cache type. On Vulkan, turbo flash attention always uses the scalar
shader (not coopmat1/coopmat2).

## How it is wired

- Cache writes (`SET_ROWS`) normalize, rotate and quantize each 128-value group. The cache holds rotated values.
- The attention graph rotates Q with the same transform (`GGML_OP_TURBO_WHT`), so the scores do not change, and rotates
  the attention output back when V is a turbo type.
- Heads that are not a multiple of 128 are zero-padded to the next multiple of 128 in the cache. The scores do not
  change, but the memory does: with head size 64, a turbo cache takes twice the bits per value shown above (turbo4 at head
  size 64 is about as large as q8_0). Turbo types pay off for head sizes 128 and 256.
- Context shift (K-shift) is not possible with a turbo K cache: RoPE cannot be applied to rotated, quantized keys.
  `--context-shift` is disabled automatically, as for other models that cannot shift.
- MLA models (DeepSeek V2/V3 and similar) are not supported.

## Environment variables

| variable                 | default | effect |
|--------------------------|---------|--------|
| `TURBO_AUTO_ASYMMETRIC`  | `1`     | with the same turbo type for K and V and a GQA ratio >= 6, K is stored as `q8_0` (the K error is shared by every q head of a K/V head). `0` disables it |
| `TURBO_LAYER_ADAPTIVE`   | `7` if V is `turbo2`, else `0` | per-layer types for models with >= 8 layers: `1` K/V of the first 4 and last 4 layers in `q8_0`; `2` K/V of the last 8 layers in `q8_0`; `5` V of the first 2 and last 2 layers in `turbo4`, other V layers `turbo2`; `6` V of the last 8 layers in `turbo4`, other V layers `turbo2`; `7` V of the first 2 and last 2 layers in `q8_0`, other V layers `turbo2`; `0` off |

The upstream attention rotation (`LLAMA_ATTN_ROT_DISABLE`) is unchanged for the other cache types; it is skipped for
turbo caches, which rotate by themselves.

## Quality

Check the quality on your model before relying on a turbo cache: compare against an f16 cache with
`llama-perplexity --kl-divergence-base` / `--kl-divergence`. Models with attention sinks (gpt-oss) are sensitive to any
K quantization; prefer quantizing V only there. As a rule of thumb, `turbo4`/`turbo5` V with `q8_0` or `turbo6` K is the
safe end, `turbo2` the aggressive end.

## Tests

- `test-turbo-quant`: the CPU quantizer (SIMD) writes the same bytes as the reference codec, round-trip quality of each
  type, the dot products, the rotation op, and the attention kernels against attention over the decoded K/V.
- `test-backend-ops -o SET_ROWS`, `-o TURBO_WHT`, `-o FLASH_ATTN_EXT`: the Vulkan kernels against the CPU.
