# Performance Analysis: FLOPs Budget Before Optimizing

## Why this matters

Before benchmarking a new model's cache-dit optimizations (cache / CP / TP / quantization
/ compile), compute the **FLOPs budget** first. The split between attention and GEMM
compute decides which optimizations can possibly pay off:

- `fp8-per-row` / `svdq` quantization only accelerates **GEMM** (the `nn.Linear` calls).
- `--compile` fuses elementwise ops and (marginally) improves GEMM scheduling; it **never
  rewrites the attention kernel**.
- Context parallelism and cache are the only levers that shrink the **attention** share
  (CP splits the O(S^2) work across ranks, cache skips whole steps).
- `--attn ffpa_fp8 / ffpa_fp4` quantizes the attention computation itself — the only
  single-GPU lever against an attention-dominated model.

A model where attention is 90% of the FLOPs **cannot** show a meaningful fp8 speedup no
matter how well the quantization is wired: 2x on 10% of the time saves 5%. This is the
Motif-Video lesson (PR #1142): fp8-per-row measured 1.08x and svdq-nvfp4 1.09x and both
were **at their theoretical ceiling** — the integration was fine; the workload was
attention-dominated. `--attn ffpa_fp8` then delivered 2.14x on the same model, confirming
the analysis.

## The formulas

Given per-block: hidden size `d`, heads `H`, head dim `dh = d/H`, FFN inner `dff`,
sequence length `S` (joint: image + text tokens), image tokens `N`, text tokens `T`:

```
F_attn  (per attention layer) = 4 * S^2 * dh * H            # QK^T + PV, both O(S^2)
F_gemm  (per Linear)          = 2 * d_in * d_out * tokens   # O(S), tokens = its作用域
```

Key subtleties that change the answer by orders of magnitude:

1. **Sequence length is NOT the token count of one frame.** Multiply the full latent
   grid: `S = ceil(F_frames / (2 * patch_t)) * (H_px / (8 * patch_h)) * (W_px / (8 * patch_w)) + T`.
   Motif-Video 480x832x81f with patch (1,2,2) and VAE /8 gives
   `41 * 30 * 52 + 512 = 64472` — a 64k sequence. Getting this wrong by 20x invalidates
   the whole budget (a naive "1600 tokens" guess made attention look cheap).
2. **A projection's GEMM cost scales with ITS OWN token count.** Joint-attention
   `add_q/add_k/add_v` run on the text stream (`T` ~ 512), not the image stream; AdaLN
   `norm1.linear` runs **per-sample** (batch dim only) and is negligible — do not count
   it as per-token.
3. Attention is quadratic in `S`; GEMM is linear. Long-sequence video models (large
   resolutions * frame counts * small patch sizes) are therefore attention-dominated;
   small-sequence image models (e.g. 1-4k tokens) are GEMM-dominated. The crossover
   sits roughly where `4 * S * dh * H ≈ Σ 2 * d_in * d_out` over the block's linears.

Worked example (Motif-Video-2B, 480x832x81f): attention ~919 TFLOPs/forward, GEMM
~102 TFLOPs/forward, elementwise ~3 TFLOPs → **attention 90% / GEMM 10%**. Sanity-check
the budget against measured wall time (GPU peak rates: GEMM ~400 TFLOPS bf16, cudnn
SDPA ~350 TFLOPS effective): predicted ~3.7s per (dual-CFG) step vs measured 7.4s — a
match within elementwise/launch overhead validates the whole model of the workload.

## The diagnostic ladder (when a speedup looks wrong)

Use this order — each step is cheap and eliminates one hypothesis:

1. **Transformer-level bit-exactness** (functor on/off) rules out integration bugs
   before touching performance. If bit-exact, the number is the model's truth.
2. **Check quantization coverage from logs**: regional line should list the block
   classes, "Quantized Linear Layers: N (total)" with 0 skipped. If N is small, suspect
   a missing `_repeated_blocks` declaration (see `declare_repeated_blocks` in
   `cache_dit/utils.py`; diffusers models mostly declare it themselves).
3. **`torch._dynamo.explain(block)`** counts graph breaks on a real block — rules out
   (or in) the "inspect.signature / graph break" class of compile failures. Note: modern
   Dynamo handles `inspect.signature(processor.__call__)` (Motif measured 0 breaks).
4. **Rule out SDPA math fallback by memory, not by timing**: math materializes
   `(B, H, S, S)` additive/bool mask. Motif at 64k would need ~96 GB; the process ran in
   16.8 GB — math impossible, no profiling needed. (The classic "compile dispatches SDPA
   to math" failure shows up as an OOM or 10x slowdown, never as a 1.02x.)
5. **Back out the VAE share from cache timing**: `30t + X = t_base`,
   `k*t + X = t_cache` (k = effective cached-step fraction) gives both the per-step
   transformer time and the VAE + pre/post floor `X`. Motif: X ≈ 66.5s = 23% of
   end-to-end — every transformer-side speedup is diluted by 1/(1 - 0.23).
6. **FLOPs budget** (above) to set the theoretical ceiling of each optimization, then
   compare measured vs ceiling. Measured ≈ ceiling → nothing left to fix; measured ≪
   ceiling → integration issue worth digging.

## Motif-Video reference results (RTX PRO 5000, 480x832x81f x 30 steps)

| Configuration | Time | Speedup | PSNR | SSIM |
|---|---|---|---|---|
| bf16 baseline | 287.6s | 1.00x | - | - |
| DBCache (thr 0.12) | 158.7s | 1.81x | 41.39 | 0.9706 |
| CP2 (Ulysses) | 159.5s | 1.80x | 42.33 | 0.9719 |
| CP2 + fp8 + compile | 150.1s | 1.92x | 35.25 | 0.9556 |
| fp8-per-row + compile | 265.0s | 1.08x | 35.51 | 0.9544 |
| svdq-nvfp4-r64 + compile | 264.5s | 1.09x | 32.22 | 0.9332 |
| compile only | 282.2s | 1.02x | 42.44 | 0.9716 |
| **--attn ffpa_fp8 (1 GPU)** | **134.2s** | **2.14x** | **41.55** | **0.9702** |
| --attn ffpa_fp4 (1 GPU) | 119.1s | 2.41x | 32.43 | 0.9152 |
| **--attn ffpa_fp4 + cache** | **69.7s** | **4.12x** | 32.40 | 0.9141 |

Readings from this table:

- GEMM-only quantization (fp8/svdq) at 1.08-1.09x **matches the 10% GEMM share
  ceiling** — integration was never the problem.
- `--compile` at 1.02x matches the ~0.3% elementwise share it can fuse.
- `ffpa_fp8` is the accuracy-preserving single-GPU winner (2.14x @ 41.6 dB); `ffpa_fp4`
  trades ~9 dB for another 0.3x; `ffpa_fp4 + cache` stacks to 4.12x with no extra
  accuracy loss on top of fp4.
- **CP + ffpa is currently blocked by the mask allowlist**: templated CP dispatch only
  accepts `native / cudnn / flash_varlen / npu` ops with a non-None `attn_mask`
  (`cache_dit/attention/backends/register.py`, `_ATTENTION_OPS_ALLOW_ATTN_MASK`).
  Models with a persistent text-padding mask (Motif t2v) must choose between CP and
  ffpa until ffpa ops join the allowlist.

## Next: turn the budget into a table

Once the budget says which levers *can* pay off, run the matrix instead of ad-hoc commands:
[`./perf_matrix.md`](./perf_matrix.md) — row checklist (Tier A 1/2 GPUs by default, Tier B 4 GPUs /
Tier C 8 GPUs on request), command templates, and the measurement/table conventions. Pass/fail gates:
[`./testing.md`](./testing.md) §7.6.
