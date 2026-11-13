# Cache-DiT native FP4/NVFP4: measured state and bottlenecks

Status: recorded 2026-11-13 on the native NVFP4 work branch.
Scope: `csrc/kernels/fp4/` and `src/cache_dit/quantization/fp4/`.
Re-run the benchmark and profiler before reusing these numbers; they are a single-GPU snapshot.

## Environment and artifacts

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, sm120a, physical GPU7.
- PyTorch 2.15.0+cu132, TorchAO 0.18.0, MSLK installed from the environment clone.
- Python: `/workspace/dev/miniconda3/envs/cdit-fp4/bin/python`.
- Build: `CACHE_DIT_BUILD_CUDA=1 CACHE_DIT_CUDA_ARCH_LIST=120a bash tools/build_fast.sh install -e . --no-build-isolation --no-deps`.
- Operator benchmark: `bench/kernels/bench_fp4.py`.
- Raw results: `.tmp/native_nvfp4/bench/bench_fp4_flux_scaled_mm.json`.
- E2E images/logs: `.tmp/native_nvfp4/e2e/`, `.tmp/native_nvfp4/logs/`.
- nsys: `.tmp/native_nvfp4/profiles/nsys_e2e_native_custom.nsys-rep`,
  `nsys_e2e_torchao.nsys-rep`.
- NCU: `.tmp/native_nvfp4/profiles/ncu_fp4_gemm.ncu-rep`,
  `ncu_torchao_gemm.ncu-rep`.

## Implementation contract

- Two independent public operators: `fp4_quantize` and `fp4_gemm`.
- `fp4_quantize` returns packed E2M1 `float4_e2m1fn_x2`, canonical 128x4 UE4M3
  block scales, and optional per-row FP32 global scales.
- Two-level recipe: activation per-row, weight per-output-channel, then per-16
  block scale; the epilogue folds the global scales and optional BF16 bias.
- Constraints: BF16, `K % 64 == 0`, `N % 8 == 0`; M tails supported.
- Native kernel guards: host rejects non-SM120; device body uses
  `__CUDA_ARCH__ >= 1200` with a trap fallback.
- `fp4_quantize` uses IEEE `__fdiv_rn` for global/block scale and reciprocal so
  `--use_fast_math` cannot change quantization bits.

## Correctness state

- Native quantize matches an independent PyTorch reference, including zero rows,
  zero groups, tiny underflow, 128-row scale padding and tail shapes.
- Native GEMM matches TorchAO `_scaled_mm` bit-for-bit when given the same
  single-level pack.
- Two-level GEMM against dequantized FP32 `F.linear`: relative L2 below 3e-3.
- Native CuTe GEMM matches the original LeetCUDA implementation bit-for-bit on
  the same pack.
- The default `torch._scaled_mm` path folds two global scales in BF16 after the
  GEMM. Its error against the dequantized reference is about 0.0041; the native
  kernel is more accurate because it folds both scales in FP32 before BF16.
- `tests/kernels/test_fp4_ops.py` and `tests/quantization/test_fp4_backend.py`
  cover pack/swizzle, tails, zero and underflow, bias, fake tensors, fullgraph
  compile and CUDA Graph.

## Operator performance

Six FLUX shapes, CUDA Event median of 50 iterations × 3 rounds. Microseconds.
`torchao_gemm` is `torch._scaled_mm` on the same single-level pack.

| shape | cd quantize | TorchAO quantize | cd `_scaled_mm` | TorchAO GEMM | native WS | cd compiled linear | TorchAO compiled linear |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4096x3072x3072 | 30.6 | 85.0 | 117.2 | 116.1 | 178.9 | 163.8 | 226.8 |
| 4096x12288x3072 | 30.6 | 83.2 | 385.2 | 384.2 | 645.5 | 592.2 | 621.4 |
| 4096x3072x12288 | 136.8 | 395.2 | 399.6 | 397.9 | 658.7 | 633.5 | 759.2 |
| 4608x3072x3072 | 32.7 | 85.8 | 127.7 | 127.2 | 197.5 | 180.0 | 263.2 |
| 4608x3072x15360 | 276.8 | 560.0 | 548.6 | 547.9 | 908.0 | 898.9 | 1094.3 |
| 4608x12288x3072 | 32.1 | 85.5 | 430.4 | 429.6 | 721.4 | 673.8 | 724.7 |

Geometric-mean ratios:

- cache-dit `_scaled_mm` / TorchAO GEMM: **1.004x**.
- native WS CuTe kernel / TorchAO GEMM: **1.626x** (native kernel is slower).
- cache-dit compiled linear / TorchAO compiled linear: **0.818x** (cache-dit
  is faster end-to-end per layer).

## FLUX e2e

FLUX.1-dev, 1024x1024, 28 steps, seed 42, warmup 1, repeat 3.

| arm | eager | compile |
|---|---:|---:|
| BF16 | 17.03 s | 14.90 s |
| TorchAO NVFP4 | 9.70 s | 5.56 s |
| cache-dit NVFP4, default `_scaled_mm` | 9.55 s | 5.69 s |
| cache-dit NVFP4, native CuTe GEMM | 9.24 s | 7.20 s |

Image quality of the default path:

- native eager vs BF16 eager: PSNR 30.21 dB, SSIM 0.873.
- native compile vs BF16 compile: PSNR 30.15 dB, SSIM 0.881.
- native vs TorchAO eager: PSNR 33.63 dB, SSIM 0.910.
- native vs TorchAO compile: PSNR 31.90 dB, SSIM 0.902.

The default path is within roughly 2% of TorchAO compile e2e and clearly better
than the native CuTe GEMM. The native GEMM is faster in eager because it avoids
the two separate global-scale multiply passes, but loses badly under
`torch.compile`.

## nsys/NCU: where time goes

nsys, one warmup + one measured forward, 27664 = 494 × 28 × 2 linears per arm:

| kernel group | cache-dit native CuTe GEMM | TorchAO |
|---|---:|---:|
| NVFP4 GEMM | 7.335 s | 4.321 s |
| activation quantize | 1.167 s (`row_amax` + `quantize`) | 0.508 s (`quantize_nvfp4`) + fused abs-max reductions |

The GEMM delta is about 3.01 s over two forwards, or 1.51 s per forward. That
accounts for nearly all of the 1.57 s e2e gap (`7.20 s` vs `5.63 s`).

Per-shape nsys aggregation (average per launch):

| layer class | calls | native | TorchAO |
|---|---:|---:|---:|
| N=3072, M=4608 | 8512 | 365.5 us | 215.8 us |
| N=12288, M=4608 | 2128 | 713.1 us | 415.4 us |
| N=3072, M=4096 | 5320 | 262.8 us | 155.6 us |
| N=12288, M=4096 | 1064 | 630.8 us | 366.5 us |
| N=15360, small M | 2128 | 101.1 us | 56.6 us |

NCU on `4096x3072x3072`:

| metric | native WS | TorchAO CUTLASS |
|---|---:|---:|
| duration | 128.3 us | 103.6 us |
| SM throughput | 61.1% | 77.8% |
| Tensor pipe active | 64.8% | 80.4% |
| occupancy | 18.2% | 20.8% |
| registers | 168 | 168 |
| shared memory | 38.9 KB | 89.1 KB |
| long-scoreboard stall | 1.58 | 0.99 |

Interpretation: the native kernel is not DRAM-bound (DRAM about 7.5%). It has
too little MMA work per TMA stage: `BK=64` with one K64 MMA step, versus
TorchAO's `128x128x256` CUTLASS tile. It therefore issues more TMA/barrier
iterations per K, has lower Tensor-pipe utilization and a larger load-latency
stall. `--use_fast_math` was not the cause; the quantize path uses IEEE division.

## Compile path findings

- `fp4_quantize_torch` was implemented and made bit-identical to the custom
  quantizer. Inductor lowered it to four kernels; compiled quantize was about
  145-160 us versus 31 us for the custom kernel.
- Replacing the compile path with it raised FLUX compile e2e from 7.21 s to
  7.99 s. It was reverted.
- The actual compile problem was the native CuTe GEMM, not the opaque quantize
  boundary. Defaulting to `torch._scaled_mm` reduced compile e2e to 5.69 s.
- TorchAO does **not** use an MSLK NVFP4 GEMM. Its NVFP4 linear uses
  `torch._scaled_mm` (cuBLAS/CUTLASS). MSLK is used only for Triton activation
  quantization (`mslk.quantize.triton.fp4_quantize.triton_quantize_nvfp4`).

## Effective and rejected changes

Retained:

- `torch._scaled_mm` as the default `fp4_gemm` path; native CuTe kernel remains
  available with `use_native=True`.
- N%16 fallback to native when `_scaled_mm` cannot represent the tail.
- IEEE division in the native quantizer for bit stability.
- FP8 `use_native=False` default for per-row/per-tensor; per-block stays native.

Rejected:

- Pure-Torch compile-visible FP4 quantizer: bit-exact but 5x slower compiled.
- Claiming an MSLK FP4 GEMM path: no such operator is used by TorchAO's NVFP4
  linear.
- Native CuTe GEMM as the default: 1.63x slower than `_scaled_mm` on production
  shapes and 1.5 s slower per FLUX forward under compile.

## Reproduction

```bash
cd cache-dit

CUDA_VISIBLE_DEVICES=7 /workspace/dev/miniconda3/envs/cdit-fp4/bin/python \
  bench/kernels/bench_fp4.py --warmup 10 --iters 50 --rounds 3 \
  --output .tmp/native_nvfp4/bench/bench_fp4_flux_scaled_mm.json

CUDA_VISIBLE_DEVICES=7 HF_HUB_OFFLINE=1 \
  /workspace/dev/miniconda3/envs/cdit-fp4/bin/python -m cache_dit.generate flux \
  --model-path /workspace/dev/vipdev/hf_models/FLUX.1-dev \
  --height 1024 --width 1024 --steps 28 --seed 42 --warmup 1 --repeat 3 \
  --nvfp4 --quantize-backend cache_dit --compile
```

## Current compile gap after the `_scaled_mm` switch

The FP4 code is intentionally frozen at the current mixed state:

- `fp4_gemm(..., use_native=False)` is the default and calls `torch._scaled_mm`.
- `fp4_quantize` remains the native CUDA quantizer in both eager and compiled
  paths. Do not replace it with `fp4_quantize_torch`.
- `fp4_gemm(..., use_native=True)` keeps the SM120 CuTe kernel for development.

Latest FLUX compile e2e was `5.69 s` vs TorchAO `5.56 s` (about 2.3% slower).
A later nsys capture on both cached graphs reported `5.77 s` vs `5.63 s`
(`0.14 s` difference), with:

- FP4 GEMM: `4.311 s` vs `4.322 s`.
- FlashAttention: `4.104 s` vs `4.123 s`.
- Activation quantize: `1.009 s` (native row-amax + quantize) vs `0.510 s`
  MSLK quantize plus about `1.111 s` of abs-max reductions.
- Total GPU kernel time: `13.70 s` vs `15.53 s` for two forwards.

The GEMM and quantization kernels are therefore not the remaining cause. GPU
work is lower on the cache-dit side, while wall time is slightly higher. The
remaining `0.13-0.14 s` sits outside reported kernel durations: compiled-graph
host dispatch/launch gaps, graph scheduling, or run-to-run variance. It has not
been separated from noise by an interleaved repeat yet. Do not claim a specific
host-gap number until that measurement is run.

## Future FP4 GEMM kernel tuning

The native CuTe kernel remains useful as a fallback and as a place to beat
cuBLAS, but it must be gated by kernel-only measurements. Baseline:
native WS is `1.626x` TorchAO `_scaled_mm` geometric mean, while the default
external path is `1.004x`. A tuning change is only useful if it moves the native
kernel toward or past parity without changing the two-level scale contract.

Priority order, based on NCU rather than tile folklore:

1. **More MMA work per TMA stage.** NCU shows native Tensor-pipe active 64.8%,
   SM 61.1%, long-scoreboard 1.58, versus TorchAO Tensor 80.4%, SM 77.8%,
   long-scoreboard 0.99. The native loop consumes `BK=64` with one K64 MMA step
   per stage; TorchAO uses a `128x128x256` CUTLASS tile. First experiment:
   `BK=128`/`256` or four K64 atom steps per TMA stage. Do not merely add
   pipeline stages; with one MMA per stage that only increases smem and does not
   fix compute-per-stage.
2. **Pipeline depth after BK changes.** Re-evaluate 6/8 stages and the WS/non-WS
   split only after the bigger K tile is numerically stable. Watch shared-memory
   budget, output overlay, registers (currently 168) and occupancy (18.2%).
3. **Shape dispatch.** Revisit `128x256` for `N=3072` and small-M/tail cases only
   after the mainloop change. Current e2e uses mostly `N=3072` and `N=12288`;
   do not tune to one shape and claim a matrix-wide win.
4. **Epilogue fusion.** The external path must fold two global scales in BF16
   today; a native epilogue can multiply both scales and add bias in FP32 before
   the single BF16 conversion. If the native mainloop reaches parity, this is its
   real advantage. Do not trade it for a BF16 intermediate just to save a
   register.
5. **Persistent/stream-K only after mainloop parity.** Tail/small-M occupancy is
   not the primary gap at M>=4096. Avoid adding persistent scheduling before the
   per-CTA Tensor utilization problem is solved.
6. **Acceptance gates.** Kernel-only median over the six FLUX shapes; require
   at least 5% improvement for a retained change, no register spill, and NCU
   Tensor utilization/SM improvement in the expected direction. Always re-run
   correctness (pack, scale, tails, bias, zero rows) before performance.

Do not repeat these without new evidence: raising stages alone, changing only
tile-N, or tuning based on the `4096x3072x3072` shape while ignoring `K=12288`
and `N=15360`.

## Future quantize fusion and deep optimization

The native quantize path is already faster than TorchAO's MSLK quantize, so
fusion is an optional next step, not the explanation of the current compile gap.

1. **Single-pass row/group quantization kernel.** Today eager launches
   `row_amax_fp4_kernel` then `quantize_fp4_kernel`; over two forwards they cost
   about `0.426 s + 0.583 s`. Fuse them: one CTA per row computes all 16-element
   group maxima, derives the row maximum from them, then does the second read and
   writes packed qdata, canonical scales and the global scale. This removes one
   launch per linear and one global row-amax buffer. Acceptance: bit-identical
   qdata/scales and a kernel-only win over the two-kernel chain on all six
   shapes, not only the short-K case.
2. **Fuse quantization with the preceding pointwise producer.** The high-value
   seams in FLUX are RMSNorm/GELU/SiLU/modulation feeding a quantized linear.
   A model-level `producer + fp4_quantize` kernel can avoid writing BF16
   activations and reading them back, but it must be a separate explicit op with
   a bit-exact BF16 fallback. Do not hide it inside `fp4_quantize`.
3. **Fuse quantization into the GEMM prologue only with A-tile reuse.** A naive
   per-CTA prologue would re-quantize the same A tile for every N tile. Only
   consider this with a persistent/2D grid or a producer-consumer split that
   proves each A tile is quantized once; otherwise keep the standalone
   quantizer.
4. **Compile-visible implementation must be a fused CUDA kernel.** The rejected
   pure-Torch path lowered to four kernels and cost 145-160 us versus 31 us for
   the custom quantizer. A future compile path may use a single fused CUDA
   kernel exposed as a custom op if host gaps are proven to dominate; a
   Triton implementation is outside this repository's supported backend policy.
5. **Measure before fusing.** The nsys gap analysis does not currently justify
   more quantize work: native quantize is cheaper than TorchAO's. First establish
   whether the 0.14 s is host gap or noise with one interleaved repeat; only then
   decide whether to spend effort on a fused producer or on graph/launch
   plumbing.
6. **Do not change the recipe.** Per-row activation and per-output-channel weight
   global scales plus per-16 UE4M3 block scales are the contract. Switching to
   per-tensor scaling would remove a pointwise pass but invalidates the accuracy
   comparison; make it an explicit opt-in experiment, never a silent default.
