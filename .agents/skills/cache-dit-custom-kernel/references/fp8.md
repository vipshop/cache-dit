# Cache-DiT native FP8: performance findings and deferred work

Status: recorded 2026-11-12 after commits `0dfb17e6`, `b6a15127`, `ceb786a5`.
Scope: native FP8 in `csrc/kernels/fp8/` and `src/cache_dit/quantization/fp8/`.
Do not apply these optimizations without re-running the current benchmark and NCU; the numbers below are a 2026-11-12 snapshot on one GPU.

## Environment and artifacts

- GPU: NVIDIA RTX PRO 5000 72GB Blackwell, sm120a.
- PyTorch 2.15.0+cu132, TorchAO 0.18.0.
- Operator benchmark: `bench/kernels/bench_fp8.py`.
- Raw results: `.tmp/fp8_backend/bench_fp8_ops_ws.json`, `bench_fp8_square_ws.json`,
  `bench_fp8_multi_shape_suite.json`.
- nsys: `.tmp/fp8_backend/nsys_kernels.nsys-rep`.
- NCU: `.tmp/fp8_backend/ncu_row_nonws.ncu-rep`, `ncu_block_ws.ncu-rep`.
- E2E: `.tmp/fp8_backend/e2e/` and `.tmp/fp8_backend/report.md`.

## Measured state

kernel-only, `4096 x 3072 x 3072`, BF16 output, 50 launches:

| path | kernel | time |
|---|---|---:|
| activation per-row quantize | `quantize_per_row_kernel` | 16.18 us |
| activation per-block quantize | `quantize_per_block_activation_kernel` | 15.27 us |
| per-row non-WS | `fp8_gemm_scale_kernel` 128x256x128/s2 | 195.46 us |
| per-row WS | `fp8_gemm_scale_ws_kernel` | 196.84 us |
| TorchAO per-row | CUTLASS `_scaled_mm` | 181.09 us |
| per-block non-WS | `fp8_gemm_block_kernel` 128x128x128/s2 | 266.96 us |
| per-block WS | `fp8_gemm_block_ws_kernel` | 241.69 us |
| TorchAO per-block | Triton `blockwise_fp8_gemm_kernel` | 224.58 us |

Six FLUX layer shapes, CUDA Event median ratios (cache-dit / TorchAO):

| mode | quantize | GEMM non-WS | GEMM WS | eager chain |
|---|---:|---:|---:|---:|
| per_row | 0.128x | 1.054x | 1.064x | 0.88x |
| per_tensor | 0.147x | 1.059x | 1.055x | 0.90x |
| per_block | 0.079x | 0.999x | 0.921x | 0.69x |

E2E FLUX.1-dev, compile, 1024^2, 28 steps:

| backend | per_row | per_tensor | per_block |
|---|---:|---:|---:|
| TorchAO | 7.03 s | 7.38 s | 8.37 s |
| cache-dit | 8.01 s | 8.01 s | 9.47 s |

The custom quantize kernel is 6.8x-12.7x faster in eager mode, but the compiled E2E is 0.6-1.1 s slower. The gap is structural: TorchAO's pure-Python quantization gets fused by Inductor, while the native path exposes two opaque custom ops with an intermediate `qdata`/`scale` round-trip, plus a 5%-6% slower per-row GEMM.

### Pure-Torch quantize under compile

Implemented in `src/cache_dit/kernels/fp8.py::fp8_quantize_torch` and selected from
`CacheDitFP8Linear.forward` when `torch.compiler.is_compiling()` is true. It reproduces
the custom kernel's qdata and scale bit-for-bit for all three modes and the weight-block
tail case, but keeps the ops visible to Inductor.

Compiled E2E after switching to the pure-Torch quantizer (FLUX 1024^2, 28 steps, r3):

| mode | before | after | TorchAO |
|---|---:|---:|---:|
| per_row | 8.01 s | 7.42 s | 7.03 s |
| per_tensor | 8.01 s | 7.53 s | 7.38 s |
| per_block | 9.47 s | 8.82 s | 8.37 s |

- Memory stays 22.87 GiB for every arm.
- qdata/scale are bit-identical to the custom kernel. Do **not** use E2E PSNR to judge this
  change: image PSNR on this workload has a wide run-to-run spread (observed 27-35 dB on
  fixed seeds), larger than the A/B difference. Time and memory are the only conclusions
  drawn here; the bitwise quantizer test is the correctness gate.
- Remaining gap is per-row/per-block GEMM, not quantize.
- Profiler asserts `quantize_per_row_kernel` is absent and `fp8_gemm_scale_kernel` remains
  in the compiled graph (`test_compiled_linear_uses_fusible_torch_quantize`), and a separate
  test asserts the eager path still invokes `quantize_per_row_kernel`.

## Multi-shape GEMM matrix (2026-11-12)

19 shapes, `bench/kernels/bench_fp8.py --suite sweep`, warmup 5 / iters 20, CUDA Event
medians including host launch overhead. `ratio = cache-dit / TorchAO`; per-row uses default
non-WS, per-block default WS. The matrix includes M/N tails and a multi-K-block shape; all
K are 128-aligned because the per-block kernel requires it.

| shape | per-row ratio | per-block ratio |
|---|---:|---:|
| 128x3072x3072 | 1.876 | 0.997 |
| 512x3072x3072 | 1.517 | 0.700 |
| 1024x3072x3072 | 0.941 | 0.801 |
| 2048x3072x3072 | 1.020 | 0.873 |
| 4096x3072x3072 | 1.094 | 0.905 |
| 4608x3072x3072 | 1.001 | 0.912 |
| 8192x3072x3072 | 1.049 | 0.907 |
| 4096x12288x3072 | 1.065 | 0.908 |
| 4096x3072x12288 | 1.045 | 0.857 |
| 4608x12288x3072 | 1.076 | 0.915 |
| 4608x3072x15360 | 0.964 | 0.992 |
| 4096x15360x3072 | 1.081 | 0.913 |
| 4096x3072x15360 | 1.080 | 0.986 |
| 4032x3072x3072 | 1.051 | 0.912 |
| 4096x3200x3072 | 1.080 | 0.918 |
| 4096x3072x3328 | 1.102 | 0.875 |
| 1024x4096x4096 | 1.342 | 0.888 |
| 4096x4096x4096 | 1.066 | 0.912 |
| 8192x8192x8192 | 1.044 | 0.928 |

Summary:

- per-row: wins 2/19, median ratio 1.066, range 0.941-1.876. It is broadly behind
  `_scaled_mm` on aligned production shapes. The worst regression is at small M
  (128/512: 1.88x/1.52x), where the 128x256 tile yields only 12/48 CTAs; this is a tile
  granularity problem, not evidence that host descriptor setup dominates. Treat per-row
  GEMM as the primary optimization target (tile/CTA selection first).
- per-block: wins 19/19, median ratio 0.912, range 0.700-0.997. WS is not a
  universally-losing path; it wins every tested shape against TorchAO's Triton kernel.
- per-row WS is slower than non-WS in 15/19 shapes, median WS/non-WS = 1.010; keep the
  mode-specific default.
- per-block non-WS wins 14/19 but is 7%-17% slower than WS on every tested shape; keep WS
  default.

## Confirmed bottlenecks

### 1. Per-row/epilogue register pressure

- NCU per-row non-WS: SM 72.15%, memory 40.67%, occupancy 16.32%, 255 regs/thread, 100.35 KB smem, 1 CTA/SM, no local spill.
- Top stalls: `math_pipe_throttle` 3.16, `wait` 2.38, `long_scoreboard` 0.229.
- The kernel is compute-bound but register/smem-limited to one CTA.
- WS per-row is not the fix: 384 threads lower the ptxas static register ceiling to 168, while the per-row epilogue needs ~194; `cuobjdump -res-usage` reports `STACK:144` for WS vs 24 for non-WS. Hence `fp8_gemm(..., use_ws=None)` intentionally uses non-WS for per-row/per-tensor.
- Per-block WS fits: 168 regs, 67.58 KB smem, no spill, SM 57.33%, occupancy 18.56%; top stalls `wait` 1.18 and `math_pipe_throttle` 1.12. WS is ~8% faster than non-WS there.

### 2. Compile graph boundary

`CacheDitFP8Linear.forward` currently calls `fp8_quantize` then `fp8_gemm`, both `torch.library` custom ops. Inductor treats each as opaque:

- activation quantization cannot fuse with surrounding pointwise work;
- the qdata and scale round-trip through global memory remains;
- the GEMM prologue cannot consume a BF16 tensor directly.

Do not conclude from the fast eager quantize kernel that E2E should be faster.

### 3. Per-row GEMM is behind `_scaled_mm`

Production shapes and kernel-only runs show a consistent 5%-6% deficit. Possible causes to test with NCU, not yet proven:

- accumulator/tile shape (128x256x128/s2) leaves only one CTA/SM at 255 registers;
- TMA and descriptor setup overhead;
- epilogue (scale lookup, BF16 convert, STSM, TMA store) is a larger fraction at 3072-wide N;
- missing cluster/L2 swizzle or persistent scheduling compared with CUTLASS.

### 4. Per-block is not a current priority

Per-block WS wins all 19 shapes in the operator matrix (median 0.912x vs TorchAO), but the single-shape kernel-only number remains mixed: cache-dit 241.69 us vs TorchAO Triton 224.58 us at `4096x3072x3072`. Treat per-block as "do not optimize now": the compiled per-layer total is roughly at parity (quantize 30.4 vs 28.7 us; GEMM+bias 248.7 vs 244.4 us on `4096x3072`), and the remaining E2E delta needs graph-level evidence before changing kernels.

### 5. Host-side fixed costs

Every launch rebuilds three TMA descriptors on the host. Single-layer compiled timing shows `host + launch` about 36.9 us for per-row vs 27.9 us for TorchAO at `4096x3072x3072`; the exact split between descriptor construction, pybind dispatch and graph entry has not been isolated. CUDA Graph removes it from replay; eager/custom-op E2E does not.

## Convergence target

The task is converged only when, for every supported FP8 mode, cache-dit's full
**quantize + GEMM** chain is equal to or faster than TorchAO under `torch.compile`, with
equivalent qdata/scale semantics. Optimizing an isolated kernel is not sufficient; report
the compiled per-layer and E2E numbers together.

Custom quantize vs the fused `fp8_quantize_torch` compiled kernel (device time,
`4096x3072x3072`):

| mode | custom quantize kernel | Torch compile fused | conclusion |
|---|---:|---:|---|
| per_row | 16.18 us | 10.18 us | custom is slower |
| per_tensor | ~20.4 us (3 kernels) | 20.83 us (3 fused kernels) | device parity; custom loses on host/launch in the full chain |
| per_block | 15.27 us | 30.35 us | custom is faster |

So the custom quantize kernel is a real optimization target for per_row/per_tensor
(especially full-chain wall time), but per_block must not be replaced blindly: its custom
kernel is still ahead of the fused pure-Torch path.

## Current optimization priority (2026-11-12, converged)

This is the agreed order after the 19-shape and compiled-linear decomposition. Do not
reorder without new measurements.

1. **Per-row GEMM tile/occupancy (primary).** Per-row is behind `_scaled_mm` in 17/19 shapes, median 1.066x. At `4096x3072x3072` the device GEMM is 198.3 vs 183.5 us; NCU shows SM 72%, occupancy 16.3%, 255 regs/thread, 1 CTA/SM. Start with tile/CTA selection and register pressure, not quantization. Target: match `_scaled_mm` without regressing production shapes.
2. **Custom quantize kernel for per_row/per_tensor.** Close the gap to the fused Torch path: per-row kernel 16.18 vs 10.18 us; per-tensor is device-parity but loses through extra launches/host work. Optimize reduction/launch structure and host overhead; keep the eager custom path fast. Do not change per-block without evidence, where custom is already 15.27 vs 30.35 us.
3. **Per-launch TMA descriptor and custom-op overhead.** Compiled per-layer wall time is 36.9 us above device kernels (vs 27.9 us for TorchAO) for per-row. Attack descriptor construction/reuse, pybind/dispatcher overhead, and graph entry; measure host/device separately before changing ABI. Do not attribute the small-M regression to host overhead: at M=128/512 the 128x256 tile leaves too few CTAs, which belongs to priority 1.
4. **Per-block: no optimization work now.** Quantize is at parity after the pure-Torch compile path; GEMM is already faster than TorchAO on 19/19 shapes. Only revisit with an E2E trace showing a concrete per-block hotspot.

## Deferred optimization directions

Priority is evidence-driven and deliberately not implemented yet.

1. **Fuse activation quantization into the GEMM prologue.** Let the kernel consume BF16 A directly, compute row/block scales while TMA stages A, and avoid qdata/scale global round-trips. This is a longer-term follow-up, not ahead of per-row GEMM tuning.
2. **Pure-Torch quantize under `torch.compile` (implemented 2026-11-12).** `fp8_quantize_torch` is selected by `torch.compiler.is_compiling()`; eager keeps the custom kernel. E2E measurements for all three modes are in the table above. Remaining work is to decide whether graph-break/resume paths should force the custom kernel (currently they can execute the unfused pure-Torch version).
3. **Per-row WS epilogue split by N.** Process the epilogue in two 128-column groups, reducing peak live registers below the 168 ceiling and eliminating the 144 B/thread spill. Verify with `cuobjdump -res-usage` (`STACK:0`) before claiming a gain.
4. **Per-row tile/occupancy sweep details.** Use NCU to test 128x128/s3, 256x128/s2, register caps, and stage counts; this supports current priority 1.
5. **Per-block rescale reduction.** Deferred; do not start until per-block shows a measured regression.
6. **TMA descriptor reuse.** Cache weight descriptors per module and use CUDA Graph or descriptor update for dynamic activations; supports current priority 2.
7. **Compile path with one fused custom op.** Deferred follow-up to priority 2: one `quantized_linear_bf16` graph node instead of two, benchmarked separately.

## Reproduction checklist

```bash
cd cache-dit
CACHE_DIT_BUILD_CUDA=1 CACHE_DIT_CUDA_ARCH_LIST=120a \
  bash tools/build_fast.sh install -e . --no-build-isolation --no-deps

CUDA_VISIBLE_DEVICES=4 /workspace/dev/miniconda3/envs/cdit/bin/python \
  bench/kernels/bench_fp8.py --suite sweep --modes per_row per_block --skip-compiled \
  --warmup 5 --iters 20 --output .tmp/fp8_backend/bench_fp8_multi_shape.json

CUDA_VISIBLE_DEVICES=4 nsys profile --trace=cuda --capture-range=cudaProfilerApi \
  --capture-range-end=stop -o .tmp/fp8_backend/nsys_kernels \
  /workspace/dev/miniconda3/envs/cdit/bin/python .tmp/fp8_backend/profile_kernels.py
```

When changing the WS default, rerun `tests/kernels/test_fp8_ops.py`, especially the `use_ws` parity, K-phase, CUDA Graph, and profiler kernel-name assertions.

## Per-row follow-up on 2026-11-12: retained gains and open E2E gap

Branch `opt/fp8-per-row-compile`, environment `cdit-fp8-per-row-opt`, RTX PRO 5000 sm120a. Artifacts are under `.tmp/fp8_per_row_opt/`. This is an intermediate improvement: the requested stable 5% E2E win over TorchAO + compile was not reached, and no PR is claimed or created.

### Retained implementation changes

- Per-row quantize uses eight warps per row. For short K it reuses the BF16 input through dynamic shared memory rather than reading global memory twice; long K keeps the two-pass global path. Reciprocal constants and FP8 pack conversion already fold in SASS, so no unsafe reciprocal rewrite was introduced. The standalone `4096x3072` kernel improved from 15.856 to 9.536 us; the Inductor fused Torch quantize comparison was 14.336 us.
- Per-row GEMM uses M4N2 warps and shape dispatch: 64x64 for tiny M, 128x128 for small M, 128x192 for medium M with N < 4096, and 128x256 otherwise. All tiles use two stages. At `4096x3072x3072`, the BN192 non-WS kernel is 181.760 us and WS is 181.712 us, versus 184.016 us for TorchAO `_scaled_mm`. NCU shows tensor-pipe utilization 76.8%, 84 KB shared memory, and one CTA/SM.
- `fp8_quantize_gemm` is a single host/custom-op entry for per-row linear. It calls the existing quantize and GEMM functions in C++; it is not a CUDA kernel fusion. Eager per-row `QW8A8Linear` uses this entry. All compiled modes use `fp8_quantize_torch` plus the native GEMM so Inductor can fuse quantization. Fake, fullgraph, public export, WS/non-WS bitwise, CUDA Graph, and dispatch tests cover the native operators.

Six production-shape compiled-linear CUDA Event medians (native / TorchAO, microseconds):

| shape | native | TorchAO | native / TorchAO |
|---|---:|---:|---:|
| 4096x3072x3072 | 221.68 | 216.21 | 1.0253 |
| 4096x12288x3072 | 734.34 | 737.42 | 0.9958 |
| 4096x3072x12288 | 797.20 | 864.11 | 0.9226 |
| 4608x3072x3072 | 241.31 | 258.64 | 0.9330 |
| 4608x3072x15360 | 1079.14 | 1202.30 | 0.8976 |
| 4608x12288x3072 | 825.57 | 826.62 | 0.9987 |

The aggregate latency ratio is 0.9498 (about 5.0% less total latency), but the geometric-mean ratio is 0.9610 and the first shape is slower. Do not describe this as a stable 5% per-shape win.

### Rejected experiments

- CuTe `copy`/`copy_if` direct R-to-G epilogue: numerically correct but poorly coalesced compared with STSM+TMA. Typical GEMM regressed from about 182 to 268 us and compiled linear regressed 26%-36%; fully reverted.
- BF16 producer prologue fusion: each N-tile repeatedly read and quantized the same activation. Compiled linear regressed 1.9x-3.9x; fully reverted. qdata materialization is costly, but repeated quantization across N-tiles is worse.
- Compile-time pure-Torch quantization plus optimized native GEMM did not solve the E2E gap either; aggregate compiled-linear improvement was about 4.7%.
- `CACHE_DIT_EPILOGUE_PROLOGUE_FUSION=1` improved TorchAO E2E but did not materially improve the native custom-op path.

### FLUX and validation status

FLUX.1-dev, 1024x1024, 28 steps, seed 42, warmup 1, repeat 3, 494/494 quantized layers on both FP8 arms:

| arm | default fusion setting | epilogue/prologue fusion enabled |
|---|---:|---:|
| TorchAO + compile | 7.43 s | 7.03 s |
| native cache-dit + compile | 7.62 s | 7.61 s |

The default-setting native arm is about 2.6% slower than TorchAO, so the stable 5% E2E gate failed. A formal-step nsys decomposition found native GEMM slightly faster in total (142.62 versus 143.68 ms), but native quantize plus surrounding Inductor kernels remained more expensive because custom-op boundaries prevent equivalent fusion.

Final native-vs-TorchAO image metrics: PSNR 40.081 dB, SSIM 0.98115. Native-vs-BF16: PSNR 35.189 dB, SSIM 0.95087. Images were visually checked and showed no structural corruption.

Final checks before the intermediate commit: 187 FP8/quantization/compile-config tests passed, `pre-commit run --files ...` passed after formatting, and `git diff --check` passed. The E2E performance gate remains open and the measurements must not be described as a stable win over TorchAO.

## 2026-11-13 hybrid follow-up and pure-Torch quantizer analysis

All three E2E arms below ran sequentially on the same idle GPU5 with `CACHE_DIT_EPILOGUE_PROLOGUE_FUSION=1`, 1024x1024, 28 steps, seed 42, warmup 1 and repeat 3. The hybrid arm used `fp8_quantize_torch` only under `torch.compile`, followed by the optimized native per-row GEMM; eager still used the native fused host op.

| arm | inference time | relative result |
|---|---:|---|
| native fused custom op + compile | 7.54 s | baseline |
| pure-Torch compile quantize + native GEMM | 7.04 s | 6.6% faster than native custom-op path |
| TorchAO + compile | 6.93 s | hybrid remains 1.6% slower |

The same hybrid arm won 5/6 production compiled-linear shapes. Its aggregate and geometric-mean latency ratios to TorchAO were 0.9548 and 0.9584, respectively. This confirms that graph-visible quantization is materially better than the opaque custom-op path after enabling fusion, but it still does not establish a stable 5% E2E win over TorchAO.

Pure-Torch per-row quantizer microanalysis at `4096x3072`:

- TorchAO inference `_choose_scale_float8` computes `abs/amax` in the input dtype, but also computes `max_abs / 448` in that dtype before converting the scale to FP32. Cache-dit must not copy the latter rounding if it wants to preserve its current scale/qdata contract.
- Replacing `x.float().abs().amax()` with `x.abs().amax().float()` is bit-equivalent for BF16 abs/max, including an exhaustive all-65536-bit-pattern test. However, Inductor's `codegen_upcast_to_fp32` still generates an FP32 reduction. Measured Event time was 31.07 us versus 30.88-31.10 us for the current form; disabling upcast gave 30.91 us. There is no useful performance gain.
- Removing `clamp(-448, 448)` is bit-equivalent under the tested PyTorch 2.15 saturating FP8 conversion. Standalone quantize improved from about 31.1 to 30.2 us, but aggregate compiled-linear improvement was only 0.05% (one shape improved 1.49%, others were within noise). Treat this as optional cleanup, not a meaningful E2E optimization.
- Replacing per-element division with a row reciprocal and multiplication gave 1,476 mismatches across the 65,536 BF16 bit patterns, in both eager and compiled modes, with no speedup (about 30.3 us). It is rejected.

The hybrid path was subsequently promoted into `QW8A8Linear`: all compiled modes use `fp8_quantize_torch`, while only eager per-row uses the fused host op. Epilogue/prologue fusion now defaults to enabled and can still be disabled with `CACHE_DIT_EPILOGUE_PROLOGUE_FUSION=0`; `codegen_upcast_to_fp32` remains enabled. The no-clamp experiment stayed in `.tmp/fp8_per_row_opt/`; production `fp8_quantize_torch` was not changed by that experiment. Raw logs and generated Inductor code are retained in the same directory.

## 2026-11-13 final E2E record and convergence decision

The optimization task is stopped at the current hybrid implementation. TorchAO + compile was faster in the final model matrix, so further native GEMM/Inductor tuning is deferred unless a new requirement appears. This section records the remaining gap without claiming convergence on performance.

### FLUX.1 steady-state NSYS decomposition

Both hybrid and TorchAO had exactly 35,136 kernel launches in the captured 28-step inference. Total GPU time differed by 159.6 ms: native FP8 GEMM contributed +114.8 ms and Inductor quantization regions +45.0 ms; attention was 3.2 ms faster and other Triton kernels were 3.1 ms faster. Timeline gaps were 22.2 ms for hybrid and 25.6 ms for TorchAO, so host launch gaps were not the main wall-time cause. The largest quantization-region discrepancy was a `_where_9` kernel (96.83 vs 40.41 us in NSYS), while both backends contained the cache-dit zero-scale `where` protection. `_patch_torchao_zero_row_scale` is called by `quantize_ao`, so the 6.93 s TorchAO baseline also included this protection; it was not an unprotected upstream-TorchAO comparison.

`ones_like` was lowered to a scalar `tl.full([1,1], 1.0)`, not a real M-element allocation. A four-way `[M]/[M,1]` x `ones_like/scalar` test was bit-exact, but scalar/keepdim did not improve the full linear. TorchAO per-row scale was `[M,1]` and cache-dit's was `[M]`; matching the rank did not provide a reliable gain. These are not retained as performance optimizations.

### Three-model performance matrix

All runs used GPU5, 1024x1024, seed 42, warmup 1, repeat 3, fusion enabled, and no DBCache. Qwen used its built-in 8-step schedule; both Klein models used 4 steps. Native arms used pure-Torch compile quantization plus native FP8 GEMM. The safe-off runs explicitly used `safe_scale=False`; safe-on used `True`.

| model | BF16 + compile | native safe on | native safe off | TorchAO + compile | safe-on vs TorchAO |
|---|---:|---:|---:|---:|---:|
| Qwen-Image-2.1-Turbo | 3.16 s | 1.97 s | 1.97 s | 1.95 s | 1.0% slower |
| FLUX.2-klein-4B | 0.92 s | 0.71 s | 0.71 s | 0.60 s | 18.3% slower |
| FLUX.2-klein-9B | 1.71 s | 1.40 s | 1.41 s | 1.08 s | 29.6% slower |

### Three-model accuracy versus BF16

| model | native safe on PSNR/SSIM | native safe off PSNR/SSIM | TorchAO PSNR/SSIM |
|---|---|---|---|
| Qwen-Image-2.1-Turbo | 40.313 / 0.97683 | 40.073 / 0.97529 | 40.420 / 0.97701 |
| FLUX.2-klein-4B | 34.481 / 0.95311 | 35.240 / 0.96832 | 35.652 / 0.95607 |
| FLUX.2-klein-9B | 36.606 / 0.96505 | 36.669 / 0.96802 | 36.773 / 0.96840 |

Safe-on versus safe-off PSNR was 41.056, 37.294, and 38.465 dB, respectively. Separately compiled runs were not pixel-identical, so these numbers are recorded as compile run-to-run variation rather than a proof that the switch changes or preserves final images. Visual inspection found no black images, NaNs, or structural corruption. Images, logs, and metrics are in `.tmp/fp8_per_row_opt/e2e_models/`.

### Final safe-scale and documentation state

`fp8_quantize_torch(..., safe_scale=True)` is now the default. Disabling it is an explicit optimization for callers that guarantee nonzero amax in every row/tensor/block; all-zero units then produce NaN qdata. Native CUDA quantization remains safe for zero units. Tests cover default-on behavior, nonzero-input bit equality, the unsafe zero-unit contract, activation blocks, and weight blocks.

The main per-row quantizer and scale GEMM kernels now have detailed comments covering their mathematical contract, Vec8 layout, hierarchical reduction, shared-memory reuse, TMA/mbarrier pipeline, M4N2 tiling, guarded tails, FP32 epilogue, WS/non-WS synchronization, and BF16 output staging. Final validation after these cleanup changes was 191 tests passed, with pre-commit and `git diff --check` passing. Performance optimization is intentionally paused here.

## 2026-11-13 external GEMM default (`use_native=False`)

`fp8_gemm` now defaults to `use_native=False`:

- `per_row` and `per_tensor` use `torch._scaled_mm` (cuBLAS/CUTLASS). This is the
  same operator TorchAO uses for these modes, and it accepts the exact cache-dit
  scales: rowwise uses `a_scale[:, None]` and `b_scale[None, :]`.
- `per_block` keeps the cache-dit native kernel. `_scaled_mm` cannot express the
  `(1, 128) x (128, 128)` block scales used by this path.
- If `N % 16 != 0`, the default deterministically falls back to the native
  kernel because `_scaled_mm` requires both matmul dimensions to be multiples
  of 16.
- `use_native=True` forces the native CuTe kernel for every mode; `use_ws` then
  selects the native mainloop. `fp8_quantize_gemm` remains the fused native
  per-row eager op and is no longer on the default high-level path.

Six FLUX shapes, CUDA Event medians of 50 iterations x 3 rounds. Ratios are
cache-dit default GEMM / TorchAO GEMM:

| mode | geometric-mean ratio |
|---|---:|
| per_row | 1.001x |
| per_tensor | 0.990x |
| per_block (native) | 0.913x |

Compiled Linear remains much more variable because TorchAO's compile path
sometimes falls into much slower kernels on the large-N shapes; the raw
per-shape numbers are in `.tmp/native_nvfp4/bench/bench_fp8_scaled_mm.json`.
Do not use these numbers as a stable e2e win claim without rerunning the exact
model/compile cache.

Validation: `tests/kernels/test_fp8_ops.py` and
`tests/quantization/test_fp8_backend.py` pass with both the default external
path and explicit `use_native=True`; CUDA Graph, fake, WS/non-WS and kernel-name
tests were updated to select the native path explicitly.

### FLUX `float8_per_row` compile e2e

One paired run, 1024x1024, 28 steps, seed 42, `--compile`, `--repeat 3`.
`--warmup 1 --warmup-num-inference-steps 1` was used so the warmup only triggers
compilation instead of running another 28-step pass; the reported timing is
still the measured 28-step inference.

| arm | inference time |
|---|---:|
| cache-dit per-row, `fp8_quantize_torch` + `torch._scaled_mm` | 7.02 s |
| TorchAO per-row + compile | 6.99 s |

The difference is 0.03 s (0.4%), i.e. parity within one-run noise. Image metrics:

- cache-dit vs BF16 compile: PSNR 35.69 dB, SSIM 0.9609.
- TorchAO vs BF16 compile: PSNR 35.53 dB, SSIM 0.9482.
- cache-dit vs TorchAO: PSNR 39.44 dB, SSIM 0.9793.

Artifacts: `.tmp/native_nvfp4/e2e/fp8_per_row_native_compile.png`,
`.tmp/native_nvfp4/e2e/fp8_per_row_torchao_compile.png`,
`.tmp/native_nvfp4/logs/e2e_fp8_per_row_*.log`.
