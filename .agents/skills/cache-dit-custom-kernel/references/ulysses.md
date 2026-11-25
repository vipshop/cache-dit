# Ulysses FP8 通信：现状、契约与实测

Status: recorded 2026-11-12 after commits `c40cea09` and `b5ae66f1`.
Scope: `csrc/kernels/ulysses/`, `src/cache_dit/kernels/ulysses.py`,
`src/cache_dit/distributed/communicate/primitives.py`.
Machine: RTX PRO 5000 72GB, sm120a, PCIe, no NVLink; tests use GPU 0-3.
Do not quote these absolute numbers on NVLink or other GPUs without remeasurement.

## 1. What changed

The former `csrc/kernels/comm/` family and `fp8_comm_*` operators were renamed and
moved to first-class Ulysses semantics:

```text
csrc/kernels/ulysses/
  ulysses.h / ulysses.cpp              register_ulysses_ops
  fp8_common.cuh                       shared device helpers
  fp8_per_token_quant.cu
  fp8_per_token_dequant.cu
  fp8_qkv_permute_quant.cu
  fp8_qkv_permute_dequant.cu
  fp8_o_permute_quant.cu
  fp8_o_permute_dequant.cu

src/cache_dit/kernels/ulysses.py       public wrappers + torch.library
```

- C++ namespace: `cache_dit::ulysses`.
- Torch namespace: `cache_dit_cuda_ops::ulysses_fp8_*`.
- Python public names: `ulysses_fp8_*`.
- `setup.py` source helper: `_required_ulysses_sources()`.
- Single extension entry `csrc/kernels/ops.cpp` calls `register_ulysses_ops`.
- Old `fp8_comm_*` names, `kernels/comm.py`, and `csrc/kernels/comm/` are removed;
  there is no compatibility alias. Historical docs/examples using those names are stale.

Reason for the semantics: QKV/O permute layouts are specific to Ulysses' head-to-sequence
exchange. The per-token quantize/dequantize core is generic, but the family is not the Ring
transport and must not be advertised as such. See `references/ring.md` for the Ring plan.

## 2. Operator contracts

All builders are BF16 input on CUDA; the compiled extension is required.

### per-token quant/dequant

```python
ulysses_fp8_per_token_quant(x, packed_head_size=None) -> fp8_e4m3
ulysses_fp8_per_token_dequant(x, head_size=None) -> bf16
```

- Input is flattened to `[rows, D]`; row output is `[rows, packed_head_size]`.
- Default packed width is `D + 2`.
- BF16 scale is stored at byte offset `D`; `[D + 2, packed_head_size)` is zeroed.
- Dequant must receive the original `D` whenever padding is present.
- `packed_head_size` must be at least `D + 2`.

### QKV permute quant/dequant

```python
ulysses_fp8_qkv_permute_quant(x, packed_head_size=None) -> fp8_e4m3
ulysses_fp8_qkv_permute_dequant(quant_x, head_size=None) -> bf16
```

- Input `[B, S, P, N, D]`, last dimension stride 1.
- Output `[P, S, B, N, packed_head_size]`, default `D + 4`.
- FP32 scale is stored at byte offset `D`; `[D + 4, packed_head_size)` is zeroed.
- Dequant input `[S, B, N, packed_head_size]` -> `[B, S, N, D]`.
- `packed_head_size` must be at least `D + 4`.

### O permute quant/dequant

```python
ulysses_fp8_o_permute_quant(x, partitions, packed_head_size) -> fp8_e4m3
ulysses_fp8_o_permute_dequant(quant_x, head_size) -> bf16
```

- Input `[B, S_GLOBAL, H_LOCAL, D]`, last dimension stride 1.
- Output `[P, H_LOCAL, B, S_LOCAL, packed_head_size]`.
- BF16 scale at offset `D`; padding zeroed.
- Dequant input `[H_GLOBAL, B, S_LOCAL, packed_head_size]` ->
  `[B, S_LOCAL, H_GLOBAL, D]`.
- `S_GLOBAL % partitions == 0`; the O FP8 path also rejects rank-dependent head padding,
  so global heads must divide by world size. Disable `ulysses_float8` if either constraint
  fails.

## 3. Alignment rules and padding strategy

NCCL on this PCIe host drops to roughly 1/5-1/7 bandwidth when the per-rank all-to-all chunk
is not 16-byte aligned. The fix must be per-chunk, not a naive flat-buffer tail pad.

### BF16 paths

BF16 has no scale. If one row is 16-byte aligned, every chunk is aligned regardless of the
number of rows:

```python
alignment = 16 // element_size          # 8 elements for BF16
packed_D = ceil(D / alignment) * alignment
```

`_padded_last_dim(D, element_size)` is the shape-only helper. It has no `units` list and no
`any`; when `D` is already a multiple of 8 for BF16 it returns `D`, and
`_pad_last_dim`/`_unpad_last_dim` return the original tensor. Common model `D=64/128` is
therefore a strict no-op.

This is applied to BF16 standard QKV, standard O, UAA QKV, and UAA O. The padding is zero,
the sequence split sizes are unchanged, and the output is bitwise equal to the old unpadded
layout.

Do not generalize the shape-only rule to FP8: FP8 rows include the scale bytes, so
`D+2 -> next multiple of 16` overpads (for `D=128`, 130 -> 144, about 10.8% more traffic).

### FP8 paths

FP8 keeps a chunk-minimal search:

```python
packed = min_width                       # D+2 or D+4
while any((rows_per_chunk * packed) % 16 != 0 for rows_per_chunk in all_split_rows):
  packed += 1
```

- Standard paths: one uniform chunk size.
- UAA paths: use all gathered send and receive split sizes, so every rank selects the same
  `packed` value while split sizes remain row counts.
- Quant kernels write zero padding, and dequant reads only the real payload plus scale.
- The O path previously used `(P,H_L,B,S_L,D+2)`, which misaligned Wan-480p/CP4; the new
  fused O path chooses the minimal aligned width and removes the two full permutes around
  quantize/dequantize.

## 4. K FP8

`ulysses_float8_k=False` is the default.

- It requires `ulysses_float8=True`; otherwise configuration raises `ValueError`.
- When enabled, K uses the same QKV FP8 implementation as Q/V.
- LSE always remains BF16.
- CLI: `--ulysses-float8-k` / `--ufp8k`.
- Parallel config suffix: `UF8K_` (UAA: `F8K_`) so e2e output names do not collide.

## 5. Measured results

FP8 microbenchmark, CUDA Event, warmup and repeated runs:

| Case | Before | After | Result |
|---|---:|---:|---|
| Wan shape `seq=32760, heads=40`, CP4, whole Q/K/V/O layer | 9.5535 ms (0.86x of BF16) | 4.9953 ms (1.64x) | 1.91x |
| Same, O direction only | 5.8716 ms (0.37x) | 1.2610 ms (1.73x) | 4.66x |
| FLUX shape `seq=4096, heads=24`, CP2, whole layer | 0.6826 ms (1.54x) | 0.6620 ms (1.59x) | 1.03x |

BF16 auto padding, 4-rank:

| Shape | Legacy | Padded | Result |
|---|---:|---:|---|
| `D=66`, `S_LOCAL=8191`, `H=8` | 1.2303 ms | 0.2929 ms | 4.20x |
| `D=128`, `S_LOCAL=8191`, `H=8` | 0.4534 ms | 0.4513 ms | 1.00x, no-op |

FLUX.1-dev no cache/quantize/compile, seed 0, 1024^2, warmup 2 / repeat 2:

| Config | Time | PSNR vs single | SSIM |
|---|---:|---:|---:|
| CP2 BF16 | 10.11 s | 38.66881 | 0.96339 |
| CP2 FP8 QVO | 9.52 s | 38.54188 | 0.97597 |
| CP2 FP8 QKVO | 9.34 s | 35.74742 | 0.97453 |
| CP4 BF16 | 5.96 s | 39.51267 | 0.97099 |
| CP4 FP8 QVO | 5.58 s | 39.83260 | 0.98359 |
| CP4 FP8 QKVO | 5.45 s | 38.60712 | 0.97976 |

The final BF16 CP2 image is bitwise identical to the pre-change CP2 image
(`PSNR=inf`, `SSIM=1.00000`).

## 6. Tests and artifacts

Regression command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m pytest -q \
  tests/kernels/test_cuda_ulysses_fp8_ops.py \
  tests/kernels/test_ops_dispatch.py \
  tests/kernels/test_cuda_ext_manifest.py \
  tests/parallelism/test_distributed_primitives.py \
  tests/parallelism/test_topology.py \
  tests/parallelism/test_ring_buffer_reuse.py \
  tests/parallelism/test_templated_usp.py \
  tests/parallelism/test_fp8_a2a_alignment.py \
  tests/parallelism/test_bf16_a2a_alignment.py
```

Expected: 144 passed. Key guarantees:

- padded FP8 payload, scale, and dequantized BF16 are bitwise equal to the legacy path;
- padding poisoned with `255` does not change dequantization;
- 4-rank standard and UAA FP8 layouts match BF16;
- 4-rank BF16 padded layouts are bitwise equal to legacy unpadded layouts;
- optional `packed_head_size`/`head_size` support fake tensors and `fullgraph=True`.

Evidence:

- `.tmp/comm_opt_dev/report.md`
- `.tmp/comm_opt_dev/logs/test_final_all.log`
- `.tmp/comm_opt_dev/logs/final_fp8_wan_cp4.log`
- `.tmp/comm_opt_dev/logs/bench_bf16_pad_shape_only.log`
- `.tmp/comm_opt_dev/logs/e2e_metrics_final.log`

## 7. Known limits

- `fast_h3`-style `_all_to_all_single_qkv/o_custom_heads` BF16 paths do not yet have the
  automatic pad; common `D` multiples of 8 are already aligned.
- UAA FP8 O still uses the per-token layout and was not fused into an O-specific kernel.
- No Ring FP8 transport is implemented. See `references/ring.md`.
- The O FP8 path intentionally keeps the legacy epsilon `1e-4`; QKV uses `1e-6`. Do not
  unify them without a numerical gate.

## 8. Next optimization direction: packed QKV all-to-all

Status: **candidate, not implemented, not evaluated end-to-end**. Do not treat the current
estimates as a no-go decision; the async path has not been measured with a packed kernel.

### 8.1 Motivation

Today Q/K/V use three independent Ulysses all-to-all calls. A 3-to-1 packed call could reduce
three collective launches and two fixed `alpha` costs to one:

```text
q,k,v [B,S_L,H,D]
  -> pack [B,S_L,P,H_L,3,D]       (or 3*D in the last dimension)
  -> one all_to_all_single
  -> [S_G,B,H_L,3,D]
  -> unpack three [B,S_G,H_L,D]
```

At `alpha = 36 us` on this PCIe host, the theoretical saving is `2 * alpha = 72 us` per
layer, independent of payload size. The relative gain is therefore largest for small
payloads (CP8 image) and close to zero for video.

### 8.2 Two paths must be evaluated separately

1. **Synchronous `UlyssesAttention.forward`**
   - Current code sends Q, waits for Q, then sends K/V and waits.
   - Q/K/V are all available when attention starts, so a packed a2a is mechanically simple.
   - This is the lowest-risk place to prototype.

2. **`ulysses_async=True` patches (FLUX, Qwen-Image, etc.)**
   - Current sequence already overlaps: `send_v` with Q projection, `send_q` with K projection.
   - A naive packed a2a waits for K, then packs and sends, which can lose the existing overlap.
   - This does **not** prove packed a2a is bad. It means the experiment must measure
     *exposed* communication and compare:
     - current: hidden V/Q sends plus a late K send;
     - packed: one later send after pack;
     - possible variants: projection epilogue writing directly into the packed buffer,
       partial packing, or a separate stream for the pack.
   - Keep packed a2a behind an explicit flag and allow per-model enable/disable.

### 8.3 Required implementation pieces

- A fused pack/unpack path. Do not use `torch.cat` + `permute` as the production path: it
  adds a full relayout and can consume the entire `2 * alpha` saving.
- BF16: pack kernel writing `[P,S,B,H_L,3,D]` in one pass; unpack can use views plus the
  existing contiguous restore.
- FP8: a fused three-plane `permute + quant` kernel and a fused dequant/unpack kernel,
  reusing the `ulysses_fp8_*` per-token math and alignment rules.
- A `send_qkv` entry on `_All2AllComm` with its own metadata; keep `send_q/k/v` unchanged as
  the fallback.
- Eligibility guard: equal q/k/v shapes, same dtype/device, no GQA replicate-KV strategy,
  no `custom_heads` path, and no UAA uneven split initially.
- Alignment: the packed row width changes the per-rank chunk size; compute it for the packed
  layout and apply the same 16-byte rule.

### 8.4 Measurement plan

Before productionizing, run a dedicated A/B probe:

1. Per-layer a2a + pack/unpack timing for CP4 and CP8 image, and a Wan/video shape.
2. Both synchronous and `ulysses_async=True` paths.
3. Report:
   - total a2a time;
   - pack/unpack time;
   - exposed communication time (not only total);
   - e2e denoise time;
   - memory peak.
4. Accuracy: BF16 packed path must be `torch.equal` to three separate sends. FP8 packed
   path must match the existing Q/K/V FP8 path bitwise or within the established tolerance.
5. Model matrix: FLUX CP4/CP8, one video model, one non-async model.

### 8.5 Decision criteria

Proceed if at least one of these holds:

- CP8 image exposed communication drops by a measurable margin and e2e improves or is neutral;
- a video/long-sequence case shows a positive e2e result;
- CPU launch overhead is a demonstrated bottleneck and the packed path reduces it without
  increasing exposed GPU time.

Do not enable by default until the async overlap regression is measured. Record negative
results too, especially if pack/unpack or lost overlap eats the `2 * alpha` saving.

### 8.6 Initial A/B results (2026-11-12)

Environment: world=4, GPU 0-3, BF16, PCIe, no NVLink. The packed path uses three
`copy_` calls into a packed buffer, one a2a, and three unpack copies, matching the copy
bytes of the separate path. No fused pack kernel yet.

Pure a2a, no projection gaps:

| QKV payload | three_sync | three_async | packed | packed vs async |
|---|---:|---:|---:|---:|
| 9.0 MiB | 0.3081 ms | 0.3010 ms | 0.2927 ms | 1.028x |
| 18.0 MiB | 0.5565 ms | 0.5411 ms | 0.5218 ms | 1.037x |
| 36.0 MiB | 1.0455 ms | 1.0386 ms | 1.0192 ms | 1.019x |
| 72.0 MiB | 1.9390 ms | 1.8572 ms | 1.9556 ms | 0.950x |
| 239.9 MiB | 6.0167 ms | 5.8774 ms | 6.4578 ms | 0.910x |

With a projection GEMM between sends (models the async patch, not a quality test):

| Case | async + projection | packed + projection | packed / async |
|---|---:|---:|---:|
| image, QKV 18 MiB | 0.7432 ms | 0.9002 ms | 1.211x slower |
| video480p, QKV 239.9 MiB | 9.2289 ms | 11.3148 ms | 1.226x slower |

Findings:

- Packed wins only for small pure-comm payloads, roughly up to 36 MiB QKV on this host.
  At 72 MiB the extra pack/unpack scheduling already outweighs the two saved alphas.
- Once the real async overlap is modeled, packed is 21-23% slower: losing V/Q overlap is
  more expensive than saving two collectives.
- This does not close the direction. A projection epilogue that writes directly into the
  packed buffer removes the pre-pack copy and could change the crossover; that needs a
  fused QKV projection / epilogue experiment.
- A fused FP8 three-plane quant/dequant path is also still unmeasured.

Artifacts:

- `.tmp/comm_opt_dev/bench/probe_qkv_pack_a2a.py`
- `.tmp/comm_opt_dev/bench/probe_qkv_pack_overlap.py`
- `.tmp/comm_opt_dev/logs/bench_qkv_pack_a2a_sweep.log`
- `.tmp/comm_opt_dev/logs/bench_qkv_pack_overlap.log`

### 8.7 Compile interaction (important for the decision)

Eager async sends overlap with projections, but `torch.compile` changes both sides of that
trade:

- compiled projections/norm/rope are faster, so the compute window available to hide the
  V/Q sends shrinks;
- each async `send_*` crosses an opaque Python/collective boundary and introduces a graph
  break. Three sends split the compiled region at three points; a packed path has only one
  communication boundary.
- inductor's `reorder_for_compute_comm_overlap` cannot recover overlap across those opaque
  boundaries in the same way as a traceable collective.

FLUX.1-dev CP2 e2e, seed 0, 1024^2, warmup 2 / repeat 2:

| Path | Eager | Compile | Async gain |
|---|---:|---:|---:|
| sync Ulysses | 10.09 s | 8.89 s | - |
| async Ulysses | 9.25 s | 8.61 s | eager 8.3%, compile 3.1% |

So async still wins under compile, but the advantage shrinks by about two thirds. This
matches the L20 documentation trend (async benefit drops from about 4.8% to 2%).

Compile-aware micro A/B with projection GEMM, image CP4, QKV 18 MiB:

| Path | Eager | Compile |
|---|---:|---:|
| async + projection | 0.7482 ms | 0.8124 ms |
| packed + projection | 0.8993 ms | 0.8777 ms |
| packed / async | 1.202x slower | 1.080x slower |

The packed/async gap narrows from 20% to 8% because packed has one communication graph
break while async has three. The compiled async micro case is also slower than eager:
the three graph-break boundaries cost more than the compiled GEMMs save at this small
shape. Full-model compile still wins because the rest of the transformer benefits.

Implications:

- Do not evaluate packed QKV using eager timings only. The compile path is the relevant
  production path for many users.
- A packed implementation may become competitive after amortizing graph breaks, but it still
  needs a real implementation with a fused projection epilogue or fused pack kernel; the
  current torch packed prototype is not a production candidate.
- Keep async enabled under compile for now: it is still 3% better than sync on this host.
  But do not quote eager overlap gains for compiled runs.
- Measure both eager and compile in any future packed QKV A/B, and report graph-break count
  or nsys exposed-communication time.

Artifacts:

- `.tmp/comm_opt_dev/logs/compile_sync_eager.log`
- `.tmp/comm_opt_dev/logs/compile_async_eager.log`
- `.tmp/comm_opt_dev/logs/compile_sync_compile.log`
- `.tmp/comm_opt_dev/logs/compile_async_compile.log`
- `.tmp/comm_opt_dev/bench/probe_qkv_pack_compile.py`
- `.tmp/comm_opt_dev/logs/bench_qkv_pack_compile.log`

## 9. Per-group FP8 scales (2026-11-25)

Status: implemented and validated on `dev` with torch 2.15.0+cu132, RTX PRO 5000 sm120a,
GPU 4-7. The public config is `ulysses_float8_quant_type`; the CLI is
`--ulysses-float8-quant-type {per-token,per-group-16,per-group-32}`. The default remains
`per-token`, including the original output suffix and bit-exact kernel path.

### 9.1 Contract

- The six existing operators accept a trailing `group_size=0/16/32`; zero launches the unchanged
  legacy kernels. The Python bridge omits the new argument for zero so the default path also works
  with an older extension. Group dequant requires an explicit `head_size`.
- Packed rows are `[D FP8 payload][ceil(D/G) scales][zero padding]`. QKV keeps FP32 scales and
  epsilon `1e-6`; O and the generic/UAA path keep BF16 scales and epsilon `1e-4`. A final partial
  group is supported.
- `min_width = D + ceil(D/G) * scale_bytes`, followed by the existing chunk-minimal 16-byte search
  over every send/receive split. Do not replace this with row-width-only alignment.
- Quantization uses FP32 `inv_scale = 1 / scale` and the existing clamp -> FP16 -> FP8 rounding
  sequence. Group scales are read and written with `memcpy`; do not assume scale or row alignment.
- Group kernels use 2/4 lanes per 16/32-element group on the aligned vector path. D=24/40/72/264
  and D=66/68 cover fast-path tails and the scalar path. D=384/512 covers multiple warp iterations.

### 9.2 Validation

- `tests/kernels/test_cuda_ulysses_fp8_ops.py`: 91 passed, including byte-exact payload/scales and
  BF16 dequant reference, aligned vs misaligned views, padding poison, fake/fullgraph, empty,
  strided sequence, and an outlier reconstruction gate.
- Related dispatch/manifest/config/NCCL/BF16/Ring regression: 141 passed. The 4-rank FP8 test
  covers standard and UAA QKV/O for all three granularities and spies the actual `group_size`
  passed to all six operators.
- Full evidence and scripts are under `.tmp/ulysses_fp8_per_group/`; see `report.md`,
  `results.json`, `logs/`, `images/`, `comparison_cp4_qvo.png`, and `diff_cp4_qvo.png`.

### 9.3 Measured performance and precision

Operator CUDA Event medians, D=128. Times are milliseconds; values in parentheses are relative to
per-token.

| Op | per-token | group-16 | group-32 |
|---|---:|---:|---:|
| QKV quant | 0.010739 | 0.012228 (1.139x) | 0.012345 (1.149x) |
| QKV dequant | 0.007765 | 0.009950 (1.281x) | 0.009967 (1.284x) |
| O quant | 0.010765 | 0.012151 (1.129x) | 0.012322 (1.145x) |
| O dequant | 0.007554 | 0.009521 (1.260x) | 0.009517 (1.260x) |

Production QKV+O all-to-all, seq=4096, heads=24, D=128. `layer = 3*QKV + O`; packed widths are
QKV/O and every chunk is 16-byte aligned. FLUX shapes have zero padding in all three modes, so the
new scales increase traffic instead of fitting into existing padding.

| CP | Mode | Layer ms | vs per-token | Speedup vs BF16 | Packed widths | rel-L2 QKV/O |
|---|---|---:|---:|---:|---|---|
| 2 | per-token | 0.647601 | 1.000x | 1.583x | 132/130 | 0.025775/0.025825 |
| 2 | group-16 | 0.759348 | 1.173x | 1.350x | 160/144 | 0.022349/0.022412 |
| 2 | group-32 | 0.715284 | 1.105x | 1.433x | 144/136 | 0.024033/0.024104 |
| 4 | per-token | 0.469728 | 1.000x | 1.593x | 132/130 | 0.025767/0.025818 |
| 4 | group-16 | 0.545375 | 1.161x | 1.372x | 160/144 | 0.022348/0.022401 |
| 4 | group-32 | 0.508686 | 1.083x | 1.471x | 144/136 | 0.024031/0.024085 |

FLUX.1-dev, 1024^2, 28 steps, seed 0, `_sdpa_cudnn`, compile, warmup 2 / repeat 5, no cache or
weight quantization:

| CP | Mode | QVO s | QKVO s |
|---|---|---:|---:|
| 2 | per-token | 7.91 | 7.75 |
| 2 | group-16 | 8.02 | 7.92 |
| 2 | group-32 | 7.99 | 7.87 |
| 4 | per-token | 4.72 | 4.59 |
| 4 | group-16 | 4.80 | 4.74 |
| 4 | group-32 | 4.79 | 4.67 |

The group-16 operator-level reconstruction error is lower, but e2e PSNR does not improve
monotonically. CP4 group-32 QVO repeated in a fresh process is only 26.42 dB / 0.96434 SSIM from
its original image, so sampling/compile drift is larger than the granularity ranking. Do not claim
an e2e precision win from a single image pair. Visual comparison preserves composition and readable
text in all rows. The observed e2e cost is about 1-3%, within single-run noise, while the isolated
full communication layer is 8-17% slower; group modes remain opt-in.

## 10. FP6 E2M3 and unified communication configuration (2026-11-25)

Status: implemented and validated on `dev` with torch 2.15.0+cu132, sm120a, GPU 4-7. Full evidence:
`.tmp/ulysses_fp6/report.md` and `results.json`.

### 10.1 Public configuration and naming

- Use only `quant_communicate_kwargs={"bits": 6|8, "quant_type": ..., "quantize_k": ...}`.
  Empty means BF16. FP8 defaults to `per_token`; FP6 defaults to `per_group_32`. Both accept explicit
  `per_token`, `per_group_16`, and `per_group_32`. K is opt-in; LSE remains BF16.
- CLI: `--quant-communicate-params 'bits=6,quant_type=per_group_32,quantize_k=false'`. All old
  `--ulysses-float8*` options and their config fields are removed.
- `ulysses_fN_*` dispatches with keyword-only `bits=8`; direct `ulysses_fp6_*` and `ulysses_fp8_*`
  remain public. `_call_ulysses_*` bridge functions are never in `__all__`.
- Quantized primitives are `_all_to_all_*_fN_*`; row-width helpers are `_fN_*`. Selectors take
  `bits=None/0/6/8`: None follows config, zero forces BF16, and 6/8 select fN. The unused
  `_all_to_all_{qkv,o}_async_fn` wrappers and re-exports were removed.

### 10.2 FP6 packed contract and numerical pitfalls

- Four E2M3 codes occupy three little-endian bytes in a `uint8` transport tensor. Payload and total
  row width are 4-byte aligned; every rank chunk still uses the minimal 16-byte search. Both payload
  padding and trailing padding are zeroed.
- Aligned fast path: 16 elements/lane, 12 output bytes, three aligned u32 stores. Scalar path: four
  elements/lane with masked tails. QKV keeps FP32 scale/1e-6; O/generic keeps BF16 scale/1e-4.
- Require native conversion support (host check currently `major >= 10`; validated only on sm120a).
  Do not silently accept emulation on sm89. The CUDA source has `__has_include(<cuda_fp6.h>)` stubs
  for older toolkits.
- Use explicit IEEE `__fdiv_rn`/`__fmul_rn` for FP6 scale/normalization when byte-exact references
  matter: the extension builds with fast math. Independently, PyTorch CUDA FP32 division can be one
  ulp away from the correctly rounded scale (`1.640625/7.5`); generate the reference scale with FP64
  division and cast to FP32 before comparing bits.
- Round the minimum packed width up to the width step before searching chunk alignment. Starting
  at `payload+scales=2 mod 4` and incrementing by four can otherwise loop forever for some chunk
  row counts. Keep the D=72/BF16-scale regression.

### 10.3 Validation and measured outcome

- FP6 operators: 74 passed; FP8 operator regression: 91 passed; export/config/selector/manifest:
  152 passed; 4-rank FP6/FP8 x three granularities: 6 passed.
- D=128 QKV widths for per-token/group-16/group-32: FP8 132/160/144, FP6 100/128/112. O widths:
  FP8 130/144/136, FP6 100/112/104. FP6 quant/dequant is slower despite smaller payloads.
- FP6 FLUX CP2/CP4 QVO/QKVO times are approximately parity to 2% faster than the FP8 controls.
  Operator rel-L2 is ordered per-group-16 < per-group-32 < per-token for both bit widths. The CP4
  FP6 group-32 repeat floor is 37.02 dB / 0.99023 SSIM, so do not claim a stable e2e precision
  ranking from these image pairs. Visual outputs preserve composition and readable text.
- Second-level/global scale quantization is not part of this implementation; evaluate it separately
  from the bit-width/granularity change. See the design-only roadmaps in Sections 11 and 12.

## 11. Quant/dequant kernel optimization roadmap (design only, 2026-11-25)

Status: analysis and proposed experiments only; none of these optimizations is implemented yet. The
first goal is to reduce device-side quant/dequant cost without changing the FP8/FP6 wire format or
its numerical contract. Do not start with larger loads, more unrolling, or approximate arithmetic
before profiling the current kernels.

### 11.1 Current bottlenecks visible from the source

- **One warp per row underuses lanes.** FP8 uses eight elements per lane (`kPackSize=8`), so D=128
  activates only 16 of 32 lanes in the per-token path. FP6 uses 16 elements per lane
  (`kFp6FastPack=16`), so D=128 activates only 8 lanes in quant and dequant. D=64 is worse. Group
  kernels inherit this row-level underutilization even when their intra-group reduction is balanced.
- **Per-token quantization reads BF16 twice.** `fp8_per_token_quant.cu:15-17` calls `row_absmax`
  and then reloads the row in `quantize_row`. FP6 has the same two-pass structure and, worse,
  `row_absmax` uses eight elements/lane while `quantize_row_fp6` uses 16 elements/lane
  (`fp6_common.cuh:138-178`). The different mappings prevent simple register reuse.
- **Scalar conversion and packing work is substantial.** FP8 currently performs clamp -> FP16 ->
  FP8 per element. FP6 performs scalar float -> E2M3 conversion, unpacks codes to byte arrays, then
  spends shifts/masks assembling 12 bytes. Dequantization reverses those steps. Pair intrinsics are
  available in the installed CUDA headers and should be evaluated before hand-written LUTs.
- **Hot loops still contain runtime-only work.** Group count, lanes/group, payload width, integer
  division/modulo, alignment branches, scale broadcasts, and zero-padding loops are repeated per
  row. Fully unrolled local arrays are expected to stay in registers, but this must be proven from
  SASS rather than assumed.
- **The current measurements support this priority.** FP6 reduces packed bytes but is slower than
  FP8 in isolated quant/dequant. Optimizing the communication width again before reducing these
  device costs is unlikely to address the main bottleneck.

### 11.2 Mandatory profiling baseline

Run NCU before every optimization and again after it. Keep CUDA Event timings for the full public
op, but use NCU device duration to separate kernel work from `torch.empty`, torch.library, and
launch overhead.

Minimum matrix: generic/QKV/O; quant/dequant; D=64/128/256; group=0/16/32; aligned, tail, and
misaligned inputs; small and large row counts. Include the FLUX D=128 shapes and a small-grid case
so a lane-mapping change cannot win on large inputs while starving the GPU on small inputs.

Record at least:

- Kernel duration, SASS instruction count, registers/thread, occupancy and eligible/active warps.
- Local loads/stores or spills, global bytes, L2 hit rate, and warp stall breakdown.
- Shuffle, conversion, integer division, and packing instruction counts from PTX/SASS inspection.
- Host wall time versus device time. Do not attribute allocation or wrapper overhead to the CUDA
  kernel. Buffer reuse is a separate optimization track with async lifetime requirements.

Suggested first candidates are `fp6_per_token_{quant,dequant}` and
`fp8_qkv_permute_{quant,dequant}` at D=128. Save baseline logs and SASS before editing.

### 11.3 Priority 0: fix row-to-lane mapping and remove the second read

1. **Map multiple rows to one warp.** For D=128, FP8 can assign two rows/warp (16 lanes/row), and
   FP6 can assign four rows/warp (8 lanes/row). D=64 permits four and eight rows/warp respectively.
   Use a row-local segment:

   ```text
   lanes_per_row = D / elements_per_lane
   row_in_warp   = lane / lanes_per_row
   lane_in_row   = lane % lanes_per_row
   ```

   Per-token absmax must become a segmented reduction over `lanes_per_row`; group reductions must
   remain inside both the row segment and the quantization group. Do not use a full-warp reduction
   after this change. Add a small-row-count fallback so fewer blocks do not reduce occupancy.
   Increasing to one CTA per row is not a useful alternative for D=64/128.

2. **Fuse absmax and quantization with cached values.** For common D=64/128 specializations, load
   each BF16 value once, keep it in registers through absmax/shuffle, and quantize the cached value.
   FP6 must use the same 16-element mapping for both stages. Larger D should keep a bounded tile or
   the existing two-pass fallback instead of spilling an entire row to local memory. Group kernels
   already reuse values within an iteration; the per-token path is the first target.

These two changes address inactive lanes and duplicate global reads before micro-tuning arithmetic.
Validate them independently so their effects are not conflated.

### 11.4 Priority 1: paired conversion and register-native bit packing

- Evaluate `__nv_cvt_float2_to_fp6x2` and `__nv_cvt_fp6x2_to_halfraw2` for FP6. Consume the packed
  FP6x2 bytes directly when assembling the three u32 stores; avoid expanding every code into a
  separate byte array and then re-masking it. `PRMT` or funnel shifts are optional follow-ups, not
  assumptions of a win.
- Evaluate `__nv_cvt_halfraw2_to_fp8x2` after `__float22half2_rn` for FP8. This preserves the
  current clamp -> FP16 -> FP8 double-rounding sequence. Replacing it with direct float -> FP8 is a
  numerical-contract change and must not be presented as a drop-in optimization.
- Use paired conversion for dequantization and keep BF16x2/float2 values in native registers.
  `__bfloat1622float2`, `__habs2`, and `__hmax2` are candidates only after NaN, zero, subnormal,
  rounding, and SASS checks. A 64/256-entry lookup table is not the default: indexed constant loads
  can replay and shared-memory lookups add traffic and possible bank conflicts.
- Check SASS for local-memory traffic. Use bounded `uint4`/`uint2` register objects and fully
  unrolled small packs; if FP6 arrays spill, reduce values per lane or split the kernel rather than
  increasing unrolling blindly.

### 11.5 Priority 2: fold constants and simplify hot loops

- Start with `GroupSize` and scale-dtype templates. Add D=64/128 specializations only if profiling
  shows meaningful loop/index overhead; retain a generic D fallback to control binary size.
- With compile-time D/group, fold group count, lanes/group, payload bytes, scale offset, tail size,
  and padding length. Power-of-two lane counts should become masks/shifts rather than integer
  div/mod. Avoid a large template explosion across all shapes.
- Hoist alignment decisions out of `load_fp8_pack`/`store_fp8_pack` and FP6 pack helpers. Aligned
  hot paths should not repeat per-tile pointer branches; keep separate misaligned fallbacks.
- Eliminate zero-length padding loops at compile time and use vector stores for real padding.
  Vector-load scale arrays once per row when profitable, then broadcast from registers/shuffles.
- For QKV/O permute kernels, profile int64 index division. Consider 32-bit indexing and grid-axis
  coordinate mapping for partition/head/sequence before adding approximate index arithmetic.

Constant-folding arithmetic must preserve bits. In particular, do not replace `x / C` with
`x * (1.0f / C)`, or change `__fdiv_rn`/`__fmul_rn`, without an exhaustive check over all possible
BF16 absmax inputs and the subsequent scale/inv-scale results. Otherwise keep the current division.

### 11.6 Priority 3: occupancy, memory policy, and host-side overhead

- Tune 128/256/512-thread blocks and rows/block only after register and occupancy measurements.
  `__launch_bounds__` is acceptable only when it does not introduce spills. Multi-row mapping must
  retain enough blocks for small inputs.
- Non-temporal loads/stores are a later experiment for a proven memory-bound kernel; they can also
  change NCCL/cache interaction. Do not add shared-memory staging or extra barriers by default.
- Output allocation and wrapper overhead need a separate benchmark. A workspace pool may help, but
  quant/dequant output cannot be reused until the corresponding all-to-all and consumer have
  finished. Correct async lifetime is more important than allocation savings.
- Longer term, fusing quantization into the producer or dequantization into the consumer can remove
  an entire BF16 round trip. This is a larger architectural change and comes after the local kernel
  wins are established.

### 11.7 Experiment order and acceptance gates

Land and measure one change at a time: multi-row mapping -> per-token one-pass reuse -> paired
conversion/bit packing -> constant/hot-loop cleanup -> block and memory tuning.

Every candidate must report before/after device time, SASS instructions, registers, occupancy,
local/global bytes, public-op latency, and full communication-layer latency. Required correctness
includes all current byte-exact FP8/FP6 tests, tails, zero rows, padding poison, misaligned views,
fake/fullgraph, and four-rank NCCL tests. Reject a candidate that improves NCU duration but causes
register spills, small-grid regression, accuracy drift, or material FLUX e2e regression.

### 11.8 Fused QKV mega quant/dequant and one all-to-all (design only)

The current Q, K, and V paths each perform quantize/permute, all-to-all, and dequantize/restore.
A complementary optimization is to execute **one BF16/FP8/FP6 fused all-to-all** and one mega
quant/dequant launch over the union of QKV rows. This targets collective launch latency, host work,
allocations, and small-message overhead rather than arithmetic instructions, so it can be evaluated
in parallel with the lane-mapping work above.

The structural entry points can stay small:

```text
_FusedAll2AllComm
_all_to_all_fused_qkv_{bf16,fN}_async          # standard/even splits
_all_to_all_fused_qkv_any_{bf16,fN}_async      # UAA/uneven splits
```

`_FusedAll2AllComm` should cache metadata, offsets, and the selected implementation once, expose a
`send_qkv(q, k, v)` operation, and return one async handle whose wait yields `(q, k, v)`. Keep the
existing separate communicator as a fallback. No new public quantization option is required for the
experiment.

**The fused chunk layout is the correctness-critical contract.** For every source/destination
partition, the bytes sent in one fused chunk must be exactly:

```text
fused_chunk(partition) = q_chunk(partition) || k_chunk(partition) || v_chunk(partition)
```

Do not flatten as `[all Q partitions][all K partitions][all V partitions]`: all-to-all would split
that buffer across ranks and assign the wrong tensor/sequence ownership. For even splits, use a
partition-major layout with QKV inside each partition. For UAA, derive every split and byte offset
from the three old per-tensor splits so each fused send/receive chunk is the concatenation of the
legacy chunks. Each legacy chunk is already 16-byte aligned, so their sums remain aligned without
inventing new padding.

Mixed formats must be supported from the first design:

- Q and V use the configured FP8/FP6 mode. K uses that same mode only when `quantize_k=True`;
  otherwise K remains BF16. LSE is outside this fusion and remains BF16.
- Q/K/V segment widths and scale offsets may therefore differ. Store explicit per-partition,
  per-tensor byte offsets rather than assuming one packed row width.
- GQA can have different Q and K/V head counts. Require matching B/S/D for the first implementation
  and fall back to the separate path otherwise; do not silently flatten incompatible shapes.

The mega quant kernel should read all three inputs and write the fused output directly, with no
intermediate packed Q/K/V tensors and no host `torch.cat`. The mega dequant kernel performs the
inverse operation into three outputs. A grid over the union of logical tensor rows is sufficient;
initially keep a warp within one tensor/format to avoid mixed-format intra-warp divergence. BF16
uses the same fused copy/permute structure without quantization. After the layout is proven, apply
the multi-row lane mapping and paired conversions from Sections 11.3-11.4.

A low-risk implementation sequence is:

1. Build a reference fused buffer from legacy packed chunks and verify it byte-for-byte before
   enabling the collective. This is a correctness oracle, not the production fast path.
2. Add the fused primitive and communicator with one all-to-all, then add direct mega
   quant/dequant kernels to remove temporary buffers and extra launches.
3. Benchmark separate `3 * (quant + a2a + dequant)` against fused `1 * (mega quant + a2a + mega
   dequant)` at CP2/CP4, BF16 and every FP8/FP6 granularity. Measure host calls, NCCL duration,
   whole-attention-layer time, and FLUX e2e, not just the mega kernels.

Required tests include byte equality with the three legacy paths, `quantize_k` on/off, GQA head
counts, uneven sequence splits, aligned/tail/misaligned inputs, sync/async, fake/fullgraph, and
four-rank NCCL. A single larger collective may change overlap or perform differently across PCIe
and NVLink; keep size/shape-based fallback and reject fusion if the collective regression exceeds
the launch and kernel savings.

## 12. Global-scale / second-level quantization design (design only, 2026-11-25)

Status: precision research plan only; no public API or kernel implementation yet. A global scale is
not automatically an accuracy improvement. It can also be merely a way to compress group scales,
and FP6 payload rounding may dominate all scale-rounding errors.

### 12.1 Decompose the error before choosing a format

For group `g`, let `A_g = max(abs(x_g))` and `F = 7.5` for FP6 or `448` for FP8. The current scale is
approximately `s_g = max(A_g / F, epsilon)`. Quantization uses `Q(x_i / s_g)`; dequantization uses
the stored scale. O/generic currently store BF16 scales, while QKV stores FP32 scales.

Run a PyTorch-only ablation first:

1. Current payload + current stored scale.
2. Current payload + oracle FP32 scale on every path (scale-rounding upper bound).
3. Current payload + proposed normalized local scale.
4. BF16/FP16/FP8 local-scale variants with the same global-scale computation.

Report rel-L2, RMSE, max absolute error, and error per group. Include random normal data, one large
outlier per group, high dynamic-range rows, zero rows, tiny values, and tails. If oracle FP32 scales
do not materially improve over the current format, second-level scales are the wrong target; stop
and work on payload rounding or group granularity instead. Do not infer precision from one FLUX
image because sampling/compile drift is larger than the granularity effect.

### 12.2 Freeze the meaning of "global" first

- **Row-global (recommended first experiment): one global scale per row/token.** It needs no
  cross-row reduction and fits the existing warp-per-row model. The overhead is 2-4 bytes per row,
  negligible relative to the payload.
- **Tensor-global: one scale for an entire Q/K/V/O tensor.** This needs an extra global amax pass or
  a block-reduction plus synchronization before quantization. It also allows a single outlier to
  affect every row. Its extra kernel/synchronization cost can exceed the scale savings, so it is a
  separate experiment, not the default design.

### 12.3 Candidate normalized-scale contract

For a row with `M = max(abs(x))`, define for `M > 0`:

```text
S   = 2^ceil(log2(M / F))          # exact power-of-two global scale
l_g = A_g / (F * S)                # normalized group scale, in (0, 1]
s_g = S * l_g                      # effective scale used by quantization
```

For `M == 0`, use `S = 1` and zero payload with the existing epsilon-safe scale rule. To preserve
the current epsilon semantics, use `l_g = max(A_g / (F * S), epsilon / S)` before local-scale
storage. Quantization still uses the full effective `s_g`; dequantization uses `S * stored(l_g)`.
An independent reference must reproduce that asymmetry and the exact rounding order. A second
variant that quantizes using the already-rounded effective scale is a different numerical contract
and must be evaluated separately.

The packed layout becomes:

```text
[payload][global S][stored local scales][zero padding]
```

Use FP32 `S` initially for a simple reference implementation. Because `S` is a power of two,
multiplying or dividing by it is exact except at overflow/underflow boundaries. Tail groups still
use only valid elements, and padding remains zero.

Candidates to compare:

- **FP32 global + FP16 local scales (precision candidate for O/generic).** FP16 has 10 mantissa bits
  versus BF16's 7 at the same two bytes/group, plus four bytes/row for `S`. This can improve scale
  precision without a significant traffic increase. It does not change the FP6/FP8 payload codebook.
- **FP32 global + FP8 local scales (primarily a compression candidate).** This may reduce scale
  bytes, including QKV's current FP32 scales, but must not be marketed as an accuracy improvement
  unless the measurements show one.
- **FP32 local oracle.** A measurement upper bound, not necessarily a production format.

Reject rows whose normalized local scales underflow FP16; keep a direct-scale fallback for extreme
dynamic range. Preserve zero rows and the existing NaN/Inf behavior. Second-payload residual
quantization is out of scope for this near-zero-traffic design because it approximately doubles
payload traffic.

### 12.4 Implementation and acceptance sequence

1. Implement only a scalar/reference path after the oracle ablation supports the idea. Compare
   current direct scales, FP16 local scales, and FP8 local scales on identical inputs.
2. Keep the experiment internal; do not add public config keys until a winning format and scope are
   frozen. If adopted, `quant_communicate_kwargs` is the correct future extension point.
3. Add byte-exact packed-layout tests, scale/dequant reference tests, tails, zero/tiny/outlier rows,
   padding poison, fake/fullgraph, and four-rank NCCL coverage before vectorizing the new path.
4. Measure packed bytes, scale bytes, quant/dequant latency, full all-to-all latency, and FLUX
   PSNR/SSIM over multiple seeds. A precision claim requires a consistent operator-level gain and
   an e2e effect above the repeat-noise floor; a byte reduction alone is a compression result.
5. Only after precision and traffic are proven, apply the Section 11 paired-conversion, lane-mapping,
   and hot-loop optimizations to the new layout.
