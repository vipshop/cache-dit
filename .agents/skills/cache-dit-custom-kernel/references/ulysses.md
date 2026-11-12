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
