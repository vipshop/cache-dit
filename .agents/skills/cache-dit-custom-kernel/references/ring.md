# Ring 通信：现状、Ring2 实测与 FP8 可行性

Status: recorded 2026-11-12 after commit `c40cea09`.
Scope: `src/cache_dit/distributed/parallel/ring.py`,
`src/cache_dit/distributed/communicate/primitives.py` (`_RingP2PComm`).
Machine: RTX PRO 5000 72GB, sm120a, PCIe, no NVLink; tests use GPU 0-3.
No Ring FP8 transport is implemented. This file records the current transport, the measured
buffer-reuse result, and the design constraints for a future FP8 experiment.

## 1. Current transport

`RingAttention.forward` loops over `world_size` steps:

```text
for step in range(world_size):
  if step + 1 != world_size:
    next_k, next_v = comm.send_recv_kv(key, value, k_buffer, v_buffer)
  out, lse = forward_op(query, key, value, ...)
  merge(prev_out, prev_lse, out, lse)
  if step + 1 != world_size:
    comm.wait()
    key, value = next_k, next_v
```

- `_RingP2PComm.send_recv_kv` calls `send_recv` twice (K and V), then one `commit()`.
- `commit()` is a single `dist.batch_isend_irecv(self._ops)` carrying four `P2POp`s:
  `isend(K)`, `irecv(K)`, `isend(V)`, `irecv(V)`.
- Therefore "merge the K/V p2p commits" is already done; repeating it is a no-op.
- The p2p is launched before the current hop's attention and `wait()` happens after it, so
  p2p is already overlapped with current-hop compute.
- Ring does not do an all-gather of K/V. Each hop rotates the whole local K/V shard.
- `convert_to_fp32` affects only the attention state merge, not the p2p dtype.
- Ring LSE is produced by attention and merged locally; it is not communicated.

The p2p buffer is currently BF16. The transport accepts an optional receive tensor, which is
what makes buffer reuse possible without changing the collective.

## 2. Buffer reuse change

Before: every hop allocated `torch.empty_like(key)` and `torch.empty_like(value)` inside
`send_recv` (two allocations per hop; four for K/V over a 2-hop ring).

After: `RingAttention.forward` preallocates two pairs of contiguous K/V receive buffers and
alternates by hop parity:

```text
slot[i % 2] = receive destination for hop i
send source for hop i = slot[(i - 1) % 2]
```

The two slots are always different, and `wait()` completes before the next hop consumes the
received tensors. `world_size == 2` allocates only one pair. Buffers use
`memory_format=torch.contiguous_format`; reusing a non-contiguous `empty_like` could silently
scramble NCCL's linear payload.

This change does **not** alter send/compute/wait ordering, K/V ordering, or numerical output.
The 4-rank regression asserts buffer IDs are exactly `[A, B, A]` and the output is equal to
the pre-change implementation.

## 3. Measured Ring2 performance

FLUX CP2 local shape: `B=1, S_LOCAL=2048, H=24, D=128`, K/V 12.0 MiB each, 2 ranks.

| Case | Legacy (allocate per hop) | Reuse buffers | Difference |
|---|---:|---:|---:|
| `RingAttention` 1 hop + merge | 0.7088 ms | 0.6982 ms | -0.0106 ms (1.015x) |
| K/V p2p rotation x10 | 6.2332 ms | 6.2239 ms | -0.0092 ms (1.001x) |
| FLUX Ring2 e2e, 1024^2, 28 steps | 9.30 s | 9.28 s | -0.02 s (~0.2%) |

E2E output comparison: `PSNR=inf`, `SSIM=1.00000`.

Interpretation:

- The receive-buffer allocation is a caching-allocator hit, worth only about 10 us per hop.
- Ring2 latency is dominated by the 24 MiB K/V p2p payload, not by allocator calls.
- The observed 0.2% e2e difference is at the noise floor. This change is a lifecycle/allocator
  cleanup, not a performance lever.
- Do not cite the buffer reuse as a Ring speedup. The measured evidence only supports
  "does not regress, removes per-hop allocations, preserves bitwise output".

Artifacts:

- `.tmp/comm_opt_dev/logs/bench_ring2_buffer.log`
- `.tmp/comm_opt_dev/logs/e2e_ring2_before.log`
- `.tmp/comm_opt_dev/logs/e2e_ring2_after.log`
- `.tmp/comm_opt_dev/logs/ring2_metrics.log`
- `.tmp/comm_opt_dev/bench/probe_ring2_buffer.py`

## 4. Ring FP8 feasibility

Technically feasible. Ring differs from Ulysses in one important way: there is no head or
sequence permute. The transport simply rotates the whole local K/V shard, so a Ring FP8 path
only needs per-token quantize/dequantize and buffer management.

A safe design:

```text
local K/V BF16
  -> quantize once to FP8 packets with per-token scale
  -> hop i: isend/irecv FP8 packets
  -> compute hop i with the current BF16 K/V
  -> wait
  -> dequantize received FP8 packet into a BF16 compute buffer
  -> relay the original FP8 packet on hop i+1
```

The relay must forward the original FP8 packet, not re-quantize the dequantized BF16 tensor.
Re-quantizing each hop would compound error in Ring4/Ring8.

### 4.1 What can be reused

The per-token math in `csrc/kernels/ulysses/fp8_per_token_*.cu` is generic over
`[rows, D]`, so it can quantize/dequantize K/V. However, the public names are deliberately
Ulysses-specific. A Ring implementation should either:

- add generic `comm_fp8_per_token_*` operators while keeping `ulysses_fp8_*` as Ulysses
  wrappers, or
- add `ring_fp8_*` wrappers over the same kernel.

Do not call `ulysses_fp8_*` from Ring as the public contract; it makes the transport semantics
ambiguous and was the reason the family was renamed after the Ulysses refactor.

### 4.2 Alignment and layout

- Pack the per-token scale into each row; do not send a separate scale tensor.
- Choose the FP8 row width so the entire p2p packet is 16-byte aligned. The packet size is
  `B * S_LOCAL * H * packed_width`; for Ring there is one peer, so the per-chunk rule becomes
  the whole packet size.
- BF16 `D` is usually a multiple of 8 or 16, so the BF16 Ring payload is already aligned.
  FP8 `D+2`/`D+4` reintroduces the same alignment cliff as Ulysses, so Ring FP8 must pad.
- No head padding is needed for Ring.
- Keep a BF16 compute copy plus FP8 relay packets. The extra FP8 packet memory is about
  0.5x the BF16 K/V footprint; the BF16 receive/dequant buffer can be reused.

### 4.3 Accuracy

- Quantize V first; V FP8 directly perturbs attention output but is usually less sensitive
  than K.
- K FP8 perturbs `Q @ K^T` logits; require a real model PSNR/SSIM gate before enabling it.
- LSE remains BF16.
- Use the same per-token scale grain as Ulysses; do not invent per-channel K scaling without
  a separate accuracy study.
- A Ring4/Ring8 relay path should be tested for cumulative error; a Ring2 test cannot expose
  it because each rank only receives one peer shard once.

### 4.4 Expected performance

Using the Ring2 measurements above:

| Item | BF16 | Estimated FP8 |
|---|---:|---:|
| K/V bytes per hop | 24 MiB | about 12.3 MiB including scale |
| p2p transfer | about 0.70 ms | about 0.38-0.45 ms |
| quantize + dequantize | 0 | about 0.05-0.10 ms |
| RingAttention one hop | 0.71 ms | about 0.48-0.55 ms |

But Ring already overlaps p2p with the current-hop attention. For FLUX image, per-layer
compute is much larger than 0.7 ms, so most of the p2p saving is hidden. The e2e upside on
FLUX Ring2 is expected to be below 0.5%, and may be negative if dequantization becomes a
serial bubble.

The worthwhile cases are:

- Ring4/Ring8, where the number of hops and bytes per layer grow;
- video or long-sequence K/V payloads;
- models with a low compute-to-p2p ratio;
- deployments where Ring p2p is not fully hidden by attention.

## 5. Recommended next steps

1. Build a V-only FP8 Ring2/Ring4 prototype; keep K BF16.
2. Add packet-level alignment and a 4-rank relay accuracy test.
3. Run FLUX Ring2 and a video Ring4 matrix: latency, PSNR, SSIM, and visual inspection.
4. Only then add K FP8 behind an explicit `ring_float8_k` switch.
5. Keep the default off; Ring2 image does not justify enabling it.

Do not prioritize Ring2 FP8 as an e2e optimization before measuring Ring4/video; current
evidence says the FLUX Ring2 gain is within noise.
