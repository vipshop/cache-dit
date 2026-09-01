<!-- PTX ISA 9.1 -->

# Blackwell (sm_100) -- tcgen05 & New Features

## sm_100 / sm_100a / sm_100f Target Differences

| Target | Features enabled |
|--------|-----------------|
| `sm_100` | Virtual arch, no tcgen05 |
| `sm_100a` | All tcgen05, `.kind::i8`, `.kind::mxf4nvf4`, `.scale_vec::1X/2X/4X`, `scale-input-d` |
| `sm_100f` | Most tcgen05 (not `.kind::i8` alone, not `.scale_vec::NX`), `.block16/.block32`, `setmaxnreg`, introduced PTX 8.8 |

All tcgen05 instructions in a kernel **must** use the same `.cta_group` value.

## .blocksareclusters Directive

### Syntax
```ptx
.blocksareclusters
```
### Constraints
- Introduced PTX ISA 9.0.
- Specifies that CUDA thread blocks are mapped to clusters.
- Kernel-level directive.

## Tensor Memory (TMEM)

- 512 columns x 128 lanes (rows) per CTA, each cell 32 bits.
- Address: bits[31:16] = lane, bits[15:0] = column.
- Allocation unit: 32 columns, power of 2, range [32, 512].
- Divided into 4 chunks: warp N in warpgroup accesses lanes `[32*N, 32*N+31]`.

## tcgen05.alloc / dealloc / relinquish_alloc_permit

### Syntax
```ptx
tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols;
tcgen05.dealloc.cta_group.sync.aligned.b32               taddr, nCols;
tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- `nCols` in [32, 512], power of 2. Warp-level collective. Must dealloc before kernel exit.
- `.cta_group::2`: one warp from each peer CTA collectively; may block.

## tcgen05.mma

### Syntax
```ptx
// Dense, no block scaling:
tcgen05.mma.cta_group.kind [d-tmem], a-desc, b-desc, idesc,
    {disable-output-lane}, enable-input-d {, scale-input-d};
tcgen05.mma.cta_group.kind [d-tmem], [a-tmem], b-desc, idesc,
    {disable-output-lane}, enable-input-d {, scale-input-d};

// With block scaling (mx kinds):
tcgen05.mma.cta_group.kind.block_scale{.scale_vectorsize}
    [d-tmem], a-desc, b-desc, idesc,
    [scale-A-tmem], [scale-B-tmem], enable-input-d;

.kind     = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8,
              .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Variants
- `tcgen05.mma.sp` -- sparse A matrix (adds `[sp-meta-tmem]` operand).
- `tcgen05.mma.ws` -- weight stationary (only `.cta_group::1`).
- `tcgen05.mma.ws.sp` -- weight stationary + sparse A.
- `.collector::a::{fill,use,lastuse,discard}` (activation stationary, A buffer).
- `.collector::bN::{fill,use,lastuse,discard}` (weight stationary, N=0-3).
- `.ashift` -- shifts A rows down by 1 in TMEM (M=128 or 256 only).
- `scale-input-d` -- `D = A*B + D * 2^(-scale)`, scale in [0,15], `.kind::f16`/`.kind::tf32` only (`sm_100a`).

### Shape/Type Summary (cta_group::1, dense, no .ws)

| `.kind` | dtype | atype/btype | M | N | K |
|---------|-------|-------------|---|---|---|
| `f16` | f16/f32 | f16, bf16 | 64, 128 | 8..256 step 8 | 16 |
| `tf32` | f32 | tf32 | 64, 128 | 8..256 step 8 | 8 |
| `f8f6f4` | f16/f32 | e4m3,e5m2,e2m3,e3m2,e2m1 | 64, 128 | 8..256 step 8 | 32 |
| `i8` | s32 | s8, u8 | 64, 128 | 8,16,24,32,48..256 step 16 | 32 |
| `mxf8f6f4` | f32 | above x ue8m0 | 128 | 8..256 step 8 | 32 |
| `mxf4` | f32 | e2m1 x ue8m0 | 128 | 8..256 step 8 | 64 |
| `mxf4nvf4` | f32 | e2m1 x ue8m0/ue4m3 | 128 | 8..256 step 8 | 64 |

**cta_group::2**: M doubles (128/256), N steps become 16.
**ws shapes** (cta_group::1 only): M={32,64,128}, N={64,128,256}.

### Instruction Descriptor (idesc, 32-bit register)

| Bits | Field | Encoding |
|------|-------|----------|
| 0-1 | Sparsity selector | 0-3 |
| 2 | Sparse flag | 0=dense, 1=sparse |
| 3 | Saturate (i8 only) | 0/1 |
| 4-5 | dtype | f16=0, f32=1, s32=2 |
| 7-9 | atype | kind-dependent |
| 10-12 | btype | kind-dependent |
| 13 | Negate A | 0/1 |
| 14 | Negate B | 0/1 |
| 15 | Transpose A | 0/1 |
| 16 | Transpose B | 0/1 |
| 17-22 | N >> 3 | |
| 24-28 | M >> 4 | |
| 30-31 | Max shift (.ws B-reuse) | 0=none, 1=8, 2=16, 3=32 |

### Block Scaling (.scale_vectorsize)

| Qualifier | Alias for | Applies to |
|-----------|-----------|------------|
| `.scale_vec::1X` | `.block32` (mxf8f6f4) | `sm_100a` |
| `.scale_vec::2X` | `.block32` (mxf4, mxf4nvf4) | `sm_100a` |
| `.scale_vec::4X` | `.block16` (mxf4nvf4) | `sm_100a` |
| `.block16` | -- | `sm_100f`, `sm_110f` |
| `.block32` | -- | `sm_100f`, `sm_110f` |

### Sparse Matrices

| `.kind` | Sparsity pattern |
|---------|-----------------|
| `tf32` | 1:2 |
| `f16/f8f6f4/mxf8f6f4/i8` | 2:4 |
| `mxf4/mxf4nvf4` | 4:8 pairwise structured |

### Example
```ptx
tcgen05.mma.cta_group::1.kind::tf32 [taddr0], adesc, bdesc, idesc, {m0,m1,m2,m3}, p;
tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale
    [taddr2], [taddr1], bdesc, idesc, [sf_a], [sf_b], p;
tcgen05.mma.ws.cta_group::1.kind::i8.collector::b2::use
    [taddr2], [taddr1], bdesc, idesc, p;
```

## tcgen05.cp -- Shared Memory to TMEM

### Syntax
```ptx
tcgen05.cp.cta_group.shape{.multicast}{.dst_fmt.src_fmt} [taddr], s-desc;
.shape     = { .128x256b, .4x256b, .128x128b, .64x128b, .32x128b }
.multicast = { .warpx2::02_13, .warpx2::01_23, .warpx4 }
.src_fmt   = { .b6x16_p32, .b4x16_p64 }
.dst_fmt   = { .b8x16 }
```
### Constraints
- `.64x128b` requires `.warpx2::02_13` or `.warpx2::01_23`.
- `.32x128b` requires `.warpx4`.
- Decompression: 4-bit->8-bit (`.b4x16_p64`->`.b8x16`), 6-bit->8-bit (`.b6x16_p32`->`.b8x16`).

### Example
```ptx
tcgen05.cp.cta_group::1.128x256b [taddr], sdesc;
tcgen05.cp.cta_group::2.128x128b.b8x16.b6x16_p32 [taddr], sdesc;
```

## tcgen05.ld / tcgen05.st

### Syntax
```ptx
tcgen05.ld.sync.aligned.shape.num{.pack::16b}.b32   r, [taddr];
tcgen05.st.sync.aligned.shape.num{.unpack::16b}.b32  [taddr], r;
.shape = { .16x64b, .16x128b, .16x256b, .32x32b, .16x32bx2 }
.num   = { .x1, .x2, .x4, .x8, .x16, .x32, .x64, .x128 }
```
### Variants
- `tcgen05.ld.red` -- load with `.min`/`.max` reduction (`.32x32b` or `.16x32bx2`, `.x2` minimum).
- `.16x32bx2` takes additional `immHalfSplitoff` immediate operand.

### Register count per .num

| .num | .32x32b/.16x64b/.16x32bx2 | .16x128b | .16x256b |
|------|---------------------------|----------|----------|
| .x1 | 1 | 2 | 4 |
| .x2 | 2 | 4 | 8 |
| .x4 | 4 | 8 | 16 |
| .x8 | 8 | 16 | 32 |
| .x16 | 16 | 32 | 64 |
| .x32 | 32 | 64 | 128 |
| .x64 | 64 | 128 | N/A |
| .x128 | 128 | N/A | N/A |

## tcgen05.shift

### Syntax
```ptx
tcgen05.shift.cta_group.down [taddr];
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- Shifts 32-byte elements down by one row (all rows except last). Lane of `taddr` must be aligned to 32.

## tcgen05.fence

### Syntax
```ptx
tcgen05.fence::before_thread_sync ;
tcgen05.fence::after_thread_sync  ;
```
### Constraints
- `before_thread_sync`: orders prior async tcgen05 ops before subsequent sync/execution ops.
- `after_thread_sync`: orders subsequent async tcgen05 ops after prior sync/execution ops.

## tcgen05.commit

### Syntax
```ptx
tcgen05.commit.cta_group.mbarrier::arrive::one{.shared::cluster}{.multicast::cluster}.b64
    [mbar] {, ctaMask};
.cta_group = { .cta_group::1, .cta_group::2 }
```
### Constraints
- Tracks completion of prior async tcgen05 ops (mma/cp/shift) from current thread.
- Triggers arrive-on with count=1 at cluster scope. Optional `.multicast::cluster` with 16-bit `ctaMask`.

## tcgen05.wait

### Syntax
```ptx
tcgen05.wait::ld.sync.aligned;
tcgen05.wait::st.sync.aligned;
```
### Constraints
- Blocks until all prior `tcgen05.ld` (or `.st`) from executing thread have completed.

## 2CTA / CTA Pair Mode

- **CTA pair**: two CTAs in a cluster whose `%cluster_ctarank` differs only in bit 0.
- `.cta_group::2`: tcgen05 ops access TMEM of both CTAs in the pair.
- `.cta_group::1`: operate on current CTA's TMEM only.

### Issue Granularity

| Operation | cta_group::1 | cta_group::2 |
|-----------|-------------|-------------|
| mma, cp, shift, commit | 1 thread | 1 thread from CTA pair |
| alloc, dealloc, relinquish | 1 warp | 1 warp from each peer CTA (blocking) |
| ld, st, wait | 1 warp (N/A) | N/A |
| fence | 1 thread (N/A) | N/A |

### Example (dealloc with 2CTA)
```ptx
// Both CTA0 and CTA1 warps must participate:
barrier.cluster.arrive;
barrier.cluster.wait;
tcgen05.dealloc.cta_group::2.sync.aligned.b32 taddr, 32;
exit;
```

## Shared Memory Descriptor (64-bit)

| Bits | Field |
|------|-------|
| 0-13 | Matrix start addr `(addr & 0x3FFFF) >> 4` |
| 16-29 | Leading dim byte offset/addr (encoded same way) |
| 32-45 | Stride dim byte offset |
| 46-48 | Fixed `0b001` |
| 49-51 | Matrix base offset |
| 52 | Leading dim mode: 0=relative, 1=absolute |
| 61-63 | Swizzle: 0=none, 1=128B+32B atom, 2=128B, 4=64B, 6=32B |

## Pipelined Instruction Pairs

| Producer -> Consumer | Same cta_group, additional constraints |
|---------------------|-----------------------------------------|
| `mma -> mma` | Same accumulator and shape |
| `cp -> mma` | Same cta_group |
| `shift -> mma` | Same cta_group |
| `mma -> shift` | Same cta_group |
| `shift -> cp.4x256b` | Same cta_group |
| `mma/cp/shift -> commit` | Implicit pipeline |
| `ld -> wait::ld` | Implicit pipeline |
| `st -> wait::st` | Implicit pipeline |
