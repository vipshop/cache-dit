<!-- PTX ISA 9.1 -->

# Async Copy & TMA Operations

## cp.async (per-thread, non-bulk)

### Syntax

```ptx
cp.async.COP.shared{::cta}.global{.L2::cache_hint}{.L2::prefetch_size}
        [dst], [src], cp-size{, src-size}{, cache-policy};
cp.async.COP.shared{::cta}.global{.L2::cache_hint}{.L2::prefetch_size}
        [dst], [src], cp-size{, ignore-src}{, cache-policy};

.COP        = { .ca, .cg }
cp-size     = { 4, 8, 16 }       // bytes; .cg requires cp-size=16
```

### Constraints

- `sm_80`+, PTX 7.0+.
- `.ca`: cache all levels. `.cg`: L2 only, forces `cp-size=16`.
- Optional `src-size` (u32, < cp-size): copies `src-size` bytes, zero-fills rest.
- Optional predicate `ignore-src`: if true, writes zeros to dst (PTX 7.5+).
- Weak memory operation; no ordering without explicit sync.
- Alignment: `dst` and `src` aligned to `cp-size`.

### Example

```ptx
cp.async.ca.shared.global  [shrd], [gbl + 4], 4;
cp.async.cg.shared.global  [%r2], [%r3], 16;
cp.async.ca.shared.global  [shrd], [gbl], 4, p;       // predicated ignore
```

## cp.async.commit_group / cp.async.wait_group

### Syntax

```ptx
cp.async.commit_group ;
cp.async.wait_group N ;        // N = integer constant; wait until <= N groups pending
cp.async.wait_all ;            // equivalent to commit_group + wait_group 0
```

### Constraints

- `sm_80`+, PTX 7.0+.
- Groups complete in commit order. No ordering within a group.
- Two `cp.async` ops writing to the same location within one group is undefined.

### Example

```ptx
cp.async.ca.shared.global [buf0], [gbl0], 16;
cp.async.commit_group ;                          // group 0
cp.async.ca.shared.global [buf1], [gbl1], 16;
cp.async.commit_group ;                          // group 1
cp.async.wait_group 1 ;   // group 0 complete; group 1 may still be in flight
```

## cp.async.bulk (bulk linear copy)

### Syntax

```ptx
// global -> shared::cta (mbarrier completion)
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes{.L2::cache_hint}
        [dstMem], [srcMem], size, [mbar]{, cache-policy};

// global -> shared::cluster (optional multicast)
cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes
        {.multicast::cluster}{.L2::cache_hint}
        [dstMem], [srcMem], size, [mbar]{, ctaMask}{, cache-policy};

// shared::cta -> shared::cluster (mbarrier completion)
cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
        [dstMem], [srcMem], size, [mbar];

// shared::cta -> global (bulk_group completion)
cp.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}{.cp_mask}
        [dstMem], [srcMem], size{, cache-policy}{, byteMask};
```

### Constraints

- `sm_90`+, PTX 8.0+.
- `size` (u32): must be multiple of 16.
- `dstMem`, `srcMem`: must be 16-byte aligned.
- `.multicast::cluster`: 16-bit `ctaMask`, each bit = destination CTA %ctaid. Optimized on sm_90a/sm_100+.
- `.cp_mask` + 16-bit `byteMask`: per-byte mask within each 16B chunk (sm_100+, PTX 8.6+).
- Complete-tx on mbarrier has `.release` semantics at `.cluster` scope.

### Variants

| Direction | Completion Mechanism |
|---|---|
| global -> shared::cta | `.mbarrier::complete_tx::bytes` |
| global -> shared::cluster | `.mbarrier::complete_tx::bytes` |
| shared::cta -> shared::cluster | `.mbarrier::complete_tx::bytes` |
| shared::cta -> global | `.bulk_group` |

### Example

```ptx
cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes
        [dstMem], [srcMem], size, [mbar];
cp.async.bulk.global.shared::cta.bulk_group [dstMem], [srcMem], size;
```

## cp.async.bulk.tensor (TMA tensor copy)

### Syntax

```ptx
// global -> shared (load)
cp.async.bulk.tensor.DIM.DST.global{.LOAD_MODE}.mbarrier::complete_tx::bytes
        {.multicast::cluster}{.cta_group}{.L2::cache_hint}
        [dstMem], [tensorMap, {coords}], [mbar]{, im2colInfo}{, ctaMask}{, cache-policy};

// shared -> global (store)
cp.async.bulk.tensor.DIM.global.shared::cta{.LOAD_MODE}.bulk_group{.L2::cache_hint}
        [tensorMap, {coords}], [srcMem]{, cache-policy};

.DIM       = { .1d, .2d, .3d, .4d, .5d }
.DST       = { .shared::cta, .shared::cluster }
.LOAD_MODE = { .tile, .tile::gather4, .tile::scatter4,
               .im2col, .im2col::w, .im2col::w::128, .im2col_no_offs }
.cta_group = { .cta_group::1, .cta_group::2 }
```

### Constraints

- `sm_90`+, PTX 8.0+.
- `tensorMap` (u64): generic address of 128-byte opaque tensor-map object (`.param`/`.const`/`.global`). Accessed via tensormap proxy.
- `tensorCoords`: vector of `.s32`, length = `.dim` (except gather4/scatter4: always 5).
- `.tile::gather4`/`.im2col::w`: sm_100+ for shared::cluster, sm_100+ for shared::cta.
- `.tile::scatter4`, `.im2col::w::128`, `.cta_group`: sm_100+, PTX 8.6+.
- `.cta_group::2`: signal mbarrier in peer-CTA of a CTA-pair.
- Loads: mbarrier completion. Stores: bulk async-group completion.

### Example

```ptx
cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes
        [sMem], [tensorMap, {x, y}], [mbar];

cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group
        [tensorMap, {x}], [sMem];
```

## cp.reduce.async.bulk (bulk linear reduction)

### Syntax

```ptx
// shared::cta -> shared::cluster (mbarrier)
cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes
        .REDOP.TYPE  [dstMem], [srcMem], size, [mbar];

// shared::cta -> global (bulk_group)
cp.reduce.async.bulk.global.shared::cta.bulk_group{.L2::cache_hint}
        .REDOP.TYPE  [dstMem], [srcMem], size{, cache-policy};

.REDOP = { .and, .or, .xor, .add, .inc, .dec, .min, .max }
```

### Constraints

- `sm_90`+, PTX 8.0+.
- `size`: multiple of 16, both addresses 16-byte aligned.
- `.add.f32` flushes subnormals. `.add.{f16,bf16}` requires `.noftz` qualifier (preserves subnormals).
- Each reduction has `.relaxed.gpu` memory ordering.

### Variants (redOp x type)

| `.redOp` | shared::cluster types | global types |
|---|---|---|
| `.add` | `.u32`, `.s32`, `.u64` | `.u32`, `.s32`, `.u64`, `.f32`, `.f64`, `.f16`, `.bf16` |
| `.min`, `.max` | `.u32`, `.s32` | `.u32`, `.s32`, `.u64`, `.s64`, `.f16`, `.bf16` |
| `.inc`, `.dec` | `.u32` | `.u32` |
| `.and`, `.or`, `.xor` | `.b32` | `.b32`, `.b64` |

### Example

```ptx
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [dstMem], [srcMem], size;
cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [dstMem], [srcMem], size;
```

## cp.reduce.async.bulk.tensor (tensor reduction)

### Syntax

```ptx
cp.reduce.async.bulk.tensor.DIM.global.shared::cta.REDOP{.LOAD_MODE}.bulk_group
        {.L2::cache_hint}  [tensorMap, {coords}], [srcMem]{, cache-policy};

.REDOP     = { .add, .min, .max, .inc, .dec, .and, .or, .xor }
.LOAD_MODE = { .tile, .im2col_no_offs }
```

### Constraints

- `sm_90`+, PTX 8.0+. Direction: shared::cta -> global only.
- Element type determined by tensor-map. Same redOp/type table as cp.reduce.async.bulk (global column).

### Example

```ptx
cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group
        [tensorMap, {tc0, tc1}], [sMem];
```

## Bulk Async-Group Completion

### Syntax

```ptx
cp.async.bulk.commit_group ;
cp.async.bulk.wait_group N ;          // wait until <= N bulk groups pending
cp.async.bulk.wait_group.read N ;     // wait for source reads only
```

### Constraints

- `sm_90`+, PTX 8.0+. Separate from non-bulk `cp.async.commit_group`.
- `.read` modifier: wait only until source reads complete (source can be reused; destination may not yet be written).

## Tensor-map (Section 5.5.8)

128-byte opaque object in `.const`, `.param`, or `.global` space. Created via CUDA host API (`cuTensorMapEncodeTiled`, etc.). Encodes:

| Property | Description |
|---|---|
| Element type | `.u8`, `.u16`, `.u32`, `.s32`, `.u64`, `.f16`, `.bf16`, `.tf32`, `.f32`, `.f64`, sub-byte types |
| Dimensions | 1D-5D, sizes and strides per dimension |
| Bounding box | Size per dimension (must be multiple of 16 bytes) |
| Swizzle mode | None, 32B, 64B, 96B, 128B (with atomicity sub-modes: 16B, 32B, 32B+8B-flip, 64B) |
| Interleave | None, 8-byte (NC/8DHWC8), 16-byte (NC/16HWC16) |
| OOB fill | Zero fill or OOB-NaN fill |

## Async Proxy

`cp{.reduce}.async.bulk` operations execute in the async proxy. Cross-proxy access requires `fence.proxy.async`. Completion includes an implicit generic-async proxy fence.

## Architecture Summary

| Instruction | Min SM | PTX |
|---|---|---|
| `cp.async` | sm_80 | 7.0 |
| `cp.async.bulk` | sm_90 | 8.0 |
| `cp.async.bulk.tensor` | sm_90 | 8.0 |
| `.multicast::cluster` | sm_90 (optimized sm_90a) | 8.0 |
| `.cp_mask` | sm_100 | 8.6 |
| `.cta_group::2` | sm_100 | 8.6 |
| `.tile::gather4`/`.scatter4` | sm_100 | 8.6 |
| `.im2col::w`/`::w::128` | sm_100 | 8.6 |
