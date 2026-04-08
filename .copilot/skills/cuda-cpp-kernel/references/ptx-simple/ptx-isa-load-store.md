<!-- PTX ISA 9.1 -->
# PTX Load, Store, Atomic, Reduction, and Data Movement Instructions

## ld

### Syntax

```ptx
ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type d, [a]{.unified}{, cache-policy};
ld{.weak}{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.L2::prefetch_size}{.vec}.type d, [a]{, cache-policy};
ld.volatile{.ss}{.level::prefetch_size}{.vec}.type d, [a];
ld.relaxed.scope{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.L2::prefetch_size}{.vec}.type d, [a]{, cache-policy};
ld.acquire.scope{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.L2::prefetch_size}{.vec}.type d, [a]{, cache-policy};
ld.mmio.relaxed.sys{.global}.type d, [a];
```

### Variants

| Qualifier | Values |
|-----------|--------|
| `.ss` | `.const`, `.global`, `.local`, `.param{::entry,::func}`, `.shared{::cta,::cluster}` |
| `.cop` | `.ca`, `.cg`, `.cs`, `.lu`, `.cv` |
| `.scope` | `.cta`, `.cluster`, `.gpu`, `.sys` |
| `.vec` | `.v2`, `.v4`, `.v8` |
| `.type` | `.b8`, `.b16`, `.b32`, `.b64`, `.b128`, `.u8`-`.u64`, `.s8`-`.s64`, `.f32`, `.f64` |

### Constraints

- `.weak` is default when no `.volatile`/`.relaxed`/`.acquire` specified
- `.relaxed`/`.acquire`: only `.global`/`.shared`; `.cop` NOT allowed
- `.volatile`: `.global`/`.shared`/`.local`; `.cop` NOT allowed
- `.mmio`: `.global` only; requires `.relaxed` + `.sys`
- `.v8` only for `.b32`/`.u32`/`.s32`/`.f32` in `.global`
- `.v4` with 64-bit types (`.b64`/`.u64`/`.s64`/`.f64`) only in `.global`
- `.b128`: scalar 128-bit load, `sm_70`+
- `.v8.b32`/`.v4.b64` 256-bit loads: L2 eviction priority requires `sm_100`+
- Sink symbol `_` usable in `.v8`/`.v4` vector expressions
- Alignment: naturally aligned to access size (vec_count x element_size)
- Cache hints: see ptx-isa-cache-hints.md

### Example

```ptx
ld.global.f32 d, [a];
ld.shared.v4.b32 Q, [p];
ld.global.relaxed.gpu.u32 %r0, [gbl];
ld.shared.acquire.gpu.u32 %r1, [sh];
ld.global.L1::evict_last.u32 d, [p];
ld.global.L2::128B.b32 %r0, [gbl];
ld.global.L2::evict_last.v8.f32 {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7}, [addr];
ld.global.b128 %r0, [gbl];
ld.global.mmio.relaxed.sys.u32 %r3, [gbl];
```

## st

### Syntax

```ptx
st{.weak}{.ss}{.cop}{.L2::cache_hint}{.vec}.type [a], b{, cache-policy};
st{.weak}{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.vec}.type [a], b{, cache-policy};
st.volatile{.ss}{.vec}.type [a], b;
st.relaxed.scope{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.vec}.type [a], b{, cache-policy};
st.release.scope{.ss}{.L1::evict_*}{.L2::evict_*}{.L2::cache_hint}{.vec}.type [a], b{, cache-policy};
st.mmio.relaxed.sys{.global}.type [a], b;
```

### Variants

| Qualifier | Values |
|-----------|--------|
| `.ss` | `.global`, `.local`, `.param::func`, `.shared{::cta,::cluster}` |
| `.cop` | `.wb`, `.cg`, `.cs`, `.wt` |
| `.scope` | `.cta`, `.cluster`, `.gpu`, `.sys` |
| `.vec` | `.v2`, `.v4`, `.v8` |
| `.type` | `.b8`-`.b128`, `.u8`-`.u64`, `.s8`-`.s64`, `.f32`, `.f64` |

### Constraints

Same rules as `ld` for `.weak`/`.volatile`/`.relaxed`/`.release` mutual exclusivity, vec/type restrictions, and alignment. Stores to `.const` are illegal.

### Example

```ptx
st.global.f32 [a], b;
st.global.v4.s32 [p], Q;
st.global.relaxed.sys.u32 [gbl], %r0;
st.shared.release.cta.u32 [sh], %r1;
st.global.L1::no_allocate.f32 [p], a;
st.global.b128 [a], b;
st.global.L2::evict_last.v8.f32 [addr], {%r0, _, %r2, %r3, %r4, %r5, %r6, %r7};
```

## atom

### Syntax

```ptx
// Scalar
atom{.sem}{.scope}{.space}.op{.L2::cache_hint}.type d, [a], b{, cache-policy};
atom{.sem}{.scope}{.space}.cas.type d, [a], b, c;   // compare-and-swap (3 operands)
atom{.sem}{.scope}{.space}.cas.b16 d, [a], b, c;
atom{.sem}{.scope}{.space}.cas.b128 d, [a], b, c;
atom{.sem}{.scope}{.space}.exch{.L2::cache_hint}.b128 d, [a], b{, cache-policy};

// Half-precision (requires .noftz)
atom{.sem}{.scope}{.space}.add.noftz{.L2::cache_hint}.{f16,f16x2,bf16,bf16x2} d, [a], b;

// Vector (.global only, sm_90+)
atom{.sem}{.scope}{.global}.add{.L2::cache_hint}.{v2,v4}.f32 d, [a], b;
atom{.sem}{.scope}{.global}.op.noftz{.L2::cache_hint}.{v2,v4,v8}.{f16,bf16} d, [a], b;
atom{.sem}{.scope}{.global}.op.noftz{.L2::cache_hint}.{v2,v4}.{f16x2,bf16x2} d, [a], b;

.space = { .global, .shared{::cta,::cluster} }
.sem   = { .relaxed, .acquire, .release, .acq_rel }  // default: .relaxed
.scope = { .cta, .cluster, .gpu, .sys }               // default: .gpu
```

### Variants

| Operation | Valid Scalar Types |
|-----------|-------------------|
| `.and`, `.or`, `.xor` | `.b32`, `.b64` |
| `.cas` | `.b16`, `.b32`, `.b64`, `.b128` |
| `.exch` | `.b32`, `.b64`, `.b128` |
| `.add` | `.u32`, `.u64`, `.s32`, `.s64`, `.f32`, `.f64` |
| `.inc`, `.dec` | `.u32` |
| `.min`, `.max` | `.u32`, `.u64`, `.s32`, `.s64` |
| `.add.noftz` | `.f16`, `.f16x2`, `.bf16`, `.bf16x2` |

Vector ops (`sm_90`+, `.global` only):

| Vec | `.f16`/`.bf16` | `.f16x2`/`.bf16x2` | `.f32` |
|-----|----------------|---------------------|--------|
| `.v2` | add, min, max | add, min, max | add |
| `.v4` | add, min, max | add, min, max | add |
| `.v8` | add, min, max | -- | -- |

### Constraints

- Atomicity for packed/vector types is per-element, not across the entire access
- `.b128` cas/exch requires `sm_90`+
- Use `_` as destination for fire-and-forget reductions: `atom.global.add.s32 _, [a], 1;`
- Two `atom`/`red` ops are atomic w.r.t. each other only if each specifies a scope that includes the other
- `atom.add.f32` on global flushes subnormals; on shared it does not
- `.noftz` required for `.f16`/`.f16x2`/`.bf16`/`.bf16x2` adds (preserves subnormals)

### Example

```ptx
atom.global.add.s32 d, [a], 1;
atom.global.cas.b32 d, [p], my_val, my_new_val;
atom.global.acquire.sys.inc.u32 ans, [gbl], %r0;
atom.add.noftz.f16x2 d, [a], b;
atom.global.v4.f32.add {%f0,%f1,%f2,%f3}, [gbl], {%f0,%f1,%f2,%f3};
atom.global.v8.f16.max.noftz {%h0,...,%h7}, [gbl], {%h0,...,%h7};
```

## red

### Syntax

```ptx
// Scalar
red{.sem}{.scope}{.space}.op{.L2::cache_hint}.type [a], b{, cache-policy};
red{.sem}{.scope}{.space}.add.noftz{.L2::cache_hint}.{f16,f16x2,bf16,bf16x2} [a], b;

// Vector (.global only, sm_90+)
red{.sem}{.scope}{.global}.add{.L2::cache_hint}.{v2,v4}.f32 [a], b;
red{.sem}{.scope}{.global}.op.noftz{.L2::cache_hint}.{v2,v4,v8}.{f16,bf16} [a], b;
red{.sem}{.scope}{.global}.op.noftz{.L2::cache_hint}.{v2,v4}.{f16x2,bf16x2} [a], b;

.space = { .global, .shared{::cta,::cluster} }
.sem   = { .relaxed, .release }                       // NO .acquire/.acq_rel (unlike atom)
.scope = { .cta, .cluster, .gpu, .sys }               // default: .gpu
```

### Variants

Same op/type table as `atom` except: no `.cas`, no `.exch`, no `.b128`. Same vector support table.

### Constraints

Same atomicity/scope rules as `atom`. No return value (unlike `atom`).

### Example

```ptx
red.global.add.s32 [a], 1;
red.global.sys.add.u32 [a], 1;
red.add.noftz.f16x2 [a], b;
red.global.v4.f32.add [gbl], {%f0,%f1,%f2,%f3};
red.global.v8.bf16.min.noftz [gbl], {%h0,%h1,%h2,%h3,%h4,%h5,%h6,%h7};
```

## mov

### Syntax

```ptx
// Register/immediate/address move
mov.type d, a;
mov.type d, avar;          // non-generic address of variable
mov.type d, avar+imm;
mov.u32  d, fname;         // device function address
mov.u64  d, kernel;        // entry function address

.type = { .pred, .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64, .f32, .f64 }

// Pack/unpack (vector <-> scalar)
mov.btype d, a;
.btype = { .b16, .b32, .b64, .b128 }
```

### Constraints

- For address of variable: places non-generic address (use `cvta` to convert to generic)
- `.b128` pack/unpack requires `sm_70`+
- Sink `_` allowed in unpack destination

### Example

```ptx
mov.f32 d, a;
mov.u32 ptr, A;              // address of A
mov.b32 %r1, {a, b};         // pack two .u16 -> .b32
mov.b64 {lo, hi}, %x;        // unpack .b64 -> two .u32
mov.b128 {%b1, %b2}, %y;     // unpack .b128 -> two .b64
```

## cvt

### Syntax

```ptx
cvt{.irnd}{.ftz}{.sat}.dtype.atype d, a;      // integer rounding
cvt{.frnd}{.ftz}{.sat}.dtype.atype d, a;      // float rounding

// Packed conversions (selected common forms)
cvt.frnd{.relu}{.satfinite}.f16x2.f32 d, a, b;
cvt.frnd{.relu}{.satfinite}.bf16x2.f32 d, a, b;
cvt.rn.satfinite{.relu}.f8x2type.f32 d, a, b;
cvt.rn{.relu}.f16x2.f8x2type d, a;

.irnd = { .rni, .rzi, .rmi, .rpi }
.frnd = { .rn, .rz, .rm, .rp }
.dtype/.atype = { .u8-.u64, .s8-.s64, .bf16, .f16, .f32, .f64 }
.f8x2type = { .e4m3x2, .e5m2x2 }
```

### Constraints

- Rounding mandatory for: float-to-float narrowing, float-to-int, int-to-float, all packed conversions
- `.satfinite` mandatory for FP8/FP6/FP4 destination types
- `.ftz`: only when source or dest is `.f32`; flushes subnormals to sign-preserving zero
- `.sat`: clamps integers to MININT..MAXINT; clamps floats to [0.0, 1.0]
- `.relu`: clamps negative to 0; applies to `.f16`/`.bf16`/`.tf32` and packed dest types

### Example

```ptx
cvt.f32.s32 f, i;
cvt.rni.f32.f32 x, y;                              // round to nearest int
cvt.rn.relu.f16.f32 b, f;
cvt.rz.f16x2.f32 b1, f, f1;                        // pack two f32 -> f16x2
cvt.rn.satfinite.e4m3x2.f32 d, a, b;               // two f32 -> e4m3x2
cvt.rn.f16x2.e4m3x2 d, a;                          // unpack e4m3x2 -> f16x2
```

## cvta

### Syntax

```ptx
cvta.space.size p, a;           // state-space addr -> generic
cvta.space.size p, var;         // variable -> generic
cvta.to.space.size p, a;        // generic -> state-space addr

.space = { .const, .global, .local, .shared{::cta,::cluster}, .param{::entry} }
.size  = { .u32, .u64 }
```

### Constraints

- `sm_20`+; `.param` requires `sm_70`+; `::cluster` requires `sm_90`+
- Use `isspacep` to guard against invalid generic-to-specific conversions

### Example

```ptx
cvta.global.u64 gptr, myVar;
cvta.shared::cta.u32 p, As+4;
cvta.to.global.u32 p, gptr;
```

## isspacep

### Syntax

```ptx
isspacep.space p, a;

.space = { .const, .global, .local, .shared{::cta,::cluster}, .param{::entry} }
```

### Constraints

- `p` is `.pred`; `a` is `.u32` or `.u64` generic address
- `isspacep.global` returns 1 for `.param` addresses (`.param` window is within `.global`)
- `::cta` only returns 1 for executing CTA's shared memory; `::cluster` for any CTA in cluster

### Example

```ptx
isspacep.global isglbl, gptr;
isspacep.shared::cluster isclust, sptr;
```

## prefetch

### Syntax

```ptx
prefetch{.space}.level [a];
prefetch.global.level::eviction_priority [a];
prefetchu.L1 [a];
prefetch{.tensormap_space}.tensormap [a];

.space = { .global, .local }
.level = { .L1, .L2 }
.level::eviction_priority = { .L2::evict_last, .L2::evict_normal }
.tensormap_space = { .const, .param }
```

### Constraints

- `sm_20`+; eviction priority requires `sm_80`+; `.tensormap` requires `sm_90`+
- Prefetch to shared memory is a no-op
- `prefetchu.L1` requires generic address; no-op if address maps to const/local/shared

### Example

```ptx
prefetch.global.L1 [ptr];
prefetch.global.L2::evict_last [ptr];
prefetchu.L1 [addr];
prefetch.const.tensormap [tmap_ptr];
```
