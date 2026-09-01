<!-- PTX ISA 9.1 -->
# Cache Operators, Eviction Policies & L2 Cache Hints

## Cache Operators on `ld` / `st` (9.7.9.1)

PTX ISA 2.0+. `sm_20`+. Performance hints only -- no effect on memory consistency.

### Load Cache Operators

| Operator | Name | Behavior |
|----------|------|----------|
| `.ca` | Cache at all levels (default) | Allocates in L1 and L2 with normal eviction. L1 not coherent across SMs for global data. |
| `.cg` | Cache at global level | Bypasses L1, caches only in L2. |
| `.cs` | Cache streaming | Evict-first policy in L1 and L2. On `.local` addresses behaves as `.lu`. |
| `.lu` | Last use | Avoids write-back of soon-discarded lines. On `.global` behaves as `.cs`. |
| `.cv` | Don't cache (volatile) | Invalidates matching L2 line, re-fetches on every load. |

### Store Cache Operators

| Operator | Name | Behavior |
|----------|------|----------|
| `.wb` | Write-back (default) | Writes back coherent levels with normal eviction. |
| `.cg` | Cache at global level | Bypasses L1, caches only in L2. |
| `.cs` | Cache streaming | Evict-first allocation to limit pollution. |
| `.wt` | Write-through | Writes through L2 to system memory. |

### Constraints

- `.cop` qualifiers are mutually exclusive with `.relaxed`/`.acquire`/`.release`/`.volatile`.
- Only valid on `.weak` (default) memory ordering.

---

## Cache Eviction Priority Hints (9.7.9.2)

PTX ISA 7.4+. `.global` state space only (or generic pointing to `.global`).

| Priority | Meaning | Applicable Levels |
|----------|---------|-------------------|
| `evict_normal` | Default priority | L1, L2 |
| `evict_first` | Evicted first -- streaming data | L1, L2 |
| `evict_last` | Evicted last -- persistent data | L1, L2 |
| `evict_unchanged` | Do not change existing priority | L1 only |
| `no_allocate` | Do not allocate to cache | L1 only |

### Syntax on `ld` / `st`

```ptx
.level1::eviction_priority = { .L1::evict_normal, .L1::evict_unchanged,
                               .L1::evict_first, .L1::evict_last, .L1::no_allocate };
.level2::eviction_priority = { .L2::evict_normal, .L2::evict_first, .L2::evict_last };
```

### Architecture Requirements

| Qualifier | PTX ISA | Target |
|-----------|---------|--------|
| `.L1::evict_*` / `.L1::no_allocate` | 7.4 | `sm_70`+ |
| `.L2::evict_*` on `ld`/`st` | 8.8 | `sm_100`+ |
| `.L2::cache_hint` | 7.4 | `sm_80`+ |

### Example

```ptx
ld.global.L1::evict_last.u32                    d, [p];
st.global.L1::no_allocate.f32                   [p], a;
ld.global.L2::evict_last.L1::evict_last.v4.u64  {r0, r1, r2, r3}, [addr];
```

---

## L2 Prefetch Size Hints

```ptx
.level::prefetch_size = { .L2::64B, .L2::128B, .L2::256B };
```

| Qualifier | PTX ISA | Target |
|-----------|---------|--------|
| `.L2::64B` / `.L2::128B` | 7.4 | `sm_75`+ |
| `.L2::256B` | 7.4 | `sm_80`+ |

Only valid for `.global` state space. Performance hint only.

### Example

```ptx
ld.global.L2::64B.b32   %r0, [gbl];
ld.global.L2::128B.f64  %r1, [gbl];
ld.global.L2::256B.f64  %r2, [gbl];
```

---

## `createpolicy` (9.7.9.18)

Creates a 64-bit opaque cache eviction policy for use with `.L2::cache_hint` on `ld`/`st`.

PTX ISA 7.4+. `sm_80`+.

### Syntax

```ptx
// Range-based
createpolicy.range{.global}.level::primary{.level::secondary}.b64
    cache-policy, [a], primary-size, total-size;

// Fraction-based
createpolicy.fractional.level::primary{.level::secondary}.b64
    cache-policy{, fraction};

// Convert CUDA access property
createpolicy.cvt.L2.b64  cache-policy, access-property;

.level::primary   = { .L2::evict_last, .L2::evict_normal,
                      .L2::evict_first, .L2::evict_unchanged };
.level::secondary = { .L2::evict_first, .L2::evict_unchanged };
```

### Range-Based Policy

Defines three address ranges relative to base `a`:

| Range | Span | Applied Priority |
|-------|------|-----------------|
| Primary | `[a .. a + primary_size - 1]` | `primary` |
| Trailing secondary | `[a + primary_size .. a + total_size - 1]` | `secondary` |
| Preceding secondary | `[a - (total_size - primary_size) .. a - 1]` | `secondary` |
| Outside | -- | Unspecified |

- `primary_size` <= `total_size`. Max `total_size` = 4 GB.
- Default `secondary` = `.L2::evict_unchanged`.

### Fraction-Based Policy

Each access has probability `fraction` of receiving `primary` priority; remainder gets `secondary`.
Valid range: `(0.0, 1.0]`. Default `fraction` = `1.0`. Default `secondary` = `.L2::evict_unchanged`.

### Example

```ptx
createpolicy.fractional.L2::evict_last.b64                      pol, 1.0;
createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64  pol, 0.5;
createpolicy.range.L2::evict_last.L2::evict_first.b64           pol, [ptr], 0x100000, 0x200000;
createpolicy.cvt.L2.b64                                         pol, access-prop;

// Usage with ld/st:
ld.global.L2::cache_hint.b64  x, [p], pol;
st.global.L2::cache_hint.b32  [a], b, pol;
```

---

## `prefetch` / `prefetchu` (9.7.9.15)

### Syntax

```ptx
prefetch{.space}.level                    [a];
prefetch.global.level::eviction_priority  [a];
prefetchu.L1                              [a];
prefetch{.tensormap_space}.tensormap       [a];

.space                    = { .global, .local };
.level                    = { .L1, .L2 };
.level::eviction_priority = { .L2::evict_last, .L2::evict_normal };
.tensormap_space          = { .const, .param };
```

### Constraints

- No state space: generic addressing.
- Prefetch to `.shared`: no-op.
- `prefetchu.L1` requires generic address; no-op for `.const`, `.local`, `.shared`.
- `.tensormap` prefetches for subsequent `cp.async.bulk.tensor`.

### Architecture Requirements

| Feature | PTX ISA | Target |
|---------|---------|--------|
| `prefetch` / `prefetchu` | 2.0 | `sm_20`+ |
| `.level::eviction_priority` | 7.4 | `sm_80`+ |
| `.tensormap` | 8.0 | `sm_90`+ |

### Example

```ptx
prefetch.global.L1              [ptr];
prefetch.global.L2::evict_last  [ptr];
prefetchu.L1                    [addr];
prefetch.const.tensormap        [ptr];
```

---

## `applypriority` (9.7.9.16)

Changes eviction priority of an existing L2 cache line.

PTX ISA 7.4+. `sm_80`+.

### Syntax

```ptx
applypriority{.global}.level::eviction_priority  [a], size;

.level::eviction_priority = { .L2::evict_normal };
```

### Constraints

- `size` must be `128`. Address `a` must be 128-byte aligned.
- `.global` only (or generic to `.global`).
- Only `.L2::evict_normal` supported (demote from `evict_last` back to normal).

### Example

```ptx
applypriority.global.L2::evict_normal [ptr], 128;
```

---

## `discard` (9.7.9.17)

Discards L2 cache lines without writing back to memory.

PTX ISA 7.4+. `sm_80`+.

### Syntax

```ptx
discard{.global}.level  [a], size;

.level = { .L2 };
```

### Constraints

- Semantically a weak write of an **unstable indeterminate value** -- subsequent reads may return different values.
- `size` must be `128`. Address `a` must be 128-byte aligned.
- `.global` only (or generic to `.global`).

### Example

```ptx
discard.global.L2 [ptr], 128;
ld.weak.u32 r0, [ptr];
ld.weak.u32 r1, [ptr];
// r0 and r1 may differ!
```

---

## Architecture Requirements Summary

| Feature | PTX ISA | Min SM |
|---------|---------|--------|
| Cache operators (`.ca`/`.cg`/`.cs`/`.lu`/`.cv`/`.wb`/`.wt`) | 2.0 | `sm_20` |
| `prefetch` / `prefetchu` | 2.0 | `sm_20` |
| `.L1::evict_*` / `.L1::no_allocate` | 7.4 | `sm_70` |
| `.L2::64B` / `.L2::128B` prefetch size | 7.4 | `sm_75` |
| `.L2::256B` prefetch size | 7.4 | `sm_80` |
| `.L2::cache_hint` | 7.4 | `sm_80` |
| `createpolicy` | 7.4 | `sm_80` |
| `applypriority` | 7.4 | `sm_80` |
| `discard` | 7.4 | `sm_80` |
| `prefetch` with eviction priority | 7.4 | `sm_80` |
| `prefetch.tensormap` | 8.0 | `sm_90` |
| `.L2::evict_*` on `ld`/`st` | 8.8 | `sm_100` |

---

## Quick Reference: Typical Usage Patterns

```ptx
// --- Streaming load (evict early) ---
ld.global.cs.f32                          val, [ptr];
ld.global.L1::evict_first.f32             val, [ptr];

// --- Persistent data (keep in cache) ---
ld.global.L1::evict_last.f32              val, [ptr];

// --- L2-only caching (bypass L1) ---
ld.global.cg.f32                          val, [ptr];
st.global.cg.f32                          [ptr], val;

// --- L2 cache hint with policy ---
createpolicy.fractional.L2::evict_last.b64 pol, 1.0;
ld.global.L2::cache_hint.f32              val, [ptr], pol;
st.global.L2::cache_hint.f32              [ptr], val, pol;

// --- Prefetch to L2 with evict_last ---
prefetch.global.L2::evict_last            [ptr];

// --- Demote from evict_last back to normal ---
applypriority.global.L2::evict_normal     [ptr], 128;

// --- Discard dirty L2 line (avoid writeback) ---
discard.global.L2                         [ptr], 128;

// --- Write-through store ---
st.global.wt.f32                          [ptr], val;
```
