<!-- PTX ISA 9.1 -->

# PTX ISA 9.1 -- Memory Spaces & Fences

---

## 1. State Spaces Overview

| Space | Addressable | Access | Sharing | Notes |
|-------|:-:|--------|---------|-------|
| `.reg` | No | R/W | per-thread | 1/8/16/32/64/128-bit scalar; 16/32/64/128-bit vector; `.pred` is 1-bit |
| `.sreg` | No | RO | per-CTA | Predefined (e.g. `%tid`, `%ctaid`, `%clock`) |
| `.const` | Yes | RO | per-grid | 64 KB static + 10x64 KB driver-allocated banks; initialized to zero by default |
| `.global` | Yes | R/W | context | Initialized to zero by default; visible across grids |
| `.local` | Yes | R/W | per-thread | Stack-allocated (ABI); private per-thread |
| `.param` (kernel) | Yes | RO | per-grid | Accessed via `ld.param::entry`; address via `mov` |
| `.param` (func) | Restricted | R/W | per-thread | `ld.param::func` / `st.param::func`; address taken -> spills to `.local` |
| `.shared` | Yes | R/W | per-cluster | Default sub-qualifier `::cta`; `::cluster` for cross-CTA access |

---

## 2. `.global` State Space (Section 5.1.4)

### Syntax
```ptx
.global .type varname;
.global .type varname = initializer;
.global .align N .type varname[size];
```

### Access Instructions
`ld.global`, `st.global`, `atom.global`, `red.global`

### Constraints
- Addresses are 32-bit or 64-bit.
- Access must be naturally aligned to access size.
- Uninitialized globals default to zero.

---

## 3. `.shared` State Space (Section 5.1.7)

### Syntax
```ptx
.shared .type varname;
.shared .align N .b8 buffer[size];
```

### Sub-qualifiers

| Sub-qualifier | Meaning | Default for |
|---------------|---------|-------------|
| `::cta` | Shared memory of the executing CTA | `ld.shared`, `st.shared`, etc. |
| `::cluster` | Shared memory of any CTA in the cluster | Must be explicit |

### Access Instructions
`ld.shared{::cta, ::cluster}`, `st.shared{::cta, ::cluster}`, `atom.shared{::cta, ::cluster}`

### Constraints
- Variables declared in `.shared` refer to the current CTA's memory.
- Use `mapa` to obtain `.shared::cluster` address of a variable in another CTA.
- `::cluster` requires `sm_90+`.

### Example
```ptx
.shared .align 16 .b8 smem[4096];

ld.shared::cta.u32      r0, [smem];       // local CTA
st.shared::cluster.u32  [remote_addr], r1; // cross-CTA in cluster
```

---

## 4. `.local` State Space (Section 5.1.5)

### Syntax
```ptx
.local .type varname;
.local .align N .b8 stack_buf[size];
```

### Constraints
- Must be declared at function scope (ABI mode).
- Allocated on per-thread stack.
- Accessed via `ld.local`, `st.local`.

---

## 5. `.const` State Space (Section 5.1.3)

### Syntax
```ptx
.const .type varname = value;
.const .align N .b8 data[size] = { ... };
```

### Constraints
- 64 KB for static constants.
- Additional 10x64 KB banks allocated by driver (pointers passed as kernel params).
- Each buffer must fit entirely within one 64 KB region.
- Accessed via `ld.const`.

---

## 6. `.param` State Space (Section 5.1.6)

### Kernel Parameters

```ptx
.entry foo ( .param .b32 N,
             .param .align 8 .b8 buffer[64] )
{
    .reg .u32 %n;
    ld.param.u32 %n, [N];
}
```

### `.ptr` Attribute (for pointer params)

```ptx
.param .type .ptr .space .align N varname
.space = { .const, .global, .local, .shared }
```

```ptx
.entry bar ( .param .u32 param1,
             .param .u32 .ptr.global.align 16 param2,
             .param .u32 .ptr.const.align 8  param3,
             .param .u32 .ptr.align 16       param4 )  // generic address
```

Default alignment when `.align` omitted: 4 bytes. PTX ISA 2.2+.

### Device Function Parameters

```ptx
.func foo ( .reg .b32 N, .param .align 8 .b8 buffer[12] )
{
    ld.param.f64 %d, [buffer];
    ld.param.s32 %y, [buffer+8];
}
```

- Input params: `ld.param::func`. Return params: `st.param::func`.
- Taking address of a function input param via `mov` forces it to `.local`.

---

## 7. Generic Addressing (Section 6.4.1.1)

When a memory instruction omits the state space qualifier, it uses generic addressing.

### Address Windows

| Window | Mapping |
|--------|---------|
| `.const` | Falls within const window -> const access |
| `.local` | Falls within local window -> local access |
| `.shared` | Falls within shared window -> shared access |
| `.param` (kernel) | Contained within `.global` window |
| Everything else | `.global` |

### `cvta` -- Convert Address

```ptx
cvta{.space}.size  dst, src;       // state-space -> generic
cvta.to{.space}.size  dst, src;    // generic -> state-space

.space = { .const, .global, .local, .shared{::cta, ::cluster}, .param{::entry} }
.size  = { .u32, .u64 }
```

### `isspacep` -- Test Address Space

```ptx
isspacep.space  p, a;
.space = { .const, .global, .local, .shared{::cta, ::cluster}, .param::entry }
```

Sets predicate `p` to `True` if generic address `a` falls within the specified space window.

---

## 8. Memory Fences: `fence` / `membar` (Section 9.7.13.4)

### 8.1 Thread Fence (`fence`)

```ptx
fence{.sem}.scope;

.sem   = { .sc, .acq_rel, .acquire, .release }   // default: .acq_rel
.scope = { .cta, .cluster, .gpu, .sys }
```

| Variant | Semantics | Use case |
|---------|-----------|----------|
| `fence.acq_rel.scope` | Lightweight acquire-release fence | Most synchronization patterns |
| `fence.sc.scope` | Sequential consistency fence | Restore SC ordering (slower) |
| `fence.acquire.scope` | One-directional acquire | Pair with prior release |
| `fence.release.scope` | One-directional release | Pair with subsequent acquire |

### Constraints
- `fence` requires `sm_70+`.
- `.acquire` / `.release` qualifiers require `sm_90+`.
- `.cluster` scope requires `sm_90+`.

### Example
```ptx
fence.acq_rel.gpu;
fence.sc.sys;
fence.acquire.cluster;
```

### 8.2 Restricted Fences

```ptx
// Operation-restricted fence (mbarrier init ordering)
fence.mbarrier_init.release.cluster;

// Sync-restricted fences (shared memory scope)
fence.acquire.sync_restrict::shared::cluster.cluster;
fence.release.sync_restrict::shared::cta.cluster;
```

| Qualifier | `.sem` must be | `.scope` must be | Effect restricted to |
|-----------|---------------|-----------------|---------------------|
| `.mbarrier_init` | `.release` | `.cluster` | Prior `mbarrier.init` ops on `.shared::cta` |
| `.sync_restrict::shared::cta` | `.release` | `.cluster` | Ops on `.shared::cta` objects |
| `.sync_restrict::shared::cluster` | `.acquire` | `.cluster` | Ops on `.shared::cluster` objects |

Requires `sm_90+`.

### 8.3 Legacy `membar`

```ptx
membar.level;
.level = { .cta, .gl, .sys }
```

| `membar` level | Equivalent `fence` scope |
|---------------|-------------------------|
| `.cta` | `fence.sc.cta` |
| `.gl` | `fence.sc.gpu` |
| `.sys` | `fence.sc.sys` |

On `sm_70+`, `membar` is a synonym for `fence.sc`. `membar.{cta,gl}` supported on all targets. `membar.sys` requires `sm_20+`.

---

## 9. Proxy Fences (Section 9.7.13.4)

Proxy fences order memory accesses across different memory proxies (generic, async, texture, virtual aliases).

### 9.1 Bi-directional Proxy Fence

```ptx
fence.proxy.proxykind;
membar.proxy.proxykind;      // synonym on sm_70+

.proxykind = { .alias, .async, .async.global, .async.shared::{cta, cluster} }
```

| `.proxykind` | Orders between |
|-------------|---------------|
| `.alias` | Virtually aliased addresses to the same physical location |
| `.async` | Async proxy and generic proxy (all state spaces) |
| `.async.global` | Async proxy and generic proxy (`.global` only) |
| `.async.shared::cta` | Async proxy and generic proxy (`.shared::cta` only) |
| `.async.shared::cluster` | Async proxy and generic proxy (`.shared::cluster` only) |

### 9.2 Uni-directional Proxy Fence (tensormap)

```ptx
fence.proxy.tensormap::generic.release.scope;
fence.proxy.tensormap::generic.acquire.scope [addr], 128;

.scope = { .cta, .cluster, .gpu, .sys }
```

Used after modifying a tensormap (`tensormap.replace`) and before issuing tensor copies that use the updated map. The acquire form takes an address operand and size (must be 128). Address must be in `.global` via generic addressing.

### Constraints
- `fence.proxy` requires `sm_70+`.
- `membar.proxy` requires `sm_60+`.
- `.async` proxy variants require `sm_90+`.
- `.tensormap::generic` requires `sm_90+`.

### Example: tensormap update pattern
```ptx
tensormap.replace.tile.global_address.global.b1024.b64 [gbl], new_addr;
fence.proxy.tensormap::generic.release.gpu;
cvta.global.u64 tmap, gbl;
fence.proxy.tensormap::generic.acquire.gpu [tmap], 128;
cp.async.bulk.tensor.1d.shared::cluster.global.tile [addr0], [tmap, {tc0}], [mbar0];
```

---

## 10. Scopes (Section 8.5)

| Scope | Thread set |
|-------|-----------|
| `.cta` | All threads in the same CTA |
| `.cluster` | All threads in the same cluster |
| `.gpu` | All threads on the same device (including other grids) |
| `.sys` | All threads across all devices + host |

Warp is NOT a scope in the memory consistency model.

---

## 11. Operation Ordering Qualifiers (Section 8.4)

| Qualifier | Meaning |
|-----------|---------|
| `.relaxed` | Strong, no ordering beyond data dependency |
| `.acquire` | Subsequent ops cannot move before this |
| `.release` | Prior ops cannot move after this |
| `.acq_rel` | Combined acquire + release |
| `.volatile` | Equivalent to `.relaxed.sys` with extra constraints (deprecated for sync) |
| `.mmio` | For memory-mapped I/O; preserves operation count; not cached |
| `.weak` | Default for plain `ld`/`st`; no ordering guarantees |
