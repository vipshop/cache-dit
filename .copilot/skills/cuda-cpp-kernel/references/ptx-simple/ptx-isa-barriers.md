<!-- PTX ISA 9.1 -->

## bar.sync / bar.arrive / bar.red

### Syntax

```ptx
bar{.cta}.sync   a{, b};
bar{.cta}.arrive a, b;
bar{.cta}.red.popc.u32 d, a{, b}, {!}c;
bar{.cta}.red.op.pred  p, a{, b}, {!}c;

barrier{.cta}.sync{.aligned}           a{, b};
barrier{.cta}.arrive{.aligned}         a, b;
barrier{.cta}.red.popc{.aligned}.u32   d, a{, b}, {!}c;
barrier{.cta}.red.op{.aligned}.pred    p, a{, b}, {!}c;

.op = { .and, .or };
```

### Variants

| Form | Behavior |
|------|----------|
| `.sync` | Arrive + wait for all participants. Full memory ordering. |
| `.arrive` | Arrive only, no wait. Requires thread count `b`. |
| `.red.popc` | Arrive + wait + population count of predicate `c`. Result in `.u32` `d`. |
| `.red.and`/`.or` | Arrive + wait + predicate reduction. Result in `.pred` `p`. |

`bar.sync` is equivalent to `barrier.cta.sync.aligned`. 16 barriers per CTA (0..15). Operand `b` must be a multiple of warp size.

### Constraints

- `bar` forms: all targets (immediate barrier), `sm_20+` (register operands, `.arrive`, `.red`)
- `barrier` forms: `sm_30+`
- Do not mix `.red` with `.sync`/`.arrive` on the same active barrier

### Example

```ptx
st.shared [r0], r1;
bar.cta.sync 1;
ld.shared r2, [r3];

bar.cta.red.and.pred r3, 1, p;
```

## bar.warp.sync

### Syntax

```ptx
bar.warp.sync membermask;
```

### Constraints

- `membermask`: `.b32`, bit per lane. Executing thread must be in mask.
- Provides memory ordering among participating threads.
- `sm_30+`

### Example

```ptx
st.shared.u32 [r0], r1;
bar.warp.sync 0xffffffff;
ld.shared.u32 r2, [r3];
```

## barrier.cluster

### Syntax

```ptx
barrier.cluster.arrive{.sem}{.aligned};
barrier.cluster.wait{.acquire}{.aligned};

.sem = { .release, .relaxed }
```

### Variants

| Instruction | Default sem | Behavior |
|-------------|-------------|----------|
| `.arrive` | `.release` | Mark arrival, no wait. |
| `.wait` | `.acquire` | Block until all cluster threads arrived. |

Auto-reinitializes on completion. Each thread arrives exactly once per phase. `.relaxed` on arrive removes memory ordering (use explicit `fence` if needed).

### Constraints

- `sm_90+`
- `.acquire`, `.relaxed`, `.release` qualifiers: PTX ISA 8.0+

### Example

```ptx
ld.shared::cluster.u32 r0, [addr];
barrier.cluster.arrive.aligned;
barrier.cluster.wait.aligned;
st.shared::cluster.u32 [addr], r1;
```

## mbarrier.init

### Syntax

```ptx
mbarrier.init{.shared{::cta}}.b64 [addr], count;
```

### Constraints

- `count` range: [1, 2^20 - 1]. Sets phase=0, pending=count, expected=count, tx-count=0.
- Object: `.b64`, 8-byte aligned, in `.shared` memory.
- Must call `mbarrier.inval` before re-init or repurposing memory.
- `sm_80+`

### Example

```ptx
mbarrier.init.shared::cta.b64 [shMem], 12;
```

## mbarrier.arrive

### Syntax

```ptx
mbarrier.arrive{.sem.scope}{.shared{::cta}}.b64           state, [addr]{, count};
mbarrier.arrive{.sem.scope}{.shared::cluster}.b64              _, [addr]{, count};
mbarrier.arrive.expect_tx{.sem.scope}{.shared{::cta}}.b64 state, [addr], txCount;
mbarrier.arrive.expect_tx{.sem.scope}{.shared::cluster}.b64    _, [addr], txCount;
mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64  state, [addr], count;

.sem   = { .release, .relaxed }   // default: .release
.scope = { .cta, .cluster }      // default: .cta
```

### Variants

| Variant | Behavior |
|---------|----------|
| basic | Decrements pending count by `count` (default 1). Returns opaque `state`. |
| `.expect_tx` | Fused: tx-count += txCount, then arrive with count=1. |
| `.noComplete` | Must not cause phase completion (UB otherwise). Required on `sm_8x` with explicit count. |
| `.shared::cluster` | Remote arrive. Must use sink `_` as destination. |

### Constraints

- `sm_80+`. `.expect_tx`, `.cluster`, count without `.noComplete`: `sm_90+`. `.relaxed`: `sm_90+`.

### Example

```ptx
mbarrier.arrive.shared.b64 %r0, [shMem];
mbarrier.arrive.release.cluster.b64 _, [remoteAddr], cnt;
mbarrier.arrive.expect_tx.release.cluster.b64 _, [remoteAddr], tx_count;
```

## mbarrier.test_wait / mbarrier.try_wait

### Syntax

```ptx
mbarrier.test_wait{.sem.scope}{.shared{::cta}}.b64        waitComplete, [addr], state;
mbarrier.test_wait.parity{.sem.scope}{.shared{::cta}}.b64 waitComplete, [addr], phaseParity;

mbarrier.try_wait{.sem.scope}{.shared{::cta}}.b64         waitComplete, [addr], state
                                                            {, suspendTimeHint};
mbarrier.try_wait.parity{.sem.scope}{.shared{::cta}}.b64  waitComplete, [addr], phaseParity
                                                            {, suspendTimeHint};

.sem   = { .acquire, .relaxed }   // default: .acquire
.scope = { .cta, .cluster }      // default: .cta
```

### Variants

| Instruction | Blocking | Notes |
|-------------|----------|-------|
| `test_wait` | No | Returns `True` if phase complete. |
| `try_wait` | Potentially | Thread may suspend. `suspendTimeHint` in nanoseconds. |
| `.parity` | -- | Uses phase parity (0=even, 1=odd) instead of opaque `state`. |

On `True` return with `.acquire`: all prior `.release` arrive memory ops by participants are visible.

### Constraints

- `test_wait`: `sm_80+`. `try_wait`: `sm_90+`. `.cluster` scope, `.relaxed`: `sm_90+`.
- Only valid for current incomplete phase (`False`) or immediately preceding phase (`True`).

### Example

```ptx
// Spin loop with test_wait
waitLoop:
  mbarrier.test_wait.shared.b64 complete, [shMem], state;
  @!complete nanosleep.u32 20;
  @!complete bra waitLoop;

// Hardware-managed suspend with try_wait
waitLoop:
  mbarrier.try_wait.shared.b64 complete, [shMem], state;
  @!complete bra waitLoop;
```

## mbarrier.pending_count

### Syntax

```ptx
mbarrier.pending_count.b64 count, state;
```

### Constraints

- `state` must be from a prior `mbarrier.arrive.noComplete` or `mbarrier.arrive_drop.noComplete`.
- `count` is `.u32` pending arrival count at time of that arrive.
- `sm_80+`

### Example

```ptx
mbarrier.arrive.noComplete.b64 state, [shMem], 1;
mbarrier.pending_count.b64 %r1, state;
```

## elect.sync

### Syntax

```ptx
elect.sync d|p, membermask;
```

### Constraints

- Elects one leader thread from `membermask`. Deterministic (same mask = same leader).
- `d`: `.b32` laneid of elected thread (can use sink `_`).
- `p`: `.pred`, `True` only for the elected thread.
- Executing thread must be in `membermask`. All threads in mask must execute before any resume.
- `sm_90+`

### Example

```ptx
elect.sync %r0|%p0, 0xffffffff;
```

## griddepcontrol

### Syntax

```ptx
griddepcontrol.action;

.action = { .launch_dependents, .wait }
```

### Variants

| Action | Behavior |
|--------|----------|
| `.launch_dependents` | Signals that runtime-designated dependent grids may launch once all CTAs issue this or complete. Idempotent per CTA. |
| `.wait` | Blocks until all prerequisite grids complete. Memory from prerequisites visible. |

### Constraints

- If prerequisite uses `.launch_dependents`, dependent must use `.wait`.
- `sm_90+`

### Example

```ptx
griddepcontrol.launch_dependents;
griddepcontrol.wait;
```

## mbarrier.expect_tx / mbarrier.complete_tx

### Syntax

```ptx
mbarrier.expect_tx{.sem.scope}{.space}.b64  [addr], txCount;
mbarrier.complete_tx{.sem.scope}{.space}.b64 [addr], txCount;

.sem   = { .relaxed }
.scope = { .cta, .cluster }
.space = { .shared{::cta}, .shared::cluster }
```

### Variants

| Instruction | Effect on tx-count |
|-------------|--------------------|
| `expect_tx` | tx-count += txCount |
| `complete_tx` | tx-count -= txCount (simulates async completion without actual async op) |

### Constraints

- `.sem` and `.scope` must be specified together.
- `sm_90+`

### Example

```ptx
mbarrier.expect_tx.b64 [addr], 32;
mbarrier.complete_tx.shared.b64 [mbarObj], 512;
```

## mbarrier shared memory scope support

| Operation | `.shared::cta` | `.shared::cluster` |
|-----------|:-:|:-:|
| `mbarrier.arrive` | Supported (returns state) | Supported (no return, use `_`) |
| `mbarrier.expect_tx` | Supported | Supported |
| `mbarrier.complete_tx` | Supported | Supported |
| Other ops (init, inval, test_wait, try_wait, pending_count) | Supported | Not supported |

## fence / membar

Covered in `ptx-isa-memory-spaces.md`. Key barrier-related fences:

```ptx
fence.mbarrier_init.release.cluster;          // after mbarrier.init, before cluster arrive
fence.proxy.async::generic.acquire.sync_restrict::shared::cluster.cluster;  // acquire remote barrier state
fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;     // release local barrier state
```
