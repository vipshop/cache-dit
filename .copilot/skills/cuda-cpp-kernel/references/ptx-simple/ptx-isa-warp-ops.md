<!-- PTX ISA 9.1 -->

## shfl.sync

### Syntax

```ptx
shfl.sync.mode.b32  d[|p], a, b, c, membermask;

.mode = { .up, .down, .bfly, .idx };
```

### Variants

| Mode    | Source lane `j`                        | Predicate `p` true when |
|---------|----------------------------------------|-------------------------|
| `.up`   | `lane - b`                             | `j >= maxLane`          |
| `.down` | `lane + b`                             | `j <= maxLane`          |
| `.bfly` | `lane ^ b`                             | `j <= maxLane`          |
| `.idx`  | `minLane \| (b[4:0] & ~segmask[4:0])` | `j <= maxLane`          |

Operand `c` packs two fields: `c[4:0]` = clamp value, `c[12:8]` = segment mask.

```
segmask[4:0] = c[12:8]
maxLane = (lane & segmask) | (cval & ~segmask)
minLane = (lane & segmask)
```

When `p` is false (out of range), the thread copies its own `a`. Only `.b32` type supported.

Sub-warp width W (power of 2): set `segmask = ~(W-1) & 0x1f`, `cval = W-1` for down/bfly/idx, `cval = 0` for up.

### Constraints

- `membermask`: 32-bit; executing thread must be set in mask, else undefined.
- Sourcing from an inactive thread or one not in `membermask` is undefined.
- sm_6x and below: all threads in `membermask` must execute the same `shfl.sync` in convergence.
- **PTX**: 6.0+. **Target**: sm_30+.

### Example

```ptx
// Butterfly reduction across full warp
shfl.sync.bfly.b32  Ry, Rx, 0x10, 0x1f, 0xffffffff;
add.f32             Rx, Ry, Rx;
shfl.sync.bfly.b32  Ry, Rx, 0x8,  0x1f, 0xffffffff;
add.f32             Rx, Ry, Rx;

// Inclusive prefix scan using .up
shfl.sync.up.b32  Ry|p, Rx, 0x1, 0x0, 0xffffffff;
@p add.f32        Rx, Ry, Rx;
```

---

## vote.sync

### Syntax

```ptx
vote.sync.mode.pred   d, {!}a, membermask;
vote.sync.ballot.b32  d, {!}a, membermask;

.mode = { .all, .any, .uni };
```

### Variants

| Mode      | Dest type | Result                                                                 |
|-----------|-----------|------------------------------------------------------------------------|
| `.all`    | `.pred`   | True if `a` is True for all non-exited threads in membermask.          |
| `.any`    | `.pred`   | True if `a` is True for any thread in membermask.                      |
| `.uni`    | `.pred`   | True if `a` has the same value in all non-exited threads in membermask.|
| `.ballot` | `.b32`    | Bit `i` of `d` = predicate of lane `i`. Non-membermask threads contribute 0. |

Negate the source predicate (`!a`) to compute `.none` (via `.all`) or `.not_all` (via `.any`).

### Constraints

- `membermask`: 32-bit; executing thread must be set in mask.
- sm_6x and below: all threads in `membermask` must execute the same `vote.sync` in convergence.
- **PTX**: 6.0+. **Target**: sm_30+.
- Non-sync `vote` deprecated PTX 6.0, removed for sm_70+ at PTX 6.4.

### Example

```ptx
vote.sync.all.pred     p, q, 0xffffffff;
vote.sync.ballot.b32   r1, p, 0xffffffff;
```

---

## match.sync

### Syntax

```ptx
match.any.sync.type  d, a, membermask;
match.all.sync.type  d[|p], a, membermask;

.type = { .b32, .b64 };
```

### Variants

| Mode   | `d` (b32 mask)                                                      | `p` (pred)                       |
|--------|---------------------------------------------------------------------|----------------------------------|
| `.any` | Mask of non-exited threads in membermask whose `a` equals this thread's `a`. | N/A                              |
| `.all` | Mask of non-exited threads if all have same `a`; else `0`.          | True if all match, false otherwise. Sink `_` allowed for `d` or `p`. |

Operand `a` has instruction type (`.b32` or `.b64`). Destination `d` is always `.b32`.

### Constraints

- `membermask`: 32-bit; executing thread must be set in mask.
- **PTX**: 6.0+. **Target**: sm_70+.

### Example

```ptx
match.any.sync.b32  d, a, 0xffffffff;
match.all.sync.b64  d|p, a, mask;
```

---

## redux.sync

### Syntax

```ptx
// Integer arithmetic
redux.sync.op.type   dst, src, membermask;
.op   = { .add, .min, .max }
.type = { .u32, .s32 }

// Bitwise
redux.sync.op.b32    dst, src, membermask;
.op   = { .and, .or, .xor }

// Floating-point
redux.sync.op{.abs}{.NaN}.f32  dst, src, membermask;
.op   = { .min, .max }
```

### Variants

| Category   | Operations              | Types           | Notes                                                                              |
|------------|-------------------------|-----------------|-------------------------------------------------------------------------------------|
| Arithmetic | `.add`, `.min`, `.max`  | `.u32`, `.s32`  | `.add` result truncated to 32 bits.                                                 |
| Bitwise    | `.and`, `.or`, `.xor`   | `.b32`          |                                                                                     |
| Float      | `.min`, `.max`          | `.f32`          | `.abs`: reduce absolute values. `.NaN`: propagate NaN (without it, NaN inputs skipped; result NaN only if all inputs NaN). `+0.0 > -0.0`. |

All participating threads receive the same result in `dst`.

### Constraints

- `membermask`: 32-bit; executing thread must be set in mask.
- Integer/bitwise: **PTX** 7.0+, **Target** sm_80+.
- `.f32`: **PTX** 8.6+, **Target** sm_100a (sm_100f from PTX 8.8).
- `.abs`, `.NaN`: **PTX** 8.6+, **Target** sm_100a (sm_100f from PTX 8.8).

### Example

```ptx
redux.sync.add.s32          dst, src, 0xff;
redux.sync.xor.b32          dst, src, mask;
redux.sync.min.abs.NaN.f32  dst, src, mask;
```

---

## activemask

### Syntax

```ptx
activemask.b32  d;
```

### Variants

None. Single form only. Destination `d` is a 32-bit register.

### Constraints

- Not a synchronization point; merely reads current execution mask.
- Active, predicated-on threads contribute 1; exited, inactive, or predicated-off threads contribute 0.
- **PTX**: 6.2+. **Target**: sm_30+.

### Example

```ptx
activemask.b32  %r1;
```

---

## Quick Reference

| Instruction   | PTX  | Min Target | Sync? | Type suffixes                       |
|---------------|------|------------|-------|-------------------------------------|
| `shfl.sync`   | 6.0  | sm_30      | Yes   | `.b32`                              |
| `vote.sync`   | 6.0  | sm_30      | Yes   | `.pred` (mode), `.b32` (ballot)     |
| `match.sync`  | 6.0  | sm_70      | Yes   | `.b32`, `.b64`                      |
| `redux.sync`  | 7.0  | sm_80      | Yes   | `.u32`, `.s32`, `.b32`, `.f32`      |
| `activemask`  | 6.2  | sm_30      | No    | `.b32`                              |

All `.sync` warp instructions require `membermask` (32-bit, bit `i` = lane `i`). Use `0xffffffff` for full-warp. Executing thread **must** be in `membermask`.
