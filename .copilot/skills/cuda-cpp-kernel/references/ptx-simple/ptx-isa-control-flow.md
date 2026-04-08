<!-- PTX ISA 9.1 -->

# PTX Control Flow & Predicated Execution

## Predicated Execution (`@p` / `@!p`)

### Syntax

```ptx
@{!}p  instruction;
```

### Variants

| Guard    | Behavior                                        |
|----------|-------------------------------------------------|
| `@p`     | Execute instruction when predicate `p` is true  |
| `@!p`    | Execute instruction when predicate `p` is false |
| *(none)* | Execute unconditionally                         |

Predicate registers are declared as `.reg .pred`:

```ptx
.reg .pred p, q, r;
```

### Constraints

- All PTX instructions accept an optional guard predicate.
- No direct conversion between predicates and integers. Use `selp` to materialize:
  ```ptx
  selp.u32 %r1, 1, 0, %p;    // %r1 = %p ? 1 : 0
  ```
- Predicate manipulation: `and`, `or`, `xor`, `not`, `mov` on `.pred` operands.

### Example

```ptx
setp.eq.f32  p, y, 0;          // is y zero?
@!p div.f32  ratio, x, y;      // skip division when y==0
@q  bra      L23;              // conditional branch
```

## `setp` -- Comparison Operators

### Syntax

```ptx
setp.CmpOp.type  p, a, b;
setp.CmpOp.type  p|q, a, b;    // set p = result, q = !result
```

### Variants

**Integer / Bit-Size Comparisons:**

| Meaning  | Signed | Unsigned | Bit-Size |
|----------|--------|----------|----------|
| a == b   | `eq`   | `eq`     | `eq`     |
| a != b   | `ne`   | `ne`     | `ne`     |
| a < b    | `lt`   | `lo`     | n/a      |
| a <= b   | `le`   | `ls`     | n/a      |
| a > b    | `gt`   | `hi`     | n/a      |
| a >= b   | `ge`   | `hs`     | n/a      |

**Floating-Point -- Ordered** (either operand NaN => result is False):

`eq`, `ne`, `lt`, `le`, `gt`, `ge`

**Floating-Point -- Unordered** (either operand NaN => result is True):

`equ`, `neu`, `ltu`, `leu`, `gtu`, `geu`

**NaN Testing:**

| Meaning                    | Operator |
|----------------------------|----------|
| !isNaN(a) && !isNaN(b)     | `num`    |
| isNaN(a) \|\| isNaN(b)     | `nan`    |

### Constraints

- Unsigned ordering operators: `lo` (lower), `ls` (lower-or-same), `hi` (higher), `hs` (higher-or-same).
- Bit-size types support only `eq` and `ne`.

### Example

```ptx
setp.lt.s32   p, i, n;         // p = (i < n)
setp.geu.f32  p|q, a, b;       // p = (a >= b || NaN), q = !(...)
```

## `bra` -- Branch

### Syntax

```ptx
@p   bra{.uni}  tgt;            // conditional branch to label
     bra{.uni}  tgt;            // unconditional branch
```

### Variants

| Modifier | Meaning                                                       |
|----------|---------------------------------------------------------------|
| *(none)* | Potentially divergent branch                                  |
| `.uni`   | Non-divergent: all active threads share same predicate/target |

### Constraints

- Branch target `tgt` must be a label (no indirect branching via `bra`).
- PTX ISA 1.0+. All target architectures.

### Example

```ptx
bra.uni  L_exit;               // uniform unconditional jump
@q       bra  L23;             // conditional branch
```

## `brx.idx` -- Indirect Branch

### Syntax

```ptx
@p   brx.idx{.uni}  index, tlist;
     brx.idx{.uni}  index, tlist;
```

### Variants

- `index`: `.u32` register, zero-based index into `tlist`.
- `tlist`: label of a `.branchtargets` directive (must be in local function scope).
- `.uni`: asserts non-divergent (all active threads have identical index and predicate).

### Constraints

- Behavior undefined if `index >= length(tlist)`.
- `.branchtargets` must be defined before use; labels must be within the current function.
- PTX ISA 6.0+. Requires `sm_30`.

### Example

```ptx
.function foo () {
    .reg .u32 %r0;
    L1: ...
    L2: ...
    L3: ...
    ts: .branchtargets L1, L2, L3;
    @p brx.idx %r0, ts;
}
```

## `call` -- Function Call

### Syntax

```ptx
// direct call
call{.uni} (ret-param), func, (param-list);
call{.uni} func, (param-list);
call{.uni} func;

// indirect call via pointer + call table
call{.uni} (ret-param), fptr, (param-list), flist;

// indirect call via pointer + prototype
call{.uni} (ret-param), fptr, (param-list), fproto;
```

### Variants

| Form     | Target                 | Extra operand                          |
|----------|------------------------|----------------------------------------|
| Direct   | symbolic function name | none                                   |
| Indirect | register `fptr`        | `flist` (`.calltargets` / jump table)  |
| Indirect | register `fptr`        | `fproto` (`.callprototype`)            |

- `.uni`: asserts non-divergent call.
- Arguments: pass-by-value (registers, immediates, or `.param` variables).

### Constraints

- Direct call: PTX ISA 1.0+, all architectures.
- Indirect call: PTX ISA 2.1+, requires `sm_20`.
- `flist`: complete target list allows backend optimization of calling convention.
- `fproto`: incomplete target list forces ABI calling convention. Undefined behavior if callee does not match prototype.

### Example

```ptx
    call     init;                          // no args
    call.uni g, (a);                        // uniform call
@p  call     (d), h, (a, b);               // return value in d

// indirect via jump table
.global .u32 jmptbl[3] = { foo, bar, baz };
    call (retval), %r0, (x, y), jmptbl;

// indirect via .calltargets
Ftgt: .calltargets foo, bar, baz;
    call (retval), %r0, (x, y), Ftgt;

// indirect via .callprototype
Fproto: .callprototype _ (.param .u32 _, .param .u32 _);
    call %fptr, (x, y), Fproto;
```

## `ret` -- Return

### Syntax

```ptx
ret{.uni};
```

### Variants

| Modifier | Meaning                                               |
|----------|-------------------------------------------------------|
| *(none)* | Divergent return: suspends threads until all are ready |
| `.uni`   | Non-divergent: all active threads return together      |

### Constraints

- Move return values into return parameter variables before executing `ret`.
- A `ret` in a top-level entry routine terminates the thread.
- PTX ISA 1.0+. All target architectures.

### Example

```ptx
    ret;
@p  ret;
```

## `exit` -- Thread Exit

### Syntax

```ptx
exit;
```

### Variants

None.

### Constraints

- Barriers exclusively waiting on arrivals from exited threads are always released.
- PTX ISA 1.0+. All target architectures.

### Example

```ptx
    exit;
@p  exit;
```

## `nanosleep` -- Thread Sleep

### Syntax

```ptx
nanosleep.u32  t;
```

### Variants

- `t`: `.u32` register or immediate value specifying sleep duration in nanoseconds.

### Constraints

- Sleep duration is approximate, guaranteed in interval `[0, 2*t]`.
- Maximum sleep duration: 1 millisecond.
- Implementation may reduce per-thread sleep so all sleeping threads in a warp wake together.
- PTX ISA 6.3+. Requires `sm_70`.

### Example

```ptx
.reg .b32  r;
.reg .pred p;

nanosleep.u32  r;              // sleep for r nanoseconds
nanosleep.u32  42;             // sleep for ~42 ns
@p nanosleep.u32 r;            // predicated sleep
```

## Thread Divergence

### Syntax

Control-flow instructions accept an optional `.uni` suffix:

```ptx
bra.uni   tgt;
call.uni  func;
ret.uni;
```

### Variants

| Thread state  | Definition                                |
|---------------|-------------------------------------------|
| **Uniform**   | All threads in the CTA take the same path |
| **Divergent** | Threads take different control-flow paths  |

### Constraints

- All control-flow instructions are assumed divergent unless marked `.uni`.
- The code generator automatically determines re-convergence points for divergent branches.
- Marking branches `.uni` when provably non-divergent lets the compiler skip divergence handling.
- Divergent CTAs may have lower performance than uniform CTAs.

### Example

```ptx
// Compiler can optimize knowing all threads branch the same way
bra.uni  loop_top;

// Divergent: threads may take different paths
@p bra   else_branch;
```
