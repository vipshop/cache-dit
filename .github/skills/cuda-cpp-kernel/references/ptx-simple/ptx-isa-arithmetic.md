<!-- PTX ISA 9.1 -->

# PTX Arithmetic Instructions

## Integer add / sub

### Syntax
```
add.type      d, a, b;
add{.sat}.s32 d, a, b;
sub.type      d, a, b;
sub{.sat}.s32 d, a, b;

.type = { .u16, .u32, .u64, .s16, .s32, .s64, .u16x2, .s16x2 };
```

### Constraints
- `.sat` applies only to `.s32` (clamps to MININT..MAXINT)
- `.u16x2` / `.s16x2`: operands are `.b32`, SIMD parallel on half-words; requires **sm_90+** (PTX 8.0)

### Example
```
add.sat.s32 c, c, 1;
add.u16x2   u, v, w;
sub.s32     c, a, b;
```

## Integer mul

### Syntax
```
mul.mode.type d, a, b;
.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64, .s16, .s32, .s64 };
```

### Constraints
- `.wide`: d is 2x width of a/b; supported only for 16-bit and 32-bit types
- `.hi` / `.lo`: d is same width, returns upper / lower half of full product

### Example
```
mul.wide.s32 z, x, y;   // 32*32 -> 64-bit result
mul.lo.s16   fa, fxs, fys;
```

## Integer mad

### Syntax
```
mad.mode.type     d, a, b, c;
mad.hi.sat.s32    d, a, b, c;
.mode = { .hi, .lo, .wide };
.type = { .u16, .u32, .u64, .s16, .s32, .s64 };
```

### Constraints
- Same `.wide` / `.hi` / `.lo` rules as `mul`
- `.sat` only for `.s32` in `.hi` mode

## Integer div / rem

### Syntax
```
div.type d, a, b;
rem.type d, a, b;
.type = { .u16, .u32, .u64, .s16, .s32, .s64 };
```
Division by zero yields unspecified machine-specific value.

## Integer abs / neg

### Syntax
```
abs.type d, a;
neg.type d, a;
.type = { .s16, .s32, .s64 };   // signed only
```

## Integer min / max

### Syntax
```
min.atype       d, a, b;
min{.relu}.btype d, a, b;
max.atype       d, a, b;
max{.relu}.btype d, a, b;

.atype = { .u16, .u32, .u64, .u16x2, .s16, .s64 };
.btype = { .s16x2, .s32 };
```

### Constraints
- `.relu` clamps negative results to 0; applies to `.s16x2`, `.s32`
- SIMD `.u16x2` / `.s16x2` and `.relu` require **sm_90+** (PTX 8.0)

## Bit Manipulation (popc, clz, bfind, brev, bfe, bfi, fns, bmsk, szext)

| Instruction | Syntax | Types | Min SM |
|---|---|---|---|
| `popc` | `popc.type d, a` | `.b32, .b64` | sm_20 |
| `clz` | `clz.type d, a` | `.b32, .b64` | sm_20 |
| `bfind` | `bfind{.shiftamt}.type d, a` | `.u32, .u64, .s32, .s64` | sm_20 |
| `brev` | `brev.type d, a` | `.b32, .b64` | sm_20 |
| `bfe` | `bfe.type d, a, b, c` | `.u32, .u64, .s32, .s64` | sm_20 |
| `bfi` | `bfi.type f, a, b, c, d` | `.b32, .b64` | sm_20 |
| `fns` | `fns.b32 d, mask, base, offset` | `.b32` only | sm_30 |
| `bmsk` | `bmsk.mode.b32 d, a, b` (.mode={.clamp,.wrap}) | `.b32` | sm_70 |
| `szext` | `szext.mode.type d, a, b` (.mode={.clamp,.wrap}) | `.u32, .s32` | sm_70 |

- `popc`, `clz` destination is always `.u32`
- `bfind` returns `0xFFFFFFFF` if no non-sign bit found; `.shiftamt` returns left-shift amount instead
- `bfe`: b = start pos, c = length (both 0..255); sign-extends for signed types
- `bfi`: inserts bit field from a into b at position c with length d

## Integer Dot Product (dp4a, dp2a)

### Syntax
```
dp4a.atype.btype         d, a, b, c;
dp2a.mode.atype.btype    d, a, b, c;
.atype = .btype = { .u32, .s32 };
.mode  = { .lo, .hi };            // dp2a only
```

### Constraints
- Requires **sm_61+**
- `dp4a`: 4-way byte dot product accumulated into 32-bit d
- `dp2a`: 2-way 16-bit x 8-bit dot product; `.lo`/`.hi` selects which half of b

## Extended-Precision Integer (add.cc, addc, sub.cc, subc, mad.cc, madc)

### Syntax
```
add.cc.type       d, a, b;          // carry-out to CC.CF
addc{.cc}.type    d, a, b;          // carry-in from CC.CF
sub.cc.type       d, a, b;          // borrow-out to CC.CF
subc{.cc}.type    d, a, b;          // borrow-in from CC.CF
mad{.hi,.lo}.cc.type  d, a, b, c;   // carry-out
madc{.hi,.lo}{.cc}.type d, a, b, c; // carry-in, optional carry-out

.type = { .u32, .s32, .u64, .s64 };
```

### Constraints
- CC register is implicit, single carry flag bit; not preserved across calls
- 32-bit: all targets; 64-bit: **sm_20+**
- `mad.cc` / `madc`: **sm_20+**

### Example
```
// 128-bit addition: [x4,x3,x2,x1] = [y4,y3,y2,y1] + [z4,z3,z2,z1]
add.cc.u32  x1, y1, z1;
addc.cc.u32 x2, y2, z2;
addc.cc.u32 x3, y3, z3;
addc.u32    x4, y4, z4;
```

---

## FP32/FP64 add / sub / mul

### Syntax
```
{add,sub,mul}{.rnd}{.ftz}{.sat}.f32   d, a, b;
{add,sub,mul}{.rnd}{.ftz}.f32x2       d, a, b;
{add,sub,mul}{.rnd}.f64               d, a, b;

.rnd = { .rn, .rz, .rm, .rp };   // default .rn
```

### Constraints

| Modifier | `.f32` | `.f64` | `.f32x2` |
|---|---|---|---|
| `.rn, .rz` | all targets | all targets | sm_100+ |
| `.rm, .rp` | sm_20+ | sm_13+ | sm_100+ |
| `.ftz` | yes | n/a | yes |
| `.sat` | yes (clamps [0,1]) | n/a | n/a |

- No explicit `.rnd` => default `.rn`; optimizer may fold mul+add into fma
- Explicit `.rnd` prevents aggressive optimization

## FP32/FP64 fma

### Syntax
```
fma.rnd{.ftz}{.sat}.f32   d, a, b, c;
fma.rnd{.ftz}.f32x2       d, a, b, c;
fma.rnd.f64               d, a, b, c;

.rnd = { .rn, .rz, .rm, .rp };   // REQUIRED, no default
```

### Constraints
- Computes `a*b+c` in infinite precision, then rounds once => true FMA
- `.f32`: **sm_20+**; `.f64`: **sm_13+**; `.f32x2`: **sm_100+**
- `fma.f64` is identical to `mad.f64`

### Example
```
fma.rn.ftz.f32 w, x, y, z;
fma.rn.f64     d, a, b, c;
```

## FP32/FP64 mad

`mad.rnd.{f32,f64}` is identical to `fma.rnd.{f32,f64}` on sm_20+. Rounding modifier required for sm_20+.

## FP32/FP64 div

### Syntax
```
div.approx{.ftz}.f32   d, a, b;   // fast, max 2 ulp error
div.full{.ftz}.f32     d, a, b;   // full-range approx, max 2 ulp, no rounding
div.rnd{.ftz}.f32      d, a, b;   // IEEE 754 compliant
div.rnd.f64            d, a, b;   // IEEE 754 compliant

.rnd = { .rn, .rz, .rm, .rp };
```

### Constraints
- `div.approx.f32`: all targets; for `|b|` in `[2^-126, 2^126]`, max 2 ulp
- `div.full.f32`: all targets; full-range, max 2 ulp, no rounding modifier
- `div.rnd.f32`: **sm_20+**
- `div.rnd.f64`: `.rn` **sm_13+**; `.rz,.rm,.rp` **sm_20+**

## FP32/FP64 abs / neg

```
abs{.ftz}.f32 d, a;     neg{.ftz}.f32 d, a;
abs.f64       d, a;     neg.f64       d, a;
```
`.ftz` flushes subnormals. `.f64` requires **sm_13+**.

## FP32/FP64 min / max

### Syntax
```
{min,max}{.ftz}{.NaN}{.xorsign.abs}.f32 d, a, b;
{min,max}{.ftz}{.NaN}{.abs}.f32         d, a, b, c;   // 3-input
{min,max}.f64                           d, a, b;
```

### Constraints
- Default: NaN inputs propagate non-NaN operand (`minNum`/`maxNum` semantics)
- `.NaN`: result is canonical NaN if any input is NaN; **sm_80+**
- `.xorsign.abs`: sign = XOR of input signs, magnitude = min/max of |a|,|b|; **sm_86+**
- 3-input: **sm_100+**
- `-0.0 < +0.0`

## FP32/FP64 rcp / sqrt / rsqrt

| Instruction | Syntax | Precision | Min SM |
|---|---|---|---|
| `rcp.approx{.ftz}.f32` | `d = 1/a` | max 1 ulp | all |
| `rcp.rnd{.ftz}.f32` | IEEE 754 | exact | sm_20 |
| `rcp.rnd.f64` | IEEE 754 | exact | sm_13 (.rn) / sm_20 |
| `rcp.approx.ftz.f64` | gross approx (20-bit mantissa) | low | sm_20 |
| `sqrt.approx{.ftz}.f32` | `d = sqrt(a)` | max rel err 2^-23 | all |
| `sqrt.rnd{.ftz}.f32` | IEEE 754 | exact | sm_20 |
| `sqrt.rnd.f64` | IEEE 754 | exact | sm_13 (.rn) / sm_20 |
| `rsqrt.approx{.ftz}.f32` | `d = 1/sqrt(a)` | max rel err 2^-22.9 | all |
| `rsqrt.approx.f64` | approx | emulated, slow | sm_13 |
| `rsqrt.approx.ftz.f64` | gross approx (20-bit mantissa) | low | sm_20 |

`.rnd = { .rn, .rz, .rm, .rp }` -- required (no default) for IEEE variants.

## FP32 Transcendentals (sin, cos, lg2, ex2, tanh)

### Syntax
```
sin.approx{.ftz}.f32   d, a;
cos.approx{.ftz}.f32   d, a;
lg2.approx{.ftz}.f32   d, a;
ex2.approx{.ftz}.f32   d, a;
tanh.approx.f32        d, a;      // sm_75+
```

### Precision

| Instruction | Max Error | Range |
|---|---|---|
| `sin`, `cos` | 2^-20.5 abs | [-2pi, 2pi] |
| `sin`, `cos` | 2^-14.7 abs | [-100pi, 100pi] |
| `lg2` | 2^-22 abs/rel | full range |
| `ex2` | 2 ulp | full range |
| `tanh` | 2^-11 rel | full range |

`.approx` is required (PTX 1.4+). `tanh` does not support `.ftz`.

---

## Half Precision (f16/bf16) add / sub / mul

### Syntax
```
{add,sub,mul}{.rnd}{.ftz}{.sat}.f16    d, a, b;
{add,sub,mul}{.rnd}{.ftz}{.sat}.f16x2  d, a, b;
{add,sub,mul}{.rnd}.bf16               d, a, b;
{add,sub,mul}{.rnd}.bf16x2             d, a, b;

.rnd = { .rn };   // only .rn supported
```

### Constraints
- `.f16` / `.f16x2`: **sm_53+** (PTX 4.2)
- `.bf16` / `.bf16x2`: **sm_90+** (PTX 7.8)
- `.ftz`: f16 only; `.sat`: f16 only (clamps [0,1])
- SIMD x2 variants: operands are `.b32`, parallel on packed half-words

## Half Precision fma

### Syntax
```
fma.rnd{.ftz}{.sat}.f16          d, a, b, c;
fma.rnd{.ftz}{.sat}.f16x2        d, a, b, c;
fma.rnd{.ftz}.relu.f16           d, a, b, c;
fma.rnd{.ftz}.relu.f16x2         d, a, b, c;
fma.rnd{.relu}.bf16              d, a, b, c;
fma.rnd{.relu}.bf16x2            d, a, b, c;
fma.rnd.oob{.relu}.type          d, a, b, c;

.rnd = { .rn };
```

### Constraints
- Base f16/f16x2: **sm_53+**
- `.relu` (clamp negative to 0): f16 **sm_80+**, bf16 **sm_80+**
- `.oob` (force 0 if operand is OOB NaN): **sm_90+** (PTX 8.1)

### Example
```
fma.rn.f16         d0, a0, b0, c0;
fma.rn.relu.bf16x2 f2, f0, f1, f1;
fma.rn.oob.relu.f16x2 p3, p1, p2, p2;
```

## Half Precision abs / neg

```
abs{.ftz}.f16   d, a;     neg{.ftz}.f16   d, a;
abs{.ftz}.f16x2 d, a;     neg{.ftz}.f16x2 d, a;
abs.bf16        d, a;     neg.bf16        d, a;
abs.bf16x2      d, a;     neg.bf16x2      d, a;
```
f16: **sm_53+**; bf16: **sm_80+**.

## Half Precision min / max

### Syntax
```
{min,max}{.ftz}{.NaN}{.xorsign.abs}.f16    d, a, b;
{min,max}{.ftz}{.NaN}{.xorsign.abs}.f16x2  d, a, b;
{min,max}{.NaN}{.xorsign.abs}.bf16         d, a, b;
{min,max}{.NaN}{.xorsign.abs}.bf16x2       d, a, b;
```
Requires **sm_80+**. `.xorsign.abs` requires **sm_86+**. Same NaN semantics as f32 min/max.

## Half Precision tanh / ex2

```
tanh.approx.type d, a;           // .type = { .f16, .f16x2, .bf16, .bf16x2 }
ex2.approx.type  d, a;           // .type = { .f16, .f16x2 }
ex2.approx.ftz.type d, a;        // .type = { .bf16, .bf16x2 }
```

| | f16 max error | bf16 max error | f16 min SM | bf16 min SM |
|---|---|---|---|---|
| `tanh` | 2^-10.987 abs | 2^-8 abs | sm_75 | sm_90 |
| `ex2` | 2^-9.9 rel | 2^-7 rel | sm_75 | sm_90 |

`ex2.bf16` requires `.ftz`; `ex2.f16` does not.

---

## Mixed Precision FP (sm_100+)

### Syntax
```
add{.rnd}{.sat}.f32.atype   d, a, c;      // d = cvt(a) + c
sub{.rnd}{.sat}.f32.atype   d, a, c;      // d = cvt(a) - c
fma.rnd{.sat}.f32.abtype    d, a, b, c;   // d = cvt(a)*cvt(b) + c

.atype = .abtype = { .f16, .bf16 };
.rnd   = { .rn, .rz, .rm, .rp };
```

### Constraints
- All require **sm_100+** (PTX 8.6)
- Input a (and b for fma) is converted from f16/bf16 to f32 before operation
- `.sat` clamps result to [0.0, 1.0]
- `fma`: rounding modifier required (no default)
- `add`, `sub`: default `.rn`

### Example
```
fma.rn.sat.f32.f16 fd, ha, hb, fc;
add.rz.f32.bf16    fd, ba, fc;
```
