# PTX ISA 9.1 -- Data Types & Conversions

Reference for PTX type system, register declarations, and the `cvt` conversion instruction.
Source: NVIDIA PTX ISA 9.1 specification.

## 1. Fundamental Types (Section 5.2.1)

Every register variable and instruction operand carries a type specifier. The fundamental types are:

| Basic Type       | Specifiers                              | Register Widths  |
|------------------|-----------------------------------------|------------------|
| Signed integer   | `.s8`, `.s16`, `.s32`, `.s64`           | 8/16/32/64 bits  |
| Unsigned integer | `.u8`, `.u16`, `.u32`, `.u64`           | 8/16/32/64 bits  |
| Floating-point   | `.f16`, `.f16x2`, `.f32`, `.f64`        | 16/32/32/64 bits |
| Bits (untyped)   | `.b8`, `.b16`, `.b32`, `.b64`, `.b128`  | 8-128 bits       |
| Predicate        | `.pred`                                 | 1 bit            |

Type compatibility rules:
- Signed and unsigned integers of the same size are compatible.
- Bit-size types are compatible with any fundamental type of the same width.

### Sub-word restrictions (Section 5.2.2)

`.u8`, `.s8`, `.b8` types are restricted to `ld`, `st`, and `cvt` instructions only. In practice,
8-bit and 16-bit values are held in 32-bit registers and operated on after widening.

## 2. Alternate Floating-Point Formats (Section 5.2.3)

These are *not* fundamental types. They are instruction-type qualifiers used with `cvt` and MMA
instructions. Values are stored in bit-size registers of the appropriate width.

| Format   | Bits | Exponent | Mantissa | Register Type | Notes                                |
|----------|------|----------|----------|---------------|--------------------------------------|
| `.bf16`  | 16   | 8        | 7        | `.b16`        | Same range as f32, reduced precision |
| `.tf32`  | 32   | 8        | >=10     | `.b32`        | MMA-only; layout is impl-defined     |
| `.e4m3`  | 8    | 4        | 3        | `.b8`/packed  | No infinity; NaN = 0x7f/0xff         |
| `.e5m2`  | 8    | 5        | 2        | `.b8`/packed  | FP8 format                           |
| `.e2m3`  | 6    | 2        | 3        | packed `.b16` | No infinity/NaN; 2 MSB bits = 0     |
| `.e3m2`  | 6    | 3        | 2        | packed `.b16` | No infinity/NaN; 2 MSB bits = 0     |
| `.e2m1`  | 4    | 2        | 1        | `.b8` (x2)    | No infinity/NaN (FP4)                |
| `.ue8m0` | 8    | 8        | 0        | packed `.b16` | Unsigned; exponent-only scaling      |

### Fixed-point format

| Format  | Bits | Description                                      | Register Type |
|---------|------|--------------------------------------------------|---------------|
| `.s2f6` | 8    | Signed 2's complement: 2 int bits + 6 frac bits | packed `.b16` |

## 3. Packed Data Types (Section 5.2.5)

Packed types bundle 2 or 4 scalar elements for SIMD-style operations.

| Packed Type   | Elements | Element Type | Declared As         |
|---------------|----------|--------------|---------------------|
| `.f16x2`      | 2        | `.f16`       | `.f16x2` or `.b32`  |
| `.bf16x2`     | 2        | `.bf16`      | `.b32`              |
| `.e4m3x2`     | 2        | `.e4m3`      | `.b16`              |
| `.e5m2x2`     | 2        | `.e5m2`      | `.b16`              |
| `.e2m3x2`     | 2        | `.e2m3`      | `.b16`              |
| `.e3m2x2`     | 2        | `.e3m2`      | `.b16`              |
| `.e2m1x2`     | 2        | `.e2m1`      | `.b8`               |
| `.ue8m0x2`    | 2        | `.ue8m0`     | `.b16`              |
| `.e4m3x4`     | 4        | `.e4m3`      | `.b32`              |
| `.e5m2x4`     | 4        | `.e5m2`      | `.b32`              |
| `.e2m1x4`     | 4        | `.e2m1`      | `.b16`              |
| `.e2m3x4`     | 4        | `.e2m3`      | `.b32`              |
| `.e3m2x4`     | 4        | `.e3m2`      | `.b32`              |

## 4. Vector Types & Variables (Section 5.4.2)

Vectors of length 2 or 4 are declared with `.v2` or `.v4` prefixes. Maximum total width is 128 bits
(so `.v4 .f64` is illegal). Three-element vectors should use `.v4` with padding.

```ptx
.reg    .v4 .f32 accel;       // 4x32-bit float vector (128 bits)
.global .v2 .u16 uv;          // 2x16-bit unsigned vector
.global .v4 .b8  mask;        // 4x8-bit byte vector

// Parameterized register names
.reg .b32 %r<100>;            // declares %r0 .. %r99
```

Default alignment is the overall vector size (e.g., `.v4 .f32` aligns to 16 bytes).

## 5. Scalar Conversion Rules (Section 6.5)

The `cvt` instruction converts between types. The conversion method depends on source/destination
category:

| Conversion           | Method           | Rounding Required? |
|----------------------|------------------|--------------------|
| int -> wider int     | `sext` / `zext`  | No                 |
| int -> narrower int  | `chop` (truncate)| No                 |
| int -> float         | `s2f` / `u2f`    | Yes (FP rounding)  |
| float -> int         | `f2s` / `f2u`    | Yes (int rounding) |
| float -> wider float | `f2f` (exact)    | No                 |
| float -> narrower FP | `f2f` (lossy)    | Yes (FP rounding)  |
| same type/size       | identity / `f2f` | No (unless rounding to int) |

Key rules:
- `sext` = sign-extend, `zext` = zero-extend, `chop` = keep low bits.
- If the destination register is wider than the destination format, the result is extended after
  chopping. Extension type (sign or zero) depends on the destination format.
- Float-to-int conversions saturate (clamp) to the destination range by default.
- Out-of-range float-to-float: IEEE 754 Inf for `.f32`/`.f64`; ~131,000 for `.f16`.

## 6. Rounding Modifiers (Section 6.5.2)

### Floating-point rounding (for int-to-float, float-to-narrower-float)

| Modifier | Description                                         |
|----------|-----------------------------------------------------|
| `.rn`    | Round to nearest even (default IEEE 754 mode)       |
| `.rna`   | Round to nearest, ties away from zero               |
| `.rz`    | Round towards zero (truncation)                     |
| `.rm`    | Round towards negative infinity (floor)             |
| `.rp`    | Round towards positive infinity (ceil)              |
| `.rs`    | Stochastic rounding (uses random bits operand)      |

### Integer rounding (for float-to-int, float-to-same-size-float rounding)

| Modifier | Description                                         |
|----------|-----------------------------------------------------|
| `.rni`   | Round to nearest integer, ties to even              |
| `.rzi`   | Round towards zero                                  |
| `.rmi`   | Round towards negative infinity                     |
| `.rpi`   | Round towards positive infinity                     |

When rounding is required it is mandatory -- omitting it is a compile error.

## 7. The `cvt` Instruction (Section 9.7.9.21)

### Basic syntax

```ptx
cvt{.irnd}{.ftz}{.sat}.dtype.atype         d, a;   // integer rounding
cvt{.frnd}{.ftz}{.sat}.dtype.atype         d, a;   // FP rounding

// Fundamental type pairs
.dtype = .atype = { .u8, .u16, .u32, .u64,
                    .s8, .s16, .s32, .s64,
                    .bf16, .f16, .f32, .f64 };
```

### Packed / alternate-format syntax

```ptx
// f32 -> packed f16x2 / bf16x2
cvt.frnd{.relu}{.satfinite}.f16x2.f32      d, a, b;
cvt.frnd{.relu}{.satfinite}.bf16x2.f32     d, a, b;

// f32 -> tf32
cvt.rna{.satfinite}.tf32.f32               d, a;

// f32 -> FP8 packed pair
cvt.rn.satfinite{.relu}.e4m3x2.f32         d, a, b;
cvt.rn.satfinite{.relu}.e5m2x2.f32         d, a, b;

// FP8 packed pair -> f16x2 (upconvert)
cvt.rn{.relu}.f16x2.e4m3x2                 d, a;
cvt.rn{.relu}.f16x2.e5m2x2                 d, a;

// f32 -> FP4 (e2m1x2)
cvt.rn.satfinite{.relu}.e2m1x2.f32         d, a, b;
// f32 x4 -> packed FP8x4 / FP4x4 with stochastic rounding
cvt.rs{.relu}.satfinite.e4m3x4.f32         d, {a, b, e, f}, rbits;
cvt.rs{.relu}.satfinite.e2m1x4.f32         d, {a, b, e, f}, rbits;
```

### Saturation modifiers

| Modifier      | Effect                                                    |
|---------------|-----------------------------------------------------------|
| `.sat`        | Clamps integers to MININT..MAXINT; floats to [0.0, 1.0]  |
| `.satfinite`  | NaN -> NaN (or MAX_NORM for formats without NaN); Inf -> MAX_NORM |
| `.relu`       | Clamps negative results to +0; NaN -> canonical NaN      |
| `.ftz`        | Flush .f32 subnormals to sign-preserving zero             |

`.satfinite` is mandatory when converting to `.e4m3x2`, `.e5m2x2`, `.e2m1x2`, `.e2m3x2`,
`.e3m2x2`, and their x4 variants.

### Packing semantics for `cvt` with packed destination

For `f16x2`/`bf16x2` destinations from two `.f32` inputs:
- `d[31:16] = convert(a)`  (upper half)
- `d[15:0]  = convert(b)`  (lower half)

For `e4m3x2`/`e5m2x2` destinations from two `.f32` inputs:
- `d[15:8] = convert(a)`
- `d[7:0]  = convert(b)`

For `e2m1x2` destinations:
- `d[7:4] = convert(a)`
- `d[3:0] = convert(b)`

### Common examples

```ptx
// Basic scalar conversions
cvt.f32.s32      f, i;            // int32 -> float32 (exact for small values)
cvt.s32.f64      j, r;            // float64 -> int32 (saturates by default)
cvt.rni.f32.f32  x, y;            // round f32 to nearest integer, keep as f32

// f16 / bf16 conversions
cvt.rn.f16.f32        h, f;       // f32 -> f16
cvt.rn.relu.f16.f32   h, f;       // f32 -> f16 with ReLU clamp
cvt.f32.f16           f, h;       // f16 -> f32 (exact)
cvt.rn.bf16.f32       b, f;       // f32 -> bf16
cvt.f32.bf16          f, b;       // bf16 -> f32

// Packed f16x2 from two f32 values
cvt.rz.f16x2.f32                d, a, b;
cvt.rn.relu.satfinite.f16x2.f32 d, a, b;

// FP8 conversions (sm_89+)
cvt.rn.satfinite.e4m3x2.f32     d, a, b;   // two f32 -> packed e4m3x2
cvt.rn.f16x2.e4m3x2             d, a;      // packed e4m3x2 -> f16x2

// tf32 conversion (sm_80+)
cvt.rna.satfinite.tf32.f32       d, a;

// Stochastic rounding (sm_100a+)
cvt.rs.f16x2.f32   d, a, b, rbits;
```

## 8. The `cvt.pack` Instruction (Section 9.7.9.22)

Converts and packs two 32-bit integers into narrower integer fields within a 32-bit destination.
Used for quantization pipelines.

```ptx
cvt.pack.sat.convertType.abType         d, a, b;
cvt.pack.sat.convertType.abType.cType   d, a, b, c;

// .convertType = { .u16, .s16, .u8, .s8, .u4, .s4, .u2, .s2 }
// .abType      = { .s32 }
// .cType       = { .b32 }   // provides upper bits via c
```

When operand `c` is present, converted `a` and `b` are packed into the low bits of `d`, and
remaining upper bits are copied from `c`. This enables iterative packing of multiple values.

```ptx
// Pack four s32 values into four u8 lanes of a single u32
cvt.pack.sat.u8.s32.b32   %r1, %r2, %r3, 0;     // pack first two into low 16 bits
cvt.pack.sat.u8.s32.b32   %r4, %r5, %r6, %r1;   // pack next two, shift previous up
```

Requires `sm_72+` (sub-byte types `.u4`/`.s4`/`.u2`/`.s2` require `sm_75+`).

## 9. Alternate-Format Conversion Matrix (Table 16)

Supported `cvt` float-to-float conversions among alternate formats (f2f = valid):

| Source \ Dest | f16 | f32 | bf16 | e4m3 | e5m2 | e2m3 | e3m2 | e2m1 | ue8m0 |
|---------------|-----|-----|------|------|------|------|------|------|-------|
| **f16**       | --  | f2f | f2f  | f2f  | f2f  | f2f  | f2f  | f2f  | --    |
| **f32**       | f2f | --  | f2f  | f2f  | f2f  | f2f  | f2f  | f2f  | f2f   |
| **bf16**      | f2f | f2f | --   | f2f  | f2f  | f2f  | f2f  | f2f  | f2f   |
| **e4m3**      | f2f | --  | --   | --   | --   | --   | --   | --   | --    |
| **e5m2**      | f2f | --  | --   | --   | --   | --   | --   | --   | --    |
| **e2m3**      | f2f | --  | --   | --   | --   | --   | --   | --   | --    |
| **e3m2**      | f2f | --  | --   | --   | --   | --   | --   | --   | --    |
| **e2m1**      | f2f | --  | --   | --   | --   | --   | --   | --   | --    |
| **ue8m0**     | --  | --  | f2f  | --   | --   | --   | --   | --   | --    |

Narrow FP formats (e4m3, e5m2, e2m3, e3m2, e2m1) can only upconvert to `.f16` (via packed x2
instructions). Downconversion from `.f16`, `.f32`, or `.bf16` to these formats is supported.
`ue8m0` converts only to/from `.bf16`.
