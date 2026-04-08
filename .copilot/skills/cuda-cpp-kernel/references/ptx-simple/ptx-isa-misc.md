<!-- PTX ISA 9.1 -->

## prmt -- Byte Permute
### Syntax
```ptx
prmt.b32{.mode}  d, a, b, c;
.mode = { .f4e, .b4e, .rc8, .ecl, .ecr, .rc16 };
```
### Variants
**Default (no mode):** `c` provides four 4-bit selectors in `c[15:12]`, `c[11:8]`, `c[7:4]`, `c[3:0]`. Each selector's 3 LSBs pick a byte (0..7) from `{b, a}` = `{b7..b4, b3..b0}`. MSB of selector enables sign-extension of that byte.

| Mode | Description |
|------|-------------|
| `.f4e` | Forward 4 extract: sliding window `{a,b}` shifted right by `c[1:0]` bytes |
| `.b4e` | Backward 4 extract: reverse sliding window |
| `.rc8` | Replicate byte `c[1:0]` to all 4 positions |
| `.ecl` | Edge clamp left |
| `.ecr` | Edge clamp right |
| `.rc16` | Replicate halfword `c[0]` to both halves |

### Constraints
- All target architectures. PTX ISA 2.0+.
### Example
```ptx
prmt.b32      d, a, b, 0x3210;  // identity permute
prmt.b32      d, a, b, 0x0123;  // reverse bytes
prmt.b32.f4e  d, a, b, c;       // funnel extract
```

---

## bfe -- Bit Field Extract
### Syntax
```ptx
bfe.type  d, a, b, c;
.type = { .u32, .u64, .s32, .s64 };
```
### Variants
- `.u32`/`.u64`: zero-extends extracted field
- `.s32`/`.s64`: sign-extends using bit at `min(pos+len-1, msb)`
### Constraints
- `b`: start position (0..255), `c`: field length (0..255). If len==0 or start > msb, result is 0 (unsigned) or sign-filled (signed). Requires `sm_20`+. PTX ISA 2.0+.
### Example
```ptx
bfe.u32  d, a, 8, 4;   // extract 4 bits starting at bit 8
```

---

## bfi -- Bit Field Insert
### Syntax
```ptx
bfi.type  f, a, b, c, d;
.type = { .b32, .b64 };
```
### Constraints
- Inserts low `d` bits of `a` into `b` starting at position `c`. If len==0 or start > msb, result is `b`. Requires `sm_20`+. PTX ISA 2.0+.
### Example
```ptx
bfi.b32  f, a, b, 8, 4;  // insert 4 bits of a into b at bit 8
```

---

## dp4a -- 4-Way Byte Dot Product Accumulate
### Syntax
```ptx
dp4a.atype.btype  d, a, b, c;
.atype = .btype = { .u32, .s32 };
```
### Constraints
- `a`, `b`: 32-bit values holding 4 packed bytes. Computes `d = c + sum(a_byte[i] * b_byte[i])` for i=0..3. Bytes sign/zero-extended per type. Requires `sm_61`+. PTX ISA 5.0+.
### Example
```ptx
dp4a.u32.u32  d, a, b, c;
dp4a.s32.u32  d, a, b, c;  // signed a bytes, unsigned b bytes
```

---

## dp2a -- 2-Way Dot Product Accumulate
### Syntax
```ptx
dp2a.mode.atype.btype  d, a, b, c;
.atype = .btype = { .u32, .s32 };
.mode = { .lo, .hi };
```
### Constraints
- `a`: 2 packed 16-bit values. `b`: 4 packed bytes. `.lo` uses bytes 0..1 of `b`, `.hi` uses bytes 2..3. Computes `d = c + sum(a_half[i] * b_byte[sel+i])`. Requires `sm_61`+. PTX ISA 5.0+.
### Example
```ptx
dp2a.lo.s32.u32  d, a, b, c;
```

---

## lop3 -- Arbitrary 3-Input Logic
### Syntax
```ptx
lop3.b32         d, a, b, c, immLut;
lop3.BoolOp.b32  d|p, a, b, c, immLut, q;
.BoolOp = { .or, .and };
```
### Variants
`immLut` encodes the truth table for `F(a,b,c)`:
```
ta = 0xF0;  tb = 0xCC;  tc = 0xAA;
immLut = F(ta, tb, tc);
```

| Function | immLut |
|----------|--------|
| `a & b & c` | `0x80` |
| `a \| b \| c` | `0xFE` |
| `a & b & ~c` | `0x40` |
| `(a & b \| c) ^ a` | `0x1A` |

### Constraints
- 256 possible operations. Optional `.BoolOp` computes `p = (d != 0) BoolOp q`. `_` allowed as sink for `d`. Requires `sm_50`+. `.BoolOp` requires `sm_70`+. PTX ISA 4.3+.
### Example
```ptx
lop3.b32      d, a, b, c, 0x80;       // d = a & b & c
lop3.or.b32   d|p, a, b, c, 0x3f, q;
```

---

## shf -- Funnel Shift
### Syntax
```ptx
shf.l.mode.b32  d, a, b, c;   // left shift
shf.r.mode.b32  d, a, b, c;   // right shift
.mode = { .clamp, .wrap };
```
### Variants
Shifts the 64-bit value `{b[63:32], a[31:0]}` by amount `c`. `shf.l` writes MSBs to `d`; `shf.r` writes LSBs to `d`.
```
// .clamp: n = min(c, 32)    .wrap: n = c & 0x1f
shf.l:  d = (b << n) | (a >> (32-n))
shf.r:  d = (b << (32-n)) | (a >> n)
```
### Constraints
- Requires `sm_32`+. PTX ISA 3.1+. Use for multi-word shifts and 32-bit rotates (`a == b`).
### Example
```ptx
shf.r.clamp.b32  r1, r0, r0, n;  // rotate right by n
shf.l.clamp.b32  r7, r2, r3, n;  // 128-bit left shift step
```

---

## shl / shr -- Shift Left / Right
### Syntax
```ptx
shl.type  d, a, b;    .type = { .b16, .b32, .b64 };
shr.type  d, a, b;    .type = { .b16, .b32, .b64, .u16, .u32, .u64, .s16, .s32, .s64 };
```
### Constraints
- `b` is always `.u32`. Shifts > register width clamped to N. Signed `shr` fills with sign bit; unsigned/untyped fills with 0. All targets. PTX ISA 1.0+.
### Example
```ptx
shl.b32  q, a, 2;
shr.s32  i, i, 1;   // arithmetic right shift
```

---

## nanosleep -- Thread Suspension
### Syntax
```ptx
nanosleep.u32  t;   // t: register or immediate (nanoseconds)
```
### Constraints
- Duration in `[0, 2*t]`. Max 1 ms. Warp threads may wake together. Requires `sm_70`+. PTX ISA 6.3+.
### Example
```ptx
@!done nanosleep.u32 20;
```

---

## getctarank -- Get CTA Rank of Shared Memory Address
### Syntax
```ptx
getctarank{.shared::cluster}.type  d, a;
.type = { .u32, .u64 };
```
### Constraints
- `d`: 32-bit CTA rank. `a`: shared memory address. Requires `sm_90`+. PTX ISA 7.8+.
### Example
```ptx
getctarank.shared::cluster.u32  rank, addr;
```

---

## setmaxnreg -- Adjust Warp Register Count
### Syntax
```ptx
setmaxnreg.action.sync.aligned.u32  imm-reg-count;
.action = { .inc, .dec };
```
### Constraints
- `imm-reg-count`: 24..256, multiple of 8. `.dec` releases registers; `.inc` requests (blocks until available). All warps in a warpgroup must execute the same instruction. Must synchronize between successive calls. New registers from `.inc` are undefined. Requires `sm_90a`+. PTX ISA 8.0+.
### Example
```ptx
setmaxnreg.dec.sync.aligned.u32 64;
setmaxnreg.inc.sync.aligned.u32 192;
```

---

## Special Registers

### Thread / Block / Grid Identification

| Register | Type | Description |
|----------|------|-------------|
| `%tid.{x,y,z}` | `.u32` | Thread ID within CTA. Range `[0, %ntid-1)` per dim |
| `%ntid.{x,y,z}` | `.u32` | CTA dimensions. Max x,y=1024; z=64 (sm_20+) |
| `%laneid` | `.u32` | Lane within warp (0..WARP_SZ-1) |
| `%warpid` | `.u32` | Warp ID within CTA (may change at runtime) |
| `%nwarpid` | `.u32` | Max warp IDs. `sm_20`+ |
| `%ctaid.{x,y,z}` | `.u32` | CTA ID within grid |
| `%nctaid.{x,y,z}` | `.u32` | Grid dimensions |
| `%smid` | `.u32` | SM identifier (may change at runtime) |
| `%nsmid` | `.u32` | Max SM IDs (not contiguous). `sm_20`+ |
| `%gridid` | `.u64` | Grid launch identifier |

### Cluster Registers (sm_90+)

| Register | Type | Description |
|----------|------|-------------|
| `%clusterid.{x,y,z}` | `.u32` | Cluster ID within grid |
| `%nclusterid.{x,y,z}` | `.u32` | Number of clusters per grid |
| `%cluster_ctaid.{x,y,z}` | `.u32` | CTA ID within cluster |
| `%cluster_nctaid.{x,y,z}` | `.u32` | Number of CTAs per cluster |
| `%cluster_ctarank` | `.u32` | Flat CTA rank within cluster |
| `%cluster_nctarank` | `.u32` | Total CTAs in cluster |
| `%is_explicit_cluster` | `.pred` | Whether cluster launch was explicit |

### Timing and Performance

| Register | Type | Description |
|----------|------|-------------|
| `%clock` | `.u32` | 32-bit cycle counter (wraps) |
| `%clock_hi` | `.u32` | Upper 32 bits of `%clock64`. `sm_20`+ |
| `%clock64` | `.u64` | 64-bit cycle counter. `sm_20`+ |
| `%globaltimer` | `.u64` | 64-bit nanosecond timer. `sm_30`+ |
| `%globaltimer_lo/hi` | `.u32` | Lower/upper 32 bits of `%globaltimer` |

### Shared Memory Size

| Register | Type | Description |
|----------|------|-------------|
| `%total_smem_size` | `.u32` | Total smem (static+dynamic, excl. reserved). `sm_20`+ |
| `%dynamic_smem_size` | `.u32` | Dynamically allocated smem. `sm_20`+ |
| `%aggr_smem_size` | `.u32` | Total smem including reserved region. `sm_90`+ |

### Lane Masks

| Register | Description |
|----------|-------------|
| `%lanemask_eq` | Bit set at own lane position |
| `%lanemask_le` | Bits set at positions <= own lane |
| `%lanemask_lt` | Bits set at positions < own lane |
| `%lanemask_ge` | Bits set at positions >= own lane |
| `%lanemask_gt` | Bits set at positions > own lane |

All `.u32`, require `sm_20`+.

```ptx
mov.u32  %r1, %tid.x;
mov.u32  %r2, %ctaid.x;
mov.u32  %r3, %laneid;
mov.u64  %rd1, %clock64;
mov.u32  %r4, %cluster_ctarank;
mov.u32  %r5, %lanemask_lt;
```
