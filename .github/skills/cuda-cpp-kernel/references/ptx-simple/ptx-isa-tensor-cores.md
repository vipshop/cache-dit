# PTX ISA 9.1 -- Tensor Core Instructions (mma, wgmma, ldmatrix)

Reference for GPU kernel engineers working with NVIDIA tensor core instructions
in PTX. Covers warp-level `mma`, warpgroup-level `wgmma.mma_async`, and
the `ldmatrix`/`stmatrix` data movement instructions.

---

## 1. Warp-Level `mma.sync` (Section 9.7.14.5.14)

Performs `D = A * B + C` within a single warp (32 threads). All threads must
execute the same instruction (`.sync.aligned`).

### Syntax

```ptx
mma.sync.aligned.shape.alayout.blayout.dtype.atype.btype.ctype  d, a, b, c;
```

For most shapes (m16n8k*), layout is fixed: `.row.col` (A is row-major,
B is column-major). Only the legacy `.m8n8k4` supports arbitrary `.row/.col`
on both operands.

### Shape x Type Table

| Data type | Shapes | Acc (D/C) | Min arch |
|-----------|--------|-----------|----------|
| `.f16` | m8n8k4, m16n8k8, m16n8k16 | `.f16` or `.f32` | sm_70 / sm_75 / sm_80 |
| `.bf16` | m16n8k8, m16n8k16 | `.f32` | sm_80 |
| `.tf32` | m16n8k4, m16n8k8 | `.f32` | sm_80 |
| `.e4m3`/`.e5m2` (FP8) | m16n8k16, m16n8k32 | `.f16` or `.f32` | sm_89 |
| `.e3m2`/`.e2m3`/`.e2m1` | m16n8k32 (with `.kind::f8f6f4`) | `.f32` | sm_120a |
| `.f64` | m8n8k4, m16n8k4, m16n8k8, m16n8k16 | `.f64` | sm_80 / sm_90 |
| `.u8`/`.s8` | m8n8k16, m16n8k16, m16n8k32 | `.s32` | sm_75 / sm_80 |
| `.u4`/`.s4` | m8n8k32, m16n8k32, m16n8k64 | `.s32` | sm_75 / sm_80 |
| `.b1` (xor/and.popc) | m8n8k128, m16n8k128, m16n8k256 | `.s32` | sm_75 / sm_80 |

Block-scaled MMA (`.block_scale`, `.kind::mxf4`, `.kind::mxf8f6f4`) with
scale matrices requires sm_120a.

### Type constraints

- m16n8k8: `.dtype` == `.ctype`, `.atype` == `.btype`.
- m16n8k16, m16n8k32: `.dtype` == `.ctype`.

### Example

```ptx
.reg .f16x2 %Ra<4>, %Rb<2>, %Rc<2>, %Rd<2>;
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
  {%Rd0, %Rd1},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1};

.reg .b32 %Ra<4>, %Rb<2>;
.reg .f32 %Rc<4>, %Rd<4>;
mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32
  {%Rd0, %Rd1, %Rd2, %Rd3},
  {%Ra0, %Ra1, %Ra2, %Ra3},
  {%Rb0, %Rb1},
  {%Rc0, %Rc1, %Rc2, %Rc3};
```

### Fragment layout (m16n8k16, f16)

Each thread holds a fragment determined by `groupID = laneid >> 2` and
`threadID_in_group = laneid % 4`. The C/D accumulator fragment contains
elements at rows `groupID` (for c0,c1) and `groupID+8` (for c2,c3),
with columns `threadID_in_group * 2 + (i & 0x1)`.

---

## 2. `ldmatrix` / `stmatrix` (Sections 9.7.14.5.15-16)

Warp-collective loads/stores of 8x8 matrices from/to shared memory, laid out
for direct use as `mma` operands.

### ldmatrix syntax

```ptx
ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type  r, [p];

.shape = {.m8n8, .m16n16, .m8n16}
.num   = {.x1, .x2, .x4}       // number of matrices
.type  = {.b16, .b8}
.ss    = {.shared{::cta}}
```

### stmatrix syntax

```ptx
stmatrix.sync.aligned.shape.num{.trans}{.ss}.type  [p], r;

.shape = {.m8n8, .m16n8}
.num   = {.x1, .x2, .x4}
.type  = {.b16, .b8}
```

### Key details

| Feature | ldmatrix | stmatrix |
|---------|----------|----------|
| Min arch | sm_75 | sm_90 |
| 16-bit shape | m8n8 (x1/x2/x4) | m8n8 (x1/x2/x4) |
| 8-bit shape | m16n16 (x1/x2), m8n16 | m16n8 (x1/x2/x4) |
| `.trans` | optional (mandatory for m16n16) | optional (mandatory for m16n8) |

**Thread-to-address mapping**: threads 0-7 provide addresses for matrix 0,
threads 8-15 for matrix 1, etc. (for `.x1`, only threads 0-7 are used).
Each address is the start of an 8-element row (16 bytes for .b16).

### Example

```ptx
// Load four 8x8 matrices of f16 from shared memory
.reg .b64 addr;
.reg .b32 d<4>;
ldmatrix.sync.aligned.m8n8.x4.b16 {d0, d1, d2, d3}, [addr];

// Store one 8x8 matrix transposed
stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [addr], {d0};
```

---

## 3. Warpgroup-Level `wgmma.mma_async` (Section 9.7.15.5.2)

Asynchronous MMA across a **warpgroup** (4 consecutive warps = 128 threads).
Operates on much larger tiles than warp-level `mma`. Requires **sm_90a**.

### Syntax

```ptx
// A from shared memory (descriptor):
wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
  d, a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b{, imm-trans-a, imm-trans-b};

// A from registers:
wgmma.mma_async.sync.aligned.shape.dtype.atype.btype
  d, a, b-desc, scale-d, imm-scale-a, imm-scale-b{, imm-trans-b};
```

- `scale-d`: predicate. If false, computes `D = A*B` (no accumulate).
- `imm-scale-a/b`: 1 or -1 (negate elements of A/B).
- `imm-trans-a/b`: 0 or 1 (transpose, only for `.f16`/`.bf16` descriptor variants).

### Shape x Type Table

All shapes have M=64. N ranges from 8 to 256 in steps of 8. K depends on type.

| atype/btype | K | Accumulator (D) | N range |
|-------------|---|-----------------|---------|
| `.f16` | 16 | `.f16` or `.f32` | 8..256 (step 8) |
| `.bf16` | 16 | `.f32` | 8..256 (step 8) |
| `.tf32` | 8 | `.f32` | 8..256 (step 8) |
| `.e4m3`/`.e5m2` (FP8) | 32 | `.f16` or `.f32` | 8..256 (step 8) |
| `.u8`/`.s8` | 32 | `.s32` | 8..256 (step 16) |
| `.b1` (and.popc) | 256 | `.s32` | 8..256 (step 16) |

Matrix B **must** be in shared memory (via descriptor). Matrix A can be in
registers or shared memory (via descriptor).

### Matrix Descriptor Format (64-bit)

| Bits | Field |
|------|-------|
| 13-0 | `encode(start_address)` |
| 29-16 | `encode(leading_dim_byte_offset)` |
| 45-32 | `encode(stride_dim_byte_offset)` |
| 51-49 | Base offset (for swizzle alignment) |
| 63-62 | Swizzle mode: 0=none, 1=128B, 2=64B, 3=32B |

Where `encode(x) = (x & 0x3FFFF) >> 4`. Shared memory addresses must be
16-byte aligned.

### Example

```ptx
.reg .f32   f32d<4>;
.reg .f16x2 f16a<4>;
.reg .b64   descA, descB;
.reg .pred  scaleD;

// A from registers, B from descriptor
wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16
  {f32d0, f32d1, f32d2, f32d3},
  {f16a0, f16a1, f16a2, f16a3},
  descB,
  1, -1, -1, 1;       // scaleD=true, negate A, negate B, transpose B

// Both from descriptors (FP8)
wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e5m2
  {f32d0, ..., f32d63},
  descA, descB,
  scaleD, 1, 1;
```

---

## 4. wgmma Lifecycle: fence / commit_group / wait_group

The `wgmma.mma_async` instruction runs in the **async proxy**. You must bracket
it with synchronization instructions:

```ptx
// 1. Fence: orders prior register writes before wgmma reads them
wgmma.fence.sync.aligned;

// 2. Issue one or more MMAs
wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 ...;
wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 ...;

// 3. Commit: batch all pending mma_async ops into a "wgmma-group"
wgmma.commit_group.sync.aligned;

// 4. Wait: block until N or fewer groups remain pending
wgmma.wait_group.sync.aligned N;
//   N=0 means wait for ALL groups to complete
```

### Rules

- **fence** is required before the first `mma_async` and whenever you modify
  registers (accumulator or A fragments) between `mma_async` calls.
  Exception: back-to-back `mma_async` with same-shape accumulators do not need
  an intervening fence.
- **commit_group** batches all uncommitted `mma_async` ops. An empty commit
  creates an empty group.
- **wait_group N** waits until at most N groups are pending. Accessing
  accumulator registers before the corresponding group has been waited on is
  undefined behavior.
- All three instructions require `.sync.aligned` -- all threads in the
  warpgroup must execute them uniformly.
- An implicit `fence.proxy.async` makes completed results visible to the
  generic proxy after `wait_group` returns.

### Pipeline pattern

```ptx
// Initialize accumulators
mov.f32 d0, 0.0;  mov.f32 d1, 0.0; ...

wgmma.fence.sync.aligned;

// K-loop body: issue mma, commit, optionally wait
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
  {d0, ..., d63}, descA, descB, 1, 1, 1, 0, 0;
wgmma.commit_group.sync.aligned;

// ... next iteration can overlap with prior group ...

wgmma.wait_group.sync.aligned 0;     // drain all
// Now safe to read d0..d63
```

---

## 5. Sparse MMA (`mma.sp` and `wgmma.mma_async.sp`)

Both warp-level and warpgroup-level MMA support 2:4 structured sparsity on
matrix A. The sparse variants double the K dimension for the same register
cost:

| Level | Dense shape example | Sparse shape |
|-------|-------------------|--------------|
| mma | m16n8k16 (f16) | m16n8k32.sp |
| wgmma | m64nNk16 (f16/bf16) | m64nNk32.sp |
| wgmma | m64nNk32 (e4m3/e5m2) | m64nNk64.sp |

Sparse variants require a sparsity metadata register (`sp-meta`, 32-bit) and
a selector constant (`sp-sel`, 0..3) that identifies which metadata
quadrant to use.

---

## Architecture Summary

| Instruction | Minimum arch | Notes |
|------------|-------------|-------|
| `mma.sync` (f16, m8n8k4) | sm_70 | Legacy, optimized for Volta only |
| `mma.sync` (f16 m16n8k8, int8/4/1) | sm_75 | Turing |
| `mma.sync` (f16 m16n8k16, bf16, tf32, f64, int larger shapes) | sm_80 | Ampere |
| `mma.sync` (e4m3/e5m2 FP8) | sm_89 | Ada Lovelace |
| `mma.sync` (e3m2/e2m3/e2m1, block_scale) | sm_120a | Next-gen |
| `ldmatrix` (.b16, m8n8) | sm_75 | |
| `stmatrix` (.b16, m8n8) | sm_90 | Hopper |
| `wgmma.mma_async` | sm_90a | Hopper (warpgroup) |
| `wgmma.fence/commit/wait` | sm_90a | |
