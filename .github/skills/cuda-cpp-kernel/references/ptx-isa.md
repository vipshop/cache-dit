# PTX ISA Reference

**Related guides:** cuda-runtime.md (high-level API), cuda-driver.md (low-level API)

## Table of Contents

- [Local Documentation](#local-documentation) — 405 markdown files, 2.3MB
- [When to Use PTX Documentation](#when-to-use-ptx-documentation) — Inspecting code, inline PTX, TensorCore ops
- [Quick Search Examples](#quick-search-examples) — WGMMA fragments, TMA swizzling, specific instructions
- [Documentation Structure](#documentation-structure) — Chapter organization
- [Common PTX Concepts](#common-ptx-concepts) — Registers, address spaces, instructions, inline PTX
- [Inspecting PTX](#inspecting-ptx) — Generate and extract PTX from binaries
- [Key TensorCore Sections](#key-tensorcore-sections) — WMMA, WGMMA, TensorCore Gen5
- [Memory Consistency Model](#memory-consistency-model) — Async operations, barriers
- [Quick Reference Workflow](#quick-reference-workflow) — How to navigate docs

## Local Documentation

**Complete PTX ISA 9.1 documentation is available locally at `ptx-docs/`**

The documentation has been converted to markdown with:
- ✅ All tables, code blocks, and formatting preserved
- ✅ 405 files organized by chapter
- ✅ Full searchability with grep/ripgrep
- ✅ Section numbers for precise navigation
- ✅ 1049 diagrams referenced as URLs (NVIDIA CDN)

**Note:** Documentation is local and searchable with grep. Links to online resources provided for reference only.

## When to Use PTX Documentation

Consult PTX reference when:

1. **Inspecting generated code** — Understanding nvcc output via `cuobjdump -ptx`
2. **Writing inline PTX** — Low-level optimizations using `asm volatile`
3. **Debugging code generation** — Verifying compiler behavior
4. **Understanding SASS** — PTX is intermediate, SASS is final, but PTX reveals intent
5. **Using TensorCore operations** — WMMA, WGMMA, TMA instructions
6. **Memory operations** — Async copy, tensor memory, swizzling modes

## Quick Search Examples

### Find Register Fragment Layout for WGMMA

```bash
# Search for wgmma register fragments
grep -r "register fragment" ptx-docs/9-instruction-set/ | grep -i wgmma

# Find specific shape (m64nNk16)
find ptx-docs -name "*wgmma*" -type f
# Then read: ptx-docs/9-instruction-set/9.7.15.5-asynchronous-warpgroup-level-matrix-multiply-accumulate-operation-usingwgmmamma_asyncinstruction.md
```

**Answer location**: Section 9.7.15.5.1.1 documents all WGMMA register fragment layouts for different matrix shapes (m64nNk16, m64nNk8, m64nNk32, m64nNk256).

### Disable TMA Swizzling

```bash
# Find swizzling information
grep -r "swizzle_mode" ptx-docs/9-instruction-set/ | grep -i "no swizzling"
```

**Answer**: Use `tensormap.replace` instruction with `.swizzle_mode` field set to value `0` (No swizzling). See `ptx-docs/9-instruction-set/9.7.9.28-data-movement-and-conversion-instructionstensormapreplace.md`

### Find Tensor Swizzling Modes

```bash
# Find all swizzling documentation
find ptx-docs -name "*swizzl*" -type f
# Read: ptx-docs/5-state-spaces-types-and-variables/5.5.7-swizzling-modes.md
```

### Find Specific Instruction

```bash
# Search for any instruction (e.g., mbarrier)
grep -r "mbarrier.init" ptx-docs/9-instruction-set/

# Or find by section number
find ptx-docs -name "9.7.13.15*"
```

### Find Data Types and Memory Spaces

```bash
# State spaces
grep -r "state space" ptx-docs/5-state-spaces-types-and-variables/

# Fundamental types
find ptx-docs -name "5.2.1*"
```

## Documentation Structure

```
ptx-docs/
├── 1-introduction/                    # PTX overview
├── 2-programming-model/               # Thread hierarchy, memory
├── 3-ptx-machine-model/              # SIMT architecture
├── 4-syntax/                         # PTX syntax rules
├── 5-state-spaces-types-and-variables/ # Memory spaces, data types, tensors
├── 6-instruction-operands/           # Operand types
├── 7-abstracting-the-abi/           # Functions, calling conventions
├── 8-memory-consistency-model/       # Memory ordering, atomics
├── 9-instruction-set/               # Complete instruction reference (213 files!)
│   ├── 9.7.1-*                      # Integer arithmetic
│   ├── 9.7.3-*                      # Floating point
│   ├── 9.7.9-*                      # Data movement (includes TMA)
│   ├── 9.7.14-*                     # WMMA (warp-level MMA)
│   ├── 9.7.15-*                     # WGMMA (warpgroup-level MMA)
│   └── 9.7.16-*                     # TensorCore Gen5 (Hopper)
├── 10-special-registers/            # %tid, %ctaid, %clock64, etc.
├── 11-directives/                   # .version, .target, .entry
├── 12-descriptions-ofpragmastrings/ # Pragma directives
├── 13-release-notes/                # Version history
└── INDEX.md                         # Complete table of contents
```

## Common PTX Concepts

### Registers

```ptx
.reg .b32 %r<10>;     // 10 32-bit registers
.reg .f32 %f<5>;      // 5 32-bit float registers
.reg .b64 %rd<3>;     // 3 64-bit registers
.reg .pred %p<2>;     // 2 predicate registers
```

### Address Spaces

```ptx
.global   // Device global memory
.shared   // Block shared memory
.local    // Thread local memory
.const    // Constant memory
.param    // Kernel parameters
```

### Common Instructions

```ptx
// Arithmetic
add.s32 %r1, %r2, %r3;     // r1 = r2 + r3
mul.lo.s32 %r1, %r2, %r3;  // r1 = (r2 * r3) low bits
mad.lo.s32 %r1, %r2, %r3, %r4;  // r1 = r2*r3 + r4

// Memory
ld.global.f32 %f1, [%rd1];      // Load from global
st.global.f32 [%rd1], %f1;      // Store to global
ld.shared.f32 %f1, [%r1];       // Load from shared

// Control
@%p1 bra target;    // Conditional branch
bar.sync 0;         // Barrier

// Special registers
mov.u32 %r1, %tid.x;    // Thread ID X
mov.u32 %r2, %ctaid.x;  // Block ID X
mov.u32 %r3, %ntid.x;   // Block dim X
```

### Inline PTX in CUDA

```cuda
__device__ int myAdd(int a, int b) {
    int result;
    asm("add.s32 %0, %1, %2;"
        : "=r"(result)
        : "r"(a), "r"(b));
    return result;
}
```

### Constraint Codes

```
r  — 32-bit register
l  — 64-bit register
f  — 32-bit float register
d  — 64-bit float register
n  — immediate integer
```

## Inspecting PTX

```bash
# Generate PTX file
nvcc -ptx program.cu -o program.ptx

# Extract from binary
cuobjdump -ptx ./program > extracted.ptx

# Find specific kernel
cuobjdump -ptx ./program | grep -A 200 ".entry myKernel"

# Show resource usage (registers, shared mem)
cuobjdump -res-usage ./program
```

## Key TensorCore Sections

### WMMA (Warp-Level Matrix Operations)
- **Location**: `ptx-docs/9-instruction-set/9.7.14-*`
- **Use for**: sm_70+ (Volta, Turing, Ampere)
- **Operations**: `wmma.load`, `wmma.mma`, `wmma.store`

### WGMMA (Warpgroup-Level Async MMA)
- **Location**: `ptx-docs/9-instruction-set/9.7.15-*`
- **Use for**: sm_90+ (Hopper)
- **Operations**: `wgmma.mma_async`, register fragment layouts
- **Key file**: `9.7.15.5-asynchronous-warpgroup-level-matrix-multiply-accumulate-operation-usingwgmmamma_asyncinstruction.md`

### TensorCore Gen5 (TMA, TMEM)
- **Location**: `ptx-docs/9-instruction-set/9.7.16-*`
- **Use for**: sm_100+ (Blackwell)
- **Operations**: Tensor memory, TMA operations, specialized MMA

## Memory Consistency Model

For understanding async operations, memory ordering, and barriers:
- **Location**: `ptx-docs/8-memory-consistency-model/`
- **Topics**: Scopes, memory operations, ordering, axioms

## Quick Reference Workflow

1. **Know what you're looking for** → Use grep to find it
2. **Know the section** → Navigate directly by number
3. **Browse instructions** → Check `9-instruction-set/` directory
4. **Need examples** → Look at the instruction's file, has syntax + examples
5. **Verify against HTML** → Cross-reference with online docs if unsure

## External Resources

- Official PTX docs online: https://docs.nvidia.com/cuda/parallel-thread-execution/
- PTX ISA PDF: https://docs.nvidia.com/cuda/pdf/ptx_isa_9.1.pdf
- CUTLASS library: https://github.com/NVIDIA/cutlass (uses PTX heavily)
