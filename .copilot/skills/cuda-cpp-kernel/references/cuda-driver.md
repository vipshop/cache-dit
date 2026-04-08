# CUDA Driver API Reference

**Related guides:** cuda-runtime.md (high-level alternative), ptx-isa.md (instruction-level)

## Table of Contents

- [Local Documentation](#local-documentation) — 128 markdown files, 0.8MB
- [When to Use CUDA Driver API Documentation](#when-to-use-cuda-driver-api-documentation) — Context management, module loading, virtual memory
- [Quick Search Examples](#quick-search-examples) — Error codes, contexts, module loading, virtual memory
- [Driver API vs Runtime API](#driver-api-vs-runtime-api) — Key differences and when to use each
- [Documentation Structure](#documentation-structure) — Modules and data structures organization
- [Notable Large Files](#notable-large-files) — Memory, graph, types, context, stream, exec
- [Search Tips](#search-tips) — Function names, error codes, types, structs
- [Common Workflows](#common-workflows) — Loading kernels, virtual memory, graph programming
- [Troubleshooting](#troubleshooting) — Common errors and fixes

## Local Documentation

**Complete CUDA Driver API 13.1 documentation is available locally at `cuda-driver-docs/`**

The documentation has been converted to markdown with:
- ✅ All function signatures, parameters, and return values preserved
- ✅ 128 files organized by module and data structures (0.8 MB)
- ✅ Full searchability with grep/ripgrep
- ✅ Type and function names preserved (redundant URLs removed)
- ✅ Detailed descriptions and notes
- ✅ Navigation, duplicate TOC, "See also" sections, URLs, and footer removed (76% size reduction)

**Note:** Documentation is local and searchable with grep. Links to online resources provided for reference only.

## When to Use CUDA Driver API Documentation

Consult Driver API reference when:

1. **Low-level context management** — Explicit context creation, switching, and destruction
2. **Module and function loading** — Loading PTX/CUBIN modules, getting kernel functions
3. **Driver-level error codes** — Looking up meaning of `CUresult` values (CUDA_SUCCESS, etc.)
4. **Virtual memory management** — cuMemMap, cuMemAddressReserve, fine-grained memory control
5. **Advanced features** — Green contexts, multicast, tensor maps, checkpointing
6. **Interoperability** — OpenGL, Direct3D 9/10/11, VDPAU, EGL integration
7. **Graph programming** — Advanced graph node types and conditional graphs
8. **Understanding Runtime internals** — Runtime API is built on top of Driver API

## Quick Search Examples

### Look Up Error Code Meaning

```bash
# Find what CUDA_ERROR_INVALID_VALUE means
grep -r "CUDA_ERROR_INVALID_VALUE" cuda-driver-docs/
```

**Answer location**: Error codes are documented in `modules/group__cuda__error.md` and throughout function documentation.

### Find Context Management Functions

```bash
# Find context API functions
cat cuda-driver-docs/modules/group__cuda__ctx.md | grep -A 5 "cuCtxCreate"
```

**Answer**: Full cuCtxCreate, cuCtxDestroy, cuCtxPushCurrent, cuCtxPopCurrent documentation with parameters and behavior.

### Understanding Module Loading

```bash
# Search for module loading
grep -A 20 "cuModuleLoad" cuda-driver-docs/modules/group__cuda__module.md
```

**Answer**: Documentation for cuModuleLoad, cuModuleLoadData, cuModuleGetFunction, and JIT compilation.

### Find Virtual Memory API

```bash
# Find virtual memory management
ls cuda-driver-docs/modules/*va*.md
# Read: cuda-driver-docs/modules/group__cuda__va.md
```

**Answer**: Complete virtual memory API including cuMemMap, cuMemAddressReserve, cuMemCreate for fine-grained control.

### Find CUdeviceptr vs void* Differences

```bash
# Search for device pointer documentation
grep -r "CUdeviceptr" cuda-driver-docs/modules/group__cuda__types.md
```

**Answer**: Driver API uses CUdeviceptr (unsigned integer) vs Runtime API's void* pointers.

## Driver API vs Runtime API

**Key differences** documented in the files:

- **Runtime API** (`cudaXxx`) — Higher-level, single-context, implicit initialization
- **Driver API** (`cuXxx`) — Lower-level, multi-context, explicit initialization required

**When to use Driver API:**
- Need explicit context control (multiple devices, context sharing)
- Loading PTX/CUBIN modules at runtime
- Building tools/libraries that need low-level control
- Interoperating with graphics APIs

**When to use Runtime API:**
- Standard CUDA programming (99% of use cases)
- Single-context applications
- Simpler API with less boilerplate

## Documentation Structure

```
cuda-driver-docs/
├── modules/                                    # 50 API module files
│   ├── group__cuda__types.md                  # Data types (242KB!)
│   ├── group__cuda__error.md                  # Error handling
│   ├── group__cuda__initialize.md             # cuInit
│   ├── group__cuda__device.md                 # Device management
│   ├── group__cuda__ctx.md                    # Context management (119KB)
│   ├── group__cuda__module.md                 # Module loading
│   ├── group__cuda__mem.md                    # Memory management (715KB!)
│   ├── group__cuda__va.md                     # Virtual memory
│   ├── group__cuda__malloc__async.md          # Stream-ordered allocator
│   ├── group__cuda__stream.md                 # Stream management (120KB)
│   ├── group__cuda__event.md                  # Event management
│   ├── group__cuda__exec.md                   # Kernel execution (115KB)
│   ├── group__cuda__graph.md                  # CUDA graphs (370KB!)
│   ├── group__cuda__green__contexts.md        # Green contexts (79KB)
│   ├── group__cuda__checkpoint.md             # Checkpointing
│   ├── group__cuda__multicast.md              # Multicast
│   ├── group__cuda__tensor__memory.md         # Tensor maps
│   └── ...                                     # Interop, textures, etc.
├── data-structures/                            # 80 struct files
│   ├── structcudevprop__v1.md                 # Device properties
│   ├── structcuda__memcpy3d__v2.md            # 3D memcpy params
│   ├── structcuda__kernel__node__params__v3.md # Kernel node config
│   ├── structcumemallocationprop__v1.md       # Virtual memory props
│   └── ...
└── INDEX.md                                    # Complete table of contents
```

## Notable Large Files

The following files contain extensive API documentation:

1. **group__cuda__mem.md** (715 KB) — Complete memory management API
2. **group__cuda__graph.md** (370 KB) — CUDA graph programming
3. **group__cuda__types.md** (242 KB) — All type definitions and enums
4. **group__cuda__ctx.md** (119 KB) — Context management
5. **group__cuda__stream.md** (120 KB) — Stream operations
6. **group__cuda__exec.md** (115 KB) — Kernel execution control

## Search Tips

1. **Function names**: Driver API uses `cuXxx` (camelCase after cu)
   ```bash
   grep "cuMemAlloc" cuda-driver-docs/modules/group__cuda__mem.md
   ```

2. **Error codes**: Start with `CUDA_ERROR_`
   ```bash
   grep -r "CUDA_ERROR_LAUNCH_FAILED" cuda-driver-docs/
   ```

3. **Types**: Start with `CU` (e.g., CUdevice, CUcontext, CUstream)
   ```bash
   grep "CUcontext" cuda-driver-docs/modules/group__cuda__types.md
   ```

4. **Structs**: Usually `CUDA_` prefix or `CU` prefix
   ```bash
   ls cuda-driver-docs/data-structures/structcuda__*.md
   ls cuda-driver-docs/data-structures/structcu*.md
   ```

## Common Workflows

### Loading and Running a Kernel

```bash
# 1. Initialization
grep "cuInit" cuda-driver-docs/modules/group__cuda__initialize.md

# 2. Get device and create context
grep "cuDeviceGet" cuda-driver-docs/modules/group__cuda__device.md
grep "cuCtxCreate" cuda-driver-docs/modules/group__cuda__ctx.md

# 3. Load module
grep "cuModuleLoad" cuda-driver-docs/modules/group__cuda__module.md

# 4. Get function
grep "cuModuleGetFunction" cuda-driver-docs/modules/group__cuda__module.md

# 5. Launch kernel
grep "cuLaunchKernel" cuda-driver-docs/modules/group__cuda__exec.md
```

### Virtual Memory Management

```bash
# Reserve address range
grep "cuMemAddressReserve" cuda-driver-docs/modules/group__cuda__va.md

# Create physical memory
grep "cuMemCreate" cuda-driver-docs/modules/group__cuda__va.md

# Map memory to address
grep "cuMemMap" cuda-driver-docs/modules/group__cuda__va.md

# Set access permissions
grep "cuMemSetAccess" cuda-driver-docs/modules/group__cuda__va.md
```

### Graph Programming

```bash
# Create graph
grep "cuGraphCreate" cuda-driver-docs/modules/group__cuda__graph.md

# Add nodes
grep "cuGraphAddKernelNode" cuda-driver-docs/modules/group__cuda__graph.md

# Instantiate and launch
grep "cuGraphInstantiate" cuda-driver-docs/modules/group__cuda__graph.md
grep "cuGraphLaunch" cuda-driver-docs/modules/group__cuda__graph.md
```

## Troubleshooting

### "CUDA_ERROR_NOT_INITIALIZED"
- Search: `grep "CUDA_ERROR_NOT_INITIALIZED" cuda-driver-docs/`
- **Cause**: cuInit() was not called
- **Fix**: Call cuInit(0) before any Driver API calls

### "CUDA_ERROR_INVALID_CONTEXT"
- Search: `grep "CUDA_ERROR_INVALID_CONTEXT" cuda-driver-docs/`
- **Cause**: No context is current or context was destroyed
- **Fix**: Use cuCtxPushCurrent() or cuCtxSetCurrent()

### "CUDA_ERROR_LAUNCH_FAILED"
- Search: `grep "CUDA_ERROR_LAUNCH_FAILED" cuda-driver-docs/`
- **Cause**: Kernel launch error (many possible reasons)
- **Fix**: Check kernel parameters, shared memory size, register usage

## Version Information

- **CUDA Toolkit Version**: 13.1
- **Documentation Date**: December 4, 2025
- **Total Size**: 0.8 MB (76% reduction from 3.6 MB raw)
- **Files**: 50 modules + 80 data structures + 1 index
- **Source**: https://docs.nvidia.com/cuda/cuda-driver-api/
