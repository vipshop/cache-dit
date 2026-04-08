# CUDA Runtime API Reference

**Related guides:** ptx-isa.md (instruction-level), cuda-driver.md (low-level alternative)

## Table of Contents

- [Local Documentation](#local-documentation) — 107 markdown files, 0.9MB
- [When to Use CUDA Runtime Documentation](#when-to-use-cuda-runtime-documentation) — Error codes, API details, device properties
- [Quick Search Examples](#quick-search-examples) — Error codes, cudaDeviceProp, contexts, streams, memory pools
- [Documentation Structure](#documentation-structure) — Modules and data structures organization
- [Common Use Cases](#common-use-cases) — Error handling, device properties, streams, memory
- [Function Documentation Format](#function-documentation-format) — How functions are documented
- [Data Structure Documentation Format](#data-structure-documentation-format) — How structs are documented
- [Module Organization](#module-organization) — Core device & memory, execution control, advanced features, interop, utilities
- [Quick Reference Workflow](#quick-reference-workflow) — How to find information
- [Common Questions Answered](#common-questions-answered) — Practical search examples

## Local Documentation

**Complete CUDA Runtime API 13.1 documentation is available locally at `cuda-runtime-docs/`**

The documentation has been converted to markdown with:
- ✅ All function signatures, parameters, and return values preserved
- ✅ 107 files organized by module and data structures (0.9 MB)
- ✅ Full searchability with grep/ripgrep
- ✅ Type and function names preserved (redundant URLs removed)
- ✅ Detailed descriptions and notes
- ✅ Navigation, duplicate content, URLs, and boilerplate removed (83% size reduction)

**Note:** Documentation is local and searchable with grep. Links to online resources provided for reference only.

## When to Use CUDA Runtime Documentation

Consult Runtime API reference when:

1. **Understanding error codes** — Looking up meaning of `cudaError_t` values
2. **API function details** — Parameters, return values, behavior of CUDA functions
3. **Data structures** — Fields and usage of structs like `cudaDeviceProp`, `cudaMemcpy3DParms`
4. **Context and stream behavior** — Understanding default context, stream ordering, synchronization
5. **Memory management** — Malloc variants, unified memory, memory pools, async operations
6. **Device management** — Querying device properties, setting device, limits
7. **Graph operations** — CUDA graphs, nodes, instantiation
8. **Interoperability** — OpenGL, Direct3D, external resources

## Quick Search Examples

### Look Up Error Code Meaning

```bash
# Find what cudaErrorInvalidValue means
grep -r "cudaErrorInvalidValue" cuda-runtime-docs/
```

**Answer location**: Error codes are documented in `modules/group__cudart__error.md` with descriptions and cross-references.

### Find cudaDeviceProp Fields

```bash
# Find the cudaDeviceProp structure documentation
grep -r "struct cudaDeviceProp" cuda-runtime-docs/data-structures/
# Read: cuda-runtime-docs/data-structures/structcudadeviceprop.md
```

**Answer**: All device property fields (maxThreadsPerBlock, sharedMemPerBlock, etc.) are documented with types and descriptions.

### Understanding Default Context

```bash
# Search for default context documentation
grep -ri "default context" cuda-runtime-docs/modules/
```

**Answer location**: Context behavior is documented in modules like `group__cudart__execution__context.md` and `group__cudart__driver.md`.

### Find Stream Synchronization Behavior

```bash
# Find stream sync documentation
grep -A 20 "cudaStreamSynchronize" cuda-runtime-docs/modules/group__cudart__stream.md
```

**Answer**: Full function documentation with parameters, return values, and synchronization semantics.

### Find Memory Pool Functions

```bash
# Search for memory pool API
ls cuda-runtime-docs/modules/*pool*.md
# Read: cuda-runtime-docs/modules/group__cudart__memory__pools.md
```

## Documentation Structure

```
cuda-runtime-docs/
├── modules/                               # 41 API module files
│   ├── group__cudart__device.md          # Device management
│   ├── group__cudart__memory.md          # Memory management (491KB!)
│   ├── group__cudart__stream.md          # Stream management
│   ├── group__cudart__event.md           # Event management
│   ├── group__cudart__execution.md       # Execution control
│   ├── group__cudart__error.md           # Error handling
│   ├── group__cudart__graph.md           # CUDA graphs
│   ├── group__cudart__occupancy.md       # Occupancy calculator
│   ├── group__cudart__memory__pools.md   # Stream-ordered allocator
│   ├── group__cudart__unified.md         # Unified addressing
│   ├── group__cudart__peer.md            # Peer device access
│   └── ...                                # Graphics interop, textures, etc.
├── data-structures/                       # 66 struct/union files
│   ├── structcudadeviceprop.md           # Device properties
│   ├── structcudamemcpy3dparms.md        # 3D memcpy parameters
│   ├── structcudalaunchconfig__t.md      # Launch configuration
│   ├── structcudapointerattributes.md    # Pointer query results
│   └── ...
└── INDEX.md                               # Complete table of contents
```

## Common Use Cases

### Error Handling

```cuda
cudaError_t err = cudaMalloc(&ptr, size);
if (err != cudaSuccess) {
    // What does this error mean?
    // Search: grep "cudaErrorMemoryAllocation" cuda-runtime-docs/
    printf("Error: %s\n", cudaGetErrorString(err));
}
```

**Documentation**: `modules/group__cudart__error.md` has all error codes and error handling functions.

### Device Properties

```cuda
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);
// What fields are available in prop?
// Read: data-structures/structcudadeviceprop.md
```

**Documentation**: `data-structures/structcudadeviceprop.md` lists all ~80 fields with types and descriptions.

### Stream Behavior

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);
// Is this stream synchronizing or non-blocking?
// Search: grep -A 10 "cudaStreamCreate" cuda-runtime-docs/
```

**Documentation**: `modules/group__cudart__stream.md` documents all stream functions and their synchronization behavior.

### Memory Management

```cuda
// What's the difference between cudaMalloc, cudaMallocManaged, cudaMallocAsync?
// Search: grep -A 5 "cudaMallocManaged\|cudaMallocAsync" cuda-runtime-docs/
```

**Documentation**: `modules/group__cudart__memory.md` (491KB) has comprehensive memory API documentation.

## Function Documentation Format

Each function is documented with:

```
__host__ return_type functionName( parameters )

Brief description.

Parameters:
  param1 - Description
  param2 - Description

Returns:
  cudaSuccess, cudaErrorInvalidValue, ... (with links)

Description:
  Detailed explanation of behavior, constraints, and notes.

See also:
  Related functions
```

## Data Structure Documentation Format

Each struct/union is documented with:

```
struct structName

Brief description.

Public Members:
  type fieldName - Description
  type fieldName - Description
  ...

Detailed Description:
  Usage information and notes.
```

## Module Organization

Modules are organized by functional area:

1. **Core Device & Memory**
   - Device Management (`group__cudart__device.md`)
   - Memory Management (`group__cudart__memory.md`)
   - Unified Addressing (`group__cudart__unified.md`)
   - Memory Pools (`group__cudart__memory__pools.md`)

2. **Execution Control**
   - Stream Management (`group__cudart__stream.md`)
   - Event Management (`group__cudart__event.md`)
   - Execution Control (`group__cudart__execution.md`)
   - Occupancy (`group__cudart__occupancy.md`)

3. **Advanced Features**
   - Graph Management (`group__cudart__graph.md`)
   - External Resources (`group__cudart__extres__interop.md`)
   - Peer Access (`group__cudart__peer.md`)
   - Execution Context (`group__cudart__execution__context.md`)

4. **Interoperability**
   - OpenGL (`group__cudart__opengl.md`)
   - Direct3D 9/10/11 (`group__cudart__d3d*.md`)
   - EGL (`group__cudart__egl.md`)
   - VDPAU (`group__cudart__vdpau.md`)

5. **Utilities**
   - Error Handling (`group__cudart__error.md`)
   - Version Management (`group__cudart____version.md`)
   - Profiler Control (`group__cudart__profiler.md`)
   - Driver Interaction (`group__cudart__driver.md`)

## Quick Reference Workflow

1. **Know the function name** → grep to find it
2. **Know the module** → Read the specific module file
3. **Need a struct** → Check `data-structures/` directory
4. **Browse all functions** → Check `INDEX.md` for complete list
5. **Error code lookup** → Search in `modules/group__cudart__error.md`

## Common Questions Answered

### "What error code is -1?"
```bash
grep -r "cudaError" cuda-runtime-docs/modules/group__cudart__error.md | grep -- "-1\|enum"
```

### "What does cudaMemcpyAsync do differently?"
```bash
grep -A 20 "cudaMemcpyAsync" cuda-runtime-docs/modules/group__cudart__memory.md
```

### "What fields are in cudaDeviceProp?"
```bash
cat cuda-runtime-docs/data-structures/structcudadeviceprop.md
```

### "How do I create a stream?"
```bash
grep -A 10 "cudaStreamCreate" cuda-runtime-docs/modules/group__cudart__stream.md
```

### "What's the default context?"
```bash
grep -ri "primary context\|default context" cuda-runtime-docs/modules/
```

### "What memory functions are async?"
```bash
grep -r "cudaMemcpyAsync\|cudaMallocAsync\|cudaFreeAsync" cuda-runtime-docs/modules/
```

## External Resources

- Official CUDA Runtime API docs: https://docs.nvidia.com/cuda/cuda-runtime-api/
- CUDA C Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA C Best Practices: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
