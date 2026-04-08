# NVTX Patterns Reference

## Table of Contents

- [Overview](#overview) — Custom markers for profiling
- [Setup](#setup) — Include header, compilation, profiling
- [Basic API](#basic-api) — Range push/pop, colors, marks, named threads
- [Common Patterns](#common-patterns) — Phase tracking, iteration tracking, kernel wrapper, CPU vs GPU time, color-coded categories, RAII wrapper, conditional profiling
- [Analysis with nsys](#analysis-with-nsys) — NVTX summary, correlation with CUDA
- [Best Practices](#best-practices) — Consistency, hierarchy, overhead considerations
- [Overhead Considerations](#overhead-considerations) — Performance impact

## Overview

NVIDIA Tools Extension (NVTX) provides custom markers for profiling. Use when kernel-level granularity isn't enough.

## Setup

### Include Header

```cuda
#include <nvtx3/nvToolsExt.h>
```

### Compilation

```bash
nvcc program.cu -lnvToolsExt -o program
```

### Profiling

```bash
nsys profile --trace=cuda,nvtx -o report ./program
nsys stats report.nsys-rep --report nvtx_sum
```

## Basic API

### Range Push/Pop

```cuda
nvtxRangePush("Region Name");
// ... code ...
nvtxRangePop();
```

Ranges can nest:
```cuda
nvtxRangePush("Outer");
  nvtxRangePush("Inner");
  // ...
  nvtxRangePop();
nvtxRangePop();
```

### Range with Color

```cuda
nvtxEventAttributes_t attr = {0};
attr.version = NVTX_VERSION;
attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
attr.colorType = NVTX_COLOR_ARGB;
attr.color = 0xFF00FF00;  // Green
attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
attr.message.ascii = "Green Region";
nvtxRangePushEx(&attr);
// ...
nvtxRangePop();
```

### Marks (Instant Events)

```cuda
nvtxMark("Checkpoint reached");
```

### Named Threads

```cuda
nvtxNameOsThread(pthread_self(), "Worker Thread");
```

## Common Patterns

### Pattern: Phase Tracking

```cpp
void runPipeline() {
    nvtxRangePush("Pipeline");
    
    nvtxRangePush("Load Data");
    loadData();
    nvtxRangePop();
    
    nvtxRangePush("Preprocess");
    preprocess();
    nvtxRangePop();
    
    nvtxRangePush("Inference");
    inference();
    nvtxRangePop();
    
    nvtxRangePush("Postprocess");
    postprocess();
    nvtxRangePop();
    
    nvtxRangePop();
}
```

### Pattern: Iteration Tracking

```cpp
for (int i = 0; i < iterations; i++) {
    char name[64];
    snprintf(name, sizeof(name), "Iteration %d", i);
    nvtxRangePush(name);
    
    processIteration(i);
    
    nvtxRangePop();
}
```

### Pattern: Kernel Wrapper

```cpp
void launchMyKernel(float* data, int n) {
    nvtxRangePush("MyKernel");
    
    myKernel<<<grid, block>>>(data, n);
    cudaDeviceSynchronize();  // Include sync in range
    
    nvtxRangePop();
}
```

### Pattern: CPU vs GPU Time

```cpp
void compute() {
    nvtxRangePush("CPU Prep");
    prepareData();
    nvtxRangePop();
    
    nvtxRangePush("GPU Compute");
    kernel<<<grid, block>>>(data);
    cudaDeviceSynchronize();
    nvtxRangePop();
    
    nvtxRangePush("CPU Post");
    processResults();
    nvtxRangePop();
}
```

### Pattern: Color-Coded Categories

```cpp
// Define colors for different categories
#define COLOR_MEMORY  0xFFFF0000  // Red
#define COLOR_COMPUTE 0xFF00FF00  // Green
#define COLOR_IO      0xFF0000FF  // Blue

void pushColoredRange(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

void process() {
    pushColoredRange("Allocate", COLOR_MEMORY);
    allocate();
    nvtxRangePop();
    
    pushColoredRange("Compute", COLOR_COMPUTE);
    compute();
    nvtxRangePop();
    
    pushColoredRange("Save", COLOR_IO);
    save();
    nvtxRangePop();
}
```

### Pattern: RAII Wrapper (C++)

```cpp
class NvtxRange {
public:
    NvtxRange(const char* name) { nvtxRangePush(name); }
    ~NvtxRange() { nvtxRangePop(); }
    
    // Non-copyable
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

// Usage
void process() {
    NvtxRange range("Process");
    // ... automatically pops when scope exits
}
```

### Pattern: Conditional Profiling

```cpp
#ifdef ENABLE_NVTX
#define NVTX_PUSH(name) nvtxRangePush(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name)
#define NVTX_POP()
#endif

void process() {
    NVTX_PUSH("Process");
    // ...
    NVTX_POP();
}
```

Compile with:
```bash
nvcc -DENABLE_NVTX program.cu -lnvToolsExt -o program
```

## Analysis with nsys

### Get NVTX Summary

```bash
nsys profile --trace=cuda,nvtx -o report ./program
nsys stats report.nsys-rep --report nvtx_sum
```

Output shows:
- Total time per named range
- Instance count
- Average/min/max duration
- Percentage of total time

### Correlate with CUDA

```bash
nsys stats report.nsys-rep --report nvtx_sum --report cuda_gpu_kern_sum
```

Compare NVTX phases with kernel execution time.

### Export for Analysis

```bash
nsys stats report.nsys-rep --report nvtx_sum --format csv > nvtx.csv
```

## Best Practices

1. **Be consistent** — Use same naming convention throughout
2. **Don't over-instrument** — Too many ranges add overhead
3. **Include sync in ranges** — When measuring GPU work, include `cudaDeviceSynchronize()`
4. **Use hierarchy** — Nest ranges to show structure
5. **Color-code categories** — Makes timeline easier to read
6. **Make conditional** — Use macros to disable in production

## Overhead Considerations

NVTX has minimal overhead, but:
- Avoid in tight loops
- Don't create millions of ranges
- Keep names short (they're stored)
- Use conditional compilation for production builds
