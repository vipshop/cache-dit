---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/lazy-loading.html
---

# 4.7. Lazy Loading

## 4.7.1. Introduction

Lazy loading reduces program initialization time by waiting to load CUDA modules until they are needed. Lazy loading is particularly effective for programs that only use a small number of the kernels they include, as is common when using libraries. Lazy loading is designed to be invisible to the user when the CUDA programming model is followed. [Potential Hazards](#lazy-loading-potential-hazards) explains this in detail. As of CUDA 12.3 lazy Loading is enabled by default on all platforms, but can be controlled via the `CUDA_MODULE_LOADING` environment variable.

## 4.7.2. Change History

Table 17 Select Lazy Loading Changes by CUDA Version CUDA Version | Change  
---|---  
12.3 | Lazy loading performance improved. Now enabled by default for Windows.  
12.2 | Lazy loading enabled by default for Linux.  
11.7 | Lazy loading first introduced, disabled by default.  
  
## 4.7.3. Requirements for Lazy Loading

Lazy loading is a joint feature of both the CUDA runtime and driver. Lazy loading is only available when the runtime and driver version requirements are satisfied.

### 4.7.3.1. CUDA Runtime Version Requirement

Lazy loading is available starting in CUDA runtime version 11.7. As CUDA runtime is usually linked statically into programs and libraries, only programs and libraries from or compiled with CUDA 11.7+ toolkit will benefit from lazy loading. Libraries compiled using older CUDA runtime versions will load all modules eagerly.

### 4.7.3.2. CUDA Driver Version Requirement

Lazy loading requires driver version 515 or newer. Lazy loading is not available for driver versions older than 515, even when using CUDA toolkit 11.7 or newer.

### 4.7.3.3. Compiler Requirements

Lazy loading does not require any compiler support. Both SASS and PTX compiled with pre-11.7 compilers can be loaded with lazy loading enabled, and will see full benefits of the feature. However, the version 11.7+ CUDA runtime is still required, as described above.

### 4.7.3.4. Kernel Requirements

Lazy loading does not affect modules containing managed variables, which will still be loaded eagerly.

## 4.7.4. Usage

### 4.7.4.1. Enabling & Disabling

Lazy loading is enabled by setting the `CUDA_MODULE_LOADING` environment variable to `LAZY`. Lazy loading can be disabled by setting the `CUDA_MODULE_LOADING` environment variable to `EAGER`. As of CUDA 12.3, lazy loading is enabled by default on all platforms.

### 4.7.4.2. Checking if Lazy Loading is Enabled at Runtime

The `cuModuleGetLoadingMode` API in the CUDA driver API can be used to determine if lazy loading is enabled. Note that CUDA must be initialized before running this function. Sample usage is shown in the snippet below.
    
    
    #include "<cuda.h>"
    #include "<assert.h>"
    #include "<iostream>"
    
    int main() {
            CUmoduleLoadingMode mode;
    
            assert(CUDA_SUCCESS == cuInit(0));
            assert(CUDA_SUCCESS == cuModuleGetLoadingMode(&mode));
    
            std::cout << "CUDA Module Loading Mode is " << ((mode == CU_MODULE_LAZY_LOADING) ? "lazy" : "eager") << std::endl;
    
            return 0;
    }
    

### 4.7.4.3. Forcing a Module to Load Eagerly at Runtime

Loading kernels and variables happens automatically, without any need for explicit loading. Kernels can be loaded explicitly even without executing them by doing the following:

  * The `cuModuleGetFunction()` function will cause a module to be loaded into device memory

  * The `cudaFuncGetAttributes()` function will cause a kernel to be loaded into device memory


Note

`cuModuleLoad()` does not guarantee that a module will be loaded immediately.

## 4.7.5. Potential Hazards

Lazy loading is designed so that it should not require any modifications to applications to use it. That said, there are some caveats, especially when applications are not fully compliant with the CUDA programming model, as described below.

### 4.7.5.1. Impact on Concurrent Kernel Execution

Some programs incorrectly assume that concurrent kernel execution is guaranteed. A deadlock can occur if cross-kernel synchronization is required, but kernel execution has been serialized. To minimize the impact of lazy loading on concurrent kernel execution, do the following:

  * preload all kernels that you hope to execute concurrently prior to launching them or

  * run application with `CUDA_MODULE_LOADING = EAGER` to force loading data eagerly without forcing each function to load eagerly


### 4.7.5.2. Large Memory Allocations

Lazy loading delays memory allocation for CUDA modules from program initialization until closer to execution time. If an application allocates the entire VRAM on startup, CUDA can fail to allocate memory for modules at runtime. Possible solutions:

  * use `cudaMallocAsync()` instead of an allocator that allocates the entire VRAM on startup

  * add some buffer to compensate for the delayed loading of kernels

  * preload all kernels that will be used in the program before trying to initialize the allocator


### 4.7.5.3. Impact on Performance Measurements

Lazy loading may skew performance measurements by moving CUDA module initialization into the measured execution window. To avoid this:

  * do at least one warmup iteration prior to measurement

  * preload the benchmarked kernel prior to launching it
