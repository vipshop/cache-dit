---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
---

# 5.4. C/C++ Language Extensions

## 5.4.1. Function and Variable Annotations

### 5.4.1.1. Execution Space Specifiers

The execution space specifiers `__host__`, `__device__`, and `__global__` indicate whether a function executes on the host or the device.

Table 38 Execution Space Specifier Execution Space Specifier | Executed on | Callable from  
---|---|---  
Host | Device | Host | Device  
`__host__`, no specifier | ✅ | ❌ | ✅ | ❌  
`__device__` | ❌ | ✅ | ❌ | ✅  
`__global__` | ❌ | ✅ | ✅ | ✅  
`__host__ __device__` | ✅ | ✅ | ✅ | ✅  
  
* * *

Constraints for `__global__` functions:

  * Must return `void`.

  * Cannot be a member of a `class`, `struct`, or `union`.

  * Requires an execution configuration as described in [Kernel Configuration](#execution-configuration).

  * Does not support recursion.

  * Refer to `__global__` [function parameters](cpp-language-support.html#global-function-parameters) for additional restrictions.


Calls to a `__global__` function are asynchronous. They return to the host thread before the device completes execution.

* * *

Functions declared with `__host__ __device__` are compiled for both the host and the device. The `__CUDA_ARCH__` [macro](#cuda-arch-macro) can be used to differentiate host and device code paths:
    
    
    __host__ __device__ void func() {
    #if defined(__CUDA_ARCH__)
        // Device code path
    #else
        // Host code path
    #endif
    }
    

### 5.4.1.2. Memory Space Specifiers

The memory space specifiers `__device__`, `__managed__`, `__constant__`, and `__shared__` indicate the storage location of a variable on the device.

The following table summarizes the memory space properties:

Table 39 Memory Space Specifier Memory Space Specifier | Location | Accessible by | Lifetime | Unique instance  
---|---|---|---|---  
`__device__` | Device global memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](../03-advanced/driver-api.html#driver-api-context) | Per device  
`__constant__` | Device constant memory | Device Threads (grid) / CUDA Runtime API | Program/[CUDA context](../03-advanced/driver-api.html#driver-api-context) | Per device  
`__managed__` | Host and Device (automatic) | Host/Device Threads | Program | Per program  
`__shared__` | Device (streaming multiprocessor) | Block Threads | Block | Block  
no specifier | Device (registers) | Single Thread | Single Thread | Single Thread  
  
* * *

  * Both `__device__` and `__constant__` variables can be accessed from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html) functions `cudaGetSymbolAddress()`, `cudaGetSymbolSize()`, `cudaMemcpyToSymbol()`, and `cudaMemcpyFromSymbol()`.

  * `__constant__` variables are read-only in device code and can only be modified from the host using the [CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html).


The following example illustrates how to use these APIs:
    
    
    __device__   float device_var       = 4.0f; // Variable in device memory
    __constant__ float constant_mem_var = 4.0f; // Variable in constant memory
                                                // For readability, the following example focuses on a device variable.
    int main() {
        float* device_ptr;
        cudaGetSymbolAddress((void**) &device_ptr, device_var);        // Gets address of device_var
    
        size_t symbol_size;
        cudaGetSymbolSize(&symbol_size, device_var);                   // Retrieves the size of the symbol (4 bytes).
    
        float host_var;
        cudaMemcpyFromSymbol(&host_var, device_var, sizeof(host_var)); // Copies from device to host.
    
        host_var = 3.0f;
        cudaMemcpyToSymbol(device_var, &host_var, sizeof(host_var));   // Copies from host to device.
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/vYjP8GGv3).

#### 5.4.1.2.1. `__shared__` Memory

`__shared__` memory variables can have a static size, which is determined at compile time, or a dynamic size, which is determined at kernel launch time. See the [Kernel Configuration](#execution-configuration) section for details on specifying the shared memory size at run time.

Shared memory constraints:

  * Variables with a dynamic size must be declared as an external array or as a pointer.

  * Variables with a static size cannot be initialized in their declaration.


The following example illustrates how to declare and size `__shared__` variables:
    
    
    extern __shared__ char dynamic_smem_pointer[];
    // extern __shared__ char* dynamic_smem_pointer; alternative syntax
    
    __global__ void kernel() { // or a __device__ function
        __shared__ int smem_var1[4];                  // static size
        auto smem_var2 = (int*) dynamic_smem_pointer; // dynamic size
    }
    
    int main() {
        size_t shared_memory_size = 16;
        kernel<<<1, 1, shared_memory_size>>>();
        cudaDeviceSynchronize();
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/nPjvd1frb).

#### 5.4.1.2.2. `__managed__` Memory

`__managed__` variables have the following restrictions:

  * The address of a `__managed__` variable is not a constant expression.

  * A `__managed__` variable shall not have a reference type `T&`.

  * The address or value of a `__managed__` variable shall not be used when the CUDA runtime may not be in a valid state, including the following cases:

    * In static/dynamic initialization or destruction of an object with `static` or `thread_local` storage duration.

    * In code that executes after `exit()` has been called. For example, a function marked with `__attribute__((destructor))`.

    * In code that executes when the CUDA runtime may not be initialized. For example, a function marked with `__attribute__((constructor))`.

  * A `__managed__` variable cannot be used as an unparenthesized id-expression argument to a `decltype()` expression.

  * `__managed__` variables have the same coherence and consistency behavior as specified for [dynamically allocated managed memory](../02-basics/understanding-memory.html#memory-unified-memory).

  * See also the restrictions for [local variables](cpp-language-support.html#local-variables).


Here are examples of legal and illegal uses of `__managed__` variables:
    
    
    #include <cassert>
    
    __device__ __managed__ int global_var = 10; // OK
    
    int* ptr = &global_var;                     // ERROR: use of a managed variable in static initialization
    
    struct MyStruct1 {
        int field;
        MyStruct1() : field(global_var) {};
    };
    
    struct MyStruct2 {
        ~MyStruct2() { global_var = 10; }
    };
    
    MyStruct1 temp1; // ERROR: use of managed variable in dynamic initialization
    
    MyStruct2 temp2; // ERROR: use of managed variable in the destructor of
                     //        object with static storage duration
    
    __device__ __managed__ const int const_var = 10;         // ERROR: const-qualified type
    
    __device__ __managed__ int&      reference = global_var; // ERROR: reference type
    
    template <int* Addr>
    struct MyStruct3 {};
    
    MyStruct3<&global_var> temp;     // ERROR: address of managed variable is not a constant expression
    
    __global__ void kernel(int* ptr) {
        assert(ptr == &global_var);  // OK
        global_var = 20;             // OK
    }
    
    int main() {
        int* ptr = &global_var;      // OK
        kernel<<<1, 1>>>(ptr);
        cudaDeviceSynchronize();
        global_var++;                // OK
        decltype(global_var) var1;   // ERROR: managed variable used as unparenthesized argument to decltype
    
        decltype((global_var)) var2; // OK
    }
    

### 5.4.1.3. Inlining Specifiers

The following specifiers can be used to control inlining for `__host__` and `__device__` functions:

  * `__noinline__`: Instructs `nvcc` not to inline the function.

  * `__forceinline__`: Forces `nvcc` to inline the function within a single translation unit.

  * `__inline_hint__`: Enables aggressive inlining across translation units when using [Link-Time Optimization](../02-basics/nvcc.html#nvcc-link-time-optimization).


These specifiers are mutually exclusive.

### 5.4.1.4. `__restrict__` Pointers

`nvcc` supports restricted pointers via the `__restrict__` keyword.

Pointer aliasing occurs when two or more pointers refer to overlapping memory regions. This can inhibit optimizations such as code reordering and common sub-expression elimination.

A restrict-qualified pointer is a promise from the programmer that for the lifetime of the pointer, the memory it points to will only be accessed through that pointer. This allows the compiler to perform more aggressive optimizations.

  * all threads that access the device function only read from it; or

  * at most one thread writes to it, and no other thread reads from it.


The following example illustrates an aliasing issue and demonstrates how using a restricted pointer can help the compiler reduce the number of instructions:
    
    
    __device__
    void device_function(const float* a, const float* b, float* c) {
        c[0] = a[0] * b[0];
        c[1] = a[0] * b[0];
        c[2] = a[0] * b[0] * a[1];
        c[3] = a[0] * a[1];
        c[4] = a[0] * b[0];
        c[5] = b[0];
        ...
    }
    

Because the pointers `a`, `b`, and `c` may be aliased, any write through `c` could modify elements of `a` or `b`. To guarantee functional correctness, the compiler cannot load `a[0]` and `b[0]` into registers, multiply them, and store the result in both `c[0]` and `c[1]`. This is because the results would differ from the abstract execution model if `a[0]` and `c[0]` were at the same location. The compiler cannot take advantage of the common sub-expression. Similarly, the compiler cannot reorder the computation of `c[4]` with the computations of `c[0]` and `c[1]` because a preceding write to `c[3]` could alter the inputs to the computation of `c[4]`.

By declaring `a`, `b`, and `c` as restricted pointers, the programmer informs the compiler that the pointers are not aliased. This means that writing to `c` will never overwrite the elements of `a` or `b`. This changes the function prototype as follows:
    
    
    __device__
    void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c);
    

Note that all pointer arguments must be restricted for the compiler optimizer to be effective. With the addition of the `__restrict__` keywords, the compiler can reorder and perform common sub-expression elimination at will while maintaining identical functionality to the abstract execution model.
    
    
    __device__
    void device_function(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c) {
        float t0 = a[0];
        float t1 = b[0];
        float t2 = t0 * t1;
        float t3 = a[1];
        c[0]     = t2;
        c[1]     = t2;
        c[4]     = t2;
        c[2]     = t2 * t3;
        c[3]     = t0 * t3;
        c[5]     = t1;
        ...
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/6KeTqarnW).

The result is a reduced number of memory accesses and computations, balanced by an increase in register pressure from caching loads and common sub-expressions in registers.

Since register pressure is a critical issue in many CUDA codes, the use of restricted pointers can negatively impact performance by reducing occupancy.

* * *

Accesses to `__global__` function `const` pointers marked with `__restrict__` are compiled as read-only cache loads, similar to the [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc) `ld.global.nc` or `__ldg()` [low-level load and store functions](#low-level-load-store-functions) instructions.
    
    
    __global__
    void kernel1(const float* in, float* out) {
        *out = *in; // PTX: ld.global
    }
    
    __global__
    void kernel2(const float* __restrict__ in, float* out) {
        *out = *in;  // PTX: ld.global.nc
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/drsTEPa8s).

### 5.4.1.5. `__grid_constant__` Parameters

Annotating a `__global__` function parameter with `__grid_constant__` prevents the compiler from creating a per-thread copy of the parameter. Instead, all threads in the grid will access the parameter through a single address, which can improve performance.

The `__grid_constant__` parameter has the following properties:

  * It has the lifetime of the kernel.

  * It is private to a single kernel, meaning the object is not accessible to threads from other grids, including sub-grids.

  * All threads in the kernel see the same address.

  * It is read-only. Modifying a `__grid_constant__` object or any of its sub-objects, including `mutable` members, is undefined behavior.


Requirements:

  * Kernel parameters annotated with `__grid_constant__` must have `const`-qualified non-reference types.

  * All function declarations must be consistent with any `__grid_constant__` parameters.

  * Function template specializations must match the primary template declaration with respect to any `__grid_constant__` parameters.

  * Function template instantiations must also match the primary template declaration with respect to any `__grid_constant__` parameters.


Examples:
    
    
    struct MyStruct {
        int         x;
        mutable int y;
    };
    
    __device__ void external_function(const MyStruct&);
    
    __global__ void kernel(const __grid_constant__ MyStruct s) {
        // s.x++; // Compile error: tried to modify read-only memory
        // s.y++; // Undefined Behavior: tried to modify read-only memory
    
        // Compiler will NOT create a per-thread local copy of "s":
        external_function(s);
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/Goq9jrEeo).

### 5.4.1.6. Annotation Summary

The following table summarizes the CUDA annotations and reports which execution space each annotation applies to and where it is valid.

Table 40 Annotation Summary Annotation | `__host__` / `__device__` / `__host__ __device__` | `__global__`  
---|---|---  
[__noinline__](#inline-specifiers), [__forceinline__](#inline-specifiers), [__inline_hint__](#inline-specifiers) | Function | ❌  
[__restrict__](#restrict) | Pointer Parameter | Pointer Parameter  
[__grid_constant__](#grid-constant) | ❌ | Parameter  
[__launch_bounds__](#launch-bounds) | ❌ | Function  
[__maxnreg__](#maximum-number-of-registers-per-thread) | ❌ | Function  
[__cluster_dims__](#cluster-dimensions) | ❌ | Function  
  
## 5.4.2. Built-in Types and Variables

### 5.4.2.1. Host Compiler Type Extensions

The use of non-standard arithmetic types is permitted by CUDA, as long as the host compiler supports it. The following types are supported:

  * 128-bit integer type `__int128`.

    * Supported on Linux when the host compiler defines the `__SIZEOF_INT128__` macro.

  * 128-bit floating-point types `__float128` and `_Float128` are available on GPU devices with compute capability 10.0 and later. A constant expression of `__float128` type may be processed by the compiler in a floating-point representation with lower precision.

    * Supported on Linux x86 when the host compiler defines the `__SIZEOF_FLOAT128__` or `__FLOAT128__` macros.

  * `_Complex` [types](https://www.gnu.org/software/c-intro-and-ref/manual/html_node/Complex-Data-Types.html) are only supported in host code.


### 5.4.2.2. Built-in Variables

The values used to specify and retrieve the kernel configuration for the grid and blocks along the x, y, and z dimensions are of type `dim3`. The variables used to obtain the block and thread indices are of type `uint3`. Both `dim3` and `uint3` are trivial structures consisting of three unsigned values named `x`, `y`, and `z`. In C++11 and later, the default value of all components of `dim3` is 1.

Built-in device-only variables:

  * `dim3 gridDim`: contains the dimensions of the grid, namely the number of thread blocks, along the x, y, and z dimensions.

  * `dim3 blockDim`: contains the dimensions of the thread block, namely the number of threads, along the x, y, and z dimensions.

  * `uint3 blockIdx`: contains the block index within the grid, along the x, y, and z dimensions.

  * `uint3 threadIdx`: contains the thread index within the block, along the x, y, and z dimensions.

  * `int warpSize` : A run-time value defined as the number of threads in a warp, commonly `32`. See also [Warps and SIMT](../01-introduction/programming-model.html#programming-model-warps-simt) for the definition of a warp.


### 5.4.2.3. Built-in Types

CUDA provides vector types derived from basic integer and floating-point types that are supported for both the host and the device. The following table shows the available vector types.

Table 41 Vector Types C++ Fundamental Type | Vector X1 | Vector X2 | Vector X3 | Vector X4  
---|---|---|---|---  
`signed char` | `char1` | `char2` | `char3` | `char4`  
`unsigned char` | `uchar1` | `uchar2` | `uchar3` | `uchar4`  
`signed short` | `short1` | `short2` | `short3` | `short4`  
`unsigned short` | `ushort1` | `ushort2` | `ushort3` | `ushort4`  
`signed int` | `int1` | `int2` | `int3` | `int4`  
`unsigned` | `uint1` | `uint2` | `uint3` | `uint4`  
`signed long` | `long1` | `long2` | `long3` | `long4_16a/long4_32a`  
`unsigned long` | `ulong1` | `ulong2` | `ulong3` | `ulong4_16a/ulong4_32a`  
`signed long long` | `longlong1` | `longlong2` | `longlong3` | `longlong4_16a/longlong4_32a`  
`unsigned long long` | `ulonglong1` | `ulonglong2` | `ulonglong3` | `ulonglong4_16a/ulonglong4_32a`  
`float` | `float1` | `float2` | `float3` | `float4`  
`double` | `double1` | `double2` | `double3` | `double4_16a/double4_32a`  
  
Note that `long4`, `ulong4`, `longlong4`, `ulonglong4`, and `double4` have been deprecated in CUDA 13, and may be removed in a future release.

* * *

The following table details the byte size and alignment requirements of the vector types:

Table 42 Alignment Requirements Type | Size | Alignment  
---|---|---  
`char1`, `uchar1` | 1 | 1  
`char2`, `uchar2` | 2 | 2  
`char3`, `uchar3` | 3 | 1  
`char4`, `uchar4` | 4 | 4  
`short1`, `ushort1` | 2 | 2  
`short2`, `ushort2` | 4 | 4  
`short3`, `ushort3` | 6 | 2  
`short4`, `ushort4` | 8 | 8  
`int1`, `uint1` | 4 | 4  
`int2`, `uint2` | 8 | 8  
`int3`, `uint3` | 12 | 4  
`int4`, `uint4` | 16 | 16  
`long1`, `ulong1` | 4/8 ***** | 4/8 *****  
`long2`, `ulong2` | 8/16 ***** | 8/16 *****  
`long3`, `ulong3` | 12/24 ***** | 4/8 *****  
`long4`, `ulong4` (deprecated) | 16/32 ***** | 16 *****  
`long4_16a`, `ulong4_16a` | 16/32 ***** | 16  
`long4_32a`, `ulong4_32a` | 16/32 ***** | 32  
`longlong1`, `ulonglong1` | 8 | 8  
`longlong2`, `ulonglong2` | 16 | 16  
`longlong3`, `ulonglong3` | 24 | 8  
`longlong4`, `ulonglong4` (deprecated) | 32 | 16  
`longlong4_16a`, `ulonglong4_16a` | 32 | 16  
`longlong4_32a`, `ulonglong4_32a` | 32 | 32  
`float1` | 4 | 4  
`float2` | 8 | 8  
`float3` | 12 | 4  
`float4` | 16 | 16  
`double1` | 8 | 8  
`double2` | 16 | 16  
`double3` | 24 | 8  
`double4` (deprecated) | 32 | 16  
`double4_16a` | 32 | 16  
`double4_32a` | 32 | 32  
  
***** `long` is 4 bytes on C++ LLP64 data model (Windows 64-bit), while it is 8 bytes on C++ LP64 data model (Linux 64-bit).

* * *

Vector types are structures. Their first, second, third, and fourth components are accessible through the `x`, `y`, `z`, and `w` fields, respectively.
    
    
    int sum(int4 value) {
        return value.x + value.y + value.z + value.w;
    }
    

They all have a factory function of the form `make_<type_name>()`; for example:
    
    
    int4 add_one(int x, int y, int z, int w) {
        return make_int4(x + 1, y + 1, z + 1, w + 1);
    }
    

If host code is not compiled with `nvcc`, the vector types and related functions can be imported by including the `cuda_runtime.h` header provided in the CUDA toolkit.

## 5.4.3. Kernel Configuration

Any call to a `__global__` function must specify an _execution configuration_ for that call. This execution configuration defines the dimensions of the grid and blocks that will be used to execute the function on the device, as well as the associated [stream](../02-basics/asynchronous-execution.html#cuda-streams).

The execution configuration is specified by inserting an expression in the form `<<<grid_dim, block_dim, dynamic_smem_bytes, stream>>>` between the function name and the parenthesized argument list, where:

  * `grid_dim` is of type [dim3](#built-in-variables) and specifies the dimension and size of the grid, such that `grid_dim.x * grid_dim.y * grid_dim.z` equals the number of blocks being launched;

  * `block_dim` is of type [dim3](#built-in-variables) and specifies the dimension and size of each block, such that `block_dim.x * block_dim.y * block_dim.z` equals the number of threads per block;

  * `dynamic_smem_bytes` is an optional `size_t` argument that defaults to zero. It specifies the number of bytes in shared memory that are dynamically allocated per block for this call in addition to the statically allocated memory. This memory is used by `extern __shared__` arrays (see [__shared__ Memory](#shared-memory-specifier)).

  * `stream` is of type `cudaStream_t` (pointer) and specifies the associated stream. `stream` is an optional argument that defaults to `NULL`.


The following example shows a kernel function declaration and call:
    
    
    __global__ void kernel(float* parameter);
    
    kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>(parameter);
    

The arguments for the execution configuration are evaluated before the arguments for the actual function.

The function call fails if `grid_dim` or `block_dim` exceeds the maximum sizes allowed for the device, as specified in [Compute Capabilities](compute-capabilities.html#compute-capabilities), or if `dynamic_smem_bytes` is greater than the available shared memory after accounting for statically allocated memory.

### 5.4.3.1. Thread Block Cluster

Compute capability 9.0 and higher allow users to specify compile-time thread block cluster dimensions so that the kernels can use the [cluster hierarchy](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters) in CUDA. The compile-time cluster dimension can be specified using the `__cluster_dims__` attribute with the following syntax: `__cluster_dims__([x, [y, [z]]])`. The example below shows a compile-time cluster size of 2 in the X dimension and 1 in the Y and Z dimensions.
    
    
    __global__ void __cluster_dims__(2, 1, 1) kernel(float* parameter);
    

The default form of `__cluster_dims__()` specifies that a kernel is to be launched as a grid cluster. If a cluster dimension is not specified, the user can specify it at launch time. Failing to specify a dimension at launch time will result in a launch-time error.

The dimensions of the thread block cluster can also be specified at runtime, and the kernel with the cluster can be launched using the `cudaLaunchKernelEx` API. This API takes a configuration argument of type `cudaLaunchConfig_t`, a kernel function pointer, and kernel arguments. The example below shows runtime kernel configuration.
    
    
    __global__ void kernel(float parameter1, int parameter2) {}
    
    int main() {
        cudaLaunchConfig_t config = {0};
        // The grid dimension is not affected by cluster launch, and is still enumerated
        // using the number of blocks.
        // The grid dimension should be a multiple of cluster size.
        config.gridDim          = dim3{4};  // 4 blocks
        config.blockDim         = dim3{32}; // 32 threads per block
        config.dynamicSmemBytes = 1024;     // 1 KB
    
        cudaLaunchAttribute attribute[1];
        attribute[0].id               = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size in X-dimension
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;
        config.attrs    = attribute;
        config.numAttrs = 1;
    
        float parameter1 = 3.0f;
        int   parameter2 = 4;
        cudaLaunchKernelEx(&config, kernel, parameter1, parameter2);
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/M67r3a5zM).

### 5.4.3.2. Launch Bounds

As discussed in the [Kernel Launch and Occupancy](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy) section, using fewer registers allows more threads and thread blocks to reside on a multiprocessor, which improves performance.

Therefore, the compiler uses heuristics to minimize register usage while keeping [register spilling](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-registers) and instruction count to a minimum. Applications can optionally aid these heuristics by providing additional information to the compiler in the form of launch bounds that are specified using the `__launch_bounds__()` qualifier in the definition of a `__global__` function:
    
    
    __global__ void
    __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor, maxBlocksPerCluster)
    MyKernel(...) {
        ...
    }
    

  * `maxThreadsPerBlock` specifies the maximum number of threads per block with which the application will ever launch `MyKernel()`; it compiles to the `.maxntid` PTX directive.

  * `minBlocksPerMultiprocessor` is optional and specifies the desired minimum number of resident blocks per multiprocessor; it compiles to the `.minnctapersm` PTX directive.

  * `maxBlocksPerCluster` is optional and specifies the desired maximum number of thread blocks per cluster with which the application will ever launch `MyKernel()`; it compiles to the `.maxclusterrank` PTX directive.


If launch bounds are specified, the compiler first derives the upper limit, `L`, on the number of registers that the kernel should use. This ensures that `minBlocksPerMultiprocessor` blocks (or a single block, if `minBlocksPerMultiprocessor` is not specified) of `maxThreadsPerBlock` threads can reside on the multiprocessor. See the [occupancy](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-kernel-launch-and-occupancy) section for the relationship between the number of registers used by a kernel and the number of registers allocated per block. The compiler then optimizes register usage as follows:

  * If the initial register usage exceeds `L`, the compiler reduces it until it is less than or equal to `L`. This usually results in increased local memory usage and/or a higher number of instructions.

  * If the initial register usage is lower than `L`

    * If `maxThreadsPerBlock` is specified but `minBlocksPerMultiprocessor` is not, the compiler uses `maxThreadsPerBlock` to determine the register usage thresholds for the transitions between `n` and `n + 1` resident blocks. This occurs when using one less register makes room for an additional resident block. Then, the compiler applies similar heuristics as when no launch bounds are specified.

    * If both `minBlocksPerMultiprocessor` and `maxThreadsPerBlock` are specified, the compiler may increase register usage up to `L` in order to reduce the number of instructions and better hide the latency of single-threaded instructions.


A kernel will fail to launch if it is executed with:

  * more threads per block than its launch bound `maxThreadsPerBlock`.

  * more thread blocks per cluster than its launch bound `maxBlocksPerCluster`.


The per-thread resources required by a CUDA kernel may limit the maximum block size in an undesirable way. To maintain forward compatibility with future hardware and toolkits, and to ensure that at least one thread block can run on a streaming multiprocessor, developers should include the single argument `__launch_bounds__(maxThreadsPerBlock)` which specifies the largest block size with which the kernel will launch. Failure to do so could result in “too many resources requested for launch” errors. Providing the two-argument version of `__launch_bounds__(maxThreadsPerBlock,minBlocksPerMultiprocessor)` can improve performance in some cases. The best value for `minBlocksPerMultiprocessor` should be determined through a detailed analysis of each kernel.

The optimal launch bounds for a kernel typically differ across major architecture revisions. The following code sample illustrates how this is managed in device code with the `__CUDA_ARCH__` [macro](#cuda-arch-macro).
    
    
    #define THREADS_PER_BLOCK  256
    
    #if __CUDA_ARCH__ >= 900
        #define MY_KERNEL_MAX_THREADS  (2 * THREADS_PER_BLOCK)
        #define MY_KERNEL_MIN_BLOCKS   3
    #else
        #define MY_KERNEL_MAX_THREADS  THREADS_PER_BLOCK
        #define MY_KERNEL_MIN_BLOCKS   2
    #endif
    
    __global__ void
    __launch_bounds__(MY_KERNEL_MAX_THREADS, MY_KERNEL_MIN_BLOCKS)
    MyKernel(...) {
        ...
    }
    

When `MyKernel` is invoked with the maximum number of threads per block, which is specified as the first parameter of `__launch_bounds__()`, it is tempting to use `MY_KERNEL_MAX_THREADS` as the number of threads per block in the execution configuration:
    
    
    // Host code
    MyKernel<<<blocksPerGrid, MY_KERNEL_MAX_THREADS>>>(...);
    

However, this will not work, since `__CUDA_ARCH__` is undefined in host code as mentioned in the [Execution Space Specifiers](#execution-space-specifiers) section. Therefore, `MyKernel` will launch with 256 threads per block. The number of threads per block should instead be determined:

  * Either at compile time using a macro or constant that does not depend on `__CUDA_ARCH__`, for example
        
        // Host code
        MyKernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(...);
        

  * Or at runtime based on the compute capability
        
        // Host code
        cudaGetDeviceProperties(&deviceProp, device);
        int threadsPerBlock = (deviceProp.major >= 9) ? 2 * THREADS_PER_BLOCK : THREADS_PER_BLOCK;
        MyKernel<<<blocksPerGrid, threadsPerBlock>>>(...);
        


The `--resource-usage` compiler option reports register usage. The [CUDA profiler](https://docs.nvidia.com/nsight-compute/NsightCompute/index.html#occupancy-calculator) reports occupancy, which can be used to derive the number of resident blocks.

### 5.4.3.3. Maximum Number of Registers per Thread

To enable low-level performance tuning, CUDA C++ offers the `__maxnreg__()` function qualifier, which passes performance tuning information to the backend optimizing compiler. The `__maxnreg__()` qualifier specifies the maximum number of registers that can be allocated to a single thread in a thread block. In the definition of a `__global__` function:
    
    
    __global__ void
    __maxnreg__(maxNumberRegistersPerThread)
    MyKernel(...) {
        ...
    }
    

The `maxNumberRegistersPerThread` variable specifies the maximum number of registers to be allocated to a single thread in a thread block of the kernel `MyKernel()`; it compiles to the `.maxnreg` PTX directive.

The `__launch_bounds__()` and `__maxnreg__()` qualifiers cannot be applied to the same kernel together.

The `--maxrregcount <N>` compiler option can be used to control register usage for all `__global__` functions in a file. This option is ignored for kernel functions with the `__maxnreg__` qualifier.

## 5.4.4. Synchronization Primitives

### 5.4.4.1. Thread Block Synchronization Functions
    
    
    void __syncthreads();
    int  __syncthreads_count(int predicate);
    int  __syncthreads_and(int predicate);
    int  __syncthreads_or(int predicate);
    

The intrinsics coordinate communication among threads within the same block. When threads in a block access the same addresses in shared or global memory, read-after-write, write-after-read, or write-after-write hazards can occur. These hazards can be avoided by synchronizing threads between such accesses.

The intrinsics have the following semantics:

  * `__syncthreads*()` wait until all non-exited threads in the thread block simultaneously reach the same `__syncthreads*()` intrinsic call in the program or exit.

  * `__syncthreads*()` provide memory ordering among participating threads: the call to `__syncthreads*()` intrinsics strongly happens before (see [C++ specification [intro.races]](https://eel.is/c++draft/intro.races)) any participating thread is unblocked from the wait or exits.


The following example shows how to use `__syncthreads()` to synchronize threads within a thread block and safely sum the elements of an array shared among the threads:
    
    
    // assuming blockDim.x is 128
    __global__ void example_syncthreads(int* input_data, int* output_data) {
        __shared__ int shared_data[128];
        // Every thread writes to a distinct element of 'shared_data':
        shared_data[threadIdx.x] = input_data[threadIdx.x];
    
        // All threads synchronize, guaranteeing all writes to 'shared_data' are ordered 
        // before any thread is unblocked from '__syncthreads()':
        __syncthreads();
    
        // A single thread safely reads 'shared_data':
        if (threadIdx.x == 0) {
            int sum = 0;
            for (int i = 0; i < blockDim.x; ++i) {
                sum += shared_data[i];
            }
            output_data[blockIdx.x] = sum;
        }
    }
    

The `__syncthreads*()` intrinsics are permitted in conditional code, but only if the condition evaluates uniformly across the entire thread block. Otherwise, execution may hang or produce unintended side effects.

The following example demonstrates a valid behavior:
    
    
    // assuming blockDim.x is 128
    __global__ void syncthreads_valid_behavior(int* input_data, int* output_data) {
        __shared__ int shared_data[128];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        if (blockIdx.x > 0) { // CORRECT, uniform condition across all block threads
            __syncthreads();
            output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
        }
    }
    

while the following examples exhibit invalid behavior, such as kernel hang, or undefined behavior:
    
    
    // assuming blockDim.x is 128
    __global__ void syncthreads_invalid_behavior1(int* input_data, int* output_data) {
        __shared__ int shared_data[256];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        if (threadIdx.x > 0) { // WRONG, non-uniform condition
            __syncthreads();   // Undefined Behavior
            output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
        }
    }
    
    
    
    // assuming blockDim.x is 128
    __global__ void syncthreads_invalid_behavior2(int* input_data, int* output_data) {
        __shared__ int shared_data[256];
        shared_data[threadIdx.x] = input_data[threadIdx.x];
        for (int i = 0; i < blockDim.x; ++i) {
            if (i == threadIdx.x) { // WRONG, non-uniform condition
                __syncthreads();    // Undefined Behavior
            }
        }
        output_data[threadIdx.x] = shared_data[128 - threadIdx.x];
    }
    

* * *

`__syncthreads()` **variants with predicate** :
    
    
    int __syncthreads_count(int predicate);
    

is identical to `__syncthreads()` except that it evaluates a predicate for all non-exited threads in the block and returns the number of threads for which the predicate evaluates to a non-zero value.
    
    
    int __syncthreads_and(int predicate);
    

is identical to `__syncthreads()` except that it evaluates the predicate for all non-exited threads in the block. It returns a non-zero value if and only if the predicate evaluates to a non-zero value for all of them.
    
    
    int __syncthreads_or(int predicate);
    

is identical to `__syncthreads()` except that it evaluates the predicate for all non-exited threads in the block. It returns a non-zero value if and only if the predicate evaluates to a non-zero value one or more of them.

### 5.4.4.2. Warp Synchronization Function
    
    
    void __syncwarp(unsigned mask = 0xFFFFFFFF);
    

The intrinsic function `__syncwarp()` coordinates communication between the threads within the same warp. When some threads within a warp access the same addresses in shared or global memory, potential read-after-write, write-after-read, or write-after-write hazards may occur. These data hazards can be avoided by synchronizing the threads between these accesses.

Calling `__syncwarp(mask)` provides memory ordering among the participating threads within a warp named in `mask`: the call to `__syncwarp(mask)` strongly happens before (see [C++ specification [intro.races]](https://eel.is/c++draft/intro.races)) any warp thread named in `mask` is unblocked from the wait or exits.

The functions are subject to the [Warp __sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

The following example demonstrates how to use `__syncwarp()` to synchronize threads within a warp to safely access a shared memory array:
    
    
    __global__ void example_syncwarp(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];
    
            __syncwarp(); // equivalent to __syncwarp(0xFFFFFFFF)
            if (threadIdx.x == 0)
                output_data[0] = shared_data[1];
        }
    }
    

### 5.4.4.3. Memory Fence Functions

The CUDA programming model assumes a weakly ordered memory model. In other words, the order in which a CUDA thread writes data to shared memory, global memory, page-locked host memory, or the memory of a peer device is not necessarily the order in which another CUDA or host thread observes the data being written. Reading from or writing to the same memory location without memory fences or synchronization results in undefined behavior.

In the following example, thread 1 executes `writeXY()`, while thread 2 executes `readXY()`.
    
    
    __device__ int X = 1, Y = 2;
    
    __device__ void writeXY() {
        X = 10;
        Y = 20;
    }
    
    __device__ void readXY() {
        int B = Y;
        int A = X;
    }
    

The two threads simultaneously read and write to the same memory locations, `X` and `Y`. Any data race results in undefined behavior and has no defined semantics. Therefore, the resulting values for `A` and `B` can be anything.

Memory fence and synchronization functions enforce a [sequentially consistent ordering](https://en.cppreference.com/w/cpp/atomic/memory_order) of memory accesses. These functions differ in the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes) in which orderings are enforced, but are independent of the accessed memory space, including shared memory, global memory, page-locked host memory, and the memory of a peer device.

Hint

It is suggested to use `cuda::atomic_thread_fence` provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic/atomic_thread_fence.html) whenever possible for safety and portability reasons.

**Block-level memory fence**

CUDA C++
    
    
    // <cuda/atomic> header
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_block);
    

ensures that:

  * All writes to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are observed by all threads in the calling thread’s block as occurring before all writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`;

  * All reads from all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are ordered before all reads from all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`.


Intrinsics
    
    
    void __threadfence_block();
    

ensures that:

  * All writes to all memory made by the calling thread before the call to `__threadfence_block()` are observed by all threads in the calling thread’s block as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_block()`;

  * All reads from all memory made by the calling thread before the call to `__threadfence_block()` are ordered before all reads from all memory made by the calling thread after the call to `__threadfence_block()`.


**Device-level memory fence**

CUDA C++
    
    
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    

ensures that:

  * No writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()`.


Intrinsics
    
    
    void __threadfence();
    

ensures that:

  * No writes to all memory made by the calling thread after the call to `__threadfence()` are observed by any thread in the device as occurring before any write to all memory made by the calling thread before the call to `__threadfence()`.


**System-level memory fence**

CUDA C++
    
    
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_system);
    

ensures that:

  * All writes to all memory made by the calling thread before the call to `cuda::atomic_thread_fence()` are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to `cuda::atomic_thread_fence()`.


Intrinsics
    
    
    void __threadfence_system();
    

ensures that:

  * All writes to all memory made by the calling thread before the call to `__threadfence_system()` are observed by all threads in the device, host threads, and all threads in peer devices as occurring before all writes to all memory made by the calling thread after the call to `__threadfence_system()`.


In the previous code sample, we can insert memory fences in the code as follows:

CUDA C++
    
    
    #include <cuda/atomic>
    
    __device__ int X = 1, Y = 2;
    
    __device__ void writeXY() {
        X = 10;
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
        Y = 20;
    }
    
    __device__ void readXY() {
        int B = Y;
        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
        int A = X;
    }
    

Intrinsics
    
    
    __device__ int X = 1, Y = 2;
    
    __device__ void writeXY() {
        X = 10;
        __threadfence();
        Y = 20;
    }
    
    __device__ void readXY() {
        int B = Y;
        __threadfence();
        int A = X;
    }
    

For this code, the following outcomes can be observed:

  * `A` equal to 1 and `B` equal to 2, namely `readXY()` is executed before `writeXY()`,

  * `A` equal to 10 and `B` equal to 20, namely `writeXY()` is executed before `readXY()`.

  * `A` equal to 10 and `B` equal to 2.

  * The case where `A` is 1 and `B` is 20 is not possible, as the memory fence ensures that the write to `X` is visible before the write to `Y`.


If threads 1 and 2 belong to the same block, it is enough to use a block-level fence. If threads 1 and 2 do not belong to the same block, a device-level fence must be used if they are CUDA threads from the same device, and a system-level fence must be used if they are CUDA threads from two different devices.

A common use case is illustrated by the following code sample, where threads consume data produced by other threads. This kernel computes the sum of an array of N numbers in a single call.

  * Each block first sums a subset of the array and stores the result in global memory.

  * When all the blocks have finished, the last block reads each of these partial sums from global memory and adds them together to obtain the final result.

  * To determine which block finished last, each block atomically increments a counter to signal completion of computing and storing its partial sum (see the [Atomic Functions](#atomic-functions) section for further details). The last block receives a counter value equal to `gridDim.x - 1`.


Without a fence between storing the partial sum and incrementing the counter, the counter may increment before the partial sum is stored. This could cause the counter to reach `gridDim.x - 1` and allow the last block to start reading partial sums before they are updated in memory.

Note

The memory fence only affects the order in which memory operations are executed; it does not guarantee visibility of these operations to other threads.

In the code sample below, the visibility of the memory operations on the `result` variable is ensured by declaring it as `volatile`. For more details, see the `volatile`-[qualified variables](cpp-language-support.html#volatile-qualifier) section.
    
    
    #include <cuda/atomic>
    
    __device__ int count = 0;
    
    __global__ void sum(const float*    array,
                        int             N,
                        volatile float* result) {
        __shared__ bool isLastBlockDone;
        // Each block sums a subset of the input array.
        float partialSum = calculatePartialSum(array, N);
    
        if (threadIdx.x == 0) {
            // Thread 0 of each block stores the partial sum to global memory.
            // The compiler will use a store operation that bypasses the L1 cache
            // since the "result" variable is declared as volatile.
            // This ensures that the threads of the last block will read the correct
            // partial sums computed by all other blocks.
            result[blockIdx.x] = partialSum;
    
            // Thread 0 makes sure that the increment of the "count" variable is
            // only performed after the partial sum has been written to global memory.
            cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_device);
    
            // Thread 0 signals that it is done.
            int count_old = atomicInc(&count, gridDim.x);
    
            // Thread 0 determines if its block is the last block to be done.
            isLastBlockDone = (count_old == (gridDim.x - 1));
        }
        // Synchronize to make sure that each thread reads the correct value of
        // isLastBlockDone.
        __syncthreads();
    
        if (isLastBlockDone) {
            // The last block sums the partial sums stored in result[0 .. gridDim.x-1]
            float totalSum = calculateTotalSum(result);
    
            if (threadIdx.x == 0) {
                // Thread 0 of last block stores the total sum to global memory and
                // resets the count variable, so that the next kernel call works
                // properly.
                result[0] = totalSum;
                count     = 0;
            }
        }
    }
    

## 5.4.5. Atomic Functions

Atomic functions perform read-modify-write operations on shared data, making them appear to execute in a single step. Atomicity ensures that each operation either completes fully or not at all, providing all participating threads with a consistent view of the data.

CUDA provides atomic functions in four ways:

Extended CUDA C++ atomic functions, [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html).
    

  * They are allowed in both host and device code.

  * They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.

  * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.


Standard C++ atomic functions, [cuda::std::atomic](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [cuda::std::atomic_ref](https://en.cppreference.com/w/cpp/atomic/atomic_ref.html).
    

  * They are allowed in both host and device code.

  * They follow the [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) semantics.

  * They do not allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.


Compiler [built-in atomic functions](#built-in-atomic-functions), `__nv_atomic_<op>()`.
    

  * They have been available since CUDA 12.8.

  * They are only allowed in device code.

  * They follow the [C++ standard atomic memory order](https://en.cppreference.com/w/cpp/atomic/memory_order.html) semantics.

  * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations.

  * They have the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html).

  * They support a subset of the data types allowed by [cuda::std::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html) and [cuda::std::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), except for 128-bit data types.


[Legacy atomic functions](#legacy-atomic-functions), `atomic<Op>()`.
    

  * They are only allowed in device code.

  * They only support `memory_order_relaxed` [C++ atomic memory semantics](https://en.cppreference.com/w/cpp/atomic/memory_order.html).

  * They allow specifying the [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes) of the atomic operations as part of the function name.

  * Unlike [built-in atomic functions](#built-in-atomic-functions), legacy atomic functions only ensure atomicity and do not introduce synchronization points (fences).

  * They support a subset of the data types allowed by [built-in atomic functions](#built-in-atomic-functions). The atomic `add` operation supports additional data types.


Hint

Using the [Extended CUDA C++ atomic functions](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html) provided by `libcu++` is recommended for efficiency, safety, and portability.

### 5.4.5.1. Legacy Atomic Functions

Legacy atomic functions perform atomic read-modify-write operations on a 32-, 64-, or 128-bit word stored in global or shared memory. For example, the `atomicAdd()` function reads a word at a specific address in global or shared memory, adds a number to it, and writes the result back to the same address.

  * Atomic functions can only be used in device functions.

  * For vector types such as `__half2`, `__nv_bfloat162`, `float2`, and `float4`, the read-modify-write operation is performed on each element of the vector. The entire vector is not guaranteed to be atomic in a single access.


The atomic functions described in this section have a [memory ordering](https://en.cppreference.com/w/cpp/atomic/memory_order) of `cuda::std::memory_order_relaxed` and are only atomic at a particular [thread scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes):

  * Atomic APIs without a suffix, for example `atomicAdd`, are atomic at scope `cuda::thread_scope_device`.

  * Atomic APIs with the `_block` suffix, for example, `atomicAdd_block`, are atomic at scope `cuda::thread_scope_block`.

  * Atomic APIs with the `_system` suffix, for example, `atomicAdd_system`, are atomic at scope `cuda::thread_scope_system` if they meet particular [conditions](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity).


The following example shows the CPU and GPU atomically updating an integer value at address `addr`:
    
    
    #include <cuda_runtime.h>
    
    __global__ void atomicAdd_kernel(int* addr) {
        atomicAdd_system(addr, 10);
    }
    
    void test_atomicAdd(int device_id) {
        int* addr;
        cudaMallocManaged(&addr, 4);
        *addr = 0;
    
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device_id);
        if (deviceProp.concurrentManagedAccess != 1) {
            return; // the device does not coherently access managed memory concurrently with the CPU
        }
    
        atomicAdd_kernel<<<...>>>(addr);
        __sync_fetch_and_add(addr, 10);  // CPU atomic operation
    }
    

* * *

Note that any atomic operation can be implemented based on `atomicCAS()` (Compare and Swap). For example, `atomicAdd()` for single-precision floating-point numbers can be implemented as follows:
    
    
    #include <cuda/memory>
    #include <cuda/std/bit>
    
    __device__ float customAtomicAdd(float* d_ptr, float value) {
        volatile unsigned* d_ptr_unsigned = reinterpret_cast<unsigned*>(d_ptr);
        unsigned  old_value      = *d_ptr_unsigned;
        unsigned  assumed;
        do {
            assumed                          = old_value;
            float    assumed_float           = cuda::std::bit_cast<float>(assumed);
            float    expected_value          = assumed_float + value;
            unsigned expected_value_unsigned = cuda::std::bit_cast<unsigned>(expected_value);
            old_value                        = atomicCAS(d_ptr_unsigned, assumed, expected_value_unsigned);
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old_value);
        return cuda::std::bit_cast<float>(old_value);
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/676e5bc7a).

#### 5.4.5.1.1. `atomicAdd()`
    
    
    T atomicAdd(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old + val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicAdd()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`, `float`, `double`, `__half2`, `__half`.

  * `__nv_bfloat16`, `__nv_bfloat162` on devices of compute capability 8.x and higher.

  * `float2`, `float4` on devices of compute capability 9.x and higher, and only supported for global memory addresses.


The atomicity of `atomicAdd()` applied to vector types, for example `__half2` or `float4`, is guaranteed separately for each of the components; the entire vector is not guaranteed to be atomic as a single access.

#### 5.4.5.1.2. `atomicSub()`
    
    
    T atomicSub(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old - val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicSub()` supports the following data types:

  * `int`, `unsigned`


#### 5.4.5.1.3. `atomicInc()`
    
    
    unsigned atomicInc(unsigned* address, unsigned val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old >= val ? 0 : (old + 1)`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

#### 5.4.5.1.4. `atomicDec()`
    
    
    unsigned atomicDec(unsigned* address, unsigned val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `(old == 0 || old > val) ? val : (old - 1)`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

#### 5.4.5.1.5. `atomicAnd()`
    
    
    T atomicAnd(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old & val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicAnd()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`.


#### 5.4.5.1.6. `atomicOr()`
    
    
    T atomicOr(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old | val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicOr()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`.


#### 5.4.5.1.7. `atomicXor()`
    
    
    T atomicXor(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old ^ val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicXor()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`.


#### 5.4.5.1.8. `atomicMin()`
    
    
    T atomicMin(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes the minimum of `old` and `val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicMin()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`, `long long`.


#### 5.4.5.1.9. `atomicMax()`
    
    
    T atomicMax(T* address, T val);
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes the maximum of `old` and `val`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicMax()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`, `long long`.


#### 5.4.5.1.10. `atomicExch()`
    
    
    T atomicExch(T* address, T val);
    
    
    
    template<typename T>
    T atomicExch(T* address, T val); // only 128-bit types, compute capability 9.x and higher
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Stores `val` back to memory at the same address.


The function returns the `old` value.

`atomicExch()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`, `float`.


The C++ template function `atomicExch()` supports 128-bit types with the following requirements:

  * Compute capability 9.x and higher.

  * `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.

  * `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.

  * For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.


#### 5.4.5.1.11. `atomicCAS()`
    
    
    T atomicCAS(T* address, T compare, T val);
    
    
    
    template<typename T>
    T atomicCAS(T* address, T compare, T val);  // only 128-bit types, compute capability 9.x and higher
    

The function performs the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old == compare ? val : old`.

  3. Stores the result back to memory at the same address.


The function returns the `old` value.

`atomicCAS()` supports the following data types:

  * `int`, `unsigned`, `unsigned long long`, `unsigned short`.


The C++ template function `atomicCAS()` supports 128-bit types with the following requirements:

  * Compute capability 9.x and higher.

  * `T` must be aligned to 16 bytes, namely `alignof(T) >= 16`.

  * `T` must be trivially copyable, namely `std::is_trivially_copyable_v<T>`.

  * For C++03 and older: `T` must be trivially constructible, namely `std::is_default_constructible_v<T>`.


### 5.4.5.2. Built-in Atomic Functions

CUDA 12.8 and later support CUDA compiler built-in functions for atomic operations, following the same memory ordering semantics as [C++ standard atomic operations](https://en.cppreference.com/w/cpp/atomic/atomic.html) and the CUDA [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes). The functions follow the [GNU’s atomic built-in function signature](https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) with an extra argument for thread scope.

`nvcc` defines the macro `__CUDACC_DEVICE_ATOMIC_BUILTINS__` when built-in atomic functions are supported.

Below are listed the raw enumerators for the [memory orders](https://en.cppreference.com/w/cpp/atomic/atomic.html) and [thread scopes](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#libcudacxx-extended-api-memory-model-thread-scopes), which are used as the `order` and `scope` arguments of the built-in atomic functions:
    
    
    // atomic memory orders
    enum {
       __NV_ATOMIC_RELAXED,
       __NV_ATOMIC_CONSUME,
       __NV_ATOMIC_ACQUIRE,
       __NV_ATOMIC_RELEASE,
       __NV_ATOMIC_ACQ_REL,
       __NV_ATOMIC_SEQ_CST
    };
    
    
    
    // thread scopes
    enum {
       __NV_THREAD_SCOPE_THREAD,
       __NV_THREAD_SCOPE_BLOCK,
       __NV_THREAD_SCOPE_CLUSTER,
       __NV_THREAD_SCOPE_DEVICE,
       __NV_THREAD_SCOPE_SYSTEM
    };
    

  * The memory order corresponds to [C++ standard atomic operations’ memory order](https://en.cppreference.com/w/cpp/atomic/memory_order).

  * The thread scope follows the `cuda::thread_scope` [definition](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes).

  * `__NV_ATOMIC_CONSUME` memory order is currently implemented using stronger `__NV_ATOMIC_ACQUIRE` memory order.

  * `__NV_THREAD_SCOPE_THREAD` thread scope is currently implemented using wider `__NV_THREAD_SCOPE_BLOCK` thread scope.


Example:
    
    
    __device__ T __nv_atomic_load_n(T*  pointer,
                                    int memory_order,
                                    int thread_scope = __NV_THREAD_SCOPE_SYSTEM);
    

Atomic built-in functions have the following restrictions:

  * They can only be used in device functions.

  * They cannot operate on local memory.

  * The addresses of these functions cannot be taken.

  * The `order` and `scope` arguments must be integer literals; they cannot be variables.

  * The thread scope `__NV_THREAD_SCOPE_CLUSTER` is supported on architectures `sm_90` and higher.


Example of unsupported cases:
    
    
     // Not permitted in a host function
     __host__ void bar() {
         unsigned u1 = 1, u2 = 2;
         __nv_atomic_load(&u1, &u2, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
     }
    
     // Not permitted to be applied to local memory
    __device__ void foo() {
       unsigned a = 1, b;
       __nv_atomic_load(&a, &b, __NV_ATOMIC_RELAXED, __NV_THREAD_SCOPE_SYSTEM);
    }
    
     // Not permitted as a template default argument.
     // The function address cannot be taken.
     template<void *F = __nv_atomic_load_n>
     class X {
         void *f = F; // The function address cannot be taken.
     };
    
     // Not permitted to be called in a constructor initialization list.
     class Y {
         int a;
     public:
         __device__ Y(int *b): a(__nv_atomic_load_n(b, __NV_ATOMIC_RELAXED)) {}
     };
    

#### 5.4.5.2.1. `__nv_atomic_fetch_add()`, `__nv_atomic_add()`
    
    
    __device__ T    __nv_atomic_fetch_add(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_add      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old + val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_add` returns the `old` value.

  * `__nv_atomic_add` has no return value.


The functions support the following data types:

  * `int`, `unsigned`, `unsigned long long`, `float`, `double`.


#### 5.4.5.2.2. `__nv_atomic_fetch_sub()`, `__nv_atomic_sub()`
    
    
    __device__ T    __nv_atomic_fetch_sub(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_sub      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old - val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_sub` returns the `old` value.

  * `__nv_atomic_sub` has no return value.


The functions support the following data types:

  * `int`, `unsigned`, `unsigned long long`, `float`, `double`.


#### 5.4.5.2.3. `__nv_atomic_fetch_and()`, `__nv_atomic_and()`
    
    
    __device__ T    __nv_atomic_fetch_and(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_and      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old & val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_and` returns the `old` value.

  * `__nv_atomic_and` has no return value.


The functions support the following data types:

  * Any integral type of size 4 or 8 bytes.


#### 5.4.5.2.4. `__nv_atomic_fetch_or()`, `__nv_atomic_or()`
    
    
    __device__ T    __nv_atomic_fetch_or(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_or      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old | val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_or` returns the `old` value.

  * `__nv_atomic_or` has no return value.


The functions support the following data types:

  * Any integral type of size 4 or 8 bytes.


#### 5.4.5.2.5. `__nv_atomic_fetch_xor()`, `__nv_atomic_xor()`
    
    
    __device__ T    __nv_atomic_fetch_xor(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_xor      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes `old ^ val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_xor` returns the `old` value.

  * `__nv_atomic_xor` has no return value.


The functions support the following data types:

  * Any integral type of size 4 or 8 bytes.


#### 5.4.5.2.6. `__nv_atomic_fetch_min()`, `__nv_atomic_min()`
    
    
    __device__ T    __nv_atomic_fetch_min(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_min      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes the minimum of `old` and `val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_min` returns the `old` value.

  * `__nv_atomic_min` has no return value.


The functions support the following data types:

  * `unsigned`, `int`, `unsigned long long`, `long long`.


#### 5.4.5.2.7. `__nv_atomic_fetch_max()`, `__nv_atomic_max()`
    
    
    __device__ T    __nv_atomic_fetch_max(T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_max      (T* address, T val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Computes the maximum of `old` and `val`.

  3. Stores the result back to memory at the same address.


  * `__nv_atomic_fetch_max` returns the `old` value.

  * `__nv_atomic_max` has no return value.


The functions support the following data types:

  * `unsigned`, `int`, `unsigned long long`, `long long`


#### 5.4.5.2.8. `__nv_atomic_exchange()`, `__nv_atomic_exchange_n()`
    
    
    __device__ T    __nv_atomic_exchange_n(T* address, T val,          int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_exchange  (T* address, T* val, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. `__nv_atomic_exchange_n` stores `val` to where `address` points to.

`__nv_atomic_exchange` stores `old` to where `ret` points to and stores the value located at the address `val` to where `address` points to.


  * `__nv_atomic_exchange_n` returns the `old` value.

  * `__nv_atomic_exchange` has no return value.


The functions support the following data types:

  * Any data type of size of 4, 8 or 16 bytes.

  * The 16-byte data type is supported on devices of compute capability 9.x and higher.


#### 5.4.5.2.9. `__nv_atomic_compare_exchange()`, `__nv_atomic_compare_exchange_n()`
    
    
    __device__ bool __nv_atomic_compare_exchange  (T* address, T* expected, T* desired, bool weak, int success_order, int failure_order,
                                                   int scope = __NV_THREAD_SCOPE_SYSTEM);
    
    __device__ bool __nv_atomic_compare_exchange_n(T* address, T* expected, T desired, bool weak, int success_order, int failure_order,
                                                   int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. Compare `old` with the value where `expected` points to.

  3. If they are equal, the return value is `true` and `desired` is stored to where `address` points to. Otherwise, it returns `false` and `old` is stored to where `expected` points to.


The parameter `weak` is ignored and it picks the stronger memory order between `success_order` and `failure_order` to execute the compare-and-exchange operation.

The functions support the following data types:

  * Any data type of size of 2, 4, 8 or 16 bytes.

  * The 16-byte data type is supported on devices with compute capability 9.x and higher.


#### 5.4.5.2.10. `__nv_atomic_load()`, `__nv_atomic_load_n()`
    
    
    __device__ void __nv_atomic_load  (T* address, T* ret, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ T    __nv_atomic_load_n(T* address,         int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. `__nv_atomic_load` stores `old` to where `ret` points to.

`__nv_atomic_load_n` returns `old`.


The functions support the following data types:

  * Any data type of size 1, 2, 4, 8 or 16 bytes.


`order` cannot be `__NV_ATOMIC_RELEASE` or `__NV_ATOMIC_ACQ_REL`.

#### 5.4.5.2.11. `__nv_atomic_store()`, `__nv_atomic_store_n()`
    
    
    __device__ void __nv_atomic_store  (T* address, T* val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    __device__ void __nv_atomic_store_n(T* address, T  val, int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

The functions perform the following operations in one atomic transaction:

  1. Reads the `old` value located at the address `address` in global or shared memory.

  2. `__nv_atomic_store` reads the value where `val` points to and stores to where `address` points to.

`__nv_atomic_store_n` stores `val` to where `address` points to.


`order` cannot be `__NV_ATOMIC_CONSUME`, `__NV_ATOMIC_ACQUIRE` or `__NV_ATOMIC_ACQ_REL`.

#### 5.4.5.2.12. `__nv_atomic_thread_fence()`
    
    
    __device__ void __nv_atomic_thread_fence(int order, int scope = __NV_THREAD_SCOPE_SYSTEM);
    

This atomic function establishes an ordering between memory accesses requested by this thread based on the specified memory order. The thread scope parameter specifies the set of threads that may observe the ordering effect of this operation.

## 5.4.6. Warp Functions

The following section describes the warp functions that allow threads within a warp to communicate with each other and perform computations.

Hint

It is suggested to use the `CUB` [Warp-Wide “Collective” Primitives](https://nvidia.github.io/cccl/cub/api_docs/warp_wide.html#warp-wide-collective-primitives) to perform warp operations whenever possible for efficiency, safety, and portability reasons.

### 5.4.6.1. Warp Active Mask
    
    
    unsigned __activemask();
    

The function returns a 32-bit integer mask representing all currently active threads in the calling warp. The Nth bit is set if the Nth lane in the warp is active when `__activemask()` is called. [Inactive threads](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes) are represented by 0 bits in the returned mask. Threads that have exited the program are always marked as inactive.

Warning

`__activemask()` cannot be used to determine which warp lanes execute a given branch. This function is intended for opportunistic warp-level programming and only provides an instantaneous snapshot of the active threads within a warp.
    
    
    // Check whether at least one thread's predicate evaluates to true
    if (pred) {
        // Invalid: the value of 'at_least_one' is non-deterministic
        // and could vary between executions.
        at_least_one = __activemask() > 0;
    }
    

Note that threads convergent at an `__activemask()` call are not guaranteed to remain convergent at subsequent instructions unless those instructions are warp synchronizing intrinsics (`__sync`).

For example, the compiler could reorder instructions, and the set of active threads might not be preserved:
    
    
    unsigned mask      = __activemask();              // Assume mask == 0xFFFFFFFF (all bits set, all threads active)
    int      predicate = threadIdx.x % 2 == 0;        // 1 for even threads, 0 for odd threads
    int      result    = __any_sync(mask, predicate); // Active threads might not be preserved
    

### 5.4.6.2. Warp Vote Functions
    
    
    int      __all_sync   (unsigned mask, int predicate);
    int      __any_sync   (unsigned mask, int predicate);
    unsigned __ballot_sync(unsigned mask, int predicate);
    

The warp vote functions enable the threads of a given [warp](../01-introduction/programming-model.html#programming-model-warps-simt) to perform a reduction-and-broadcast operation. These functions take an integer `predicate` as input from each non-exited thread in the warp and compare those values with zero. The results of the comparisons are then combined (reduced) across the [active threads](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes) of the warp in one of the following ways, broadcasting a single return value to each participating thread:

`__all_sync(unsigned mask, predicate)`:
    

Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for all of them.

`__any_sync(unsigned mask, predicate)`:
    

Evaluates `predicate` for all non-exited threads in `mask` and returns non-zero if `predicate` evaluates to non-zero for one or more of them.

`__ballot_sync(unsigned mask, predicate)`:
    

Evaluates `predicate` for all non-exited threads in `mask` and returns an integer whose Nth bit is set if `predicate` evaluates to non-zero for the Nth thread of the warp and the Nth thread is active. Otherwise, the Nth bit is zero.

The functions are subject to the [Warp __sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.3. Warp Match Functions

Hint

It is suggested to use the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/warp/warp_match_all.html) `cuda::device::warp_match_all()` function as a generalized and safer alternative to `__match_all_sync` function.
    
    
    unsigned __match_any_sync(unsigned mask, T value);
    unsigned __match_all_sync(unsigned mask, T value, int *pred);
    

The warp match functions perform a broadcast-and-compare operation of a variable between non-exited threads within a [warp](../01-introduction/programming-model.html#programming-model-warps-simt).

`__match_any_sync`
    

Returns the mask of non-exited threads that have the same bitwise `value` in `mask`.

`__match_all_sync`
    

Returns `mask` if all non-exited threads in `mask` have the same bitwise `value`; otherwise 0 is returned. Predicate `pred` is set to `true` if all non-exited threads in `mask` have the same bitwise `value`; otherwise the predicate is set to false.

`T` can be `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.

The functions are subject to the [Warp __sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.4. Warp Reduce Functions

Hint

It is suggested to use the `CUB` [Warp-Wide “Collective” Primitives](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html#_CPPv4I0_iEN3cub10WarpReduceE) to perform a Warp Reduction whenever possible for efficiency, safety, and portability reasons.

Supported by devices of compute capability 8.x or higher.
    
    
    T        __reduce_add_sync(unsigned mask, T value);
    T        __reduce_min_sync(unsigned mask, T value);
    T        __reduce_max_sync(unsigned mask, T value);
    
    unsigned __reduce_and_sync(unsigned mask, unsigned value);
    unsigned __reduce_or_sync (unsigned mask, unsigned value);
    unsigned __reduce_xor_sync(unsigned mask, unsigned value);
    

The `__reduce_<op>_sync` intrinsics perform a reduction operation on the data provided in `value` after synchronizing all non-exited threads named in `mask`.

`__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`
    

Returns the result of applying an arithmetic add, min, or max reduction operation on the values provided in `value` by each non-exited thread named in `mask`. `T` can be an `unsigned` or `signed` integer.

`__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`
    

Returns the result of applying a bitwise AND, OR, or XOR reduction operation on the values provided in `value` by each non-exited thread named in `mask`.

The functions are subject to the [Warp __sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Warning

These intrinsics do not provide any memory ordering.

### 5.4.6.5. Warp Shuffle Functions

Hint

It is suggested to use the [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/warp/warp_shuffle.html#libcudacxx-extended-api-warp-warp-shuffle) `cuda::device::warp_shuffle()` functions as a generalized and safer alternative to `__shfl_sync()` and `__shfl_<op>_sync()` intrinsics.
    
    
    T __shfl_sync     (unsigned mask, T value, int      srcLane,  int width=warpSize);
    T __shfl_up_sync  (unsigned mask, T value, unsigned delta,    int width=warpSize);
    T __shfl_down_sync(unsigned mask, T value, unsigned delta,    int width=warpSize);
    T __shfl_xor_sync (unsigned mask, T value, int      laneMask, int width=warpSize);
    

Warp shuffle functions exchange a value between non-exited threads within a [warp](../01-introduction/programming-model.html#programming-model-warps-simt) without the use of shared memory.

`__shfl_sync()`: Direct copy from indexed lane.
    

The intrinsic function returns the value of `value` held by the thread whose ID is given by `srcLane`.

  * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.

  * If `srcLane` is outside the range `[0, width - 1]`, the result corresponds to the value held by the `srcLane % width`, which is within the same subsection.


* * *

`__shfl_up_sync()`: Copy from a lane with a lower ID than the caller’s.
    

The intrinsic function calculates a source lane ID by subtracting `delta` from the caller’s lane ID. The value of `value` held by the resulting lane ID is returned: in effect, `value` is shifted up the warp by `delta` lanes.

  * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.

  * The source lane index will not wrap around the value of `width`, so the lower `delta` lanes will remain unchanged.   


* * *

`__shfl_down_sync()`: Copy from a lane with a higher ID than the caller’s.
    

The intrinsic function calculates a source lane ID by adding `delta` to the caller’s lane ID. The value of `value` held by the resulting lane ID is returned: this has the effect of shifting `value` down the warp by `delta` lanes.

  * If `width` is less than `warpSize`, then each subsection of the warp behaves as a separate entity with a starting logical lane ID of 0.

  * As for `__shfl_up_sync()`, the ID number of the source lane will not wrap around the value of width and so the upper `delta` lanes will effectively remain unchanged.   


* * *

`__shfl_xor_sync()`: Copy from a lane based on bitwise XOR of own lane ID.
    

The intrinsic function calculates a source lane ID by performing a bitwise XOR of the caller’s lane ID and `laneMask`: the value of `value` held by the resulting lane ID is returned. This mode implements a butterfly addressing pattern, which is used in tree reduction and broadcast.

  * If `width` is less than `warpSize`, then each group of `width` consecutive threads are able to access elements from earlier groups. However, if they attempt to access elements from later groups of threads their own value of `value` will be returned.


* * *

`T` can be:

  * `int`, `unsigned`, `long`, `unsigned long`, `long long`, `unsigned long long`, `float` or `double`.

  * `__half` and `__half2` with the `cuda_fp16.h` header included.

  * `__nv_bfloat16` and `__nv_bfloat162` with the `cuda_bf16.h` header included.


Threads may only read data from another thread that is actively participating in the intrinsics. If the target thread is [inactive](../03-advanced/advanced-kernel-programming.html#simt-architecture-notes), the retrieved value is undefined.

`width` must be a power of two in the range `[1, warpSize]`, namely 1, 2, 4, 8, 16, or 32. Other values will produce undefined results.

The functions are subject to the [Warp __sync Intrinsic Constraints](#warp-sync-intrinsic-constraints).

Examples of valid warp shuffle usage:
    
    
    int laneId = threadIdx.x % warpSize;
    int data   = ...
    
    // all warp threads get 'data' from lane 0
    int result1 = __shfl_sync(0xFFFFFFFF, data, 0);
    
    if (laneId < 4) {
        // lanes 0, 1, 2, 3 get 'data' from lane 1
        int result2 = __shfl_sync(0xb1111, data, 1);
    }
    
    // lanes [0 - 15] get 'data' from lane 0
    // lanes [16 - 31] get 'data' from lane 16
    int result3 = __shfl_sync(0xFFFFFFFF, value, warpSize / 2);
    
    // each lane gets 'data' from the lane two positions above
    // lanes 30, 31 get their original value
    int result4 = __shfl_down_sync(0xFFFFFFFF, data, 2);
    

Examples of invalid warp shuffle usage:
    
    
    int laneId = threadIdx.x % warpSize;
    int value  = ...
     // undefined behavior: lane 0 does not participate in the call
    int result = (laneId > 0) ? __shfl_sync(0xFFFFFFFF, value, 0) : 0;
    
    if (laneId <= 4) {
        // undefined behavior: destination lanes 5, 6 are not active for lanes 3, 4
        result = __shfl_down_sync(0b11111, value, 2);
    }
    
    // undefined behavior: width is not a power of 2
    __shfl_sync(0xFFFFFFFF, value, 0, /*width=*/31);
    

Warning

These intrinsics do not imply a memory barrier. They do not guarantee any memory ordering.

Example 1: Broadcast of a single value across a warp

CUDA C++
    
    
    #include <cassert>
    #include <cuda/warp>
    
    __global__ void warp_broadcast_kernel(int input) {
        int laneId = threadIdx.x % 32;
        int value;
        if (laneId == 0) { // unused variable for all threads except lane 0
            value = input;
        }
        value = cuda::device::warp_shuffle_idx(value, 0); // Synchronize all threads in warp, and get "value" from lane 0
        assert(value == input);
    }
    
    int main() {
        warp_broadcast_kernel<<<1, 32>>>(1234);
        cudaDeviceSynchronize();
        return 0;
    }
    

Intrinsics
    
    
    #include <assert.h>
    
    __global__ void warp_broadcast_kernel(int input) {
        int laneId = threadIdx.x % 32;
        int value;
        if (laneId == 0) { // unused variable for all threads except lane 0
            value = input;
        }
        value = __shfl_sync(0xFFFFFFFF, value, 0); // Synchronize all threads in warp, and get "value" from lane 0
        assert(value == input);
    }
    
    int main() {
        warp_broadcast_kernel<<<1, 32>>>(1234);
        cudaDeviceSynchronize();
        return 0;
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/E3E3Y5e4e).

Example 2: Inclusive plus-scan across sub-partitions of 8 threads

Hint

It is suggested to use the [cub::WarpScan](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html) function for efficient and generalized warp scan functions.

CUDA C++
    
    
    #include <cstdio>
    #include <cub/cub.cuh>
    
    __global__ void scan_sub_partition_with_8_threads_kernel() {
        using WarpScan    = cub::WarpScan<int, 8>;
        using TempStorage = typename WarpScan::TempStorage;
        __shared__ TempStorage temp_storage;
    
        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        int partial_sum;
        WarpScan(temp_storage).InclusiveSum(value, partial_sum);
        printf("Thread %d final value = %d\n", threadIdx.x, partial_sum);
    }
    
    int main() {
        scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

Intrinsics
    
    
    #include <stdio.h>
    
    __global__ void scan_sub_partition_with_8_threads_kernel() {
        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        // Loop to accumulate scan within my partition.
        // Scan requires log2(8) == 3 steps for 8 threads
        for (int delta = 1; delta <= 4; delta *= 2) {
            int tmp         = __shfl_up_sync(0xFFFFFFFF, value, delta, /*width=*/8); // read from laneId - delta
            int source_lane = laneId % 8 - delta;
            if (source_lane >= 0) // lanes with 'source_lane < 0' have their value unchanged
                value += tmp;
        }
        printf("Thread %d final value = %d\n", threadIdx.x, value);
    }
    
    int main() {
        scan_sub_partition_with_8_threads_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/Tohd38edc).

Example 3: Reduction across a warp

Hint

It is suggested to use the [cub::WarpReduce](https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpReduce.html) function for efficient and generalized warp reduction functions.

CUDA C++
    
    
    #include <cstdio>
    #include <cub/cub.cuh>
    #include <cuda/warp>
    
    __global__ void warp_reduce_kernel() {
        using WarpReduce  = cub::WarpReduce<int>;
        using TempStorage = typename WarpReduce::TempStorage;
        __shared__ TempStorage temp_storage;
    
        int laneId     = threadIdx.x % 32;
        int value      = 31 - laneId; // starting value to accumulate
        auto aggregate = WarpReduce(temp_storage).Sum(value);
        aggregate      = cuda::device::warp_shuffle_idx(aggregate, 0);
        printf("Thread %d final value = %d\n", threadIdx.x, aggregate);
    }
    
    int main() {
        warp_reduce_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

Intrinsics
    
    
    #include <stdio.h>
    
    __global__ void warp_reduce_kernel() {
        int laneId = threadIdx.x % 32;
        int value  = 31 - laneId; // starting value to accumulate
        // Use XOR mode to perform butterfly reduction
        // A full-warp reduction requires log2(32) == 5 steps
        for (int i = 1; i <= 16; i *= 2)
            value += __shfl_xor_sync(0xFFFFFFFF, value, i);
        // "value" now contains the sum across all threads
        printf("Thread %d final value = %d\n", threadIdx.x, value);
    }
    
    int main() {
        warp_reduce_kernel<<<1, 32>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/T94nfGMzG).

### 5.4.6.6. Warp `__sync` Intrinsic Constraints

All warp `__sync` intrinsics, such as:

  * `__shfl_sync`, `__shfl_up_sync`, `__shfl_down_sync`, `__shfl_xor_sync`

  * `__match_any_sync`, `__match_all_sync`

  * `__reduce_add_sync`, `__reduce_min_sync`, `__reduce_max_sync`, `__reduce_and_sync`, `__reduce_or_sync`, `__reduce_xor_sync`

  * `__syncwarp`


use the `mask` parameter to indicate which warp threads participate in the call. This parameter ensures proper convergence before the hardware executes the intrinsic.

Each bit in the `mask` corresponds to a thread’s lane ID (`threadIdx.x % warpSize`). The intrinsic waits until all non-exited warp threads specified in the `mask` reach the call.

The following constraints must be met for correct execution:

  * Each calling thread must have its corresponding bit set in the `mask`.

  * Each non-calling thread must have its corresponding bit set to zero in the `mask`. Exited threads are ignored.

  * All non-exited threads specified in the `mask` must execute the intrinsic with the same `mask` value.

  * Warp threads may call the intrinsic concurrently with different `mask` values, provided the masks are disjoint. Such condition is valid even in divergent control flow.


The behavior of warp `__sync` functions is invalid, such as kernel hang, or undefined if:

  * A calling thread is not specified in the `mask`.

  * A non-exited thread specified in the `mask` fails to either eventually exit or call the intrinsic at the same program point with the same `mask` value.

  * In conditional code, all conditions must evaluate identically across all non-exited threads specified in the `mask`.


Note

The intrinsics achieve the best efficiency when all warp threads participate in the call, namely when the `mask` is set to `0xFFFFFFFF`.

Examples of valid warp intrinsics usage:
    
    
    __global__ void valid_examples() {
        if (threadIdx.x < 4) {        // threads 0, 1, 2, 3 are active
            __all_sync(0b1111, pred); // CORRECT, threads 0, 1, 2, 3 participate in the call
        }
    
        if (threadIdx.x == 0)
            return; // exit
        // CORRECT, all non-exited threads participate in the call
        __all_sync(0xFFFFFFFF, pred);
    }
    

Disjoint `mask` examples:
    
    
    __global__ void example_syncwarp_with_mask(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];
    
            unsigned mask = threadIdx.x < 16 ? 0xFFFF : 0xFFFF0000; // CORRECT
            __syncwarp(mask);
            if (threadIdx.x == 0 || threadIdx.x == 16)
                output_data[threadIdx.x] = shared_data[threadIdx.x + 1];
        }
    }
    
    
    
    __global__ void example_syncwarp_with_mask_branches(int* input_data, int* output_data) {
        if (threadIdx.x < warpSize) {
            __shared__ int shared_data[warpSize];
            shared_data[threadIdx.x] = input_data[threadIdx.x];
    
            if (threadIdx.x < 16) {
                unsigned mask = 0xFFFF; // CORRECT
                __syncwarp(mask);
                output_data[threadIdx.x] = shared_data[15 - threadIdx.x];
            }
            else {
                unsigned mask = 0xFFFF0000; // CORRECT
                __syncwarp(mask);
                output_data[threadIdx.x] = shared_data[31 - threadIdx.x];
            }
        }
    }
    

Examples of invalid warp intrinsics usage:
    
    
    if (threadIdx.x < 4) {           // threads 0, 1, 2, 3 are active
        __all_sync(0b0000011, pred); // WRONG, threads 2, 3 are active but not set in mask
        __all_sync(0b1111111, pred); // WRONG, threads 4, 5, 6 are not active but set in mask
    }
    
    // WRONG, participating threads have a different and overlapping mask
    __all_sync(threadIdx.x == 0 ? 1 : 0xFFFFFFFF, pred);
    

## 5.4.7. CUDA-Specific Macros

### 5.4.7.1. `__CUDA_ARCH__`

The macro `__CUDA_ARCH__` represents the [virtual architecture](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#virtual-architecture-macros) of the NVIDIA GPU for which the code is being compiled. Its value may differ from the device’s actual compute capability. This macro enables the writing of code paths that are specialized for particular GPU architectures, which may be necessary for optimal performance or to use architecture-specific features and instructions. The macro can also be used to distinguish between host and device code.

`__CUDA_ARCH__` is only defined in device code, namely in the `__device__`, `__host__ __device__`, and `__global__` functions. The value of the macro is associated with the `nvcc` option `compute_<version>`, with the relation `__CUDA_ARCH__ = <version> * 10`.

Example:
    
    
    nvcc --generate-code arch=compute_80,code=sm_90 prog.cu
    

defines `__CUDA_ARCH__` as `800`.

* * *

`__CUDA_ARCH__` **Constraints**

**1.** The type signatures of the following entities shall not depend on whether `__CUDA_ARCH__` is defined, nor on its value.

  * `__global__` functions and function templates.

  * `__device__` and `__constant__` variables.

  * Textures and surfaces.


Example:
    
    
    #if !defined(__CUDA_ARCH__)
        typedef int my_type;
    #else
        typedef double my_type;
    #endif
    
    __device__ my_type my_var;           // ERROR: my_var's type depends on __CUDA_ARCH__
    
    __global__ void kernel(my_type in) { // ERROR: kernel's type depends on __CUDA_ARCH__
        ...
    }
    

**2.** If a `__global__` function template is instantiated and launched from the host, then it must be instantiated with the same template arguments, regardless of whether `__CUDA_ARCH__` is defined or its value.

Example:
    
    
    __device__ int result;
    
    template <typename T>
    __global__ void kernel(T in) {
        result = in;
    }
    
    __host__ __device__ void host_device_function(void) {
    #if !defined(__CUDA_ARCH__)
        kernel<<<1, 1>>>(1); // ERROR: "kernel<int>" instantiation only
                                //        when __CUDA_ARCH__ is undefined!
    #endif
    }
    
    int main(void) {
        host_device_function();
        cudaDeviceSynchronize();
        return 0;
    }
    

**3.** In separate compilation mode, the presence or absence of a function or variable definition with external linkage shall not depend on the definition of `__CUDA_ARCH__` or on its value.

Example:
    
    
    #if !defined(__CUDA_ARCH__)
        void host_function(void) {} // ERROR: The definition of host_function()
                                    //        is only present when __CUDA_ARCH__
                                    //        is undefined
    #endif
    

**4.** In separate compilation, the preprocessor macro `__CUDA_ARCH__` must not be used in headers to prevent objects from having different behaviors. Alternatively, all objects must be compiled for the same virtual architecture. If a weak or template function is defined in a header and its behavior depends on `__CUDA_ARCH__`, then instances of that function in different objects could conflict if those objects are compiled for different compute architectures.

For example, if a header file `a.h` contains:
    
    
    template<typename T>
    __device__ T* get_ptr() {
    #if __CUDA_ARCH__ == 900
        return nullptr; /* no address */
    #else
        __shared__ T arr[256];
        return arr;
    #endif
    }
    

Then if `a.cu` and `b.cu` both include `a.h` and instantiate `get_ptr()` for the same type, and `b.cu` expects a non-`NULL` address, and compile with:
    
    
    nvcc -arch=compute_70 -dc a.cu
    nvcc -arch=compute_80 -dc b.cu
    nvcc -arch=sm_80 a.o b.o
    
    Only one version of the ``get_ptr()`` function is used at link time, so the behavior depends on which version is chosen. To avoid this issue, either ``a.cu`` and ``b.cu`` must be compiled for the same compute architecture, or ``__CUDA_ARCH__`` should not be used in the shared header function.
    

The compiler does not guarantee that a diagnostic will be generated for the unsupported uses of `__CUDA_ARCH__` described above.

### 5.4.7.2. `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__`

The macros `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are defined to identify GPU devices with [architecture-](compute-capabilities.html#compute-capabilities-architecture-specific-features) and [family-](compute-capabilities.html#compute-capabilities-family-specific-features) specific features, respectively. See [Feature Set Compiler Targets](compute-capabilities.html#compute-capabilities-feature-set-compiler-targets) section for more information.

Similarly to `__CUDA_ARCH__`, `__CUDA_ARCH_SPECIFIC__` and `__CUDA_ARCH_FAMILY_SPECIFIC__` are only defined in the device code, namely in the `__device__`, `__host__ __device__`, and `__global__` functions. The macros are associated with the `nvcc` options `compute_<version>a` and `compute_<version>f`.
    
    
    nvcc --generate-code arch=compute_100a,code=sm_100a prog.cu
    

  * `__CUDA_ARCH__ == 1000`.

  * `__CUDA_ARCH_SPECIFIC__ == 1000`.

  * `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.


    
    
    nvcc --generate-code arch=compute_100f,code=sm_103f prog.cu
    

  * `__CUDA_ARCH__ == 1000`.

  * `__CUDA_ARCH_FAMILY_SPECIFIC__ == 1000`.

  * `__CUDA_ARCH_SPECIFIC__` is not defined.


    
    
    nvcc -arch=sm_100 prog.cu
    

  * `__CUDA_ARCH__ == 1000`.

  * `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.

  * `__CUDA_ARCH_SPECIFIC__` is not defined.


    
    
    nvcc -arch=sm_100a prog.cu
    # equivalent to:
    nvcc --generate-code arch=sm_100a,compute_100,compute_100a prog.cu
    

  * `__CUDA_ARCH__ == 1000`.

  * `__CUDA_ARCH_FAMILY_SPECIFIC__` is not defined.

  * `__CUDA_ARCH_SPECIFIC__ == 1000` and `__CUDA_ARCH_SPECIFIC__` not defined are both generated.


### 5.4.7.3. CUDA Feature Testing Macros

`nvcc` provides the following preprocessor macros for feature testing. The macros are defined when a particular feature is supported by the CUDA front-end compiler.

  * `__CUDACC_DEVICE_ATOMIC_BUILTINS__`: Supports [device atomic compiler builtins](#built-in-atomic-functions).

  * `__NVCC_DIAG_PRAGMA_SUPPORT__`: Supports [diagnostic control pragmas](#nv-diagnostic-pragmas).

  * `__CUDACC_EXTENDED_LAMBDA__`: Supports [extended lambdas](cpp-language-support.html#extended-lambdas). Enabled by `--expt-extended-lambda` or `--extended-lambda` flag.

  * `__CUDACC_RELAXED_CONSTEXPR__`: Support for [relaxed constexpr functions](cpp-language-support.html#constexpr-functions). Enabled by the `--expt-relaxed-constexpr` flag.


### 5.4.7.4. `__nv_pure__` Attribute

In C/C++, a pure function has no side effects on its parameters and can access global variables, though it does not modify them.

CUDA provides `__nv_pure__` attribute supported for both host and device functions. The compiler translates `__nv_pure__` to the `pure` GNU attribute or to the Microsoft Visual Studio `noalias` attribute.
    
    
    __device__ __nv_pure__
    int add(int a, int b) {
        return a + b;
    }
    

## 5.4.8. CUDA-Specific Functions

### 5.4.8.1. Address Space Predicate Functions

Address space predicate functions are used to determine the address space of a pointer.

Hint

It is suggested to use the `cuda::device::is_address_from()` and `cuda::device::is_object_from()` functions provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory/is_address_from.html) as a portable and safer alternative to Address Space Predicate intrinsic functions.
    
    
    __device__ unsigned __isGlobal      (const void* ptr);
    __device__ unsigned __isShared      (const void* ptr);
    __device__ unsigned __isConstant    (const void* ptr);
    __device__ unsigned __isGridConstant(const void* ptr);
    __device__ unsigned __isLocal       (const void* ptr);
    

The functions return `1` if `ptr` contains the generic address of an object in the specified address space, `0` otherwise. Their behavior is unspecified if the argument is a `NULL` pointer.

  * `__isGlobal()`: global memory space.

  * `__isShared()`: shared memory space.

  * `__isConstant()`: constant memory space.

  * `__isGridConstant()`: kernel parameter annotated with `__grid_constant__`.

  * `__isLocal()`: local memory space.


### 5.4.8.2. Address Space Conversion Functions

CUDA pointers (`T*`) can access objects regardless of where the objects are stored. For example, an `int*` can access `int` objects whether they reside in global or shared memory.

Address space conversion functions are used to convert between generic addresses and addresses in specific address spaces. These functions are useful when the compiler cannot determine a pointer’s address space, for example, when crossing translation units or interacting with PTX instructions.
    
    
    __device__ size_t __cvta_generic_to_global  (const void* ptr); // PTX: cvta.to.global
    __device__ size_t __cvta_generic_to_shared  (const void* ptr); // PTX: cvta.to.shared
    __device__ size_t __cvta_generic_to_constant(const void* ptr); // PTX: cvta.to.const
    __device__ size_t __cvta_generic_to_local   (const void* ptr); // PTX: cvta.to.local
    
    
    
    __device__ void* __cvta_global_to_generic  (size_t raw_ptr); // PTX: cvta.global
    __device__ void* __cvta_shared_to_generic  (size_t raw_ptr); // PTX: cvta.shared
    __device__ void* __cvta_constant_to_generic(size_t raw_ptr); // PTX: cvta.const
    __device__ void* __cvta_local_to_generic   (size_t raw_ptr); // PTX: cvta.local
    

As an example of inter-operating with PTX instructions, the `ld.shared.s32 r0, [ptr];` PTX instruction expects `ptr` to refer to the shared memory address space. A CUDA program with an `int*` pointer to an object in `__shared__` memory needs to convert this pointer to the shared address space before passing it to the PTX instruction by calling `__cvta_generic_to_shared` as follows:
    
    
    __shared__ int smem_var;
    smem_var        = 42;
    size_t smem_ptr = __cvta_generic_to_shared(&smem_var);
    int    output;
    asm volatile("ld.shared.s32 %0, [%1];" : "=r"(output) : "l"(smem_ptr) : "memory");
    assert(output == 42);
    

A common optimization that exploits these address representations is reducing data structure size by leveraging the fact that the address ranges of shared, local, and constant spaces are smaller than 32 bits, which allows storing 32-bit addresses instead of 64-bit pointers and save registers. Additionally, 32-bit arithmetic is faster than 64-bit arithmetic. To obtain the 32-bit integer representation of these addresses, truncate the 64-bit value to 32 bits by casting from an unsigned 64-bit integer to an unsigned 32-bit integer:
    
    
    __shared__ int smem_var;
    uint32_t       smem_ptr_32bit = static_cast<uint32_t>(__cvta_generic_to_shared(&smem_var));
    

To recover a generic address from such a 32-bit representation, zero-extend the address back to an unsigned 64-bit integer and then call the corresponding address space conversion function:
    
    
    size_t smem_ptr_64bit = static_cast<size_t>(smem_ptr_32bit); // zero-extend to 64 bits
    void*  generic_ptr    = __cvta_shared_to_generic(smem_ptr_64bit);
    assert(generic_ptr == &smem_var);
    

* * *

### 5.4.8.3. Low-Level Load and Store Functions
    
    
    T __ldg(const T* address);
    

The function `__ldg()` performs a read-only L1/Tex cache load. It supports all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

* * *
    
    
    T __ldcg(const T* address);
    T __ldca(const T* address);
    T __ldcs(const T* address);
    T __ldlu(const T* address);
    T __ldcv(const T* address);
    

The functions perform a load using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

* * *
    
    
    void __stwb(T* address, T value);
    void __stcg(T* address, T value);
    void __stcs(T* address, T value);
    void __stwt(T* address, T value);
    

The functions perform a store using the cache operator specified in the [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) guide. They support all C++ fundamental types, CUDA vector types (except x3 components), and extended floating-point types, such as `__half`, `__half2`, `__nv_bfloat16`, and `__nv_bfloat162`.

### 5.4.8.4. `__trap()`

Hint

It is suggested to use the `cuda::std::terminate()` function provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) ([C++ reference](https://en.cppreference.com/w/cpp/error/terminate.html)) as a portable alternative to `__trap()`.

A trap operation can be initiated by calling the `__trap()` function from any device thread.
    
    
    void __trap();
    

Execution of the kernel is aborted, raising an interrupt in the host program. Calling `__trap()` results in a corrupted CUDA context, causing subsequent CUDA calls and kernel invocations to fail.

### 5.4.8.5. `__nanosleep()`
    
    
    __device__ void __nanosleep(unsigned nanoseconds);
    

The function `__nanosleep(ns)` suspends the thread for a sleep duration of approximately `ns` nanoseconds. The maximum sleep duration is approximately one millisecond.

Example:

The following code implements a mutex with exponential back-off.
    
    
    __device__ void mutex_lock(unsigned* mutex) {
        unsigned ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1) {
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
        }
    }
    
    __device__ void mutex_unlock(unsigned *mutex) {
        atomicExch(mutex, 0);
    }
    

### 5.4.8.6. Dynamic Programming eXtension (DPX) Instructions

The DPX set of functions enables finding minimum and maximum values, as well as fused addition and minimum/maximum for up to three 16- or 32-bit signed or unsigned integer parameters. There is an optional ReLU, namely clamping to zero, feature.

Comparison functions:

  * Three parameters. Semantic: `max(a, b, c)`, `min(a, b, c)`.


    
    
         int __vimax3_s32  (     int,      int,      int);
    unsigned __vimax3_s16x2(unsigned, unsigned, unsigned);
    unsigned __vimax3_u32  (unsigned, unsigned, unsigned);
    unsigned __vimax3_u16x2(unsigned, unsigned, unsigned);
    
         int __vimin3_s32  (     int,      int,      int);
    unsigned __vimin3_s16x2(unsigned, unsigned, unsigned);
    unsigned __vimin3_u32  (unsigned, unsigned, unsigned);
    unsigned __vimin3_u16x2(unsigned, unsigned, unsigned);
    

  * Two parameters, with ReLU. Semantic: `max(a, b, 0)`, `max(min(a, b), 0)`.


    
    
         int __vimax_s32_relu  (     int,      int);
    unsigned __vimax_s16x2_relu(unsigned, unsigned);
    
         int __vimin_s32_relu  (     int,      int);
    unsigned __vimin_s16x2_relu(unsigned, unsigned);
    

  * Three parameters, with ReLU. Semantic: `max(a, b, c, 0)`, `max(min(a, b, c), 0)`.


    
    
         int __vimax3_s32_relu  (     int,      int,      int);
    unsigned __vimax3_s16x2_relu(unsigned, unsigned, unsigned);
    
         int __vimin3_s32_relu  (     int,      int,      int);
    unsigned __vimin3_s16x2_relu(unsigned, unsigned, unsigned);
    

  * Two parameters, also returning which parameter was smaller/larger:


    
    
         int __vibmax_s32  (     int,      int, bool* pred);
    unsigned __vibmax_u32  (unsigned, unsigned, bool* pred);
    unsigned __vibmax_s16x2(unsigned, unsigned, bool* pred);
    unsigned __vibmax_u16x2(unsigned, unsigned, bool* pred);
    
         int __vibmin_s32  (     int,      int, bool* pred);
    unsigned __vibmin_u32  (unsigned, unsigned, bool* pred);
    unsigned __vibmin_s16x2(unsigned, unsigned, bool* pred);
    unsigned __vibmin_u16x2(unsigned, unsigned, bool* pred);
    

Fused addition and minimum/maximum:

  * Three parameters, comparing (first + second) with the third. Semantic: `max(a + b, c)`, `min(a + b, c)`


    
    
         int __viaddmax_s32  (     int,     int,       int);
    unsigned __viaddmax_s16x2(unsigned, unsigned, unsigned);
    unsigned __viaddmax_u32  (unsigned, unsigned, unsigned);
    unsigned __viaddmax_u16x2(unsigned, unsigned, unsigned);
    
         int __viaddmin_s32  (     int,     int,       int);
    unsigned __viaddmin_s16x2(unsigned, unsigned, unsigned);
    unsigned __viaddmin_u32  (unsigned, unsigned, unsigned);
    unsigned __viaddmin_u16x2(unsigned, unsigned, unsigned);
    

  * Three parameters, with ReLU, comparing (first + second) with the third and a zero. Semantic: `max(a + b, c, 0)`, `max(min(a + b, c), 0)`


    
    
         int __viaddmax_s32_relu  (     int,      int,      int);
    unsigned __viaddmax_s16x2_relu(unsigned, unsigned, unsigned);
    
         int __viaddmin_s32_relu  (     int,      int,      int);
    unsigned __viaddmin_s16x2_relu(unsigned, unsigned, unsigned);
    

These instructions are hardware-accelerated or software emulated depending on compute capability. See [Arithmetic Instructions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions) section for the compute capability requirements.

The full API can be found in [CUDA Math API documentation](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SIMD.html).

* * *

The DPX is an exceptionally useful tool for implementing dynamic programming algorithms such as the Smith-Waterman and Needleman-Wunsch algorithms in genomics and the Floyd-Warshall algorithm in route optimization.

Maximum value of three signed 32-bit integers, with ReLU:
    
    
    int a           = -15;
    int b           = 8;
    int c           = 5;
    int max_value_0 = __vimax3_s32_relu(a, b, c); // max(-15, 8, 5, 0) = 8
    int d           = -2;
    int e           = -4;
    int max_value_1 = __vimax3_s32_relu(a, d, e); // max(-15, -2, -4, 0) = 0
    

Minimum value of the sum of two 32-bit signed integers, another 32-bit signed integer and a zero (ReLU):
    
    
    int a           = -5;
    int b           = 6;
    int c           = -2;
    int max_value_0 = __viaddmax_s32_relu(a, b, c); // max(-5 + 6, -2, 0) = max(1, -2, 0) = 1
    int d           = 4;
    int max_value_1 = __viaddmax_s32_relu(a, d, c); // max(-5 + 4, -2, 0) = max(-1, -2, 0) = 0
    

Minimum value of two unsigned 32-bit integers and determining which value is smaller:
    
    
    unsigned a = 9;
    unsigned b = 6;
    bool     smaller_value;
    unsigned min_value = __vibmin_u32(a, b, &smaller_value); // min_value is 6, smaller_value is true
    

Maximum values of three pairs of unsigned 16-bit integers:
    
    
    unsigned a         = 0x00050002;
    unsigned b         = 0x00070004;
    unsigned c         = 0x00020006;
    unsigned max_value = __vimax3_u16x2(a, b, c); // max(5, 7, 2) and max(2, 4, 6), so max_value is 0x00070006
    

## 5.4.9. Compiler Optimization Hints

Compiler optimization hints decorate code with additional information to help the compiler optimize generated code.

  * The built-in functions are always available in the device code.

  * Host code support depends on the host compiler.


### 5.4.9.1. `#pragma unroll`

The compiler unrolls small loops with a known trip count by default. However, the `#pragma unroll` directive can be used to control the unrolling of any given loop. This directive must be placed immediately before the loop and only applies to that loop.

An integral constant expression may optionally follow. The following are cases for an integral constant expression:

  * If it is absent, the loop will be completely unrolled if its trip count is constant.

  * If it evaluates to `0` or `1`, the loop will not be unrolled.

  * If it is a non-positive integer or greater than `INT_MAX`, the pragma will be ignored, and a warning will be issued.


Examples:
    
    
    struct MyStruct {
        static constexpr int value = 4;
    };
    
    inline constexpr int Count = 4;
    
    __device__ void foo(int* p1, int* p2) {
        // no argument specified, the loop will be completely unrolled
        #pragma unroll
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 2;
    
        // unroll value = 5
        #pragma unroll (Count + 1)
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 4;
    
        // unroll value = 1, loop unrolling disabled
        #pragma unroll 1
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 8;
    
        // unroll value = 4
        #pragma unroll (MyStruct::value)
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 16;
    
        // negative value, pragma unroll ignored
        #pragma unroll -1
        for (int i = 0; i < 12; ++i)
            p1[i] += p2[i] * 2;
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/fPMK55PxE).

### 5.4.9.2. `__builtin_assume_aligned()`

Hint

It is suggested to use the `cuda::std::assume_aligned()` function provided by [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) ([C++ reference](https://en.cppreference.com/w/cpp/memory/assume_aligned.html)) as a portable and safer alternative to the built-in functions.
    
    
    void* __builtin_assume_aligned(const void* ptr, size_t align)
    void* __builtin_assume_aligned(const void* ptr, size_t align, <integral type> offset)
    

The built-in functions enable the compiler to assume that the returned pointer is aligned to at least `align` bytes.

  * The three parameter version enables the compiler to assume that `(char*) ptr - offset` is aligned to at least `align` bytes.


`align` must be a power of two and an integer literal.

Examples:
    
    
    void* res1 = __builtin_assume_aligned(ptr, 32);    // compiler can assume 'res1' is at least 32-byte aligned
    void* res2 = __builtin_assume_aligned(ptr, 32, 8); // compiler can assume 'res2 = (char*) ptr - 8' is at least 32-byte aligned
    

### 5.4.9.3. `__builtin_assume()` and `__assume()`
    
    
    void __builtin_assume(bool predicate)
    void __assume        (bool predicate) // only with Microsoft Compiler
    

The built-in function enables the compiler to assume that the boolean argument is true. If the argument is false at runtime, the behavior is undefined. Note that if the argument has side effects, the behavior is unspecified.

Example:
    
    
    __device__ bool is_greater_than_zero(int value) {
        return value > 0;
    }
    
    __device__ bool f(int value) {
        __builtin_assume(value > 0);
        return is_greater_than_zero(value); // returns true, without evaluating the condition
    }
    

### 5.4.9.4. `__builtin_expect()`
    
    
    long __builtin_expect(long input, long expected)
    

The built-in function tells the compiler that `input` is expected to equal `expected`, and returns the value of `input`. It is typically used to provide branch prediction information to the compiler. It behaves like the C++20 `[[likely]]` and `[[unlikely]]` [attributes](https://en.cppreference.com/w/cpp/language/attributes/likely).

Example:
    
    
    // indicate to the compiler that likely "var == 0"
    if (__builtin_expect(var, 0))
        doit();
    

### 5.4.9.5. `__builtin_unreachable()`
    
    
    void __builtin_unreachable(void)
    

The built-in function tells the compiler that the control flow will never reach the point at which the function is called. If the control flow does reach this point at runtime, the program has undefined behavior.

This function is useful for avoiding code generation of unreachable branches and disabling compiler warnings for unreachable code.

Example:
    
    
    // indicates to the compiler that the default case label is never reached.
    switch (in) {
        case 1:  return 4;
        case 2:  return 10;
        default: __builtin_unreachable();
    }
    

### 5.4.9.6. Custom ABI Pragmas

The `#pragma nv_abi` directive enables applications compiled in [separate compilation](../02-basics/nvcc.html#nvcc-separate-compilation) mode to achieve performance similar to that of [whole program compilation](../02-basics/nvcc.html#nvcc-separate-compilation) by preserving the number of registers used by a function.

The syntax for using this pragma is as follows, where `EXPR` refers to any integral constant expression:
    
    
    #pragma nv_abi preserve_n_data(EXPR) preserve_n_control(EXPR)
    

  * The arguments that follow `#pragma nv_abi` are optional and may be provided in any order; however, at least one argument is required.

  * The `preserve_n` arguments limit the number of registers preserved during a function call:

    * `preserve_n_data(EXPR)` limits the number of data registers.

    * `preserve_n_control(EXPR)` limits the number of control registers.


The `#pragma nv_abi` directive can be placed immediately before a device function declaration or definition.
    
    
    #pragma nv_abi preserve_n_data(16)
    __device__ void dev_func();
    
    #pragma nv_abi preserve_n_data(16) preserve_n_control(8)
    __device__ int dev_func() {
        return 0;
    }
    

Alternatively, it can be placed directly before an indirect function call within a C++ expression statement inside a device function. Note that while indirect function calls to free functions are supported, indirect calls to function references or class member functions are not supported.
    
    
    __device__ int dev_func1();
    
    struct MyStruct {
        __device__ int member_func2();
    };
    
    __device__ void test() {
        auto* dev_func_ptr = &dev_func1; // type: int (*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v1 = dev_func_ptr();         // CORRECT, indirect call
    
        #pragma nv_abi preserve_n_control(8)
        int v2 = dev_func1();            // WRONG, direct call; the pragma has no effect
                                         // dev_func1 has type: int(void)
    
        auto& dev_func_ref = &dev_func1; // type: int (&)(void)
        #pragma nv_abi preserve_n_control(8)
        int v3 = dev_func_ref();         // WRONG, call to a reference
                                         // the pragma has no effect
    
        auto member_function_ptr = &MyStruct::member_func2; // type: int (MyStruct::*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v4 = member_function_ptr();  // WRONG, indirect call to member function
                                         // the pragma has no effect
    }
    

When applied to a device function’s declaration or definition, the pragma modifies the custom ABI properties for any calls to that function. When placed at an indirect function call site, it affects the ABI properties only for that specific call. Note that the pragma only affects indirect function calls when placed at a call site; it has no effect on direct function calls.
    
    
    #pragma nv_abi preserve_n_control(8)
    __device__ int dev_func3();
    
    __device__ int dev_func4();
    
    __device__ void test() {
        int v1 = dev_func3();            // CORRECT, the pragma affects the direct call
    
        auto* dev_func_ptr = &dev_func4; // type: int (*)(void)
        #pragma nv_abi preserve_n_control(8)
        int v2 = dev_func_ptr();         // CORRECT, the pragma affects the indirect call
    
        int v3 = dev_func_ptr();         // WRONG, the pragma has no effect
    }
    

Note that a program is ill-formed if the pragma arguments for a function declaration and its corresponding definition do not match.

## 5.4.10. Debugging and Diagnostics

### 5.4.10.1. Assertion
    
    
    void assert(int expression);
    

The `assert()` macro stops kernel execution if `expression` is equal to zero. If the program is run within a debugger, a breakpoint is triggered, allowing the debugger to be used to inspect the current state of the device. Otherwise, each thread for which `expression` is equal to zero prints a message to stderr after synchronizing with the host via `cudaDeviceSynchronize()`, `cudaStreamSynchronize()`, or `cudaEventSynchronize()`. The format of this message is as follows:
    
    
    <filename>:<line number>:<function>:
    block: [blockIdx.x,blockIdx.y,blockIdx.z],
    thread: [threadIdx.x,threadIdx.y,threadIdx.z]
    Assertion `<expression>` failed.
    

Execution of the kernel is aborted, raising an interrupt in the host program. The `assert()` macro results in a corrupted CUDA context, causing any subsequent CUDA calls or kernel invocations to fail with `cudaErrorAssert`.

The kernel execution is unaffected if `expression` is different from zero.

For example, the following program from source file `test.cu`
    
    
    #include <assert.h>
    
     __global__ void testAssert(void) {
         int is_one        = 1;
         int should_be_one = 0;
    
         // This will have no effect
         assert(is_one);
    
         // This will halt kernel execution
         assert(should_be_one);
     }
    
     int main(void) {
         testAssert<<<1,1>>>();
         cudaDeviceSynchronize();
         return 0;
     }
    

will output:
    
    
    test.cu:11: void testAssert(): block: [0,0,0], thread: [0,0,0] Assertion `should_be_one` failed.
    

Assertions are intended for debugging purposes. Since they can affect performance, it is recommended that they be disabled in production code. They can be disabled at compile time by defining the `NDEBUG` preprocessor macro before including `assert.h` or `<cassert>`, or by using the compiler flag `-DNDEBUG`. Note that the expression should not have side effects; otherwise, disabling the assertion will affect the functionality of the code.

### 5.4.10.2. Breakpoint Function

The execution of a kernel function can be suspended by calling the `__brkpt()` function from any device thread.
    
    
    void __brkpt();
    

### 5.4.10.3. Diagnostic Pragmas

The following pragmas can be used to manage the severity of errors that are triggered when a specific diagnostic message is raised.
    
    
    #pragma nv_diag_suppress
    #pragma nv_diag_warning
    #pragma nv_diag_error
    #pragma nv_diag_default
    #pragma nv_diag_once
    

The uses of these pragmas are as follows:
    
    
    #pragma nv_diag_xxx <error_number1>, <error_number2> ...
    

The affected diagnostic is specified using the error number shown in the warning message. Any diagnostic can be changed to an error, but only warnings can have their severity suppressed or restored after being changed to an error. The `nv_diag_default` pragma returns the severity of a diagnostic to the severity that was in effect before any other pragmas were issued, namely, the normal severity of the message as modified by any command-line options. The following example suppresses the `declared but never referenced` warning of `foo()`:
    
    
    #pragma nv_diag_suppress 177 // "declared but never referenced"
    void foo() {
        int i = 0;
    }
    
    #pragma nv_diag_default 177
    void bar() {
        int i = 0;
    }
    

The following pragmas may be used to save and restore the current diagnostic pragma state:
    
    
    #pragma nv_diagnostic push
    #pragma nv_diagnostic pop
    

Examples:
    
    
    #pragma nv_diagnostic push
    #pragma nv_diag_suppress 177 // "declared but never referenced"
    void foo() {
        int i = 0;
    }
    
    #pragma nv_diagnostic pop
    void bar() {
        int i = 0; // raise a warning
    }
    

Note that these directives only affect the `nvcc` CUDA front-end compiler. They have no effect on the host compiler.

`nvcc` defines the macro `__NVCC_DIAG_PRAGMA_SUPPORT__` when diagnostic pragmas are supported.

## 5.4.11. Warp Matrix Functions

C++ warp matrix operations leverage Tensor Cores to accelerate matrix problems of the form `D=A*B+C`. These operations are supported on mixed-precision floating point data for devices of compute capability 7.0 or higher. This requires co-operation from all threads in a [warp](../01-introduction/programming-model.html#programming-model-warps-simt). In addition, these operations are allowed in conditional code only if the condition evaluates identically across the entire [warp](../01-introduction/programming-model.html#programming-model-warps-simt), otherwise the code execution is likely to hang.

### 5.4.11.1. Description

All following functions and types are defined in the namespace `nvcuda::wmma`. Sub-byte operations are considered preview, i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This extra functionality is defined in the `nvcuda::wmma::experimental` namespace.
    
    
    template<typename Use, int m, int n, int k, typename T, typename Layout=void> class fragment;
    
    void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm);
    void load_matrix_sync(fragment<...> &a, const T* mptr, unsigned ldm, layout_t layout);
    void store_matrix_sync(T* mptr, const fragment<...> &a, unsigned ldm, layout_t layout);
    void fill_fragment(fragment<...> &a, const T& v);
    void mma_sync(fragment<...> &d, const fragment<...> &a, const fragment<...> &b, const fragment<...> &c, bool satf=false);
    

`fragment`
    

An overloaded class containing a section of a matrix distributed across all threads in the warp. The mapping of matrix elements into `fragment` internal storage is unspecified and subject to change in future architectures.

Only certain combinations of template arguments are allowed. The first template parameter specifies how the fragment will participate in the matrix operation. Acceptable values for `Use` are:

  * `matrix_a` when the fragment is used as the first multiplicand, `A`,

  * `matrix_b` when the fragment is used as the second multiplicand, `B`, or

  * `accumulator` when the fragment is used as the source or destination accumulators (`C` or `D`, respectively).

The `m`, `n` and `k` sizes describe the shape of the warp-wide matrix tiles participating in the multiply-accumulate operation. The dimension of each tile depends on its role. For `matrix_a` the tile takes dimension `m x k`; for `matrix_b` the dimension is `k x n`, and `accumulator` tiles are `m x n`.

The data type, `T`, may be `double`, `float`, `__half`, `__nv_bfloat16`, `char`, or `unsigned char` for multiplicands and `double`, `float`, `int`, or `__half` for accumulators. As documented in [Element Types and Matrix Sizes](#wmma-type-sizes), limited combinations of accumulator and multiplicand types are supported. The Layout parameter must be specified for `matrix_a` and `matrix_b` fragments. `row_major` or `col_major` indicate that elements within a matrix row or column are contiguous in memory, respectively. The `Layout` parameter for an `accumulator` matrix should retain the default value of `void`. A row or column layout is specified only when the accumulator is loaded or stored as described below.


`load_matrix_sync`
    

Waits until all warp lanes have arrived at load_matrix_sync and then loads the matrix fragment a from memory. `mptr` must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. `ldm` describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for `__half` element type or multiple of 4 for `float` element type. (i.e., multiple of 16 bytes in both cases). If the fragment is an `accumulator`, the `layout` argument must be specified as either `mem_row_major` or `mem_col_major`. For `matrix_a` and `matrix_b` fragments, the layout is inferred from the fragment’s `layout` parameter. The values of `mptr`, `ldm`, `layout` and all template parameters for `a` must be the same for all threads in the warp. This function must be called by all threads in the warp, or the result is undefined.

`store_matrix_sync`
    

Waits until all warp lanes have arrived at store_matrix_sync and then stores the matrix fragment a to memory. `mptr` must be a 256-bit aligned pointer pointing to the first element of the matrix in memory. `ldm` describes the stride in elements between consecutive rows (for row major layout) or columns (for column major layout) and must be a multiple of 8 for `__half` element type or multiple of 4 for `float` element type. (i.e., multiple of 16 bytes in both cases). The layout of the output matrix must be specified as either `mem_row_major` or `mem_col_major`. The values of `mptr`, `ldm`, `layout` and all template parameters for a must be the same for all threads in the warp.

`fill_fragment`
    

Fill a matrix fragment with a constant value `v`. Because the mapping of matrix elements to each fragment is unspecified, this function is ordinarily called by all threads in the warp with a common value for `v`.

`mma_sync`
    

Waits until all warp lanes have arrived at mma_sync, and then performs the warp-synchronous matrix multiply-accumulate operation `D=A*B+C`. The in-place operation, `C=A*B+C`, is also supported. The value of `satf` and template parameters for each matrix fragment must be the same for all threads in the warp. Also, the template parameters `m`, `n` and `k` must match between fragments `A`, `B`, `C` and `D`. This function must be called by all threads in the warp, or the result is undefined.

If `satf` (saturate to finite value) mode is `true`, the following additional numerical properties apply for the destination accumulator:

  * If an element result is +Infinity, the corresponding accumulator will contain `+MAX_NORM`

  * If an element result is -Infinity, the corresponding accumulator will contain `-MAX_NORM`

  * If an element result is NaN, the corresponding accumulator will contain `+0`


Because the map of matrix elements into each thread’s `fragment` is unspecified, individual matrix elements must be accessed from memory (shared or global) after calling `store_matrix_sync`. In the special case where all threads in the warp will apply an element-wise operation uniformly to all fragment elements, direct element access can be implemented using the following `fragment` class members.
    
    
    enum fragment<Use, m, n, k, T, Layout>::num_elements;
    T fragment<Use, m, n, k, T, Layout>::x[num_elements];
    

As an example, the following code scales an `accumulator` matrix tile by half.
    
    
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> frag;
    float alpha = 0.5f; // Same value for all threads in warp
    /*...*/
    for(int t=0; t<frag.num_elements; t++)
    frag.x[t] *= alpha;
    

### 5.4.11.2. Alternate Floating Point

Tensor Cores support alternate types of floating point operations on devices with compute capability 8.0 and higher.

`__nv_bfloat16`
    

This data format is an alternate fp16 format that has the same range as f32 but reduced precision (7 bits). You can use this data format directly with the `__nv_bfloat16` type available in `cuda_bf16.h`. Matrix fragments with `__nv_bfloat16` data types are required to be composed with accumulators of `float` type. The shapes and operations supported are the same as with `__half`.

`tf32`
    

This data format is a special floating-point format supported by Tensor Cores, with the same range as f32 and reduced precision (>=10 bits). The internal layout of this format is implementation-defined. To use this floating-point format with WMMA operations, the input matrices must be manually converted to tf32 precision.

To facilitate conversion, a new intrinsic `__float_to_tf32` is provided. While the input and output arguments to the intrinsic are of `float` type, the output will be `tf32` numerically. This new precision is intended to be used with Tensor Cores only, and if mixed with other `float`type operations, the precision and range of the result will be undefined.

Once an input matrix (`matrix_a` or `matrix_b`) is converted to tf32 precision, the combination of a `fragment` with `precision::tf32` precision, and a data type of `float` to `load_matrix_sync` will take advantage of this new capability. Both the accumulator fragments must have `float` data types. The only supported matrix size is 16x16x8 (m-n-k).

The elements of the fragment are represented as `float`, hence the mapping from `element_type<T>` to `storage_element_type<T>` is:
    
    
    precision::tf32 -> float
    

### 5.4.11.3. Double Precision

Tensor Cores support double-precision floating point operations on devices with compute capability 8.0 and higher. To use this new functionality, a `fragment` with the `double` type must be used. The `mma_sync` operation will be performed with the .rn (rounds to nearest even) rounding modifier.

### 5.4.11.4. Sub-byte Operations

Sub-byte WMMA operations provide a way to access the low-precision capabilities of Tensor Cores. They are considered a preview feature i.e. the data structures and APIs for them are subject to change and may not be compatible with future releases. This functionality is available via the `nvcuda::wmma::experimental` namespace:
    
    
    namespace experimental {
        namespace precision {
            struct u4; // 4-bit unsigned
            struct s4; // 4-bit signed
            struct b1; // 1-bit
       }
        enum bmmaBitOp {
            bmmaBitOpXOR = 1, // compute_75 minimum
            bmmaBitOpAND = 2  // compute_80 minimum
        };
        enum bmmaAccumulateOp { bmmaAccumulateOpPOPC = 1 };
    }
    

For 4 bit precision, the APIs available remain the same, but you must specify `experimental::precision::u4` or `experimental::precision::s4` as the fragment data type. Since the elements of the fragment are packed together, `num_storage_elements` will be smaller than `num_elements` for that fragment. The `num_elements` variable for a sub-byte fragment, hence returns the number of elements of sub-byte type `element_type<T>`. This is true for single bit precision as well, in which case, the mapping from `element_type<T>` to `storage_element_type<T>` is as follows:
    
    
    experimental::precision::u4 -> unsigned (8 elements in 1 storage element)
    experimental::precision::s4 -> int (8 elements in 1 storage element)
    experimental::precision::b1 -> unsigned (32 elements in 1 storage element)
    T -> T  //all other types
    

The allowed layouts for sub-byte fragments is always `row_major` for `matrix_a` and `col_major` for `matrix_b`.

For sub-byte operations the value of `ldm` in `load_matrix_sync` should be a multiple of 32 for element type `experimental::precision::u4` and `experimental::precision::s4` or a multiple of 128 for element type `experimental::precision::b1` (i.e., multiple of 16 bytes in both cases).

Note

Support for the following variants for MMA instructions is deprecated and will be removed in sm_90:

>   * `experimental::precision::u4`
> 
>   * `experimental::precision::s4`
> 
>   * `experimental::precision::b1` with `bmmaBitOp` set to `bmmaBitOpXOR`
> 
> 


`bmma_sync`
    

Waits until all warp lanes have executed bmma_sync, and then performs the warp-synchronous bit matrix multiply-accumulate operation `D = (A op B) + C`, where `op` consists of a logical operation `bmmaBitOp` followed by the accumulation defined by `bmmaAccumulateOp`. The available operations are:

`bmmaBitOpXOR`, a 128-bit XOR of a row in `matrix_a` with the 128-bit column of `matrix_b`

`bmmaBitOpAND`, a 128-bit AND of a row in `matrix_a` with the 128-bit column of `matrix_b`, available on devices with compute capability 8.0 and higher.

The accumulate op is always `bmmaAccumulateOpPOPC` which counts the number of set bits.

### 5.4.11.5. Restrictions

The special format required by tensor cores may be different for each major and minor device architecture. This is further complicated by threads holding only a fragment (opaque architecture-specific ABI data structure) of the overall matrix, with the developer not allowed to make assumptions on how the individual parameters are mapped to the registers participating in the matrix multiply-accumulate.

Since fragments are architecture-specific, it is unsafe to pass them from function A to function B if the functions have been compiled for different link-compatible architectures and linked together into the same device executable. In this case, the size and layout of the fragment will be specific to one architecture and using WMMA APIs in the other will lead to incorrect results or potentially, corruption.

An example of two link-compatible architectures, where the layout of the fragment differs, is sm_70 and sm_75.
    
    
    fragA.cu: void foo() { wmma::fragment<...> mat_a; bar(&mat_a); }
    fragB.cu: void bar(wmma::fragment<...> *mat_a) { // operate on mat_a }
    
    
    
    // sm_70 fragment layout
    $> nvcc -dc -arch=compute_70 -code=sm_70 fragA.cu -o fragA.o
    // sm_75 fragment layout
    $> nvcc -dc -arch=compute_75 -code=sm_75 fragB.cu -o fragB.o
    // Linking the two together
    $> nvcc -dlink -arch=sm_75 fragA.o fragB.o -o frag.o
    

This undefined behavior might also be undetectable at compilation time and by tools at runtime, so extra care is needed to make sure the layout of the fragments is consistent. This linking hazard is most likely to appear when linking with a legacy library that is both built for a different link-compatible architecture and expecting to be passed a WMMA fragment.

Note that in the case of weak linkages (for example, a CUDA C++ inline function), the linker may choose any available function definition which may result in implicit passes between compilation units.

To avoid these sorts of problems, the matrix should always be stored out to memory for transit through external interfaces (e.g. `wmma::store_matrix_sync(dst, …);`) and then it can be safely passed to `bar()` as a pointer type [e.g. `float *dst`].

Note that since sm_70 can run on sm_75, the above example sm_75 code can be changed to sm_70 and correctly work on sm_75. However, it is recommended to have sm_75 native code in your application when linking with other sm_75 separately compiled binaries.

### 5.4.11.6. Element Types and Matrix Sizes

Tensor Cores support a variety of element types and matrix sizes. The following table presents the various combinations of `matrix_a`, `matrix_b` and `accumulator` matrix supported:

Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k)  
---|---|---|---  
__half | __half | float | 16x16x16  
__half | __half | float | 32x8x16  
__half | __half | float | 8x32x16  
__half | __half | __half | 16x16x16  
__half | __half | __half | 32x8x16  
__half | __half | __half | 8x32x16  
unsigned char | unsigned char | int | 16x16x16  
unsigned char | unsigned char | int | 32x8x16  
unsigned char | unsigned char | int | 8x32x16  
signed char | signed char | int | 16x16x16  
signed char | signed char | int | 32x8x16  
signed char | signed char | int | 8x32x16  
  
Alternate floating-point support:

Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k)  
---|---|---|---  
__nv_bfloat16 | __nv_bfloat16 | float | 16x16x16  
__nv_bfloat16 | __nv_bfloat16 | float | 32x8x16  
__nv_bfloat16 | __nv_bfloat16 | float | 8x32x16  
precision::tf32 | precision::tf32 | float | 16x16x8  
  
Double Precision Support:

Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k)  
---|---|---|---  
double | double | double | 8x8x4  
  
Experimental support for sub-byte operations:

Matrix A | Matrix B | Accumulator | Matrix Size (m-n-k)  
---|---|---|---  
precision::u4 | precision::u4 | int | 8x8x32  
precision::s4 | precision::s4 | int | 8x8x32  
precision::b1 | precision::b1 | int | 8x8x128  
  
### 5.4.11.7. Example

The following code implements a 16x16x16 matrix multiplication in a single warp.
    
    
    #include <mma.h>
    using namespace nvcuda;
    
    __global__ void wmma_ker(half *a, half *b, float *c) {
       // Declare the fragments
       wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
       wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
       wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
       // Initialize the output to zero
       wmma::fill_fragment(c_frag, 0.0f);
    
       // Load the inputs
       wmma::load_matrix_sync(a_frag, a, 16);
       wmma::load_matrix_sync(b_frag, b, 16);
    
       // Perform the matrix multiplication
       wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    
       // Store the output
       wmma::store_matrix_sync(c, c_frag, 16, wmma::mem_row_major);
    }
