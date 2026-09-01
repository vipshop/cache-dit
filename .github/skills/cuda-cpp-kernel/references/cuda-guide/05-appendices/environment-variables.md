---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
---

# 5.2. CUDA Environment Variables  
  
The following section lists the CUDA environment variables. Those related to the Multi-Process Service (MPS) are documented in the [GPU Deployment and Management Guide](https://docs.nvidia.com/deploy/mps/index.html#environment-variables).

## 5.2.1. Device Enumeration and Properties

### 5.2.1.1. `CUDA_VISIBLE_DEVICES`

The environment variable controls which GPU devices are visible to a CUDA application and in what order they are enumerated.

  * If the variable is not set, all GPU devices are visible.

  * If the variable is set to an empty string, no GPU devices are visible.


**Possible Values** : A comma-separated sequence of GPU identifiers.

GPU identifiers are provided as:

  * **Integer indices** : These correspond to the ordinal number of the GPU in the system, as determined by `nvidia-smi`, starting from 0. For example, setting `CUDA_VISIBLE_DEVICES=2,1` makes device 0 not visible and enumerates device 2 before device 1.

    * If an invalid index is encountered, only devices with indices that appear before the invalid index in the list are visible. For example, setting `CUDA_VISIBLE_DEVICES=0,2,-1,1` makes devices 0 and 2 visible, while device 1 is not visible because it appears after the invalid index `-1`.

  * **GPU UUID strings** : These should follow the same format as given by `nvidia-smi -L`, such as `GPU-8932f937-d72c-4106-c12f-20bd9faed9f6`. However, for convenience, abbreviated forms are allowed; simply specify enough digits from the beginning of the GPU UUID to uniquely identify that GPU in the target system. For example, `CUDA_VISIBLE_DEVICES=GPU-8932f937` may be a valid way to refer to the above GPU UUID, assuming no other GPU in the system shares this prefix.

  * [Multi-Instance GPU (MIG)](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) support: `MIG-<GPU-UUID>/<GPU instance ID>/<compute instance ID>`. For example, `MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2`. Only single MIG instance enumeration is supported.


The device count returned by the `cudaGetDeviceCount()` API includes only the visible devices, so CUDA APIs that use integer device identifiers only support ordinals in the range [0, visible device count - 1]. The enumeration order of the GPU devices determines the ordinal values. For example, with `CUDA_VISIBLE_DEVICES=2,1`, calling `cudaSetDevice(0)` will set device 2 as the current device, as it is enumerated first and assigned an ordinal of 0. Calling `cudaGetDevice(&device_ordinal)` after that will also set `device_ordinal` to 0, which corresponds to device 2.

**Examples** :
    
    
    nvidia-smi -L # Get list of GPU UUIDs
    CUDA_VISIBLE_DEVICES=0,1
    CUDA_VISIBLE_DEVICES=GPU-8932f937-d72c-4106-c12f-20bd9faed9f6
    CUDA_VISIBLE_DEVICES=MIG-GPU-8932f937-d72c-4106-c12f-20bd9faed9f6/1/2
    

* * *

### 5.2.1.2. `CUDA_DEVICE_ORDER`

The environment variable controls the order in which CUDA enumerates the available devices.

**Possible Values** :

  * `FASTEST_FIRST`: The available devices are enumerated from fastest to slowest using a simple heuristic (default).

  * `PCI_BUS_ID`: The available devices are enumerated by PCI bus ID in ascending order. The PCI bus IDs can be obtained with `nvidia-smi --query-gpu=name,pci.bus_id`.


**Examples** :
    
    
    CUDA_DEVICE_ORDER=FASTEST_FIRST
    CUDA_DEVICE_ORDER=PCI_BUS_ID
    nvidia-smi --query-gpu=name,pci.bus_id # Get list of PCI bus IDs
    

* * *

### 5.2.1.3. `CUDA_MANAGED_FORCE_DEVICE_ALLOC`

The environment variable alters how [Unified Memory](../02-basics/understanding-memory.html#memory-unified-memory) is physically stored in multi-GPU systems.

**Possible Values** : Numerical value, either zero or non-zero.

  * **Non-zero value** : Forces the driver to use device memory for physical storage. All devices used in the process that support managed memory must be peer-to-peer compatible. Otherwise, `cudaErrorInvalidDevice` is returned.

  * `0`: Default behavior.


**Examples** :
    
    
    CUDA_MANAGED_FORCE_DEVICE_ALLOC=0
    CUDA_MANAGED_FORCE_DEVICE_ALLOC=1 # force device memory
    

* * *

## 5.2.2. JIT Compilation

### 5.2.2.1. `CUDA_CACHE_DISABLE`

The environment variable controls the behavior of the on-disk [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache. Disabling the JIT cache forces PTX to CUBIN compilation for a CUDA application each time it is executed, unless the CUBIN code for the running architecture is found in the binary.

Disabling the JIT cache increases an application’s load time during initial execution. However, it can be useful for reducing the application’s disk space and for diagnosing differences across driver versions or build flags.

**Possible Values** :

  * `1`: Disables PTX JIT caching.

  * `0`: Enables PTX JIT caching (default).


**Examples** :
    
    
    CUDA_CACHE_DISABLE=1 # disables caching
    CUDA_CACHE_DISABLE=0 # enables caching
    

* * *

### 5.2.2.2. `CUDA_CACHE_PATH`

The environment variable specifies the directory path for the [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache.

**Possible Values** : The absolute path to the cache directory (with appropriate access permissions). The default values are:

  * on Windows, `%APPDATA%\NVIDIA\ComputeCache`

  * on Linux, `~/.nv/ComputeCache`


**Example** :
    
    
    CUDA_CACHE_PATH=~/tmp
    

* * *

### 5.2.2.3. `CUDA_CACHE_MAXSIZE`

The environment variable specifies the [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) cache size in bytes. Binaries that exceed this size are not cached. If needed, older binaries are evicted from the cache to make room for newer ones.

**Possible Values** : Number of bytes. The default values are:

  * On desktop/server platforms, `1073741824` (1 GiB)

  * On embedded platforms, `268435456` (256 MiB)


`4294967296` (4 GiB) is the maximum size.

**Example** :
    
    
    CUDA_CACHE_MAXSIZE=268435456 # 256 MiB
    

* * *

### 5.2.2.4. `CUDA_FORCE_PTX_JIT` and `CUDA_FORCE_JIT`

The environment variables instruct the CUDA driver to ignore any CUBIN embedded in an application and perform [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) of the embedded PTX code instead.

Forcing JIT compilation increases an application’s load time during initial execution. However, it can be used to validate that PTX code is embedded in an application and that its Just-In-Time compilation is functioning properly. This ensures [forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/) with future architectures.

`CUDA_FORCE_PTX_JIT` overrides `CUDA_FORCE_JIT`.

**Possible Values** :

  * `1`: Forces PTX JIT compilation.

  * `0`: Default behavior.


**Example** :
    
    
    CUDA_FORCE_PTX_JIT=1
    

* * *

### 5.2.2.5. `CUDA_DISABLE_PTX_JIT` and `CUDA_DISABLE_JIT`

The environment variables disable the [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) of embedded PTX code and use the compatible CUBIN embedded in an application.

A kernel will fail to load if it does not have embedded binary code, or if the embedded binary was compiled for an incompatible architecture. These environment variables can be used to validate that an application has compatible CUBIN code generated for each kernel. See the [Binary Compatibility](../01-introduction/cuda-platform.html#cuda-platform-compute-binary-compatibility) section for more details.

`CUDA_DISABLE_PTX_JIT` overrides `CUDA_DISABLE_JIT`.

**Possible Values** :

  * `1`: Disables PTX JIT compilation.

  * `0`: Default behavior.


**Example** :
    
    
    CUDA_DISABLE_PTX_JIT=1
    

* * *

### 5.2.2.6. `CUDA_FORCE_PRELOAD_LIBRARIES`

The environment variable affects the preloading of libraries required for [NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) and [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation).

**Possible Values** :

  * `1`: This forces the driver to preload the libraries required for [NVVM](https://docs.nvidia.com/cuda/nvvm-ir-spec/) and [Just-In-Time (JIT) compilation](../01-introduction/cuda-platform.html#cuda-platform-just-in-time-compilation) during initialization. This increases the memory footprint and the time required for CUDA driver initialization. Setting this environment variable is necessary to avoid certain deadlock situations involving multiple threads.

  * `0`: Default behavior.


**Example** :
    
    
    CUDA_FORCE_PRELOAD_LIBRARIES=1
    

* * *

## 5.2.3. Execution

### 5.2.3.1. `CUDA_LAUNCH_BLOCKING`

The environment variable specifies whether to disable or enable asynchronous kernel launches.

Disabling asynchronous execution results in slower execution but is useful for debugging. It forces GPU work to run synchronously from the CPU’s perspective. This allows CUDA API errors to be observed at the exact API call that triggered them, rather than later in the execution. Synchronous execution is useful for debugging purposes.

**Possible Values** :

  * `1`: Disables asynchronous execution.

  * `0`: Asynchronous execution (default).


**Example** :
    
    
    CUDA_LAUNCH_BLOCKING=1
    

* * *

### 5.2.3.2. `CUDA_DEVICE_MAX_CONNECTIONS`

The environment variable controls the number of concurrent compute and copy engine connections (work queues), setting both to the specified value. If independent GPU tasks, namely kernels or copy operations launched from different CUDA streams, map to the same work queue, a false dependency is created which can lead to GPU work serialization, since the same underlying resource(s) are used. To reduce the probability of such false dependencies, it is recommended that the work queue count, controlled via this environment variable, be greater than or equal to the number of active CUDA streams per context.

Setting this environment variable also modifies the number of copy connections, unless they are explicitly set via the `CUDA_DEVICE_MAX_COPY_CONNECTIONS` environment variable.

**Possible Values** : `1` to `32` connections, default is `8` (assumes no MPS)

**Example** :
    
    
    CUDA_DEVICE_MAX_CONNECTIONS=16
    

* * *

### 5.2.3.3. `CUDA_DEVICE_MAX_COPY_CONNECTIONS`

The environment variable controls the number of concurrent copy connections (work queues) involved in copy operations. It affects only devices of [compute capability](compute-capabilities.html#compute-capabilities) 8.0 and above.

The `CUDA_DEVICE_MAX_COPY_CONNECTIONS` overrides the value of copy connections set via `CUDA_DEVICE_MAX_CONNECTIONS`, if both were set.

**Possible Values** : `1` to `32` connections, default is `8` (assumes no MPS)

**Example** :
    
    
    CUDA_DEVICE_MAX_COPY_CONNECTIONS=16
    

* * *

### 5.2.3.4. `CUDA_SCALE_LAUNCH_QUEUES`

The environment variable specifies the scaling factor for the size of the queues available for launching work (command buffer), namely the total number of pending kernels or host/device copy operations that can be enqueued on a device.

**Possible Values** : `0.25x`, `0.5x`, `2x`, `4x`

  * Any value other than `0.25x`, `0.5x`, `2x` or `4x` is interpreted as `1x`.


**Example** :
    
    
    CUDA_SCALE_LAUNCH_QUEUES=2x
    

* * *

### 5.2.3.5. `CUDA_GRAPHS_USE_NODE_PRIORITY`

The environment variable controls the CUDA graph’s execution priority relative to the stream priority it inherits from the stream in which it is launched.

`CUDA_GRAPHS_USE_NODE_PRIORITY` overrides the [cudaGraphInstantiateFlagUseNodePriority](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1gd4d586536547040944c05249ee26bc62) flag on graph instantiation.

**Possible Values** :

  * `0`: Inherit the priority of the stream the graph is launched into (default).

  * `1`: Honor per-node launch priorities. The CUDA runtime treats node-level priorities as a scheduling hint for ready-to-run graph nodes.


**Example** :
    
    
    CUDA_GRAPHS_USE_NODE_PRIORITY=1
    

* * *

### 5.2.3.6. `CUDA_DEVICE_WAITS_ON_EXCEPTION`

The environment variable controls the behavior of a CUDA application when an exception (error) occurs.

When enabled, a CUDA application will halt and wait when a device-side exception occurs, allowing a debugger, such as [cuda-gdb](https://docs.nvidia.com/cuda/cuda-gdb/index.html), to be attached to inspect the live GPU state before the process exits or continues.

**Possible Values** :

  * `0`: Default behavior.

  * `1`: Halt when a device exception occurs.


**Example** :
    
    
    CUDA_DEVICE_WAITS_ON_EXCEPTION=1
    

* * *

### 5.2.3.7. `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`

The environment variable controls the default “set-aside” portion of the GPU’s L2 cache reserved for [persisting accesses](../04-special-topics/l2-cache-control.html#l2-set-aside), expressed as a percentage of the L2 size.

It is relevant for GPUs that support persistent L2 cache, specifically devices with [compute capability](compute-capabilities.html#compute-capabilities) 8.0 or higher when using the [CUDA Multi-Process Service (MPS)](https://docs.nvidia.com/deploy/mps/index.html). The environment variable must be set before starting the CUDA MPS Control Daemon, namely before running the `nvidia-cuda-mps-control -d` command.

**Possible Values** : Percentage value between 0 and 100, default is 0.

**Example** :
    
    
    CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT=25 # 25%
    

* * *

### 5.2.3.8. `CUDA_DISABLE_PERF_BOOST`

On Linux hosts, setting this environment variable to 1 prevents boosting the device performance state, instead pstate can be selected implicitly based on various heuristics. This option can potentially be used to reduce power consumption, but may result in higher latency in certain scenarios due to dynamic performance state selection.

**Example** :
    
    
    CUDA_DISABLE_PERF_BOOST=1 # perf boost disabled, Linux only.
    CUDA_DISABLE_PERF_BOOST=0 # default behavior
    

### 5.2.3.9. `CUDA_AUTO_BOOST` [[deprecated]]

The environment variable affects the GPU clock “auto boost” behavior, namely dynamic clock boosting. It overrides the “auto boost” option of the `nvidia-smi` tool, namely `nvidia-smi --auto-boost-default=0`.

Note

This environment variable is deprecated. It is strongly suggested to use `nvidia-smi --applications-clocks=<memory,graphics>` or the [NVML API](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceCommands.html#group__nvmlDeviceCommands) instead of the `CUDA_AUTO_BOOST` environment variable.

* * *

## 5.2.4. Module Loading

### 5.2.4.1. `CUDA_MODULE_LOADING`

The environment variable affects how the CUDA runtime loads modules, specifically how it initializes the device code.

**Possible Values** :

  * `DEFAULT`: Default behavior, equivalent to `LAZY`.

  * `LAZY`: The loading of specific kernels is delayed until a CUDA function handle, `CUfunc`, is extracted using the `cuModuleGetFunction()` or `cuKernelGetFunction()` API calls. In this case, the data from the CUBIN is loaded when the first kernel in the CUBIN is loaded or when the first variable in the CUBIN is accessed.

    * The driver loads the required code on the first call to a kernel; subsequent calls incur no extra overhead. This reduces startup time and GPU memory footprint.

  * `EAGER`: Fully loads CUDA modules and kernels at program initialization. All kernels and data from a CUBIN, FATBIN, or PTX file are fully loaded upon the corresponding `cuModuleLoad*` and `cuLibraryLoad*` driver API call.

    * Higher startup time and GPU memory footprint. Kernel launch overhead is predictable.


**Examples** :
    
    
    CUDA_MODULE_LOADING=EAGER
    CUDA_MODULE_LOADING=LAZY
    

* * *

### 5.2.4.2. `CUDA_MODULE_DATA_LOADING`

The environment variable affects how the CUDA runtime loads data associated to modules.

This is a complementary setting to the kernel-focused setting in `CUDA_MODULE_LOADING`. This environment variable does not affect the `LAZY` or `EAGER` loading of kernels. Data loading behavior is inherited from `CUDA_MODULE_LOADING` if this environment variable is not set.

**Possible Values** :

  * `DEFAULT`: Default behavior, equivalent to `LAZY`.

  * `LAZY`: The loading of module data is delayed until a CUDA function handle, `CUfunc`, is required. In this case, the data from the CUBIN is loaded when the first kernel in the CUBIN is loaded or when the first variable in the CUBIN is accessed.

    * Lazy data loads can require context synchronization, which can slow down concurrent execution.

  * `EAGER`: All data from a CUBIN, FATBIN, or PTX file are fully loaded upon the corresponding `cuModuleLoad*` and `cuLibraryLoad*` API call.


**Example** :
    
    
    CUDA_MODULE_DATA_LOADING=EAGER
    

### 5.2.4.3. `CUDA_BINARY_LOADER_THREAD_COUNT`

Sets the number of CPU threads to use when loading device binaries. When set to 0, the number of CPU threads used is set to a default value of 1.

**Possible Values** :

>   * Integer number of threads to use. Defaults to 0, which uses 1 thread.
> 
> 


**Example** :
    
    
    CUDA_BINARY_LOADER_THREAD_COUNT=4
    

* * *

## 5.2.5. CUDA Error Log Management

### 5.2.5.1. `CUDA_LOG_FILE`

The environment variable specifies a location where descriptive error log messages will be printed as they occur for supported CUDA API calls that returned an error.

For example, if one attempts to launch a kernel with an invalid grid configuration, such as `kernel<<<1, dim3(1,1,128)>>>(...)`, that kernel will fail to launch and `cudaGetLastError()` will return a generic `invalid configuration argument` error.   
If the `CUDA_LOG_FILE` environment variable is set, the user can see the following descriptive error message in the log: `[CUDA][E] Block Dimensions (1,1,128) include one or more values that exceed the device limit of (1024,1024,64)` and easily determine that the specified z-dimension of the block was invalid. See [Error Log Management](../04-special-topics/error-log-management.html#error-log-management) for more details.

**Possible Values** : `stdout`, `stderr`, or a valid file path (with appropriate access permissions)

**Examples** :
    
    
    CUDA_LOG_FILE=stdout
    CUDA_LOG_FILE=/tmp/dbg_cuda_log
