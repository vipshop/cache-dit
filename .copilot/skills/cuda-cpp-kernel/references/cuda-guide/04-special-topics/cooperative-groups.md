---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
---

# 4.4. Cooperative Groups

## 4.4.1. Introduction

Cooperative Groups are an extension to the CUDA programming model for organizing groups of collaborating threads. Cooperative Groups allow developers to control the granularity at which threads are collaborating, helping them to express richer, more efficient parallel decompositions. Cooperative Groups also provide implementations of common parallel primitives like scan and parallel reduce.

Historically, the CUDA programming model has provided a single, simple construct for synchronizing cooperating threads: a barrier across all threads of a thread block, as implemented with the `__syncthreads()` intrinsic function. In an effort to express broader patterns of parallel interaction, many performance-oriented programmers have resorted to writing their own ad hoc and unsafe primitives for synchronizing threads within a single warp, or across sets of thread blocks running on a single GPU. Whilst the performance improvements achieved have often been valuable, this has resulted in an ever-growing collection of brittle code that is expensive to write, tune, and maintain over time and across GPU generations. Cooperative Groups provides a safe and future-proof mechanism for writing performant code.

The full Cooperative Groups API is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-partition-header).

## 4.4.2. Cooperative Group Handle & Member Functions

Cooperative Groups are managed via a Cooperative Group Handle. The Cooperative Group handle allows participating threads to learn their position in the group, the group size, and other group information. Select group member functions are shown in the following table.

Table 10 Select Member Functions Accessor | Returns  
---|---  
`thread_rank()` | The rank of the calling thread.  
`num_threads()` | The total number of threads in the group .  
`thread_index()` | A 3-Dimensional index of the thread within the launched block.  
`dim_threads()` | The 3D dimensions of the launched block in units of threads.  
  
A complete list pf member functions is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-common-header).

## 4.4.3. Default Behavior / Groupless Execution

Groups representing the grid and thread blocks are implicitly created based on the kernel launch configuration. These “implicit” groups provide a starting point that developers can explicitly decompose into finer grained groups. Implicit groups can be accessed using the following methods:

Table 11 Cooperative Groups Implicitly Created by CUDA Runtime Accessor | Group Scope  
---|---  
`this_thread_block()` | Returns the handle to a group containing all threads in current thread block.  
`this_grid()` | Returns the handle to a group containing all threads in the grid.  
`coalesced_threads()` [[1]](#cgfn2) | Returns the handle to a group of currently active threads in a warp.  
`this_cluster()` [[2]](#cgfn3) | Returns the handle to a group of threads in the current cluster.  
  
[[1](#id3)]

The `coalesced_threads()` operator returns the set of active threads at that point in time, and makes no guarantee about which threads are returned (as long as they are active) or that they will stay coalesced throughout execution.

[[2](#id4)]

The `this_cluster()` assumes a 1x1x1 cluster when a non-cluster grid is launched. Requires Compute Capability 9.0 or greater.

More information is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-common-header).

### 4.4.3.1. Create Implicit Group Handles As Early As Possible

For best performance it is recommended that you create a handle for the implicit group upfront (as early as possible, before any branching has occurred) and use that handle throughout the kernel.

### 4.4.3.2. Only Pass Group Handles by Reference

It is recommended that you pass group handles by reference to functions when passing a group handle into a function. Group handles must be initialized at declaration time, as there is no default constructor. Copy-constructing group handles is discouraged.

## 4.4.4. Creating Cooperative Groups

Groups are created by partitioning a parent group into subgroups. When a group is partitioned, a group handle is created to manage the resulting subgroup. The following partitioning operations are available to developers:

Table 12 Cooperative Group Partitioning Operations Partition Type | Description  
---|---  
tiled_partition | Divides parent group into a series of fixed-size subgroups arranged in a one-dimensional, row-major format.  
stride_partition | Divides parent group into equally-sized subgroups where threads are assigned to subgroups in a round-robin manner.  
labeled_partition | Divides parent group into one-dimensional subgroups based on a conditional label, which can be any integral type.  
binary_partition | Specialized form of labeled partitioning where label can only be “0” or “1”.  
  
The following example shows how a tiled partition is created:
    
    
    namespace cg = cooperative_groups;
    // Obtain the current thread's cooperative group
    cg::thread_block my_group = cg::this_thread_block();
    
    // Partition the cooperative group into tiles of size 8
    cg::thread_block_tile<8> my_subgroup = cg::tiled_partition<8>(cta);
    
    // do work as my_subgroup
    

The best partitioning strategy to use depends on the context. More information is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-partition-header).

### 4.4.4.1. Avoiding Group Creation Hazards

Partitioning a group is a collective operation and all threads in the group must participate. If the group was created in a conditional branch that not all threads reach, this can lead to deadlocks or data corruption.

## 4.4.5. Synchronization

Prior to the introduction of Cooperative Groups, the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary. Cooperative groups allows developers to synchronize groups of cooperating threads at different granularities.

### 4.4.5.1. Sync

You can synchronize a group by calling the collective `sync()` function. Like `__syncthreads()`, the `sync()` function makes the following guarantees: \- All memory accesses (e.g., reads and writes) made by threads in the group before the synchronization point are visible to all threads in the group after the synchronization point. \- All threads in the group reach the synchronization point before any thread is allowed to proceed beyond it.

The following example shows a `cooperative_groups::sync()` that is equivalent to `__syncthreads()`.
    
    
    namespace cg = cooperative_groups;
    
    cg::thread_block my_group = cg::this_thread_block();
    
    // Synchronize threads in the block
    cg::sync(my_group);
    

Cooperative groups can be used to synchronize the entire grid. As of CUDA 13, cooperative groups can no longer be used for multi-device synchronization. For details see the [Large Scale Groups](#cooperative-groups-large-scale-groups) section.

More information about synchronization is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-sync-header).

### 4.4.5.2. Barriers

Cooperative Groups provides a barrier API similar to `cuda::barrier` that can be used for more advanced synchronization. Cooperative Groups barrier API differs from `cuda::barrier` in a few key ways: \- Cooperative Groups barriers are automatically initialized \- All threads in the group must arrive and wait at the barrier once per phase. \- `barrier_arrive` returns an `arrival_token` object that must be passed into the corresponding `barrier_wait`, where it is consumed and cannot be used again.

Programmers must take care to avoid hazards when using Cooperative Groups barriers: \- No collective operations can be used by a group between after calling `barrier_arrive` and before calling `barrier_wait`. \- `barrier_wait` only guarantees that all threads in the group have called `barrier_arrive`. `barrier_wait` does NOT guarantee that all threads have called `barrier_wait`.
    
    
    namespace cg = cooperative_groups;
    
    cg::thread_block my_group = this_block();
    
    auto token = cluster.barrier_arrive();
    
    // Optional: Do some local processing to hide the synchronization latency
    local_processing(block);
    
    // Make sure all other blocks in the cluster are running and initialized shared data before accessing dsmem
    cluster.barrier_wait(std::move(token));
    

## 4.4.6. Collective Operations

Cooperative Groups includes a set of collective operations that can be performed by a group of threads. These operations require participation of all threads in the specified group in order to complete the operation.

All threads in the group must pass the same values for corresponding arguments to each collective call, unless different values are explicitly allowed in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-partition-header). Otherwise the behavior of the call is undefined.

### 4.4.6.1. Reduce

The `reduce` function is used to perform a parallel reduction on the data provided by each thread in the specified group. The type of reduction must be specified by providing one of the operators shown in the following table.

Table 13 Cooperative Groups Reduction Operators Operator | Returns  
---|---  
plus | Sum of all values in group  
less | Minimum value  
greater | Maximum value  
bit_and | Bitwise AND reduction  
bit_or | Bitwise OR reduction  
bit_xor | Bitwise XOR reduction  
  
Hardware acceleration is used for reductions when available (requires Compute Capability 8.0 or greater). A software fallback is available for older hardware where hardware acceleration is not available. Only 4B types are accelerated by hardware.

More information about reductions is available in the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-reduce-header).

The following example shows how to use `cooperative_groups::reduce()` to perform a block-wide sum reduction.
    
    
    namespace cg = cooperative_groups;
    
    cg::thread_block my_group = cg::this_thread_block();
    
    int val = data[threadIdx.x];
    
    int sum = cg::reduce(cta, val, cg::plus<int>());
    
    // Store the result from the reduction
    if (my_group.thread_rank() == 0) {
       result[blockIdx.x] = sum;
    }
    

### 4.4.6.2. Scans

Cooperative Groups includes implementations of `inclusive_scan` and `exclusive_scan` that can be used on arbitrary group sizes. The functions perform a scan operation on the data provided by each thread named in the specified group.

Programmers can optionally specify a reduction operator, as listed in [Reduction Operators Table](#cooperative-groups-reduction-operators) above.
    
    
    namespace cg = cooperative_groups;
    
    cg::thread_block my_group = cg::this_thread_block();
    
    int val = data[my_group.thread_rank()];
    
    int exclusive_sum = cg::exclusive_scan(my_group, val, cg::plus<int>());
    
    result[my_group.thread_rank()] = exclusive_sum;
    

More information about scans is available in the [Cooperative Groups Scan API](../05-appendices/device-callable-apis.html#cg-api-scan-header).

### 4.4.6.3. Invoke One

Cooperative Groups provides an `invoke_one` function for use when a single thread must perform a serial portion of work on behalf of a group. \- `invoke_one` selects a single arbitrary thread from the calling group and uses that thread to call the supplied invocable function using the supplied arguments. \- `invoke_one_broadcast` is the same as `invoke_one` except the result of the call is also broadcast to all threads in the group.

The thread selection mechanism is not guaranteed to be deterministic.

The following example shows basic `invoke_one` utilization.
    
    
    namespace cg = cooperative_groups;
    cg::thread_block my_group = cg::this_thread_block();
    
    // Ensure only one thread in the thread block prints the message
    cg::invoke_one(my_group, []() {
       printf("Hello from one thread in the block!");
    });
    
    // Synchronize to make sure all threads wait until the message is printed
    cg::sync(my_group);
    

Communication or synchronization within the calling group is not allowed inside the invocable function. Communication with threads outside of the calling group is allowed.

## 4.4.7. Asynchronous Data Movement

Cooperative Groups `memcpy_async` functionality in CUDA provides a way to perform asynchronous memory copies between global memory and shared memory. `memcpy_async` is particularly useful for optimizing memory transfers and overlapping computation with data transfer to improve performance.

The `memcpy_async` function is used to start an asynchronous load from global memory to shared memory. `memcpy_async` is intended to be used like a “prefetch” where data is loaded before it is needed.

The `wait` function forces all threads in a group to wait until the asynchronous memory transfer is completed. `wait` must be called by all threads in the group before the data can be accessed in shared memory.

The following example shows how to use `memcpy_async` and `wait` to prefetch data.
    
    
    namespace cg = cooperative_groups;
    
    cg::thread_group my_group = cg::this_thread_block();
    
    __shared__ int shared_data[];
    
    // Perform an asynchronous copy from global memory to shared memory
    cg::memcpy_async(my_group, shared_data + my_group.rank(), input + my_group.rank(), sizeof(int));
    
    // Hide latency by doing work here. Cannot use shared_data
    
    // Wait for the asynchronous copy to complete
    cg::wait(my_group);
    
    // Prefetched data is now available
    

See the [Cooperative Groups API](../05-appendices/device-callable-apis.html#cg-api-async-header) for more information.

### 4.4.7.1. Memcpy Async Alignment Requirements

`memcpy_async` is only asynchronous if the source is global memory and the destination is shared memory and both are at least 4-byte aligned. For achieving best performance: an alignment of 16 bytes for both shared memory and global memory is recommended.

## 4.4.8. Large Scale Groups

Cooperative Groups allows for large groups that span the entire grid. All Cooperative Group functionality described previously is available to these large groups, with one notable exception: synchronizing the entire grid requires using the `cudaLaunchCooperativeKernel` runtime launch API.

Multi-device launch APIs and related references for Cooperative Groups have been removed as of CUDA 13.

### 4.4.8.1. When to use `cudaLaunchCooperativeKernel`

`cudaLaunchCooperativeKernel` is a CUDA runtime API function used to launch a single-device kernel that employs cooperative groups, specifically designed for executing kernels that require inter-block synchronization. This function ensures that all threads in the kernel can synchronize and cooperate across the entire grid, which is not possible with traditional CUDA kernels that only allow synchronization within individual thread blocks. `cudaLaunchCooperativeKernel` ensures that the kernel launch is atomic, i.e. if the API call succeeds, then the provided number of thread blocks will launch on the specified device.

It is good practice to first ensure the device supports cooperative launches by querying the device attribute `cudaDevAttrCooperativeLaunch`:
    
    
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    

which will set `supportsCoopLaunch` to 1 if the property is supported on device 0. Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these:

  * The Linux platform without MPS

  * The Linux platform with MPS and on a device with compute capability 7.0 or higher

  * The latest Windows platform
