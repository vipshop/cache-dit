---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html
---

# 4.3. Stream-Ordered Memory Allocator

## 4.3.1. Introduction

Managing memory allocations using `cudaMalloc` and `cudaFree` causes the GPU to synchronize across all executing CUDA streams. The stream-ordered memory allocator enables applications to order memory allocation and deallocation with other work launched into a CUDA stream such as kernel launches and asynchronous copies. This improves application memory use by taking advantage of stream-ordering semantics to reuse memory allocations. The allocator also allows applications to control the allocator’s memory caching behavior. When set up with an appropriate release threshold, the caching behavior allows the allocator to avoid expensive calls into the OS when the application indicates it is willing to accept a bigger memory footprint. The allocator also supports easy and secure allocation sharing between processes.

Stream-Ordered Memory Allocator:

>   * Reduces the need for custom memory management abstractions, and makes it easier to create high-performance custom memory management for applications that need it.
> 
>   * Enables multiple libraries to share a common memory pool managed by the driver. This can reduce excess memory consumption.
> 
>   * Allows, the driver to perform optimizations based on its awareness of the allocator and other stream management APIs.
> 
> 


Note

Nsight Compute and the Next-Gen CUDA debugger is aware of the allocator since CUDA 11.3.

## 4.3.2. Memory Management

`cudaMallocAsync` and `cudaFreeAsync` are the APIs which enable stream-ordered memory management. `cudaMallocAsync` returns an allocation and `cudaFreeAsync` frees an allocation. Both APIs accept stream arguments to define when the allocation will become and stop being available for use. These functions allow memory operations to be tied to specific CUDA streams, enabling them to occur without blocking the host or other streams. Application performance can be improved by avoiding potentially costly synchronization of `cudaMalloc` and `cudaFree`.

These APIs can be used for further performance optimization through memory pools, which manage and reuse large blocks of memory for more efficient allocation and deallocation. Memory pools help reduce overhead and prevent fragmentation, improving performance in scenarios with frequent memory allocation operations.

### 4.3.2.1. Allocating Memory

The `cudaMallocAsync` function triggers asynchronous memory allocation on the GPU, linked to a specific CUDA stream. `cudaMallocAsync` allows memory allocation to occur without hindering the host or other streams, eliminating the need for expensive synchronization.

Note

`cudaMallocAsync` ignores the current device/context when determining where the allocation will reside. Instead, `cudaMallocAsync` determines the appropriate device based on the specified memory pool or the supplied stream.

The listing below illustrates a fundamental use pattern: the memory is allocated, used, and then freed back into the same stream.
    
    
    void *ptr;
    size_t size = 512;
    cudaMallocAsync(&ptr, size, cudaStreamPerThread);
    // do work using the allocation
    kernel<<<..., cudaStreamPerThread>>>(ptr, ...);
    // An asynchronous free can be specified without synchronizing the cpu and GPU
    cudaFreeAsync(ptr, cudaStreamPerThread);
    

Note

When accessing allocation from a stream other than the stream that made the allocation, the user must guarantee that the access occurs after the allocation operation, otherwise the behavior is undefined.

### 4.3.2.2. Freeing Memory

`cudaFreeAsync()` asynchronously frees device memory in a stream-ordered fashion, meaning the memory deallocation is assigned to a specific CUDA stream and does not block the host or other streams.

The user must guarantee that the free operation happens after the allocation operation and any uses of the allocation. Any use of the allocation after the free operation starts results in undefined behavior.

Events and/or stream synchronizing operations should be used to guarantee any access to the allocation from other streams is complete before the free operation begins, as illustrated in the following example.
    
    
    cudaMallocAsync(&ptr, size, stream1);
    cudaEventRecord(event1, stream1);
    //stream2 must wait for the allocation to be ready before accessing
    cudaStreamWaitEvent(stream2, event1);
    kernel<<<..., stream2>>>(ptr, ...);
    cudaEventRecord(event2, stream2);
    // stream3 must wait for stream2 to finish accessing the allocation before
    // freeing the allocation
    cudaStreamWaitEvent(stream3, event2);
    cudaFreeAsync(ptr, stream3);
    

Memory allocated with `cudaMalloc()` can be freed with with `cudaFreeAsync()`. As above, all accesses to the memory must be complete before the free operation begins.
    
    
    cudaMalloc(&ptr, size);
    kernel<<<..., stream>>>(ptr, ...);
    cudaFreeAsync(ptr, stream);
    

Likewise, memory allocated with `cudaMallocAsync` can be freed with `cudaFree()`. When freeing such allocations through the `cudaFree()` API, the driver assumes that all accesses to the allocation are complete and performs no further synchronization. The user can use `cudaStreamQuery` / `cudaStreamSynchronize` / `cudaEventQuery` / `cudaEventSynchronize` / `cudaDeviceSynchronize` to guarantee that the appropriate asynchronous work is complete and that the GPU will not try to access the allocation.
    
    
    cudaMallocAsync(&ptr, size,stream);
    kernel<<<..., stream>>>(ptr, ...);
    // synchronize is needed to avoid prematurely freeing the memory
    cudaStreamSynchronize(stream);
    cudaFree(ptr);
    

## 4.3.3. Memory Pools

Memory pools encapsulate virtual address and physical memory resources that are allocated and managed according to the pools attributes and properties. The primary aspect of a memory pool is the kind and location of memory it manages.

All calls to `cudaMallocAsync` use resources from memory pool. If a memory pool is not specified, `cudaMallocAsync` uses the current memory pool of the supplied stream’s device. The current memory pool for a device may be set with `cudaDeviceSetMempool` and queried with `cudaDeviceGetMempool`. Each device has a default memory pool, which is active if `cudaDeviceSetMempool` has not been called.

The API `cudaMallocFromPoolAsync` and [c++ overloads of cudaMallocAsync](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1ga31efcffc48981621feddd98d71a0feb) allow a user to specify the pool to be used for an allocation without setting it as the current pool. The APIs `cudaDeviceGetDefaultMempool` and `cudaMemPoolCreate` return handles to memory pools. `cudaMemPoolSetAttribute` and `cudaMemPoolGetAttribute` control the attributes of memory pools.

Note

The mempool current to a device will be local to that device. So allocating without specifying a memory pool will always yield an allocation local to the stream’s device.

### 4.3.3.1. Default/Implicit Pools

The default memory pool of a device can be retrieved by calling `cudaDeviceGetDefaultMempool`. Allocations from the default memory pool of a device are non-migratable device allocation located on that device. These allocations will always be accessible from that device. The accessibility of the default memory pool can be modified with `cudaMemPoolSetAccess` and queried with `cudaMemPoolGetAccess`. Since the default pools do not need to be explicitly created, they are sometimes referred to as implicit pools. The default memory pool of a device does not support IPC.

### 4.3.3.2. Explicit Pools

`cudaMemPoolCreate` creates an explicit pool. This allows applications to request properties for their allocation beyond what is provided by the default/implicit pools. These include properties such as IPC capability, maximum pool size, allocations resident on a specific CPU NUMA node on supported platforms etc.
    
    
    // create a pool similar to the implicit pool on device 0
    int device = 0;
    cudaMemPoolProps poolProps = { };
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = device;
    poolProps.location.type = cudaMemLocationTypeDevice;
    
    cudaMemPoolCreate(&memPool, &poolProps));
    

The following code snippet illustrates an example of creating an IPC capable memory pool on a valid CPU NUMA node.
    
    
    // create a pool resident on a CPU NUMA node that is capable of IPC sharing (via a file descriptor).
    int cpu_numa_id = 0;
    cudaMemPoolProps poolProps = { };
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = cpu_numa_id;
    poolProps.location.type = cudaMemLocationTypeHostNuma;
    poolProps.handleType = cudaMemHandleTypePosixFileDescriptor;
    
    cudaMemPoolCreate(&ipcMemPool, &poolProps));
    

### 4.3.3.3. Device Accessibility for Multi-GPU Support

Like allocation accessibility controlled through the virtual memory management APIs, memory pool allocation accessibility does not follow `cudaDeviceEnablePeerAccess` or `cuCtxEnablePeerAccess`. For memory pools, the API `cudaMemPoolSetAccess` modifies what devices can access allocations from a pool. By default, allocations are accessible only from the device where the allocations are located. This access cannot be revoked. To enable access from other devices, the accessing device must be peer capable with the memory pool’s device. This can be verified with `cudaDeviceCanAccessPeer`. If the peer capability is not checked, the set access may fail with `cudaErrorInvalidDevice`. However, if no allocations had been made from the pool, the `cudaMemPoolSetAccess` call may succeed even when the devices are not peer capable. In this case, the next allocation from the pool will fail.

It is worth noting that `cudaMemPoolSetAccess` affects all allocations from the memory pool, not just future ones. Likewise, the accessibility reported by `cudaMemPoolGetAccess` applies to all allocations from the pool, not just future ones. Changing the accessibility settings of a pool for a given GPU frequently is not recommended. That is, once a pool is made accessible from a given GPU, it should remain accessible from that GPU for the lifetime of the pool.
    
    
    // snippet showing usage of cudaMemPoolSetAccess:
    cudaError_t setAccessOnDevice(cudaMemPool_t memPool, int residentDevice,
                  int accessingDevice) {
        cudaMemAccessDesc accessDesc = {};
        accessDesc.location.type = cudaMemLocationTypeDevice;
        accessDesc.location.id = accessingDevice;
        accessDesc.flags = cudaMemAccessFlagsProtReadWrite;
    
        int canAccess = 0;
        cudaError_t error = cudaDeviceCanAccessPeer(&canAccess, accessingDevice,
                  residentDevice);
        if (error != cudaSuccess) {
            return error;
        } else if (canAccess == 0) {
            return cudaErrorPeerAccessUnsupported;
        }
    
        // Make the address accessible
        return cudaMemPoolSetAccess(memPool, &accessDesc, 1);
    }
    

### 4.3.3.4. Enabling Memory Pools for IPC

Memory pools can be enabled for interprocess communication (IPC) to allow easy, efficient and secure sharing of GPU memory between processes. CUDA’s IPC memory pools provide the same security benefits as CUDA’s [virtual memory management APIs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#virtual-memory-management).

There are two steps to sharing memory between processes with memory pools: the processes first needs to share access to the pool, then share specific allocations from that pool. The first step establishes and enforces security. The second step coordinates what virtual addresses are used in each process and when mappings need to be valid in the importing process.

#### 4.3.3.4.1. Creating and Sharing IPC Memory Pools

Sharing access to a pool involves retrieving an OS-native handle to the pool with `cudaMemPoolExportToShareableHandle()`, transferring the handle to the importing process using OS-native IPC mechanisms, and then creating an imported memory pool with the `cudaMemPoolImportFromShareableHandle()` API. For `cudaMemPoolExportToShareableHandle` to succeed, the memory pool must have been created with the requested handle type specified in the pool properties structure.

Please reference [samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/streamOrderedAllocationIPC) for the appropriate IPC mechanisms to transfer the OS-native handle between processes. The rest of the procedure can be found in the following code snippets.
    
    
    // in exporting process
    // create an exportable IPC capable pool on device 0
    cudaMemPoolProps poolProps = { };
    poolProps.allocType = cudaMemAllocationTypePinned;
    poolProps.location.id = 0;
    poolProps.location.type = cudaMemLocationTypeDevice;
    
    // Setting handleTypes to a non zero value will make the pool exportable (IPC capable)
    poolProps.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    
    cudaMemPoolCreate(&memPool, &poolProps));
    
    // FD based handles are integer types
    int fdHandle = 0;
    
    
    // Retrieve an OS native handle to the pool.
    // Note that a pointer to the handle memory is passed in here.
    cudaMemPoolExportToShareableHandle(&fdHandle,
                 memPool,
                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                 0);
    
    // The handle must be sent to the importing process with the appropriate
    // OS-specific APIs.
    
    
    
    // in importing process
     int fdHandle;
    // The handle needs to be retrieved from the exporting process with the
    // appropriate OS-specific APIs.
    // Create an imported pool from the shareable handle.
    // Note that the handle is passed by value here.
    cudaMemPoolImportFromShareableHandle(&importedMemPool,
              (void*)fdHandle,
              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
              0);
    

#### 4.3.3.4.2. Set Access in the Importing Process

Imported memory pools are initially only accessible from their resident device. The imported memory pool does not inherit any accessibility set by the exporting process. The importing process needs to enable access with `cudaMemPoolSetAccess` from any GPU it plans to access the memory from.

If the imported memory pool belongs to a device that is not visible to importing process, the user must use the `cudaMemPoolSetAccess` API to enable access from the GPUs the allocations will be used on. (See [Device Accessibility for Multi-GPU Support](#stream-ordered-deviceaccessibility))

#### 4.3.3.4.3. Creating and Sharing Allocations from an Exported Pool

Once the pool has been shared, allocations made with `cudaMallocAsync()` from the pool in the exporting process can be shared with processes that have imported the pool. Since the pool’s security policy is established and verified at the pool level, the OS does not need extra bookkeeping to provide security for specific pool allocations. In other words, the opaque `cudaMemPoolPtrExportData` required to import a pool allocation may be sent to the importing process using any mechanism.

While allocations may be exported and imported without synchronizing with the allocating stream in any way, the importing process must follow the same rules as the exporting process when accessing the allocation. Specifically, access to the allocation must happen after the allocation operation in the allocating stream executes. The two following code snippets show `cudaMemPoolExportPointer()` and `cudaMemPoolImportPointer()` sharing the allocation with an IPC event used to guarantee that the allocation isn’t accessed in the importing process before the allocation is ready.
    
    
    // preparing an allocation in the exporting process
    cudaMemPoolPtrExportData exportData;
    cudaEvent_t readyIpcEvent;
    cudaIpcEventHandle_t readyIpcEventHandle;
    
    // ipc event for coordinating between processes
    // cudaEventInterprocess flag makes the event an ipc event
    // cudaEventDisableTiming  is set for performance reasons
    
    cudaEventCreate(&readyIpcEvent, cudaEventDisableTiming | cudaEventInterprocess)
    
    // allocate from the exporting mem pool
    cudaMallocAsync(&ptr, size,exportMemPool, stream);
    
    // event for sharing when the allocation is ready.
    cudaEventRecord(readyIpcEvent, stream);
    cudaMemPoolExportPointer(&exportData, ptr);
    cudaIpcGetEventHandle(&readyIpcEventHandle, readyIpcEvent);
    
    // Share IPC event and pointer export data with the importing process using
    //  any mechanism. Here we copy the data into shared memory
    shmem->ptrData = exportData;
    shmem->readyIpcEventHandle = readyIpcEventHandle;
    // signal consumers data is ready
    
    
    
    // Importing an allocation
    cudaMemPoolPtrExportData *importData = &shmem->prtData;
    cudaEvent_t readyIpcEvent;
    cudaIpcEventHandle_t *readyIpcEventHandle = &shmem->readyIpcEventHandle;
    
    // Need to retrieve the ipc event handle and the export data from the
    // exporting process using any mechanism.  Here we are using shmem and just
    // need synchronization to make sure the shared memory is filled in.
    
    cudaIpcOpenEventHandle(&readyIpcEvent, readyIpcEventHandle);
    
    // import the allocation. The operation does not block on the allocation being ready.
    cudaMemPoolImportPointer(&ptr, importedMemPool, importData);
    
    // Wait for the prior stream operations in the allocating stream to complete before
    // using the allocation in the importing process.
    cudaStreamWaitEvent(stream, readyIpcEvent);
    kernel<<<..., stream>>>(ptr, ...);
    

When freeing the allocation, the allocation must be freed in the importing process before it is freed in the exporting process. The following code snippet demonstrates the use of CUDA IPC events to provide the required synchronization between the `cudaFreeAsync` operations in both processes. Access to the allocation from the importing process is obviously restricted by the free operation in the importing process side. It is worth noting that `cudaFree` can be used to free the allocation in both processes and that other stream synchronization APIs may be used instead of CUDA IPC events.
    
    
    // The free must happen in importing process before the exporting process
    kernel<<<..., stream>>>(ptr, ...);
    
    // Last access in importing process
    cudaFreeAsync(ptr, stream);
    
    // Access not allowed in the importing process after the free
    cudaIpcEventRecord(finishedIpcEvent, stream);
    
    
    
    // Exporting process
    // The exporting process needs to coordinate its free with the stream order
    // of the importing process’s free.
    cudaStreamWaitEvent(stream, finishedIpcEvent);
    kernel<<<..., stream>>>(ptrInExportingProcess, ...);
    
    // The free in the importing process doesn’t stop the exporting process
    // from using the allocation.
    cudFreeAsync(ptrInExportingProcess,stream);
    

#### 4.3.3.4.4. IPC Export Pool Limitations

IPC pools currently do not support releasing physical blocks back to the OS. As a result the `cudaMemPoolTrimTo` API has no effect and the `cudaMemPoolAttrReleaseThreshold` is effectively ignored. This behavior is controlled by the driver, not the runtime and may change in a future driver update.

#### 4.3.3.4.5. IPC Import Pool Limitations

Allocating from an import pool is not allowed; specifically, import pools cannot be set current and cannot be used in the `cudaMallocFromPoolAsync` API. As such, the allocation reuse policy attributes do not have meaning for these pools.

IPC Import pools, like IPC export pools, currently do not support releasing physical blocks back to the OS.

The resource usage stat attribute queries only reflect the allocations imported into the process and the associated physical memory.

## 4.3.4. Best Practices and Tuning

### 4.3.4.1. Query for Support

An application can determine whether or not a device supports the stream-ordered memory allocator by calling `cudaDeviceGetAttribute()` (see [developer blog](https://developer.nvidia.com/blog/cuda-pro-tip-the-fast-way-to-query-device-properties/)) with the device attribute `cudaDevAttrMemoryPoolsSupported`.

IPC memory pool support can be queried with the `cudaDevAttrMemoryPoolSupportedHandleTypes` device attribute. This attribute was added in CUDA 11.3, and older drivers will return `cudaErrorInvalidValue` when this attribute is queried.
    
    
    int driverVersion = 0;
    int deviceSupportsMemoryPools = 0;
    int poolSupportedHandleTypes = 0;
    cudaDriverGetVersion(&driverVersion);
    if (driverVersion >= 11020) {
        cudaDeviceGetAttribute(&deviceSupportsMemoryPools,
                               cudaDevAttrMemoryPoolsSupported, device);
    }
    if (deviceSupportsMemoryPools != 0) {
        // `device` supports the Stream-Ordered Memory Allocator
    }
    
    if (driverVersion >= 11030) {
        cudaDeviceGetAttribute(&poolSupportedHandleTypes,
                  cudaDevAttrMemoryPoolSupportedHandleTypes, device);
    }
    if (poolSupportedHandleTypes & cudaMemHandleTypePosixFileDescriptor) {
       // Pools on the specified device can be created with posix file descriptor-based IPC
    }
    

Performing the driver version check before the query avoids hitting a `cudaErrorInvalidValue` error on drivers where the attribute was not yet defined. One can use `cudaGetLastError` to clear the error instead of avoiding it.

### 4.3.4.2. Physical Page Caching Behavior

By default, the allocator tries to minimize the physical memory owned by a pool. To minimize the OS calls to allocate and free physical memory, applications must configure a memory footprint for each pool. Applications can do this with the release threshold attribute (`cudaMemPoolAttrReleaseThreshold`).

The release threshold is the amount of memory in bytes a pool should hold onto before trying to release memory back to the OS. When more than the release threshold bytes of memory are held by the memory pool, the allocator will try to release memory back to the OS on the next call to stream, event or device synchronize. Setting the release threshold to UINT64_MAX will prevent the driver from attempting to shrink the pool after every synchronization.
    
    
    Cuuint64_t setVal = UINT64_MAX;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    

Applications that set `cudaMemPoolAttrReleaseThreshold` high enough to effectively disable memory pool shrinking may wish to explicitly shrink a memory pool’s memory footprint. `cudaMemPoolTrimTo` allows applications to do so. When trimming a memory pool’s footprint, the `minBytesToKeep` parameter allows an application to hold onto a specified amount of memory, for example the amount it expects to need in a subsequent phase of execution.
    
    
    Cuuint64_t setVal = UINT64_MAX;
    cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, &setVal);
    
    // application phase needing a lot of memory from the stream-ordered allocator
    for (i=0; i<10; i++) {
        for (j=0; j<10; j++) {
            cudaMallocAsync(&ptrs[j],size[j], stream);
        }
        kernel<<<...,stream>>>(ptrs,...);
        for (j=0; j<10; j++) {
            cudaFreeAsync(ptrs[j], stream);
        }
    }
    
    // Process does not need as much memory for the next phase.
    // Synchronize so that the trim operation will know that the allocations are no
    // longer in use.
    cudaStreamSynchronize(stream);
    cudaMemPoolTrimTo(mempool, 0);
    
    // Some other process/allocation mechanism can now use the physical memory
    // released by the trimming operation.
    

### 4.3.4.3. Resource Usage Statistics

Querying the `cudaMemPoolAttrReservedMemCurrent` attribute of a pool reports the current total physical GPU memory consumed by the pool. Querying the `cudaMemPoolAttrUsedMemCurrent` of a pool returns the total size of all of the memory allocated from the pool and not available for reuse.

The`cudaMemPoolAttr*MemHigh` attributes are watermarks recording the max value achieved by the respective `cudaMemPoolAttr*MemCurrent` attribute since last reset. They can be reset to the current value by using the `cudaMemPoolSetAttribute` API.
    
    
    // sample helper functions for getting the usage statistics in bulk
    struct usageStatistics {
        cuuint64_t reserved;
        cuuint64_t reservedHigh;
        cuuint64_t used;
        cuuint64_t usedHigh;
    };
    
    void getUsageStatistics(cudaMemoryPool_t memPool, struct usageStatistics *statistics)
    {
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemCurrent, statistics->reserved);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, statistics->reservedHigh);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemCurrent, statistics->used);
        cudaMemPoolGetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, statistics->usedHigh);
    }
    
    
    // resetting the watermarks will make them take on the current value.
    void resetStatistics(cudaMemoryPool_t memPool)
    {
        cuuint64_t value = 0;
        cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReservedMemHigh, &value);
        cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrUsedMemHigh, &value);
    }
    

### 4.3.4.4. Memory Reuse Policies

In order to service an allocation request, the driver attempts to reuse memory that was previously freed via `cudaFreeAsync()` before attempting to allocate more memory from the OS. For example, memory freed in a stream can be reused immediately in a subsequent allocation request on the same stream. When a stream is synchronized with the CPU, the memory that was previously freed in that stream becomes available for reuse for an allocation in any stream. Reuse policies can be applied to both default and explicit memory pools.

The stream-ordered allocator has a few controllable allocation policies. The pool attributes `cudaMemPoolReuseFollowEventDependencies`, `cudaMemPoolReuseAllowOpportunistic`, and `cudaMemPoolReuseAllowInternalDependencies` control these policies and are detailed below. These policies can be enabled or disabled through a call to `cudaMemPoolSetAttribute`. Upgrading to a newer CUDA driver may change, enhance, augment and/or reorder the enumeration of the reuse policies.

#### 4.3.4.4.1. cudaMemPoolReuseFollowEventDependencies

Before allocating more physical GPU memory, the allocator examines dependency information established by CUDA events and tries to allocate from memory freed in another stream.
    
    
    cudaMallocAsync(&ptr, size, originalStream);
    kernel<<<..., originalStream>>>(ptr, ...);
    cudaFreeAsync(ptr, originalStream);
    cudaEventRecord(event,originalStream);
    
    // waiting on the event that captures the free in another stream
    // allows the allocator to reuse the memory to satisfy
    // a new allocation request in the other stream when
    // cudaMemPoolReuseFollowEventDependencies is enabled.
    cudaStreamWaitEvent(otherStream, event);
    cudaMallocAsync(&ptr2, size, otherStream);
    

#### 4.3.4.4.2. cudaMemPoolReuseAllowOpportunistic

When the `cudaMemPoolReuseAllowOpportunistic` policy is enabled, the allocator examines freed allocations to see if the free operations stream order semantic has been met, for example the stream has passed the point of execution indicated by the free operation. When this policy is disabled, the allocator will still reuse memory made available when a stream is synchronized with the CPU. Disabling this policy does not stop the `cudaMemPoolReuseFollowEventDependencies` from applying.
    
    
    cudaMallocAsync(&ptr, size, originalStream);
    kernel<<<..., originalStream>>>(ptr, ...);
    cudaFreeAsync(ptr, originalStream);
    
    
    // after some time, the kernel finishes running
    wait(10);
    
    // When cudaMemPoolReuseAllowOpportunistic is enabled this allocation request
    // can be fulfilled with the prior allocation based on the progress of originalStream.
    cudaMallocAsync(&ptr2, size, otherStream);
    

#### 4.3.4.4.3. cudaMemPoolReuseAllowInternalDependencies

Failing to allocate and map more physical memory from the OS, the driver will look for memory whose availability depends on another stream’s pending progress. If such memory is found, the driver will insert the required dependency into the allocating stream and reuse the memory.
    
    
    cudaMallocAsync(&ptr, size, originalStream);
    kernel<<<..., originalStream>>>(ptr, ...);
    cudaFreeAsync(ptr, originalStream);
    
    // When cudaMemPoolReuseAllowInternalDependencies is enabled
    // and the driver fails to allocate more physical memory, the driver may
    // effectively perform a cudaStreamWaitEvent in the allocating stream
    // to make sure that future work in ‘otherStream’ happens after the work
    // in the original stream that would be allowed to access the original allocation.
    cudaMallocAsync(&ptr2, size, otherStream);
    

#### 4.3.4.4.4. Disabling Reuse Policies

While the controllable reuse policies improve memory reuse, users may want to disable them. Allowing opportunistic reuse (such as `cudaMemPoolReuseAllowOpportunistic`) introduces run to run variance in allocation patterns based on the interleaving of CPU and GPU execution. Internal dependency insertion (such as `cudaMemPoolReuseAllowInternalDependencies`) can serialize work in unexpected and potentially non-deterministic ways when the user would rather explicitly synchronize an event or stream on allocation failure.

### 4.3.4.5. Synchronization API Actions

One of the optimizations that comes with the allocator being part of the CUDA driver is integration with the synchronize APIs. When the user requests that the CUDA driver synchronize, the driver waits for asynchronous work to complete. Before returning, the driver will determine what frees the synchronization guaranteed to be completed. These allocations are made available for allocation regardless of specified stream or disabled allocation policies. The driver also checks `cudaMemPoolAttrReleaseThreshold` here and releases any excess physical memory that it can.

## 4.3.5. Addendums

### 4.3.5.1. cudaMemcpyAsync Current Context/Device Sensitivity

In the current CUDA driver, any async `memcpy` involving memory from `cudaMallocAsync` should be done using the specified stream’s context as the calling thread’s current context. This is not necessary for `cudaMemcpyPeerAsync`, as the device primary contexts specified in the API are referenced instead of the current context.

### 4.3.5.2. cudaPointerGetAttributes Query

Invoking `cudaPointerGetAttributes` on an allocation after invoking `cudaFreeAsync` on it results in undefined behavior. Specifically, it does not matter if an allocation is still accessible from a given stream: the behavior is still undefined.

### 4.3.5.3. cudaGraphAddMemsetNode

`cudaGraphAddMemsetNode` does not work with memory allocated via the stream ordered allocator. However, memsets of the allocations can be stream captured.

### 4.3.5.4. Pointer Attributes

The `cudaPointerGetAttributes` query works on stream-ordered allocations. Since stream-ordered allocations are not context associated, querying `CU_POINTER_ATTRIBUTE_CONTEXT` will succeed but return NULL in `*data`. The attribute `CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL` can be used to determine the location of the allocation: this can be useful when selecting a context for making p2h2p copies using `cudaMemcpyPeerAsync`. The attribute `CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE` was added in CUDA 11.3 and can be useful for debugging and for confirming which pool an allocation comes from before doing IPC.

### 4.3.5.5. CPU Virtual Memory

When using CUDA stream-ordered memory allocator APIs, avoid setting VRAM limitations with “ulimit -v” as this is not supported.
