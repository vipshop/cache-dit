# 10. Memory Optimizations


Memory optimizations are the most important area for performance. The goal is to maximize the use of the hardware by maximizing bandwidth. Bandwidth is best served by using as much fast memory and as little slow-access memory as possible. This chapter discusses the various kinds of memory on the host and device and how best to set up data items to use the memory effectively.


##  10.1. Data Transfer Between Host and Device 

The peak theoretical bandwidth between the device memory and the GPU is much higher (898 GB/s on the NVIDIA Tesla V100, for example) than the peak theoretical bandwidth between host memory and device memory (16 GB/s on the PCIe x16 Gen3). Hence, for best overall application performance, it is important to minimize data transfer between the host and the device, even if that means running kernels on the GPU that do not demonstrate any speedup compared with running them on the host CPU.

Note

**High Priority:** Minimize data transfer between the host and the device, even if it means running some kernels on the device that do not show performance gains when compared with running them on the host CPU.

Intermediate data structures should be created in device memory, operated on by the device, and destroyed without ever being mapped by the host or copied to host memory.

Also, because of the overhead associated with each transfer, batching many small transfers into one larger transfer performs significantly better than making each transfer separately, even if doing so requires packing non-contiguous regions of memory into a contiguous buffer and then unpacking after the transfer.

Finally, higher bandwidth between the host and the device is achieved when using _page-locked_ (or _pinned_) memory, as discussed in the CUDA C++ Programming Guide and the [Pinned Memory](#pinned-memory) section of this document.

###  10.1.1. Pinned Memory 

Page-locked or pinned memory transfers attain the highest bandwidth between the host and the device. On PCIe x16 Gen3 cards, for example, pinned memory can attain roughly 12 GB/s transfer rates.

Pinned memory is allocated using the `cudaHostAlloc()` functions in the Runtime API. The `bandwidthTest` CUDA Sample shows how to use these functions as well as how to measure memory transfer performance.

For regions of system memory that have already been pre-allocated, `cudaHostRegister()` can be used to pin the memory on-the-fly without the need to allocate a separate buffer and copy the data into it.

Pinned memory should not be overused. Excessive use can reduce overall system performance because pinned memory is a scarce resource, but how much is too much is difficult to know in advance. Furthermore, the pinning of system memory is a heavyweight operation compared to most normal system memory allocations, so as with all optimizations, test the application and the systems it runs on for optimal performance parameters.

###  10.1.2. Asynchronous and Overlapping Transfers with Computation 

Data transfers between the host and the device using `cudaMemcpy()` are blocking transfers; that is, control is returned to the host thread only after the data transfer is complete. The `cudaMemcpyAsync()` function is a non-blocking variant of `cudaMemcpy()` in which control is returned immediately to the host thread. In contrast with `cudaMemcpy()`, the asynchronous transfer version _requires_ pinned host memory (see [Pinned Memory](#pinned-memory)), and it contains an additional argument, a stream ID. A _stream_ is simply a sequence of operations that are performed in order on the device. Operations in different streams can be interleaved and in some cases overlapped - a property that can be used to hide data transfers between the host and the device.

Asynchronous transfers enable overlap of data transfers with computation in two different ways. On all CUDA-enabled devices, it is possible to overlap host computation with asynchronous data transfers and with device computations. For example, [Asynchronous and Overlapping Transfers with Computation](#asynchronous-transfers-and-overlapping-transfers-with-computation) demonstrates how host computation in the routine `cpuFunction()` is performed while data is transferred to the device and a kernel using the device is executed.

Overlapping computation and data transfers
    
    
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, 0);
    kernel<<<grid, block>>>(a_d);
    cpuFunction();
    

The last argument to the `cudaMemcpyAsync()` function is the stream ID, which in this case uses the default stream, stream 0. The kernel also uses the default stream, and it will not begin execution until the memory copy completes; therefore, no explicit synchronization is needed. Because the memory copy and the kernel both return control to the host immediately, the host function `cpuFunction()` overlaps their execution.

In [Asynchronous and Overlapping Transfers with Computation](#asynchronous-transfers-and-overlapping-transfers-with-computation), the memory copy and kernel execution occur sequentially. On devices that are capable of concurrent copy and compute, it is possible to overlap kernel execution on the device with data transfers between the host and the device. Whether a device has this capability is indicated by the `asyncEngineCount` field of the `cudaDeviceProp` structure (or listed in the output of the `deviceQuery` CUDA Sample). On devices that have this capability, the overlap once again requires pinned host memory, and, in addition, the data transfer and kernel must use different, non-default streams (streams with non-zero stream IDs). Non-default streams are required for this overlap because memory copy, memory set functions, and kernel calls that use the default stream begin only after all preceding calls on the device (in any stream) have completed, and no operation on the device (in any stream) commences until they are finished.

[Asynchronous and Overlapping Transfers with Computation](#asynchronous-transfers-and-overlapping-transfers-with-computation) illustrates the basic technique.

Concurrent copy and execute
    
    
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
    kernel<<<grid, block, 0, stream2>>>(otherData_d);
    

In this code, two streams are created and used in the data transfer and kernel executions as specified in the last arguments of the `cudaMemcpyAsync` call and the kernel’s execution configuration.

[Asynchronous and Overlapping Transfers with Computation](#asynchronous-transfers-and-overlapping-transfers-with-computation) demonstrates how to overlap kernel execution with asynchronous data transfer. This technique could be used when the data dependency is such that the data can be broken into chunks and transferred in multiple stages, launching multiple kernels to operate on each chunk as it arrives. [Sequential copy and execute](#sequential-copy-and-execute) and [Staged concurrent copy and execute](#staged-concurrent-copy-and-execute) demonstrate this. They produce equivalent results. The first segment shows the reference sequential implementation, which transfers and operates on an array of _N_ floats (where _N_ is assumed to be evenly divisible by nThreads).

Sequential copy and execute
    
    
    cudaMemcpy(a_d, a_h, N*sizeof(float), dir);
    kernel<<<N/nThreads, nThreads>>>(a_d);
    

[Staged concurrent copy and execute](#staged-concurrent-copy-and-execute) shows how the transfer and kernel execution can be broken up into nStreams stages. This approach permits some overlapping of the data transfer and execution.

Staged concurrent copy and execute
    
    
    size=N*sizeof(float)/nStreams;
    for (i=0; i<nStreams; i++) {
        offset = i*N/nStreams;
        cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
        kernel<<<N/(nThreads*nStreams), nThreads, 0,
                 stream[i]>>>(a_d+offset);
    }
    

(In [Staged concurrent copy and execute](#staged-concurrent-copy-and-execute), it is assumed that _N_ is evenly divisible by `nThreads*nStreams`.) Because execution within a stream occurs sequentially, none of the kernels will launch until the data transfers in their respective streams complete. Current GPUs can simultaneously process asynchronous data transfers and execute kernels. GPUs with a single copy engine can perform one asynchronous data transfer and execute kernels whereas GPUs with two copy engines can simultaneously perform one asynchronous data transfer from the host to the device, one asynchronous data transfer from the device to the host, and execute kernels. The number of copy engines on a GPU is given by the `asyncEngineCount` field of the `cudaDeviceProp` structure, which is also listed in the output of the `deviceQuery` CUDA Sample. (It should be mentioned that it is not possible to overlap a blocking transfer with an asynchronous transfer, because the blocking transfer occurs in the default stream, so it will not begin until all previous CUDA calls complete. It will not allow any other CUDA call to begin until it has completed.) A diagram depicting the timeline of execution for the two code segments is shown in [Figure 1](#timeline-comparison-for-copy-and-kernel-execution-figure), and `nStreams` is equal to 4 for [Staged concurrent copy and execute](#staged-concurrent-copy-and-execute) in the bottom half of the figure.

![Timeline comparison for copy and kernel execution](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/timeline-comparison-for-copy-and-kernel-execution.png)

Figure 1 Timeline comparison for copy and kernel execution

Top
    

Sequential

Bottom
    

Concurrent

For this example, it is assumed that the data transfer and kernel execution times are comparable. In such cases, and when the execution time (_tE_) exceeds the transfer time (_tT_), a rough estimate for the overall time is _tE + tT/nStreams_ for the staged version versus _tE + tT_ for the sequential version. If the transfer time exceeds the execution time, a rough estimate for the overall time is _tT + tE/nStreams_.

###  10.1.3. Zero Copy 

_Zero copy_ is a feature that was added in version 2.2 of the CUDA Toolkit. It enables GPU threads to directly access host memory. For this purpose, it requires mapped pinned (non-pageable) memory. On integrated GPUs (i.e., GPUs with the integrated field of the CUDA device properties structure set to 1), mapped pinned memory is always a performance gain because it avoids superfluous copies as integrated GPU and CPU memory are physically the same. On discrete GPUs, mapped pinned memory is advantageous only in certain cases. Because the data is not cached on the GPU, mapped pinned memory should be read or written only once, and the global loads and stores that read and write the memory should be coalesced. Zero copy can be used in place of streams because kernel-originated data transfers automatically overlap kernel execution without the overhead of setting up and determining the optimal number of streams.

Note

**Low Priority:** Use zero-copy operations on integrated GPUs for CUDA Toolkit version 2.2 and later.

The host code in [Zero-copy host code](#zero-copy-host-code) shows how zero copy is typically set up.

Zero-copy host code
    
    
    float *a_h, *a_map;
    ...
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.canMapHostMemory)
        exit(0);
    cudaSetDeviceFlags(cudaDeviceMapHost);
    cudaHostAlloc(&a_h, nBytes, cudaHostAllocMapped);
    cudaHostGetDevicePointer(&a_map, a_h, 0);
    kernel<<<gridSize, blockSize>>>(a_map);
    

In this code, the `canMapHostMemory` field of the structure returned by `cudaGetDeviceProperties()` is used to check that the device supports mapping host memory to the device’s address space. Page-locked memory mapping is enabled by calling `cudaSetDeviceFlags()` with `cudaDeviceMapHost`. Note that `cudaSetDeviceFlags()` must be called prior to setting a device or making a CUDA call that requires state (that is, essentially, before a context is created). Page-locked mapped host memory is allocated using `cudaHostAlloc()`, and the pointer to the mapped device address space is obtained via the function `cudaHostGetDevicePointer()`. In the code in [Zero-copy host code](#zero-copy-host-code), `kernel()` can reference the mapped pinned host memory using the pointer `a_map` in exactly the same was as it would if a_map referred to a location in device memory.

Note

Mapped pinned host memory allows you to overlap CPU-GPU memory transfers with computation while avoiding the use of CUDA streams. But since any repeated access to such memory areas causes repeated CPU-GPU transfers, consider creating a second area in device memory to manually cache the previously read host memory data.

###  10.1.4. Unified Virtual Addressing 

Devices of [compute capability](#cuda-compute-capability) 2.0 and later support a special addressing mode called _Unified Virtual Addressing_ (UVA) on 64-bit Linux and Windows. With UVA, the host memory and the device memories of all installed supported devices share a single virtual address space.

Prior to UVA, an application had to keep track of which pointers referred to device memory (and for which device) and which referred to host memory as a separate bit of metadata (or as hard-coded information in the program) for each pointer. Using UVA, on the other hand, the physical memory space to which a pointer points can be determined simply by inspecting the value of the pointer using `cudaPointerGetAttributes()`.

Under UVA, pinned host memory allocated with `cudaHostAlloc()` will have identical host and device pointers, so it is not necessary to call `cudaHostGetDevicePointer()` for such allocations. Host memory allocations pinned after-the-fact via `cudaHostRegister()`, however, will continue to have different device pointers than their host pointers, so `cudaHostGetDevicePointer()` remains necessary in that case.

UVA is also a necessary precondition for enabling peer-to-peer (P2P) transfer of data directly across the PCIe bus or NVLink for supported GPUs in supported configurations, bypassing host memory.

See the CUDA C++ Programming Guide for further explanations and software requirements for UVA and P2P.


##  10.2. Device Memory Spaces 

CUDA devices use several memory spaces, which have different characteristics that reflect their distinct usages in CUDA applications. These memory spaces include global, local, shared, texture, and registers, as shown in [Figure 2](#memory-spaces-cuda-device-figure).

![Memory spaces on a CUDA device](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/memory-spaces-on-cuda-device.png)

Figure 2 Memory spaces on a CUDA device

Of these different memory spaces, global memory is the most plentiful; see Features and Technical Specifications of the CUDA C++ Programming Guide for the amounts of memory available in each memory space at each [compute capability](#cuda-compute-capability) level. Global, local, and texture memory have the greatest access latency, followed by constant memory, shared memory, and the register file.

The various principal traits of the memory types are shown in [Table 1](#salient-features-device-memory-table).

Table 1 Salient Features of Device Memory Memory | Location on/off chip | Cached | Access | Scope | Lifetime  
---|---|---|---|---|---  
Register | On | n/a | R/W | 1 thread | Thread  
Local | Off | Yes†† | R/W | 1 thread | Thread  
Shared | On | n/a | R/W | All threads in block | Block  
Global | Off | † | R/W | All threads + host | Host allocation  
Constant | Off | Yes | R | All threads + host | Host allocation  
Texture | Off | Yes | R | All threads + host | Host allocation  
† Cached in L1 and L2 by default on devices of compute capability 6.0 and 7.x; cached only in L2 by default on devices of lower compute capabilities, though some allow opt-in to caching in L1 as well via compilation flags. |  |  |  |  |   
†† Cached in L1 and L2 by default except on devices of compute capability 5.x; devices of compute capability 5.x cache locals only in L2. |  |  |  |  |   
  
In the case of texture access, if a texture reference is bound to a linear array in global memory, then the device code can write to the underlying array. Texture references that are bound to CUDA arrays can be written to via surface-write operations by binding a surface to the same underlying CUDA array storage). Reading from a texture while writing to its underlying global memory array in the same kernel launch should be avoided because the texture caches are read-only and are not invalidated when the associated global memory is modified.

###  10.2.1. Coalesced Access to Global Memory 

A very important performance consideration in programming for CUDA-capable GPU architectures is the coalescing of global memory accesses. Global memory loads and stores by threads of a warp are coalesced by the device into as few as possible transactions.

Note

**High Priority:** Ensure global memory accesses are coalesced whenever possible.

The access requirements for coalescing depend on the compute capability of the device and are documented in the CUDA C++ Programming Guide.

For devices of compute capability 6.0 or higher, the requirements can be summarized quite easily: the concurrent accesses of the threads of a warp will coalesce into a number of transactions equal to the number of 32-byte transactions necessary to service all of the threads of the warp.

For certain devices of compute capability 5.2, L1-caching of accesses to global memory can be optionally enabled. If L1-caching is enabled on these devices, the number of required transactions is equal to the number of required 128-byte aligned segments.

Note

On devices of compute capability 6.0 or higher, L1-caching is the default, however the data access unit is 32-byte regardless of whether global loads are cached in L1 or not.

On devices with GDDR memory, accessing memory in a coalesced way is even more important when ECC is turned on. Scattered accesses increase ECC memory transfer overhead, especially when writing data to global memory.

Coalescing concepts are illustrated in the following simple examples. These examples assume compute capability 6.0 or higher and that accesses are for 4-byte words, unless otherwise noted.

####  10.2.1.1. A Simple Access Pattern 

The first and simplest case of coalescing can be achieved by any CUDA-enabled device of compute capability 6.0 or higher: the _k_ -th thread accesses the _k_ -th word in a 32-byte aligned array. Not all threads need to participate.

For example, if the threads of a warp access adjacent 4-byte words (e.g., adjacent `float` values), four coalesced 32-byte transactions will service that memory access. Such a pattern is shown in Figure 3 <coalesced-access-figure>.

![Coalesced access](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/coalesced-access.png)

Figure 3 Coalesced access

This access pattern results in four 32-byte transactions, indicated by the red rectangles.

If from any of the four 32-byte segments only a subset of the words are requested (e.g. if several threads had accessed the same word or if some threads did not participate in the access), the full segment is fetched anyway. Furthermore, if accesses by the threads of the warp had been permuted within or accross the four segments, still only four 32-byte transactions would have been performed by a device with [compute capability](#cuda-compute-capability) 6.0 or higher.

####  10.2.1.2. A Sequential but Misaligned Access Pattern 

If sequential threads in a warp access memory that is sequential but not aligned with a 32-byte segment, five 32-byte segments will be requested, as shown in [Figure 4](#misaligned-sequential-addresses-fall-5-32-byte-l2-cache-seqments).

![Misaligned sequential addresses that fall within five 32-byte segments](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/misaligned-sequential-addresses.png)

Figure 4 Misaligned sequential addresses that fall within five 32-byte segments

Memory allocated through the CUDA Runtime API, such as via `cudaMalloc()`, is guaranteed to be aligned to at least 256 bytes. Therefore, choosing sensible thread block sizes, such as multiples of the warp size (i.e., 32 on current GPUs), facilitates memory accesses by warps that are properly aligned. (Consider what would happen to the memory addresses accessed by the second, third, and subsequent thread blocks if the thread block size was not a multiple of warp size, for example.)

####  10.2.1.3. Effects of Misaligned Accesses 

It is easy and informative to explore the ramifications of misaligned accesses using a simple copy kernel, such as the one in [A copy kernel that illustrates misaligned accesses](#a-copy-kernel-that-illustrates-misaligned-accesses).

A copy kernel that illustrates misaligned accesses
    
    
    __global__ void offsetCopy(float *odata, float* idata, int offset)
    {
        int xid = blockIdx.x * blockDim.x + threadIdx.x + offset;
        odata[xid] = idata[xid];
    }
    

In [A copy kernel that illustrates misaligned accesses](#a-copy-kernel-that-illustrates-misaligned-accesses), data is copied from the input array `idata` to the output array, both of which exist in global memory. The kernel is executed within a loop in host code that varies the parameter `offset` from 0 to 32 (for example, [Figure 4](#misaligned-sequential-addresses-fall-5-32-byte-l2-cache-seqments) corresponds to this misalignments). The effective bandwidth for the copy with various offsets on an NVIDIA Tesla V100 ([compute capability](#cuda-compute-capability) 7.0) is shown in [Figure 5](#performance-offsetcopy-kernel-figure).

![Performance of offsetCopy kernel](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/performance-of-offsetcopy-kernel.png)

Figure 5 Performance of offsetCopy kernel

For the NVIDIA Tesla V100, global memory accesses with no offset or with offsets that are multiples of 8 words result in four 32-byte transactions. The achieved bandwidth is approximately 790 GB/s. Otherwise, five 32-byte segments are loaded per warp, and we would expect approximately 4/5th of the memory throughput achieved with no offsets.

In this particular example, the offset memory throughput achieved is, however, approximately 9/10th, because adjacent warps reuse the cache lines their neighbors fetched. So while the impact is still evident it is not as large as we might have expected. It would have been more so if adjacent warps had not exhibited such a high degree of reuse of the over-fetched cache lines.

####  10.2.1.4. Strided Accesses 

As seen above, in the case of misaligned sequential accesses, caches help to alleviate the performance impact. It may be different with non-unit-strided accesses, however, and this is a pattern that occurs frequently when dealing with multidimensional data or matrices. For this reason, ensuring that as much as possible of the data in each cache line fetched is actually used is an important part of performance optimization of memory accesses on these devices.

To illustrate the effect of strided access on effective bandwidth, see the kernel `strideCopy()` in [A kernel to illustrate non-unit stride data copy](#a-kernel-to-illustrate-non-unit-stride-data-copy), which copies data with a stride of stride elements between threads from `idata` to `odata`.

A kernel to illustrate non-unit stride data copy
    
    
    __global__ void strideCopy(float *odata, float* idata, int stride)
    {
        int xid = (blockIdx.x*blockDim.x + threadIdx.x)*stride;
        odata[xid] = idata[xid];
    }
    

[Figure 6](#adjacent-threads-accessing-memory-with-stride-of-2-figure) illustrates such a situation; in this case, threads within a warp access words in memory with a stride of 2. This action leads to a load of eight L2 cache segments per warp on the Tesla V100 (compute capability 7.0).

![Adjacent threads accessing memory with a stride of 2](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/adjacent-threads-accessing-memory-with-stride-of-2.png)

Figure 6 Adjacent threads accessing memory with a stride of 2

A stride of 2 results in a 50% of load/store efficiency since half the elements in the transaction are not used and represent wasted bandwidth. As the stride increases, the effective bandwidth decreases until the point where 32 32-byte segments are loaded for the 32 threads in a warp, as indicated in [Figure 7](#performance-of-stridecopy-kernel-figure).

![Performance of strideCopy kernel](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/performance-of-stridecopy-kernel.png)

Figure 7 Performance of strideCopy kernel

As illustrated in [Figure 7](#performance-of-stridecopy-kernel-figure), non-unit-stride global memory accesses should be avoided whenever possible. One method for doing so utilizes shared memory, which is discussed in the next section.

###  10.2.2. L2 Cache 

Starting with CUDA 11.0, devices of compute capability 8.0 and above have the capability to influence persistence of data in the L2 cache. Because L2 cache is on-chip, it potentially provides higher bandwidth and lower latency accesses to global memory.

For more details refer to the L2 Access Management section in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#L2_access_intro).

####  10.2.2.1. L2 Cache Access Window 

When a CUDA kernel accesses a data region in the global memory repeatedly, such data accesses can be considered to be _persisting_. On the other hand, if the data is only accessed once, such data accesses can be considered to be _streaming_. A portion of the L2 cache can be set aside for persistent accesses to a data region in global memory. If this set-aside portion is not used by persistent accesses, then streaming or normal data accesses can use it.

The L2 cache set-aside size for persisting accesses may be adjusted, within limits:
    
    
    cudaGetDeviceProperties(&prop, device_id);
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, prop.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for persisting accesses */
    

Mapping of user data to L2 set-aside portion can be controlled using an access policy window on a CUDA stream or CUDA graph kernel node. The example below shows how to use the access policy window on a CUDA stream.
    
    
    cudaStreamAttrValue stream_attribute;                                         // Stream level attributes data structure
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(ptr); // Global Memory data pointer
    stream_attribute.accessPolicyWindow.num_bytes = num_bytes;                    // Number of bytes for persisting accesses.
                                                                                  // (Must be less than cudaDeviceProp::accessPolicyMaxWindowSize)
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                          // Hint for L2 cache hit ratio for persisting accesses in the num_bytes region
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting; // Type of access property on cache hit
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;  // Type of access property on cache miss.
    
    //Set the attributes to a CUDA stream of type cudaStream_t
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);
    

The access policy window requires a value for `hitRatio` and `num_bytes`. Depending on the value of the `num_bytes` parameter and the size of L2 cache, one may need to tune the value of `hitRatio` to avoid thrashing of L2 cache lines.

####  10.2.2.2. Tuning the Access Window Hit-Ratio 

The `hitRatio` parameter can be used to specify the fraction of accesses that receive the `hitProp` property. For example, if the `hitRatio` value is 0.6, 60% of the memory accesses in the global memory region [ptr..ptr+num_bytes) have the persisting property and 40% of the memory accesses have the streaming property. To understand the effect of `hitRatio` and `num_bytes`, we use a sliding window micro benchmark.

This microbenchmark uses a 1024 MB region in GPU global memory. First, we set aside 30 MB of the L2 cache for persisting accesses using `cudaDeviceSetLimit()`, as discussed above. Then, as shown in the figure below, we specify that the accesses to the first `freqSize * sizeof(int)` bytes of the memory region are persistent. This data will thus use the L2 set-aside portion. In our experiment, we vary the size of this persistent data region from 10 MB to 60 MB to model various scenarios where data fits in or exceeds the available L2 set-aside portion of 30 MB. Note that the NVIDIA Tesla A100 GPU has 40 MB of total L2 cache capacity. Accesses to the remaining data of the memory region (i.e., streaming data) are considered normal or streaming accesses and will thus use the remaining 10 MB of the non set-aside L2 portion (unless part of the L2 set-aside portion is unused).

[![Mapping Persistent data accesses to set-aside L2 in sliding window experiment](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/sliding-window-l2.png)](_images/sliding-window-l2.png)

Figure 8 Mapping Persistent data accesses to set-aside L2 in sliding window experiment

Consider the following kernel code and access window parameters, as the implementation of the sliding window experiment.
    
    
    __global__ void kernel(int *data_persistent, int *data_streaming, int dataSize, int freqSize) {
        int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
        /*Each CUDA thread accesses one element in the persistent data section
          and one element in the streaming data section.
          Because the size of the persistent memory region (freqSize * sizeof(int) bytes) is much
          smaller than the size of the streaming memory region (dataSize * sizeof(int) bytes), data
          in the persistent region is accessed more frequently*/
    
        data_persistent[tid % freqSize] = 2 * data_persistent[tid % freqSize];
        data_streaming[tid % dataSize] = 2 * data_streaming[tid % dataSize];
    }
    
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
    stream_attribute.accessPolicyWindow.num_bytes = freqSize * sizeof(int);   //Number of bytes for persisting accesses in range 10-60 MB
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;                      //Hint for cache hit ratio. Fixed value 1.0
    

The performance of the above kernel is shown in the chart below. When the persistent data region fits well into the 30 MB set-aside portion of the L2 cache, a performance increase of as much as 50% is observed. However, once the size of this persistent data region exceeds the size of the L2 set-aside cache portion, approximately 10% performance drop is observed due to thrashing of L2 cache lines.

[![The performance of the sliding-window benchmark with fixed hit-ratio of 1.0](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/l2-hitratio-before.png)](_images/l2-hitratio-before.png)

Figure 9 The performance of the sliding-window benchmark with fixed hit-ratio of 1.0

In order to optimize the performance, when the size of the persistent data is more than the size of the set-aside L2 cache portion, we tune the `num_bytes` and `hitRatio` parameters in the access window as below.
    
    
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(data_persistent);
    stream_attribute.accessPolicyWindow.num_bytes = 20*1024*1024;                                  //20 MB
    stream_attribute.accessPolicyWindow.hitRatio  = (20*1024*1024)/((float)freqSize*sizeof(int));  //Such that up to 20MB of data is resident.
    

We fix the `num_bytes` in the access window to 20 MB and tune the `hitRatio` such that a random 20 MB of the total persistent data is resident in the L2 set-aside cache portion. The remaining portion of this persistent data will be accessed using the streaming property. This helps in reducing cache thrashing. The results are shown in the chart below, where we see good performance regardless of whether the persistent data fits in the L2 set-aside or not.

[![The performance of the sliding-window benchmark with tuned hit-ratio](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/l2-hitratio-after.png)](_images/l2-hitratio-after.png)

Figure 10 The performance of the sliding-window benchmark with tuned hit-ratio

###  10.2.3. Shared Memory 

Because it is on-chip, shared memory has much higher bandwidth and lower latency than local and global memory - provided there are no bank conflicts between the threads, as detailed in the following section.

####  10.2.3.1. Shared Memory and Memory Banks 

To achieve high memory bandwidth for concurrent accesses, shared memory is divided into equally sized memory modules (_banks_) that can be accessed simultaneously. Therefore, any memory load or store of _n_ addresses that spans _n_ distinct memory banks can be serviced simultaneously, yielding an effective bandwidth that is _n_ times as high as the bandwidth of a single bank.

However, if multiple addresses of a memory request map to the same memory bank, the accesses are serialized. The hardware splits a memory request that has bank conflicts into as many separate conflict-free requests as necessary, decreasing the effective bandwidth by a factor equal to the number of separate memory requests. The one exception here is when multiple threads in a warp address the same shared memory location, resulting in a broadcast. In this case, multiple broadcasts from different banks are coalesced into a single multicast from the requested shared memory locations to the threads.

To minimize bank conflicts, it is important to understand how memory addresses map to memory banks and how to optimally schedule memory requests.

On devices of compute capability 5.x or newer, each bank has a bandwidth of 32 bits every clock cycle, and successive 32-bit words are assigned to successive banks. The warp size is 32 threads and the number of banks is also 32, so bank conflicts can occur between any threads in the warp. See [Compute Capability 5.x](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability-5-x) for further details.

####  10.2.3.2. Shared Memory in Matrix Multiplication (C=AB) 

Shared memory enables cooperation between threads in a block. When multiple threads in a block use the same data from global memory, shared memory can be used to access the data from global memory only once. Shared memory can also be used to avoid uncoalesced memory accesses by loading and storing data in a coalesced pattern from global memory and then reordering it in shared memory. Aside from memory bank conflicts, there is no penalty for non-sequential or unaligned accesses by a warp in shared memory.

The use of shared memory is illustrated via the simple example of a matrix multiplication C = AB for the case with A of dimension Mxw, B of dimension wxN, and C of dimension MxN. To keep the kernels simple, M and N are multiples of 32, since the warp size (w) is 32 for current devices.

A natural decomposition of the problem is to use a block and tile size of wxw threads. Therefore, in terms of wxw tiles, A is a column matrix, B is a row matrix, and C is their outer product; see [Figure 11](#shared-memory-in-matrix-multiplication-c-ab-block-column-matrix-a-multiplied-block-row-matrix-b-product-matrix-c). A grid of N/w by M/w blocks is launched, where each thread block calculates the elements of a different tile in C from a single tile of A and a single tile of B.

![Block-column matrix multiplied by block-row matrix. Block-column matrix \(A\) multiplied by block-row matrix \(B\) with resulting product matrix \(C\).](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/matrix-multiplication-block-column-by-block-row.png)

Figure 11 Block-column matrix multiplied by block-row matrix. Block-column matrix (A) multiplied by block-row matrix (B) with resulting product matrix (C).

To do this, the `simpleMultiply` kernel ([Unoptimized matrix multiplication](#unoptimized-matrix-multiplication-example)) calculates the output elements of a tile of matrix C.

Unoptimized matrix multiplication
    
    
    __global__ void simpleMultiply(float *a, float* b, float *c,
                                   int N)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        for (int i = 0; i < TILE_DIM; i++) {
            sum += a[row*TILE_DIM+i] * b[i*N+col];
        }
        c[row*N+col] = sum;
    }
    

In [Unoptimized matrix multiplication](#unoptimized-matrix-multiplication-example), `a`, `b`, and `c` are pointers to global memory for the matrices A, B, and C, respectively; `blockDim.x`, `blockDim.y`, and `TILE_DIM` are all equal to w. Each thread in the wxw-thread block calculates one element in a tile of C. `row` and `col` are the row and column of the element in C being calculated by a particular thread. The `for` loop over `i` multiplies a row of A by a column of B, which is then written to C.

The effective bandwidth of this kernel is 119.9 GB/s on an NVIDIA Tesla V100. To analyze performance, it is necessary to consider how warps access global memory in the `for` loop. Each warp of threads calculates one row of a tile of C, which depends on a single row of A and an entire tile of B as illustrated in [Figure 12](#shared-memory-in-matrix-multiplication-c-ab-computing-row-c-tile-c-row-a-tile-b).

![Computing a row of a tile. Computing a row of a tile in C using one row of A and an entire tile of B.](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/computing-row-of-tile.png)

Figure 12 Computing a row of a tile. Computing a row of a tile in C using one row of A and an entire tile of B.

For each iteration _i_ of the `for` loop, the threads in a warp read a row of the B tile, which is a sequential and coalesced access for all compute capabilities.

However, for each iteration _i_ , all threads in a warp read the same value from global memory for matrix A, as the index `row*TILE_DIM+i` is constant within a warp. Even though such an access requires only 1 transaction on devices of compute capability 2.0 or higher, there is wasted bandwidth in the transaction, because only one 4-byte word out of 8 words in a 32-byte cache segment is used. We can reuse this cache line in subsequent iterations of the loop, and we would eventually utilize all 8 words; however, when many warps execute on the same multiprocessor simultaneously, as is generally the case, the cache line may easily be evicted from the cache between iterations _i_ and _i+1_.

The performance on a device of any compute capability can be improved by reading a tile of A into shared memory as shown in [Using shared memory to improve the global memory load efficiency in matrix multiplication](#using-shared-memory-to-improve-the-global-memory-load-efficiency-in-matrix-multiplication).

Using shared memory to improve the global memory load efficiency in matrix multiplication
    
    
    __global__ void coalescedMultiply(float *a, float* b, float *c,
                                      int N)
    {
        __shared__ float aTile[TILE_DIM][TILE_DIM];
    
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
        __syncwarp();
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* b[i*N+col];
        }
        c[row*N+col] = sum;
    }
    

In [Using shared memory to improve the global memory load efficiency in matrix multiplication](#using-shared-memory-to-improve-the-global-memory-load-efficiency-in-matrix-multiplication), each element in a tile of A is read from global memory only once, in a fully coalesced fashion (with no wasted bandwidth), to shared memory. Within each iteration of the `for` loop, a value in shared memory is broadcast to all threads in a warp. Instead of a `__syncthreads()`synchronization barrier call, a `__syncwarp()` is sufficient after reading the tile of A into shared memory because only threads within the warp that write the data into shared memory read this data. This kernel has an effective bandwidth of 144.4 GB/s on an NVIDIA Tesla V100. This illustrates the use of the shared memory as a _user-managed cache_ when the hardware L1 cache eviction policy does not match up well with the needs of the application or when L1 cache is not used for reads from global memory.

A further improvement can be made to how [Using shared memory to improve the global memory load efficiency in matrix multiplication](#using-shared-memory-to-improve-the-global-memory-load-efficiency-in-matrix-multiplication) deals with matrix B. In calculating each of the rows of a tile of matrix C, the entire tile of B is read. The repeated reading of the B tile can be eliminated by reading it into shared memory once ([Improvement by reading additional data into shared memory](#improvement-by-reading-additional-data-into-shared-memory)).

Improvement by reading additional data into shared memory
    
    
    __global__ void sharedABMultiply(float *a, float* b, float *c,
                                     int N)
    {
        __shared__ float aTile[TILE_DIM][TILE_DIM],
                         bTile[TILE_DIM][TILE_DIM];
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
        bTile[threadIdx.y][threadIdx.x] = b[threadIdx.y*N+col];
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* bTile[i][threadIdx.x];
        }
        c[row*N+col] = sum;
    }
    

Note that in [Improvement by reading additional data into shared memory](#improvement-by-reading-additional-data-into-shared-memory), a `__syncthreads()` call is required after reading the B tile because a warp reads data from shared memory that were written to shared memory by different warps. The effective bandwidth of this routine is 195.5 GB/s on an NVIDIA Tesla V100. Note that the performance improvement is not due to improved coalescing in either case, but to avoiding redundant transfers from global memory.

The results of the various optimizations are summarized in [Table 2](#performance-improvements-optimizing-c-ab-matrix-table).

Table 2 Performance Improvements Optimizing C = AB Matrix Multiply Optimization | NVIDIA Tesla V100  
---|---  
No optimization | 119.9 GB/s  
Coalesced using shared memory to store a tile of A | 144.4 GB/s  
Using shared memory to eliminate redundant reads of a tile of B | 195.5 GB/s  
  
Note

**Medium Priority:** Use shared memory to avoid redundant transfers from global memory.

####  10.2.3.3. Shared Memory in Matrix Multiplication (C=AAT) 

A variant of the previous matrix multiplication can be used to illustrate how strided accesses to global memory, as well as shared memory bank conflicts, are handled. This variant simply uses the transpose of A in place of B, so C = AAT.

A simple implementation for C = AAT is shown in [Unoptimized handling of strided accesses to global memory](#unoptimized-handling-of-strided-accesses-to-global-memory).

Unoptimized handling of strided accesses to global memory
    
    
    __global__ void simpleMultiply(float *a, float *c, int M)
    {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        for (int i = 0; i < TILE_DIM; i++) {
            sum += a[row*TILE_DIM+i] * a[col*TILE_DIM+i];
        }
        c[row*M+col] = sum;
    }
    

In the example above, the _row_ -th, _col_ -th element of C is obtained by taking the dot product of the _row_ -th and _col_ -th rows of A. The effective bandwidth for this kernel is 12.8 GB/s on an NVIDIA Tesla V100. These results are substantially lower than the corresponding measurements for the C = AB kernel. The difference is in how threads in a half warp access elements of A in the second term, `a[col*TILE_DIM+i]`, for each iteration `i`. For a warp of threads, `col` represents sequential columns of the transpose of A, and therefore `col*TILE_DIM` represents a strided access of global memory with a stride of w, resulting in plenty of wasted bandwidth.

The way to avoid strided access is to use shared memory as before, except in this case a warp reads a row of A into a column of a shared memory tile, as shown in [An optimized handling of strided accesses using coalesced reads from global memory](#an-optimized-handling-of-strided-accesses-using-coalesced-reads-from-global-memory).

An optimized handling of strided accesses using coalesced reads from global memory
    
    
    __global__ void coalescedMultiply(float *a, float *c, int M)
    {
        __shared__ float aTile[TILE_DIM][TILE_DIM],
                         transposedTile[TILE_DIM][TILE_DIM];
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0f;
        aTile[threadIdx.y][threadIdx.x] = a[row*TILE_DIM+threadIdx.x];
        transposedTile[threadIdx.x][threadIdx.y] =
            a[(blockIdx.x*blockDim.x + threadIdx.y)*TILE_DIM +
            threadIdx.x];
        __syncthreads();
        for (int i = 0; i < TILE_DIM; i++) {
            sum += aTile[threadIdx.y][i]* transposedTile[i][threadIdx.x];
        }
        c[row*M+col] = sum;
    }
    

[An optimized handling of strided accesses using coalesced reads from global memory](#an-optimized-handling-of-strided-accesses-using-coalesced-reads-from-global-memory) uses the shared `transposedTile` to avoid uncoalesced accesses in the second term in the dot product and the shared `aTile` technique from the previous example to avoid uncoalesced accesses in the first term. The effective bandwidth of this kernel is 140.2 GB/s on an NVIDIA Tesla V100.These results are lower than those obtained by the final kernel for C = AB. The cause of the difference is shared memory bank conflicts.

The reads of elements in `transposedTile` within the for loop are free of conflicts, because threads of each half warp read across rows of the tile, resulting in unit stride across the banks. However, bank conflicts occur when copying the tile from global memory into shared memory. To enable the loads from global memory to be coalesced, data are read from global memory sequentially. However, this requires writing to shared memory in columns, and because of the use of wxw tiles in shared memory, this results in a stride between threads of w banks - every thread of the warp hits the same bank (Recall that w is selected as 32). These many-way bank conflicts are very expensive. The simple remedy is to pad the shared memory array so that it has an extra column, as in the following line of code.
    
    
    __shared__ float transposedTile[TILE_DIM][TILE_DIM+1];
    

This padding eliminates the conflicts entirely, because now the stride between threads is w+1 banks (i.e., 33 for current devices), which, due to modulo arithmetic used to compute bank indices, is equivalent to a unit stride. After this change, the effective bandwidth is 199.4 GB/s on an NVIDIA Tesla V100, which is comparable to the results from the last C = AB kernel.

The results of these optimizations are summarized in [Table 3](#performance-inmprovements-optimizing-c-aa-matrix-multiplication-table).

Table 3 Performance Improvements Optimizing C = AAT Matrix Multiplication Optimization | NVIDIA Tesla V100  
---|---  
No optimization | 12.8 GB/s  
Using shared memory to coalesce global reads | 140.2 GB/s  
Removing bank conflicts | 199.4 GB/s  
  
These results should be compared with those in [Table 2](#performance-improvements-optimizing-c-ab-matrix-table). As can be seen from these tables, judicious use of shared memory can dramatically improve performance.

The examples in this section have illustrated three reasons to use shared memory:

  * To enable coalesced accesses to global memory, especially to avoid large strides (for general matrices, strides are much larger than 32)

  * To eliminate (or reduce) redundant loads from global memory

  * To avoid wasted bandwidth


####  10.2.3.4. Asynchronous Copy from Global Memory to Shared Memory 

CUDA 11.0 introduces an _async-copy_ feature that can be used within device code to explicitly manage the asynchronous copying of data from global memory to shared memory. This feature enables CUDA kernels to overlap copying data from global to shared memory with computation. It also avoids an intermediary register file access traditionally present between the global memory read and the shared memory write.

For more details refer to the `memcpy_async` section in the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async_data_operations).

To understand the performance difference between synchronous copy and asynchronous copy of data from global memory to shared memory, consider the following micro benchmark CUDA kernels for demonstrating the synchronous and asynchronous approaches. Asynchronous copies are hardware accelerated for NVIDIA A100 GPU.
    
    
    template <typename T>
    __global__ void pipeline_kernel_sync(T *global, uint64_t *clock, size_t copy_count) {
      extern __shared__ char s[];
      T *shared = reinterpret_cast<T *>(s);
    
      uint64_t clock_start = clock64();
    
      for (size_t i = 0; i < copy_count; ++i) {
        shared[blockDim.x * i + threadIdx.x] = global[blockDim.x * i + threadIdx.x];
      }
    
      uint64_t clock_end = clock64();
    
      atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                clock_end - clock_start);
    }
    
    template <typename T>
    __global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count) {
      extern __shared__ char s[];
      T *shared = reinterpret_cast<T *>(s);
    
      uint64_t clock_start = clock64();
    
      //pipeline pipe;
      for (size_t i = 0; i < copy_count; ++i) {
        __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                                &global[blockDim.x * i + threadIdx.x], sizeof(T));
      }
      __pipeline_commit();
      __pipeline_wait_prior(0);
    
      uint64_t clock_end = clock64();
    
      atomicAdd(reinterpret_cast<unsigned long long *>(clock),
                clock_end - clock_start);
    }
    

The synchronous version for the kernel loads an element from global memory to an intermediate register and then stores the intermediate register value to shared memory. In the asynchronous version of the kernel, instructions to load from global memory and store directly into shared memory are issued as soon as `__pipeline_memcpy_async()` function is called. The `__pipeline_wait_prior(0)` will wait until all the instructions in the pipe object have been executed. Using asynchronous copies does not use any intermediate register. Not using intermediate registers can help reduce register pressure and can increase kernel occupancy. Data copied from global memory to shared memory using asynchronous copy instructions can be cached in the L1 cache or the L1 cache can be optionally bypassed. If individual CUDA threads are copying elements of 16 bytes, the L1 cache can be bypassed. This difference is illustrated in [Figure 13](#async-copy-sync-vs-async-figure).

[![Comparing Synchronous vs Asynchronous Copy from Global Memory to Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/sync-vs-async.png)](_images/sync-vs-async.png)

Figure 13 Comparing Synchronous vs Asynchronous Copy from Global Memory to Shared Memory

We evaluate the performance of both kernels using elements of size 4B, 8B and 16B per thread i.e., using `int`, `int2` and `int4` for the template parameter. We adjust the `copy_count` in the kernels such that each thread block copies from 512 bytes up to 48 MB. The performance of the kernels is shown in [Figure 14](#async-copy-async-perf-figure).

[![Comparing Performance of Synchronous vs Asynchronous Copy from Global Memory to Shared Memory](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/_images/async-perf.png)](_images/async-perf.png)

Figure 14 Comparing Performance of Synchronous vs Asynchronous Copy from Global Memory to Shared Memory

From the performance chart, the following observations can be made for this experiment.

  * Best performance with synchronous copy is achieved when the `copy_count` parameter is a multiple of 4 for all three element sizes. The compiler can optimize groups of 4 load and store instructions. This is evident from the saw tooth curves.

  * Asynchronous copy achieves better performance in nearly all cases.

  * The async-copy does not require the `copy_count` parameter to be a multiple of 4, to maximize performance through compiler optimizations.

  * Overall, best performance is achieved when using asynchronous copies with an element of size 8 or 16 bytes.


###  10.2.4. Local Memory 

Local memory is so named because its scope is local to the thread, not because of its physical location. In fact, local memory is off-chip. Hence, access to local memory is as expensive as access to global memory. In other words, the term _local_ in the name does not imply faster access.

Local memory is used only to hold automatic variables. This is done by the `nvcc` compiler when it determines that there is insufficient register space to hold the variable. Automatic variables that are likely to be placed in local memory are large structures or arrays that would consume too much register space and arrays that the compiler determines may be indexed dynamically.

Inspection of the PTX assembly code (obtained by compiling with `-ptx` or `-keep` command-line options to `nvcc`) reveals whether a variable has been placed in local memory during the first compilation phases. If it has, it will be declared using the `.local` mnemonic and accessed using the `ld.local` and `st.local` mnemonics. If it has not, subsequent compilation phases might still decide otherwise, if they find the variable consumes too much register space for the targeted architecture. There is no way to check this for a specific variable, but the compiler reports total local memory usage per kernel (lmem) when run with the`--ptxas-options=-v` option.

###  10.2.5. Texture Memory 

The read-only texture memory space is cached. Therefore, a texture fetch costs one device memory read only on a cache miss; otherwise, it just costs one read from the texture cache. The texture cache is optimized for 2D spatial locality, so threads of the same warp that read texture addresses that are close together will achieve best performance. Texture memory is also designed for streaming fetches with a constant latency; that is, a cache hit reduces DRAM bandwidth demand, but not fetch latency.

In certain addressing situations, reading device memory through texture fetching can be an advantageous alternative to reading device memory from global or constant memory.

####  10.2.5.1. Additional Texture Capabilities 

If textures are fetched using `tex1D()`,`tex2D()`, or `tex3D()` rather than `tex1Dfetch()`, the hardware provides other capabilities that might be useful for some applications such as image processing, as shown in [Table 4](#useful-features-for-tex1d-tex2d-tex3d-fetches-table).

Table 4 Useful Features for tex1D(), tex2D(), and tex3D() Fetches Feature | Use | Caveat  
---|---|---  
Filtering | Fast, low-precision interpolation between texels | Valid only if the texture reference returns floating-point data  
Normalized texture coordinates | Resolution-independent coding | None  
Addressing modes | Automatic handling of boundary cases1 | Can be used only with normalized texture coordinates  
1 The automatic handling of boundary cases in the bottom row of Table 4 refers to how a texture coordinate is resolved when it falls outside the valid addressing range. There are two options: _clamp_ and _wrap_. If _x_ is the coordinate and _N_ is the number of texels for a one-dimensional texture, then with clamp, _x_ is replaced by _0_ if _x_ < 0 and by 1-1/_N_ if 1 _<__x_. With wrap, _x_ is replaced by _frac(x)_ where _frac(x) = x - floor(x)_. Floor returns the largest integer less than or equal to _x_. So, in clamp mode where _N_ = 1, an _x_ of 1.3 is clamped to 1.0; whereas in wrap mode, it is converted to 0.3 |  |   
  
Within a kernel call, the texture cache is not kept coherent with respect to global memory writes, so texture fetches from addresses that have been written via global stores in the same kernel call return undefined data. That is, a thread can safely read a memory location via texture if the location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread within the same kernel call.

###  10.2.6. Constant Memory 

There is a total of 64 KB constant memory on a device. The constant memory space is cached. As a result, a read from constant memory costs one memory read from device memory only on a cache miss; otherwise, it just costs one read from the constant cache. Accesses to different addresses by threads within a warp are serialized, thus the cost scales linearly with the number of unique addresses read by all threads within a warp. As such, the constant cache is best when threads in the same warp accesses only a few distinct locations. If all threads of a warp access the same location, then constant memory can be as fast as a register access.

###  10.2.7. Registers 

Generally, accessing a register consumes zero extra clock cycles per instruction, but delays may occur due to register read-after-write dependencies and register memory bank conflicts.

The compiler and hardware thread scheduler will schedule instructions as optimally as possible to avoid register memory bank conflicts. An application has no direct control over these bank conflicts. In particular, there is no register-related reason to pack data into vector data types such as `float4` or `int4` types.

####  10.2.7.1. Register Pressure 

Register pressure occurs when there are not enough registers available for a given task. Even though each multiprocessor contains thousands of 32-bit registers (see Features and Technical Specifications of the CUDA C++ Programming Guide), these are partitioned among concurrent threads. To prevent the compiler from allocating too many registers, use the `-maxrregcount=N` compiler command-line option or the launch bounds kernel definition qualifier (see Execution Configuration of the CUDA C++ Programming Guide) to control the maximum number of registers to allocated per thread.


##  10.3. Allocation 

Device memory allocation and de-allocation via `cudaMalloc()` and `cudaFree()` are expensive operations. It is recommended to use `cudaMallocAsync()` and `cudaFreeAsync()` which are stream ordered pool allocators to manage device memory.


##  10.4. NUMA Best Practices 

Some recent Linux distributions enable automatic NUMA balancing (or “[AutoNUMA](https://lwn.net/Articles/488709/)”) by default. In some instances, operations performed by automatic NUMA balancing may degrade the performance of applications running on NVIDIA GPUs. For optimal performance, users should manually tune the NUMA characteristics of their application.

The optimal NUMA tuning will depend on the characteristics and desired hardware affinities of each application and node, but in general applications computing on NVIDIA GPUs are advised to choose a policy that disables automatic NUMA balancing. For example, on IBM Newell POWER9 nodes (where the CPUs correspond to NUMA nodes 0 and 8), use:
    
    
    numactl --membind=0,8
    

to bind memory allocations to the CPUs.
