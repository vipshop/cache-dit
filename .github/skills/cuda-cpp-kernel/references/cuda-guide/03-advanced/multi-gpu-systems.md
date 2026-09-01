---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html
---

# 3.4. Programming Systems with Multiple GPUs

Multi-GPU programming allows an application to address problem sizes and achieve performance levels beyond what is possible with a single GPU by exploiting the larger aggregate arithmetic performance, memory capacity, and memory bandwidth provided by multi-GPU systems.

CUDA enables multi-GPU programming through host APIs, driver infrastructure, and supporting GPU hardware technologies:

  * Host thread CUDA context management

  * Unified memory addressing for all processors in the system

  * Peer-to-peer bulk memory transfers between GPUs

  * Fine-grained peer-to-peer GPU load/store memory access

  * Higher level abstractions and supporting system software such as CUDA interprocess communication, parallel reductions using [NCCL](https://developer.nvidia.com/nccl), and communication using NVLink and/or GPU-Direct RDMA with APIs such as [NVSHMEM](https://developer.nvidia.com/nvshmem) and MPI


At the most basic level, multi-GPU programming requires the application to manage multiple active CUDA contexts concurrently, distribute data to the GPUs, launch kernels on the GPUs to complete their work, and to communicate or collect the results so that they can be acted upon by the application. The details of how this is done differ depending on the most effective mapping of an application’s algorithms, available parallelism, and existing code structure to a suitable multi-GPU programming approach. Some of the most common multi-GPU programming approaches include:

  * A single host thread driving multiple GPUs

  * Multiple host threads, each driving their own GPU

  * Multiple single-threaded host processes, each driving their own GPU

  * Multiple host processes containing multiple threads, each driving their own GPU

  * Multi-node NVLink-connected clusters, with GPUs driven by threads and processes running within multiple operating system instances across the cluster nodes


GPUs can communicate with each other through memory transfers and peer accesses between device memories, covering each of the multi-device work distribution approaches listed above. High performance, low-latency GPU communications are supported by querying for and enabling the use of peer-to-peer GPU memory access, and leveraging NVLink to achieve high bandwidth transfers and finer-grained load/store operations between devices.

CUDA unified virtual addressing permits communication between multiple GPUs within the same host process with minimal additional steps to query and enable the use of high performance peer-to-peer memory access and transfers, e.g., via NVLink.

Communication between multiple GPUs managed by different host processes is supported through the use of interprocess communication (IPC) and Virtual memory Management (VMM) APIs. An introduction to high level IPC concepts and intra-node CUDA IPC APIs are discussed in the [Interprocess Communication](../04-special-topics/inter-process-communication.html#interprocess-communication) section. AdvancedVirtual Memory Management (VMM) APIs support both intra-node and multi-node IPC, are usable on both Linux and Windows operating systems, and allow per-allocation granularity control over IPC sharing of memory buffers as described in [Virtual Memory Management](../04-special-topics/virtual-memory-management.html#virtual-memory-management).

CUDA itself provides the APIs needed to implement collective operations within a group of GPUs, potentially including the host, but it does not provide high level multi-GPU collective APIs itself. Multi-GPU collectives are provided by higher abstraction CUDA communication libraries such as [NCCL](https://developer.nvidia.com/nccl) and [NVSHMEM](https://developer.nvidia.com/nvshmem).

## 3.4.1. Multi-Device Context and Execution Management

The first steps that are required to for an application to use multiple GPUs are to enumerate the available GPU devices, select among the available devices as appropriate based on their hardware properties, CPU affinity, and connectivity to peers, and to create CUDA contexts for each device that the application will use.

### 3.4.1.1. Device Enumeration

The following code sample shows how to query number of CUDA-enabled devices, enumerate each of the devices, and query their properties.
    
    
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n",
               device, deviceProp.major, deviceProp.minor);
    }
    

### 3.4.1.2. Device Selection

A host thread can set the device it is currently operating on at any time by calling `cudaSetDevice()`. Device memory allocations and kernel launches are made on the current device; streams and events are created in association with the currently set device. Until a call to `cudaSetDevice()` is made by the host thread, the current device defaults to device 0.

The following code sample illustrates how setting the current device affects subsequent memory allocation and kernel execution operations.
    
    
    size_t size = 1024 * sizeof(float);
    cudaSetDevice(0);            // Set device 0 as current
    float* p0;
    cudaMalloc(&p0, size);       // Allocate memory on device 0
    MyKernel<<<1000, 128>>>(p0); // Launch kernel on device 0
    
    cudaSetDevice(1);            // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);       // Allocate memory on device 1
    MyKernel<<<1000, 128>>>(p1); // Launch kernel on device 1
    

### 3.4.1.3. Multi-Device Stream, Event, and Memory Copy Behavior

A kernel launch will fail if it is issued to a stream that is not associated to the current device as illustrated in the following code sample.
    
    
    cudaSetDevice(0);               // Set device 0 as current
    cudaStream_t s0;
    cudaStreamCreate(&s0);          // Create stream s0 on device 0
    MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 0 in s0
    
    cudaSetDevice(1);               // Set device 1 as current
    cudaStream_t s1;
    cudaStreamCreate(&s1);          // Create stream s1 on device 1
    MyKernel<<<100, 64, 0, s1>>>(); // Launch kernel on device 1 in s1
    
    // This kernel launch will fail, since stream s0 is not associated to device 1:
    MyKernel<<<100, 64, 0, s0>>>(); // Launch kernel on device 1 in s0
    

A memory copy will succeed even if it is issued to a stream that is not associated to the current device.

`cudaEventRecord()` will fail if the input event and input stream are associated to different devices.

`cudaEventElapsedTime()` will fail if the two input events are associated to different devices.

`cudaEventSynchronize()` and `cudaEventQuery()` will succeed even if the input event is associated to a device that is different from the current device.

`cudaStreamWaitEvent()` will succeed even if the input stream and input event are associated to different devices. `cudaStreamWaitEvent()` can therefore be used to synchronize multiple devices with each other.

Each device has its own [default stream](../02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream), so commands issued to the default stream of a device may execute out of order or concurrently with respect to commands issued to the default stream of any other device.

## 3.4.2. Multi-Device Peer-to-Peer Transfers and Memory Access

### 3.4.2.1. Peer-to-Peer Memory Transfers

CUDA can perform memory transfers between devices and will take advantage of dedicated copy engines and NVLink hardware to maximize performance when peer-to-peer memory access is possible.

`cudaMemcpy` can be used with the copy type `cudaMemcpyDeviceToDevice` or `cudaMemcpyDefault`.

Otherwise, copies must be performed using `cudaMemcpyPeer()`, `cudaMemcpyPeerAsync()`, `cudaMemcpy3DPeer()`, or `cudaMemcpy3DPeerAsync()` as illustrated in the following code sample.
    
    
    cudaSetDevice(0);                   // Set device 0 as current
    float* p0;
    size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);              // Allocate memory on device 0
    
    cudaSetDevice(1);                   // Set device 1 as current
    float* p1;
    cudaMalloc(&p1, size);              // Allocate memory on device 1
    
    cudaSetDevice(0);                   // Set device 0 as current
    MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
    
    cudaSetDevice(1);                   // Set device 1 as current
    cudaMemcpyPeer(p1, 1, p0, 0, size); // Copy p0 to p1
    MyKernel<<<1000, 128>>>(p1);        // Launch kernel on device 1
    

A copy (in the implicit _NULL_ stream) between the memories of two different devices:

  * does not start until all commands previously issued to either device have completed and

  * runs to completion before any commands (see [Asynchronous Execution](../02-basics/asynchronous-execution.html#asynchronous-execution)) issued after the copy to either device can start.


Consistent with the normal behavior of streams, an asynchronous copy between the memories of two devices may overlap with copies or kernels in another stream.

If peer-to-peer access is enabled between two devices, e.g., as described in [Peer-to-Peer Memory Access](#multi-gpu-peer-to-peer-memory-access), peer-to-peer memory copies between these two devices no longer need to be staged through the host and are therefore faster.

### 3.4.2.2. Peer-to-Peer Memory Access

Depending on the system properties, specifically the PCIe and/or NVLink topology, devices are able to address each other’s memory (i.e., a kernel executing on one device can dereference a pointer to the memory of the other device). Peer-to-peer memory access is supported between two devices if `cudaDeviceCanAccessPeer()` returns true for the specified devices.

Peer-to-peer memory access must be enabled between two devices by calling `cudaDeviceEnablePeerAccess()` as illustrated in the following code sample. On non-NVSwitch enabled systems, each device can support a system-wide maximum of eight peer connections.

A unified virtual address space is used for both devices (see [Unified Virtual Address Space](../02-basics/understanding-memory.html#memory-unified-virtual-address-space)), so the same pointer can be used to address memory from both devices as shown in the code sample below.
    
    
    cudaSetDevice(0);                   // Set device 0 as current
    float* p0;
    size_t size = 1024 * sizeof(float);
    cudaMalloc(&p0, size);              // Allocate memory on device 0
    MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
    
    cudaSetDevice(1);                   // Set device 1 as current
    cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                        // with device 0
    
    // Launch kernel on device 1
    // This kernel launch can access memory on device 0 at address p0
    MyKernel<<<1000, 128>>>(p0);
    

Note

The use of `cudaDeviceEnablePeerAccess()` to enable peer memory access operates globally on all previous and subsequent GPU memory allocations on the peer device. Enabling peer access to a device via `cudaDeviceEnablePeerAccess()` adds runtime cost to device memory allocation operations on that peer due to the need make the allocations immediately accessible to the current device and any other peers that also have access, adding multiplicative overhead that scales with the number of peer devices.

A more scalable alternative to enabling peer memory access for all device memory allocations is to make use of CUDA Virtual Memory Management APIs to explicitly allocate peer-accessible memory regions only as-needed, at allocation time. By requesting peer-accessibility explicitly during memory allocation, the runtime cost of memory allocations are unharmed for allocations not accessible to peers, and peer-accessible data structures are correctly scoped for improved software debugging and reliability (see ref::virtual-memory-management).

### 3.4.2.3. Peer-to-Peer Memory Consistency

Synchronization operations must be used to enforce the ordering and correctness of memory accesses by concurrently executing threads in grids distributed across multiple devices. Threads synchronizing across devices operate at the `thread_scope_system` [synchronization scope](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#thread-scopes). Similarly, memory operations fall within the `thread_scope_system` [memory synchronization domain](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#memory-synchronization-domains).

CUDA ref::atomic-functions can perform read-modify-write operations on an object in peer device memory when only a single GPU is accessing that object. The requirements and limitations for peer atomicity are described in the CUDA memory model [atomicity requirements](https://nvidia.github.io/cccl/libcudacxx/extended_api/memory_model.html#atomicity) discussion.

### 3.4.2.4. Multi-Device Managed Memory

Managed memory can be used on multi-GPU systems with peer-to-peer support. The detailed requirements for concurrent multi-device managed memory access and APIs for GPU-exclusive access to managed memory are described in [Multi-GPU](../04-special-topics/unified-memory.html#um-legacy-multi-gpu).

### 3.4.2.5. Host IOMMU Hardware, PCI Access Control Services, and VMs

On Linux specifically, CUDA and the display driver do not support IOMMU-enabled bare-metal PCIe peer-to-peer memory transfer. However, CUDA and the display driver do support IOMMU via virtual machine pass through. The IOMMU must be disabled when running Linux on a bare metal system to prevent silent device memory corruption. Conversely, the IOMMU should be enabled and the VFIO driver be used for PCIe pass through for virtual machines.

On Windows the IOMMU limitation above does not exist.

See also [Allocating DMA Buffers on 64-bit Platforms](https://download.nvidia.com/XFree86/Linux-x86_64/510.85.02/README/dma_issues.html).

Additionally, PCI Access Control Services (ACS) can be enabled on systems that support IOMMU. The PCI ACS feature redirects all PCI point-to-point traffic through the CPU root complex, which can cause significant performance loss due to the reduction in overall bisection bandwidth.
