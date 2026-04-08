# 2. Programming Model


##  2.1. [A Highly Multithreaded Coprocessor](#highly-multithreaded-coprocessor)

The GPU is a compute device capable of executing a very large number of threads in parallel. It operates as a coprocessor to the main CPU, or host: In other words, data-parallel, compute-intensive portions of applications running on the host are off-loaded onto the device.

More precisely, a portion of an application that is executed many times, but independently on different data, can be isolated into a kernel function that is executed on the GPU as many different threads. To that effect, such a function is compiled to the PTX instruction set and the resulting kernel is translated at install time to the target GPU instruction set.


##  2.2. [Thread Hierarchy](#thread-hierarchy)

The batch of threads that executes a kernel is organized as a grid. A grid consists of either cooperative thread arrays or clusters of cooperative thread arrays as described in this section and illustrated in [Figure 1](#grid-of-clusters-grid-with-ctas) and [Figure 2](#grid-of-clusters-grid-with-clusters). _Cooperative thread arrays (CTAs)_ implement CUDA thread blocks and clusters implement CUDA thread block clusters.

###  2.2.1. [Cooperative Thread Arrays](#cooperative-thread-arrays)

The _Parallel Thread Execution (PTX)_ programming model is explicitly parallel: a PTX program specifies the execution of a given thread of a parallel thread array. A _cooperative thread array_ , or CTA, is an array of threads that execute a kernel concurrently or in parallel.

Threads within a CTA can communicate with each other. To coordinate the communication of the threads within the CTA, one can specify synchronization points where threads wait until all threads in the CTA have arrived.

Each thread has a unique thread identifier within the CTA. Programs use a data parallel decomposition to partition inputs, work, and results across the threads of the CTA. Each CTA thread uses its thread identifier to determine its assigned role, assign specific input and output positions, compute addresses, and select work to perform. The thread identifier is a three-element vector `tid`, (with elements `tid.x`, `tid.y`, and `tid.z`) that specifies the thread’s position within a 1D, 2D, or 3D CTA. Each thread identifier component ranges from zero up to the number of thread ids in that CTA dimension.

Each CTA has a 1D, 2D, or 3D shape specified by a three-element vector `ntid` (with elements `ntid.x`, `ntid.y`, and `ntid.z`). The vector `ntid` specifies the number of threads in each CTA dimension.

Threads within a CTA execute in SIMT (single-instruction, multiple-thread) fashion in groups called _warps_. A _warp_ is a maximal subset of threads from a single CTA, such that the threads execute the same instructions at the same time. Threads within a warp are sequentially numbered. The warp size is a machine-dependent constant. Typically, a warp has 32 threads. Some applications may be able to maximize performance with knowledge of the warp size, so PTX includes a run-time immediate constant, `WARP_SZ`, which may be used in any instruction where an immediate operand is allowed.

###  2.2.2. [Cluster of Cooperative Thread Arrays](#cluster-of-cooperative-thread-arrays)

Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. The executing CTA has to make sure that the shared memory of the peer CTA exists before communicating with it via shared memory and the peer CTA hasn’t exited before completing the shared memory operation.

Threads within the different CTAs in a cluster can synchronize and communicate with each other via shared memory. Cluster-wide barriers can be used to synchronize all the threads within the cluster. Each CTA in a cluster has a unique CTA identifier within its cluster (_cluster_ctaid_). Each cluster of CTAs has 1D, 2D or 3D shape specified by the parameter _cluster_nctaid_. Each CTA in the cluster also has a unique CTA identifier (_cluster_ctarank_) across all dimensions. The total number of CTAs across all the dimensions in the cluster is specified by _cluster_nctarank_. Threads may read and use these values through predefined, read-only special registers `%cluster_ctaid`, `%cluster_nctaid`, `%cluster_ctarank`, `%cluster_nctarank`.

Cluster level is applicable only on target architecture `sm_90` or higher. Specifying cluster level during launch time is optional. If the user specifies the cluster dimensions at launch time then it will be treated as explicit cluster launch, otherwise it will be treated as implicit cluster launch with default dimension 1x1x1. PTX provides read-only special register `%is_explicit_cluster` to differentiate between explicit and implicit cluster launch.

###  2.2.3. [Grid of Clusters](#grid-of-clusters)

There is a maximum number of threads that a CTA can contain and a maximum number of CTAs that a cluster can contain. However, clusters with CTAs that execute the same kernel can be batched together into a grid of clusters, so that the total number of threads that can be launched in a single kernel invocation is very large. This comes at the expense of reduced thread communication and synchronization, because threads in different clusters cannot communicate and synchronize with each other.

Each cluster has a unique cluster identifier (_clusterid_) within a grid of clusters. Each grid of clusters has a 1D, 2D , or 3D shape specified by the parameter _nclusterid_. Each grid also has a unique temporal grid identifier (_gridid_). Threads may read and use these values through predefined, read-only special registers `%tid`, `%ntid`, `%clusterid`, `%nclusterid`, and `%gridid`.

Each CTA has a unique identifier (_ctaid_) within a grid. Each grid of CTAs has 1D, 2D, or 3D shape specified by the parameter _nctaid_. Thread may use and read these values through predefined, read-only special registers `%ctaid` and `%nctaid`.

Each kernel is executed as a batch of threads organized as a grid of clusters consisting of CTAs where cluster is optional level and is applicable only for target architectures `sm_90` and higher. [Figure 1](#grid-of-clusters-grid-with-ctas) shows a grid consisting of CTAs and [Figure 2](#grid-of-clusters-grid-with-clusters) shows a grid consisting of clusters.

Grids may be launched with dependencies between one another - a grid may be a dependent grid and/or a prerequisite grid. To understand how grid dependencies may be defined, refer to the section on _CUDA Graphs_ in the _Cuda Programming Guide_.

![Grid with CTAs](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-CTAs.png)

Figure 1 Grid with CTAs

![Grid with clusters](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/grid-with-clusters.png)

Figure 2 Grid with clusters

A cluster is a set of cooperative thread arrays (CTAs) where a CTA is a set of concurrent threads that execute the same kernel program. A grid is a set of clusters consisting of CTAs that execute independently.


##  2.3. [Memory Hierarchy](#memory-hierarchy)

PTX threads may access data from multiple state spaces during their execution as illustrated by [Figure 3](#memory-hierarchy-memory-hierarchy-with-clusters) where cluster level is introduced from target architecture `sm_90` onwards. Each thread has a private local memory. Each thread block (CTA) has a shared memory visible to all threads of the block and to all active blocks in the cluster and with the same lifetime as the block. Finally, all threads have access to the same global memory.

There are additional state spaces accessible by all threads: the constant, param, texture, and surface state spaces. Constant and texture memory are read-only; surface memory is readable and writable. The global, constant, param, texture, and surface state spaces are optimized for different memory usages. For example, texture memory offers different addressing modes as well as data filtering for specific data formats. Note that texture and surface memory is cached, and within the same kernel call, the cache is not kept coherent with respect to global memory writes and surface memory writes, so any texture fetch or surface read to an address that has been written to via a global or a surface write in the same kernel call returns undefined data. In other words, a thread can safely read some texture or surface memory location only if this memory location has been updated by a previous kernel call or memory copy, but not if it has been previously updated by the same thread or another thread from the same kernel call.

The global, constant, and texture state spaces are persistent across kernel launches by the same application.

Both the host and the device maintain their own local memory, referred to as _host memory_ and _device memory_ , respectively. The device memory may be mapped and read or written by the host, or, for more efficient transfer, copied from the host memory through optimized API calls that utilize the device’s high-performance _Direct Memory Access (DMA)_ engine.

![Memory Hierarchy](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/memory-hierarchy-with-clusters.png)

Figure 3 Memory Hierarchy
