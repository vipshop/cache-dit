# 3. Heterogeneous Computing


CUDA programming involves running code on two different platforms concurrently: a _host_ system with one or more CPUs and one or more CUDA-enabled NVIDIA GPU _devices_.


While NVIDIA GPUs are frequently associated with graphics, they are also powerful arithmetic engines capable of running thousands of lightweight threads in parallel. This capability makes them well suited to computations that can leverage parallel execution.


However, the device is based on a distinctly different design from the host system, and it’s important to understand those differences and how they determine the performance of CUDA applications in order to use CUDA effectively.


##  3.1. Differences between Host and Device 

The primary differences are in threading model and in separate physical memories:

Threading resources
    

Execution pipelines on host systems can support a limited number of concurrent threads. For example, servers that have two 32 core processors can run only 64 threads concurrently (or small multiple of that if the CPUs support simultaneous multithreading). By comparison, the _smallest_ executable unit of parallelism on a CUDA device comprises 32 threads (termed a _warp_ of threads). Modern NVIDIA GPUs can support up to 2048 active threads concurrently per multiprocessor (see Features and Specifications of the CUDA C++ Programming Guide) On GPUs with 80 multiprocessors, this leads to more than 160,000 concurrently active threads.

Threads
    

Threads on a CPU are generally heavyweight entities. The operating system must swap threads on and off CPU execution channels to provide multithreading capability. Context switches (when two threads are swapped) are therefore slow and expensive. By comparison, threads on GPUs are extremely lightweight. In a typical system, thousands of threads are queued up for work (in warps of 32 threads each). If the GPU must wait on one warp of threads, it simply begins executing work on another. Because separate registers are allocated to all active threads, no swapping of registers or other state need occur when switching among GPU threads. Resources stay allocated to each thread until it completes its execution. In short, CPU cores are designed to _minimize latency_ for a small number of threads at a time each, whereas GPUs are designed to handle a large number of concurrent, lightweight threads in order to _maximize throughput_.

RAM
    

The host system and the device each have their own distinct attached physical memories [1](#fn1). As the host and device memories are separated, items in the host memory must occasionally be communicated between device memory and host memory as described in [What Runs on a CUDA-Enabled Device?](#what-runs-on-cuda-enabled-device).

These are the primary hardware differences between CPU hosts and GPU devices with respect to parallel programming. Other differences are discussed as they arise elsewhere in this document. Applications composed with these differences in mind can treat the host and device together as a cohesive heterogeneous system wherein each processing unit is leveraged to do the kind of work it does best: sequential work on the host and parallel work on the device.


##  3.2. What Runs on a CUDA-Enabled Device? 

The following issues should be considered when determining what parts of an application to run on the device:

  * The device is ideally suited for computations that can be run on numerous data elements simultaneously in parallel. This typically involves arithmetic on large data sets (such as matrices) where the same operation can be performed across thousands, if not millions, of elements at the same time. This is a requirement for good performance on CUDA: the software must use a large number (generally thousands or tens of thousands) of concurrent threads. The support for running numerous threads in parallel derives from CUDA’s use of a lightweight threading model described above.

  * To use CUDA, data values must be transferred from the host to the device. These transfers are costly in terms of performance and should be minimized. (See [Data Transfer Between Host and Device](#data-transfer-between-host-and-device).) This cost has several ramifications:

    * The complexity of operations should justify the cost of moving data to and from the device. Code that transfers data for brief use by a small number of threads will see little or no performance benefit. The ideal scenario is one in which many threads perform a substantial amount of work.

For example, transferring two matrices to the device to perform a matrix addition and then transferring the results back to the host will not realize much performance benefit. The issue here is the number of operations performed per data element transferred. For the preceding procedure, assuming matrices of size NxN, there are N2 operations (additions) and 3N2 elements transferred, so the ratio of operations to elements transferred is 1:3 or O(1). Performance benefits can be more readily achieved when this ratio is higher. For example, a matrix multiplication of the same matrices requires N3 operations (multiply-add), so the ratio of operations to elements transferred is O(N), in which case the larger the matrix the greater the performance benefit. The types of operations are an additional factor, as additions have different complexity profiles than, for example, trigonometric functions. It is important to include the overhead of transferring data to and from the device in determining whether operations should be performed on the host or on the device.

    * Data should be kept on the device as long as possible. Because transfers should be minimized, programs that run multiple kernels on the same data should favor leaving the data on the device between kernel calls, rather than transferring intermediate results to the host and then sending them back to the device for subsequent calculations. So, in the previous example, had the two matrices to be added already been on the device as a result of some previous calculation, or if the results of the addition would be used in some subsequent calculation, the matrix addition should be performed locally on the device. This approach should be used even if one of the steps in a sequence of calculations could be performed faster on the host. Even a relatively slow kernel may be advantageous if it avoids one or more transfers between host and device memory. [Data Transfer Between Host and Device](#data-transfer-between-host-and-device) provides further details, including the measurements of bandwidth between the host and the device versus within the device proper.

  * For best performance, there should be some coherence in memory access by adjacent threads running on the device. Certain memory access patterns enable the hardware to coalesce groups of reads or writes of multiple data items into one operation. Data that cannot be laid out so as to enable _coalescing_ , or that doesn’t have enough locality to use the L1 or texture caches effectively, will tend to see lesser speedups when used in computations on GPUs. A noteworthy exception to this are completely random memory access patterns. In general, they should be avoided, because compared to peak capabilities any architecture processes these memory access patterns at a low efficiency. However, compared to cache based architectures, like CPUs, latency hiding architectures, like GPUs, tend to cope better with completely random memory access patterns.


[1](#id3)
    

On Systems on a Chip with integrated GPUs, such as NVIDIA® Tegra®, host and device memory are physically the same, but there is still a logical distinction between host and device memory. See the [Application Note on CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote) for details.
