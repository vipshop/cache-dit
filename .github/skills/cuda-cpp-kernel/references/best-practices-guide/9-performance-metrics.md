# 9. Performance Metrics


When attempting to optimize CUDA code, it pays to know how to measure performance accurately and to understand the role that bandwidth plays in performance measurement. This chapter discusses how to correctly measure performance using CPU timers and CUDA events. It then explores how bandwidth affects performance metrics and how to mitigate some of the challenges it poses.


##  9.1. Timing 

CUDA calls and kernel executions can be timed using either CPU or GPU timers. This section examines the functionality, advantages, and pitfalls of both approaches.

###  9.1.1. Using CPU Timers 

Any CPU timer can be used to measure the elapsed time of a CUDA call or kernel execution. The details of various CPU timing approaches are outside the scope of this document, but developers should always be aware of the resolution their timing calls provide.

When using CPU timers, it is critical to remember that many CUDA API functions are asynchronous; that is, they return control back to the calling CPU thread prior to completing their work. All kernel launches are asynchronous, as are memory-copy functions with the `Async` suffix on their names. Therefore, to accurately measure the elapsed time for a particular call or sequence of CUDA calls, it is necessary to synchronize the CPU thread with the GPU by calling `cudaDeviceSynchronize()` immediately before starting and stopping the CPU timer. `cudaDeviceSynchronize()`blocks the calling CPU thread until all CUDA calls previously issued by the thread are completed.

Although it is also possible to synchronize the CPU thread with a particular stream or event on the GPU, these synchronization functions are not suitable for timing code in streams other than the default stream. `cudaStreamSynchronize()` blocks the CPU thread until all CUDA calls previously issued into the given stream have completed. `cudaEventSynchronize()` blocks until a given event in a particular stream has been recorded by the GPU. Because the driver may interleave execution of CUDA calls from other non-default streams, calls in other streams may be included in the timing.

Because the default stream, stream 0, exhibits serializing behavior for work on the device (an operation in the default stream can begin only after all preceding calls in any stream have completed; and no subsequent operation in any stream can begin until it finishes), these functions can be used reliably for timing in the default stream.

Be aware that CPU-to-GPU synchronization points such as those mentioned in this section imply a stall in the GPU’s processing pipeline and should thus be used sparingly to minimize their performance impact.

###  9.1.2. Using CUDA GPU Timers 

The CUDA event API provides calls that create and destroy events, record events (including a timestamp), and convert timestamp differences into a floating-point value in milliseconds. [How to time code using CUDA events](#how-to-time-code-using-cuda-events-figure) illustrates their use.

How to time code using CUDA events
    
    
    cudaEvent_t start, stop;
    float time;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord( start, 0 );
    kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y,
                               NUM_REPS);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    
    cudaEventElapsedTime( &time, start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
    

Here `cudaEventRecord()` is used to place the `start` and `stop` events into the default stream, stream 0. The device will record a timestamp for the event when it reaches that event in the stream. The `cudaEventElapsedTime()` function returns the time elapsed between the recording of the `start` and `stop` events. This value is expressed in milliseconds and has a resolution of approximately half a microsecond. Like the other calls in this listing, their specific operation, parameters, and return values are described in the _CUDA Toolkit Reference Manual_. Note that the timings are measured on the GPU clock, so the timing resolution is operating-system-independent.


##  9.2. Bandwidth 

Bandwidth - the rate at which data can be transferred - is one of the most important gating factors for performance. Almost all changes to code should be made in the context of how they affect bandwidth. As described in [Memory Optimizations](#memory-optimizations) of this guide, bandwidth can be dramatically affected by the choice of memory in which data is stored, how the data is laid out and the order in which it is accessed, as well as other factors.

To measure performance accurately, it is useful to calculate theoretical and effective bandwidth. When the latter is much lower than the former, design or implementation details are likely to reduce bandwidth, and it should be the primary goal of subsequent optimization efforts to increase it.

Note

**High Priority:** Use the effective bandwidth of your computation as a metric when measuring performance and optimization benefits.

###  9.2.1. Theoretical Bandwidth Calculation 

Theoretical bandwidth can be calculated using hardware specifications available in the product literature. For example, the NVIDIA Tesla V100 uses HBM2 (double data rate) RAM with a memory clock rate of 877 MHz and a 4096-bit-wide memory interface.

Using these data items, the peak theoretical memory bandwidth of the NVIDIA Tesla V100 is 898 GB/s:

\\(\left. \left( 0.877 \times 10^{9} \right. \times (4096/8) \times 2 \right) \div 10^{9} = 898\text{GB/s}\\)

In this calculation, the memory clock rate is converted in to Hz, multiplied by the interface width (divided by 8, to convert bits to bytes) and multiplied by 2 due to the double data rate. Finally, this product is divided by 109 to convert the result to GB/s.

Note

Some calculations use 10243 instead of 109 for the final calculation. In such a case, the bandwidth would be 836.4 GiB/s. It is important to use the same divisor when calculating theoretical and effective bandwidth so that the comparison is valid.

Note

On GPUs with GDDR memory with ECC enabled the available DRAM is reduced by 6.25% to allow for the storage of ECC bits. Fetching ECC bits for each memory transaction also reduced the effective bandwidth by approximately 20% compared to the same GPU with ECC disabled, though the exact impact of ECC on bandwidth can be higher and depends on the memory access pattern. HBM2 memories, on the other hand, provide dedicated ECC resources, allowing overhead-free ECC protection.[2](#fn2)

###  9.2.2. Effective Bandwidth Calculation 

Effective bandwidth is calculated by timing specific program activities and by knowing how data is accessed by the program. To do so, use this equation:

\\(\text{Effective\ bandwidth} = \left( {\left( B_{r} + B_{w} \right) \div 10^{9}} \right) \div \text{time}\\)

Here, the effective bandwidth is in units of GB/s, Br is the number of bytes read per kernel, Bw is the number of bytes written per kernel, and time is given in seconds.

For example, to compute the effective bandwidth of a 2048 x 2048 matrix copy, the following formula could be used:

\\(\text{Effective\ bandwidth} = \left( {\left( 2048^{2} \times 4 \times 2 \right) \div 10^{9}} \right) \div \text{time}\\)

The number of elements is multiplied by the size of each element (4 bytes for a float), multiplied by 2 (because of the read _and_ write), divided by 109 (or 1,0243) to obtain GB of memory transferred. This number is divided by the time in seconds to obtain GB/s.

###  9.2.3. Throughput Reported by Visual Profiler 

For devices with [compute capability](#cuda-compute-capability) of 2.0 or greater, the Visual Profiler can be used to collect several different memory throughput measures. The following throughput metrics can be displayed in the Details or Detail Graphs view:

  * Requested Global Load Throughput

  * Requested Global Store Throughput

  * Global Load Throughput

  * Global Store Throughput

  * DRAM Read Throughput

  * DRAM Write Throughput


The Requested Global Load Throughput and Requested Global Store Throughput values indicate the global memory throughput requested by the kernel and therefore correspond to the effective bandwidth obtained by the calculation shown under [Effective Bandwidth Calculation](#effective-bandwidth-calculation).

Because the minimum memory transaction size is larger than most word sizes, the actual memory throughput required for a kernel can include the transfer of data not used by the kernel. For global memory accesses, this actual throughput is reported by the Global Load Throughput and Global Store Throughput values.

It’s important to note that both numbers are useful. The actual memory throughput shows how close the code is to the hardware limit, and a comparison of the effective or requested bandwidth to the actual bandwidth presents a good estimate of how much bandwidth is wasted by suboptimal coalescing of memory accesses (see [Coalesced Access to Global Memory](#coalesced-access-to-global-memory)). For global memory accesses, this comparison of requested memory bandwidth to actual memory bandwidth is reported by the Global Memory Load Efficiency and Global Memory Store Efficiency metrics.

[2](#id24)
    

As an exception, scattered writes to HBM2 see some overhead from ECC but much less than the overhead with similar access patterns on ECC-protected GDDR5 memory.
