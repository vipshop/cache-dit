---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/pipelines.html
---

# 4.10. Pipelines

Pipelines, introduced in [Advanced Synchronization Primitives](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives), are a mechanism for staging work and coordinating multi-buffer producer–consumer patterns, commonly used to overlap compute with [asynchronous data copies](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies).

This section provides details on how to use pipelines mainly via the `cuda::pipeline` API (with pointers to primitives where applicable).

## 4.10.1. Initialization

A `cuda::pipeline` can be created at different thread scopes. For a scope other than `cuda::thread_scope_thread`, a `cuda::pipeline_shared_state<scope, count>` object is required to coordinate the participating threads. This state encapsulates the finite resources that allow a pipeline to process up to `count` concurrent stages.
    
    
    // Create a pipeline at thread scope
    constexpr auto scope = cuda::thread_scope_thread;
    cuda::pipeline<scope> pipeline = cuda::make_pipeline();
    
    
    
    // Create a pipeline at block scope
    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto pipeline = cuda::make_pipeline(group, &shared_state);
    

Pipelines can be either _unified_ or _partitioned_. In a unified pipeline, all the participating threads are both producers and consumers. In a partitioned pipeline, each participating thread is either a producer or a consumer and its role cannot change during the lifetime of the pipeline object. A thread-local pipeline cannot be partitioned. To create a partitioned pipeline, we need to provide either the number of producers or the role of the thread to `cuda::make_pipeline()`.
    
    
    // Create a partitioned pipeline at block scope where only thread 0 is a producer
    constexpr auto scope = cuda::thread_scope_block;
    constexpr auto stages_count = 2;
    __shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
    auto thread_role = (group.thread_rank() == 0) ? cuda::pipeline_role::producer : cuda::pipeline_role::consumer;
    auto pipeline = cuda::make_pipeline(group, &shared_state, thread_role);
    

To support partitioning, a shared `cuda::pipeline` incurs additional overheads, including using a set of shared memory barriers per stage for synchronization. These are used even when the pipeline is unified and could use `__syncthreads()` instead. Thus, it is preferable to use thread-local pipelines which avoid these overheads when possible.

## 4.10.2. Submitting Work

Committing work to a pipeline stage involves:

>   * Collectively _acquiring_ the pipeline _head_ from a set of producer threads using `pipeline.producer_acquire()`.
> 
>   * Submitting asynchronous operations, e.g., `memcpy_async`, to the pipeline head.
> 
>   * Collectively _committing_ (advancing) the pipeline head using `pipeline.producer_commit()`.
> 
> 


If all resources are in use, `pipeline.producer_acquire()` blocks producer threads until the resources of the next pipeline stage are released by consumer threads.

## 4.10.3. Consuming Work

Consuming work from a previously committed stage involves:

>   * Collectively waiting for the stage to complete, e.g., using `pipeline.consumer_wait()` to wait on the tail (oldest) stage, from a set of consumer threads.
> 
>   * Collectively _releasing_ the stage using `pipeline.consumer_release()`.
> 
> 


With `cuda::pipeline<cuda:thread_scope_thread>` one can also use the `cuda::pipeline_consumer_wait_prior<N>()` friend function to wait for all except the last N stages to complete, similar to `__pipeline_wait_prior(N)` in the primitives API.

## 4.10.4. Warp Entanglement

The pipeline mechanism is shared among CUDA threads in the same warp. This sharing causes sequences of submitted operations to be entangled within a warp, which can impact performance under certain circumstances.

**Commit**. The commit operation is coalesced such that the pipeline’s sequence is incremented once for all converged threads that invoke the commit operation and their submitted operations are batched together. If the warp is fully converged, the sequence is incremented by one and all submitted operations will be batched in the same stage of the pipeline; if the warp is fully diverged, the sequence is incremented by 32 and all submitted operations will be spread to different stages.

  * Let _PB_ be the warp-shared pipeline’s _actual_ sequence of operations.

`PB = {BP0, BP1, BP2, …, BPL}`

  * Let _TB_ be a thread’s _perceived_ sequence of operations, as if the sequence were only incremented by this thread’s invocation of the commit operation.

`TB = {BT0, BT1, BT2, …, BTL}`


> The `pipeline::producer_commit()` return value is from the thread’s _perceived_ batch sequence.

  * An index in a thread’s perceived sequence always aligns to an equal or larger index in the actual warp-shared sequence. The sequences are equal only when all commit operations are invoked from fully converged threads.

`BTn ≡ BPm` where `n <= m`


For example, when a warp is fully diverged:

  * The warp-shared pipeline’s actual sequence would be: `PB = {0, 1, 2, 3, ..., 31}` (`PL=31`).

  * The perceived sequence for each thread of this warp would be:

    * Thread 0: `TB = {0}` (`TL=0`)

    * Thread 1: `TB = {0}` (`TL=0`)

    * `…`

    * Thread 31: `TB = {0}` (`TL=0`)


**Wait**. A CUDA thread invokes `pipeline::consumer_wait()` or `pipeline_consumer_wait_prior<N>()` to wait for batches in the _perceived_ sequence `TB` to complete. Note that `pipeline::consumer_wait()` is equivalent to `pipeline_consumer_wait_prior<N>()`, where `N = PL`.

The _wait prior_ variants wait for batches in the _actual_ sequence at least up to and including `PL-N`. Since `TL <= PL`, waiting for batch up to and including `PL-N` includes waiting for batch `TL-N`. Thus, when `TL < PL`, the thread will unintentionally wait for additional, more recent batches. In the extreme fully-diverged warp example above, each thread could wait for all 32 batches.

Note

It is recommended that commit invocations are by converged threads to not over-wait, by keeping threads’ perceived sequence of batches aligned with the actual sequence.

When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking commit operations.

## 4.10.5. Early Exit

When a thread that is participating in a pipeline must exit early, that thread must explicitly drop out of participation before exiting using `cuda::pipeline::quit()`. The remaining participating threads can proceed normally with subsequent operations.

## 4.10.6. Tracking Asynchronous Memory Operations

The following example demonstrates how to collectively copy data from global to shared memory with asynchronous memory copies using a pipeline to keep track of the copy operations. Each thread uses its own pipeline to independently submit memory copies and then wait for them to complete and consume the data. For more details on asynchronous data copies, see [Section 3.2.5](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies).

CUDA C++ `cuda::pipeline`
    
    
    #include <cuda/pipeline>
    
    __global__ void example_kernel(const float *in)
    {
        constexpr int block_size = 128;
        __shared__ __align__(sizeof(float)) float buffer[4 * block_size];
    
        // Create a unified pipeline per thread
        cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
    
        // First stage of memory copies
        pipeline.producer_acquire();
        // Every thread fetches one element of the first block
        cuda::memcpy_async(buffer, in, sizeof(float), pipeline);
        pipeline.producer_commit();
    
        // Second stage of memory copies
        pipeline.producer_acquire();
        // Every thread fetches one element of the second and third block
        cuda::memcpy_async(buffer + block_size, in + block_size, sizeof(float), pipeline);
        cuda::memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float), pipeline);
        pipeline.producer_commit();
    
        // Third stage of memory copies
        pipeline.producer_acquire();
        // Every thread fetches one element of the last block
        cuda::memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float), pipeline);
        pipeline.producer_commit();
    
        // Wait for the oldest stage (waits for first stage)
        pipeline.consumer_wait();
        pipeline.consumer_release();
    
        // __syncthreads();
        // Use data from the first stage
    
        // Wait for the oldest stage (waits for second stage)
        pipeline.consumer_wait();
        pipeline.consumer_release();
    
        // __syncthreads();
        // Use data from the second stage
    
        // Wait for the oldest stage (waits for third stage)
        pipeline.consumer_wait();
        pipeline.consumer_release();
    
        // __syncthreads();
        // Use data from the third stage
    }
      
  
---  
  
CUDA C primitives
    
    
    #include <cuda_pipeline.h>
    
    __global__ void example_kernel(const float *in)
    {
        constexpr int block_size = 128;
        __shared__ __align__(sizeof(float)) float buffer[4 * block_size];
    
        // First batch of memory copies
        // Every thread fetches one element of the first block
        __pipeline_memcpy_async(buffer, in, sizeof(float));
        __pipeline_commit();
    
        // Second batch of memory copies
        // Every thread fetches one element of the second and third block
        __pipeline_memcpy_async(buffer + block_size, in + block_size, sizeof(float));
        __pipeline_memcpy_async(buffer + 2 * block_size, in + 2 * block_size, sizeof(float));
        __pipeline_commit();
    
        // Third batch of memory copies
        // Every thread fetches one element of the last block
        __pipeline_memcpy_async(buffer + 3 * block_size, in + 3 * block_size, sizeof(float));
        __pipeline_commit();
    
        // Wait for all except the last two batches of memory copies (waits for first batch)
        __pipeline_wait_prior(2);
    
        // __syncthreads();
        // Use data from the first batch
    
        // Wait for all except the last batch of memory copies (waits for second batch)
        __pipeline_wait_prior(1);
    
        // __syncthreads();
        // Use data from the second batch
    
        // Wait for all batches of memory copies (waits for third batch)
        __pipeline_wait_prior(0);
    
        // __syncthreads();
        // Use data from the last batch
    }
      
  
---  
  
## 4.10.7. Producer-Consumer Pattern using Pipelines

In [Section 4.9.7](async-barriers.html#asynchronous-barriers-producer-consumer), we showed how a thread block can be spatially partitioned to implement a producer-consumer pattern using [asynchronous barriers](async-barriers.html#asynchronous-barriers). With `cuda::pipeline`, this can be simplified using a single partitioned pipeline with one stage per data buffer instead of two asynchronous barriers per buffer.

CUDA C++ `cuda::pipeline`
    
    
    #include <cuda/pipeline>
    #include <cooperative_groups.h>
    
    #pragma nv_diag_suppress static_var_with_dynamic_init
    
    using pipeline = cuda::pipeline<cuda::thread_scope_block>;
    
    __device__ void produce(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *in, int N)
    {
      if (batch < num_batches)
      {
        pipe.producer_acquire();
        /* copy data from in(batch) to buffer(stage) using asynchronous memory copies */
        pipe.producer_commit();
      }
    }
    
    __device__ void consume(pipeline &pipe, int num_stages, int stage, int num_batches, int batch, float *buffer, int buffer_len, float *out, int N)
    {
      pipe.consumer_wait();
      /* consume buffer(stage) and update out(batch) */
      pipe.consumer_release();
    }
    
    __global__ void producer_consumer_pattern(float *in, float *out, int N, int buffer_len)
    {
      auto block = cooperative_groups::this_thread_block();
    
      /* Shared memory buffer declared below is of size 2 * buffer_len
         so that we can alternatively work between two buffers.
         buffer_0 = buffer and buffer_1 = buffer + buffer_len */
      __shared__ extern float buffer[];
    
      const int num_batches = N / buffer_len;
    
      // Create a partitioned pipeline with 2 stages where half the threads are producers and the other half are consumers.
      constexpr auto scope = cuda::thread_scope_block;
      constexpr int num_stages = 2;
      cuda::std::size_t producer_count = block.size() / 2;
      __shared__ cuda::pipeline_shared_state<scope, num_stages> shared_state;
      pipeline pipe = cuda::make_pipeline(block, &shared_state, producer_count);
    
      // Fill the pipeline
      if (block.thread_rank() < producer_count)
      {
        for (int s = 0; s < num_stages; ++s)
        {
          produce(pipe, num_stages, s, num_batches, s, buffer, buffer_len, in, N);
        }
      }
    
      // Process the batches
      int stage = 0;
      for (size_t b = 0; b < num_batches; ++b)
      {
        if (block.thread_rank() < producer_count)
        {
          // Prefetch the next batch
          produce(pipe, num_stages, stage, num_batches, b + num_stages, buffer, buffer_len, in, N);
        }
        else
        {
          // Consume the oldest batch
          consume(pipe, num_stages, stage, num_batches, b, buffer, buffer_len, out, N);
        }
        stage = (stage + 1) % num_stages;
      }
    }
      
  
---  
  
In this example, we use half of the threads in the thread block as producers and the other half as consumers. As a first step, we need to create a `cuda::pipeline` object. Since we want some threads to be producers and some to be consumers, we need to use a **partitioned** pipeline with `cuda::thread_scope_block`. Partitioned pipelines require a `cuda::pipeline_shared_state` to coordinate the participating threads. We initialize the state for a 2-stage pipeline in thread-block scope and then call `cuda::make_pipeline()`. Next, producer threads fill the pipeline by submitting asynchronous copies from `in` to `buffer`. At this point all data copies are in-flight. Finally, in the main loop, we go over all of the batches of data and depending on whether a thread is a producer or consumer, we either submit another asynchronous copy for a future batch or consume the current batch.
