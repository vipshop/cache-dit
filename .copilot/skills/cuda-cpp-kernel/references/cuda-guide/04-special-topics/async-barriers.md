---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html
---

# 4.9. Asynchronous Barriers

Asynchronous barriers, introduced in [Advanced Synchronization Primitives](../03-advanced/advanced-kernel-programming.html#advanced-kernels-advanced-sync-primitives), extend CUDA synchronization beyond `__syncthreads()` and `__syncwarp()`, enabling fine-grained, non-blocking coordination and better overlap of communication and computation.

This section provides details on how to use asynchronous barriers mainly via the `cuda::barrier` API (with pointers to `cuda::ptx` and primitives where applicable).

## 4.9.1. Initialization

Initialization must happen before any thread begins participating in a barrier.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/barrier>
    #include <cooperative_groups.h>
    
    __global__ void init_barrier()
    {
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // A single thread initializes the total expected arrival count.
        init(&bar, block.size());
      }
      block.sync();
    }
      
  
---  
  
CUDA C++ `cuda::ptx`
    
    
    #include <cuda/ptx>
    #include <cooperative_groups.h>
    
    __global__ void init_barrier()
    {
      __shared__ uint64_t bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // A single thread initializes the total expected arrival count.
        cuda::ptx::mbarrier_init(&bar, block.size());
      }
      block.sync();
    }
      
  
---  
  
CUDA C primitives
    
    
    #include <cuda_awbarrier_primitives.h>
    #include <cooperative_groups.h>
    
    __global__ void init_barrier()
    {
      __shared__ uint64_t bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // A single thread initializes the total expected arrival count.
        __mbarrier_init(&bar, block.size());
      }
      block.sync();
    }
      
  
---  
  
Before any thread can participate in a barrier, the barrier must be initialized using the `cuda::barrier::init()` friend function. This must happen before any thread arrives on the barrier. This poses a bootstrapping challenge in that threads must synchronize before participating in the barrier, but threads are creating a barrier in order to synchronize. In this example, threads that will participate are part of a cooperative group and use `block.sync()` to bootstrap initialization. Since a whole thread block is participating in the barrier, `__syncthreads()` could also be used.

The second parameter of `init()` is the _expected arrival count_ , i.e., the number of times `bar.arrive()` will be called by participating threads before a participating thread is unblocked from its call to `bar.wait(std::move(token))`. In this and the previous examples, the barrier is initialized with the number of threads in the thread block i.e., `cooperative_groups::this_thread_block().size()`, so that all threads within the thread block can participate in the barrier.

Asynchronous barriers are flexible in specifying _how_ threads participate (split arrive/wait) and _which_ threads participate. In contrast, `this_thread_block.sync()` or `__syncthreads()` is applicable to the whole thread-block and `__syncwarp(mask)` to a specified subset of a warp. Nonetheless, if the intention of the user is to synchronize a full thread block or a full warp, we recommend using `__syncthreads()` and `__syncwarp()` respectively for better performance.

## 4.9.2. A Barrier’s Phase: Arrival, Countdown, Completion, and Reset

An asynchronous barrier counts down from the expected arrival count to zero as participating threads call `bar.arrive()`. When the countdown reaches zero, the barrier is complete for the current phase. When the last call to `bar.arrive()` causes the countdown to reach zero, the countdown is automatically and atomically reset. The reset assigns the countdown to the expected arrival count, and moves the barrier to the next phase.

A `token` object of class `cuda::barrier::arrival_token`, as returned from `token=bar.arrive()`, is associated with the current phase of the barrier. A call to `bar.wait(std::move(token))` blocks the calling thread while the barrier is in the current phase, i.e., while the phase associated with the token matches the phase of the barrier. If the phase is advanced (because the countdown reaches zero) before the call to `bar.wait(std::move(token))` then the thread does not block; if the phase is advanced while the thread is blocked in `bar.wait(std::move(token))`, the thread is unblocked.

**It is essential to know when a reset could or could not occur, especially in non-trivial arrive/wait synchronization patterns.**

  * A thread’s calls to `token=bar.arrive()` and `bar.wait(std::move(token))` must be sequenced such that `token=bar.arrive()` occurs during the barrier’s current phase, and `bar.wait(std::move(token))` occurs during the same or next phase.

  * A thread’s call to `bar.arrive()` must occur when the barrier’s counter is non-zero. After barrier initialization, if a thread’s call to `bar.arrive()` causes the countdown to reach zero then a call to `bar.wait(std::move(token))` must happen before the barrier can be reused for a subsequent call to `bar.arrive()`.

  * `bar.wait()` must only be called using a `token` object of the current phase or the immediately preceding phase. For any other values of the `token` object, the behavior is undefined.


For simple arrive/wait synchronization patterns, compliance with these usage rules is straightforward.

### 4.9.2.1. Warp Entanglement

Warp-divergence affects the number of times an arrive on operation updates the barrier. If the invoking warp is fully converged, then the barrier is updated once. If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier.

Note

It is recommended that `arrive-on(bar)` invocations are used by converged threads to minimize updates to the barrier object. When code preceding these operations diverges threads, then the warp should be re-converged, via `__syncwarp` before invoking arrive-on operations.

## 4.9.3. Explicit Phase Tracking

An asynchronous barrier can have multiple phases depending on how many times it is used to synchronize threads and memory operations. Instead of using tokens to track barrier phase flips, we can directly track a phase using the `mbarrier_try_wait_parity()` family of functions available through the `cuda::ptx` and primitives APIs.

In its simplest form, the `cuda::ptx::mbarrier_try_wait_parity(uint64_t* bar, const uint32_t& phaseParity)` function waits for a phase with a particular parity. The `phaseParity` operand is the integer parity of either the current phase or the immediately preceding phase of the barrier object. An even phase has integer parity 0 and an odd phase has integer parity 1. When we initialize a barrier, its phase has parity 0. So the valid values of `phaseParity` are 0 and 1. Explicit phase tracking can be useful when tracking [asynchronous memory operations](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies), as it allows only a single thread to arrive on the barrier and set the transaction count, while other threads only wait for a parity-based phase flip. This can be more efficient than having all threads arrive on the barrier and use tokens. This functionality is only available for shared-memory barriers at thread-block and cluster scope.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/ptx>
    #include <cooperative_groups.h>
    
    __device__ void compute(float *data, int iteration);
    
    __global__ void split_arrive_wait(int iteration_count, float *data)
    {
      using barrier_t = cuda::barrier<cuda::thread_scope_block>;
      __shared__ barrier_t bar;
      int parity = 0; // Initial phase parity is 0.
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // Initialize barrier with expected arrival count.
        init(&bar, block.size());
      }
      block.sync();
    
      for (int i = 0; i < iteration_count; ++i)
      {
        /* code before arrive */
    
        // This thread arrives. Arrival does not block a thread.
        // Get a handle to the native barrier to use with cuda::ptx API.
        (void)cuda::ptx::mbarrier_arrive(cuda::device::barrier_native_handle(bar));
    
        compute(data, i);
    
        // Wait for all threads participating in the barrier to complete mbarrier_arrive().
        // Get a handle to the native barrier to use with cuda::ptx API.
        while (!cuda::ptx::mbarrier_try_wait_parity(cuda::device::barrier_native_handle(bar), parity)) {}
        // Flip parity.
        parity ^= 1;
    
        /* code after wait */
      }
    }
      
  
---  
  
CUDA C++ `cuda::ptx`
    
    
    #include <cuda/ptx>
    #include <cooperative_groups.h>
    
    __device__ void compute(float *data, int iteration);
    
    __global__ void split_arrive_wait(int iteration_count, float *data)
    {
      __shared__ uint64_t bar;
      int parity = 0; // Initial phase parity is 0.
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // Initialize barrier with expected arrival count.
        cuda::ptx::mbarrier_init(&bar, block.size());
      }
      block.sync();
    
      for (int i = 0; i < iteration_count; ++i)
      {
        /* code before arrive */
    
        // This thread arrives. Arrival does not block a thread.
        (void)cuda::ptx::mbarrier_arrive(&bar);
    
        compute(data, i);
    
        // Wait for all threads participating in the barrier to complete mbarrier_arrive().
        while (!cuda::ptx::mbarrier_try_wait_parity(&bar, parity)) {}
        // Flip parity.
        parity ^= 1;
    
        /* code after wait */
      }
    }
      
  
---  
  
CUDA C primitives
    
    
    #include <cuda_awbarrier_primitives.h>
    #include <cooperative_groups.h>
    
    __device__ void compute(float *data, int iteration);
    
    __global__ void split_arrive_wait(int iteration_count, float *data)
    {
      __shared__ __mbarrier_t bar;
      bool parity = false; // Initial phase parity is false.
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        // Initialize barrier with expected arrival count.
        __mbarrier_init(&bar, block.size());
      }
      block.sync();
    
      for (int i = 0; i < iteration_count; ++i)
      {
        /* code before arrive */
    
        // This thread arrives. Arrival does not block a thread.
        (void)__mbarrier_arrive(&bar);
    
        compute(data, i);
    
        // Wait for all threads participating in the barrier to complete __mbarrier_arrive().
        while(!__mbarrier_try_wait_parity(&bar, parity, 1000)) {}
        parity ^= 1;
    
        /* code after wait */
      }
    }
      
  
---  
  
## 4.9.4. Early Exit

When a thread that is participating in a sequence of synchronizations must exit early from that sequence, that thread must explicitly drop out of participation before exiting. The remaining participating threads can proceed normally with subsequent arrive and wait operations.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/barrier>
    #include <cooperative_groups.h>
    
    __device__ bool condition_check();
    
    __global__ void early_exit_kernel(int N)
    {
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        init(&bar, block.size());
      }
      block.sync();
    
      for (int i = 0; i < N; ++i)
      {
        if (condition_check())
        {
          bar.arrive_and_drop();
          return;
        }
        // Other threads can proceed normally.
        auto token = bar.arrive();
    
        /* code between arrive and wait */
    
        // Wait for all threads to arrive.
        bar.wait(std::move(token));
    
        /* code after wait */
      }
    }
      
  
---  
  
CUDA C primitives
    
    
    #include <cuda_awbarrier_primitives.h>
    #include <cooperative_groups.h>
    
    __device__ bool condition_check();
    
    __global__ void early_exit_kernel(int N)
    {
      __shared__ __mbarrier_t bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        __mbarrier_init(&bar, block.size());
      }
      block.sync();
    
      for (int i = 0; i < N; ++i)
      {
        if (condition_check())
        {
          __mbarrier_token_t token = __mbarrier_arrive_and_drop(&bar);
          return;
        }
        // Other threads can proceed normally.
        __mbarrier_token_t token = __mbarrier_arrive(&bar);
    
        /* code between arrive and wait */
    
        // Wait for all threads to arrive.
        while (!__mbarrier_try_wait(&bar, token, 1000)) {}
    
        /* code after wait */
      }
    }
      
  
---  
  
The `bar.arrive_and_drop()` operation arrives on the barrier to fulfill the participating thread’s obligation to arrive in the **current** phase, and then decrements the expected arrival count for the **next** phase so that this thread is no longer expected to arrive on the barrier.

## 4.9.5. Completion Function

The `cuda::barrier` API supports an optional completion function. A `CompletionFunction` of `cuda::barrier<Scope, CompletionFunction>` is executed once per phase, after the last thread _arrives_ and before any thread is unblocked from the `wait`. Memory operations performed by the threads that arrived at the `barrier` during the phase are visible to the thread executing the `CompletionFunction`, and all memory operations performed within the `CompletionFunction` are visible to all threads waiting at the `barrier` once they are unblocked from the `wait`.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/barrier>
    #include <cooperative_groups.h>
    #include <functional>
    namespace cg = cooperative_groups;
    
    __device__ int divergent_compute(int *, int);
    __device__ int independent_computation(int *, int);
    
    __global__ void psum(int *data, int n, int *acc)
    {
      auto block = cg::this_thread_block();
    
      constexpr int BlockSize = 128;
      __shared__ int smem[BlockSize];
      assert(BlockSize == block.size());
      assert(n % BlockSize == 0);
    
      auto completion_fn = [&]
      {
        int sum = 0;
        for (int i = 0; i < BlockSize; ++i)
        {
          sum += smem[i];
        }
        *acc += sum;
      };
    
      /* Barrier storage.
         Note: the barrier is not default-constructible because
               completion_fn is not default-constructible due
               to the capture. */
      using completion_fn_t = decltype(completion_fn);
      using barrier_t = cuda::barrier<cuda::thread_scope_block,
                                      completion_fn_t>;
      __shared__ std::aligned_storage<sizeof(barrier_t),
                                      alignof(barrier_t)>
          bar_storage;
    
      // Initialize barrier.
      barrier_t *bar = (barrier_t *)&bar_storage;
      if (block.thread_rank() == 0)
      {
        assert(*acc == 0);
        assert(blockDim.x == blockDim.y == blockDim.y == 1);
        new (bar) barrier_t{block.size(), completion_fn};
        /* equivalent to: init(bar, block.size(), completion_fn); */
      }
      block.sync();
    
      // Main loop.
      for (int i = 0; i < n; i += block.size())
      {
        smem[block.thread_rank()] = data[i] + *acc;
        auto token = bar->arrive();
        // We can do independent computation here.
        bar->wait(std::move(token));
        // Shared-memory is safe to re-use in the next iteration
        // since all threads are done with it, including the one
        // that did the reduction.
      }
    }
      
  
---  
  
## 4.9.6. Tracking Asynchronous Memory Operations

Asynchronous barriers can be used to track [asynchronous memory copies](../03-advanced/advanced-kernel-programming.html#advanced-kernels-async-copies). When an asynchronous copy operation is bound to a barrier, the copy operation automatically increments the expected count of the current barrier phase upon initiation and decrements it upon completion. This mechanism ensures that the barrier’s `wait()` operation will block until all associated asynchronous memory copies have completed, providing a convenient way to synchronize multiple concurrent memory operations.

Starting with compute capability 9.0, asynchronous barriers in shared memory with thread-block or cluster scope can **explicitly** track asynchronous memory operations. We refer to these barriers as _asynchronous transaction barriers_. In addition to the expected arrival count, a barrier object can accept a **transaction count** , which can be used for tracking the completion of asynchronous transactions. The transaction count tracks the number of asynchronous transactions that are outstanding and yet to be complete, in units specified by the asynchronous memory operation (typically bytes). The transaction count to be tracked by the current phase can be set on arrival with `cuda::device::barrier_arrive_tx()` or directly with `cuda::device::barrier_expect_tx()`. When a barrier uses a transaction count, it blocks threads at the wait operation until all the producer threads have performed an arrive _and_ the sum of all the transaction counts reaches an expected value.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/barrier>
    #include <cooperative_groups.h>
    
    __global__ void track_kernel()
    {
      __shared__ cuda::barrier<cuda::thread_scope_block> bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        init(&bar, block.size());
      }
      block.sync();
    
      auto token = cuda::device::barrier_arrive_tx(bar, 1, 0);
    
      bar.wait(cuda::std::move(token));
    }
      
  
---  
  
CUDA C++ `cuda::ptx`
    
    
    #include <cuda/ptx>
    #include <cooperative_groups.h>
    
    __global__ void track_kernel()
    {
      __shared__ uint64_t bar;
      auto block = cooperative_groups::this_thread_block();
    
      if (block.thread_rank() == 0)
      {
        cuda::ptx::mbarrier_init(&bar, block.size());
      }
      block.sync();
    
      uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release, cuda::ptx::scope_cluster, cuda::ptx::space_shared, &bar, 1, 0);
    
      while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
    }
      
  
---  
  
In this example, the `cuda::device::barrier_arrive_tx()` operation constructs an arrival token object associated with the phase synchronization point for the current phase. Then, decrements the arrival count by 1 and increments the expected transaction count by 0. Since the transaction count update is 0, the barrier is not tracking any transactions. The subsequent section on [Using the Tensor Memory Accelerator (TMA)](async-copies.html#async-copies-tma) includes examples of tracking asynchronous memory operations.

## 4.9.7. Producer-Consumer Pattern Using Barriers

A thread block can be spatially partitioned to allow different threads to perform independent operations. This is most commonly done by assigning threads from different warps within the thread block to specific tasks. This technique is referred to as _warp specialization_.

This section shows an example of spatial partitioning in a producer-consumer pattern, where one subset of threads produces data that is concurrently consumed by the other (disjoint) subset of threads. A producer-consumer spatial partitioning pattern requires two one-sided synchronizations to manage a data buffer between the producer and consumer.

Producer | Consumer  
---|---  
wait for buffer to be ready to be filled | signal buffer is ready to be filled  
produce data and fill the buffer |   
signal buffer is filled | wait for buffer to be filled  
| consume data in filled buffer  
  
Producer threads wait for consumer threads to signal that the buffer is ready to be filled; however, consumer threads do not wait for this signal. Consumer threads wait for producer threads to signal that the buffer is filled; however, producer threads do not wait for this signal. For full producer/consumer concurrency this pattern has (at least) double buffering where each buffer requires two barriers.

CUDA C++ `cuda::barrier`
    
    
    #include <cuda/barrier>
    
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;
    
    __device__ void produce(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *in, int N)
    {
      for (int i = 0; i < N / buffer_len; ++i)
      {
        ready[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        barrier_t::arrival_token token = filled[i % 2].arrive(); /* buffer_(i%2) is filled */
      }
    }
    
    __device__ void consume(barrier_t ready[], barrier_t filled[], float *buffer, int buffer_len, float *out, int N)
    {
      barrier_t::arrival_token token1 = ready[0].arrive(); /* buffer_0 is ready for initial fill */
      barrier_t::arrival_token token2 = ready[1].arrive(); /* buffer_1 is ready for initial fill */
      for (int i = 0; i < N / buffer_len; ++i)
      {
        filled[i % 2].arrive_and_wait(); /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        barrier_t::arrival_token token3 = ready[i % 2].arrive(); /* buffer_(i%2) is ready to be re-filled */
      }
    }
    
    __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
    {
      constexpr int warpSize = 32;
    
      /* Shared memory buffer declared below is of size 2 * buffer_len
         so that we can alternatively work between two buffers.
         buffer_0 = buffer and buffer_1 = buffer + buffer_len */
      __shared__ extern float buffer[];
    
      /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
         while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
      #pragma nv_diag_suppress static_var_with_dynamic_init
      __shared__ barrier_t bar[4];
    
      if (threadIdx.x < 4)
      {
        init(bar + threadIdx.x, blockDim.x);
      }
      __syncthreads();
    
      if (threadIdx.x < warpSize)
      { produce(bar, bar + 2, buffer, buffer_len, in, N); }
      else
      { consume(bar, bar + 2, buffer, buffer_len, out, N); }
    }
      
  
---  
  
CUDA C++ `cuda::ptx`
    
    
    #include <cuda/ptx>
    
    __device__ void produce(barrier ready[], barrier filled[], float *buffer, int buffer_len, float *in, int N)
    {
      for (int i = 0; i < N / buffer_len; ++i)
      {
        uint64_t token1 = cuda::ptx::mbarrier_arrive(ready[i % 2]);
        while(!cuda::ptx::mbarrier_try_wait(&ready[i % 2], token1)) {} /* wait for buffer_(i%2) to be ready to be filled */
        /* produce, i.e., fill in, buffer_(i%2)  */
        uint64_t token2 = cuda::ptx::mbarrier_arrive(&filled[i % 2]); /* buffer_(i%2) is filled */
      }
    }
    
    __device__ void consume(barrier ready[], barrier filled[], float *buffer, buffer_len, float *out, int N)
    {
      uint64_t token1 = cuda::ptx::mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
      uint64_t token2 = cuda::ptx::mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
      for (int i = 0; i < N / buffer_len; ++i)
      {
        uint64_t token3 = cuda::ptx::mbarrier_arrive(&filled[i % 2]);
        while(!cuda::ptx::mbarrier_try_wait(&filled[i % 2], token3x)) {} /* wait for buffer_(i%2) to be filled */
        /* consume buffer_(i%2) */
        uint64_t token4 = cuda::ptx::mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
      }
    }
    
    __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
    {
      constexpr int warpSize = 32;
    
      /* Shared memory buffer declared below is of size 2 * buffer_len
         so that we can alternatively work between two buffers.
         buffer_0 = buffer and buffer_1 = buffer + buffer_len */
      __shared__ extern float buffer[];
    
      /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
         while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
      #pragma nv_diag_suppress static_var_with_dynamic_init
      __shared__ uint64_t bar[4];
    
      if (threadIdx.x < 4)
      {
        cuda::ptx::mbarrier_init(bar + block.thread_rank(), block.size());
      }
      __syncthreads();
    
      if (threadIdx.x < warpSize)
      {  produce(bar, bar + 2, buffer, buffer_len, in, N); }
      else
      {  consume(bar, bar + 2, buffer, buffer_len, out, N); }
    }
      
  
---  
  
CUDA C primitives
    
    
    #include <cuda_awbarrier_primitives.h>
    
    __device__ void produce(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *in, int N)
    {
      for (int i = 0; i < N / buffer_len; ++i)
      {
        __mbarrier_token_t token1 = __mbarrier_arrive(&ready[i % 2]); /* wait for buffer_(i%2) to be ready to be filled */
        while(!__mbarrier_try_wait(&ready[i % 2], token1, 1000)) {}
        /* produce, i.e., fill in, buffer_(i%2)  */
        __mbarrier_token_t token2 = __mbarrier_arrive(filled[i % 2]);  /* buffer_(i%2) is filled */
      }
    }
    
    __device__ void consume(__mbarrier_t ready[], __mbarrier_t filled[], float *buffer, int buffer_len, float *out, int N)
    {
      __mbarrier_token_t token1 = __mbarrier_arrive(&ready[0]); /* buffer_0 is ready for initial fill */
      __mbarrier_token_t token2 = __mbarrier_arrive(&ready[1]); /* buffer_1 is ready for initial fill */
      for (int i = 0; i < N / buffer_len; ++i)
      {
        __mbarrier_token_t token3 = __mbarrier_arrive(&filled[i % 2]);
        while(!__mbarrier_try_wait(&filled[i % 2], token3, 1000)) {}
        /* consume buffer_(i%2) */
        __mbarrier_token_t token4 = __mbarrier_arrive(&ready[i % 2]); /* buffer_(i%2) is ready to be re-filled */
      }
    }
    
    __global__ void producer_consumer_pattern(int N, float *in, float *out, int buffer_len)
    {
      constexpr int warpSize = 32;
    
      /* Shared memory buffer declared below is of size 2 * buffer_len
         so that we can alternatively work between two buffers.
         buffer_0 = buffer and buffer_1 = buffer + buffer_len */
      __shared__ extern float buffer[];
    
      /* bar[0] and bar[1] track if buffers buffer_0 and buffer_1 are ready to be filled,
         while bar[2] and bar[3] track if buffers buffer_0 and buffer_1 are filled-in respectively */
      #pragma nv_diag_suppress static_var_with_dynamic_init
      __shared__ __mbarrier_t bar[4];
    
      if (threadIdx.x < 4)
      {
        __mbarrier_init(bar + threadIdx.x, blockDim.x);
      }
      __syncthreads();
    
      if (threadIdx.x < warpSize)
      { produce(bar, bar + 2, buffer, buffer_len, in, N); }
      else
      { consume(bar, bar + 2, buffer, buffer_len, out, N); }
    }
      
  
---  
  
In this example, the first warp is specialized as the producer and the remaining warps are specialized as consumers. All producer and consumer threads participate (call `bar.arrive()` or `bar.arrive_and_wait()`) in each of the four barriers so the expected arrival counts are equal to `block.size()`.

A producer thread waits for the consumer threads to signal that the shared memory buffer can be filled. In order to wait for a barrier, a producer thread must first arrive on that `ready[i%2].arrive()` to get a token and then `ready[i%2].wait(token)` with that token. For simplicity, `ready[i%2].arrive_and_wait()` combines these operations.
    
    
    bar.arrive_and_wait();
    /* is equivalent to */
    bar.wait(bar.arrive());
    

Producer threads compute and fill the ready buffer, they then signal that the buffer is filled by arriving on the filled barrier, `filled[i%2].arrive()`. A producer thread does not wait at this point, instead it waits until the next iteration’s buffer (double buffering) is ready to be filled.

A consumer thread begins by signaling that both buffers are ready to be filled. A consumer thread does not wait at this point, instead it waits for this iteration’s buffer to be filled, `filled[i%2].arrive_and_wait()`. After the consumer threads consume the buffer they signal that the buffer is ready to be filled again, `ready[i%2].arrive()`, and then wait for the next iteration’s buffer to be filled.
