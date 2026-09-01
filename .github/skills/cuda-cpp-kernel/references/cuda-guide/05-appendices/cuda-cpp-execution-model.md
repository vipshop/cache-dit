---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-execution-model.html
---

# 5.8. CUDA C++ Execution model

CUDA C++ aims to provide [parallel forward progress [intro.progress.9]](https://eel.is/c++draft/intro.progress#9) for all device threads of execution, facilitating the parallelization of pre-existing C++ applications with CUDA C++.

[[intro.progress]](https://eel.is/c++draft/intro.progress)

  * [[intro.progress.7]](https://eel.is/c++draft/intro.progress#7): For a thread of execution providing [concurrent forward progress guarantees](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees), the implementation ensures that the thread will eventually make progress for as long as it has not terminated.

[Note 5: This applies regardless of whether or not other threads of execution (if any) have been or are making progress. To eventually fulfill this requirement means that this will happen in an unspecified but finite amount of time. — end note]

  * [[intro.progress.9]](https://eel.is/c++draft/intro.progress#9): For a thread of execution providing [parallel forward progress guarantees](https://eel.is/c++draft/intro.progress#9), the implementation is not required to ensure that the thread will eventually make progress if it has not yet executed any execution step; once this thread has executed a step, it provides [concurrent forward progress guarantees](https://eel.is/c++draft/intro.progress#def:concurrent_forward_progress_guarantees).

> [Note 6: This does not specify a requirement for when to start this thread of execution, which will typically be specified by the entity that creates this thread of execution. For example, a thread of execution that provides concurrent forward progress guarantees and executes tasks from a set of tasks in an arbitrary order, one after the other, satisfies the requirements of parallel forward progress for these tasks. — end note]


The CUDA C++ Programming Language is an extension of the C++ Programming Language. This section documents the modifications and extensions to the [[intro.progress]](https://eel.is/c++draft/intro.progress) section of the current [ISO International Standard ISO/IEC 14882 – Programming Language C++](https://eel.is/c++draft/) draft. Modified sections are called out explicitly and their diff is shown in **bold**. All other sections are additions.

## 5.8.1. Host threads

The forward progress provided by threads of execution created by the host implementation to execute [main](https://en.cppreference.com/w/cpp/language/main_function), [std::thread](https://en.cppreference.com/w/cpp/thread/thread), and [std::jthread](https://en.cppreference.com/w/cpp/thread/jthread) is implementation-defined behavior of the host implementation [[intro.progress]](https://eel.is/c++draft/intro.progress). General-purpose host implementations should provide concurrent forward progress.

If the host implementation provides [concurrent forward progress [intro.progress.7]](https://eel.is/c++draft/intro.progress#7), then CUDA C++ provides [parallel forward progress [intro.progress.9]](https://eel.is/c++draft/intro.progress#9) for device threads.

## 5.8.2. Device threads

Once a device thread makes progress:

  * If it is part of a [Cooperative Grid](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html#group__CUDART__EXECUTION_1g504b94170f83285c71031be6d5d15f73), all device threads in its grid shall eventually make progress.

  * Otherwise, all device threads in its [thread-block cluster](../02-basics/intro-to-cuda-cpp.html#thread-block-clusters) shall eventually make progress.

> [Note: Threads in other thread-block clusters are not guaranteed to eventually make progress. - end note.]
> 
> [Note: This implies that all device threads within its thread block shall eventually make progress. - end note.]


Modify [[intro.progress.1]](https://eel.is/c++draft/intro.progress#1) as follows (modifications in **bold**):

The implementation may assume that any **host** thread will eventually do one of the following:

>   1. terminate,
> 
>   2. invoke the function [std::this_thread::yield](https://en.cppreference.com/w/cpp/thread/yield) ([[thread.thread.this]](http://eel.is/c++draft/thread.thread.this)),
> 
>   3. make a call to a library I/O function,
> 
>   4. perform an access through a volatile glvalue,
> 
>   5. perform a synchronization operation or an atomic operation, or
> 
>   6. continue execution of a trivial infinite loop ([[stmt.iter.general]](http://eel.is/c++draft/stmt.iter.general)).
> 
> 


**The implementation may assume that any device thread will eventually do one of the following:**

>   1. **terminate** ,
> 
>   2. **make a call to a library I/O function** ,
> 
>   3. **perform an access through a volatile glvalue except if the designated object has automatic storage duration, or**
> 
>   4. **perform a synchronization operation or an atomic read operation except if the designated object has automatic storage duration.**
> 
> 

> 
> [Note: Some current limitations of device threads relative to host threads are implementation defects known to us, that we may fix over time. Examples include the undefined behavior that arises from device threads that eventually only perform volatile or atomic operations on automatic storage duration objects. However, other limitations of device threads relative to host threads are intentional choices. They enable performance optimizations that would not be possible if device threads followed the C++ Standard strictly. For example, providing forward progress to programs that eventually only perform atomic writes or fences would degrade overall performance for little practical benefit. - end note.]

Examples of forward progress guarantee differences between host and device threads due to modifications to [[intro.progress.1]](https://eel.is/c++draft/intro.progress#1).

The following examples refer to the itemized sub-clauses of the implementation assumptions for host and device threads above using “host.threads.<id>” and “device.threads.<id>”, respectively.
    
    
    1// Example: Execution.Model.Device.0
    2// Outcome: grid eventually terminates per device.threads.4 because the atomic object does not have automatic storage duration.
    3__global__ void ex0(cuda::atomic_ref<int, cuda::thread_scope_device> atom) {
    4    if (threadIdx.x == 0) {
    5        while(atom.load(cuda::memory_order_relaxed) == 0);
    6    } else if (threadIdx.x == 1) {
    7        atom.store(1, cuda::memory_order_relaxed);
    8    }
    9}
    
    
    
    1// Example: Execution.Model.Device.1
    2// Allowed outcome: No thread makes progress because device threads don't support host.threads.2.
    3__global__ void ex1() {
    4    while(true) cuda::std::this_thread::yield();
    5}
    
    
    
    1// Example: Execution.Model.Device.2
    2// Allowed outcome: No thread makes progress because device threads don't support host.threads.4
    3// for objects with automatic storage duration (see exception in device.threads.3).
    4__global__ void ex2() {
    5    volatile bool True = true;
    6    while(True);
    7}
    
    
    
    1// Example: Execution.Model.Device.3
    2// Allowed outcome: No thread makes progress because device threads don't support host.threads.5
    3// for objects with automatic storage duration (see exception in device.threads.4).
    4__global__ void ex3() {
    5    cuda::atomic<bool, cuda::thread_scope_thread> True = true;
    6    while(True.load());
    7}
    
    
    
    1// Example: Execution.Model.Device.4
    2// Allowed outcome: No thread makes progress because device threads don't support host.thread.6.
    3__global void ex4() {
    4    while(true) { /* empty */ }
    5}
    

## 5.8.3. CUDA APIs

A CUDA API call shall eventually either return or ensure at least one device thread makes progress.

CUDA query functions (e.g. [cudaStreamQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g2021adeb17905c7ec2a3c1bf125c5435), [cudaEventQuery](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g2bf738909b4a059023537eaa29d8a5b7), etc.) shall not consistently return `cudaErrorNotReady` without a device thread making progress.

> [Note: The device thread need not be “related” to the API call, e.g., an API operating on one stream or process may ensure progress of a device thread on another stream or process. - end note.]
> 
> [Note: A simple but not sufficient method to test a program for CUDA API Forward Progress conformance is to run them with following environment variables set: `CUDA_DEVICE_MAX_CONNECTIONS=1 CUDA_LAUNCH_BLOCKING=1`, and then check that the program still terminates. If it does not, the program has a bug. This method is not sufficient because it does not catch all Forward Progress bugs, but it does catch many such bugs. - end note.]

Examples of CUDA API forward progress guarantees.
    
    
     1// Example: Execution.Model.API.1
     2// Outcome: if no other device threads (e.g., from other processes) are making progress,
     3// this program terminates and returns cudaSuccess.
     4// Rationale: CUDA guarantees that if the device is empty:
     5// - `cudaDeviceSynchronize` eventually ensures that at least one device-thread makes progress, which implies that eventually `hello_world` grid and one of its device-threads start.
     6// - All thread-block threads eventually start (due to "if a device thread makes progress, all other threads in its thread-block cluster eventually make progress").
     7// - Once all threads in thread-block arrive at `__syncthreads` barrier, all waiting threads are unblocked.
     8// - Therefore all device threads eventually exit the `hello_world`` grid.
     9// - And `cudaDeviceSynchronize`` eventually unblocks.
    10__global__ void hello_world() { __syncthreads(); }
    11int main() {
    12    hello_world<<<1,2>>>();
    13    return (int)cudaDeviceSynchronize();
    14}
    
    
    
     1// Example: Execution.Model.API.2
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: the `cudaDeviceSynchronize` API below is only called if a device thread eventually makes progress and sets the flag.
     4// However, CUDA only guarantees that `producer` device thread eventually starts if the synchronization API is called.
     5// Therefore, the host thread may never be unblocked from the flag spin-loop.
     6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     7__global__ void producer() { flag.store(1); }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    producer<<<1,1>>>();
    11    while (flag.load() == 0);
    12    return cudaDeviceSynchronize();
    13}
    
    
    
     1// Example: Execution.Model.API.3
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: same as Example.Model.API.2, with the addition that a single CUDA query API call does not guarantee
     4// the device thread eventually starts, only repeated CUDA query API calls do (see Execution.Model.API.4).
     5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     6__global__ void producer() { flag.store(1); }
     7int main() {
     8    cudaHostRegister(&flag, sizeof(flag));
     9    producer<<<1,1>>>();
    10    (void)cudaStreamQuery(0);
    11    while (flag.load() == 0);
    12    return cudaDeviceSynchronize();
    13}
    
    
    
     1// Example: Execution.Model.API.4
     2// Outcome: terminates.
     3// Rationale: same as Execution.Model.API.3, but this example repeatedly calls
     4// a CUDA query API in within the flag spin-loop, which guarantees that the device thread
     5// eventually makes progress.
     6cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     7__global__ void producer() { flag.store(1); }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    producer<<<1,1>>>();
    11    while (flag.load() == 0) {
    12        (void)cudaStreamQuery(0);
    13    }
    14    return cudaDeviceSynchronize();
    15}
    

### 5.8.3.1. Dependencies

A device thread shall not start until all its dependencies have completed.

> [Note: Dependencies that prevent device threads from starting to make progress can be created, for example, via [CUDA Stream Commands](../02-basics/asynchronous-execution.html#cuda-streams). These may include dependencies on the completion of, among others, [CUDA Events](../02-basics/asynchronous-execution.html#cuda-events) and [CUDA Kernels](../02-basics/intro-to-cuda-cpp.html#kernels). - end note.]

Examples of CUDA API forward progress guarantees due to dependencies
    
    
     1// Example: Execution.Model.Stream.0
     2// Allowed outcome: eventually, no thread makes progress.
     3// Rationale: while CUDA guarantees that one device thread makes progress, since there
     4// is no dependency between `first` and `second`, it does not guarantee which thread,
     5// and therefore it could always pick the device thread from `second`, which then never
     6// unblocks from the spin-loop.
     7// That is, `second` may starve `first`.
     8cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     9__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
    10__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
    11int main() {
    12    cudaHostRegister(&flag, sizeof(flag));
    13    cudaStream_t s0, s1;
    14    cudaStreamCreate(&s0);
    15    cudaStreamCreate(&s1);
    16    first<<<1,1,0,s0>>>();
    17    second<<<1,1,0,s1>>>();
    18    return cudaDeviceSynchronize();
    19}
    
    
    
     1// Example: Execution.Model.Stream.1
     2// Outcome: terminates.
     3// Rationale: same as Execution.Model.Stream.0, but this example has a stream dependency
     4// between first and second, which requires CUDA to run the grids in order.
     5cuda::atomic<int, cuda::thread_scope_system> flag = 0;
     6__global__ void first() { flag.store(1, cuda::memory_order_relaxed); }
     7__global__ void second() { while(flag.load(cuda::memory_order_relaxed) == 0) {} }
     8int main() {
     9    cudaHostRegister(&flag, sizeof(flag));
    10    cudaStream_t s0;
    11    cudaStreamCreate(&s0);
    12    first<<<1,1,0,s0>>>();
    13    second<<<1,1,0,s0>>>();
    14    return cudaDeviceSynchronize();
    15}
