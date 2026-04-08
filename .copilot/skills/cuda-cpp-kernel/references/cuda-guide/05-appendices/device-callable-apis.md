---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/device-callable-apis.html
---

# 5.6. Device-Callable APIs and Intrinsics

This chapter contains reference material and API documentation for APIs and intrinsics which can be called from CUDA kernels and device code.

## 5.6.1. Memory Barrier Primitives Interface

The primitives API is a C-like interface to `cuda::barrier` functionality. These primitives are available by including the `<cuda_awbarrier_primitives.h>` header.

### 5.6.1.1. Data Types
    
    
    typedef /* implementation defined */ __mbarrier_t;
    typedef /* implementation defined */ __mbarrier_token_t;
    

### 5.6.1.2. Memory Barrier Primitives API
    
    
    uint32_t __mbarrier_maximum_count();
    void __mbarrier_init(__mbarrier_t* bar, uint32_t expected_count);
    

  * `bar` must be a pointer to `__shared__` memory.

  * `expected_count <= __mbarrier_maximum_count()`

  * Initialize `*bar` expected arrival count for the current and next phase to `expected_count`.


    
    
    void __mbarrier_inval(__mbarrier_t* bar);
    

  * `bar` must be a pointer to the barrier object residing in shared memory.

  * Invalidation of `*bar` is required before the corresponding shared memory can be repurposed.


    
    
    __mbarrier_token_t __mbarrier_arrive(__mbarrier_t* bar);
    

  * Initialization of `*bar` must happen before this call.

  * Pending count must not be zero.

  * Atomically decrement the pending count for the current phase of the barrier.

  * Return an arrival token associated with the barrier state immediately prior to the decrement.


    
    
    __mbarrier_token_t __mbarrier_arrive_and_drop(__mbarrier_t* bar);
    

  * Initialization of `*bar` must happen before this call.

  * Pending count must not be zero.

  * Atomically decrement the pending count for the current phase and expected count for the next phase of the barrier.

  * Return an arrival token associated with the barrier state immediately prior to the decrement.


    
    
    bool __mbarrier_test_wait(__mbarrier_t* bar, __mbarrier_token_t token);
    

  * `token` must be associated with the immediately preceding phase or current phase of `*bar`.

  * Returns `true` if `token` is associated with the immediately preceding phase of `*bar`, otherwise returns `false`.


    
    
    bool __mbarrier_test_wait_parity(__mbarrier_t* bar, bool phase_parity);
    

  * `phase_parity` must indicate the parity of either the current phase or the immediately preceding phase of `*bar`. A value of `true` corresponds to odd-numbered phases and a value of `false` corresponds to even-numbered phases.

  * Returns `true` if `phase_parity` indicates the integer parity of the immediately preceding phase of `*bar`, otherwise returns `false`.


    
    
    bool __mbarrier_try_wait(__mbarrier_t* bar, __mbarrier_token_t token, uint32_t max_sleep_nanosec);
    

  * `token` must be associated with the immediately preceding phase or current phase of `*bar`.

  * Returns `true` if `token` is associated with the immediately preceding phase of `*bar`. Otherwise, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes (returns `true`) OR before the phase completes following a system-dependent time limit (returns `false`).

  * `max_sleep_nanosec` specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit.


    
    
    bool __mbarrier_try_wait_parity(__mbarrier_t* bar, bool phase_parity, uint32_t max_sleep_nanosec);
    

  * `phase_parity` must indicate the parity of either the current phase or the immediately preceding phase of `*bar`. A value of `true` corresponds to odd-numbered phases and a value of `false` corresponds to even-numbered phases.

  * Returns `true` if `phase_parity` indicates the integer parity of the immediately preceding phase of `*bar`. Otherwise, the executing thread may be suspended. Suspended thread resumes execution when the specified phase completes (returns `true`) OR before the phase completes following a system-dependent time limit (returns `false`).

  * `max_sleep_nanosec` specifies the time limit, in nanoseconds, that may be used for the time limit instead of the system-dependent limit.


## 5.6.2. Pipeline Primitives Interface

Pipeline primitives provide a C-like interface for the functionality available in `<cuda/pipeline>`. The pipeline primitives interface is available by including the `<cuda_pipeline.h>` header. When compiling without ISO C++ 2011 compatibility, include the `<cuda_pipeline_primitives.h>` header.

Note

The pipeline primitives API only supports tracking asynchronous copies from global memory to shared memory with specific size and alignment requirements. It provides equivalent functionality to a `cuda::pipeline` object with `cuda::thread_scope_thread`.

### 5.6.2.1. `memcpy_async` Primitive
    
    
    void __pipeline_memcpy_async(void* __restrict__ dst_shared,
                                 const void* __restrict__ src_global,
                                 size_t size_and_align,
                                 size_t zfill=0);
    

  * Request that the following operation be submitted for asynchronous evaluation:
        
        size_t i = 0;
        for (; i < size_and_align - zfill; ++i) ((char*)dst_shared)[i] = ((char*)src_global)[i]; /* copy */
        for (; i < size_and_align; ++i) ((char*)dst_shared)[i] = 0; /* zero-fill */
        

  * Requirements:

    * `dst_shared` must be a pointer to the shared memory destination for the `memcpy_async`.

    * `src_global` must be a pointer to the global memory source for the `memcpy_async`.

    * `size_and_align` must be 4, 8, or 16.

    * `zfill <= size_and_align`.

    * `size_and_align` must be the alignment of `dst_shared` and `src_global`.

  * It is a race condition for any thread to modify the source memory or observe the destination memory prior to waiting for the `memcpy_async` operation to complete. Between submitting a `memcpy_async` operation and waiting for its completion, any of the following actions introduces a race condition:

    * Loading from `dst_shared`.

    * Storing to `dst_shared` or `src_global`.

    * Applying an atomic update to `dst_shared` or `src_global`.


### 5.6.2.2. Commit Primitive
    
    
    void __pipeline_commit();
    

  * Commit submitted `memcpy_async` to the pipeline as the current batch.


### 5.6.2.3. Wait Primitive
    
    
    void __pipeline_wait_prior(size_t N);
    

  * Let `{0, 1, 2, ..., L}` be the sequence of indices associated with invocations of `__pipeline_commit()` by a given thread.

  * Wait for completion of batches _at least_ up to and including `L-N`.


### 5.6.2.4. Arrive On Barrier Primitive
    
    
    void __pipeline_arrive_on(__mbarrier_t* bar);
    

  * `bar` points to a barrier in shared memory.

  * Increments the barrier arrival count by one, when all memcpy_async operations sequenced before this call have completed, the arrival count is decremented by one and hence the net effect on the arrival count is zero. It is user’s responsibility to make sure that the increment on the arrival count does not exceed `__mbarrier_maximum_count()`.


## 5.6.3. Cooperative Groups API

### 5.6.3.1. cooperative_groups.h

#### 5.6.3.1.1. class thread_block

Any CUDA programmer is already familiar with a certain group of threads: the thread block. The Cooperative Groups extension introduces a new datatype, `thread_block`, to explicitly represent this concept within the kernel.

`class thread_block;`

Constructed via:
    
    
    thread_block g = this_thread_block();
    

**Public Member Functions:**

`static void sync()`: Synchronize the threads named in the group, equivalent to `g.barrier_wait(g.barrier_arrive())`

`thread_block::arrival_token barrier_arrive()`: Arrive on the thread_block barrier, returns a token that needs to be passed into `barrier_wait()`.

`void barrier_wait(thread_block::arrival_token&& t)`: Wait on the `thread_block` barrier, takes arrival token returned from `barrier_arrive()` as an rvalue reference.

`static unsigned int thread_rank()`: Rank of the calling thread within [0, num_threads)

`static dim3 group_index()`: 3-Dimensional index of the block within the launched grid

`static dim3 thread_index()`: 3-Dimensional index of the thread within the launched block

`static dim3 dim_threads()`: Dimensions of the launched block in units of threads

`static unsigned int num_threads()`: Total number of threads in the group

Legacy member functions (aliases):

`static unsigned int size()`: Total number of threads in the group (alias of `num_threads()`)

`static dim3 group_dim()`: Dimensions of the launched block (alias of `dim_threads()`)

**Example:**
    
    
    /// Loading an integer from global into shared memory
    __global__ void kernel(int *globalInput) {
        __shared__ int x;
        thread_block g = this_thread_block();
        // Choose a leader in the thread block
        if (g.thread_rank() == 0) {
            // load from global into shared for all threads to work with
            x = (*globalInput);
        }
        // After loading data into shared memory, you want to synchronize
        // if all threads in your thread block need to see it
        g.sync(); // equivalent to __syncthreads();
    }
    

**Note:** that all threads in the group must participate in collective operations, or the behavior is undefined.

**Related:** The `thread_block` datatype is derived from the more generic `thread_group` datatype, which can be used to represent a wider class of groups.

#### 5.6.3.1.2. class cluster_group

This group object represents all the threads launched in a single cluster. The APIs are available on all hardware with Compute Capability 9.0+. In such cases, when a non-cluster grid is launched, the APIs assume a 1x1x1 cluster.

`class cluster_group;`

Constructed via:
    
    
    cluster_group g = this_cluster();
    

**Public Member Functions:**

`static void sync()`: Synchronize the threads named in the group, equivalent to `g.barrier_wait(g.barrier_arrive())`

`static cluster_group::arrival_token barrier_arrive()`: Arrive on the cluster barrier, returns a token that needs to be passed into `barrier_wait()`.

`static void barrier_wait(cluster_group::arrival_token&& t)`: Wait on the cluster barrier, takes arrival token returned from `barrier_arrive()` as a rvalue reference.

`static unsigned int thread_rank()`: Rank of the calling thread within [0, num_threads)

`static unsigned int block_rank()`: Rank of the calling block within [0, num_blocks)

`static unsigned int num_threads()`: Total number of threads in the group

`static unsigned int num_blocks()`: Total number of blocks in the group

`static dim3 dim_threads()`: Dimensions of the launched cluster in units of threads

`static dim3 dim_blocks()`: Dimensions of the launched cluster in units of blocks

`static dim3 block_index()`: 3-Dimensional index of the calling block within the launched cluster

`static unsigned int query_shared_rank(const void *addr)`: Obtain the block rank to which a shared memory address belongs

`static T* map_shared_rank(T *addr, int rank)`: Obtain the address of a shared memory variable of another block in the cluster

Legacy member functions (aliases):

`static unsigned int size()`: Total number of threads in the group (alias of `num_threads()`)

#### 5.6.3.1.3. class grid_group

This group object represents all the threads launched in a single grid. APIs other than `sync()` are available at all times, but to be able to synchronize across the grid, you need to use the cooperative launch API.

`class grid_group;`

Constructed via:
    
    
    grid_group g = this_grid();
    

**Public Member Functions:**

`bool is_valid() const`: Returns whether the grid_group can synchronize

`void sync() const`: Synchronize the threads named in the group, equivalent to `g.barrier_wait(g.barrier_arrive())`

`grid_group::arrival_token barrier_arrive()`: Arrive on the grid barrier, returns a token that needs to be passed into `barrier_wait()`.

`void barrier_wait(grid_group::arrival_token&& t)`: Wait on the grid barrier, takes arrival token returned from `barrier_arrive()` as a rvalue reference.

`static unsigned long long thread_rank()`: Rank of the calling thread within [0, num_threads)

`static unsigned long long block_rank()`: Rank of the calling block within [0, num_blocks)

`static unsigned long long cluster_rank()`: Rank of the calling cluster within [0, num_clusters)

`static unsigned long long num_threads()`: Total number of threads in the group

`static unsigned long long num_blocks()`: Total number of blocks in the group

`static unsigned long long num_clusters()`: Total number of clusters in the group

`static dim3 dim_blocks()`: Dimensions of the launched grid in units of blocks

`static dim3 dim_clusters()`: Dimensions of the launched grid in units of clusters

`static dim3 block_index()`: 3-Dimensional index of the block within the launched grid

`static dim3 cluster_index()`: 3-Dimensional index of the cluster within the launched grid

Legacy member functions (aliases):

`static unsigned long long size()`: Total number of threads in the group (alias of `num_threads()`)

`static dim3 group_dim()`: Dimensions of the launched grid (alias of `dim_blocks()`)

#### 5.6.3.1.4. class thread_block_tile

A templated version of a tiled group, where a template parameter is used to specify the size of the tile - with this known at compile time there is the potential for more optimal execution.
    
    
    template <unsigned int Size, typename ParentT = void>
    class thread_block_tile;
    

Constructed via:
    
    
    template <unsigned int Size, typename ParentT>
    _CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
    

`Size` must be a power of 2 and less than or equal to 1024. Notes section describes extra steps needed to create tiles of size larger than 32 on hardware with Compute Capability 7.5 or lower.

`ParentT` is the parent-type from which this group was partitioned. It is automatically inferred, but a value of void will store this information in the group handle rather than in the type.

**Public Member Functions:**

`void sync() const`: Synchronize the threads named in the group

`unsigned long long num_threads() const`: Total number of threads in the group

`unsigned long long thread_rank() const`: Rank of the calling thread within [0, num_threads)

`unsigned long long meta_group_size() const`: Returns the number of groups created when the parent group was partitioned.

`unsigned long long meta_group_rank() const`: Linear rank of the group within the set of tiles partitioned from a parent group (bounded by meta_group_size)

`T shfl(T var, unsigned int src_rank) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions), **Note: For sizes larger than 32 all threads in the group have to specify the same src_rank, otherwise the behavior is undefined.**

`T shfl_up(T var, int delta) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions), available only for sizes lower or equal to 32.

`T shfl_down(T var, int delta) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions), available only for sizes lower or equal to 32.

`T shfl_xor(T var, int delta) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions), available only for sizes lower or equal to 32.

`int any(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions)

`int all(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions), available only for sizes lower or equal to 32.

`unsigned int match_any(T val) const`: Refer to [Warp Match Functions](cpp-language-extensions.html#warp-match-functions), available only for sizes lower or equal to 32.

`unsigned int match_all(T val, int &pred) const`: Refer to [Warp Match Functions](cpp-language-extensions.html#warp-match-functions), available only for sizes lower or equal to 32.

Legacy member functions (aliases):

`unsigned long long size() const`: Total number of threads in the group (alias of `num_threads()`)

**Notes:**

  * `thread_block_tile` templated data structure is being used here, the size of the group is passed to the `tiled_partition` call as a template parameter rather than an argument.

  * `shfl, shfl_up, shfl_down, and shfl_xor` functions accept objects of any type when compiled with C++11 or later. This means it’s possible to shuffle non-integral types as long as they satisfy the below constraints:

    * Qualifies as trivially copyable i.e., `is_trivially_copyable<T>::value == true`

    * `sizeof(T) <= 32` for tile sizes lower or equal 32, `sizeof(T) <= 8` for larger tiles

  * On hardware with Compute Capability 7.5 or lower tiles of size larger than 32 need small amount of memory reserved for them. This can be done using `cooperative_groups::block_tile_memory` struct template that has to reside in either shared or global memory.
        
        template <unsigned int MaxBlockSize = 1024>
        struct block_tile_memory;
        

`MaxBlockSize` Specifies the maximal number of threads in the current thread block. This parameter can be used to minimize the shared memory usage of `block_tile_memory` in kernels launched only with smaller thread counts.

This `block_tile_memory` needs be then passed into `cooperative_groups::this_thread_block`, allowing the resulting `thread_block` to be partitioned into tiles of sizes larger than 32. Overload of `this_thread_block` accepting `block_tile_memory` argument is a collective operation and has to be called with all threads in the `thread_block`.

`block_tile_memory` can be used on hardware with Compute Capability 8.0 or higher in order to be able to write one source targeting multiple different Compute Capabilities. It should consume no memory when instantiated in shared memory in cases where its not required.


**Examples:**
    
    
    /// The following code will create two sets of tiled groups, of size 32 and 4 respectively:
    /// The latter has the provenance encoded in the type, while the first stores it in the handle
    thread_block block = this_thread_block();
    thread_block_tile<32> tile32 = tiled_partition<32>(block);
    thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
    
    
    
    /// The following code will create tiles of size 128 on all Compute Capabilities.
    /// block_tile_memory can be omitted on Compute Capability 8.0 or higher.
    __global__ void kernel(...) {
        // reserve shared memory for thread_block_tile usage,
        //   specify that block size will be at most 256 threads.
        __shared__ block_tile_memory<256> shared;
        thread_block thb = this_thread_block(shared);
    
        // Create tiles with 128 threads.
        auto tile = tiled_partition<128>(thb);
    
        // ...
    }
    

#### 5.6.3.1.5. class coalesced_group

In CUDA’s SIMT architecture, at the hardware level the multiprocessor executes threads in groups of 32 called warps. If there exists a data-dependent conditional branch in the application code such that threads within a warp diverge, then the warp serially executes each branch disabling threads not on that path. The threads that remain active on the path are referred to as coalesced. Cooperative Groups has functionality to discover, and create, a group containing all coalesced threads.

Constructing the group handle via `coalesced_threads()` is opportunistic. It returns the set of active threads at that point in time, and makes no guarantee about which threads are returned (as long as they are active) or that they will stay coalesced throughout execution (they will be brought back together for the execution of a collective but can diverge again afterwards).

`class coalesced_group;`

Constructed via:
    
    
    coalesced_group active = coalesced_threads();
    

**Public Member Functions:**

`void sync() const`: Synchronize the threads named in the group

`unsigned long long num_threads() const`: Total number of threads in the group

`unsigned long long thread_rank() const`: Rank of the calling thread within [0, num_threads)

`unsigned long long meta_group_size() const`: Returns the number of groups created when the parent group was partitioned. If this group was created by querying the set of active threads, for example `coalesced_threads()` the value of `meta_group_size()` will be 1.

`unsigned long long meta_group_rank() const`: Linear rank of the group within the set of tiles partitioned from a parent group (bounded by meta_group_size). If this group was created by querying the set of active threads, e.g. `coalesced_threads()` the value of `meta_group_rank()` will always be 0.

`T shfl(T var, unsigned int src_rank) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`T shfl_up(T var, int delta) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`T shfl_down(T var, int delta) const`: Refer to [Warp Shuffle Functions](cpp-language-extensions.html#warp-shuffle-functions)

`int any(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions)

`int all(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions)

`unsigned int ballot(int predicate) const`: Refer to [Warp Vote Functions](index.html#warp-vote-functions)

`unsigned int match_any(T val) const`: Refer to [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)

`unsigned int match_all(T val, int &pred) const`: Refer to [Warp Match Functions](cpp-language-extensions.html#warp-match-functions)

Legacy member functions (aliases):

`unsigned long long size() const`: Total number of threads in the group (alias of `num_threads()`)

**Notes:**

`shfl, shfl_up, and shfl_down` functions accept objects of any type when compiled with C++11 or later. This means it’s possible to shuffle non-integral types as long as they satisfy the below constraints:

  * Qualifies as trivially copyable i.e. `is_trivially_copyable<T>::value == true`

  * `sizeof(T) <= 32`


**Example:**
    
    
    /// Consider a situation whereby there is a branch in the
    /// code in which only the 2nd, 4th and 8th threads in each warp are
    /// active. The coalesced_threads() call, placed in that branch, will create (for each
    /// warp) a group, active, that has three threads (with
    /// ranks 0-2 inclusive).
    __global__ void kernel(int *globalInput) {
        // Lets say globalInput says that threads 2, 4, 8 should handle the data
        if (threadIdx.x == *globalInput) {
            coalesced_group active = coalesced_threads();
            // active contains 0-2 inclusive
            active.sync();
        }
    }
    

### 5.6.3.2. cooperative_groups/async.h

#### 5.6.3.2.1. `memcpy_async`

`memcpy_async` is a group-wide collective memcpy that utilizes hardware accelerated support for non-blocking memory transactions from global to shared memory. Given a set of threads named in the group, `memcpy_async` will move specified amount of bytes or elements of the input type through a single pipeline stage. Additionally for achieving best performance when using the `memcpy_async` API, an alignment of 16 bytes for both shared memory and global memory is required. It is important to note that while this is a memcpy in the general case, it is only asynchronous if the source is global memory and the destination is shared memory and both can be addressed with 16, 8, or 4 byte alignments. Asynchronously copied data should only be read following a call to wait or wait_prior which signals that the corresponding stage has completed moving data to shared memory.

Having to wait on all outstanding requests can lose some flexibility (but gain simplicity). In order to efficiently overlap data transfer and execution, its important to be able to kick off an **N+1**` memcpy_async` request while waiting on and operating on request **N**. To do so, use `memcpy_async` and wait on it using the collective stage-based `wait_prior` API. See [wait and wait_prior](#cg-api-async-wait) for more details.

Usage 1
    
    
    template <typename TyGroup, typename TyElem, typename TyShape>
    void memcpy_async(
      const TyGroup &group,
      TyElem *__restrict__ _dst,
      const TyElem *__restrict__ _src,
      const TyShape &shape
    );
    

Performs a copy of **``shape`` bytes**.

Usage 2
    
    
    template <typename TyGroup, typename TyElem, typename TyDstLayout, typename TySrcLayout>
    void memcpy_async(
      const TyGroup &group,
      TyElem *__restrict__ dst,
      const TyDstLayout &dstLayout,
      const TyElem *__restrict__ src,
      const TySrcLayout &srcLayout
    );
    

Performs a copy of **``min(dstLayout, srcLayout)`` elements**. If layouts are of type `cuda::aligned_size_t<N>`, both must specify the same alignment.

**Errata** The `memcpy_async` API introduced in CUDA 11.1 with both src and dst input layouts, expects the layout to be provided in elements rather than bytes. The element type is inferred from `TyElem` and has the size `sizeof(TyElem)`. If `cuda::aligned_size_t<N>` type is used as the layout, the number of elements specified times `sizeof(TyElem)` must be a multiple of N and it is recommended to use `std::byte` or `char` as the element type.

If specified shape or layout of the copy is of type `cuda::aligned_size_t<N>`, alignment will be guaranteed to be at least `min(16, N)`. In that case both `dst` and `src` pointers need to be aligned to N bytes and the number of bytes copied needs to be a multiple of N.

**Codegen Requirements:** Compute Capability 5.0 minimum, Compute Capability 8.0 for asynchronicity, C++11

`cooperative_groups/memcpy_async.h` header needs to be included.

**Example:**
    
    
    /// This example streams elementsPerThreadBlock worth of data from global memory
    /// into a limited sized shared memory (elementsInShared) block to operate on.
    #include <cooperative_groups.h>
    #include <cooperative_groups/memcpy_async.h>
    
    namespace cg = cooperative_groups;
    
    __global__ void kernel(int* global_data) {
        cg::thread_block tb = cg::this_thread_block();
        const size_t elementsPerThreadBlock = 16 * 1024;
        const size_t elementsInShared = 128;
        __shared__ int local_smem[elementsInShared];
    
        size_t copy_count;
        size_t index = 0;
        while (index < elementsPerThreadBlock) {
            cg::memcpy_async(tb, local_smem, elementsInShared, global_data + index, elementsPerThreadBlock - index);
            copy_count = min(elementsInShared, elementsPerThreadBlock - index);
            cg::wait(tb);
            // Work with local_smem
            index += copy_count;
        }
    }
    

#### 5.6.3.2.2. `wait` and `wait_prior`
    
    
    template <typename TyGroup>
    void wait(TyGroup & group);
    
    template <unsigned int NumStages, typename TyGroup>
    void wait_prior(TyGroup & group);
    

`wait` and `wait_prior` collectives allow to wait for memcpy_async copies to complete. `wait` blocks calling threads until all previous copies are done. `wait_prior` allows that the latest NumStages are still not done and waits for all the previous requests. So with `N` total copies requested, it waits until the first `N-NumStages` are done and the last `NumStages` might still be in progress. Both `wait` and `wait_prior` will synchronize the named group.

**Codegen Requirements:** Compute Capability 5.0 minimum, Compute Capability 8.0 for asynchronicity, C++11

`cooperative_groups/memcpy_async.h` header needs to be included.

**Example:**
    
    
    /// This example streams elementsPerThreadBlock worth of data from global memory
    /// into a limited sized shared memory (elementsInShared) block to operate on in
    /// multiple (two) stages. As stage N is kicked off, we can wait on and operate on stage N-1.
    #include <cooperative_groups.h>
    #include <cooperative_groups/memcpy_async.h>
    
    namespace cg = cooperative_groups;
    
    __global__ void kernel(int* global_data) {
        cg::thread_block tb = cg::this_thread_block();
        const size_t elementsPerThreadBlock = 16 * 1024 + 64;
        const size_t elementsInShared = 128;
        __align__(16) __shared__ int local_smem[2][elementsInShared];
        int stage = 0;
        // First kick off an extra request
        size_t copy_count = elementsInShared;
        size_t index = copy_count;
        cg::memcpy_async(tb, local_smem[stage], elementsInShared, global_data, elementsPerThreadBlock - index);
        while (index < elementsPerThreadBlock) {
            // Now we kick off the next request...
            cg::memcpy_async(tb, local_smem[stage ^ 1], elementsInShared, global_data + index, elementsPerThreadBlock - index);
            // ... but we wait on the one before it
            cg::wait_prior<1>(tb);
    
            // Its now available and we can work with local_smem[stage] here
            // (...)
            //
    
            // Calculate the amount fo data that was actually copied, for the next iteration.
            copy_count = min(elementsInShared, elementsPerThreadBlock - index);
            index += copy_count;
    
            // A cg::sync(tb) might be needed here depending on whether
            // the work done with local_smem[stage] can release threads to race ahead or not
            // Wrap to the next stage
            stage ^= 1;
        }
        cg::wait(tb);
        // The last local_smem[stage] can be handled here
    }
    

### 5.6.3.3. cooperative_groups/partition.h

#### 5.6.3.3.1. `tiled_partition`
    
    
    template <unsigned int Size, typename ParentT>
    thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g);
    
    
    
    thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
    

The `tiled_partition` method is a collective operation that partitions the parent group into a one-dimensional, row-major, tiling of subgroups. A total of ((size(parent)/tilesz) subgroups will be created, therefore the parent group size must be evenly divisible by the `Size`. The allowed parent groups are `thread_block` or `thread_block_tile`.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution. Functionality is limited to native hardware sizes, 1/2/4/8/16/32 and the `cg::size(parent)` must be greater than the `Size` parameter. The templated version of `tiled_partition` supports 64/128/256/512 sizes as well, but some additional steps are required on Compute Capability 7.5 or lower, refer to [class thread_block_tile](#cg-api-thread-block-tile) for details.

**Codegen Requirements:** Compute Capability 5.0 minimum, C++11 for sizes larger than 32

#### 5.6.3.3.2. `labeled_partition`
    
    
    template <typename Label>
    coalesced_group labeled_partition(const coalesced_group& g, Label label);
    
    
    
    template <unsigned int Size, typename Label>
    coalesced_group labeled_partition(const thread_block_tile<Size>& g, Label label);
    

The `labeled_partition` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a condition label and assign threads that have the same value for label into the same group.

`Label` can be any integral type.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Note:** This functionality is still being evaluated and may slightly change in the future.

**Codegen Requirements:** Compute Capability 7.0 minimum, C++11

#### 5.6.3.3.3. `binary_partition`
    
    
    coalesced_group binary_partition(const coalesced_group& g, bool pred);
    
    
    
    template <unsigned int Size>
    coalesced_group binary_partition(const thread_block_tile<Size>& g, bool pred);
    

The `binary_partition()` method is a collective operation that partitions the parent group into one-dimensional subgroups within which the threads are coalesced. The implementation will evaluate a predicate and assign threads that have the same value into the same group. This is a specialized form of `labeled_partition()`, where the label can only be 0 or 1.

The implementation may cause the calling thread to wait until all the members of the parent group have invoked the operation before resuming execution.

**Example:**
    
    
    /// This example divides a 32-sized tile into a group with odd
    /// numbers and a group with even numbers
    _global__ void oddEven(int *inputArr) {
        auto block = cg::this_thread_block();
        auto tile32 = cg::tiled_partition<32>(block);
    
        // inputArr contains random integers
        int elem = inputArr[block.thread_rank()];
        // after this, tile32 is split into 2 groups,
        // a subtile where elem&1 is true and one where its false
        auto subtile = cg::binary_partition(tile32, (elem & 1));
    }
    

### 5.6.3.4. cooperative_groups/reduce.h

#### 5.6.3.4.1. `Reduce` Operators

Below are the prototypes of function objects for some of the basic operations that can be done with `reduce`.
    
    
    namespace cooperative_groups {
      template <typename Ty>
      struct cg::plus;
    
      template <typename Ty>
      struct cg::less;
    
      template <typename Ty>
      struct cg::greater;
    
      template <typename Ty>
      struct cg::bit_and;
    
      template <typename Ty>
      struct cg::bit_xor;
    
      template <typename Ty>
      struct cg::bit_or;
    }
    

Reduce is limited to the information available to the implementation at compile time. Thus in order to make use of intrinsics introduced in CC 8.0, the `cg::` namespace exposes several functional objects that mirror the hardware. These objects appear similar to those presented in the C++ STL, with the exception of `less/greater`. The reason for any difference from the STL is that these function objects are designed to actually mirror the operation of the hardware intrinsics.

**Functional description:**

  * `cg::plus:` Accepts two values and returns the sum of both using operator+.

  * `cg::less:` Accepts two values and returns the lesser using operator<. This differs in that the **lower value is returned** rather than a Boolean.

  * `cg::greater:` Accepts two values and returns the greater using operator<. This differs in that the **greater value is returned** rather than a Boolean.

  * `cg::bit_and:` Accepts two values and returns the result of operator&.

  * `cg::bit_xor:` Accepts two values and returns the result of operator^.

  * `cg::bit_or:` Accepts two values and returns the result of operator|.


**Example:**
    
    
    {
        // cg::plus<int> is specialized within cg::reduce and calls __reduce_add_sync(...) on CC 8.0+
        cg::reduce(tile, (int)val, cg::plus<int>());
    
        // cg::plus<float> fails to match with an accelerator and instead performs a standard shuffle based reduction
        cg::reduce(tile, (float)val, cg::plus<float>());
    
        // While individual components of a vector are supported, reduce will not use hardware intrinsics for the following
        // It will also be necessary to define a corresponding operator for vector and any custom types that may be used
        int4 vec = {...};
        cg::reduce(tile, vec, cg::plus<int4>())
    
        // Finally lambdas and other function objects cannot be inspected for dispatch
        // and will instead perform shuffle based reductions using the provided function object.
        cg::reduce(tile, (int)val, [](int l, int r) -> int {return l + r;});
    }
    

#### 5.6.3.4.2. `reduce`
    
    
    template <typename TyGroup, typename TyArg, typename TyOp>
    auto reduce(const TyGroup& group, TyArg&& val, TyOp&& op) -> decltype(op(val, val));
    

`reduce` performs a reduction operation on the data provided by each thread named in the group passed in. This takes advantage of hardware acceleration (on compute 80 and higher devices) for the arithmetic add, min, or max operations and the logical AND, OR, or XOR, as well as providing a software fallback on older generation hardware. Only 4B types are accelerated by hardware.

`group`: Valid group types are `coalesced_group` and `thread_block_tile`.

`val`: Any type that satisfies the below requirements:

  * Qualifies as trivially copyable i.e. `is_trivially_copyable<TyArg>::value == true`

  * `sizeof(T) <= 32` for `coalesced_group` and tiles of size lower or equal 32, `sizeof(T) <= 8` for larger tiles

  * Has suitable arithmetic or comparative operators for the given function object.


**Note:** Different threads in the group can pass different values for this argument.

`op`: Valid function objects that will provide hardware acceleration with integral types are `plus(), less(), greater(), bit_and(), bit_xor(), bit_or()`. These must be constructed, hence the TyVal template argument is required, i.e. `plus<int>()`. Reduce also supports lambdas and other function objects that can be invoked using `operator()`

Asynchronous reduce
    
    
    template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
    void reduce_update_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);
    
    template <typename TyGroup, typename TyArg, typename TyAtomic, typename TyOp>
    void reduce_store_async(const TyGroup& group, TyAtomic& atomic, TyArg&& val, TyOp&& op);
    
    template <typename TyGroup, typename TyArg, typename TyOp>
    void reduce_store_async(const TyGroup& group, TyArg* ptr, TyArg&& val, TyOp&& op);
    

`*_async` variants of the API are asynchronously calculating the result to either store to or update a specified destination by one of the participating threads, instead of returning it by each thread. To observe the effect of these asynchronous calls, calling group of threads or a larger group containing them need to be synchronized.

  * In case of the atomic store or update variant, `atomic` argument can be either of `cuda::atomic` or `cuda::atomic_ref` available in [CUDA C++ Standard Library](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html). This variant of the API is available only on platforms and devices, where these types are supported by the CUDA C++ Standard Library. Result of the reduction is used to atomically update the atomic according to the specified `op`, eg. the result is atomically added to the atomic in case of `cg::plus()`. Type held by the `atomic` must match the type of `TyArg`. Scope of the atomic must include all the threads in the group and if multiple groups are using the same atomic concurrently, scope must include all threads in all groups using it. Atomic update is performed with relaxed memory ordering.

  * In case of the pointer store variant, result of the reduction will be weakly stored into the `dst` pointer.


### 5.6.3.5. cooperative_groups/scan.h

#### 5.6.3.5.1. `inclusive_scan` and `exclusive_scan`
    
    
    template <typename TyGroup, typename TyVal, typename TyFn>
    auto inclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));
    
    template <typename TyGroup, typename TyVal>
    TyVal inclusive_scan(const TyGroup& group, TyVal&& val);
    
    template <typename TyGroup, typename TyVal, typename TyFn>
    auto exclusive_scan(const TyGroup& group, TyVal&& val, TyFn&& op) -> decltype(op(val, val));
    
    template <typename TyGroup, typename TyVal>
    TyVal exclusive_scan(const TyGroup& group, TyVal&& val);
    

`inclusive_scan` and `exclusive_scan` performs a scan operation on the data provided by each thread named in the group passed in. Result for each thread is a reduction of data from threads with lower `thread_rank` than that thread in case of `exclusive_scan`. `inclusive_scan` result also includes the calling thread data in the reduction.

`group`: Valid group types are `coalesced_group` and `thread_block_tile`.

`val`: Any type that satisfies the below requirements:

  * Qualifies as trivially copyable i.e. `is_trivially_copyable<TyArg>::value == true`

  * `sizeof(T) <= 32` for `coalesced_group` and tiles of size lower or equal 32, `sizeof(T) <= 8` for larger tiles

  * Has suitable arithmetic or comparative operators for the given function object.


**Note:** Different threads in the group can pass different values for this argument.

`op`: Function objects defined for convenience are `plus(), less(), greater(), bit_and(), bit_xor(), bit_or()` described in [cooperative_groups/reduce.h](#cg-api-reduce-header). These must be constructed, hence the TyVal template argument is required, i.e. `plus<int>()`. `inclusive_scan` and `exclusive_scan` also supports lambdas and other function objects that can be invoked using `operator()`. Overloads without this argument use `cg::plus<TyVal>()`.

**Scan update**
    
    
    template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
    auto inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));
    
    template <typename TyGroup, typename TyAtomic, typename TyVal>
    TyVal inclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);
    
    template <typename TyGroup, typename TyAtomic, typename TyVal, typename TyFn>
    auto exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val, TyFn&& op) -> decltype(op(val, val));
    
    template <typename TyGroup, typename TyAtomic, typename TyVal>
    TyVal exclusive_scan_update(const TyGroup& group, TyAtomic& atomic, TyVal&& val);
    

`*_scan_update` collectives take an additional argument `atomic` that can be either of `cuda::atomic` or `cuda::atomic_ref` available in [CUDA C++ Standard Library](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives.html). These variants of the API are available only on platforms and devices, where these types are supported by the CUDA C++ Standard Library. These variants will perform an update to the `atomic` according to `op` with value of the sum of input values of all threads in the group. Previous value of the `atomic` will be combined with the result of scan by each thread and returned. Type held by the `atomic` must match the type of `TyVal`. Scope of the atomic must include all the threads in the group and if multiple groups are using the same atomic concurrently, scope must include all threads in all groups using it. Atomic update is performed with relaxed memory ordering.

Following pseudocode illustrates how the update variant of scan works:
    
    
    /*
     inclusive_scan_update behaves as the following block,
     except both reduce and inclusive_scan is calculated simultaneously.
    auto total = reduce(group, val, op);
    TyVal old;
    if (group.thread_rank() == selected_thread) {
        atomically {
            old = atomic.load();
            atomic.store(op(old, total));
        }
    }
    old = group.shfl(old, selected_thread);
    return op(inclusive_scan(group, val, op), old);
    */
    

`cooperative_groups/scan.h` header needs to be included.

**Example of stream compaction using exclusive_scan:**
    
    
    #include <cooperative_groups.h>
    #include <cooperative_groups/scan.h>
    namespace cg = cooperative_groups;
    
    // put data from input into output only if it passes test_fn predicate
    template<typename Group, typename Data, typename TyFn>
    __device__ int stream_compaction(Group &g, Data *input, int count, TyFn&& test_fn, Data *output) {
        int per_thread = count / g.num_threads();
        int thread_start = min(g.thread_rank() * per_thread, count);
        int my_count = min(per_thread, count - thread_start);
    
        // get all passing items from my part of the input
        //  into a contagious part of the array and count them.
        int i = thread_start;
        while (i < my_count + thread_start) {
            if (test_fn(input[i])) {
                i++;
            }
            else {
                my_count--;
                input[i] = input[my_count + thread_start];
            }
        }
    
        // scan over counts from each thread to calculate my starting
        //  index in the output
        int my_idx = cg::exclusive_scan(g, my_count);
    
        for (i = 0; i < my_count; ++i) {
            output[my_idx + i] = input[thread_start + i];
        }
        // return the total number of items in the output
        return g.shfl(my_idx + my_count, g.num_threads() - 1);
    }
    

**Example of dynamic buffer space allocation using exclusive_scan_update:**
    
    
    #include <cooperative_groups.h>
    #include <cooperative_groups/scan.h>
    namespace cg = cooperative_groups;
    
    // Buffer partitioning is static to make the example easier to follow,
    // but any arbitrary dynamic allocation scheme can be implemented by replacing this function.
    __device__ int calculate_buffer_space_needed(cg::thread_block_tile<32>& tile) {
        return tile.thread_rank() % 2 + 1;
    }
    
    __device__ int my_thread_data(int i) {
        return i;
    }
    
    __global__ void kernel() {
        __shared__ extern int buffer[];
        __shared__ cuda::atomic<int, cuda::thread_scope_block> buffer_used;
    
        auto block = cg::this_thread_block();
        auto tile = cg::tiled_partition<32>(block);
        buffer_used = 0;
        block.sync();
    
        // each thread calculates buffer size it needs
        int buf_needed = calculate_buffer_space_needed(tile);
    
        // scan over the needs of each thread, result for each thread is an offset
        // of that thread’s part of the buffer. buffer_used is atomically updated with
        // the sum of all thread's inputs, to correctly offset other tile’s allocations
        int buf_offset =
            cg::exclusive_scan_update(tile, buffer_used, buf_needed);
    
        // each thread fills its own part of the buffer with thread specific data
        for (int i = 0 ; i < buf_needed ; ++i) {
            buffer[buf_offset + i] = my_thread_data(i);
        }
    
        block.sync();
        // buffer_used now holds total amount of memory allocated
        // buffer is {0, 0, 1, 0, 0, 1 ...};
    }
    

### 5.6.3.6. cooperative_groups/sync.h

#### 5.6.3.6.1. `barrier_arrive` and `barrier_wait`
    
    
    T::arrival_token T::barrier_arrive();
    void T::barrier_wait(T::arrival_token&&);
    

`barrier_arrive` and `barrier_wait` member functions provide a synchronization API similar to `cuda::barrier` [(read more)](../04-special-topics/async-barriers.html#asynchronous-barriers). Cooperative Groups automatically initializes the group barrier, but arrive and wait operations have an additional restriction resulting from collective nature of those operations: All threads in the group must arrive and wait at the barrier once per phase. When `barrier_arrive` is called with a group, result of calling any collective operation or another barrier arrival with that group is undefined until completion of the barrier phase is observed with `barrier_wait` call. Threads blocked on `barrier_wait` might be released from the synchronization before other threads call `barrier_wait`, but only after all threads in the group called `barrier_arrive`. Group type `T` can be any of the [implicit groups](../04-special-topics/cooperative-groups.html#cooperative-groups-implicit-groups). This allows threads to do independent work after they arrive and before they wait for the synchronization to resolve, allowing to hide some of the synchronization latency. `barrier_arrive` returns an `arrival_token` object that must be passed into the corresponding `barrier_wait`. Token is consumed this way and can not be used for another `barrier_wait` call.

**Example of barrier_arrive and barrier_wait used to synchronize initialization of shared memory across the cluster:**
    
    
    #include <cooperative_groups.h>
    
    using namespace cooperative_groups;
    
    void __device__ init_shared_data(const thread_block& block, int *data);
    void __device__ local_processing(const thread_block& block);
    void __device__ process_shared_data(const thread_block& block, int *data);
    
    __global__ void cluster_kernel() {
        extern __shared__ int array[];
        auto cluster = this_cluster();
        auto block   = this_thread_block();
    
        // Use this thread block to initialize some shared state
        init_shared_data(block, &array[0]);
    
        auto token = cluster.barrier_arrive(); // Let other blocks know this block is running and data was initialized
    
        // Do some local processing to hide the synchronization latency
        local_processing(block);
    
        // Map data in shared memory from the next block in the cluster
        int *dsmem = cluster.map_shared_rank(&array[0], (cluster.block_rank() + 1) % cluster.num_blocks());
    
        // Make sure all other blocks in the cluster are running and initialized shared data before accessing dsmem
        cluster.barrier_wait(std::move(token));
    
        // Consume data in distributed shared memory
        process_shared_data(block, dsmem);
        cluster.sync();
    }
    

#### 5.6.3.6.2. `sync`
    
    
    static void T::sync();
    
    template <typename T>
    void sync(T& group);
    

`sync` synchronizes the threads named in the group. Group type `T` can be any of the existing group types, as all of them support synchronization. Its available as a member function in every group type or as a free function taking a group as parameter.

##### 5.6.3.6.2.1. Grid Synchronization

Prior to the introduction of Cooperative Groups, the CUDA programming model only allowed synchronization between thread blocks at a kernel completion boundary. The kernel boundary carries with it an implicit invalidation of state, and with it, potential performance implications.

For example, in certain use cases, applications have a large number of small kernels, with each kernel representing a stage in a processing pipeline. The presence of these kernels is required by the current CUDA programming model to ensure that the thread blocks operating on one pipeline stage have produced data before the thread block operating on the next pipeline stage is ready to consume it. In such cases, the ability to provide global inter thread block synchronization would allow the application to be restructured to have persistent thread blocks, which are able to synchronize on the device when a given stage is complete.

To synchronize across the grid, from within a kernel, you would simply use the `grid.sync()` function:
    
    
    grid_group grid = this_grid();
    grid.sync();
    

And when launching the kernel it is necessary to use, instead of the `<<<...>>>` execution configuration syntax, the `cudaLaunchCooperativeKernel` CUDA runtime launch API or the `CUDA driver equivalent`.

**Example:**

To guarantee co-residency of the thread blocks on the GPU, the number of blocks launched needs to be carefully considered. For example, as many blocks as there are SMs can be launched as follows:
    
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    // initialize, then launch
    cudaLaunchCooperativeKernel((void*)my_kernel, deviceProp.multiProcessorCount, numThreads, args);
    

Alternatively, you can maximize the exposed parallelism by calculating how many blocks can fit simultaneously per-SM using the occupancy calculator as follows:
    
    
    /// This will launch a grid that can maximally fill the GPU, on the default stream with kernel arguments
    int numBlocksPerSm = 0;
     // Number of threads my_kernel will be launched with
    int numThreads = 128;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, my_kernel, numThreads, 0);
    // launch
    void *kernelArgs[] = { /* add kernel args */ };
    dim3 dimBlock(numThreads, 1, 1);
    dim3 dimGrid(deviceProp.multiProcessorCount*numBlocksPerSm, 1, 1);
    cudaLaunchCooperativeKernel((void*)my_kernel, dimGrid, dimBlock, kernelArgs);
    

It is good practice to first ensure the device supports cooperative launches by querying the device attribute `cudaDevAttrCooperativeLaunch`:
    
    
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    

which will set `supportsCoopLaunch` to 1 if the property is supported on device 0. Only devices with compute capability of 6.0 and higher are supported. In addition, you need to be running on either of these:

  * The Linux platform without MPS

  * The Linux platform with MPS and on a device with compute capability 7.0 or higher

  * The latest Windows platform


## 5.6.4. CUDA Device Runtime

The CUDA device runtime is an API available within kernel code which provides many of the same capabilities as the CUDA Runtime API on the host. These APIs are used most often in the contexts of [CUDA Dynamic Parallelism](../04-special-topics/dynamic-parallelism.html#cuda-dynamic-parallelism) or [Device Graph Launch](../04-special-topics/cuda-graphs.html#cuda-graphs-device-graph-launch).

### 5.6.4.1. Including Device Runtime API in CUDA Code

Similar to the host-side runtime API, prototypes for the CUDA device runtime API are included automatically during program compilation. There is no need to include`cuda_device_runtime_api.h` explicitly.

### 5.6.4.2. Memory in the CUDA Device Runtime

#### 5.6.4.2.1. Configuration Options

Resource allocation for the device runtime system software is controlled via the `cudaDeviceSetLimit()` API from the host program. Limits must be set before any kernel is launched, and may not be changed while the GPU is actively running programs.

The following named limits may be set:

Limit | Behavior  
---|---  
`cudaLimitDevRuntimePendingLaunchCount` | Controls the amount of memory set aside for buffering kernel launches and events which have not yet begun to execute, due either to unresolved dependencies or lack of execution resources. When the buffer is full, an attempt to allocate a launch slot during a device side kernel launch will fail and return `cudaErrorLaunchOutOfResources`, while an attempt to allocate an event slot will fail and return `cudaErrorMemoryAllocation`. The default number of launch slots is 2048. Applications may increase the number of launch and/or event slots by setting `cudaLimitDevRuntimePendingLaunchCount`. The number of event slots allocated is twice the value of that limit.  
`cudaLimitStackSize` | Controls the stack size in bytes of each GPU thread. The CUDA driver automatically increases the per-thread stack size for each kernel launch as needed. This size isn’t reset back to the original value after each launch. To set the per-thread stack size to a different value, `cudaDeviceSetLimit()` can be called to set this limit. The stack will be immediately resized, and if necessary, the device will block until all preceding requested tasks are complete. `cudaDeviceGetLimit()` can be called to get the current per-thread stack size.  
  
#### 5.6.4.2.2. Allocation and Lifetime

`cudaMalloc()` and `cudaFree()` have distinct semantics between the host and device environments. When invoked from the host, `cudaMalloc()` allocates a new region from unused device memory. When invoked from the device runtime these functions map to device-side `malloc()` and `free()`. This implies that within the device environment the total allocatable memory is limited to the device `malloc()` heap size, which may be smaller than the available unused device memory. Also, it is an error to invoke `cudaFree()` from the host program on a pointer which was allocated by `cudaMalloc()` on the device or vice-versa.

| `cudaMalloc()` on Host | `cudaMalloc()` on Device  
---|---|---  
`cudaFree()` on Host | Supported | Not Supported  
`cudaFree()` on Device | Not Supported | Supported  
Allocation limit | Available device memory | `cudaLimitMallocHeapSize`  
  
##### 5.6.4.2.2.1. Memory Declarations

###### 5.6.4.2.2.1.1. Device and Constant Memory

Memory declared at file scope with `__device__` or `__constant__` memory space specifiers behaves identically when using the device runtime. All kernels may read or write device variables, whether the kernel was initially launched by the host or device runtime. Equivalently, all kernels will have the same view of `__constant__`s as declared at the module scope.

###### 5.6.4.2.2.1.2. Textures and Surfaces

> The device runtime does not allow creation or destruction of texture or surface objects from within device code. Texture and surface objects created from the host may be used and passed around freely on the device. Regardless of where they are created, dynamically created texture objects are always valid and may be passed to child kernels from a parent.

Note

The device runtime does not support legacy module-scope (i.e., compute capability 2.0 or Fermi-style) textures and surfaces within a kernel launched from the device. Module-scope (legacy) textures may be created from the host and used in device code as for any kernel, but may only be used by a top-level kernel (i.e., the one which is launched from the host).

###### 5.6.4.2.2.1.3. Shared Memory Variable Declarations

In CUDA C++ shared memory can be declared either as a statically sized file-scope or function-scoped variable, or as an `extern` variable with the size determined at runtime by the kernel’s caller via a launch configuration argument. Both types of declarations are valid under the device runtime.
    
    
    __global__ void permute(int n, int *data) {
       extern __shared__ int smem[];
       if (n <= 1)
           return;
    
       smem[threadIdx.x] = data[threadIdx.x];
       __syncthreads();
    
       permute_data(smem, n);
       __syncthreads();
    
       // Write back to GMEM since we can't pass SMEM to children.
       data[threadIdx.x] = smem[threadIdx.x];
       __syncthreads();
    
       if (threadIdx.x == 0) {
           permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data);
           permute<<< 1, 256, n/2*sizeof(int) >>>(n/2, data+n/2);
       }
    }
    
    void host_launch(int *data) {
        permute<<< 1, 256, 256*sizeof(int) >>>(256, data);
    }
    

###### 5.6.4.2.2.1.4. Constant Memory

Constants may not be modified from the device. They may only be modified from the host, but the behavior of modifying a constant from the host while there is a concurrent grid that access that constant at any point during its lifetime is undefined.

###### 5.6.4.2.2.1.5. Symbol Addresses

Device-side symbols (i.e., those marked `__device__`) may be referenced from within a kernel simply via the `&` operator, as all global-scope device variables are in the kernel’s visible address space. This also applies to `__constant__` symbols, although in this case the pointer will reference read-only data.

Since device-side symbols can be referenced directly, those CUDA runtime APIs which reference symbols (e.g., `cudaMemcpyToSymbol()` or `cudaGetSymbolAddress()`) are unnecessary and are not supported by the device runtime. This implies that constant data cannot be altered from within a running kernel, even ahead of a child kernel launch, as references to `__constant__` space are read-only.

### 5.6.4.3. SM Id and Warp Id

Note that in PTX `%smid` and `%warpid` are defined as volatile values. The device runtime may reschedule thread blocks onto different SMs in order to more efficiently manage resources. As such, it is unsafe to rely upon `%smid` or `%warpid` remaining unchanged across the lifetime of a thread or thread block.

### 5.6.4.4. Launch Setup APIs

[Device-Side Kernel Launch](../04-special-topics/dynamic-parallelism.html#dynamic-parallelism-device-runtime-kernel-launch) describes the syntax for launching kernels from device code using the same triple chevron launch notation as the host CUDA Runtime API.

Kernel launch is a system-level mechanism exposed through the device runtime library. It is also available directly from PTX via `cudaGetParameterBuffer()` and `cudaLaunchDevice()` APIs. It is permitted for a CUDA application to call these APIs itself, with the same requirements as for PTX. In both cases, the user is then responsible for correctly populating all necessary data structures in the correct format according to specification. Backwards compatibility is guaranteed in these data structures.

As with host-side launch, the device-side operator `<<<>>>` maps to underlying kernel launch APIs. This allows users targeting PTX will to perform a launch. The NVCC compiler front-end translates `<<<>>>` into these calls.

Table 60 New Device-only Launch Implementation Functions Runtime API Launch Functions | Description of Difference From Host Runtime Behavior (behavior is identical if no description)  
---|---  
`cudaGetParameterBuffer` | Generated automatically from `<<<>>>`. Note different API to host equivalent.  
`cudaLaunchDevice` | Generated automatically from `<<<>>>`. Note different API to host equivalent.  
  
The APIs for these launch functions are different to those of the CUDA Runtime API, and are defined as follows:
    
    
    extern   device   cudaError_t cudaGetParameterBuffer(void **params);
    extern __device__ cudaError_t cudaLaunchDevice(void *kernel,
                                            void *params, dim3 gridDim,
                                            dim3 blockDim,
                                            unsigned int sharedMemSize = 0,
                                            cudaStream_t stream = 0);
    

### 5.6.4.5. Device Management

There is no multi-GPU support from the device runtime; the device runtime is only capable of operating on the device upon which it is currently executing. It is permitted, however, to query properties for any CUDA capable device in the system.

### 5.6.4.6. API Reference

The portions of the CUDA Runtime API supported in the device runtime are detailed here. Host and device runtime APIs have identical syntax; semantics are the same except where indicated. The following table provides an overview of the API relative to the version available from the host.

Table 61 Supported API Functions Runtime API Functions | Details  
---|---  
`cudaDeviceGetCacheConfig` |   
`cudaDeviceGetLimit` |   
`cudaGetLastError` | Last error is per-thread state, not per-block state  
`cudaPeekAtLastError` |   
`cudaGetErrorString` |   
`cudaGetDeviceCount` |   
`cudaDeviceGetAttribute` | Will return attributes for any device  
`cudaGetDevice` | Always returns current device ID as would be seen from host  
`cudaStreamCreateWithFlags` | Must pass `cudaStreamNonBlocking` flag  
`cudaStreamDestroy` |   
`cudaStreamWaitEvent` |   
`cudaEventCreateWithFlags` | Must pass `cudaEventDisableTiming` flag  
`cudaEventRecord` |   
`cudaEventDestroy` |   
`cudaFuncGetAttributes` |   
`cudaMemcpyAsync` | Notes about all `memcpy/memset` functions:

  * Only async `memcpy/set` functions are supported
  * Only device-to-device `memcpy` is permitted
  * May not pass in local or shared memory pointers

  
`cudaMemcpy2DAsync`  
`cudaMemcpy3DAsync`  
`cudaMemsetAsync`  
`cudaMemset2DAsync` |   
`cudaMemset3DAsync` |   
`cudaRuntimeGetVersion` |   
`cudaMalloc` | May not call `cudaFree` on the device on a pointer created on the host, and vice-versa  
`cudaFree`  
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` |   
`cudaOccupancyMaxPotentialBlockSize` |   
`cudaOccupancyMaxPotentialBlockSizeVariableSMem` |   
  
### 5.6.4.7. API Errors and Launch Failures

As usual for the CUDA runtime, any function may return an error code. The last error code returned is recorded and may be retrieved via the `cudaGetLastError()` call. Errors are recorded per-thread, so that each thread can identify the most recent error that it has generated. The error code is of type `cudaError_t`.

Similar to a host-side launch, device-side launches may fail for many reasons (invalid arguments, etc). The user must call `cudaGetLastError()` to determine if a launch generated an error, however lack of an error after launch does not imply the child kernel completed successfully.

For device-side exceptions, e.g., access to an invalid address, an error in a child grid will be returned to the host.

### 5.6.4.8. Device Runtime Streams

The CUDA device runtime exposes special named streams which provide specific behaviors for kernels and graphs launched from the device. The named streams relevant to device graph launch are documented in [Device Launch](../04-special-topics/cuda-graphs.html#cuda-graphs-device-graph-device-launch). Two other named streams which can be used for kernels and memcpy operations in the CUDA device runtime are `cudaStreamTailLaunch` and `cudaStreamTailLaunch`. The specific behaviors of these named streams are documented in this section.

Both named and unnamed (NULL) streams are available from the device runtime. Stream handles may not be passed to parent or child grids. In other words, a stream should be treated as private to the grid in which it is created.

The host-side NULL stream’s cross-stream barrier semantic is not supported on the device (see below for details). In order to retain semantic compatibility with the host runtime, all device streams must be created using the `cudaStreamCreateWithFlags()` API, passing the `cudaStreamNonBlocking` flag. The `cudaStreamCreate()` API is not available in the CUDA device runtime.

As `cudaStreamSynchronize()` and `cudaStreamQuery()` are unsupported by the device runtime. A kernel launched into the `cudaStreamTailLaunch` stream should be used instead when the application needs to know that stream-launched child kernels have completed.

#### 5.6.4.8.1. The Implicit (NULL) Stream

Within a host program, the unnamed (NULL) stream has additional barrier synchronization semantics with other streams (see [Blocking and non-blocking streams and the default stream](../02-basics/asynchronous-execution.html#async-execution-blocking-non-blocking-default-stream) for details). The device runtime offers a single implicit, unnamed stream shared between all threads in a thread block, but as all named streams must be created with the `cudaStreamNonBlocking` flag, work launched into the NULL stream will not insert an implicit dependency on pending work in any other streams (including NULL streams of other thread blocks).

#### 5.6.4.8.2. The Fire-and-Forget Stream

The fire-and-forget named stream (`cudaStreamFireAndForget`) allows the user to launch fire-and-forget work with less boilerplate and without stream tracking overhead. It is functionally identical to, but faster than, creating a new stream per launch, and launching into that stream.

Fire-and-forget launches are immediately scheduled for launch without any dependency on the completion of previously launched grids. No other grid launches can depend on the completion of a fire-and-forget launch, except through the implicit synchronization at the end of the parent grid. So a tail launch or the next grid in parent grid’s stream won’t launch before a parent grid’s fire-and-forget work has completed.
    
    
    // In this example, C2's launch will not wait for C1's completion
    __global__ void P( ... ) {
       C1<<< ... , cudaStreamFireAndForget >>>( ... );
       C2<<< ... , cudaStreamFireAndForget >>>( ... );
    }
    

The fire-and-forget stream cannot be used to record or wait on events. Attempting to do so results in `cudaErrorInvalidValue`. The fire-and-forget stream is not supported when compiled with `CUDA_FORCE_CDP1_IF_SUPPORTED` defined. Fire-and-forget stream usage requires compilation to be in 64-bit mode.

#### 5.6.4.8.3. The Tail Launch Stream

The tail launch named stream (`cudaStreamTailLaunch`) allows a grid to schedule a new grid for launch after its completion. It should be possible to to use a tail launch to achieve the same functionality as a `cudaDeviceSynchronize()` in most cases.

Each grid has its own tail launch stream. All non-tail launch work launched by a grid is implicitly synchronized before the tail stream is kicked off. I.e. A parent grid’s tail launch does not launch until the parent grid and all work launched by the parent grid to ordinary streams or per-thread or fire-and-forget streams have completed. If two grids are launched to the same grid’s tail launch stream, the later grid does not launch until the earlier grid and all its descendent work has completed.
    
    
    // In this example, C2 will only launch after C1 completes.
    __global__ void P( ... ) {
       C1<<< ... , cudaStreamTailLaunch >>>( ... );
       C2<<< ... , cudaStreamTailLaunch >>>( ... );
    }
    

Grids launched into the tail launch stream will not launch until the completion of all work by the parent grid, including all other grids (and their descendants) launched by the parent in all non-tail launched streams, including work executed or launched after the tail launch.
    
    
    // In this example, C will only launch after all X, F and P complete.
    __global__ void P( ... ) {
       C<<< ... , cudaStreamTailLaunch >>>( ... );
       X<<< ... , cudaStreamPerThread >>>( ... );
       F<<< ... , cudaStreamFireAndForget >>>( ... )
    }
    

The next grid in the parent grid’s stream will not be launched before a parent grid’s tail launch work has completed. In other words, the tail launch stream behaves as if it were inserted between its parent grid and the next grid in its parent grid’s stream.
    
    
    // In this example, P2 will only launch after C completes.
    __global__ void P1( ... ) {
       C<<< ... , cudaStreamTailLaunch >>>( ... );
    }
    
    __global__ void P2( ... ) {
    }
    
    int main ( ... ) {
       ...
       P1<<< ... >>>( ... );
       P2<<< ... >>>( ... );
       ...
    }
    

Each grid only gets one tail launch stream. To tail launch concurrent grids, it can be done like the example below.
    
    
    // In this example,  C1 and C2 will launch concurrently after P's completion
    __global__ void T( ... ) {
       C1<<< ... , cudaStreamFireAndForget >>>( ... );
       C2<<< ... , cudaStreamFireAndForget >>>( ... );
    }
    
    __global__ void P( ... ) {
       ...
       T<<< ... , cudaStreamTailLaunch >>>( ... );
    }
    

The tail launch stream cannot be used to record or wait on events. Attempting to do so results in `cudaErrorInvalidValue`. The tail launch stream is not supported when compiled with `CUDA_FORCE_CDP1_IF_SUPPORTED` defined. Tail launch stream usage requires compilation to be in 64-bit mode.

### 5.6.4.9. ECC Errors

No notification of ECC errors is available to code within a CUDA kernel. ECC errors are reported at the host side once the entire launch tree has completed. Any ECC errors which arise during execution of a nested program will either generate an exception or continue execution (depending upon error and configuration).
