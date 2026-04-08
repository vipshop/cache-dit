<a id="cute-arch"></a>

# arch

The `cute.arch` module provides lightweight wrappers for NVVM Operation builders which implement CUDA built-in
device functions such as `thread_idx`. It integrates seamlessly with CuTe DSL types.

These wrappers enable source location tracking through the `@dsl_user_op`
decorator. The module includes the following functionality:

- Core CUDA built-in functions such as `thread_idx`, `warp_idx`, `block_dim`, `grid_dim`, `cluster_dim`, and related functions
- Memory barrier management functions including `mbarrier_init`, `mbarrier_arrive`, `mbarrier_wait`, and associated operations
- Low-level shared memory (SMEM) management capabilities, with `SmemAllocator` as the recommended interface
- Low-level tensor memory (TMEM) management capabilities, with `TmemAllocator` as the recommended interface

## API documentation

### cutlass.cute.arch.add_packed_f32x2(src_a: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], src_b: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], \*, src_c: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32] | None = None, calc_func: ~typing.Callable = <function add_packed_f32x2>, rnd: ~typing.Literal['rn', 'rz', 'rm', 'rp', 'none'] | None = 'rn', ftz=None, loc=None, ip=None) → Tuple[Float32, Float32]

### cutlass.cute.arch.alloc_smem(element_type: Type[Numeric], size_in_elems: int, alignment: int | None = None, , loc=None, ip=None) → Pointer

Statically allocates SMEM.

* **Parameters:**
  * **element_type** (*Type* *[**Numeric* *]*) – The pointee type of the pointer.
  * **size_in_elems** (*int*) – The size of the allocation in terms of number of elements of the
    pointee type
  * **alignment** (*int*) – An optional pointer alignment for the allocation
* **Returns:**
  A pointer to the start of the allocation
* **Return type:**
  Pointer

### cutlass.cute.arch.alloc_tmem(num_columns: int | Integer, smem_ptr_to_write_address: Pointer, is_two_cta=None, , arch: str = 'sm_100', loc=None, ip=None) → None

Allocates TMEM.

* **Parameters:**
  * **num_columns** (*Int*) – The number of TMEM columns to allocate
  * **smem_ptr_to_write_address** (*Pointer*) – A pointer to a SMEM buffer where the TMEM address is written
    to
  * **is_two_cta** – Optional boolean parameter for 2-CTA MMAs
  * **arch** (*str*) – The architecture of the GPU.

### cutlass.cute.arch.atomic_add(ptr, val: Numeric | Value, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric | Value

Performs an atomic addition operation.

Atomically adds val to the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Union* *[**Numeric* *,* *ir.Value* *]*) – Value to add (scalar Numeric or vector ir.Value)
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Union[Numeric, ir.Value]

### cutlass.cute.arch.atomic_and(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic bitwise AND operation.

Atomically computes bitwise AND of val with the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value for AND operation
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_cas(ptr, , cmp: Numeric, val: Numeric, sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic compare-and-swap (CAS) operation.

Atomically compares the value at the memory location with cmp. If they are equal,
stores val at the memory location and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location. Supports:
    - ir.Value (LLVM pointer)
    - cute.ptr (_Pointer instance)
  * **cmp** (*Numeric*) – Value to compare against current memory value
  * **val** (*Numeric*) – Value to store if comparison succeeds
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_exch(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic exchange operation.

Atomically exchanges val with the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value to exchange
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_max(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic maximum operation.

Atomically computes maximum of val and the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value for MAX operation
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_max_float32(ptr, value: Float32, , positive_only: bool = True, loc=None, ip=None) → Float32

Performs an atomic max operation on a float32 value in global memory.

This implementation works correctly for non-negative values (>= 0) using direct bitcast.

* **Parameters:**
  * **ptr** – Pointer to the memory location
  * **value** (*Float32*) – The float32 value to compare and potentially store (should be >= 0 for correct results)
  * **positive_only** (*bool*) – If True (default), assumes input values are non-negative.
    This parameter is provided for API compatibility and future extensions.
* **Returns:**
  The old value at the memory location
* **Return type:**
  Float32

### cutlass.cute.arch.atomic_min(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic minimum operation.

Atomically computes minimum of val and the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value for MIN operation
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_or(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic bitwise OR operation.

Atomically computes bitwise OR of val with the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value for OR operation
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.atomic_xor(ptr, val: Numeric, , sem: Literal['relaxed', 'release', 'acquire', 'acq_rel'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → Numeric

Performs an atomic bitwise XOR operation.

Atomically computes bitwise XOR of val with the value at memory location ptr and returns the old value.

* **Parameters:**
  * **ptr** – Pointer to memory location
  * **val** (*Numeric*) – Value for XOR operation
  * **sem** (*Optional* *[**Literal* *[* *"relaxed"* *,*  *"release"* *,*  *"acquire"* *,*  *"acq_rel"* *]* *]*) – Memory semantic (“relaxed”, “release”, “acquire”, “acq_rel”)
  * **scope** (*Optional* *[**Literal* *[* *"gpu"* *,*  *"cta"* *,*  *"cluster"* *,*  *"sys"* *]* *]*) – Memory scope (“gpu”, “cta”, “cluster”, “sys”)
* **Returns:**
  Old value at memory location
* **Return type:**
  Numeric

### cutlass.cute.arch.barrier(, barrier_id=None, number_of_threads=None, loc=None, ip=None) → None

Creates a barrier, optionally named.

### cutlass.cute.arch.barrier_arrive(, barrier_id=None, number_of_threads=None, loc=None, ip=None) → None

### cutlass.cute.arch.block_dim(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the number of threads in each dimension of the CTA.

### cutlass.cute.arch.block_idx(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the CTA identifier within a grid.

### cutlass.cute.arch.block_idx_in_cluster(, loc=None, ip=None) → Int32

Returns the linearized identifier of the CTA within the cluster.

### cutlass.cute.arch.block_in_cluster_dim(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the dimensions of the cluster.

### cutlass.cute.arch.block_in_cluster_idx(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the CTA index within a cluster across all dimensions.

### cutlass.cute.arch.clc_response(result_addr: Pointer, loc=None, ip=None) → Tuple[Int32, Int32, Int32, Int32]

After loading response from clusterlaunchcontrol.try_cancel instruction into 16-byte
register, it can be further queried using clusterlaunchcontrol.query_cancel instruction.
If the cluster is canceled successfully, predicate p is set to true; otherwise, it is
set to false. If the request succeeded, clusterlaunchcontrol.query_cancel.get_first_ctaid
extracts the CTA id of the first CTA in the canceled cluster. By default, the instruction
returns a .v4 vector whose first three elements are the x, y and z coordinate of first CTA
in canceled cluster.

* **Parameters:**
  **result_addr** (*Pointer*) – A pointer to the cluster launch control response address in SMEM

### cutlass.cute.arch.cluster_arrive(, aligned=None, loc=None, ip=None) → None

A cluster-wide arrive operation.

### cutlass.cute.arch.cluster_arrive_relaxed(, aligned=None, loc=None, ip=None) → None

A cluster-wide arrive operation with relaxed semantics.

### cutlass.cute.arch.cluster_dim(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the number of clusters in each dimension of the grid.

### cutlass.cute.arch.cluster_idx(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the cluster identifier within a grid.

### cutlass.cute.arch.cluster_size(, loc=None, ip=None) → Int32

Returns the number of CTA within the cluster.

### cutlass.cute.arch.cluster_wait(, loc=None, ip=None) → None

A cluster-wide wait operation.

### cutlass.cute.arch.cp_async_bulk_commit_group(, loc=None, ip=None) → None

Commits all prior initiated but uncommitted cp.async.bulk instructions.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group).

### cutlass.cute.arch.cp_async_bulk_wait_group(group, , read=None, loc=None, ip=None) → None

Waits till only a specified numbers of cp.async.bulk groups are pending.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-wait-group).

### cutlass.cute.arch.cp_async_commit_group(, loc=None, ip=None) → None

Commits all prior initiated but uncommitted cp.async instructions.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-commit-group).

### cutlass.cute.arch.cp_async_wait_group(n, , loc=None, ip=None) → None

Waits till only a specified numbers of cp.async groups are pending.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all).

### cutlass.cute.arch.cvt_f32x2_bf16x2(src_vec2, , loc=None, ip=None)

### cutlass.cute.arch.cvt_i4_bf16_intrinsic(vec_i4, length, , with_shuffle=False, loc=None, ip=None)

Fast conversion from int4 to bfloat16. It converts a vector of int4 to a vector of bfloat16.

* **Parameters:**
  * **vec_i4** (*1D vector* *of* *int4*) – The input vector of int4.
  * **length** (*int*) – The length of the input vector.
  * **with_shuffle** (*bool*) – Whether the input vec_i4 follows a specific shuffle pattern.
    If True, for consecutive 8 int4 values with indices of (0, 1, 2, 3, 4, 5, 6, 7),
    the input elements are shuffled to (0, 2, 1, 3, 4, 6, 5, 7). For tailing elements less than 8,
    the shuffle pattern is (0, 2, 1, 3) for 4 elements. No shuffle is needed for less than 4 elements.
    Shuffle could help to produce converted bf16 values in the natural order of (0, 1, 2 ,3 ,4 ,5 ,6 ,7)
    without extra prmt instructions and thus better performance.
* **Returns:**
  The output 1D vector of bfloat16 with the same length as the input vector.
* **Return type:**
  1D vector of bfloat16

### cutlass.cute.arch.cvt_i8_bf16(src_i8, , loc=None, ip=None)

### cutlass.cute.arch.cvt_i8_bf16_intrinsic(vec_i8, length, , loc=None, ip=None)

Fast conversion from int8 to bfloat16. It converts a vector of int8 to a vector of bfloat16.

* **Parameters:**
  * **vec_i8** (*1D vector* *of* *int8*) – The input vector of int8.
  * **length** (*int*) – The length of the input vector.
* **Returns:**
  The output 1D vector of bfloat16 with the same length as the input vector.
* **Return type:**
  1D vector of bfloat16

### cutlass.cute.arch.cvt_i8x2_to_bf16x2(src_vec2, , loc=None, ip=None)

### cutlass.cute.arch.cvt_i8x2_to_f32x2(src_vec2, , loc=None, ip=None)

### cutlass.cute.arch.cvt_i8x4_to_bf16x4(src_vec4, , loc=None, ip=None)

### cutlass.cute.arch.cvt_i8x4_to_f32x4(src_vec4, , loc=None, ip=None)

### cutlass.cute.arch.dealloc_tmem(tmem_ptr: Pointer, num_columns: int | Integer, is_two_cta=None, , arch: str = 'sm_100', loc=None, ip=None) → None

Deallocates TMEM using the provided pointer and number of columns.

* **Parameters:**
  * **tmem_ptr** (*Pointer*) – A pointer to the TMEM allocation to de-allocate
  * **num_columns** (*Int*) – The number of columns in the TMEM allocation
  * **is_two_cta** – Optional boolean parameter for 2-CTA MMAs

### cutlass.cute.arch.elect_one(, loc=None, ip=None) → IfOpRegion

Elects one thread within a warp.

```python
with elect_one():
    # Only one thread in the warp executes the code in this context
    pass
```

### cutlass.cute.arch.exp2(a: float | Float32, , loc=None, ip=None) → Float32

### cutlass.cute.arch.fence_acq_rel_cluster(, loc=None, ip=None) → None

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.arch.fence_acq_rel_cta(, loc=None, ip=None) → None

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.arch.fence_acq_rel_gpu(, loc=None, ip=None) → None

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.arch.fence_acq_rel_sys(, loc=None, ip=None) → None

Fence operation with acquire-release semantics.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.arch.fence_proxy(kind: Literal['alias', 'async', 'async.global', 'async.shared', 'tensormap', 'generic'], , space: Literal['cta', 'cluster'] | None = None, use_intrinsic=None, loc=None, ip=None) → None

Fence operation to ensure memory consistency between proxies.

* **Parameters:**
  * **kind** (*Literal* *[* *"alias"* *,*  *"async"* *,*  *"async.global"* *,*  *"async.shared"* *,*  *"tensormap"* *,*  *"generic"* *]*) – Proxy kind string literal:
    - “alias” : Alias proxy
    - “async” : Async proxy
    - “async.global” : Async global proxy
    - “async.shared” : Async shared proxy
    - “tensormap” : Tensormap proxy
    - “generic” : Generic proxy
  * **space** (*Optional* *[**Literal* *[* *"cta"* *,*  *"cluster"* *]* *]*) – Shared memory space scope string literal (optional):
    - “cta” : CTA (Cooperative Thread Array) scope
    - “cluster” : Cluster scope
  * **use_intrinsic** – Whether to use intrinsic version

### cutlass.cute.arch.fence_view_async_tmem_load(, kind: Literal['load', 'store'] = 'load', loc=None, ip=None) → None

Perform a fence operation on the async TMEM load or store.

#### NOTE
This function is only available on sm_100a and above.
The fence is required to synchronize the TMEM load/store
and let the pipeline release or commit the buffer.

Take a mma2acc pipeline as an example of LOAD fence, the ACC tensor is from TMEM.
``
# Start to copy ACC from TMEM to register
cute.copy(tmem_load, tACC, rACC)
fence_view_async_tmem_load()
# After fence, we can ensure the TMEM buffer is consumed totally.
# Release the buffer to let the MMA know it can overwrite the buffer.
mma2accum_pipeline.consumer_release(curr_consumer_state)
``
Take a TS GEMM kernel as an example of STORE fence, the A tensor is from TMEM.
``
# Start to copy A from register to TMEM
cute.copy(tmem_store, rA, tA)
fence_view_async_tmem_store()
# After fence, we can ensure the TMEM buffer is ready.
# Commit the buffer to let the MMA know it can start to load A.
tmem_mma_pipeline.producer_commit(curr_producer_state)
``

* **Parameters:**
  **kind** (*Literal* *[* *"load"* *,*  *"store"* *]*) – The kind of fence operation to perform (“load”, “store”).

### cutlass.cute.arch.fence_view_async_tmem_store(, kind: Literal['load', 'store'] = 'store', loc=None, ip=None) → None

Perform a fence operation on the async TMEM load or store.

#### NOTE
This function is only available on sm_100a and above.
The fence is required to synchronize the TMEM load/store
and let the pipeline release or commit the buffer.

Take a mma2acc pipeline as an example of LOAD fence, the ACC tensor is from TMEM.
``
# Start to copy ACC from TMEM to register
cute.copy(tmem_load, tACC, rACC)
fence_view_async_tmem_load()
# After fence, we can ensure the TMEM buffer is consumed totally.
# Release the buffer to let the MMA know it can overwrite the buffer.
mma2accum_pipeline.consumer_release(curr_consumer_state)
``
Take a TS GEMM kernel as an example of STORE fence, the A tensor is from TMEM.
``
# Start to copy A from register to TMEM
cute.copy(tmem_store, rA, tA)
fence_view_async_tmem_store()
# After fence, we can ensure the TMEM buffer is ready.
# Commit the buffer to let the MMA know it can start to load A.
tmem_mma_pipeline.producer_commit(curr_producer_state)
``

* **Parameters:**
  **kind** (*Literal* *[* *"load"* *,*  *"store"* *]*) – The kind of fence operation to perform (“load”, “store”).

### cutlass.cute.arch.fma_packed_f32x2(src_a: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], src_b: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], src_c: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32] | None, \*, calc_func: ~typing.Callable = <function fma_packed_f32x2>, rnd: ~typing.Literal['rn', 'rz', 'rm', 'rp', 'none'] | None = 'rn', ftz=None, loc=None, ip=None) → Tuple[Float32, Float32]

### cutlass.cute.arch.fmax(a: float | Float32, b: float | Float32, , loc=None, ip=None) → Float32

### cutlass.cute.arch.get_dyn_smem(element_type: Type[Numeric], alignment: int | None = None, , loc=None, ip=None) → Pointer

Retrieves a pointer to a dynamic SMEM allocation.

* **Parameters:**
  * **element_type** (*Type* *[**Numeric* *]*) – The pointee type of the pointer.
  * **alignment** (*int*) – An optional pointer alignment, the result pointer is offset appropriately
* **Returns:**
  A pointer to the start of the dynamic SMEM allocation with a correct
  alignement
* **Return type:**
  Pointer

### cutlass.cute.arch.get_dyn_smem_size(, loc=None, ip=None) → int

Gets the size in bytes of the dynamic shared memory that was specified at kernel launch time.
This can be used for bounds checking during shared memory allocation.

* **Returns:**
  The size of dynamic shared memory in bytes
* **Return type:**
  int

### cutlass.cute.arch.get_max_tmem_alloc_cols(compute_capability: str) → int

Get the tensor memory capacity in columns for a given compute capability.

Returns the maximum TMEM capacity in columns available for the specified
GPU compute capability.

* **Parameters:**
  **compute_capability** (*str*) – The compute capability string (e.g. “sm_100”, “sm_103”)
* **Returns:**
  The TMEM capacity in columns
* **Return type:**
  int
* **Raises:**
  **ValueError** – If the compute capability is not supported

### cutlass.cute.arch.get_min_tmem_alloc_cols(compute_capability: str) → int

Get the minimum TMEM allocation columns for a given compute capability.

Returns the minimum TMEM allocation columns available for the specified
GPU compute capability.

* **Parameters:**
  **compute_capability** (*str*) – The compute capability string (e.g. “sm_100”, “sm_103”)
* **Returns:**
  The minimum TMEM allocation columns
* **Return type:**
  int
* **Raises:**
  **ValueError** – If the compute capability is not supported

### cutlass.cute.arch.grid_dim(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the number of CTAs in each dimension of the grid.

### cutlass.cute.arch.issue_clc_query(mbar_ptr: Pointer, clc_response_ptr: Pointer, loc=None, ip=None) → None

The clusterlaunchcontrol.try_cancel instruction requests atomically cancelling the launch
of a cluster that has not started running yet. It asynchronously writes an opaque response
to shared memory indicating whether the operation succeeded or failed. On success, the
opaque response contains the ctaid of the first CTA of the canceled cluster.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier address in SMEM
  * **clc_response_ptr** (*Pointer*) – A pointer to the cluster launch control response address in SMEM

### cutlass.cute.arch.lane_idx(, loc=None, ip=None) → Int32

Returns the lane index of the current thread within the warp.

### cutlass.cute.arch.load(ptr, dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric] | VectorType, , sem: Literal['relaxed', 'acquire'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, level1_eviction_priority: Literal['evict_normal', 'evict_first', 'evict_last', 'evict_no_allocate', 'evict_unchanged'] | None = None, cop: Literal['ca', 'cg', 'cs', 'lu', 'cv'] | None = None, ss: Literal['cta', 'cluster'] | None = None, level_prefetch_size: Literal['size_64b', 'size_128b', 'size_256b'] | None = None, loc=None, ip=None) → Numeric | Value

Load a value from a memory location.

* **Parameters:**
  * **ptr** – Pointer to load from. Supports:
    - ir.Value (LLVM pointer)
    - cute.ptr (_Pointer instance)
  * **dtype** (*Union* *[*[*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]* *,* *ir.VectorType* *]*) – Data type to load. Can be:
    - Scalar: Numeric type class (Int8, Uint8, Int32, Float32, etc.)
    - Vector: ir.VectorType for vectorized load (e.g., ir.VectorType.get([4], Int64.mlir_type))
  * **sem** – Memory semantic string literal:
  * **scope** – Memory scope string literal:
  * **level1_eviction_priority** – L1 cache eviction policy string literal:
    “evict_normal” : .level1::eviction_priority = .L1::evict_normal
    “evict_first” : .level1::eviction_priority = .L1::evict_first
    “evict_last” : .level1::eviction_priority = .L1::evict_last
    “evict_no_allocate” : .level1::eviction_priority = .L1::no_allocate
    “evict_unchanged” : .level1::eviction_priority = .L1::evict_unchanged
  * **cop** – Load cache modifier string literal:
  * **ss** – Shared memory space string literal:
    “cta” : .ss = .shared::cta
    “cluster” : .ss = .shared::cluster
    None : .ss = .global
  * **level_prefetch_size** – L2 cache prefetch size hint string literal:
    “size_64b” : .level::prefetch_size = .L2::64B
    “size_128b” : .level::prefetch_size = .L2::128B
    “size_256b” : .level::prefetch_size = .L2::256B
* **Returns:**
  Loaded value (scalar Numeric or vector ir.Value)
* **Return type:**
  Union[Numeric, ir.Value]

### cutlass.cute.arch.make_warp_uniform(value: int | Integer, , loc=None, ip=None) → Int32

Provides a compiler hint indicating that the specified value is invariant across all threads in the warp,
which may enable performance optimizations.

* **Parameters:**
  **value** (*Int*) – The integer value to be marked as warp-uniform.
* **Returns:**
  The input value, marked as warp-uniform.
* **Return type:**
  Int32

### cutlass.cute.arch.mbarrier_arrive(mbar_ptr: Pointer, peer_cta_rank_in_cluster: int | Integer | None = None, arrive_count: int | Integer = 1, , loc=None, ip=None) → None

Arrives on an mbarrier.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
    the mbarrier is converted to a remote address in the peer CTA’s
    SMEM.

### cutlass.cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr: Pointer, bytes: int | Integer, peer_cta_rank_in_cluster=None, , loc=None, ip=None) → None

Arrives on a mbarrier and expects a specified number of transaction bytes.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **bytes** (*Int*) – The number of transaction bytes
  * **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
    the mbarrier is converted to a remote address in the peer CTA’s
    SMEM.

### cutlass.cute.arch.mbarrier_conditional_try_wait(cond, mbar_ptr: Pointer, phase: int | Integer, , loc=None, ip=None) → Boolean

Conditionally attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

* **Parameters:**
  * **cond** – A boolean predicate
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **phase** (*Int*) – The phase to wait for (either 0 or 1)
* **Returns:**
  A boolean value indicating whether the wait operation was successful
* **Return type:**
  Boolean

### cutlass.cute.arch.mbarrier_expect_tx(mbar_ptr: Pointer, bytes: int | Integer, peer_cta_rank_in_cluster=None, , loc=None, ip=None) → None

Expects a specified number of transaction bytes without an arrive.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **bytes** (*Int*) – The number of transaction bytes
  * **peer_cta_rank_in_cluster** – An optional CTA rank in cluster. If provided, the pointer to
    the mbarrier is converted to a remote address in the peer CTA’s
    SMEM.

### cutlass.cute.arch.mbarrier_init(mbar_ptr: Pointer, cnt: int | Integer, , loc=None, ip=None) → None

Initializes a mbarrier with the specified thread arrival count.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **cnt** (*Int*) – The arrival count of the mbarrier

### cutlass.cute.arch.mbarrier_init_fence(, loc=None, ip=None) → None

A fence operation that applies to the mbarrier initializations.

### cutlass.cute.arch.mbarrier_try_wait(mbar_ptr: Pointer, phase: int | Integer, , loc=None, ip=None) → Boolean

Attempts to wait on a mbarrier with a specified phase in a non-blocking fashion.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **phase** (*Int*) – The phase to wait for (either 0 or 1)
* **Returns:**
  A boolean value indicating whether the wait operation was successful
* **Return type:**
  Boolean

### cutlass.cute.arch.mbarrier_wait(mbar_ptr: Pointer, phase: int | Integer, , loc=None, ip=None) → None

Waits on a mbarrier with a specified phase.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **phase** (*Int*) – The phase to wait for (either 0 or 1)

### cutlass.cute.arch.mul_packed_f32x2(src_a: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], src_b: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32], \*, src_c: ~typing.Tuple[~cutlass.base_dsl.typing.Float32, ~cutlass.base_dsl.typing.Float32] | None = None, calc_func: ~typing.Callable = <function mul_packed_f32x2>, rnd: ~typing.Literal['rn', 'rz', 'rm', 'rp', 'none'] | None = 'rn', ftz=None, loc=None, ip=None) → Tuple[Float32, Float32]

### cutlass.cute.arch.popc(value: Numeric, , loc=None, ip=None) → Numeric

Performs a population count operation.

### cutlass.cute.arch.prmt(src, src_reg_shifted, prmt_indices, , loc=None, ip=None)

### cutlass.cute.arch.rcp_approx(a: float | Float32, , loc=None, ip=None)

### cutlass.cute.arch.relinquish_tmem_alloc_permit(is_two_cta=None, , loc=None, ip=None) → None

Relinquishes the right to allocate TMEM so that other CTAs potentially in a different grid can
allocate.

### cutlass.cute.arch.retrieve_tmem_ptr(element_type: Type[Numeric], alignment: int, ptr_to_buffer_holding_addr: Pointer, , loc=None, ip=None) → Pointer

Retrieves a pointer to TMEM with the provided element type and alignment.

* **Parameters:**
  * **element_type** (*Type* *[**Numeric* *]*) – The pointee type of the pointer.
  * **alignment** (*int*) – The alignment of the result pointer
  * **ptr_to_buffer_holding_addr** (*Pointer*) – A pointer to a SMEM buffer holding the TMEM address of the
    start of the allocation allocation
* **Returns:**
  A pointer to TMEM
* **Return type:**
  Pointer

### cutlass.cute.arch.setmaxregister_decrease(reg_count: int, , loc=None, ip=None)

### cutlass.cute.arch.setmaxregister_increase(reg_count: int, , loc=None, ip=None)

### cutlass.cute.arch.shuffle_sync(value: Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA), offset: int | Integer, mask: int | Integer = 4294967295, mask_and_clamp: int | Integer = 31, , kind: ShflKind = ShflKind.idx, loc=None, ip=None) → Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA)

Shuffles a value within the threads of a warp.

* **Parameters:**
  * **value** (*Numeric* *or* [*TensorSSA*](cute.md#cutlass.cute.TensorSSA)) – The value to shuffle
  * **mask** (*Int*) – A mask describing the threads participating in this operation
  * **offset** (*Int*) – A source lane or a source lane offset depending on kind
  * **mask_and_clamp** (*Int*) – An integer containing two packed values specifying a mask for logically
    splitting warps into sub-segments and an upper bound for clamping the
    source lane index.
  * **kind** (*ShflKind*) – The kind of shuffle, can be idx, up, down, or bfly
* **Returns:**
  The shuffled value
* **Return type:**
  Numeric

### cutlass.cute.arch.shuffle_sync_bfly(value: Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA), offset: int | Integer, mask: int | Integer = 4294967295, mask_and_clamp: int | Integer = 31, , kind: ShflKind = ShflKind.bfly, loc=None, ip=None) → Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA)

Shuffles a value within the threads of a warp.

* **Parameters:**
  * **value** (*Numeric* *or* [*TensorSSA*](cute.md#cutlass.cute.TensorSSA)) – The value to shuffle
  * **mask** (*Int*) – A mask describing the threads participating in this operation
  * **offset** (*Int*) – A source lane or a source lane offset depending on kind
  * **mask_and_clamp** (*Int*) – An integer containing two packed values specifying a mask for logically
    splitting warps into sub-segments and an upper bound for clamping the
    source lane index.
  * **kind** (*ShflKind*) – The kind of shuffle, can be idx, up, down, or bfly
* **Returns:**
  The shuffled value
* **Return type:**
  Numeric

### cutlass.cute.arch.shuffle_sync_down(value: Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA), offset: int | Integer, mask: int | Integer = 4294967295, mask_and_clamp: int | Integer = 31, , kind: ShflKind = ShflKind.down, loc=None, ip=None) → Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA)

Shuffles a value within the threads of a warp.

* **Parameters:**
  * **value** (*Numeric* *or* [*TensorSSA*](cute.md#cutlass.cute.TensorSSA)) – The value to shuffle
  * **mask** (*Int*) – A mask describing the threads participating in this operation
  * **offset** (*Int*) – A source lane or a source lane offset depending on kind
  * **mask_and_clamp** (*Int*) – An integer containing two packed values specifying a mask for logically
    splitting warps into sub-segments and an upper bound for clamping the
    source lane index.
  * **kind** (*ShflKind*) – The kind of shuffle, can be idx, up, down, or bfly
* **Returns:**
  The shuffled value
* **Return type:**
  Numeric

### cutlass.cute.arch.shuffle_sync_up(value: Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA), offset: int | Integer, mask: int | Integer = 4294967295, mask_and_clamp: int | Integer = 31, , kind: ShflKind = ShflKind.up, loc=None, ip=None) → Numeric | [TensorSSA](cute.md#cutlass.cute.TensorSSA)

Shuffles a value within the threads of a warp.

* **Parameters:**
  * **value** (*Numeric* *or* [*TensorSSA*](cute.md#cutlass.cute.TensorSSA)) – The value to shuffle
  * **mask** (*Int*) – A mask describing the threads participating in this operation
  * **offset** (*Int*) – A source lane or a source lane offset depending on kind
  * **mask_and_clamp** (*Int*) – An integer containing two packed values specifying a mask for logically
    splitting warps into sub-segments and an upper bound for clamping the
    source lane index.
  * **kind** (*ShflKind*) – The kind of shuffle, can be idx, up, down, or bfly
* **Returns:**
  The shuffled value
* **Return type:**
  Numeric

### cutlass.cute.arch.store(ptr, val: Numeric | Value, , level1_eviction_priority: Literal['evict_normal', 'evict_first', 'evict_last', 'evict_no_allocate', 'evict_unchanged'] | None = None, cop: Literal['wb', 'cg', 'cs', 'wt'] | None = None, ss: Literal['cta', 'cluster'] | None = None, sem: Literal['relaxed', 'release'] | None = None, scope: Literal['gpu', 'cta', 'cluster', 'sys'] | None = None, loc=None, ip=None) → None

Store a value to a memory location.

* **Parameters:**
  * **ptr** – Pointer to store to. Supports:
    - ir.Value (LLVM pointer)
    - cute.ptr (_Pointer instance)
  * **val** (*Union* *[**Numeric* *,* *ir.Value* *]*) – Value to store (scalar Numeric or vector ir.Value)
  * **level1_eviction_priority** – L1 cache eviction policy string literal:
    “evict_normal” : .level1::eviction_priority = .L1::evict_normal
    “evict_first” : .level1::eviction_priority = .L1::evict_first
    “evict_last” : .level1::eviction_priority = .L1::evict_last
    “evict_no_allocate” : .level1::eviction_priority = .L1::no_allocate
    “evict_unchanged” : .level1::eviction_priority = .L1::evict_unchanged
  * **cop** – Store cache modifier string literal:
  * **ss** – Shared memory space string literal:
    “cta” : .ss = .shared::cta
    “cluster” : .ss = .shared::cluster
    None : .ss = .global
  * **sem** – Memory semantic string literal:
  * **scope** – Memory scope string literal:

### cutlass.cute.arch.sync_threads(, loc=None, ip=None) → None

Synchronizes all threads within a CTA.

### cutlass.cute.arch.sync_warp(mask: int | Integer = 4294967295, , loc=None, ip=None) → None

Performs a warp-wide sync with an optional mask.

### cutlass.cute.arch.thread_idx(, loc=None, ip=None) → Tuple[Int32, Int32, Int32]

Returns the thread index within a CTA.

### cutlass.cute.arch.vote_all_sync(pred: Boolean, mask: int | Integer = 4294967295, , loc=None, ip=None) → Boolean

True if source predicate is True for all non-exited threads in mask. Negate the source
predicate to compute .none.

* **Parameters:**
  * **pred** (*Boolean*) – The predicate value for the current thread
  * **mask** (*Int* *,* *optional*) – A 32-bit integer mask specifying which threads participate, defaults to all
    threads (0xFFFFFFFF)
* **Returns:**
  A boolean value indicating if the source predicate is True for all non-exited
  threads in mask
* **Return type:**
  Boolean

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

### cutlass.cute.arch.vote_any_sync(pred: Boolean, mask: int | Integer = 4294967295, , loc=None, ip=None) → Boolean

True if source predicate is True for any non-exited threads in mask. Negate the source
predicate to compute .none.

* **Parameters:**
  * **pred** (*Boolean*) – The predicate value for the current thread
  * **mask** (*Int* *,* *optional*) – A 32-bit integer mask specifying which threads participate, defaults to all
    threads (0xFFFFFFFF)
* **Returns:**
  A boolean value indicating if the source predicate is True for all non-exited
  threads in mask
* **Return type:**
  Boolean

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

### cutlass.cute.arch.vote_ballot_sync(pred: Boolean, mask: int | Integer = 4294967295, , loc=None, ip=None) → Int32

Performs a ballot operation across the warp.

It copies the predicate from each thread in mask into the corresponding bit position of
destination register d, where the bit position corresponds to the thread’s lane id.

* **Parameters:**
  * **pred** (*Boolean*) – The predicate value for the current thread
  * **mask** (*Int* *,* *optional*) – A 32-bit integer mask specifying which threads participate, defaults to all threads (0xFFFFFFFF)
* **Returns:**
  A 32-bit integer where each bit represents a thread’s predicate value
* **Return type:**
  Int32

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-vote-sync).

### cutlass.cute.arch.vote_uni_sync(pred: Boolean, mask: int | Integer = 4294967295, , loc=None, ip=None) → Boolean

True f source predicate has the same value in all non-exited threads in mask. Negating
the source predicate also computes .uni

* **Parameters:**
  * **pred** (*Boolean*) – The predicate value for the current thread
  * **mask** (*Int* *,* *optional*) – A 32-bit integer mask specifying which threads participate, defaults to all
    threads (0xFFFFFFFF)
* **Returns:**
  A boolean value indicating if the source predicate is True for all non-exited
  threads in mask
* **Return type:**
  Boolean

### cutlass.cute.arch.warp_idx(, loc=None, ip=None) → Int32

Returns the warp index within a CTA.

### cutlass.cute.arch.warp_redux_sync(value: Numeric, kind: Literal['fmax', 'fmin', 'max', 'min', 'add', 'xor', 'or', 'and'], mask_and_clamp: int | Integer = 4294967295, , abs: bool = None, nan: bool = None, loc=None, ip=None) → Numeric

Perform warp-level reduction operation across threads.

Reduces values from participating threads in a warp according to the specified operation.
All threads in the mask receive the same result.

* **Parameters:**
  * **value** (*Numeric*) – Input value to reduce
  * **kind** (*Literal* *[* *"add"* *,*  *"and"* *,*  *"max"* *,*  *"min"* *,*  *"or"* *,*  *"xor"* *,*  *"fmin"* *,*  *"fmax"* *]*) – Reduction operation. Supported operations:
    - Integer types (Int32/Uint32): “add”, “and”, “max”, “min”, “or”, “xor”
    - Float types (Float32): “fmax”, “fmin” (or “max”/”min” which auto-convert to “fmax”/”fmin”)
  * **mask_and_clamp** (*Int*) – Warp participation mask (default: FULL_MASK = 0xFFFFFFFF)
  * **abs** (*bool*) – Apply absolute value before reduction (float types only)
  * **nan** (*Optional* *[**bool* *]*) – Enable NaN propagation for fmax/fmin operations (float types only)
* **Returns:**
  Reduced value (same for all participating threads)
* **Return type:**
  Numeric

### cutlass.cute.arch.warpgroup_reg_alloc(reg_count: int, , loc=None, ip=None) → None

### cutlass.cute.arch.warpgroup_reg_dealloc(reg_count: int, , loc=None, ip=None) → None
