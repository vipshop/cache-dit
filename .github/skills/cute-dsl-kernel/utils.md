# cutlass.utils

The `cutlass.utils` module contains utilities for developing kernels with CuTe DSL.

<a id="module-cutlass.utils"></a>

### *class* cutlass.utils.ClcDynamicPersistentTileScheduler(params: [ClcDynamicPersistentTileSchedulerParams](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams), cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32, clc_response_ptr: Pointer, block_idx: Tuple[Integer, Integer, Integer])

Bases: `object`

A scheduler for dynamic persistent tile execution in CUTLASS/CuTe kernels.

* **Variables:**
  * **params** – Tile schedule related params, including cluster shape.
  * **cta_id_in_cluster** – ID of the CTA within its cluster
  * **\_num_tiles_executed** – Counter for executed tiles

#### \_\_init_\_(params: [ClcDynamicPersistentTileSchedulerParams](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams), cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32, clc_response_ptr: Pointer, block_idx: Tuple[Integer, Integer, Integer])

Initializes the ClcDynamicPersistentTileScheduler with the given parameters.

* **Parameters:**
  * **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape.
  * **cta_id_in_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
  * **num_tiles_executed** (*Int32*) – Counter for executed tiles.
  * **clc_response_ptr** (*cute.Pointer*) – Pointer of the clc rsponse.
  * **block_idx** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The block index.

#### advance_to_next_work(mbarrier_addr, loc=None, ip=None)

#### create(block_idx: Tuple[Integer, Integer, Integer], grid_dim: Tuple[Integer, Integer, Integer], clc_response_ptr: Pointer, , loc=None, ip=None)

Initialize the dynamic persistent tile scheduler.

* **Parameters:**
  * **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Parameters for the persistent
    tile scheduler.
  * **block_idx** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d block index in the format (bidx, bidy, bidz).
  * **grid_dim** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d grid dimensions for kernel launch.
* **Returns:**
  A ClcDynamicPersistentTileScheduler object.
* **Return type:**
  [ClcDynamicPersistentTileScheduler](#cutlass.utils.ClcDynamicPersistentTileScheduler)

#### get_current_work(, loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### get_grid_shape(, loc=None, ip=None) → Tuple[Integer, Integer, Integer]

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

* **Parameters:**
  **params** ([*ClcDynamicPersistentTileSchedulerParams*](#cutlass.utils.ClcDynamicPersistentTileSchedulerParams)) – Parameters for grid shape calculation.
* **Returns:**
  The calculated 3d grid shape.
* **Return type:**
  Tuple[Integer, Integer, Integer]

#### initial_work_tile_info(, loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### *property* num_tiles_executed *: Int32*

#### work_tile_info_from_clc_response(result_addr: Pointer, , loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

Simulates parsing CLC response data in Python.
result_addr: 16-byte response data (simulating shared memory access)

### *class* cutlass.utils.ClcDynamicPersistentTileSchedulerParams(problem_shape_ntile_mnl: int | Integer | Tuple[Shape, ...], cluster_shape_mnk: int | Integer | Tuple[Shape, ...], , loc=None, ip=None)

Bases: `object`

A class to represent parameters for a dynamic persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

* **Variables:**
  **cluster_shape_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).

#### \_\_init_\_(problem_shape_ntile_mnl: int | Integer | Tuple[Shape, ...], cluster_shape_mnk: int | Integer | Tuple[Shape, ...], , loc=None, ip=None)

Initializes the ClcDynamicPersistentTileSchedulerParams with the given parameters.

* **Parameters:**
  * **problem_shape_ntile_mnl** (*cute.Shape*) – The shape of the problem in terms of
    number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
  * **cluster_shape_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
* **Raises:**
  **ValueError** – If cluster_shape_k is not 1.

#### get_grid_shape(, loc=None, ip=None) → Tuple[Integer, Integer, Integer]

Computes the grid shape based on the problem shape and cluster shape.

* **Returns:**
  the grid is the CTA numbers that has aligned with cluster shape.

### *class* cutlass.utils.GroupSearchResult(group_idx: Int32, cta_tile_idx_m: Int32, cta_tile_idx_n: Int32, problem_shape_m: Int32, problem_shape_n: Int32, problem_shape_k: Int32, cta_tile_count_k: Int32)

Bases: `object`

The result of the group search for grouped gemm.

* **Parameters:**
  * **group_idx** (*Int32*) – The result group index
  * **cta_tile_idx_m** (*Int32*) – CTA tile index along M dimension after rasterization
  * **cta_tile_idx_n** (*Int32*) – CTA tile index along N dimension after rasterization
  * **problem_shape_m** (*Int32*) – The M dimension of the gemm problem
  * **problem_shape_n** (*Int32*) – The N dimension of the gemm problem
  * **problem_shape_k** (*Int32*) – The K dimension of the gemm problem
  * **cta_tile_count_k** (*Int32*) – Number of tiles along K dimension

#### \_\_init_\_(group_idx: Int32, cta_tile_idx_m: Int32, cta_tile_idx_n: Int32, problem_shape_m: Int32, problem_shape_n: Int32, problem_shape_k: Int32, cta_tile_count_k: Int32) → None

### *class* cutlass.utils.GroupedGemmGroupSearchState(start_group_idx: Int32, tile_count_prev_group: Int32, tile_count_searched: Int32, found: Boolean)

Bases: `object`

The state of group index search for grouped gemm.

The state will be initialized once and updated in every round of group index search.

* **Parameters:**
  * **start_group_idx** (*Int32*) – The group idx to start the search with
  * **tile_count_prev_group** (*Int32*) – Number of tiles before the matched group
  * **tile_count_searched** (*Int32*) – Number of tiles we have searched. When the matched group is found,
    it records the number of tiles including the matched group

#### \_\_init_\_(start_group_idx: Int32, tile_count_prev_group: Int32, tile_count_searched: Int32, found: Boolean) → None

### *class* cutlass.utils.GroupedGemmTileSchedulerHelper(\*\*kwargs)

Bases: `object`

A helper to translate the raw block index (x, y, z) from tile scheduler to real CTA tile index for grouped gemm.

* **Parameters:**
  * **group_count** (*int*) – Number of groups in current grouped gemm problem
  * **tile_sched_params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameter used to create the tile scheduler this helper works with
  * **cluster_tile_shape_mnk** (*tuple* *[**int* *,* *int* *,* *int* *]*) – The shape of cluster tile as (m, n, k)
  * **search_state** ([*GroupedGemmGroupSearchState*](#cutlass.utils.GroupedGemmGroupSearchState)) – The initial search state

#### \_\_init_\_(group_count: int, tile_sched_params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), cluster_tile_shape_mnk: tuple[int, int, int], search_state: [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)) → None

#### \_compute_cta_tile_coord(cluster_tile_idx: Int32, cta_tile_coord_in_cluster: tuple, cluster_tile_count_m: Int32, cluster_tile_count_n: Int32) → tuple

Compute CTA tile indices along M and N dimensions based on the linear index within a group.

It uses the AlongM mode to decompose the linear index onto M and N dimensions.

* **Parameters:**
  * **cluster_tile_idx** (*Int32*) – The linear index within a group
  * **cta_tile_coord_in_cluster** (*tuple* *of* *Int32*) – CTA indices along M and N dimensions within a cluster
  * **cluster_tile_count_m** (*Int32*) – The number of clusters along M dimension of the matched group
  * **cluster_tile_count_n** (*Int32*) – The number of clusters along N dimension of the matched group
* **Returns:**
  A tuple containing CTA tile indices along M and N dimensions
* **Return type:**
  tuple of (Int32, Int32)

#### \_get_cluster_tile_count_mn(problem_shape: [Tensor](cute.md#cutlass.cute.Tensor)) → Int32

Compute total cluster count.

* **Parameters:**
  **problem_shape** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor containing problem shape (m, n, k, l)
* **Returns:**
  The total cluster tile count for M and N dimensions
* **Return type:**
  Int32

#### \_get_problem_for_group(problem_shape_mnkl: [Tensor](cute.md#cutlass.cute.Tensor), group_idx: Int32) → [Tensor](cute.md#cutlass.cute.Tensor)

Load gemm problem (m,n,k,l) for the specified group from global memory to register.

* **Parameters:**
  * **problem_shape_mnkl** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor in global memory with layout (group_count, 4):(4, 1)
  * **group_idx** (*Int32*) – The index of the group to load
* **Returns:**
  The problem shape tensor for the specified group
* **Return type:**
  [cute.Tensor](cute.md#cutlass.cute.Tensor)

#### \_group_search(linear_idx: Int32, problem_shape_mnkl: [Tensor](cute.md#cutlass.cute.Tensor), init_group_idx: Int32, init_tile_count_searched: Int32) → [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

Search which group the linear index belongs to.

* **Parameters:**
  * **linear_idx** (*Int32*) – The linear index to be decomposed
  * **problem_shape_mnkl** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor containing gemm problem size (M, N, K, L) for all groups
  * **init_group_idx** (*Int32*) – The group idx to start the search with
  * **init_tile_count_searched** (*Int32*) – The number of tiles we have searched
* **Returns:**
  The updated search state
* **Return type:**
  [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

#### \_group_search_and_load_problem_shape(linear_idx: Int32, problem_shape_mnkl: [Tensor](cute.md#cutlass.cute.Tensor), start_group_idx: Int32, tile_count_searched: Int32) → Tuple[Int32, [Tensor](cute.md#cutlass.cute.Tensor)]

Perform group search and load problem shape for the matched group.

* **Parameters:**
  * **linear_idx** (*Int32*) – The linear index to be decomposed
  * **problem_shape_mnkl** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor containing gemm problem size (M, N, K, L) for all groups
  * **start_group_idx** (*Int32*) – The group idx to start the search with
  * **tile_count_searched** (*Int32*) – The number of tiles we have searched
* **Returns:**
  A tuple containing the final group index and the problem shape tensor
* **Return type:**
  Tuple[Int32, [cute.Tensor](cute.md#cutlass.cute.Tensor)]

#### \_prefix_sum(value_per_thread: Int32) → Int32

Perform prefix sum within a full warp.

* **Parameters:**
  **value_per_thread** (*Int32*) – The value for this thread to contribute to the prefix sum
* **Returns:**
  The prefix sum result for this thread
* **Return type:**
  Int32

#### delinearize_z(cta_tile_coord: tuple, problem_shape_mnkl: [Tensor](cute.md#cutlass.cute.Tensor)) → [GroupSearchResult](#cutlass.utils.GroupSearchResult)

Delinearize the linear z index and return GroupSearchResult.

This function should be used by warps that need to know the CTA tile index on M and N dimensions.

* **Parameters:**
  * **cta_tile_coord** (*tuple* *of* *Int32*) – The raw CTA coordinate from tile scheduler
  * **problem_shape_mnkl** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor containing gemm problem size (M, N, K, L) for each group
* **Returns:**
  The search result containing group index and tile coordinates
* **Return type:**
  [GroupSearchResult](#cutlass.utils.GroupSearchResult)

#### search_cluster_tile_count_k(cta_tile_coord: tuple, problem_shape_mnkl: [Tensor](cute.md#cutlass.cute.Tensor)) → Tuple[Int32, Int32]

Search the matched group for given linear index and compute the number of tiles along K dimension for the matched group.

This function should be used by warps that are only interested in the number of tiles along K dimension.

* **Parameters:**
  * **cta_tile_coord** (*tuple* *of* *Int32*) – The raw CTA coordinate from tile scheduler
  * **problem_shape_mnkl** ([*cute.Tensor*](cute.md#cutlass.cute.Tensor)) – Tensor containing gemm problem size (M, N, K, L) for all groups
* **Returns:**
  A tuple containing cluster count along K dimension and the group index
* **Return type:**
  Tuple[Int32, Int32]

### *class* cutlass.utils.HardwareInfo(device_id: int = 0)

Bases: `object`

device_id: CUDA device ID to get the hardware info.

#### \_\_init_\_(device_id: int = 0)

#### \_checkCudaErrors(result) → None

#### \_cudaGetErrorEnum(error) → str

#### \_cuda_driver_version_ge(major: int, minor: int) → bool

#### \_cuda_driver_version_lt(major: int, minor: int) → bool

#### \_empty_kernel()

#### \_get_device_function() → CUfunction

Get a device function by compiling a dummy kernel using cuteDSL pipeline.

#### \_host_function()

#### get_device_multiprocessor_count() → int

#### get_l2_cache_size_in_bytes() → int

#### get_max_active_clusters(cluster_size: int, stream: CUstream = None) → int

Get the maximum number of active clusters for a given cluster size.

When a stream from a green context is provided, the occupancy calculation
will reflect the reduced SM partition of the green context.

* **Parameters:**
  * **cluster_size** (*int*) – Number of blocks per cluster (must be between 1 and 32)
  * **stream** (*driver.CUstream* *,* *optional*) – Optional CUDA stream handle. If provided (especially from a green context),
    the occupancy calculation reflects the stream’s SM partition.
* **Returns:**
  Maximum number of active clusters
* **Return type:**
  int

### *class* cutlass.utils.LayoutEnum(\*values)

Bases: `Enum`

#### COL_MAJOR *= 'col_major'*

#### ROW_MAJOR *= 'row_major'*

#### *static* from_tensor(tensor: [Tensor](cute.md#cutlass.cute.Tensor)) → [LayoutEnum](#cutlass.utils.LayoutEnum)

#### is_k_major_a()

#### is_k_major_b()

#### is_m_major_a()

#### is_m_major_c()

#### is_n_major_b()

#### is_n_major_c()

#### mma_major_mode()

#### sm90_mma_major_mode()

### *class* cutlass.utils.PersistentTileSchedulerParams

Bases: `object`

A class to represent parameters for a persistent tile scheduler.

This class is designed to manage and compute the layout of clusters and tiles
in a batched gemm problem.

* **Variables:**
  * **cluster_shape_mn** – Shape of the cluster in (m, n) dimensions (K dimension cta count must be 1).
  * **problem_layout_ncluster_mnl** – Layout of the problem in terms of
    number of clusters in (m, n, l) dimensions.

#### \_\_init_\_()

Initializes the PersistentTileSchedulerParams with the given parameters.

* **Parameters:**
  * **problem_shape_ntile_mnl** (*cute.Shape*) – The shape of the problem in terms of
    number of CTA (Cooperative Thread Array) in (m, n, l) dimensions.
  * **cluster_shape_mnk** (*cute.Shape*) – The shape of the cluster in (m, n) dimensions.
  * **swizzle_size** (*int*) – Swizzling size in the unit of cluster. 1 means no swizzle
  * **raster_along_m** (*bool*) – Rasterization order of clusters. Only used when swizzle_size > 1.
    True means along M, false means along N.
* **Raises:**
  **ValueError** – If cluster_shape_k is not 1.

#### get_grid_shape(max_active_clusters: Int32, , loc=None, ip=None) → Tuple[Integer, Integer, Integer]

Computes the grid shape based on the maximum active clusters allowed.

* **Parameters:**
  **max_active_clusters** (*Int32*) – The maximum number of active clusters that
  can run in one wave.
* **Returns:**
  A tuple containing the grid shape in (m, n, persistent_clusters).
  - m: self.cluster_shape_m.
  - n: self.cluster_shape_n.
  - persistent_clusters: Number of persistent clusters that can run.

### *class* cutlass.utils.SmemAllocator

Bases: `object`

A helper class for managing shared memory allocation on GPU.

This class manages shared memory and provides APIs for allocation of raw bytes,
numeric types, arrays, and tensors with specified layouts and alignments.

#### NOTE
- The base pointer is aligned to 1024 bytes upon initialization.
- There is no need to explicitly specify shared memory size in kernel launch.
- Currently only supports static layouts. Dynamic layouts are not supported.

**Examples**:

```python
smem = SmemAllocator()

# Allocate raw bytes
buf_ptr = smem.allocate(100)  # 100 bytes

# Allocate numeric type
int8_ptr = smem.allocate(Int8)  # 1 byte

# Define a struct
@cute.struct
class SharedStorage:
    alpha: cutlass.Float32
    x: cutlass.Int32

# Allocate struct
struct_ptr = smem.allocate(SharedStorage)  # 8 bytes

# use of struct members
struct_ptr.alpha = 1.0
struct_ptr.x = 2

# Allocate array
int8_array = smem.allocate_array(Int8, 10)  # 10 bytes

# Allocate tensor
layout = cute.make_layout((16, 16))
tensor = smem.allocate_tensor(Int8, layout)  # 256 bytes
```

#### \_\_init_\_()

Initialize a new SmemAllocator instance.

Creates a new shared memory allocator with a base pointer aligned to 1024 bytes.
Tracks the allocator instance for memory management.

* **Parameters:**
  * **loc** (*Optional* *[**ir.Location* *]*) – Source location information for debugging, defaults to None
  * **ip** (*Optional* *[**ir.InsertionPoint* *]*) – Insertion point for MLIR operations, defaults to None

#### allocate(size_or_type: int, byte_alignment: int, , loc=None, ip=None) → cute.Pointer

#### allocate(size_or_type: Type[Numeric], byte_alignment: int, , loc=None, ip=None) → cute.Pointer

#### allocate(size_or_type: [cute.struct](cute.md#cutlass.cute.struct), byte_alignment: int, , loc=None, ip=None) → cute.Pointer

Allocate a block of memory with specified size and alignment.

This method allocates a block of shared memory with the specified size and alignment requirements.
It supports allocating raw bytes, numeric types(as scalar value), and struct types.

* **Parameters:**
  * **size_or_type** (*Union* *[**int* *,* *Type* *[**Numeric* *]* *,* [*cute.struct*](cute.md#cutlass.cute.struct) *]*) – The allocation specification, which can be:
    - An integer specifying the number of bytes to allocate
    - A Numeric type (e.g., Int8, Float32) to allocate space for one element
    - A struct type to allocate space for the entire struct
  * **byte_alignment** (*int* *,* *optional*) – The minimum byte alignment requirement for the allocation, defaults to 1
  * **loc** (*Optional* *[**ir.Location* *]*) – Source location information for debugging, defaults to None
  * **ip** (*Optional* *[**ir.InsertionPoint* *]*) – Insertion point for MLIR operations, defaults to None
* **Returns:**
  For raw bytes and numeric types, returns a pointer to the allocated memory.
  For struct types, returns an initialized struct instance at the allocated location.
* **Return type:**
  cute.Pointer
* **Raises:**
  * **ValueError** – If size is negative or alignment is less than 1
  * **TypeError** – If size_or_type is not an integer, Numeric type, or struct
  * **RuntimeError** – If allocation would exceed available shared memory

#### allocate_array(element_type: Type[Numeric], num_elems: int = 1, , loc=None, ip=None)

Allocate an array of elements in shared memory.

* **Parameters:**
  * **element_type** (*Type* *[**Numeric* *]*) – The type of elements to allocate
  * **num_elems** (*int* *,* *optional*) – Number of elements to allocate, defaults to 1
* **Returns:**
  Pointer to the start of the allocated array
* **Return type:**
  cute.Pointer
* **Raises:**
  * **ValueError** – If num_elems is less than 1
  * **TypeError** – If element_type is not a Numeric type

#### allocate_tensor(element_type: Type[Numeric], layout: int | [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), byte_alignment: int = 1, swizzle: [Swizzle](cute.md#cutlass.cute.Swizzle) | None = None, , loc=None, ip=None)

Allocate a tensor in shared memory.

Note: Currently only supports static layouts. Dynamic layouts are not supported.

* **Parameters:**
  * **element_type** (*Type* *[**Numeric* *]*) – The type of elements in the tensor
  * **layout** (*Union* *[**int* *,* [*cute.Layout*](cute.md#cutlass.cute.Layout) *,* [*cute.ComposedLayout*](cute.md#cutlass.cute.ComposedLayout) *]*) – The layout specification for the tensor. Must be a static layout.
  * **byte_alignment** (*int* *,* *optional*) – The byte alignment requirement, defaults to 1
  * **swizzle** ([*cute.Swizzle*](cute.md#cutlass.cute.Swizzle) *,* *optional*) – Swizzle for position-dependent swizzling, defaults to None
* **Returns:**
  The allocated tensor with specified properties
* **Return type:**
  [cute.Tensor](cute.md#cutlass.cute.Tensor)
* **Raises:**
  * **TypeError** – If element_type is not a Numeric type or if swizzle conflicts with layout
  * **ValueError** – If allocation is not byte-aligned
  * **NotImplementedError** – If dynamic layout is specified

#### *static* capacity_in_bytes(compute_capability: str | None = None) → int

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

* **Parameters:**
  **compute_capability** (*Optional* *[**str* *]*) – The compute capability string (e.g. “70”, “75”, “80”)
* **Returns:**
  The shared memory capacity in bytes
* **Return type:**
  int
* **Raises:**
  **ValueError** – If the compute capability is not supported

### *class* cutlass.utils.StaticPersistentRuntimeTileScheduler(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), num_persistent_clusters: Int32, current_work_linear_idx: Int32, cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32, inner_mode: int = 1)

Bases: [`StaticPersistentTileScheduler`](#cutlass.utils.StaticPersistentTileScheduler)

A scheduler for static persistent runtime tile execution in CUTLASS/CuTe kernels.
This scheduler will always launch all the SMs and the scheduler will generate the real tile info for each SM.

* **Variables:**
  * **params** – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
  * **num_persistent_clusters** – Number of persistent clusters that can be launched
  * **cta_id_in_cluster** – ID of the CTA within its cluster
  * **\_num_tiles_executed** – Counter for executed tiles
  * **\_current_work_linear_idx** – Current cluster index

#### \_\_init_\_(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), num_persistent_clusters: Int32, current_work_linear_idx: Int32, cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32, inner_mode: int = 1)

Initializes the StaticPersistentTileScheduler with the given parameters.

* **Parameters:**
  * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
  * **num_persistent_clusters** (*Int32*) – Number of persistent clusters that can be launched.
  * **current_work_linear_idx** (*Int32*) – Current cluster index.
  * **cta_id_in_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
  * **num_tiles_executed** (*Int32*) – Counter for executed tiles.
  * **inner_mode** (*int*) – The inner mode along which the linear index will be decomposed first.

#### \_get_current_work_for_linear_idx(current_work_linear_idx: Int32, , loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

* **Parameters:**
  **current_work_linear_idx** (*Int32*) – The linear index of the current work.
* **Returns:**
  An object containing information about the current tile coordinates
  and validity status.
* **Return type:**
  [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### *static* create(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), block_idx: Tuple[Integer, Integer, Integer], grid_dim: Tuple[Integer, Integer, Integer], inner_mode: int = 1, , loc=None, ip=None)

Initialize the static persistent tile scheduler.

* **Parameters:**
  * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for the persistent
    tile scheduler.
  * **block_idx** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d block index in the format (bidx, bidy, bidz).
  * **grid_dim** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d grid dimensions for kernel launch.
  * **inner_mode** (*int*) – The inner mode along which the linear index will be decomposed first.
* **Returns:**
  A StaticPersistentRuntimeTileScheduler object.
* **Return type:**
  [StaticPersistentRuntimeTileScheduler](#cutlass.utils.StaticPersistentRuntimeTileScheduler)

### *class* cutlass.utils.StaticPersistentTileScheduler(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), num_persistent_clusters: Int32, current_work_linear_idx: Int32, cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32)

Bases: `object`

A scheduler for static persistent tile execution in CUTLASS/CuTe kernels.

* **Variables:**
  * **params** – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl
  * **num_persistent_clusters** – Number of persistent clusters that can be launched
  * **cta_id_in_cluster** – ID of the CTA within its cluster
  * **\_num_tiles_executed** – Counter for executed tiles
  * **\_current_work_linear_idx** – Current cluster index

#### \_\_init_\_(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), num_persistent_clusters: Int32, current_work_linear_idx: Int32, cta_id_in_cluster: int | Integer | None | Tuple[Coord, ...], num_tiles_executed: Int32)

Initializes the StaticPersistentTileScheduler with the given parameters.

* **Parameters:**
  * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Tile schedule related params, including cluster shape and problem_layout_ncluster_mnl.
  * **num_persistent_clusters** (*Int32*) – Number of persistent clusters that can be launched.
  * **current_work_linear_idx** (*Int32*) – Current cluster index.
  * **cta_id_in_cluster** (*cute.Coord*) – ID of the CTA within its cluster.
  * **num_tiles_executed** (*Int32*) – Counter for executed tiles.

#### \_get_cluster_work_idx_with_fastdivmod(current_work_linear_idx: Int32, , loc=None, ip=None) → Tuple[Int32, Int32, Int32]

FastDivmod optimized CLUSTER coordinate calculation.

CRITICAL: This should mimic problem_layout_ncluster_mnl.get_hier_coord()
which returns CLUSTER coordinates, not tile coordinates!

* **Parameters:**
  **current_work_linear_idx** (*Int32*) – Linear index in the work space
* **Returns:**
  Cluster coordinates (m, n, l) or None if FastDivmod not available
* **Return type:**
  Tuple[Int32, Int32, Int32] or None

#### \_get_current_work_for_linear_idx(current_work_linear_idx: Int32, , loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

Compute current tile coord given current_work_linear_idx and cta_id_in_cluster.

* **Parameters:**
  **current_work_linear_idx** (*Int32*) – The linear index of the current work.
* **Returns:**
  An object containing information about the current tile coordinates
  and validity status.
* **Return type:**
  [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### advance_to_next_work(, advance_count: int = 1, loc=None, ip=None)

#### *static* create(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), block_idx: Tuple[Integer, Integer, Integer], grid_dim: Tuple[Integer, Integer, Integer], , loc=None, ip=None)

Initialize the static persistent tile scheduler.

* **Parameters:**
  * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for the persistent
    tile scheduler.
  * **block_idx** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d block index in the format (bidx, bidy, bidz).
  * **grid_dim** (*Tuple* *[**Integer* *,* *Integer* *,* *Integer* *]*) – The 3d grid dimensions for kernel launch.
* **Returns:**
  A StaticPersistentTileScheduler object.
* **Return type:**
  [StaticPersistentTileScheduler](#cutlass.utils.StaticPersistentTileScheduler)

#### get_current_work(, loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### *static* get_grid_shape(params: [PersistentTileSchedulerParams](#cutlass.utils.PersistentTileSchedulerParams), max_active_clusters: Int32, , loc=None, ip=None) → Tuple[Integer, Integer, Integer]

Calculates the grid shape to be launched on GPU using problem shape,
threadblock shape, and active cluster size.

* **Parameters:**
  * **params** ([*PersistentTileSchedulerParams*](#cutlass.utils.PersistentTileSchedulerParams)) – Parameters for grid shape calculation.
  * **max_active_clusters** (*Int32*) – Maximum active clusters allowed.
* **Returns:**
  The calculated 3d grid shape.
* **Return type:**
  Tuple[Integer, Integer, Integer]

#### initial_work_tile_info(, loc=None, ip=None) → [WorkTileInfo](#cutlass.utils.WorkTileInfo)

#### *property* num_tiles_executed *: Int32*

### *class* cutlass.utils.TensorMapManager(tensormap_update_mode: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode), bytes_per_tensormap: int)

Bases: `object`

Manages TensorMap operations including initialization and updates.
Provides utilities to convert tensormap pointer to across different memory spaces.

#### \_\_init_\_(tensormap_update_mode: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode), bytes_per_tensormap: int) → None

#### bytes_per_tensormap *: int*

#### fence_tensormap_initialization(, loc=None, ip=None) → None

#### fence_tensormap_update(tensormap_ptr: Pointer, , loc=None, ip=None) → None

#### get_tensormap_ptr(ptr: Pointer, address_space=AddressSpace.gmem, , loc=None, ip=None) → Pointer

#### init_tensormap_from_atom(copy_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), dst_ptr: Pointer, warp_id: int, , loc=None, ip=None) → None

#### tensormap_update_mode *: [TensorMapUpdateMode](#cutlass.utils.TensorMapUpdateMode)*

#### update_tensormap(tensor_gmem: Tuple[[Tensor](cute.md#cutlass.cute.Tensor), ...], tma_copy_atom: Tuple[[CopyAtom](cute.md#cutlass.cute.CopyAtom), ...], tensormap_gmem_ptr: Tuple[Pointer, ...], warp_id: int, tensormap_smem_ptr: Tuple[Pointer, ...], , loc=None, ip=None) → None

### *class* cutlass.utils.TensorMapUpdateMode(\*values)

Bases: `Enum`

Enum class defining tensor map update modes.

Modes:
GMEM: Update tensormap in global memory
SMEM: Load tensormap from global memory to shared memory,
update it in shared memory, then store back to global memory

#### GMEM *= 1*

#### SMEM *= 2*

### *class* cutlass.utils.TmemAllocator(alloc_result_dst_smem_ptr: Pointer, barrier_for_retrieve: [NamedBarrier](pipeline.md#cutlass.pipeline.NamedBarrier), allocator_warp_id: int = 0, is_two_cta: bool = False, num_allocated_columns: int = 0, two_cta_tmem_dealloc_mbar_ptr: Pointer | None = None)

Bases: `object`

A class for managing tensor memory allocation on GPUs.

This class manages allocation/deallocation of tensor memory, including the mbarrier
synchronization for two cta use case.

* **Variables:**
  * **\_alloc_result_dst_smem_ptr** – The smem pointer that holds the base address of allocated tensor memory.
  * **\_barrier_for_retrieve** – The barrier for retrieving tensor memory ptr.
  * **\_allocator_warp_id** – The warp id of the allocator warp.
  * **\_is_two_cta** – Whether the allocator is for two cta.
  * **\_num_allocated_columns** – The number of columns allocated in the tensor memory.
  * **\_two_cta_tmem_dealloc_mbar_ptr** – The mbarrier pointer required when deallocating tensor memory for two cta.
  * **\_arch** – The architecture of the GPU.

#### \_\_init_\_(alloc_result_dst_smem_ptr: Pointer, barrier_for_retrieve: [NamedBarrier](pipeline.md#cutlass.pipeline.NamedBarrier), allocator_warp_id: int = 0, is_two_cta: bool = False, num_allocated_columns: int = 0, two_cta_tmem_dealloc_mbar_ptr: Pointer | None = None)

Initialize a TmemAllocator instance for managing tensor memory on Blackwell GPUs.

This initializer sets up the allocator’s state, including the shared memory (smem) pointer
holding the base address of the allocated tensor memory, barrier synchronization for
retrieving the tensor memory pointer, allocator warp ID, whether the allocator is being used
for a 2-SM configuration, number of allocated columns in tensor
memory, and the optional mbarrier pointer for deallocation in the 2-SM case.

If is_two_cta is set to True, this will initialize the mbarrier pointer required for tensor
memory deallocation across two CTAs.

* **Parameters:**
  * **alloc_result_dst_smem_ptr** (*cute.Pointer*) – The shared memory pointer that holds the base address of allocated tensor memory.
  * **barrier_for_retrieve** ([*pipeline.NamedBarrier*](pipeline.md#cutlass.pipeline.NamedBarrier)) – The named barrier for retrieving the tensor memory pointer.
  * **allocator_warp_id** (*int* *,* *optional*) – The warp ID of the allocator warp, defaults to 0.
  * **is_two_cta** (*bool* *,* *optional*) – Whether the allocator should coordinate two CTAs, defaults to False.
  * **num_allocated_columns** (*int* *,* *optional*) – The number of columns allocated in tensor memory, defaults to 0.
  * **two_cta_tmem_dealloc_mbar_ptr** (*cute.Pointer* *,* *optional*) – The mbarrier pointer required for two-CTA tensor memory deallocation, optional.
  * **loc** (*Any* *,* *optional*) – Optional codegen location for debugging and error reporting.
  * **ip** (*Any* *,* *optional*) – Optional insertion point for codegen.
* **Raises:**
  **AssertionError** – If two_cta_tmem_dealloc_mbar_ptr is None while is_two_cta is True.

#### \_init_dealloc_mbarrier(, loc=None, ip=None)

#### allocate(num_columns: int, , loc=None, ip=None)

Allocate a block of tensor memory.

This method allocates a block of tensor memory from allocator warp and returns a handle to retrieve
the allocated tensor memory address.

#### check_valid_num_columns(num_columns: int)

Check if the number of columns is valid.

This method checks if the number of columns is valid.
It checks if the number of columns is larger than 0, smaller than max capacity, a multiple of 32, and a power of two.

#### free(tmem_ptr: Pointer, num_columns: int = 0, , loc=None, ip=None)

Deallocate the tensor memory.

This method sync on mbarrier (for two cta use case) and deallocates the tensor memory from the allocator warp.
User can optionally specify the number of columns to deallocate. If not specified, all allocated columns will be deallocated.

#### relinquish_alloc_permit(, loc=None, ip=None)

Relinquish the tensor memory allocation permit.

This method relinquishes the tensor memory allocation permit for the allocator warp, promising
the allocator warp will not allocate any more tensor memory.

#### retrieve_ptr(dtype: Type[Numeric] = <class 'cutlass.base_dsl.typing.Float32'>, \*, loc=None, ip=None) → Pointer

Retrieve the pointer to the allocated tensor memory.

This method can be called by all warps after allocation has been performed
by the allocator warp.

#### wait_for_alloc(, loc=None, ip=None)

Wait for the allocator warp to finish allocation.

This method is used to synchronize the allocator warp with the other warps before retrieving tmem ptr.

### *class* cutlass.utils.TransformMode(\*values)

Bases: `Enum`

An enumeration for the possible transform modes of a mixed-input GEMM.

#### ConvertOnly *= 1*

#### ConvertScale *= 2*

### *class* cutlass.utils.WorkTileInfo(tile_idx: int | Integer | None | Tuple[Coord, ...], is_valid_tile: Boolean)

Bases: `object`

A class to represent information about a work tile.

* **Variables:**
  * **tile_idx** – The index of the tile.
  * **is_valid_tile** – Whether the tile is valid.

#### \_\_init_\_(tile_idx: int | Integer | None | Tuple[Coord, ...], is_valid_tile: Boolean)

#### *property* is_valid_tile *: Boolean*

Check latest tile returned by the scheduler is valid or not. Any scheduling
requests after all tasks completed will return an invalid tile.

* **Returns:**
  The validity of the tile.
* **Return type:**
  Boolean

#### *property* tile_idx *: int | Integer | None | Tuple[Coord, ...]*

Get the index of the tile.

* **Returns:**
  The index of the tile.
* **Return type:**
  cute.Coord

### cutlass.utils.compute_epilogue_tile_shape(cta_tile_shape: int | Integer | Tuple[Shape, ...], use_2cta_instrs: bool, layout_d: [LayoutEnum](#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], , layout_c: [LayoutEnum](#cutlass.utils.LayoutEnum) = None, elem_ty_c: Type[Numeric] | None = None, loc=None, ip=None) → int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...]

Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

* **Parameters:**
  * **cta_tile_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile, where
    cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
    corresponds to the width (N) of the tile.
  * **use_2cta_instrs** (*bool*) – A flag indicating whether the configuration is for a 2SM setup.
  * **layout_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
  * **elem_ty_d** (*Type* *[**Numeric* *]*) – The element type of output tensor D.
  * **layout_c** ([*LayoutEnum*](#cutlass.utils.LayoutEnum) *,* *optional*) – The layout enum of the input tensor C. Defaults to None.
  * **elem_ty_c** (*Union* *[**Type* *[**Numeric* *]* *,* *None* *]* *,* *optional*) – The element type for input tensor C. Defaults to None.
* **Returns:**
  Returns epilog tiler, which is used in subsequent epilog partitions.
* **Return type:**
  cute.Tile
* **Raises:**
  **ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

### cutlass.utils.compute_smem_layout(tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), mma_tiler_mnk: tuple[int, int, int], a_dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric], b_dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric], load2trans_stage_count: int, trans2mma_stage_count: int) → tuple[[ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

Compute shared memory layouts for tensor A, transformed A and tensor B.
:param tiled_mma: The tiled MMA object defining the core computation.
:type tiled_mma: cute.TiledMma
:param mma_tiler_mnk: The shape (M, N, K) of the MMA tiler.
:type mma_tiler_mnk: tuple[int, int, int]
:param a_dtype: Data type of operand A.
:type a_dtype: type[cutlass.Numeric]
:param b_dtype: Data type of operand B.
:type b_dtype: type[cutlass.Numeric]
:param load2trans_stage_count: Number of stages for load-to-transform pipeline.
:type load2trans_stage_count: int
:param trans2mma_stage_count: Number of stages for transform-to-MMA pipeline.
:type trans2mma_stage_count: int
:return: A tuple containing (smem_layout_a, smem_layout_a_transform, smem_layout_b) where:

> * smem_layout_a: Shared memory layout for tensor A
> * smem_layout_a_transform: Shared memory layout for transformed tensor A
> * smem_layout_b: Shared memory layout for tensor B
* **Return type:**
  tuple[[cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.create_cute_tensor_for_fp8(storage_tensor, dtype: Type[Numeric], leading_dim: int, source_f32_tensor=None)

Create cute tensor, handling float8 types that don’t support dlpack.

For float8 types, the storage_tensor should be uint8 (for DLPack compatibility).
The source_f32_tensor provides the actual float32 values to convert to fp8.

params storage_tensor: Tensor for DLPack (uint8 for fp8, otherwise the actual dtype)
params dtype: Target cutlass dtype
params leading_dim: Leading dimension for dynamic layout
paramas source_f32_tensor: Float32 source data for fp8 conversion (required for fp8)
return: A cute tensor with the appropriate dtype and layout

### cutlass.utils.create_initial_search_state() → [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

Create an initial search state for grouped gemm.

* **Returns:**
  A new search state with initial values
* **Return type:**
  [GroupedGemmGroupSearchState](#cutlass.utils.GroupedGemmGroupSearchState)

### cutlass.utils.get_copy_atom_a_transform(mma_dtype: [type](cute.md#cutlass.cute.Atom.type)[cutlass.Numeric], use_2cta_instrs: bool, transform_a_source: [tcgen05.OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource), a_smem_shape: cute.Shape, a_dtype: [type](cute.md#cutlass.cute.Atom.type)[cutlass.Numeric]) → [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom)

Determine the copy atom for transformed A tensor based on the operand source and tile size.

### cutlass.utils.get_divisibility(contiguous_dim_size: int, upper_bound: int = 128) → int

Calculate the largest power of 2 divisibility factor for memory alignment.

### cutlass.utils.get_gmem_layout_scale(scale_shape_mkl: tuple[int, int, int], scale_granularity_m: int, scale_granularity_k: int, scale_major_mode: [OperandMajorMode](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → [Layout](cute.md#cutlass.cute.Layout)

Get the layout of the scale tensor in global memory.
:param scale_shape_mkl: The shape of the scale tensor (M, K, L).
:type scale_shape_mkl: tuple[int, int, int]
:return: The layout of the scale tensor in global memory.
:rtype: cute.Layout

### cutlass.utils.get_num_tmem_alloc_cols(tmem_tensors: [Tensor](cute.md#cutlass.cute.Tensor) | List[[Tensor](cute.md#cutlass.cute.Tensor)], rounding=True, , arch: str = 'sm_100', loc=None, ip=None) → int

Get the total number of TMEM allocation columns for the given TMEM tensors.

* **Parameters:**
  * **tmem_tensors** (*Union* *[*[*cute.Tensor*](cute.md#cutlass.cute.Tensor) *,* *List* *[*[*cute.Tensor*](cute.md#cutlass.cute.Tensor) *]* *]*) – The TMEM tensors to get the number of allocation columns for.
  * **rounding** (*bool*) – Whether to round up the number of allocation columns to the nearest power of 2.
  * **arch** (*str*) – The architecture of the GPU.
* **Returns:**
  The total number of TMEM allocation columns.
* **Return type:**
  int
* **Raises:**
  **ValueError** – If the number of TMEM allocation columns exceeds the maximum capacity or is less than 32.

### cutlass.utils.get_smem_capacity_in_bytes(compute_capability: str | None = None) → int

Get the shared memory capacity in bytes for a given compute capability.

Returns the maximum shared memory capacity in bytes available for the specified
GPU compute capability.

* **Parameters:**
  **compute_capability** (*Optional* *[**str* *]*) – The compute capability string (e.g. “70”, “75”, “80”)
* **Returns:**
  The shared memory capacity in bytes
* **Return type:**
  int
* **Raises:**
  **ValueError** – If the compute capability is not supported

### cutlass.utils.get_smem_layout_scale(mma_tiler: tuple[int, int, int], use_2cta_instrs: bool, scale_granularity_m: int, scale_granularity_k: int, scale_major_mode: [OperandMajorMode](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), a_scale_dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric], num_scale_load2trans_stage: int) → tuple[tuple[int, int], [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

Get the layout of the scale tensor in shared memory.
:return: A tuple containing (scale_tile_shape, smem_layout_scale_per_stage, smem_layout_scale) where:

> * scale_tile_shape: The tile shape
> * smem_layout_scale_per_stage: Shared memory layout for scale tensor per stage
> * smem_layout_scale: Shared memory layout for scale tensor
* **Return type:**
  tuple[tuple[int, int], [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.get_smem_store_op(layout_d: [LayoutEnum](#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], elem_ty_acc: Type[Numeric], tiled_tmem_load: [TiledCopy](cute.md#cutlass.cute.TiledCopy), , loc=None, ip=None) → [CopyAtom](cute.md#cutlass.cute.CopyAtom)

Selects the largest vectorized smem store atom available subject to
constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

* **Parameters:**
  * **layout_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
  * **elem_ty_d** (*Type* *[**Numeric* *]*) – The element type for output tensor D.
  * **elem_ty_acc** (*Type* *[**Numeric* *]*) – The element type for accumulator.
  * **tiled_tmem_load** ([*cute.TiledCopy*](cute.md#cutlass.cute.TiledCopy)) – An instance of TiledCopy that represents the tmem load operation.
* **Returns:**
  Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
* **Return type:**
  [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom)

### cutlass.utils.get_tma_atom_kind(mcast: Boolean, use_2cta_instrs: bool, is_b: bool) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

Get the TMA atom kind based on 1) whether it’s a multicast operation,
2) whether 2CTA tcgen05.mma instruction is enabled, and
3) whether it’s a B tensor

### cutlass.utils.get_tmem_load_op(cta_tile_shape: int | Integer | Tuple[Shape, ...], layout_d: [LayoutEnum](#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], elem_ty_acc: Type[Numeric], epi_tile: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], use_2cta_instrs: bool, , loc=None, ip=None) → [CopyAtom](cute.md#cutlass.cute.CopyAtom)

Finds a performant TMEM_LOAD copy op for the selected epilogue
tile (epi_tile), element types, and tcgen05.mma instruction used.

* **Parameters:**
  * **cta_tile_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile.
  * **layout_d** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
  * **elem_ty_d** (*Type* *[**Numeric* *]*) – The element type for output tensor D.
  * **elem_ty_acc** (*Type* *[**Numeric* *]*) – The element type for accumulation.
  * **epi_tile** (*cute.Tile*) – The epilogue tile configuration.
  * **use_2cta_instrs** (*bool*) – A flag indicating whether the configuration is for 2 SMs.
* **Returns:**
  An instance of Sm100TmemLoad with the computed configuration.
* **Return type:**
  [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom)
* **Raises:**
  **ValueError** – If the function cannot handle the given combination of accumulation
  and dimension types, or if it cannot determine the appropriate configuration based on
  the input parameters.

### cutlass.utils.get_transform_a_source(a_major_mode: [OperandMajorMode](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → [OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource)

Determine the operand source for transformed A tensor based on the operand major mode.

### cutlass.utils.is_fp8_dtype(dtype: Type[Numeric]) → bool

Check if dtype is a float8 type that doesn’t support dlpack.
params dtype: The cutlass numeric type to check
type dtype: Type[cutlass.Numeric]
return: True if the dtype is Float8E5M2 or Float8E4M3FN, False otherwise

### cutlass.utils.is_valid_scale_granularity(scale_granularity_m: int, scale_granularity_k: int, a_dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric], k: int, mma_tiler_k: int) → bool

Check if the scale granularity settings are valid for the given data type and problem size.

### cutlass.utils.make_blockscaled_trivial_tiled_mma(ab_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], a_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, b_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, sf_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], sf_vec_size: int, cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup, mma_tiler_mn: ~typing.Tuple[int, int], a_source: ~cutlass.cute.nvgpu.tcgen05.mma.OperandSource = <OperandSource.SMEM>, \*, loc=None, ip=None) → [TiledMma](cute.md#cutlass.cute.TiledMma)

Make a BlockScaled tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

* **Parameters:**
  * **ab_dtype** ([*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]*) – Data type of operands A and B.
  * **a_leading_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) – Leading dimension of operand A (1 for K, 0 for M/N).
  * **b_leading_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) – Leading dimension of operand B (1 for K, 0 for M/N).
  * **sf_dtype** ([*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]*) – Data type of the Scale Factor.
  * **sf_vec_size** (*int*) – The vector size of the Scale Factor.
  * **cta_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) – The CTA group to use.
  * **mma_tiler_mn** (*Tuple* *[**int* *,* *int* *]*) – The shape (M, N, K) of the MMA tiler.
  * **a_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource)) – The source of operand A (SMEM by default or TMEM).
* **Returns:**
  A tiled MMA atom.
* **Return type:**
  [cute.TiledMma](cute.md#cutlass.cute.TiledMma)
* **Raises:**
  **TypeError** – If the data type is not supported.

### cutlass.utils.make_smem_layout_a(tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], a_dtype: Type[Numeric], num_stages: int, , is_k_major=None, loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps with:

1. Get the partitioned shape of the A tensor based on the tiled_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **tiled_mma** ([*cute.TiledMma*](cute.md#cutlass.cute.TiledMma)) – The tiled MMA used to partition tensor A
  * **mma_tiler_mnk** (*cute.cute.Tile*) – The MMA tile shape
  * **a_dtype** (*Type* *[**Numeric* *]*) – The element type for tensor A
  * **num_stages** (*int*) – The number of pipeline stages for tensor A
* **Returns:**
  SMEM layout for tensor A
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.make_smem_layout_b(tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], b_dtype: Type[Numeric], num_stages: int, , is_k_major=None, loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps:

1. Get the partitioned shape of the B tensor based on the tiled_mma & MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **tiled_mma** ([*cute.TiledMma*](cute.md#cutlass.cute.TiledMma)) – The tiled MMA which is used to partition the B tensor.
  * **mma_tiler_mnk** (*cute.cute.Tile*) – The MMA tile shape.
  * **b_dtype** (*Type* *[**Numeric* *]*) – The element type for the B tensor.
  * **num_stages** (*int*) – The stage of the B tensor.
* **Returns:**
  SMEM layout for the B tensor.
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.make_smem_layout_epi(epi_dtype: Type[Numeric], epi_layout: [LayoutEnum](#cutlass.utils.LayoutEnum), epi_tile: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], epi_stage: int, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
   the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **epi_dtype** (*Type* *[**Numeric* *]*) – The element type for the epilog tensor.
  * **epi_layout** ([*LayoutEnum*](#cutlass.utils.LayoutEnum)) – The layout enum for the epilog tensor.
  * **epi_tile** (*cute.cute.Tile*) – The epilogue tile shape.
  * **epi_stage** (*int*) – The stage of the epilog tensor.
* **Returns:**
  SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.make_trivial_tiled_mma(ab_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], a_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, b_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, acc_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup, mma_tiler_mn: ~typing.Tuple[int, int], a_source: ~cutlass.cute.nvgpu.tcgen05.mma.OperandSource = <OperandSource.SMEM>, \*, loc=None, ip=None) → [TiledMma](cute.md#cutlass.cute.TiledMma)

Make a tiled MMA atom with given data type, leading dimension, cta group and mma tile shape.
By default, the MMA atom is created with SMEM operand source for A.

* **Parameters:**
  * **ab_dtype** ([*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]*) – Data type of operands A and B.
  * **a_leading_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) – Leading dimension of operand A (1 for K, 0 for M/N).
  * **b_leading_mode** ([*tcgen05.OperandMajorMode*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) – Leading dimension of operand B (1 for K, 0 for M/N).
  * **acc_dtype** ([*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]*) – Data type of the accumulator.
  * **cta_group** ([*tcgen05.CtaGroup*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)) – The CTA group to use.
  * **mma_tiler_mn** (*Tuple* *[**int* *,* *int* *]*) – The shape (M, N, K) of the MMA tiler.
  * **a_source** ([*cutlass.cute.nvgpu.tcgen05.OperandSource*](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource)) – The source of operand A (SMEM by default or TMEM).
* **Returns:**
  A tiled MMA atom.
* **Return type:**
  [cute.TiledMma](cute.md#cutlass.cute.TiledMma)
* **Raises:**
  **TypeError** – If the data type is not supported.

### cutlass.utils.print_latex(x: [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), \*, color: Callable = <function tikz_color_bwx8>)

Prints a layout.
:param x: A layout
:type x: Union[Layout, ComposedLayout]
:param color: A function that returns TiKZ colors
:type color: Callable

### cutlass.utils.print_latex_tv(layout_tv: ~cutlass.cute.typing.Layout | ~cutlass.cute.typing.ComposedLayout, tile_mn: int | ~cutlass.base_dsl.typing.Integer | ~typing.Tuple[int | ~cutlass.base_dsl.typing.Integer | ~typing.Tuple[IntTuple, ...], ...] | ~cutlass.cute.typing.Layout, \*, color: ~typing.Callable = <function tikz_color_tv>)

Prints a tv layout for a tile M N. Everything must be static.
:param layout_tv: A static thread value layout
:type layout_tv: Union[Layout, ComposedLayout]
:param tile_mn: A static M N tile
:type tile_mn: Union[IntTuple, Layout]
:param color: A function that returns TiKZ colors
:type color: Callable

### cutlass.utils.scale_partition(src_copy_a: [TiledCopy](cute.md#cutlass.cute.TiledCopy), tCsS: [Tensor](cute.md#cutlass.cute.Tensor), transform_local_tidx: Int32, mma_dtype: [type](cute.md#cutlass.cute.Atom.type)[Numeric]) → tuple[[TiledCopy](cute.md#cutlass.cute.TiledCopy), [Tensor](cute.md#cutlass.cute.Tensor), [Tensor](cute.md#cutlass.cute.Tensor), [Tensor](cute.md#cutlass.cute.Tensor)]

Partition the scale tensor for transformation.
This method prepares the copy atom and partitions the shared memory for the scale tensor.
:param src_copy_a: Tiled copy for the source tensor
:type src_copy_a: cute.TiledCopy
:param tCsS: Scale tensor in shared memory
:type tCsS: cute.Tensor
:param transform_local_tidx: Local thread index for transformation warps
:type transform_local_tidx: cutlass.Int32
:param mma_dtype: Data type for the MMA operation
:type mma_dtype: type[cutlass.Numeric]
:return: A tuple containing (smem_thr_copy_S, tSsS_trans, tSrS_copy, tSrS) where:

> * smem_thr_copy_S: Tiled copy for the scale tensor
> * tSsS_trans: Partitioned scale tensor for transformation
> * tSrS_copy: Register fragment for the scale tensor
> * tSrS: View of scale tensor used for transformation computation
* **Return type:**
  tuple[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy), [cute.Tensor](cute.md#cutlass.cute.Tensor), [cute.Tensor](cute.md#cutlass.cute.Tensor), [cute.Tensor](cute.md#cutlass.cute.Tensor)]

### cutlass.utils.scale_tma_partition(tCsS: [cute.Tensor](cute.md#cutlass.cute.Tensor), tCgS: [cute.Tensor](cute.md#cutlass.cute.Tensor), tma_atom_s: [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom), block_in_cluster_coord_vmnk: cute.Coord, scale_cta_layout: [cute.Layout](cute.md#cutlass.cute.Layout)) → tuple[[cute.Tensor](cute.md#cutlass.cute.Tensor), [cute.Tensor](cute.md#cutlass.cute.Tensor)]

Perform TMA partition for scale tensor.
This method partitions the global memory and shared memory buffer for the scale tensor for TMA load.
:param tCsS: Input scale shared memory tensor
:type tCsS: cute.Tensor
:param tCgS: Input scale global memory tensor
:type tCgS: cute.Tensor
:param tma_atom_s: TMA copy atom for scale tensor
:type tma_atom_s: cute.CopyAtom
:param block_in_cluster_coord_vmnk: CTA coord in the cluster
:type block_in_cluster_coord_vmnk: cute.Coord
:param scale_cta_layout: Layout of CTA from the view of the scale tensor
:type scale_cta_layout: cute.Layout
:return: A tuple containing (tSsS, tSgS) where:

> * tSsS: Partitioned scale tensor in shared memory
> * tSgS: Partitioned scale tensor in global memory
* **Return type:**
  tuple[[cute.Tensor](cute.md#cutlass.cute.Tensor), [cute.Tensor](cute.md#cutlass.cute.Tensor)]

### cutlass.utils.transform_partition(transform_a_source: [OperandSource](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.OperandSource), scale_mode: [TransformMode](#cutlass.utils.TransformMode), copy_atom_a_input: [CopyAtom](cute.md#cutlass.cute.CopyAtom), copy_atom_a_transform: [CopyAtom](cute.md#cutlass.cute.CopyAtom), sA_input: [Tensor](cute.md#cutlass.cute.Tensor), A_transform: [Tensor](cute.md#cutlass.cute.Tensor), transform_local_tidx: Int32) → tuple[[TiledCopy](cute.md#cutlass.cute.TiledCopy) | None, [TiledCopy](cute.md#cutlass.cute.TiledCopy) | None, [Tensor](cute.md#cutlass.cute.Tensor), [Tensor](cute.md#cutlass.cute.Tensor)]

Partition tensors for transform input and output.
This method sets up the copy atoms and partitions the shared/tensor memory
for the transformation of tensor A.
:param transform_a_source: Where the transformed tensor A is stored (TMEM or SMEM)
:type transform_a_source: tcgen05.OperandSource
:param scale_mode: The transform mode (ConvertOnly or ConvertScale)
:type scale_mode: TransformMode
:param copy_atom_a_input: Copy atom for loading A from shared memory
:type copy_atom_a_input: cute.CopyAtom
:param copy_atom_a_transform: Copy atom for storing transformed A
:type copy_atom_a_transform: cute.CopyAtom
:param sA_input: Input tensor A in shared memory
:type sA_input: cute.Tensor
:param A_transform: Transformed tensor A in tensor or shared memory
:type A_transform: cute.Tensor
:param transform_local_tidx: Local thread index for transformation warps
:type transform_local_tidx: cutlass.Int32
:return: A tuple containing (src_copy_a, dst_copy_a, tAsA_input, tA_transform) where:

> * src_copy_a: Tiled copy for source tensor
> * dst_copy_a: Tiled copy for destination tensor
> * tAsA_input: Partitioned input tensor A
> * tA_transform: Partitioned transformed tensor A
* **Return type:**
  tuple[Optional[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy)], Optional[[cute.TiledCopy](cute.md#cutlass.cute.TiledCopy)], [cute.Tensor](cute.md#cutlass.cute.Tensor), [cute.Tensor](cute.md#cutlass.cute.Tensor)]
