<a id="utils-sm100"></a>

# Utilities for SM100

### cutlass.utils.sm100.cluster_shape_to_tma_atom_A(cluster_shape_mnk: int | Integer | Tuple[Shape, ...], atom_thr_id: [Layout](cute.md#cutlass.cute.Layout), , loc=None, ip=None) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

Select the appropriate TMA copy atom for A based on the number of SMs and the multicast flag.

* **Parameters:**
  * **cluster_shape_mnk** (*cute.Shape*) – The shape of the cluster
  * **atom_thr_id** ([*cute.Layout*](cute.md#cutlass.cute.Layout)) – The thread ID of the atom
* **Returns:**
  The appropriate TMA copy atom kind
* **Return type:**
  [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)
* **Raises:**
  * **ValueError** – If the atom_sm_cnt is invalid
  * **ValueError** – If the cluster shape is not divisible by the atom SM count

### cutlass.utils.sm100.cluster_shape_to_tma_atom_B(cluster_shape_mnk: int | Integer | Tuple[Shape, ...], atom_thr_id: [Layout](cute.md#cutlass.cute.Layout), , loc=None, ip=None) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

Select the appropriate TMA copy atom for Bbased on the number of SMs and the multicast flag.

* **Parameters:**
  * **cluster_shape_mnk** (*cute.Shape*) – The shape of the cluster
  * **atom_thr_id** ([*cute.Layout*](cute.md#cutlass.cute.Layout)) – The thread ID of the atom
* **Returns:**
  The appropriate TMA copy atom kind
* **Return type:**
  [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)
* **Raises:**
  * **ValueError** – If the atom_sm_cnt is invalid
  * **ValueError** – If the cluster shape is not divisible by the atom SM count

### cutlass.utils.sm100.cluster_shape_to_tma_atom_SFB(cluster_shape_mnk: int | Integer | Tuple[Shape, ...], atom_thr_id: [Layout](cute.md#cutlass.cute.Layout), , loc=None, ip=None) → [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) | [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)

Select the appropriate TMA copy atom for SFB based on the number of SMs and the multicast flag.

* **Parameters:**
  * **cluster_shape_mnk** (*cute.Shape*) – The shape of the cluster
  * **atom_thr_id** ([*cute.Layout*](cute.md#cutlass.cute.Layout)) – The thread ID of the atom
* **Returns:**
  The appropriate TMA copy atom kind
* **Return type:**
  [cpasync.CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) or [cpasync.CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp)
* **Raises:**
  * **ValueError** – If the atom_sm_cnt is invalid
  * **ValueError** – If the cluster shape is not divisible by the atom SM count

### cutlass.utils.sm100.compute_epilogue_tile_shape(cta_tile_shape: int | Integer | Tuple[Shape, ...], use_2cta_instrs: bool, layout_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], , layout_c: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum) = None, elem_ty_c: Type[Numeric] | None = None, loc=None, ip=None) → int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...]

Attempts to compute a reasonable epilogue tile based on block tile shape or allows the user to provide one.

* **Parameters:**
  * **cta_tile_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile, where
    cta_tile_shape[0] corresponds to the height (M) and cta_tile_shape[1]
    corresponds to the width (N) of the tile.
  * **use_2cta_instrs** (*bool*) – A flag indicating whether the configuration is for a 2SM setup.
  * **layout_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
  * **elem_ty_d** (*Type* *[**Numeric* *]*) – The element type of output tensor D.
  * **layout_c** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum) *,* *optional*) – The layout enum of the input tensor C. Defaults to None.
  * **elem_ty_c** (*Union* *[**Type* *[**Numeric* *]* *,* *None* *]* *,* *optional*) – The element type for input tensor C. Defaults to None.
* **Returns:**
  Returns epilog tiler, which is used in subsequent epilog partitions.
* **Return type:**
  cute.Tile
* **Raises:**
  **ValueError** – If the computed tile cute.size does not meet minimum requirements based on CTA dimensions.

### cutlass.utils.sm100.get_num_tmem_alloc_cols(tmem_tensors: [Tensor](cute.md#cutlass.cute.Tensor) | List[[Tensor](cute.md#cutlass.cute.Tensor)], rounding=True, , loc=None, ip=None) → int

### cutlass.utils.sm100.get_permutation_mnk(tile_shape_mnk: int | Integer | Tuple[Shape, ...], sf_vec_size: int, use_mxf8f6f4: bool, , loc=None, ip=None) → Tuple[int, int, int]

Get the permutation of M, N, K for the tiled MMA.

* **Parameters:**
  * **tile_shape_mnk** (*cute.Shape*) – The shape of the tile
  * **sf_vec_size** (*int*) – The vector size of the Scale Factor.
  * **use_mxf8f6f4** (*bool*) – Whether to use MXF8F6F4 or MXF4NVF4.
* **Returns:**
  The permutation of M, N, K
* **Return type:**
  Tuple[int, int, int]
* **Raises:**
  **ValueError** – If the tile shape is not divisible by the sf_vec_size

### cutlass.utils.sm100.get_smem_store_op(layout_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], elem_ty_acc: Type[Numeric], tiled_tmem_load: [TiledCopy](cute.md#cutlass.cute.TiledCopy), , loc=None, ip=None) → [CopyAtom](cute.md#cutlass.cute.CopyAtom)

Selects the largest vectorized smem store atom available subject to
constraint of gmem layout and chosen TMEM_LOAD’s thread-value ownership.

* **Parameters:**
  * **layout_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
  * **elem_ty_d** (*Type* *[**Numeric* *]*) – The element type for output tensor D.
  * **elem_ty_acc** (*Type* *[**Numeric* *]*) – The element type for accumulator.
  * **tiled_tmem_load** ([*cute.TiledCopy*](cute.md#cutlass.cute.TiledCopy)) – An instance of TiledCopy that represents the tmem load operation.
* **Returns:**
  Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.
* **Return type:**
  [cute.CopyAtom](cute.md#cutlass.cute.CopyAtom)

### cutlass.utils.sm100.get_tmem_load_op(cta_tile_shape: int | Integer | Tuple[Shape, ...], layout_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], elem_ty_acc: Type[Numeric], epi_tile: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], use_2cta_instrs: bool, , loc=None, ip=None) → [CopyAtom](cute.md#cutlass.cute.CopyAtom)

Finds a performant TMEM_LOAD copy op for the selected epilogue
tile (epi_tile), element types, and tcgen05.mma instruction used.

* **Parameters:**
  * **cta_tile_shape** (*cute.Shape*) – A tuple or list representing the dimensions of the CTA tile.
  * **layout_d** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum of the output tensor D.
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

### cutlass.utils.sm100.make_blockscaled_trivial_tiled_mma(ab_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], a_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, b_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, sf_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], sf_vec_size: int, cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup, mma_tiler_mn: ~typing.Tuple[int, int], a_source: ~cutlass.cute.nvgpu.tcgen05.mma.OperandSource = <OperandSource.SMEM>, \*, loc=None, ip=None) → [TiledMma](cute.md#cutlass.cute.TiledMma)

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

### cutlass.utils.sm100.make_smem_layout_a(tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], a_dtype: Type[Numeric], num_stages: int, , is_k_major=None, loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

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

### cutlass.utils.sm100.make_smem_layout_b(tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], b_dtype: Type[Numeric], num_stages: int, , is_k_major=None, loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

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

### cutlass.utils.sm100.make_smem_layout_epi(epi_dtype: Type[Numeric], epi_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), epi_tile: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], epi_stage: int, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps:

1. Select the heuristic SMEM layout atom based on the epilog tile shape,
   the epilog tensor’s majorness, and the element type.
2. cute.Tile the SMEM layout atom to the epilog tile shape.
3. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **epi_dtype** (*Type* *[**Numeric* *]*) – The element type for the epilog tensor.
  * **epi_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum for the epilog tensor.
  * **epi_tile** (*cute.cute.Tile*) – The epilogue tile shape.
  * **epi_stage** (*int*) – The stage of the epilog tensor.
* **Returns:**
  SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.sm100.make_trivial_tiled_mma(ab_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], a_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, b_leading_mode: ~cutlass.cute.nvgpu.tcgen05.mma.OperandMajorMode, acc_dtype: ~typing.Type[~cutlass.base_dsl.typing.Numeric], cta_group: ~cutlass.cute.nvgpu.tcgen05.mma.CtaGroup, mma_tiler_mn: ~typing.Tuple[int, int], a_source: ~cutlass.cute.nvgpu.tcgen05.mma.OperandSource = <OperandSource.SMEM>, \*, loc=None, ip=None) → [TiledMma](cute.md#cutlass.cute.TiledMma)

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
