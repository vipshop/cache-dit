<a id="utils-sm90"></a>

# Utilities for SM90

### cutlass.utils.sm90.compute_tile_shape_or_override(tile_shape_mnk: tuple[int, int, int], element_type: [type](cute.md#cutlass.cute.Atom.type)[Numeric], is_cooperative: bool = False, epi_tile_override: tuple[int, int] | None = None) → tuple[int, int]

Compute the epilogue tile shape or use override if provided.

* **Parameters:**
  * **tile_shape_mnk** (*Tuple* *[**int* *,* *int* *,* *int* *]*) – CTA tile shape (M,N,K)
  * **element_type** ([*type*](cute.md#cutlass.cute.Atom.type) *[**Numeric* *]*) – Data type of elements
  * **is_cooperative** (*bool*) – Whether to use cooperative approach
  * **epi_tile_override** (*Tuple* *[**int* *,* *int* *] or* *None*) – Optional override for epilogue tile shape
* **Returns:**
  Computed epilogue tile shape
* **Return type:**
  Tuple[int, int]

### cutlass.utils.sm90.get_smem_store_op(layout_d: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), elem_ty_d: Type[Numeric], elem_ty_acc: Type[Numeric], , loc=None, ip=None) → [CopyAtom](cute.md#cutlass.cute.CopyAtom)

Selects the largest vectorized smem store atom available subject to constraint of gmem layout.

## Parameters:

layout_d
: The layout enum of the output tensor D.

elem_ty_d
: The element type for output tensor D.

elem_ty_acc
: The element type for accumulator.

## Returns:

Either SmemStoreMatrix or SimtSyncCopy, based on the input parameters.

### cutlass.utils.sm90.make_smem_layout_a(a_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], a_dtype: Type[Numeric], num_stages: int, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps with:

1. Get the partitioned shape of the A tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the A tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **a_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum for tensor A
  * **mma_tiler_mnk** (*cute.cute.Tile*) – The MMA tile shape
  * **a_dtype** (*Type* *[**Numeric* *]*) – The element type for tensor A
  * **num_stages** (*int*) – The number of pipeline stages for tensor A
* **Returns:**
  SMEM layout for tensor A
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.sm90.make_smem_layout_b(b_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), mma_tiler_mnk: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], b_dtype: Type[Numeric], num_stages: int, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

This function helps with:

1. Get the partitioned shape of the B tensor based on the MMA tiler.
2. Select the heuristic SMEM layout atom based on the B tensor’s majorness, the data type, and the major mode size.
3. cute.Tile the SMEM layout atom to the MMA tile shape.
4. Stage the SMEM layout based on the number of stages.

* **Parameters:**
  * **b_layout** ([*LayoutEnum*](utils.md#cutlass.utils.LayoutEnum)) – The layout enum for tensor B
  * **mma_tiler_mnk** (*cute.cute.Tile*) – The MMA tile shape
  * **b_dtype** (*Type* *[**Numeric* *]*) – The element type for tensor B
  * **num_stages** (*int*) – The number of pipeline stages for tensor B
* **Returns:**
  SMEM layout for tensor B
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]

### cutlass.utils.sm90.make_smem_layout_epi(epi_dtype: Type[Numeric], epi_layout: [LayoutEnum](utils.md#cutlass.utils.LayoutEnum), epi_tile: int | Integer | None | [Layout](cute.md#cutlass.cute.Layout) | Tuple[Tile, ...], epi_stage: int, smem_trg_shape: [Layout](cute.md#cutlass.cute.Layout) | None = None, smem_order: tuple | None = None, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

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
  * **smem_trg_shape** ([*cute.Layout*](cute.md#cutlass.cute.Layout) *|* *None*) – Target shape for SMEM layout (optional).
  * **smem_order** (*tuple* *|* *None*) – Order for SMEM layout (optional).
* **Returns:**
  SMEM layout for epilog tensors (usually C & D which are processed in the epilog)
* **Return type:**
  Union[[cute.Layout](cute.md#cutlass.cute.Layout), [cute.ComposedLayout](cute.md#cutlass.cute.ComposedLayout)]
