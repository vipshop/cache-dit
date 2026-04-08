<a id="cute-nvgpu-common"></a>

# Common

### *class* cutlass.cute.nvgpu.CacheEvictionPriority(\*values)

Bases: `Enum`

#### EVICT_FIRST *= CacheEvictionPriority.EVICT_FIRST*

#### EVICT_LAST *= CacheEvictionPriority.EVICT_LAST*

#### EVICT_NORMAL *= CacheEvictionPriority.EVICT_NORMAL*

#### EVICT_UNCHANGED *= CacheEvictionPriority.EVICT_UNCHANGED*

#### NO_ALLOCATE *= CacheEvictionPriority.NO_ALLOCATE*

### *class* cutlass.cute.nvgpu.CopyUniversalOp

Bases: `CopyOp`

The universal Copy Operation.

When creating a Copy Atom out of this operation, the expected usage pattern is

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(
    op, 
    tensor_dtype, 
    num_bits_per_copy=64,
    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL
)
```

- `tensor_dtype` is the data type used to build the reference TV Layout (either the source         or the destination TV Layout) in unit of tensor elements and is used for partitioning by         `TiledCopy` for example
- `num_bits_per_copy` is a kw argument specifying the number of bits to copy per Atom         execution. This can be larger than the width of the above data type. When not provided,         the compiler will do a best effort at auto-vectorizing.
- `l1c_evict_priority` is a kw argument specifying the L1 cache eviction priority hint for         the copy operation. Defaults to `EVICT_NORMAL` if not provided.
- `invariant` is a kw argument specifying whether the load is invariant (read-only data         that never changes). This enables compiler optimizations like instruction reordering.         Defaults to `False` if not provided.

### *class* cutlass.cute.nvgpu.CopyUniversalTrait(value: Value)

Bases: `Trait`

### *class* cutlass.cute.nvgpu.MemoryOrder(\*values)

Bases: `Enum`

#### ACQUIRE *= MemOrderKind.ACQUIRE*

#### ACQ_REL *= MemOrderKind.ACQ_REL*

#### CONSTANT *= MemOrderKind.CONSTANT*

#### MMIO *= MemOrderKind.MMIO*

#### RELAXED *= MemOrderKind.RELAXED*

#### RELEASE *= MemOrderKind.RELEASE*

#### SC *= MemOrderKind.SC*

#### VOLATILE *= MemOrderKind.VOLATILE*

#### WEAK *= MemOrderKind.WEAK*

### *class* cutlass.cute.nvgpu.MemoryScope(\*values)

Bases: `Enum`

#### CLUSTER *= MemScopeKind.CLUSTER*

#### CTA *= MemScopeKind.CTA*

#### GPU *= MemScopeKind.GPU*

#### SYS *= MemScopeKind.SYS*

### *class* cutlass.cute.nvgpu.MmaUniversalOp(abacc_dtype: Type[Numeric])

Bases: `MmaOp`

The universal MMA Operation.

This Operation currently expects the A/B operands as well as the accumulator to share the same
data types.

* **Parameters:**
  **abacc_dtype** (*Type* *[**Numeric* *]*) – The data type for the A/B operands and the accumulator

#### abacc_dtype *: Type[Numeric]*

### *class* cutlass.cute.nvgpu.MmaUniversalTrait(value: Value)

Bases: `Trait`

### *exception* cutlass.cute.nvgpu.OpError(op: Op, message: str, suggestion: str | None = None)

Bases: `DSLBaseError`

An exception class for Op construction errors.

### cutlass.cute.nvgpu.make_tiled_tma_atom_A(op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp) | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp), gmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), smem_layout: [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), mma_tiler_mnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), cluster_shape_vmnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | None = None, , internal_type: Type[Numeric] | None = None, loc=None, ip=None) → Tuple[[CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the MK projections of the TiledMMA for A tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled_mma (considering the M-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the N-mode for A tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
   that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
   associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
   similarly to any other CuTe tensors using the algebra.

* **Parameters:**
  * **op** (*Union* *[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp) *,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) *]*) – The Copy Operation to construct an Atom for
  * **gmem_tensor** ([*Tensor*](cute.md#cutlass.cute.Tensor)) – The GMEM tensor to be loaded by this copy atom
  * **smem_layout** (*Union* *[*[*Layout*](cute.md#cutlass.cute.Layout) *,* [*ComposedLayout*](cute.md#cutlass.cute.ComposedLayout) *]*) – Shared memory layout to load the tensor into (PDSL)
  * **mma_tiler_mnk** (*Shape*) – The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
  * **tiled_mma** ([*atom.TiledMma*](cute.md#cutlass.cute.TiledMma)) – The TiledMMA that will consume the load as operands
  * **cluster_shape_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
  * **internal_type** (*Type* *[**Numeric* *]*) – An optional parameter for the internal data type to when element
    type does not match the copy type
* **Returns:**
  A copy atom for this operation and the associated TMA coord tensor
* **Return type:**
  Tuple[[atom.CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

### cutlass.cute.nvgpu.make_tiled_tma_atom_B(op: [CopyBulkTensorTileG2SOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp) | [CopyBulkTensorTileG2SMulticastOp](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp), gmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), smem_layout: [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), mma_tiler_mnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], tiled_mma: [TiledMma](cute.md#cutlass.cute.TiledMma), cluster_shape_vmnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | None = None, , internal_type: Type[Numeric] | None = None, loc=None, ip=None) → Tuple[[CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

Makes a TMA Copy atom mapping to `.tile` mode for `cp.async.bulk.tensor` PTX operation
accounting for the NK projections of the TiledMMA for B tensor loads.

Given

- a GMEM tensor
- a SMEM layout
- a MMA Tiler
- a TiledMma
- a Cluster-level shape

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to an SMEM buffer with the provided
layout and consistent with the provided Tiler & tiled_mma (considering the N-mode & K-mode).
The Cluster-level shape is used to determine the multicast factor across the M-mode for B tensor loads.

This function returns two results:

1. the Copy Atom
2. the so-called TMA tensor used to map logical coordinates of the GMEM tensor to coordinates
   that the TMA unit can consume. TMA tensors have so-called basis stride elements so that the
   associated layout can output coordinates. Otherwise, TMA tensors can be partitioned
   similarly to any other CuTe tensors using the algebra.

* **Parameters:**
  * **op** (*Union* *[*[*CopyBulkTensorTileG2SOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp) *,* [*CopyBulkTensorTileG2SMulticastOp*](cute_nvgpu_cpasync.md#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) *]*) – The Copy Operation to construct an Atom for
  * **gmem_tensor** ([*Tensor*](cute.md#cutlass.cute.Tensor)) – The GMEM tensor to be loaded by this copy atom
  * **smem_layout** (*Union* *[*[*Layout*](cute.md#cutlass.cute.Layout) *,* [*ComposedLayout*](cute.md#cutlass.cute.ComposedLayout) *]*) – Shared memory layout to load the tensor into (PDSL)
  * **mma_tiler_mnk** (*Shape*) – The MMA Tiler shape (TILE_M, TILE_N, TILE_K) in MNK dimensions
  * **tiled_mma** (*core.TiledMma*) – The TiledMMA that will consume the load as operands
  * **cluster_shape_vmnk** (*Shape*) – The Cluster-level shape in VMNK dimensions
  * **internal_type** (*Type* *[**Numeric* *]*) – An optional parameter for the internal data type to when element
    type does not match the copy type
* **Returns:**
  A Copy Atom for this Operation and the associated TMA tensor
* **Return type:**
  Tuple[[atom.CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

### cutlass.cute.nvgpu.normalize_field_to_ir_name(field, admissible_fields) → str

Normalize a field specifier to its IR logical field name.

Accepted inputs:

- Enum value present in admissible_fields (must expose \_to_ir_field_name()).
- Exact string IR name (e.g., “accum_c”, “neg_a”, “sf_a”).

Any other form is rejected.
