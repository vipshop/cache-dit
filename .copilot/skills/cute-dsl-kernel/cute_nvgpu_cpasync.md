<a id="cute-nvgpu-cpasync"></a>

# cpasync submodule

### *class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) = <CtaGroup.ONE>)

Bases: `TmaCopyOp`

Bulk tensor asynchrnous multicast GMEM to SMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

#### \_\_init_\_(cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) = <CtaGroup.ONE>) → None

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)* *= 1*

### *class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) = <CtaGroup.ONE>)

Bases: `TmaCopyOp`

Bulk tensor asynchrnous GMEM to SMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

#### \_\_init_\_(cta_group: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup) = <CtaGroup.ONE>) → None

#### cta_group *: [CtaGroup](cute_nvgpu_tcgen05.md#cutlass.cute.nvgpu.tcgen05.CtaGroup)* *= 1*

### *class* cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp

Bases: `TmaCopyOp`

Bulk tensor asynchronous SMEM to GMEM Copy Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-tensor).
This Operation uses TMA in the `.tile` mode.

#### \_\_init_\_() → None

### *class* cutlass.cute.nvgpu.cpasync.CopyDsmemStoreOp

Bases: `CopyOp`

Asynchronous Store operation to DSMEM with explicit synchronization.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st-async).

#### \_\_init_\_() → None

### *class* cutlass.cute.nvgpu.cpasync.CopyG2SOp(cache_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode) = <LoadCacheMode.ALWAYS>)

Bases: `CopyOp`

Non-bulk asynchronous GMEM to SMEM Copy Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-non-bulk-copy).

#### \_\_init_\_(cache_mode: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode) = <LoadCacheMode.ALWAYS>) → None

#### cache_mode *: [LoadCacheMode](#cutlass.cute.nvgpu.cpasync.LoadCacheMode)* *= LoadCacheMode.always*

### *class* cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp(reduction_kind: [ReductionOp](cute.md#cutlass.cute.ReductionOp) = ReductionOp.ADD)

Bases: `TmaCopyOp`

Bulk tensor asynchronous SMEM to GMEM Reduction Operation using the TMA unit.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-reduce-async-bulk).
This Operation uses TMA in the `.tile` mode.

#### \_\_init_\_(reduction_kind: [ReductionOp](cute.md#cutlass.cute.ReductionOp) = ReductionOp.ADD) → None

#### reduction_kind *: [ReductionOp](cute.md#cutlass.cute.ReductionOp)* *= 0*

### *class* cutlass.cute.nvgpu.cpasync.LoadCacheMode(\*values)

Bases: `Enum`

An enumeration for the possible cache modes of a non-bulk `cp.async` instruction.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#cache-operators).

#### ALWAYS *= LoadCacheMode.always*

#### GLOBAL *= LoadCacheMode.global_*

#### LAST_USE *= LoadCacheMode.last_use*

#### NONE *= LoadCacheMode.none*

#### STREAMING *= LoadCacheMode.streaming*

### cutlass.cute.nvgpu.cpasync.copy_tensormap(tma_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), tensormap_ptr: Pointer, , loc=None, ip=None) → None

Copies the tensormap held by a TMA Copy Atom to the memory location pointed to by the provided
pointer.

* **Parameters:**
  * **tma_atom** ([*CopyAtom*](cute.md#cutlass.cute.CopyAtom)) – The TMA Copy Atom
  * **tensormap_ptr** (*Pointer*) – The pointer to the memory location to copy the tensormap to

### cutlass.cute.nvgpu.cpasync.cp_fence_tma_desc_release(tma_desc_global_ptr: Pointer, tma_desc_shared_ptr: Pointer, , loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-tensormap-cp-fenceproxy).

### cutlass.cute.nvgpu.cpasync.create_tma_multicast_mask(cta_layout_vmnk: [Layout](cute.md#cutlass.cute.Layout), cta_coord_vmnk: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], mcast_mode: int, , loc=None, ip=None) → Int16

Computes a multicast mask for a TMA load Copy.

* **Parameters:**
  * **cta_layout_vmnk** ([*Layout*](cute.md#cutlass.cute.Layout)) – The VMNK layout of the cluster
  * **cta_coord_vmnk** (*Coord*) – The VMNK coordinate of the current CTA
  * **mcast_mode** (*int*) – The tensor mode in which to multicast
* **Returns:**
  The resulting mask
* **Return type:**
  Int16

### cutlass.cute.nvgpu.cpasync.fence_tma_desc_acquire(tma_desc_ptr: Pointer, , loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.nvgpu.cpasync.fence_tma_desc_release(, loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar).

### cutlass.cute.nvgpu.cpasync.make_tiled_tma_atom(op: [CopyBulkTensorTileG2SOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp) | [CopyBulkTensorTileG2SMulticastOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp) | [CopyBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp) | [CopyReduceBulkTensorTileS2GOp](#cutlass.cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp), gmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), smem_layout_: [Layout](cute.md#cutlass.cute.Layout) | [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), cta_tiler: int | Integer | Tuple[Shape, ...] | [Layout](cute.md#cutlass.cute.Layout) | None | Tuple[Tile, ...], num_multicast: int = 1, , internal_type: Type[Numeric] | None = None, loc=None, ip=None) → Tuple[[CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

Makes a TMA Copy Atom in the `.tile` mode to copy tiles of a GMEM tensor to/from SMEM
buffer with the given Layout.

Given

- a GMEM tensor
- a SMEM layout
- a CTA-level Tiler

this function figures out the bulk tensor asynchronous copy instruction to use with the maximum
“TMA vector length” to copy tiles of the GMEM tensor to/from an SMEM buffer with the provided
layout while maintaining consistency with the provided Tiler.

This function returns two results:

1. the Copy Atom
2. a TMA tensor that maps logical coordinates of the GMEM tensor to coordinates consumed by the        TMA unit. TMA tensors contain basis stride elements that enable their associated layout to        compute coordinates. Like other CuTe tensors, TMA tensors can be partitioned.

* **Parameters:**
  * **op** (*TMAOp*) – The TMA Copy Operation to construct an Atom
  * **gmem_tensor** ([*Tensor*](cute.md#cutlass.cute.Tensor)) – The GMEM tensor involved in the Copy
  * **smem_layout** (*Union* *[*[*Layout*](cute.md#cutlass.cute.Layout) *,* [*ComposedLayout*](cute.md#cutlass.cute.ComposedLayout) *]*) – The SMEM layout to construct the Copy Atom, either w/ or w/o the stage mode
  * **cta_tiler** (*Tiler*) – The CTA Tiler to use
  * **num_multicast** (*int*) – The multicast factor
  * **internal_type** (*Type* *[**Numeric* *]*) – Optional internal data type to use when the tensor data type is not supported by the TMA unit
* **Returns:**
  A TMA Copy Atom associated with the TMA tensor
* **Return type:**
  Tuple[[atom.CopyAtom](cute.md#cutlass.cute.CopyAtom), [Tensor](cute.md#cutlass.cute.Tensor)]

### cutlass.cute.nvgpu.cpasync.prefetch_descriptor(tma_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), , loc=None, ip=None) → None

Prefetches the TMA descriptor associated with the TMA Atom.

### cutlass.cute.nvgpu.cpasync.tma_partition(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), cta_coord: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], cta_layout: [Layout](cute.md#cutlass.cute.Layout), smem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), gmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), , loc=None, ip=None) → Tuple[[Tensor](cute.md#cutlass.cute.Tensor), [Tensor](cute.md#cutlass.cute.Tensor)]

Tiles the GMEM and SMEM tensors for the provided TMA Copy Atom.

### cutlass.cute.nvgpu.cpasync.update_tma_descriptor(tma_atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), gmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), tma_desc_ptr: Pointer, , loc=None, ip=None) → None

Updates the TMA descriptor in the memory location pointed to by the provided pointer using
information from a TMA Copy Atom and the provided GMEM tensor.

Specifically, the following fields of the TMA descriptor will be updated:

1. the GMEM tensor base address
2. the GMEM tensor shape
3. the GMEM tensor stride

Other fields of the TMA descriptor are left unchanged.

* **Parameters:**
  * **tma_atom** ([*CopyAtom*](cute.md#cutlass.cute.CopyAtom)) – The TMA Copy Atom
  * **gmem_tensor** ([*Tensor*](cute.md#cutlass.cute.Tensor)) – The GMEM tensor
  * **tensormap_ptr** (*Pointer*) – The pointer to the memory location of the descriptor to udpate
