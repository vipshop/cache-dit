<a id="cute-nvgpu-tcgen05"></a>

# tcgen05 submodule

### *class* cutlass.cute.nvgpu.tcgen05.CtaGroup(\*values)

Bases: `Enum`

An enumeration for the `cta_group`  qualifier of the MMA.

#### ONE *= 1*

#### TWO *= 2*

### *class* cutlass.cute.nvgpu.tcgen05.Field(\*values)

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

#### ACCUMULATE *= 'accum_c'*

#### NEGATE_A *= 'neg_a'*

#### NEGATE_B *= 'neg_b'*

#### SFA *= 'sf_a'*

#### SFB *= 'sf_b'*

### *class* cutlass.cute.nvgpu.tcgen05.Ld16x128bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>)

Bases: `_LdBase`

16x128b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x128b` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.Ld16x256bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>)

Bases: `_LdBase`

16x256b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x256b` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.Ld16x32bx2Op(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>)

Bases: `_LdBase`

16x32bx2 TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x32bx2` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.Ld16x64bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>)

Bases: `_LdBase`

16x64b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.16x64b` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.Ld32x32bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>)

Bases: `_LdBase`

32x32b TMEM load Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-ld).
This Operation corresponds to the `.32x32` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition) = <Repetition.x1>, pack: [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) = <Pack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.MmaF16BF16Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode))

Bases: `MmaOp`

F16/BF16 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::f16` qualifier.

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → None

#### descriptive_name *= 'tcgen05 F16/BF16 MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaFP8Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode))

Bases: `MmaOp`

F8 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → None

#### descriptive_name *= 'tcgen05 F8 MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaI8Op(ab_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode))

Bases: `MmaOp`

I8 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::i8` qualifier.

#### \_\_init_\_(ab_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → None

#### descriptive_name *= 'tcgen05 I8 MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaMXF4NVF4Op(sf_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource))

Bases: `BlockScaledMmaOp`

MXF4NVF4 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf4nvf4` qualifier.

#### \_\_init_\_(sf_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource)) → None

#### descriptive_name *= 'tcgen05 MXF4NVF4 BlockScaled MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaMXF4Op(instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource))

Bases: `BlockScaledMmaOp`

MXF4 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf4` qualifier.

#### \_\_init_\_(instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource)) → None

#### descriptive_name *= 'tcgen05 MXF4 BlockScaled MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaMXF8Op(ab_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode))

Bases: `BlockScaledMmaOp`

MXF8 tcgen05 BlockScaled MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::mxf8f6f4` qualifier.

#### \_\_init_\_(ab_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → None

#### descriptive_name *= 'tcgen05 MXF8 BlockScaled MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.MmaTF32Op(instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode))

Bases: `MmaOp`

TF32 tcgen05 MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-instructions-mma).
This Operation corresponds to the `.kind::tf32` qualifier.

#### \_\_init_\_(instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup), a_src: [OperandSource](#cutlass.cute.nvgpu.tcgen05.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.tcgen05.OperandMajorMode)) → None

#### descriptive_name *= 'tcgen05 TF32 MMA Operation'*

### *class* cutlass.cute.nvgpu.tcgen05.OperandMajorMode(\*values)

Bases: `Enum`

An enumeration for the majorness of the input operands of the MMA.

#### K *= MajorMode.k*

#### MN *= MajorMode.mn*

### *class* cutlass.cute.nvgpu.tcgen05.OperandSource(\*values)

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

#### SMEM *= MmaFragKind.smem_desc*

#### TMEM *= MmaFragKind.tmem*

### *class* cutlass.cute.nvgpu.tcgen05.Pack(\*values)

Bases: `Enum`

An enumeration for the possible packing patterns for TMEM to RMEM copies.

#### NONE *= 1*

#### PACK_16b_IN_32b *= 2*

### *class* cutlass.cute.nvgpu.tcgen05.Repetition(\*values)

Bases: `Enum`

An enumeration for the number of repetitions of a given TMEM copy within the instruction.

#### x1 *= 1*

#### x128 *= 128*

#### x16 *= 16*

#### x2 *= 2*

#### x32 *= 32*

#### x4 *= 4*

#### x64 *= 64*

#### x8 *= 8*

### *class* cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind(\*values)

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM100.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can be
used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

#### K_INTER *= 6*

#### K_SW128 *= 9*

#### K_SW32 *= 7*

#### K_SW64 *= 8*

#### MN_INTER *= 1*

#### MN_SW128 *= 4*

#### MN_SW128_32B *= 5*

#### MN_SW32 *= 2*

#### MN_SW64 *= 3*

### *class* cutlass.cute.nvgpu.tcgen05.St16x128bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>)

Bases: `_StBase`

16x128b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x128` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.St16x256bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>)

Bases: `_StBase`

16x256b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x256` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.St16x32bx2Op(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>)

Bases: `_StBase`

16x32x2b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x32x2` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.St16x64bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>)

Bases: `_StBase`

16x64b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.16x64` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.St32x32bOp(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>)

Bases: `_StBase`

32x32b TMEM store Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instructions-tcgen05-st).
This Operation corresponds to the `.32x32` qualifier.

#### \_\_init_\_(repeat: [Repetition](#cutlass.cute.nvgpu.tcgen05.Repetition), unpack: [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack) = <Unpack.NONE>) → None

### *class* cutlass.cute.nvgpu.tcgen05.TmemLoadRedOp(\*values)

Bases: `Enum`

An enumeration for the possible reduce operations for TMEM load operations.

#### MAX *= TmemLoadRedOp.max*

#### MAXABS *= TmemLoadRedOp.maxabs*

#### MIN *= TmemLoadRedOp.min*

#### MINABS *= TmemLoadRedOp.minabs*

### *class* cutlass.cute.nvgpu.tcgen05.Unpack(\*values)

Bases: `Enum`

An enumeration for the possible unpacking patterns for RMEM to TMEM copies.

#### NONE *= 1*

#### UNPACK_32b_IN_16b *= 2*

### cutlass.cute.nvgpu.tcgen05.commit(mbar_ptr: Pointer, mask=None, cta_group: [CtaGroup](#cutlass.cute.nvgpu.tcgen05.CtaGroup) = <CtaGroup.ONE>, \*, loc=None, ip=None) → None

Perform an arrive operation on a mbarrier upon completion of previous MMA operations.

* **Parameters:**
  * **mbar_ptr** (*Pointer*) – A pointer to the mbarrier in SMEM
  * **mask** (*Int*) – An optional multicast mask for the CTAs in the cluster to signal arrival to

### cutlass.cute.nvgpu.tcgen05.find_tmem_tensor_col_offset(tmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), , loc=None, ip=None) → int | Integer

Computes the TMEM column offset given a TMEM tensor.

* **Parameters:**
  **tmem_tensor** ([*Tensor*](cute.md#cutlass.cute.Tensor)) – The TMEM tensor to use to compute the columns offset
* **Returns:**
  The columns offset
* **Return type:**
  Int

### cutlass.cute.nvgpu.tcgen05.get_s2t_smem_desc_tensor(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), smem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](cute.md#cutlass.cute.Tensor)

Returns the SMEM descriptor tensor from a S2T copy atom and a SMEM tensor.

### cutlass.cute.nvgpu.tcgen05.get_tmem_copy_properties(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom)) → Tuple[int, int, int, [Pack](#cutlass.cute.nvgpu.tcgen05.Pack) | [Unpack](#cutlass.cute.nvgpu.tcgen05.Unpack)]

Returns the properties of a TMEM copy atom (number of data paths, bits, repetitions,
and whether packing/unpacking is used).

### cutlass.cute.nvgpu.tcgen05.is_tmem_load(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom)) → bool

Returns whether a CopyAtom instance is a TMEM load.

### cutlass.cute.nvgpu.tcgen05.is_tmem_store(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom)) → bool

Returns whether a CopyAtom instance is a TMEM store.

### cutlass.cute.nvgpu.tcgen05.make_s2t_copy(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), tmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), , loc=None, ip=None) → [TiledCopy](cute.md#cutlass.cute.TiledCopy)

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

### cutlass.cute.nvgpu.tcgen05.make_smem_layout_atom(kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind), element_type: Type[Numeric], , loc=None, ip=None) → [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

* **Parameters:**
  * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.tcgen05.SmemLayoutAtomKind)) – The kind of layout Atom
  * **element_type** (*Type* *[**Numeric* *]*) – The element data type to construct the layout for
* **Returns:**
  The SMEM layout atom
* **Return type:**
  [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

### cutlass.cute.nvgpu.tcgen05.make_tmem_copy(atom: [CopyAtom](cute.md#cutlass.cute.CopyAtom), tmem_tensor: [Tensor](cute.md#cutlass.cute.Tensor), , loc=None, ip=None) → [TiledCopy](cute.md#cutlass.cute.TiledCopy)

Makes a Tiled Copy instance from a TMEM Copy Atom and a TMEM tensor.

### cutlass.cute.nvgpu.tcgen05.make_umma_smem_desc(src: Pointer, layout: [Layout](cute.md#cutlass.cute.Layout), major: str, next_src: Pointer | None = None, , loc=None, ip=None)

Construct shared memory descriptor for UMMA.

The make_umma_smem_desc operation accepts an input cute.ptr (optionally a nextSrc
pointer for the second buffer in a circular buffer scheme), alongside a cute.layout
and a major attr, then constructs the shared memory descriptor and returns it.
The layout must be describing the buffer pointed to by the input pointer and the
iterator must carry valid swizzle information.

There are 5 supported swizzle variants:
- S<0, 4, 3> | SWIZZLE_NONE
- S<1, 4, 3> | SWIZZLE_32B
- S<2, 4, 3> | SWIZZLE_64B
- S<3, 4, 3> | SWIZZLE_128B
- S<2, 5, 2> | SWIZZLE_128B_BASE32B

The cute.ptr must carry shared address space and must be aligned to 16B.

* **Parameters:**
  * **src** (*Pointer*) – The source pointer to shared memory
  * **layout** ([*Layout*](cute.md#cutlass.cute.Layout)) – The layout describing the buffer
  * **major** (*str*) – The major mode attribute
  * **next_src** (*Optional* *[**Pointer* *]*) – Optional next source pointer for circular buffer scheme
* **Returns:**
  The shared memory descriptor
* **Return type:**
  SmemDescType

### cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape(atom: [Layout](cute.md#cutlass.cute.Layout), mma_tile_shape: Shape, order: IntTuple = None, , loc=None, ip=None) → [Layout](cute.md#cutlass.cute.Layout)

### cutlass.cute.nvgpu.tcgen05.tile_to_mma_shape(atom: [ComposedLayout](cute.md#cutlass.cute.ComposedLayout), mma_tile_shape: Shape, order: IntTuple = None, , loc=None, ip=None) → [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

Tiles a layout to an MMA shape.
