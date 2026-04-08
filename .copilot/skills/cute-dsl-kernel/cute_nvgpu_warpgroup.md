<a id="cute-nvgpu-warpgroup"></a>

# warpgroup submodule

### *class* cutlass.cute.nvgpu.warpgroup.Field(\*values)

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

#### ACCUMULATE *= 'accum_c'*

### *class* cutlass.cute.nvgpu.warpgroup.MmaF16BF16Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], a_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode))

Bases: `MmaOp`

F16/BF16 warpgroup MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], a_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode)) → None

#### descriptive_name *= 'warpgroup F16/BF16 MMA Operation'*

### *class* cutlass.cute.nvgpu.warpgroup.MmaF8Op(a_dtype: Type[Numeric], b_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], a_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode))

Bases: `MmaOp`

F16/BF16 warpgroup MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma).
This Operation covers the instructions using the `.e4m3` or `.e5m2` qualifiers for the input operands.

#### \_\_init_\_(a_dtype: Type[Numeric], b_dtype: Type[Numeric], acc_dtype: Type[Numeric], instruction_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], a_src: [OperandSource](#cutlass.cute.nvgpu.warpgroup.OperandSource), a_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode), b_major_mode: [OperandMajorMode](#cutlass.cute.nvgpu.warpgroup.OperandMajorMode)) → None

#### descriptive_name *= 'warpgroup F8 MMA Operation'*

### *class* cutlass.cute.nvgpu.warpgroup.OperandMajorMode(\*values)

Bases: `Enum`

An enumeration for the majorness of the input operands of the MMA.

#### K *= MajorMode.k*

#### MN *= MajorMode.mn*

### *class* cutlass.cute.nvgpu.warpgroup.OperandSource(\*values)

Bases: `Enum`

An enumeration for the source memory location of the A input operand of the MMA.

#### RMEM *= MmaFragKind.rmem*

#### SMEM *= MmaFragKind.smem_desc*

### *class* cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind(\*values)

Bases: `Enum`

Enum class for the kinds of SMEM layout atoms for SM90.

Given a swizzle kind, an SMEM layout atom is the compact layout of smallest size that can
be used to construct an SMEM layout using blocked product for operand A or B such that the
resulting layout is legal for both TMA and UMMA.

Note that there are other ways of creating legal layouts for operand A and B.

#### K_INTER *= 5*

#### K_SW128 *= 8*

#### K_SW32 *= 6*

#### K_SW64 *= 7*

#### MN_INTER *= 1*

#### MN_SW128 *= 4*

#### MN_SW32 *= 2*

#### MN_SW64 *= 3*

### cutlass.cute.nvgpu.warpgroup.commit_group(, loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-commit-group).

### cutlass.cute.nvgpu.warpgroup.fence(, loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-fence).

### cutlass.cute.nvgpu.warpgroup.make_smem_layout_atom(kind: [SmemLayoutAtomKind](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind), element_type: Type[Numeric], , loc=None, ip=None) → [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

Makes a SMEM layout Atom.

This function creates a composed layout in unit of elements consistent with the requested layout
Atom kind and element data type.

* **Parameters:**
  * **kind** ([*SmemLayoutAtomKind*](#cutlass.cute.nvgpu.warpgroup.SmemLayoutAtomKind)) – The kind of layout Atom
  * **element_type** (*Type* *[**Numeric* *]*) – The element data type to construct the layout for
* **Returns:**
  The SMEM layout atom
* **Return type:**
  [ComposedLayout](cute.md#cutlass.cute.ComposedLayout)

### cutlass.cute.nvgpu.warpgroup.wait_group(group, , loc=None, ip=None) → None

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group).
