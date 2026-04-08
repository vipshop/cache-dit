<a id="cute-nvgpu-warp"></a>

# warp submodule

### *class* cutlass.cute.nvgpu.warp.Field(\*values)

Bases: `Enum`

An enumeration for the fields of the MMA Atom that can be modified at runtime.

#### ACCUMULATE *= 'accum_c'*

#### SFA *= 'sf_a'*

#### SFB *= 'sf_b'*

### *class* cutlass.cute.nvgpu.warp.LdMatrix16x16x8bOp(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None)

Bases: `BaseOp`

16x16 `ldmatrix` Operation with transpose and optional unpacking to 8b container.
Packed source container is 16x4b elements with 64b padding
or 16x6b elements with 32b padding (total 128b per 16 elements)

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
This operation corresponds to the `.m16n16` and the `.b4x16_p64`,\`\`.b6x16_p32\`\`,\`\`.b8\`\` qualifiers.

#### \_\_init_\_(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None) → None

### *class* cutlass.cute.nvgpu.warp.LdMatrix16x8x8bOp(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None)

Bases: `BaseOp`

16x8 8b `ldmatrix` Operation with transpose

There is no direct PTX correspondance to this Op.
This actually lowers to ldmatrix with the `.m16n16` qualifier and
additional address and value permutations to match stmatrix.m16n8.trans.
Useful for vectorizing with Ampere-style 8x8 matrix thread-value layouts

#### \_\_init_\_(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None) → None

### *class* cutlass.cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None)

Bases: `BaseOp`

8x8 `ldmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-load-instruction-ldmatrix).
This operation corresponds to the `.m8n8` qualifier.

#### \_\_init_\_(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None) → None

### *class* cutlass.cute.nvgpu.warp.MmaF16BF16Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], shape_mnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...])

Bases: `WarpMmaOp`

F16/BF16 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.f16` or `.bf16` qualifiers for the input operands.

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], shape_mnk: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]) → None

#### ab_dtype *: Type[Numeric]*

#### acc_dtype *: Type[Numeric]*

#### shape_mnk *: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]*

### *class* cutlass.cute.nvgpu.warp.MmaMXF4NVF4Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], sf_type: Type[Numeric])

Bases: `MmaSM120BlockScaledOp`

MXF4NVF4 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind           = {.kind::mxf4nvf4};
.scale_vec_size = {.scale_vec::2X, .scale_vec::4X};
.stype          = {.ue8m0, .ue4m3};

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], sf_type: Type[Numeric]) → None

#### descriptive_name *= 'warp-level MXF4NVF4 MMA Operation'*

### *class* cutlass.cute.nvgpu.warp.MmaMXF4Op(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], sf_type: Type[Numeric])

Bases: `MmaSM120BlockScaledOp`

MXF4 warp-level MMA Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma).
This Operation covers the instructions using the `.e2m1` qualifiers for the input operands.
.kind           = {.kind::mxf4};
.scale_vec_size = {.scale_vec::2X};
.stype          = {.ue8m0};

#### \_\_init_\_(ab_dtype: Type[Numeric], acc_dtype: Type[Numeric], sf_type: Type[Numeric]) → None

#### descriptive_name *= 'warp-level MXF4 MMA Operation'*

### *class* cutlass.cute.nvgpu.warp.StMatrix16x8x8bOp(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None)

Bases: `BaseOp`

16x8 `stmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
This operation corresponds to the `m16n8` qualifier.

#### \_\_init_\_(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None) → None

### *class* cutlass.cute.nvgpu.warp.StMatrix8x8x16bOp(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None)

Bases: `BaseOp`

8x8 `stmatrix` Operation.

See the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-stmatrix).
This operation corresponds to the `m8n8` qualifier.

#### \_\_init_\_(transpose: bool = False, num_matrices: int = 1, unpack_bits: int | None = None) → None
