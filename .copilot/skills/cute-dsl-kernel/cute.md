<a id="cute"></a>

# cutlass.cute

### *class* cutlass.cute.AddressSpace(\*values)

Bases: `IntEnum`

Address spaces for CuTe memrefs and pointers

#### dsmem *= 7*

#### generic *= 0*

#### gmem *= 1*

#### rmem *= 5*

#### smem *= 3*

#### tmem *= 6*

### *class* cutlass.cute.Atom(op: Op, trait: Trait)

Bases: `ABC`

Atom base class.

An Atom is the composition of

- a MMA or Copy Operation;
- an internal MMA or Copy Trait.

An Operation is a pure Python class that is used to model a specific MMA or Copy instruction.
The Trait wraps the underlying IR Value and provides access to the metadata of the instruction
encoded using CuTe Layouts. When the Trait can be constructed straighforwardly from an
Operation, the `make_mma_atom` or `make_copy_atom` API should be used. There are cases where
constructing the metadata is not trivial and requires more information, for example to determine
the number of bytes copied per TMA instruction (“the TMA vector length”). In such cases,
dedicated helper functions are provided with an appropriate API such that the Atom is
constructed internally in an optimal fashion for the user.

#### \_\_init_\_(op: Op, trait: Trait) → None

#### \_abc_impl *= <_abc._abc_data object>*

#### \_unpack(, loc=None, ip=None, \*\*kwargs) → Value

#### get(field, , loc=None, ip=None) → Any

Gets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
accum = tiled_mma.get(cute.nvgpu.tcgen05.Field.ACCUMULATE)
```

The `get` method provides a way to the user to access such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

#### *property* op *: Op*

#### set(modifier, value, , loc=None, ip=None) → None

Sets runtime fields of the Atom.

Some Atoms have runtime state, for example a tcgen05 MMA Atom

```python
tiled_mma = cute.make_tiled_mma(some_tcgen05_mma_op)
tiled_mma.set(cute.nvgpu.tcgen05.Field.ACCUMULATE, True)
```

The `set` method provides a way to the user to modify such runtime state. Modifiable
fields are provided by arch-specific enumerations, for example `tcgen05.Field`. The Atom
instance internally validates the field as well as the value provided by the user to set
the field to.

#### *property* type

#### with_(, loc=None, ip=None, \*\*kwargs) → [Atom](#cutlass.cute.Atom)

Returns a new Atom with the new Operation and Trait with the given runtime state. The runtime state
is provided as keyword arguments and it is Atom-specific.

```python
tiled_copy = cute.make_tiled_copy(tma_copy_op)
new_tiled_copy = tiled_copy.with_(tma_bar_ptr=tma_bar_ptr, cache_policy=cute.CacheEvictionPriority.EVICT_LAST)
```

The `with_` method provides a way to the user to modify such runtime state or create an executable Atom
(e.g. an Executable TMA Load Atom).

### *class* cutlass.cute.CacheEvictionPriority(\*values)

Bases: `IntEnum`

Cute Cache Eviction Priority kind

#### EVICT_FIRST *= 1*

#### EVICT_LAST *= 2*

#### EVICT_NORMAL *= 0*

#### EVICT_UNCHANGED *= 3*

#### NO_ALLOCATE *= 4*

### *class* cutlass.cute.ComposedLayout

Bases: `ABC`

ComposedLayout represents the functional composition of layouts in CuTe.

**Formally:**

$$
R(c) := (inner \circ offset \circ outer)(c) := inner(offset + outer(c))
$$

where:

> - inner: The inner layout or swizzle that is applied last
> - offset: An integer tuple representing a coordinate offset
> - outer: The outer layout that is applied first

This composition allows for complex transformations of coordinates and indices,
enabling operations like tiling, partitioning, and reshaping of data.

* **Variables:**
  * **inner** – The inner layout or swizzle component
  * **offset** – The coordinate offset applied between inner and outer layouts
  * **outer** – The outer layout component
  * **max_alignment** – The maximum alignment of the composed layout

**Examples:**

```python
# Create a composed layout with inner layout, offset, and outer layout

# inner layout: (4, 8):(1, 4)
inner_layout = make_layout((4, 8))

offset = (0, 0)

# outer layout: (2, 2):(1@0, 1@1)
outer_layout = make_layout((2, 2), stride=(1 * E(0), 1 * E(1)))

# composed layout: (inner o offset o outer)
composed = make_composed_layout(inner_layout, offset, outer_layout)

# Accessing components of the composed layout
inner = composed.inner
offset = composed.offset
outer = composed.outer

# map coordinate (0, 1) to linear index
#  - outer(0, 1) = (0, 1)
#  - offset + outer(0, 1) = (0, 1)
#  - inner(0, 1) = 0 * 1 + 1 * 4 = 4
idx = crd2idx((0, 1), composed)

# Composition is used in many tiling operations
# For example, in logical_product, raked_product, and blocked_product
```

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstract property* inner

#### *abstract property* is_normal *: bool*

#### *abstract property* offset *: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]*

#### *abstract property* outer *: [Layout](#cutlass.cute.Layout)*

#### *abstract property* shape

#### *abstract property* type *: Type*

### *class* cutlass.cute.CopyAtom(op: Op, trait: Trait)

Bases: [`Atom`](#cutlass.cute.Atom)

The Copy Atom class.

#### \_abc_impl *= <_abc._abc_data object>*

#### *property* layout_dst_tv *: [Layout](#cutlass.cute.Layout)*

#### *property* layout_src_tv *: [Layout](#cutlass.cute.Layout)*

#### *property* smem_layout

Convenience property to access the SMEM layout for TMA copy atoms.

This is a shortcut for `atom.op.smem_layout` that checks if the operation
is a TMA operation and provides a clearer error message if not.

* **Returns:**
  The SMEM layout
* **Return type:**
  [Layout](#cutlass.cute.Layout) or [ComposedLayout](#cutlass.cute.ComposedLayout)
* **Raises:**
  * **TypeError** – If the operation is not a TMA operation
  * **ValueError** – If the SMEM layout is not set

Example:
: ```pycon
  >>> layout = tma_atom.smem_layout  # Instead of tma_atom.op.smem_layout
  ```

#### *property* thr_id *: [Layout](#cutlass.cute.Layout)*

#### *property* value_type *: Type[Numeric]*

### cutlass.cute.E(mode: int | List[int]) → ScaledBasis

Create a unit ScaledBasis element with the specified mode.

This function creates a ScaledBasis with value 1 and the given mode.
The mode represents the coordinate axis or dimension in the layout.

* **Parameters:**
  **mode** (*Union* *[**int* *,* *List* *[**int* *]* *]*) – The mode (dimension) for the basis element, either a single integer or a list of integers
* **Returns:**
  A ScaledBasis with value 1 and the specified mode
* **Return type:**
  ScaledBasis
* **Raises:**
  **TypeError** – If mode is not an integer or a list

**Examples:**

```python
# Create a basis element for the first dimension (mode 0)
e0 = E(0)

# Create a basis element for the second dimension (mode 1)
e1 = E(1)

# Create a basis element for a hierarchical dimension
e_hier = E([0, 1])
```

### *class* cutlass.cute.Layout(op_result)

Bases: `Value`

#### \_\_init_\_(self, value: cutlass._mlir._mlir_libs._cutlass_ir._mlir.ir.Value) → None

#### get_hier_coord(idx) → int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...]

Return the (hierarchical) ND logical coordinate corresponding to the linear index

#### *property* shape *: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]*

#### *property* stride *: int | Integer | ScaledBasis | Tuple[int | Integer | ScaledBasis | Tuple[Stride, ...], ...]*

### *class* cutlass.cute.MmaAtom(op: Op, trait: Trait)

Bases: [`Atom`](#cutlass.cute.Atom)

The MMA Atom class.

#### \_abc_impl *= <_abc._abc_data object>*

#### make_fragment_A(input, , loc=None, ip=None)

#### make_fragment_B(input, , loc=None, ip=None)

#### make_fragment_C(input, , loc=None, ip=None)

#### *property* shape_mnk *: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]*

#### *property* thr_id *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_A *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_B *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_C *: [Layout](#cutlass.cute.Layout)*

### *class* cutlass.cute.ReductionOp(\*values)

Bases: `IntEnum`

Op for cute reduce operations

#### ADD *= 0*

#### MAX *= 3*

#### MIN *= 2*

#### MUL *= 1*

### *class* cutlass.cute.Swizzle(\*args, \*\*kwargs)

Bases: `Value`

Swizzle is a transformation that permutes the elements of a layout.

Swizzles are used to rearrange data elements to improve memory access patterns
and computational efficiency.

Swizzle is defined by three parameters:
- MBase: The number of least-significant bits to keep constant
- BBits: The number of bits in the mask
- SShift: The distance to shift the mask

The mask is applied to the least-significant bits of the layout.

```default
0bxxxxxxxxxxxxxxxYYYxxxxxxxZZZxxxx
                              ^--^ MBase is the number of least-sig bits to keep constant
                 ^-^       ^-^     BBits is the number of bits in the mask
                   ^---------^     SShift is the distance to shift the YYY mask
                                      (pos shifts YYY to the right, neg shifts YYY to the left)

e.g. Given
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxZZxxx

the result is
0bxxxxxxxxxxxxxxxxYYxxxxxxxxxAAxxx where AA = ZZ `xor` YY
```

#### *property* num_base *: int*

Returns the number of least-significant bits to keep constant (M in Sw<B,M,S>).

#### *property* num_bits *: int*

Returns the number of bits in the mask (B in Sw<B,M,S>).

#### *property* num_shift *: int*

Returns the distance to shift the mask (S in Sw<B,M,S>).

### *class* cutlass.cute.SymInt(width: Literal[32, 64] = 32, , divisibility=1)

Bases: `object`

#### \_\_init_\_(width: Literal[32, 64] = 32, , divisibility=1)

#### *property* divisibility

#### *property* width

### *class* cutlass.cute.Tensor

Bases: `ABC`

Abstract base class for Tensor representations in CuTe DSL.

A CuTe Tensor is iterator with layout. A tensor evaluates the layout by mapping a
coordinate to the codomain, offsets the iterator accordingly, and dereferences
the result to obtain the tensor’s value.

**Formally:**

$$
T(c) = (E \circ L)(c) = *(E + L(c))
$$

where

> - $E$ is the iterator/engine
> - $L$ is the layout

**Notes:**

> - The tensor supports both direct element access via coordinates and slicing operations
> - Load/store operations are only supported for specific memory spaces (rmem, smem, gmem, generic)
> - For composed layouts, stride information is not directly accessible
> - Dynamic layouts do not support vector load/store operations

**Examples:**

Create tensor from torch.tensor with Host Runtime:

```python
import torch
from cutlass.cute.runtime import from_dlpack

mA = from_dlpack(torch.tensor([1, 3, 5], dtype=torch.int32))
print(mA.shape)   # (3,)
print(mA.stride)  # (1,)
print(mA.layout)  # (3,):(1,)
```

Define JIT function:

```python
@cute.jit
def add(a: Tensor, b: Tensor, res: Tensor):
    res.store(a.load() + b.load())
```

Call JIT function from python:

```python
import torch
a = torch.tensor([1, 3, 5], dtype=torch.int32)
b = torch.tensor([2, 4, 6], dtype=torch.int32)
c = torch.zeros([3], dtype=torch.int32)
mA = from_dlpack(a)
mB = from_dlpack(b)
mC = from_dlpack(c)
add(mA, mB, mC)
print(c)  # tensor([3, 7, 11], dtype=torch.int32)
```

#### \_abc_impl *= <_abc._abc_data object>*

#### *abstract property* element_type *: Type[Numeric] | Type[int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]]*

#### *abstractmethod* fill(value: Numeric) → None

#### *abstract property* iterator *: Pointer | int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]*

#### *property* layout *: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout)*

#### load(, loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

#### mark_compact_shape_dynamic(mode: int, stride_order: tuple[int, ...] | None = None, divisibility: int = 1) → [Tensor](#cutlass.cute.Tensor)

#### mark_layout_dynamic(leading_dim: int | None = None) → [Tensor](#cutlass.cute.Tensor)

#### *abstract property* memspace *: [AddressSpace](#cutlass.cute.AddressSpace)*

#### *property* shape *: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]*

#### store(data: [TensorSSA](#cutlass.cute.TensorSSA), , loc=None, ip=None)

#### *property* stride *: int | Integer | ScaledBasis | Tuple[int | Integer | ScaledBasis | Tuple[Stride, ...], ...]*

### *class* cutlass.cute.TensorSSA(value, shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], dtype: Type[Numeric])

Bases: `ArithValue`

A class representing thread local data from CuTe Tensor in value semantic and immutable.

* **Parameters:**
  * **value** (*ir.Value*) – Flatten vector as ir.Value holding logic data of SSA Tensor
  * **shape** (*Shape*) – The nested shape in CuTe of the vector
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the tensor elements
* **Variables:**
  * **\_shape** – The nested shape in CuTe of the vector
  * **\_dtype** – Data type of the tensor elements
* **Raises:**
  **ValueError** – If shape is not static

#### \_\_init_\_(value, shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], dtype: Type[Numeric])

Initialize a new TensorSSA object.

* **Parameters:**
  * **value** (*ir.Value*) – Flatten vector as ir.Value holding logic data of SSA Tensor
  * **shape** (*Shape*) – The nested shape in CuTe of the vector
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the tensor elements
* **Raises:**
  **ValueError** – If shape is not static

#### \_apply_op(op, other: [TensorSSA](#cutlass.cute.TensorSSA), flip=False, , loc, ip) → [TensorSSA](#cutlass.cute.TensorSSA)

#### \_apply_op(op, other: ArithValue, flip=False, , loc, ip) → [TensorSSA](#cutlass.cute.TensorSSA)

#### \_apply_op(op, other: int | float | bool, flip=False, , loc, ip) → [TensorSSA](#cutlass.cute.TensorSSA)

#### \_build_result(res_vect, res_shp, , row_major=False, loc=None, ip=None)

#### \_flatten_shape_and_coord(crd, , loc=None, ip=None)

#### apply_op(op, other, flip=False, , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Apply a binary operation to this tensor and another operand.

This is a public interface to the internal \_apply_op method, providing
a stable API for external users who need to apply custom operations.

Args:
: op: The operation function (e.g., operator.add, operator.mul, etc.)
  other: The other operand (TensorSSA, ArithValue, or scalar)
  flip: Whether to flip the operands (for right-hand operations)
  loc: MLIR location (optional)
  ip: MLIR insertion point (optional)

Returns:
: TensorSSA: The result of the operation

Example:
: ```pycon
  >>> tensor1 = cute.Tensor(...)
  >>> tensor2 = cute.Tensor(...)
  >>> result = tensor1.apply_op(operator.add, tensor2)
  >>> # Equivalent to: tensor1 + tensor2
  ```

#### broadcast_to(target_shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Broadcast the tensor to the target shape.

#### *property* dtype *: Type[Numeric]*

#### *property* element_type *: Type[Numeric]*

#### ir_value(, loc=None, ip=None)

#### ir_value_int8(, loc=None, ip=None)

Returns int8 ir value of Boolean tensor.
When we need to store Boolean tensor ssa, use ir_value_int8().

* **Parameters:**
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point for MLIR operations, defaults to None
* **Returns:**
  The int8 value of this Boolean
* **Return type:**
  ir.Value

#### reduce(op, init_val, reduction_profile: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], , loc=None, ip=None)

Perform reduce on selected modes with given predefined reduction op.

* **Parameters:**
  * **op** (*operator*) – The reduction operator to use (operator.add or operator.mul)
  * **init_val** (*numeric*) – The initial value for the reduction
  * **reduction_profile** (*Coord*) – Specifies which dimensions to reduce. Dimensions marked with None are kept.
* **Returns:**
  The reduced tensor
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

**Examples:**

```python
reduce(f32 o (4,))
  => f32

reduce(f32 o (4, 5))
  => f32
reduce(f32 o (4, (5, 4)), reduction_profile=(None, 1))
  => f32 o (4,)
reduce(f32 o (4, (5, 4)), reduction_profile=(None, (None, 1)))
  => f32 o (4, (5,))
```

#### reshape(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Reshape the tensor to a new shape.

* **Parameters:**
  **shape** (*Shape*) – The new shape to reshape to.
* **Returns:**
  A new tensor with the same elements but with the new shape.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)
* **Raises:**
  * **NotImplementedError** – If dynamic size is not supported
  * **ValueError** – If the new shape is not compatible with the current shape

#### *property* shape

#### to(dtype: Type[Numeric], , loc=None, ip=None)

Convert the tensor to a different numeric type.

* **Parameters:**
  **dtype** (*Type* *[**Numeric* *]*) – The target numeric type to cast to.
* **Returns:**
  A new tensor with the same shape but with elements cast to the target type.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)
* **Raises:**
  * **TypeError** – If dtype is not a subclass of Numeric.
  * **NotImplementedError** – If dtype is an unsigned integer type.

### *class* cutlass.cute.ThrCopy(op: Op, trait: Trait, thr_idx: int | Int32)

Bases: [`TiledCopy`](#cutlass.cute.TiledCopy)

The thread Copy class for modeling a thread-slice of a tiled Copy.

#### \_\_init_\_(op: Op, trait: Trait, thr_idx: int | Int32) → None

#### \_abc_impl *= <_abc._abc_data object>*

#### partition_D(dst: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### partition_S(src: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### *property* thr_idx

### *class* cutlass.cute.ThrMma(op: Op, trait: Trait, thr_idx: int | Int32)

Bases: [`TiledMma`](#cutlass.cute.TiledMma)

The thread MMA class for modeling a thread-slice of a tiled MMA.

#### \_\_init_\_(op: Op, trait: Trait, thr_idx: int | Int32) → None

#### \_abc_impl *= <_abc._abc_data object>*

#### partition_A(input_mk: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### partition_B(input_nk: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### partition_C(input_mn: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### *property* thr_idx

### *class* cutlass.cute.TiledCopy(op: Op, trait: Trait)

Bases: [`CopyAtom`](#cutlass.cute.CopyAtom)

The tiled Copy class.

#### \_abc_impl *= <_abc._abc_data object>*

#### get_slice(thr_idx: int | Int32) → [ThrCopy](#cutlass.cute.ThrCopy)

#### *property* layout_dst_tv_tiled *: [Layout](#cutlass.cute.Layout)*

#### *property* layout_src_tv_tiled *: [Layout](#cutlass.cute.Layout)*

#### *property* layout_tv_tiled *: [Layout](#cutlass.cute.Layout)*

#### retile(src, , loc=None, ip=None)

#### *property* size *: int*

#### *property* tiler_mn *: int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[Tile, ...], ...]*

### *class* cutlass.cute.TiledMma(op: Op, trait: Trait)

Bases: [`MmaAtom`](#cutlass.cute.MmaAtom)

The tiled MMA class.

#### \_abc_impl *= <_abc._abc_data object>*

#### \_partition_shape(operand_id, shape, , loc=None, ip=None)

#### \_thrfrg(operand_id, input: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

#### \_thrfrg(operand_id, input: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

#### \_thrfrg_A(input: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor)

#### \_thrfrg_B(input: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor)

#### \_thrfrg_C(input: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor)

#### get_slice(thr_idx: int | Int32) → [ThrMma](#cutlass.cute.ThrMma)

#### get_tile_size(mode_idx: int) → int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]

#### partition_shape_A(shape_mk, , loc=None, ip=None)

#### partition_shape_B(shape_nk, , loc=None, ip=None)

#### partition_shape_C(shape_mn, , loc=None, ip=None)

#### *property* permutation_mnk *: int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[Tile, ...], ...]*

#### *property* size *: int*

#### *property* thr_layout_vmnk *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_A_tiled *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_B_tiled *: [Layout](#cutlass.cute.Layout)*

#### *property* tv_layout_C_tiled *: [Layout](#cutlass.cute.Layout)*

### cutlass.cute.acos(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise arc cosine of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the arc cosine of each element in input tensor
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = acos(y)  # Compute arc cosine
```

### cutlass.cute.all_(x: [TensorSSA](#cutlass.cute.TensorSSA), , loc=None, ip=None) → Boolean

Test whether all tensor elements evaluate to True.

* **Parameters:**
  **x** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Input tensor.
* **Returns:**
  Returns a TensorSSA scalar containing True if all elements of x are True, False otherwise.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.any_(x: [TensorSSA](#cutlass.cute.TensorSSA), , loc=None, ip=None) → Boolean

Test whether any tensor element evaluates to True.

* **Parameters:**
  **x** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Input tensor.
* **Returns:**
  Returns a TensorSSA scalar containing True if any element of x is True, False otherwise.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.append(input: [Layout](#cutlass.cute.Layout), elem: [Layout](#cutlass.cute.Layout), up_to_rank=None, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.append(input: [ComposedLayout](#cutlass.cute.ComposedLayout), elem: [Layout](#cutlass.cute.Layout), up_to_rank=None, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.append(input: XTuple, elem: XTuple, up_to_rank=None, , loc=None, ip=None) → XTuple

Extend input to rank up_to_rank by appending elem to the end of input.

This function extends the input object by appending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

* **Parameters:**
  * **input** (*Union* *[**Shape* *,* *Stride* *,* *Coord* *,* *IntTuple* *,* *Tile* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – Source to be appended to
  * **elem** (*Union* *[**Shape* *,* *Stride* *,* *Coord* *,* *IntTuple* *,* *Tile* *,* [*Layout*](#cutlass.cute.Layout) *]*) – Element to append to input
  * **up_to_rank** (*Union* *[**None* *,* *int* *]* *,* *optional*) – The target rank after extension, defaults to None
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point, defaults to None
* **Returns:**
  The extended result with appended elements
* **Return type:**
  Union[Shape, Stride, Coord, IntTuple, Tile, [Layout](#cutlass.cute.Layout), [ComposedLayout](#cutlass.cute.ComposedLayout), [Tensor](#cutlass.cute.Tensor)]
* **Raises:**
  * **ValueError** – If up_to_rank is less than input’s current rank
  * **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Append to a Shape
shape = (4,4)
append(shape, 2)                   # Returns (4,4,2)

# Append to a Layout
layout = make_layout((8,8))
append(layout, make_layout((2,)))  # Returns (8,8,2):(1,8,1)

# Append with target rank
coord = (1,1)
append(coord, 0, up_to_rank=4)     # Returns (1,1,0,0)
```

Note:
: - The function preserves the structure of the input while extending it
  - Can be used to extend tensors, layouts, shapes and other CuTe types
  - When up_to_rank is specified, fills remaining positions with elem
  - Useful for tensor reshaping and layout transformations

### cutlass.cute.append_ones(t: [Layout](#cutlass.cute.Layout), up_to_rank: None | int = None, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.append_ones(t: [Tensor](#cutlass.cute.Tensor), up_to_rank: None | int = None, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.asin(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise arc sine of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the arc sine of each element in input tensor
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = asin(y)  # Compute arc sine
```

### cutlass.cute.assume(src, divby=None, , loc=None, ip=None)

### cutlass.cute.atan(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise arc tangent of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the arc tangent of each element in input tensor
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = atan(y)  # Compute arc tangent
```

### cutlass.cute.atan2(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, b: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise arc tangent of two tensors.

Computes atan2(a, b) element-wise. The function atan2(a, b) is the angle in radians
between the positive x-axis and the point given by the coordinates (b, a).

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – First input tensor (y-coordinates)
  * **b** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Second input tensor (x-coordinates)
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the arc tangent of a/b element-wise
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
y = cute.make_rmem_tensor(ptr1, layout).load()  # y coordinates
x = cute.make_rmem_tensor(ptr2, layout).load()  # x coordinates
theta = atan2(y, x)  # Compute angles
```

### cutlass.cute.autovec_copy(src: [Tensor](#cutlass.cute.Tensor), dst: [Tensor](#cutlass.cute.Tensor), \*, l1c_evict_priority: [CacheEvictionPriority](cute_nvgpu_common.md#cutlass.cute.nvgpu.CacheEvictionPriority) = <CacheEvictionPriority.EVICT_NORMAL>, loc=None, ip=None) → None

Auto-vectorization SIMT copy policy.

Given a source and destination tensors that are statically shaped, this policy figures out the
largest safe vector width that the copy instruction can take and performs the copy.

### cutlass.cute.basic_copy(src: [Tensor](#cutlass.cute.Tensor), dst: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → None

Performs a basic element-wise copy.

This functions **assumes** the following pre-conditions:
1. size(src) == size(dst)

When the src and dst shapes are static, the pre-conditions are actually verified and the
element-wise loop is fully unrolled.

* **Parameters:**
  * **src** ([*Tensor*](#cutlass.cute.Tensor)) – Source tensor
  * **dst** ([*Tensor*](#cutlass.cute.Tensor)) – Destination tensor
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None

### cutlass.cute.basic_copy_if(pred: [Tensor](#cutlass.cute.Tensor), src: [Tensor](#cutlass.cute.Tensor), dst: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → None

Performs a basic predicated element-wise copy.

This functions **assumes** the following pre-conditions:
1. size(src) == size(dst)
2. size(src) == size(pred)

When all shapes are static, the pre-conditions are actually verified and the element-wise loop
is fully unrolled.

### cutlass.cute.blocked_product(block: [Layout](#cutlass.cute.Layout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.blocked_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.ceil_div(input: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], tiler: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | [Layout](#cutlass.cute.Layout) | None | Tuple[int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[Tile, ...], ...], , loc=None, ip=None) → int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]

Compute the ceiling division of a target shape by a tiling specification.

This function computes the number of tiles required to cover the target domain.
It is equivalent to the second mode of zipped_divide(input, tiler).

* **Parameters:**
  * **input** (*Shape*) – A tuple of integers representing the dimensions of the target domain.
  * **tiler** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* *Shape* *,* *Tile* *]*) – The tiling specification.
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.
* **Returns:**
  A tuple of integers representing the number of tiles required along each dimension,
  i.e. the result of the ceiling division of the input dimensions by the tiler dimensions.
* **Return type:**
  Shape

Example:

```python
import cutlass.cute as cute
@cute.jit
def foo():
    input = (10, 6)
    tiler = (3, 4)
    result = cute.ceil_div(input, tiler)
    print(result)  # Outputs: (4, 2)
```

### cutlass.cute.coalesce(input, , target_profile: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...] = None, loc=None, ip=None)

### cutlass.cute.complement(input: [Layout](#cutlass.cute.Layout), cotarget: [Layout](#cutlass.cute.Layout) | int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

Compute the complement layout of the input layout with respect to the cotarget.

The complement of a layout A with respect to cotarget n is a layout A\* such that
for every k in Z_n and c in the domain of A, there exists a unique c\* in the domain
of A\* where k = A(c) + A\*(c\*).

This operation is useful for creating layouts that partition a space in complementary ways,
such as row and column layouts that together cover a matrix.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout)) – The layout to compute the complement of
  * **cotarget** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* *Shape* *]*) – The target layout or shape that defines the codomain
  * **loc** (*optional*) – Optional location information for IR diagnostics
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions
* **Returns:**
  The complement layout
* **Return type:**
  [Layout](#cutlass.cute.Layout)

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a right-major layout for a 4x4 matrix
    row_layout = cute.make_layout((4, 4), stride=(4, 1))
    # Create a left-major layout that complements the row layout
    col_layout = cute.complement(row_layout, 16)
    # The two layouts are complementary under 16
```

### cutlass.cute.composition(lhs: [Layout](#cutlass.cute.Layout), rhs: [Layout](#cutlass.cute.Layout) | Shape | Tile, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.composition(lhs: [ComposedLayout](#cutlass.cute.ComposedLayout), rhs: [Layout](#cutlass.cute.Layout) | Shape | Tile, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.composition(lhs: [Tensor](#cutlass.cute.Tensor), rhs: [Layout](#cutlass.cute.Layout) | Shape | Tile, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Compose two layout representations using the CuTe layout algebra.

Compose a left-hand layout (or tensor) with a right-hand operand into a new layout R, such that
for every coordinate c in the domain of the right-hand operand, the composed layout satisfies:

> R(c) = A(B(c))

where A is the left-hand operand provided as `lhs` and B is the right-hand operand provided as
`rhs`. In this formulation, B defines the coordinate domain while A applies its transformation to
B’s output, and the resulting layout R inherits the stride and shape adjustments from A.

Satisfies:
: cute.shape(cute.composition(lhs, rhs)) is compatible with cute.shape(rhs)

* **Parameters:**
  * **lhs** ([*Layout*](#cutlass.cute.Layout) *or* [*Tensor*](#cutlass.cute.Tensor)) – The left-hand operand representing the transformation to be applied.
  * **rhs** ([*Layout*](#cutlass.cute.Layout) *,* *Shape* *, or* *Tile* *, or* *int* *or* *tuple*) – The right-hand operand defining the coordinate domain. If provided as an int or tuple,
    it will be converted to a tile layout.
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.
* **Returns:**
  A new composed layout R, such that for all coordinates c in the domain of `rhs`,
  R(c) = lhs(rhs(c)).
* **Return type:**
  [Layout](#cutlass.cute.Layout) or [Tensor](#cutlass.cute.Tensor)

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout that maps (i,j) to i*4 + j
    L1 = cute.make_layout((2, 3), stride=(4, 1))
    # Create a layout that maps (i,j) to i*3 + j
    L2 = cute.make_layout((3, 4), stride=(3, 1))
    # Compose L1 and L2
    L3 = cute.composition(L1, L2)
    # L3 now maps coordinates through L2 then L1
```

### cutlass.cute.copy(atom: [CopyAtom](#cutlass.cute.CopyAtom), src: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], dst: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], , pred: [Tensor](#cutlass.cute.Tensor) | None = None, loc=None, ip=None, \*\*kwargs) → None

Facilitates data transfer between two tensors conforming to layout profile `(V, Rest...)`.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom specifying the transfer operation
  * **src** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – Source tensor or list of tensors with layout profile `(V, Rest...)`
  * **dst** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – Destination tensor or list of tensors with layout profile `(V, Rest...)`
  * **pred** (*Optional* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *optional*) – Optional predication tensor for conditional transfers, defaults to None
  * **loc** (*Any* *,* *optional*) – Source location information, defaults to None
  * **ip** (*Any* *,* *optional*) – Insertion point, defaults to None
  * **kwargs** (*Dict* *[**str* *,* *Any* *]*) – Additional copy atom specific arguments
* **Raises:**
  * **TypeError** – If source and destination element type bit widths differ
  * **ValueError** – If source and destination ranks differ
  * **ValueError** – If source and destination mode-1 sizes differ
  * **NotImplementedError** – If `V-mode` rank exceeds 2
* **Returns:**
  None
* **Return type:**
  None

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,
  and src/dst layout like `((V, Rest...), Rest...)`

The algorithm recursively processes the `V-mode`, decomposing it until reaching the minimum granularity
compatible with the provided Copy Atom’s requirements.

Source and destination tensors must be partitioned in accordance with the Copy Atom specifications.
Post-partitioning, both tensors will exhibit a `(V, Rest...)` layout profile.

The operands src and dst are variadic, each containing a variable number of tensors:

- For regular copy, src and dst contain single source and destination tensors respectively.
- For copy with auxiliary operands, src and dst contain the primary tensors followed by
  their respective auxiliary tensors.

**Precondition:** The size of mode 1 must be equal for both source and destination tensors:
`size(src, mode=[1]) == size(dst, mode=[1])`

**Examples**:

TMA copy operation with multicast functionality:

```python
cute.copy(tma_atom, src, dst, tma_bar_ptr=mbar_ptr, mcast_mask=mask, cache_policy=policy)
```

Optional predication is supported through an additional tensor parameter. For partitioned tensors with
logical profile `((ATOM_V,ATOM_REST),REST,...)`, the predication tensor must maintain profile
compatibility with `(ATOM_REST,REST,...)`.

For Copy Atoms requiring single-threaded execution, thread election is managed automatically by the
copy operation. External thread selection mechanisms are not necessary.

#### NOTE
- Certain Atoms may require additional operation-specific keyword arguments.
- Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
  for future releases.

### cutlass.cute.copy_atom_call(atom: [CopyAtom](#cutlass.cute.CopyAtom), src: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], dst: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], , pred: [Tensor](#cutlass.cute.Tensor) | None = None, loc=None, ip=None, \*\*kwargs) → None

Execute a single copy atom operation.

The copy_atom_call operation executes a copy atom with the given operands.
Source and destination tensors have layout profile `(V)`.

The `V-mode` represents either:

- A singular mode directly consumable by the provided Copy Atom
- A composite mode requiring recursive decomposition, structured as `(V, Rest...)`,

For src/dst layout like `(V, Rest...)`, the layout profile of `pred` must match `(Rest...)`.

> - Certain Atoms may require additional operation-specific keyword arguments.
> - Current implementation limits `V-mode` rank to 2 or less. Support for higher ranks is planned
>   for future releases.

Both `src` and `dst` operands are variadic, containing a variable number of tensors:

- For regular copy, `src` and `dst` each contain a single tensor.
- For copy with auxiliary operands, they contain the main tensor followed by
  auxiliary tensors. For example:

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom specifying the transfer operation
  * **src** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – Source tensor(s) with layout profile `(V)`. Can be a single Tensor
    or a list/tuple of Tensors for operations with auxiliary source operands.
  * **dst** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – Destination tensor(s) with layout profile `(V)`. Can be a single Tensor
    or a list/tuple of Tensors for operations with auxiliary destination operands.
  * **pred** (*Optional* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *optional*) – Optional predication tensor for conditional transfers, defaults to None
  * **loc** (*Any* *,* *optional*) – Source location information, defaults to None
  * **ip** (*Any* *,* *optional*) – Insertion point, defaults to None
  * **kwargs** (*Dict* *[**str* *,* *Any* *]*) – Additional copy atom specific arguments
* **Raises:**
  **TypeError** – If source and destination element type bit widths differ
* **Returns:**
  None
* **Return type:**
  None

**Examples**:

```python
# Regular copy atom operation
cute.copy_atom_call(copy_atom, src, dst)

# Predicated copy atom operation
cute.copy_atom_call(copy_atom, src, dst, pred=pred)
```

### cutlass.cute.cos(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise cosine of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor (in radians)
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the cosine of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = cos(y)  # Compute cosine
```

### cutlass.cute.cosize(a: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor), mode: List[int] = [], , loc=None, ip=None)

Return size of codomain of layout or tensor. Return static value if type is static.

For a layout `L = S:D` where `S` is the shape and `D` is the stride, the codomain size is the
minimum size needed to store all possible offsets generated by the layout. This is calculated
by taking the maximum offset plus 1.

For example, given a layout `L = (4,(3,2)):(2,(8,1))`:
: - Shape `S = (4,(3,2))`
  - Stride `D = (2,(8,1))`
  - Maximum offset = `2*(4-1) + 8*(3-1) + 1*(2-1) = 6 + 16 + 1 = 23`
  - Therefore `cosize(L) = 24`

**Examples:**

```python
L = cute.make_layout((4,(3,2)), stride=(2,(8,1))) # L = (4,(3,2)):(2,(8,1))
print(cute.cosize(L))  # => 24
```

* **Parameters:**
  * **a** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – Layout, ComposedLayout, or Tensor object
  * **mode** (*List* *[**int* *]* *,* *optional*) – List of mode(s) for cosize calculation. If empty, calculates over all modes.
    If specified, calculates cosize only for the given modes.
  * **loc** (*optional*) – Location information for diagnostics, defaults to None
  * **ip** (*optional*) – Instruction pointer for diagnostics, defaults to None
* **Returns:**
  Static size of layout or tensor (fast fold) if static, or a dynamic Value
* **Return type:**
  Union[int, Value]

### cutlass.cute.crd2idx(coord: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], layout, , loc=None, ip=None)

Convert a multi-dimensional coordinate into a value using the specified layout.

This function computes the inner product of the flattened coordinate and stride:

> index = sum(flatten(coord)[i] \* flatten(stride)[i] for i in range(len(coord)))
* **Parameters:**
  * **coord** (*Coord*) – A tuple or list representing the multi-dimensional coordinate
    (e.g., (i, j) for a 2D layout).
  * **layout** ([*Layout*](#cutlass.cute.Layout) *or* [*ComposedLayout*](#cutlass.cute.ComposedLayout)) – A layout object that defines the memory storage layout, including shape and stride,
    used to compute the inner product.
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.
* **Returns:**
  The result of applying the layout transformation to the provided coordinate.
* **Return type:**
  Any type that the layout maps to

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    L = cute.make_layout((5, 4), stride=(4, 1))
    idx = cute.crd2idx((2, 3), L)
    # Computed as: 2 * 4 + 3 = 11
    print(idx)
foo()  # Expected output: 11
```

### cutlass.cute.depth(a: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout)) → int

Returns the depth (nesting level) of a tuple, layout, or tensor.

The depth of a tuple is the maximum depth of its elements plus 1.
For an empty tuple, the depth is 1. For layouts and tensors, the depth
is determined by the depth of their shape. For non-tuple values (e.g., integers),
the depth is considered 0.

* **Parameters:**
  **a** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *,* *Any* *]*) – The object whose depth is to be determined
* **Returns:**
  The depth of the input object
* **Return type:**
  int

**Example:**

```python
>>> depth(1)
0
>>> depth((1, 2))
1
>>> depth(((1, 2), (3, 4)))
2
```

### cutlass.cute.dice(src: [Layout](#cutlass.cute.Layout), dicer: Coord, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.dice(src: [ComposedLayout](#cutlass.cute.ComposedLayout), dicer: Coord, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.dice(src: XTuple, dicer: Coord, , loc=None, ip=None) → XTuple

Keep modes in input when it is paired with an integer in dicer.

This function performs dicing operation on the input based on the dicer coordinate.
Dicing is a fundamental operation in CuTe that allows selecting specific modes from
a tensor or layout based on a coordinate pattern.

* **Parameters:**
  * **dicer** (*Coord*) – A static coordinate indicating how to dice the input
  * **input** (*Union* *[**IntTuple* *,* *Shape* *,* *Stride* *,* *Coord* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *]*) – The operand to be diced on
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  The diced result with selected modes from the input
* **Return type:**
  Union[IntTuple, Shape, Stride, Coord, [Layout](#cutlass.cute.Layout), [ComposedLayout](#cutlass.cute.ComposedLayout)]
* **Raises:**
  * **TypeError** – If dicer has an unsupported type
  * **ValueError** – If input is not provided

**Examples:**

```python
# Basic dicing of a layout
layout = make_layout((32,16,8))

# Keep only first and last modes
diced = dice((1,None,1), layout)
```

Note:
: - The dicer coordinate must be static
  - Use underscore (_) to remove a mode

### cutlass.cute.domain_offset(coord: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], tensor: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.elem_less(lhs: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | Tuple[int | Integer | Tuple[IntTuple, ...], ...] | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], rhs: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | Tuple[int | Integer | Tuple[IntTuple, ...], ...] | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], , loc=None, ip=None) → Boolean

### cutlass.cute.empty_like(a, dtype=None, , loc=None, ip=None)

Return a new TensorSSA with the same shape and type as a given array, without initializing entries.

* **Parameters:**
  * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
  * **dtype** (*Type* *[**Numeric* *]* *,* *optional*) – Overrides the data type of the result, defaults to None
* **Returns:**
  Uninitialized tensor with the same shape and type (unless overridden) as a.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.erf(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise error function of the input tensor.

The error function is defined as:
erf(x) = 2/√π ∫[0 to x] exp(-t²) dt

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the error function value for each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = erf(y)  # Compute error function
```

### cutlass.cute.exp(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise exponential of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the exponential of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = exp(y)  # Compute exponential
```

### cutlass.cute.exp2(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise base-2 exponential of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing 2 raised to the power of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = exp2(y)  # Compute 2^x
```

### cutlass.cute.filter(input: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.filter(input: [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.filter(input: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Filter a layout or tensor.

This function filters a layout or tensor according to CuTe’s filtering rules.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout) *or* [*Tensor*](#cutlass.cute.Tensor)) – The input layout or tensor to filter
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The filtered layout or tensor
* **Return type:**
  [Layout](#cutlass.cute.Layout) or [Tensor](#cutlass.cute.Tensor)
* **Raises:**
  **TypeError** – If input is not a Layout or Tensor

### cutlass.cute.filter_tuple(\*args, f: Callable)

Filter and flatten tuple elements by applying a function.

The function f should return tuples, which are then concatenated together
to produce the final result. This is useful for filtering and transforming
tuple structures in a single pass.

* **Parameters:**
  * **t** (*Union* *[**tuple* *,* *ir.Value* *,* *int* *]*) – The tuple to filter
  * **f** (*Callable*) – The function to apply to each element of t
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  A concatenated tuple of all results
* **Return type:**
  tuple

**Examples:**

```python
>>> # Keep only even numbers, wrapped in tuples
>>> filter_tuple((1, 2, 3, 4), lambda x: (x,) if x % 2 == 0 else ())
(2, 4)
>>> # Duplicate each element
>>> filter_tuple((1, 2, 3), lambda x: (x, x))
(1, 1, 2, 2, 3, 3)
```

### cutlass.cute.filter_zeros(input: [Layout](#cutlass.cute.Layout), , target_profile=None, loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.filter_zeros(input: [Tensor](#cutlass.cute.Tensor), , target_profile=None, loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Filter out zeros from a layout or tensor.

This function removes zero-stride dimensions from a layout or tensor.
Refer to [https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md)
for more layout algebra operations.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout) *or* [*Tensor*](#cutlass.cute.Tensor)) – The input layout or tensor to filter
  * **target_profile** (*Stride* *,* *optional*) – Target stride profile for the filtered result, defaults to None
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The filtered layout or tensor with zeros removed
* **Return type:**
  [Layout](#cutlass.cute.Layout) or [Tensor](#cutlass.cute.Tensor)
* **Raises:**
  **TypeError** – If input is not a Layout or Tensor

### cutlass.cute.find(t: tuple | Value | int, x: int, , loc=None, ip=None) → int | Tuple[int, ...] | None

Find the first position of a value `x` in a hierarchical structure `t`.

Searches for the first occurrence of x in t, optionally excluding positions
where a comparison value matches. The search can traverse nested structures
and returns either a single index or a tuple of indices for nested positions.

* **Parameters:**
  * **t** (*Union* *[**tuple* *,* *ir.Value* *,* *int* *]*) – The search space
  * **x** (*int*) – The static integer x to search for
* **Returns:**
  Index if found at top level, tuple of indices showing nested position, or None if not found
* **Return type:**
  Union[int, Tuple[int, …], None]

### cutlass.cute.find_if(t: tuple | Value | int, pred_fn: Callable[[tuple | Value | int, int], bool], , loc=None, ip=None) → int | Tuple[int, ...] | None

### cutlass.cute.flat_divide(target: [Layout](#cutlass.cute.Layout), tiler: Tile, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.flat_divide(target: [Tensor](#cutlass.cute.Tensor), tiler: Tile, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.flat_product(block: [Layout](#cutlass.cute.Layout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.flat_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.flatten(a: [Layout](#cutlass.cute.Layout)) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.flatten(a: [Tensor](#cutlass.cute.Tensor)) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.flatten(a: Any | Tuple[XTuple, ...]) → Any | Tuple[XTuple, ...]

Flattens a CuTe data structure into a simpler form.

For tuples, this function flattens the structure into a single-level tuple.
For layouts, it returns a new layout with flattened shape and stride.
For tensors, it returns a new tensor with flattened layout.
For other types, it returns the input unchanged.

* **Parameters:**
  **a** (*Union* *[**IntTuple* *,* *Coord* *,* *Shape* *,* *Stride* *,* [*Layout*](#cutlass.cute.Layout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – The structure to flatten
* **Returns:**
  The flattened structure
* **Return type:**
  Union[tuple, Any]

**Examples:**

```python
flatten((1, 2, 3))                      # Returns (1, 2, 3)
flatten(((1, 2), (3, 4)))               # Returns (1, 2, 3, 4)
flatten(5)                              # Returns 5
flatten(Layout(shape, stride))          # Returns Layout(flatten(shape), flatten(stride))
flatten(Tensor(layout))                 # Returns Tensor(flatten(layout))
```

### cutlass.cute.flatten_to_tuple(a: Any | Tuple[Any | Tuple[XTuple, ...], ...]) → Tuple[Any, ...]

Flattens a potentially nested tuple structure into a flat tuple.

This function recursively traverses the input structure and flattens it into
a single-level tuple, preserving the order of elements.

* **Parameters:**
  **a** (*Union* *[**IntTuple* *,* *Coord* *,* *Shape* *,* *Stride* *]*) – The structure to flatten
* **Returns:**
  A flattened tuple containing all elements from the input
* **Return type:**
  tuple

**Examples:**

```python
flatten_to_tuple((1, 2, 3))       # Returns (1, 2, 3)
flatten_to_tuple(((1, 2), 3))     # Returns (1, 2, 3)
flatten_to_tuple((1, (2, (3,))))  # Returns (1, 2, 3)
```

### cutlass.cute.front(input, , loc=None, ip=None)

Recursively get the first element of input.

This function traverses a hierarchical structure (like a layout or tensor)
and returns the first element at the deepest level. It’s particularly useful
for accessing the first stride value in a layout to determine properties like
majorness.

* **Parameters:**
  * **input** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* [*Layout*](#cutlass.cute.Layout) *,* *Stride* *]*) – The hierarchical structure to traverse
  * **loc** (*source location* *,* *optional*) – Source location where it’s called, defaults to None
  * **ip** (*insertion pointer* *,* *optional*) – Insertion pointer for IR generation, defaults to None
* **Returns:**
  The first element at the deepest level of the input structure
* **Return type:**
  Union[int, float, bool, ir.Value]

### cutlass.cute.full(shape, fill_value, dtype: Type[Numeric], , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Return a new TensorSSA of given shape and type, filled with fill_value.

* **Parameters:**
  * **shape** (*tuple*) – Shape of the new tensor.
  * **fill_value** (*scalar*) – Value to fill the tensor with.
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the tensor.
* **Returns:**
  Tensor of fill_value with the specified shape and dtype.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.full_like(a: [TensorSSA](#cutlass.cute.TensorSSA) | [Tensor](#cutlass.cute.Tensor), fill_value, dtype: None | Type[Numeric] = None, , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Return a full TensorSSA with the same shape and type as a given array.

* **Parameters:**
  * **a** (*array_like*) – The shape and data-type of a define these same attributes of the returned array.
  * **fill_value** (*array_like*) – Fill value.
  * **dtype** (*Union* *[**None* *,* *Type* *[**Numeric* *]* *]* *,* *optional*) – Overrides the data type of the result, defaults to None
* **Returns:**
  Tensor of fill_value with the same shape and type as a.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

#### SEE ALSO
[`empty_like()`](#cutlass.cute.empty_like): Return an empty array with shape and type of input.
[`ones_like()`](#cutlass.cute.ones_like): Return an array of ones with shape and type of input.
[`zeros_like()`](#cutlass.cute.zeros_like): Return an array of zeros with shape and type of input.
[`full()`](#cutlass.cute.full): Return a new array of given shape filled with value.

**Examples:**

```python
frg = cute.make_rmem_tensor((2, 3), Float32)
a = frg.load()
b = cute.full_like(a, 1.0)
```

### cutlass.cute.gemm(atom: [MmaAtom](#cutlass.cute.MmaAtom), d: [Tensor](#cutlass.cute.Tensor), a: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], b: [Tensor](#cutlass.cute.Tensor) | List[[Tensor](#cutlass.cute.Tensor)] | Tuple[[Tensor](#cutlass.cute.Tensor), ...], c: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None, \*\*kwargs) → None

The GEMM algorithm.

Computes `D <- A * B + C` where `C` and `D` can alias. Note that some MMA Atoms (e.g.
warpgroup-wide or tcgen05 MMAs) require manually setting an “accumulate” boolean field.

All tensors must be partitioned according to the provided MMA Atom.

For MMA Atoms that require single-threaded execution, the gemm op automatically handles thread
election internally. Manual thread selection is not required in such cases.

Following dispatch rules are supported:

- Dispatch [1]: (V) x (V) => (V)          => (V,1,1) x (V,1,1) => (V,1,1)
- Dispatch [2]: (M) x (N) => (M,N)        => (1,M,1) x (1,N,1) => (1,M,N)
- Dispatch [3]: (M,K) x (N,K) => (M,N)    => (1,M,K) x (1,N,K) => (1,M,N)
- Dispatch [4]: (V,M) x (V,N) => (V,M,N)  => (V,M,1) x (V,N,1) => (V,M,N)
- Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)

Operand flexibility:
- a and b can be a single Tensor (regular GEMM) or a sequence [operand, scale_factor] for block-scaled GEMM.

* **Parameters:**
  * **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom)) – MMA atom
  * **d** ([*Tensor*](#cutlass.cute.Tensor)) – Destination tensor
  * **a** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – First source tensor or sequence for advanced modes (e.g., [a, sfa])
  * **b** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* *List* *[*[*Tensor*](#cutlass.cute.Tensor) *]* *,* *Tuple* *[*[*Tensor*](#cutlass.cute.Tensor) *,*  *...* *]* *]*) – Second source tensor or sequence for advanced modes (e.g., [b, sfb])
  * **c** ([*Tensor*](#cutlass.cute.Tensor)) – Third source tensor
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point for MLIR, defaults to None
  * **kwargs** (*dict*) – Additional keyword arguments
* **Returns:**
  None
* **Return type:**
  None

### cutlass.cute.get(input: [Layout](#cutlass.cute.Layout), mode, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.get(input: [ComposedLayout](#cutlass.cute.ComposedLayout), mode, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.get(input: Any | Tuple[XTuple, ...], mode, , loc=None, ip=None) → Any | Tuple[XTuple, ...]

Extract a specific element or sub-layout from a layout or tuple.

This function recursively traverses the input according to the mode indices,
extracting the element at the specified path. For layouts, this operation
corresponds to extracting a specific sub-layout.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* *tuple*) – The input layout or tuple to extract from
  * **mode** (*List* *[**int* *]*) – Indices specifying the path to traverse for extraction
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The extracted element or sub-layout
* **Return type:**
  [Layout](#cutlass.cute.Layout), [ComposedLayout](#cutlass.cute.ComposedLayout), or element type
* **Raises:**
  * **ValueError** – If any index in mode is out of range
  * **TypeError** – If mode contains non-integer elements or if input has unsupported type
* **Postcondition:**
  `get(t, mode=find(x,t)) == x if find(x,t) != None else True`

**Examples:**

```python
layout = make_layout(((4, 8), (16, 1), 8), stride=((1, 4), (32, 0), 512))
sub_layout = get(layout, mode=[0, 1])   # 8:4
sub_layout = get(layout, mode=[1])      # (16, 1):(32, 0)
```

### cutlass.cute.get_divisibility(x: int | Integer) → int

### cutlass.cute.get_leaves(value, , loc=None, ip=None)

### cutlass.cute.get_nonswizzle_portion(layout: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout)

Extract the non-swizzle portion from a layout.

For a simple Layout, the entire layout is considered non-swizzled and is returned as-is.
For a ComposedLayout, the inner layout (non-swizzled portion) is extracted and returned,
effectively separating the base layout from any swizzle transformation that may be applied.

* **Parameters:**
  * **layout** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *]*) – A Layout or ComposedLayout from which to extract the non-swizzle portion.
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional
* **Returns:**
  The non-swizzle portion of the input layout. For Layout objects, returns the layout itself.
  For ComposedLayout objects, returns the outer layout component.
* **Return type:**
  [Layout](#cutlass.cute.Layout)
* **Raises:**
  **TypeError** – If the layout is neither a Layout nor a ComposedLayout.

### cutlass.cute.get_swizzle_portion(layout: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None) → [Swizzle](#cutlass.cute.Swizzle)

Extract or create the swizzle portion from a layout.

For a simple Layout (which has no explicit swizzle), a default identity swizzle is created.
For a ComposedLayout, the outer layout is checked and returned if it is a Swizzle object.
Otherwise, a default identity swizzle is created. The default identity swizzle has parameters
(0, 4, 3), which represents a no-op swizzle transformation.

* **Parameters:**
  * **layout** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *]*) – A Layout or ComposedLayout from which to extract the swizzle portion.
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional
* **Returns:**
  The swizzle portion of the layout. For Layout objects or ComposedLayout objects without
  a Swizzle outer component, returns a default identity swizzle (0, 4, 3). For ComposedLayout
  objects with a Swizzle outer component, returns that swizzle.
* **Return type:**
  [Swizzle](#cutlass.cute.Swizzle)
* **Raises:**
  **TypeError** – If the layout is neither a Layout nor a ComposedLayout.

### cutlass.cute.group_modes(input: [Layout](#cutlass.cute.Layout), begin: int, end: int, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.group_modes(input: [ComposedLayout](#cutlass.cute.ComposedLayout), begin: int, end: int, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.group_modes(input: [Tensor](#cutlass.cute.Tensor), begin: int, end: int, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.group_modes(input: XTuple, begin: int, end: int, , loc=None, ip=None) → XTuple

Group modes of a hierarchical tuple or layout into a single mode.

This function groups a range of modes from the input object into a single mode,
creating a hierarchical structure. For tuples, it creates a nested tuple containing
the specified range of elements. For layouts and other CuTe objects, it creates
a hierarchical representation where the specified modes are grouped together.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* *tuple* *,* *Shape* *,* *Stride* *,* *etc.*) – Input object to group modes from (layout, tuple, etc.)
  * **beg** (*int*) – Beginning index of the range to group (inclusive)
  * **end** (*int*) – Ending index of the range to group (exclusive)
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  A new object with the specified modes grouped
* **Return type:**
  Same type as input with modified structure

**Examples:**

```python
# Group modes in a tuple
t = (2, 3, 4, 5)
grouped = group_modes(t, 1, 3)  # (2, (3, 4), 5)

# Group modes in a layout
layout = make_layout((2, 3, 4, 5))
grouped_layout = group_modes(layout, 1, 3)  # Layout with shape (2, (3, 4), 5)

# Group modes in a shape
shape = make_shape(2, 3, 4, 5)
grouped_shape = group_modes(shape, 0, 2)  # Shape ((2, 3), 4, 5)
```

### cutlass.cute.has_underscore(a: Any | Tuple[Any | Tuple[XTuple, ...], ...]) → bool

### cutlass.cute.idx2crd(idx: Int, shape: Int, , loc=None, ip=None) → Int

### cutlass.cute.idx2crd(idx: IntTuple, shape: Tuple, , loc=None, ip=None) → Tuple

Convert a linear index back into a multi-dimensional coordinate using the specified layout.

Mapping from a linear index to the corresponding multi-dimensional coordinate in the layout’s coordinate space.
It essentially “unfolds” a linear index into its constituent coordinate components.

* **Parameters:**
  * **idx** ( *: int/Integer/Tuple*) – The linear index to convert back to coordinates.
  * **shape** (*Shape*) – Shape of the layout defining the size of each mode
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.
* **Returns:**
  The result of applying the layout transformation to the provided coordinate.
* **Return type:**
  Coord

**Examples:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    coord = cute.idx2crd(11, (5, 4))
    # idx2crd is always col-major 
    # For shape (m, n, l, ...), coord = (idx % m, idx // m % n, idx // m // n % l, ...
    # Computed as: (11 % 5, 11 // 5 % 4) = (1, 2)
    print(coord)

foo()  # Expected output: (1, 2)
```

### cutlass.cute.is_congruent(a: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor), b: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor)) → bool

Returns whether a is congruent to b.

Congruence is an equivalence relation between hierarchical structures.

Two objects are congruent if:
\* They have the same rank, AND
\* They are both non-tuple values, OR
\* They are both tuples AND all corresponding elements are congruent.

Congruence requires type matching at each level – scalar values match with
scalar values, and tuples match with tuples of the same rank.

* **Parameters:**
  * **a** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – First object to compare
  * **b** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – Second object to compare
* **Returns:**
  True if a and b are congruent, False otherwise
* **Return type:**
  bool

### cutlass.cute.is_int_tuple(a) → bool

### cutlass.cute.is_integer(a) → bool

Check if an object is static integer or dynamic integer

### cutlass.cute.is_major(mode, stride: int | Integer | ScaledBasis | Tuple[int | Integer | ScaledBasis | Tuple[Stride, ...], ...], , loc=None, ip=None) → bool

Check whether a mode in stride is the major mode.

### cutlass.cute.is_static(x: Any) → bool

Check if a value is statically known at compile time.

In CuTe, static values are those whose values are known at compile time,
as opposed to dynamic values which are only known at runtime.

This function checks if a value is static by recursively traversing its type hierarchy
and checking if all components are static.

Static values include:
- Python literals (bool, int, float, None)
- Static ScaledBasis objects
- Static ComposedLayout objects
- Static IR types
- Tuples containing only static values

Dynamic values include:
- Numeric objects (representing runtime values)
- Dynamic expressions
- Any tuple containing dynamic values

* **Parameters:**
  **x** (*Any*) – The value to check
* **Returns:**
  True if the value is static, False otherwise
* **Return type:**
  bool
* **Raises:**
  **TypeError** – If an unsupported type is provided

### cutlass.cute.is_weakly_congruent(a: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor), b: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor)) → bool

Returns whether a is weakly congruent to b.

Weak congruence is a partial order on hierarchical structures.

Object X is weakly congruent to object Y if:
\* X is a non-tuple value, OR
\* X and Y are both tuples of the same rank AND all corresponding elements are weakly congruent.

Weak congruence allows scalar values to match with tuples, making it useful
for determining whether an object has a hierarchical structure “up to” another.

* **Parameters:**
  * **a** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – First object to compare
  * **b** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – Second object to compare
* **Returns:**
  True if a and b are weakly congruent, False otherwise
* **Return type:**
  bool

### cutlass.cute.jit(\*dargs, \*\*dkwargs)

Decorator to mark a function for JIT compilation for Host code.

### cutlass.cute.kernel(\*dargs, \*\*dkwargs)

Decorator to mark a function for JIT compilation for GPU.

### cutlass.cute.leading_dim(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], stride: int | Integer | ScaledBasis | Tuple[int | Integer | ScaledBasis | Tuple[Stride, ...], ...]) → int | Tuple[int, ...] | None

Find the leading dimension of a shape and stride.

* **Parameters:**
  * **shape** (*Shape*) – The shape of the tensor or layout
  * **stride** (*Stride*) – The stride of the tensor or layout
* **Returns:**
  The leading dimension index or indices
* **Return type:**
  Union[int, Tuple[int, …], None]

The return value depends on the stride pattern:

> * If a single leading dimension is found, returns an integer index
> * If nested leading dimensions are found, returns a tuple of indices
> * If no leading dimension is found, returns None

### cutlass.cute.left_inverse(input: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.local_partition(target: [Tensor](#cutlass.cute.Tensor), tiler: [Layout](#cutlass.cute.Layout) | int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], index: int | Numeric, proj: Any | Tuple[Any | Tuple[XTuple, ...], ...] = 1, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.local_tile(input: [Tensor](#cutlass.cute.Tensor), tiler: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | [Layout](#cutlass.cute.Layout) | None | Tuple[int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[Tile, ...], ...], coord: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], proj: Any | Tuple[Any | Tuple[XTuple, ...], ...] = None, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.log(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise natural logarithm of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the natural logarithm of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log(y)  # Compute natural logarithm
```

### cutlass.cute.log10(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise base-10 logarithm of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the base-10 logarithm of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log10(y)  # Compute log base 10
```

### cutlass.cute.log2(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise base-2 logarithm of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the base-2 logarithm of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = log2(y)  # Compute log base 2
```

### cutlass.cute.logical_divide(target: [Layout](#cutlass.cute.Layout), tiler: Tiler, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.logical_divide(target: [Tensor](#cutlass.cute.Tensor), tiler: Tiler, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.logical_product(block: [Layout](#cutlass.cute.Layout), tiler: Tile, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.logical_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: Tile, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.make_atom(ty, values=None, , loc=None, ip=None)

This is a wrapper around the \_cute_ir.make_atom operation, providing default value for the values argument.

### cutlass.cute.make_composed_layout(inner, offset: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...], outer: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

Create a composed layout by composing an inner transformation with an outer layout.

A composed layout applies a sequence of transformations
to coordinates. The composition is defined as (inner ∘ offset ∘ outer), where the operations
are applied from right to left.

* **Parameters:**
  * **inner** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*Swizzle*](#cutlass.cute.Swizzle) *]*) – The inner transformation (can be a Layout or Swizzle)
  * **offset** (*IntTuple*) – An integral offset applied between transformations
  * **outer** ([*Layout*](#cutlass.cute.Layout)) – The outer (right-most) layout that is applied first
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  A new ComposedLayout representing the composition
* **Return type:**
  [ComposedLayout](#cutlass.cute.ComposedLayout)

**Examples:**

```python
# Create a basic layout
inner = make_layout(...)
outer = make_layout((4,4), stride=(E(0), E(1)))

# Create a composed layout with an offset
composed = make_composed_layout(inner, (2,0), outer)
```

Note:
: - The composition applies transformations in the order: outer → offset → inner
  - The stride divisibility condition must be satisfied for valid composition
  - Certain compositions (like Swizzle with scaled basis) are invalid and will raise errors
  - Composed layouts inherit many properties from the outer layout

### cutlass.cute.make_copy_atom(op: CopyOp, copy_internal_type: Type[Numeric], , loc=None, ip=None, \*\*kwargs) → [CopyAtom](#cutlass.cute.CopyAtom)

Makes a Copy Atom from a Copy Operation.

This function creates a Copy Atom from a given Copy Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters.

Example:

```python
op = cute.nvgpu.CopyUniversalOp()
atom = cute.make_copy_atom(op, tensor_dtype, num_bits_per_copy=64)
```

* **Parameters:**
  * **op** (*CopyOp*) – The Copy Operation to construct an Atom for
  * **copy_internal_type** (*Type* *[**Numeric* *]*) – An internal data type used to construct the source/destination layouts in unit of tensor elements
* **Returns:**
  The Copy Atom
* **Return type:**
  [CopyAtom](#cutlass.cute.CopyAtom)

### cutlass.cute.make_cotiled_copy(atom: [CopyAtom](#cutlass.cute.CopyAtom), atom_layout_tv: [Layout](#cutlass.cute.Layout), data_layout: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [TiledCopy](#cutlass.cute.TiledCopy)

Produce a TiledCopy from thread and value offset maps.
The TV Layout maps threads and values to the codomain of the data_layout.
It is verified that the intended codomain is valid within data_layout.
Useful when threads and values don’t care about owning specific coordinates, but
care more about the vector-width and offsets between them.

## Parameters

atom : copy atom, e.g. simt_copy and simt_async_copy, tgen05.st, etc.
atom_layout_tv : (tid, vid) -> data addr
data_layout : data coord -> data addr
loc     : source location for mlir (optional)
ip      : insertion point (optional)

## Returns

tiled_copy
: A tuple of A tiled copy and atom

### cutlass.cute.make_fragment(layout_or_shape: [Layout](#cutlass.cute.Layout) | int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], dtype: Type[Numeric], , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.make_fragment_like(src: [Tensor](#cutlass.cute.Tensor), dtype: Type[Numeric] | None, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.make_fragment_like(src: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.make_fragment_like(src: [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.make_identity_layout(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

Create an identity layout with the given shape.

An identity layout maps logical coordinates directly to themselves without any transformation.
This is equivalent to a layout with stride (1@0,1@1,…,1@(N-1)).

* **Parameters:**
  * **shape** (*Shape*) – The shape of the layout
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  A new identity Layout object with the specified shape
* **Return type:**
  [Layout](#cutlass.cute.Layout)

**Examples:**

```python
# Create a 2D identity layout with shape (4,4)
layout = make_identity_layout((4,4))     # stride=(1@0,1@1)

# Create a 3D identity layout
layout = make_identity_layout((32,16,8)) # stride=(1@0,1@1,1@2)
```

Note:
: - An identity layout is a special case where each coordinate maps to itself
  - Useful for direct coordinate mapping without any transformation

### cutlass.cute.make_identity_tensor(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Creates an identity tensor with the given shape.

An identity tensor maps each coordinate to itself, effectively creating a counting
sequence within the shape’s bounds. This is useful for generating coordinate indices
or creating reference tensors for layout transformations.

* **Parameters:**
  * **shape** (*Shape*) – The shape defining the tensor’s dimensions. Can be a simple integer
    sequence or a hierarchical structure ((m,n),(p,q))
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR operation tracking, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for MLIR operation, defaults to None
* **Returns:**
  A tensor that maps each coordinate to itself
* **Return type:**
  [Tensor](#cutlass.cute.Tensor)

**Examples:**

```python
# Create a simple 1D coord tensor
tensor = make_identity_tensor(6)  # [0,1,2,3,4,5]

# Create a 2D coord tensor
tensor = make_identity_tensor((3,2))  # [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# Create hierarchical coord tensor
tensor = make_identity_tensor(((2,1),3))
# [((0,0),0),((1,0),0),((0,0),1),((1,0),1),((0,0),2),((1,0),2)]
```

Notes:
: - The shape parameter follows CuTe’s IntTuple concept
  - Coordinates are ordered colexicographically
  - Useful for generating reference coordinates in layout transformations

### cutlass.cute.make_layout(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , stride: int | Integer | ScaledBasis | Tuple[int | Integer | ScaledBasis | Tuple[Stride, ...], ...] | None = None, loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

Create a CuTe Layout object from shape and optional stride information.

A Layout in CuTe represents the mapping between logical and physical coordinates of a tensor.
This function creates a Layout object that defines how tensor elements are arranged in memory.

* **Parameters:**
  * **shape** (*Shape*) – Shape of the layout defining the size of each mode
  * **stride** (*Union* *[**Stride* *,* *None* *]*) – Optional stride values for each mode, defaults to None
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  A new Layout object with the specified shape and stride
* **Return type:**
  [Layout](#cutlass.cute.Layout)

**Examples:**

```python
# Create a 2D compact left-most layout with shape (4,4)
layout = make_layout((4,4))                     # compact left-most layout

# Create a left-most layout with custom strides
layout = make_layout((4,4), stride=(1,4))       # left-most layout with strides (1,4)

# Create a layout for a 3D tensor
layout = make_layout((32,16,8))                 # left-most layout

# Create a layout with custom strides
layout = make_layout((2,2,2), stride=(4,1,2))   # layout with strides (4,1,2)
```

Note:
: - If stride is not provided, a default compact left-most stride is computed based on the shape
  - The resulting layout maps logical coordinates to physical memory locations
  - The layout object can be used for tensor creation and memory access patterns
  - Strides can be used to implement:
    \* Row-major vs column-major layouts
    \* Padding and alignment
    \* Blocked/tiled memory arrangements
    \* Interleaved data formats
  - Stride is keyword only argument to improve readability, e.g.
    \* make_layout((3,4), (1,4)) can be confusing with make_layout(((3,4), (1,4)))
    \* make_layout((3,4), stride=(1,4)) is more readable

### cutlass.cute.make_layout_image_mask(lay: [Layout](#cutlass.cute.Layout), coord: int | Integer | None | Tuple[int | Integer | None | Tuple[Coord, ...], ...], mode: int, , loc=None, ip=None) → Int16

Makes a 16-bit integer mask of the image of a layout sliced at a given mode
and accounting for the offset given by the input coordinate for the other modes.

### cutlass.cute.make_layout_like(input: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.make_layout_tv(thr_layout: [Layout](#cutlass.cute.Layout), val_layout: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → Tuple[int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], [Layout](#cutlass.cute.Layout)]

Create a thread-value layout by repeating the val_layout over the thr_layout.

This function creates a thread-value layout that maps between `(thread_idx, value_idx)`
coordinates and logical `(M,N)` coordinates. The thread and value layouts must be compact to ensure
proper partitioning.

This implements the thread-value partitioning pattern where data is partitioned
across threads and values within each thread.

* **Parameters:**
  * **thr_layout** ([*Layout*](#cutlass.cute.Layout)) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
  * **val_layout** ([*Layout*](#cutlass.cute.Layout)) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs within each thread
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tuple containing `tiler_mn` and `layout_tv`
* **Return type:**
  Tuple[Shape, [Layout](#cutlass.cute.Layout)]

where:
: * `tiler_mn` is tiler and `shape(tiler_mn)` is compatible with `shape(zipped_divide(x, tiler_mn))[0]`
  * `layout_tv`: Thread-value layout mapping (thread_idx, value_idx) -> (M,N)

**Example:**

The below code creates a TV Layout that maps thread/value coordinates to the logical coordinates in a `(4,6)` tensor:
: - *Tiler MN*: `(4,6)`
  - *TV Layout*: `((3,2),(2,2)):((8,2),(4,1))`

```python
thr_layout = cute.make_layout((2, 3), stride=(3, 1))
val_layout = cute.make_layout((2, 2), stride=(2, 1))
tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
```

#### TV Layout

|    | 0          | 1          | 2          | 3          | 4          | 5          |
|----|------------|------------|------------|------------|------------|------------|
|  0 | T0,<br/>V0 | T0,<br/>V1 | T1,<br/>V0 | T1,<br/>V1 | T2,<br/>V0 | T2,<br/>V1 |
|  1 | T0,<br/>V2 | T0,<br/>V3 | T1,<br/>V2 | T1,<br/>V3 | T2,<br/>V2 | T2,<br/>V3 |
|  2 | T3,<br/>V0 | T3,<br/>V1 | T4,<br/>V0 | T4,<br/>V1 | T5,<br/>V0 | T5,<br/>V1 |
|  3 | T3,<br/>V2 | T3,<br/>V3 | T4,<br/>V2 | T4,<br/>V3 | T5,<br/>V2 | T5,<br/>V3 |

### cutlass.cute.make_mma_atom(op: MmaOp, , loc=None, ip=None, \*\*kwargs) → [MmaAtom](#cutlass.cute.MmaAtom)

Makes an MMA Atom from an MMA Operation.

This function creates an MMA Atom from a given MMA Operation. Arbitrary kw arguments can be
provided for Op-specific additional parameters. They are not used as of today.

* **Parameters:**
  **op** (*MmaOp*) – The MMA Operation to construct an Atom for
* **Returns:**
  The MMA Atom
* **Return type:**
  [MmaAtom](#cutlass.cute.MmaAtom)

### cutlass.cute.make_ordered_layout(shape: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], order: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

Create a layout with a specific ordering of dimensions.

This function creates a layout where the dimensions are ordered according to the
specified order parameter, allowing for custom dimension ordering in the layout.

* **Parameters:**
  * **shape** (*Shape*) – The shape of the layout
  * **order** (*Shape*) – The ordering of dimensions
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  A new Layout object with the specified shape and dimension ordering
* **Return type:**
  [Layout](#cutlass.cute.Layout)

**Examples:**

```python
# Create a row-major layout
layout = make_ordered_layout((4,4), order=(1,0))

# Create a column-major layout
layout = make_ordered_layout((4,4), order=(0,1))         # stride=(1,4)

# Create a layout with custom dimension ordering for a 3D tensor
layout = make_ordered_layout((32,16,8), order=(2,0,1))   # stride=(128,1,16)
```

Note:
: - The order parameter specifies the ordering of dimensions from fastest-varying to slowest-varying
  - For a 2D tensor, (0,1) creates a column-major layout, while (1,0) creates a row-major layout
  - The length of order must match the rank of the shape

### cutlass.cute.make_ptr(dtype: Type[Numeric] | None, value, mem_space: [AddressSpace](#cutlass.cute.AddressSpace) = AddressSpace.generic, , assumed_align=None, loc=None, ip=None) → Pointer

### cutlass.cute.make_rmem_tensor(layout_or_shape: [Layout](#cutlass.cute.Layout) | int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], dtype: Type[Numeric], , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Creates a tensor in register memory with the specified layout/shape and data type.

This function allocates a tensor in register memory (rmem) usually on stack with
either a provided layout or creates a new layout from the given shape. The tensor
will have elements of the specified numeric data type.

* **Parameters:**
  * **layout_or_shape** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* *Shape* *]*) – Either a Layout object defining the tensor’s memory organization,
    or a Shape defining its dimensions
  * **dtype** (*Type* *[**Numeric* *]*) – The data type for tensor elements (must be a Numeric type)
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR operation tracking, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for MLIR operation, defaults to None
* **Returns:**
  A tensor allocated in register memory
* **Return type:**
  [Tensor](#cutlass.cute.Tensor)

**Examples:**

```python
# Create rmem tensor with explicit layout
layout = make_layout((128, 32))
tensor = make_rmem_tensor(layout, cutlass.Float16)

# Create rmem tensor directly from shape
tensor = make_rmem_tensor((64, 64), cutlass.Float32)
```

Notes:
: - Uses 32-byte alignment to support .128 load/store operations
  - Boolean types are stored as 8-bit integers
  - Handles both direct shapes and Layout objects

### cutlass.cute.make_rmem_tensor_like(src: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor) | [TensorSSA](#cutlass.cute.TensorSSA), dtype: Type[Numeric] | None = None, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Creates a tensor in register memory with the same shape as the input layout but
: compact col-major strides. This is equivalent to calling make_rmem_tensor(make_layout_like(tensor)).

This function allocates a tensor in register memory (rmem) usually on stack with
with the compact layout like the source. The tensor will have elements of the
specified numeric data type or the same as the source.

* **Parameters:**
  * **src** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – The source layout or tensor whose shape will be matched
  * **dtype** (*Type* *[**Numeric* *]* *,* *optional*) – The element type for the fragment tensor, defaults to None
  * **loc** (*Location* *,* *optional*) – Source location for MLIR operations, defaults to None
  * **ip** (*InsertionPoint* *,* *optional*) – Insertion point for MLIR operations, defaults to None
* **Returns:**
  A new layout or fragment tensor with matching shape
* **Return type:**
  Union[[Layout](#cutlass.cute.Layout), [Tensor](#cutlass.cute.Tensor)]

**Examples:**

Creating a rmem tensor from a tensor:

```python
smem_tensor = cute.make_tensor(smem_ptr, layout)
rmem_tensor = cute.make_rmem_tensor_like(smem_tensor, cutlass.Float32)
# frag_tensor will be a register-backed tensor with the same shape
```

Creating a fragment with a different element type:

```python
tensor = cute.make_tensor(gmem_ptr, layout)
rmem_bool_tensor = cute.make_rmem_tensor_like(tensor, cutlass.Boolean)
# bool_frag will be a register-backed tensor with Boolean elements
```

**Notes**

- When used with a Tensor, if a type is provided, it will create a new
  fragment tensor with that element type.
- For layouts with ScaledBasis strides, the function creates a fragment
  from the shape only.
- This function is commonly used in GEMM and other tensor operations to
  create register storage for intermediate results.

### cutlass.cute.make_swizzle(b, m, s, , loc=None, ip=None)

### cutlass.cute.make_tensor(iterator, layout: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

Creates a tensor by composing an engine (iterator/pointer) with a layout.

A tensor is defined as T = E ∘ L, where E is an engine (array, pointer, or counting iterator)
and L is a layout that maps logical coordinates to physical offsets. The tensor
evaluates coordinates by applying the layout mapping and dereferencing the engine
at the resulting offset.

* **Parameters:**
  * **iterator** (*Union* *[**Pointer* *,* *IntTuple* *,* *ir.Value* *]*) – Engine component that provides data access capabilities. Can be:
    - A pointer (Pointer type)
    - An integer or integer tuple for coordinate tensors
    - A shared memory descriptor (SmemDescType)
  * **layout** (*Union* *[**Shape* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *]*) – Layout component that defines the mapping from logical coordinates to
    physical offsets. Can be:
    - A shape tuple that will be converted to a layout
    - A Layout object
    - A ComposedLayout object (must be a normal layout)
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR operation tracking, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for MLIR operation, defaults to None
* **Returns:**
  A tensor object representing the composition E ∘ L
* **Return type:**
  [Tensor](#cutlass.cute.Tensor)
* **Raises:**
  * **TypeError** – If iterator type is not a supported type
  * **ValueError** – If layout is a composed layout with customized inner functions

**Examples:**

```python
# Create a tensor with row-major layout from a pointer
ptr = make_ptr(Float32, base_ptr, AddressSpace.gmem)
layout = make_layout((64, 128), stride=(128, 1))
tensor = make_tensor(ptr, layout)

# Create a tensor with hierarchical layout in shared memory
smem_ptr = make_ptr(Float16, base_ptr, AddressSpace.smem)
layout = make_layout(((128, 8), (1, 4, 1)), stride=((32, 1), (0, 8, 4096)))
tensor = make_tensor(smem_ptr, layout)

# Create a coordinate tensor
layout = make_layout(2, stride=16 * E(0))
tensor = make_tensor(5, layout)  # coordinate tensor with iterator starting at 5
```

Notes:
: - The engine (iterator) must support random access operations
  - Common engine types include raw pointers, arrays, and random-access iterators
  - The layout defines both the shape (logical dimensions) and stride (physical mapping)
  - Supports both direct coordinate evaluation T(c) and partial evaluation (slicing)
  - ComposedLayouts must be “normal” layouts (no inner functions)
  - For coordinate tensors, the iterator is converted to a counting sequence

### cutlass.cute.make_tiled_copy(atom, layout_tv, tiler_mn, , loc=None, ip=None)

Create a tiled type given a TV partitioner and tiler.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom, e.g. smit_copy and simt_async_copy, tma_load, etc.
  * **layout_tv** ([*Layout*](#cutlass.cute.Layout)) – Thread-value layout
  * **tiler_mn** (*Tiler*) – Tile size
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_A(atom, tiled_mma, , loc=None, ip=None)

Create a tiled copy out of the copy_atom that matches the A-Layout of tiled_mma.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **tiled_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_B(atom, tiled_mma, , loc=None, ip=None)

Create a tiled copy out of the copy_atom that matches the B-Layout of tiled_mma.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **tiled_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_C(atom, tiled_mma, , loc=None, ip=None)

Create a tiled copy out of the copy_atom that matches the C-Layout of tiled_mma.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **tiled_mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_C_atom(atom: [CopyAtom](#cutlass.cute.CopyAtom), mma: [TiledMma](#cutlass.cute.TiledMma), , loc=None, ip=None)

Create the smallest tiled copy that can retile LayoutC_TV for use with pipelined epilogues with subtiled stores.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **mma** ([*TiledMma*](#cutlass.cute.TiledMma)) – Tiled MMA
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)
* **Raises:**
  **ValueError** – If the number value of CopyAtom’s source layout is greater than the size of TiledMma’s LayoutC_TV

### cutlass.cute.make_tiled_copy_D(atom, tiled_copy, , loc=None, ip=None)

Create a tiled copy out of the copy_atom that matches the Dst-Layout of tiled_copy.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **tiled_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy)) – Tiled copy
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_S(atom, tiled_copy, , loc=None, ip=None)

Create a tiled copy out of the copy_atom that matches the Src-Layout of tiled_copy.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **tiled_copy** ([*TiledCopy*](#cutlass.cute.TiledCopy)) – Tiled copy
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_copy_tv(atom: [CopyAtom](#cutlass.cute.CopyAtom), thr_layout: [Layout](#cutlass.cute.Layout), val_layout: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [TiledCopy](#cutlass.cute.TiledCopy)

Create a tiled copy given separate thread and value layouts.

A TV partitioner is inferred based on the input layouts. The input thread layout
must be compact.

* **Parameters:**
  * **atom** ([*CopyAtom*](#cutlass.cute.CopyAtom)) – Copy atom
  * **thr_layout** ([*Layout*](#cutlass.cute.Layout)) – Layout mapping from `(TileM,TileN)` coordinates to thread IDs (must be compact)
  * **val_layout** ([*Layout*](#cutlass.cute.Layout)) – Layout mapping from `(ValueM,ValueN)` coordinates to value IDs
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None
* **Returns:**
  A tiled copy for the partitioner
* **Return type:**
  [TiledCopy](#cutlass.cute.TiledCopy)

### cutlass.cute.make_tiled_mma(op_or_atom: Op | [MmaAtom](#cutlass.cute.MmaAtom), atom_layout_mnk=(1, 1, 1), permutation_mnk=None, , loc=None, ip=None, \*\*kwargs) → [TiledMma](#cutlass.cute.TiledMma)

Makes a tiled MMA from an MMA Operation or an MMA Atom.

* **Parameters:**
  * **op_or_atom** (*Union* *[**Op* *,* [*MmaAtom*](#cutlass.cute.MmaAtom) *]*) – The MMA Operation or Atom
  * **atom_layout_mnk** ([*Layout*](#cutlass.cute.Layout)) – A Layout describing the tiling of Atom across threads
  * **permutation_mnk** (*Tiler*) – A permutation Tiler describing the tiling of Atom across values including any permutation of such tiling
* **Returns:**
  The resulting tiled MMA
* **Return type:**
  [TiledMma](#cutlass.cute.TiledMma)

### cutlass.cute.max_common_layout(a: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), b: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.max_common_vector(a: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), b: [Layout](#cutlass.cute.Layout) | [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → int

### cutlass.cute.mma_atom_call(atom: [MmaAtom](#cutlass.cute.MmaAtom), d: [Tensor](#cutlass.cute.Tensor), a: [Tensor](#cutlass.cute.Tensor), b: [Tensor](#cutlass.cute.Tensor), c: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None, \*\*kwargs) → None

Execute a single MMA atom operation.

The mma_atom_call operation executes an MMA atom with the given operands.
This performs a matrix multiplication and accumulation operation:
D = A \* B + C

Note: The tensors ‘d’, ‘a’, ‘b’, and ‘c’ must only have a single fragment.

* **Parameters:**
  * **atom** ([*MmaAtom*](#cutlass.cute.MmaAtom)) – The MMA atom to execute
  * **d** ([*Tensor*](#cutlass.cute.Tensor)) – Destination tensor (output accumulator)
  * **a** ([*Tensor*](#cutlass.cute.Tensor)) – First source tensor (matrix A)
  * **b** ([*Tensor*](#cutlass.cute.Tensor)) – Second source tensor (matrix B)
  * **c** ([*Tensor*](#cutlass.cute.Tensor)) – Third source tensor (input accumulator C)
  * **loc** (*Optional* *[**Location* *]* *,* *optional*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]* *,* *optional*) – Insertion point, defaults to None

Examples:

```python
# Call an MMA atom operation
cute.mma_atom_call(mma_atom, d_tensor, a_tensor, b_tensor, c_tensor)
```

### cutlass.cute.ones_like(a, dtype=None, , loc=None, ip=None)

Return a TensorSSA of ones with the same shape and type as a given array.

* **Parameters:**
  * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
  * **dtype** (*Type* *[**Numeric* *]* *,* *optional*) – Overrides the data type of the result, defaults to None
* **Returns:**
  Tensor of ones with the same shape and type (unless overridden) as a.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.prefetch(atom: [CopyAtom](#cutlass.cute.CopyAtom), src: [Tensor](#cutlass.cute.Tensor), , loc=None, ip=None) → None

The Prefetch algorithm.

The “prefetch” expects source tensors to be partitioned according to the provided Copy Atom.
Prefetch is used for loading tensors from global memory to L2.

Prefetch accepts Copy Atom but not all are allowed. Currently, only supports TMA prefetch.

```python
cute.prefetch(tma_prefetch, src)
```

For Copy Atoms that require single-threaded execution, the copy op automatically handles thread
election internally. Manual thread selection is not required in such cases.

### cutlass.cute.prepend(input: [Layout](#cutlass.cute.Layout), elem: [Layout](#cutlass.cute.Layout), up_to_rank=None, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.prepend(input: [ComposedLayout](#cutlass.cute.ComposedLayout), elem: [Layout](#cutlass.cute.Layout), up_to_rank=None, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.prepend(input: XTuple, elem: XTuple, up_to_rank=None, , loc=None, ip=None) → XTuple

Extend input to rank up_to_rank by prepending elem in front of input.

This function extends the input object by prepending elements to reach a desired rank.
It supports various CuTe types including shapes, layouts, tensors etc.

* **Parameters:**
  * **input** (*Union* *[**Shape* *,* *Stride* *,* *Coord* *,* *IntTuple* *,* *Tile* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *]*) – Source to be prepended to
  * **elem** (*Union* *[**Shape* *,* *Stride* *,* *Coord* *,* *IntTuple* *,* *Tile* *,* [*Layout*](#cutlass.cute.Layout) *]*) – Element to prepend to input
  * **up_to_rank** (*Union* *[**None* *,* *int* *]* *,* *optional*) – The target rank after extension, defaults to None
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point, defaults to None
* **Returns:**
  The extended result with prepended elements
* **Return type:**
  Union[Shape, Stride, Coord, IntTuple, Tile, [Layout](#cutlass.cute.Layout), [ComposedLayout](#cutlass.cute.ComposedLayout), [Tensor](#cutlass.cute.Tensor)]
* **Raises:**
  * **ValueError** – If up_to_rank is less than input’s current rank
  * **TypeError** – If input or elem has unsupported type

**Examples:**

```python
# Prepend to a Shape
shape = (4,4)
prepend(shape, 2)                   # Returns (2,4,4)

# Prepend to a Layout
layout = make_layout((8,8))
prepend(layout, make_layout((2,)))  # Returns (2,8,8):(1,1,8)

# Prepend with target rank
coord = (1,1)
prepend(coord, 0, up_to_rank=4)     # Returns (0,0,1,1)
```

### cutlass.cute.prepend_ones(t: [Tensor](#cutlass.cute.Tensor), up_to_rank: None | int = None, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.pretty_str(arg) → str

Constructs a concise readable pretty string.

### cutlass.cute.print_tensor(tensor: [Tensor](#cutlass.cute.Tensor) | [TensorSSA](#cutlass.cute.TensorSSA), , verbose: bool = False, loc=None, ip=None)

Print content of the tensor in human readable format.

Outputs the tensor data in a structured format showing both metadata
and the actual data values. The output includes tensor type information,
layout details, and a formatted array representation of the values.

* **Parameters:**
  * **tensor** ([*Tensor*](#cutlass.cute.Tensor)) – The tensor to print
  * **verbose** (*bool*) – If True, includes additional debug information in the output
  * **loc** (*source location* *,* *optional*) – Source location where it’s called, defaults to None
  * **ip** (*insertion pointer* *,* *optional*) – Insertion pointer for IR generation, defaults to None
* **Raises:**
  **NotImplementedError** – If the tensor type doesn’t support trivial dereferencing

**Example output:**

```text
tensor(raw_ptr<@..., Float32, generic, align(4)> o (8,5):(5,1), data=
       [[-0.4326, -0.5434,  0.1238,  0.7132,  0.8042],
        [-0.8462,  0.9871,  0.4389,  0.7298,  0.6948],
        [ 0.3426,  0.5856,  0.1541,  0.2923,  0.6976],
        [-0.1649,  0.8811,  0.1788,  0.1404,  0.2568],
        [-0.2944,  0.8593,  0.4171,  0.8998,  0.1766],
        [ 0.8814,  0.7919,  0.7390,  0.4566,  0.1576],
        [ 0.9159,  0.7577,  0.6918,  0.0754,  0.0591],
        [ 0.6551,  0.1626,  0.1189,  0.0292,  0.8655]])
```

### cutlass.cute.printf(\*args, loc=None, ip=None) → None

Print one or more values with optional formatting.

This function provides printf-style formatted printing capabilities. It can print values directly
or format them using C-style format strings. The function supports printing various types including
layouts, numeric values, tensors, and other CuTe objects.

The function accepts either:
1. A list of values to print directly
2. A format string followed by values to format

* **Parameters:**
  * **args** (*Any*) – Variable length argument list containing either:
    - One or more values to print directly
    - A format string followed by values to format
  * **loc** (*Optional* *[**Location* *]*) – Source location information for debugging, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for code generation, defaults to None
* **Raises:**
  * **ValueError** – If no arguments are provided
  * **TypeError** – If an unsupported argument type is passed

**Examples:**

Direct printing of values:

```python
a = cute.make_layout(shape=(10, 10), stride=(10, 1))
b = cutlass.Float32(1.234)
cute.printf(a, b)  # Prints values directly
```

Formatted printing:

```python
# Using format string with generic format specifiers
cute.printf("a={}, b={}", a, b)

# Using format string with C-style format specifiers
cute.printf("a={}, b=%.2f", a, b)
```

### cutlass.cute.product(a: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...] | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None)

### cutlass.cute.product_each(a: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...], , loc=None, ip=None) → int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]

### cutlass.cute.product_like(a: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...], target_profile: Any | Tuple[Any | Tuple[XTuple, ...], ...], , loc=None, ip=None) → int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]

Return product of the given IntTuple or Shape at leaves of target_profile.

This function computes products according to the structure defined by target_profile.

* **Parameters:**
  * **a** (*IntTuple* *or* *Shape*) – The input tuple or shape
  * **target_profile** (*XTuple*) – The profile that guides how products are computed
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The resulting tuple with products computed according to target_profile
* **Return type:**
  IntTuple or Shape
* **Raises:**
  * **TypeError** – If inputs have incompatible types
  * **ValueError** – If inputs have incompatible shapes

### cutlass.cute.raked_product(block: [Layout](#cutlass.cute.Layout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.raked_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.rank(a: Any | Tuple[Any | Tuple[XTuple, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout), mode: List[int] = []) → int

Returns the rank (dimensionality) of a tuple, layout, or tensor.

The rank of a tuple is its length. For layouts and tensors, the rank is
determined by the rank of their shape. For non-tuple values (e.g., integers),
the rank is considered 1 for convenience.

* **Parameters:**
  **a** (*Union* *[**XTuple* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* [*Tensor*](#cutlass.cute.Tensor) *,* *Any* *]*) – The object whose rank is to be determined
* **Returns:**
  The rank of the input object
* **Return type:**
  int

This function is used in layout algebra to determine the dimensionality
of tensors and layouts for operations like slicing and evaluation.

### cutlass.cute.recast_layout(new_type_bits: int, old_type_bits: int, src_layout: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout), , loc=None, ip=None)

Recast a layout from one data type to another.

* **Parameters:**
  * **new_type_bits** (*int*) – The new data type bits
  * **old_type_bits** (*int*) – The old data type bits
  * **src_layout** (*Union* *[*[*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *]*) – The layout to recast
  * **loc** (*optional*) – Optional location information for IR diagnostics.
  * **ip** (*optional*) – Optional instruction pointer or context for underlying IR functions.
* **Returns:**
  The recast layout
* **Return type:**
  [Layout](#cutlass.cute.Layout) or [ComposedLayout](#cutlass.cute.ComposedLayout)

**Example:**

```python
import cutlass.cute as cute
@cute.jit
def foo():
    # Create a layout
    L = cute.make_layout((2, 3, 4))
    # Recast the layout to a different data type
    L_recast = cute.recast_layout(16, 8, L)
    print(L_recast)
foo()  # Expected output: (2, 3, 4)
```

### cutlass.cute.recast_ptr(ptr: Pointer, swizzle_=None, dtype: Type[Numeric] | None = None, loc=None, ip=None) → Pointer

### cutlass.cute.recast_tensor(src: [Tensor](#cutlass.cute.Tensor), dtype: Type[Numeric], swizzle_=None, , loc=None, ip=None)

### cutlass.cute.register_jit_arg_adapter(\*dargs, \*\*dkwargs)

Register a JIT argument adapter callable

This can be used as a decorator on any callable like:

@register_jit_arg_adapter(my_py_type)
def my_adapter_for_my_py_type(arg):

> …

@register_jit_arg_adapter(my_py_type)
class MyAdapterForMyPythonType:

> …

The adapters are registered per type. If a type is already registerd, an error will be raised.

### cutlass.cute.repeat(x, n)

Creates an object by repeating x n times.

This function creates an object by repeating the input value x n times.
If n=1, returns x directly, otherwise returns a tuple of x repeated n times.

* **Parameters:**
  * **x** (*Any*) – The value to repeat
  * **n** (*int*) – Number of times to repeat x
* **Returns:**
  x if n=1, otherwise a tuple containing x repeated n times
* **Return type:**
  Union[Any, tuple]
* **Raises:**
  **ValueError** – If n is less than 1

**Examples:**

```python
repeat(1, 1)     # Returns 1
repeat(1, 3)     # Returns (1, 1, 1)
repeat(None, 4)  # Returns (None, None, None, None)
```

### cutlass.cute.repeat_as_tuple(x, n) → tuple

Creates a tuple with x repeated n times.

This function creates a tuple by repeating the input value x n times.

* **Parameters:**
  * **x** (*Any*) – The value to repeat
  * **n** (*int*) – Number of times to repeat x
* **Returns:**
  A tuple containing x repeated n times
* **Return type:**
  tuple

**Examples:**

```python
repeat_as_tuple(1, 1)     # Returns (1,)
repeat_as_tuple(1, 3)     # Returns (1, 1, 1)
repeat_as_tuple(None, 4)  # Returns (None, None, None, None)
```

### cutlass.cute.repeat_like(x, target)

Creates an object congruent to target and filled with x.

This function recursively creates a nested tuple structure that matches the structure
of the target, with each leaf node filled with the value x.

* **Parameters:**
  * **x** (*Any*) – The value to fill the resulting structure with
  * **target** (*Union* *[**tuple* *,* *Any* *]*) – The structure to mimic
* **Returns:**
  A structure matching target but filled with x
* **Return type:**
  Union[tuple, Any]

**Examples:**

```python
repeat_like(0, (1, 2, 3))      # Returns (0, 0, 0)
repeat_like(1, ((1, 2), 3))    # Returns ((1, 1), 1)
repeat_like(2, 5)              # Returns 2
```

### cutlass.cute.right_inverse(input: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.round_up(a: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...], b: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]) → int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...]

Rounds up elements of a using elements of b.

### cutlass.cute.rsqrt(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise reciprocal square root of the input tensor.

Computes 1/√x element-wise.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the reciprocal square root of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = rsqrt(y)  # Compute 1/√x
```

### cutlass.cute.select(input: [Layout](#cutlass.cute.Layout), mode, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.select(input: [ComposedLayout](#cutlass.cute.ComposedLayout), mode, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.select(input: XTuple, mode, , loc=None, ip=None) → XTuple

Select modes from input.

* **Parameters:**
  * **input** ([*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *,* *tuple*) – Input to select from
  * **mode** (*List* *[**int* *]*) – Indices specifying which dimensions or elements to select
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  A new instance with selected dimensions/elements
* **Return type:**
  [Layout](#cutlass.cute.Layout), [ComposedLayout](#cutlass.cute.ComposedLayout), tuple
* **Raises:**
  * **ValueError** – If any index in mode is out of range
  * **TypeError** – If the input type is invalid

**Examples:**

```python
# Select specific dimensions from a layout
layout = make_layout((4, 8, 16), stride=(32, 4, 1))
selected = select(layout, mode=[0, 2])  # Select mode 0 and mode 2
# Result: (4, 16):(32, 1)

# Select elements from a tuple
t = (1, 2, 3, 4, 5)
selected = select(t, mode=[0, 2, 4])  # Select mode 0, mode 2, and mode 4
# Result: (1, 3, 5)
```

### cutlass.cute.shape(input: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...] | [Tensor](#cutlass.cute.Tensor) | [Layout](#cutlass.cute.Layout) | None | Tuple[int | Integer | None | [Layout](#cutlass.cute.Layout) | Tuple[Tile, ...], ...], , mode=None, loc=None, ip=None) → int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]

Returns the shape of a tensor, layout or tiler.

For shapes, this function is identical to get.

This function extracts the shape information from the input object. For tensors and layouts,
it returns their internal shape property. For tilers, it unpacks the shape from the tile
representation.

* **Parameters:**
  * **input** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* [*Layout*](#cutlass.cute.Layout) *,* *Tile* *]*) – The object to extract shape from
  * **mode** (*Optional* *[**int* *]*) – Optional mode selector to extract specific dimensions from the shape
  * **loc** (*Optional* *[**Location* *]*) – Source location for MLIR operation tracking
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for MLIR operation
* **Returns:**
  The shape of the input object, optionally filtered by mode
* **Return type:**
  Shape

**Example:**

```python
# Get shape of a layout
l0 = cute.make_layout((2, 3, 4))
s0 = cute.shape(l0)  # => (2, 3, 4)

# Get shape of a hierarchical tiler
l1 = cute.make_layout(1)
s1 = cute.shape((l0, l1))  # => ((2, 3, 4), 1)

# Get specific mode from a shape
s2 = cute.shape(l0, mode=0)  # => 2
```

### cutlass.cute.shape_div(lhs: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], rhs: int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...], , loc=None, ip=None) → int | Integer | Tuple[int | Integer | Tuple[Shape, ...], ...]

Perform element-wise division of shapes.

This function performs element-wise division between two shapes.

* **Parameters:**
  * **lhs** (*Shape*) – Left-hand side shape
  * **rhs** (*Shape*) – Right-hand side shape
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The result of element-wise division
* **Return type:**
  Shape

### cutlass.cute.sin(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise sine of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor (in radians)
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the sine of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = sin(y)  # Compute sine
```

### cutlass.cute.size(a: int | Integer | Tuple[int | Integer | Tuple[IntTuple, ...], ...] | Tuple[int | Integer | Tuple[Shape, ...], ...] | [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | [Tensor](#cutlass.cute.Tensor), mode: List[int] = [], , loc=None, ip=None) → int | Integer

Return size of domain of layout or tensor.

Computes the size (number of elements) in the domain of a layout or tensor.
For layouts, this corresponds to the shape of the coordinate space.
See [https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/01_layout.md)
for more details on layout domains.

* **Parameters:**
  * **a** (*IntTuple* *,* *Shape* *,* [*Layout*](#cutlass.cute.Layout) *,* [*ComposedLayout*](#cutlass.cute.ComposedLayout) *or* [*Tensor*](#cutlass.cute.Tensor)) – The input object whose size to compute
  * **mode** (*list* *of* *int* *,* *optional*) – List of mode(s) for size calculation. If empty, computes total size, defaults to []
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  Static size of layout or tensor if static, otherwise a Value
* **Return type:**
  int or Value
* **Raises:**
  **ValueError** – If mode contains non-integer elements

### cutlass.cute.size_in_bytes(dtype: Type[Numeric], layout: [Layout](#cutlass.cute.Layout) | [ComposedLayout](#cutlass.cute.ComposedLayout) | None, , loc=None, ip=None) → int | Integer

Calculate the size in bytes based on its data type and layout. The result is rounded up to the nearest byte.

* **Parameters:**
  * **dtype** (*Type* *[**Numeric* *]*) – The DSL numeric data type
  * **layout** ([*Layout*](#cutlass.cute.Layout) *,* *optional*) – The layout of the elements. If None, the function returns 0
  * **loc** (*optional*) – Location information for diagnostics, defaults to None
  * **ip** (*optional*) – Instruction pointer for diagnostics, defaults to None
* **Returns:**
  The total size in bytes. Returns 0 if the layout is None
* **Return type:**
  int

### cutlass.cute.slice_(src: [Layout](#cutlass.cute.Layout), coord: Coord, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.slice_(src: \_ComposedLayout, coord: Coord, , loc=None, ip=None) → \_ComposedLayout

### cutlass.cute.slice_(src: [Tensor](#cutlass.cute.Tensor), coord: Coord, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.slice_(src: XTuple, coord: Coord, , loc=None, ip=None) → XTuple

Perform a slice operation on a source object using the given coordinate.

This function implements CuTe’s slicing operation which extracts a subset of elements
from a source object (tensor, layout, etc.) based on a coordinate pattern. The slice
operation preserves the structure of the source while selecting specific elements.

* **Parameters:**
  * **src** (*Union* *[*[*Tensor*](#cutlass.cute.Tensor) *,* [*Layout*](#cutlass.cute.Layout) *,* *IntTuple* *,* *Value* *]*) – Source object to be sliced (tensor, layout, tuple, etc.)
  * **coord** (*Coord*) – Coordinate pattern specifying which elements to select
  * **loc** (*Optional* *[**Location* *]*) – Source location information, defaults to None
  * **ip** (*Optional* *[**InsertionPoint* *]*) – Insertion point for IR generation, defaults to None
* **Returns:**
  A new object containing the sliced elements
* **Return type:**
  Union[[Tensor](#cutlass.cute.Tensor), [Layout](#cutlass.cute.Layout), IntTuple, tuple]
* **Raises:**
  **ValueError** – If the coordinate pattern is incompatible with source

**Examples:**

```python
# Layout slicing
layout = make_layout((4,4))

# Select 1st index of first mode and keep all elements in second mode
sub_layout = slice_(layout, (1, None))
```

```python
# Basic tensor slicing
tensor = make_tensor(...)           # Create a 2D tensor

# Select 1st index of first mode and keep all elements in second mode
sliced = slice_(tensor, (1, None))
```

```python
# Select 2nd index of second mode and keep all elements in first mode
sliced = slice_(tensor, (None, 2))
```

Note:
: - None represents keeping all elements in that mode
  - Slicing preserves the layout/structure of the original object
  - Can be used for:
    \* Extracting sub-tensors/sub-layouts
    \* Creating views into data
    \* Selecting specific patterns of elements

### cutlass.cute.slice_and_offset(coord, src, , loc=None, ip=None)

### cutlass.cute.sqrt(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise square root of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the square root of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = sqrt(y)  # Compute square root
```

### cutlass.cute.static(value, , loc=None, ip=None)

### *class* cutlass.cute.struct(cls)

Bases: `object`

Decorator to abstract C structure in Python DSL.

**Usage:**

```python
# Supports base_dsl scalar int/float elements, array and nested struct:
@cute.struct
class complex:
    real : cutlass.Float32
    imag : cutlass.Float32


@cute.struct
class StorageA:
    mbarA : cute.struct.MemRange[cutlass.Int64, stage]
    compA : complex
    intA : cutlass.Int16


# Supports alignment for its elements:
@cute.struct
class StorageB:
    a: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_a], 1024
    ]
    b: cute.struct.Align[
        cute.struct.MemRange[cutlass.Float32, size_b], 1024
    ]
    x: cute.struct.Align[cutlass.Int32, 16]
    compA: cute.struct.Align[complex, 16]


# Statically get size and alignment:
size = StorageB.__sizeof__()
align = StorageB.__alignof__()

# Allocate and referencing elements:
storage = allocator.allocate(StorageB)

storage.a[0] ...
storage.x ...
storage.compA.real ...
```

* **Parameters:**
  **cls** – The struct class with annotations.
* **Returns:**
  The decorated struct class.

#### *class* Align

Bases: `object`

Aligns the given type by Align[T, alignment].

#### *class* MemRange

Bases: `object`

Defines a range of memory by MemRange[T, size].

#### *class* \_AlignMeta(name, bases, dct)

Bases: `type`

Aligns the given object by setting its alignment attribute.

* **Parameters:**
  * **v** – The object to align. Must be a struct, MemRange, or a scalar type.
  * **align** – The alignment value to set.
* **Raises:**
  **TypeError** – If the object is not a struct, MemRange, or a scalar type.
* **Variables:**
  * **\_dtype** – The data type to be aligned.
  * **\_align** – The alignment of the data type.

#### \_align *= None*

#### \_dtype *= None*

#### *property* align

#### *property* dtype

#### *class* \_MemRangeData(dtype, size, base)

Bases: `object`

Represents a range of memory.

* **Parameters:**
  * **dtype** – The data type.
  * **size** – The size of the memory range in bytes.
  * **base** – The base address of the memory range.

#### \_\_init_\_(dtype, size, base)

Initializes a new memory range.

* **Parameters:**
  * **dtype** – The data type.
  * **size** – Size of the memory range in bytes. A size of **0** is accepted, but in that
    case the range can only be used for its address (e.g. as a partition marker).
  * **base** – The base address of the memory range.

#### data_ptr(, loc=None, ip=None)

Returns start pointer to the data in this memory range.

* **Returns:**
  A pointer to the start of the memory range.
* **Raises:**
  **AssertionError** – If the size of the memory range is negative.

#### get_tensor(layout, swizzle=None, dtype=None, , loc=None, ip=None)

Creates a tensor from the memory range.

* **Parameters:**
  * **layout** – The layout of the tensor.
  * **swizzle** – Optional swizzle pattern.
  * **dtype** – Optional data type; defaults to the memory range’s data type if not specified.
* **Returns:**
  A tensor representing the memory range.
* **Raises:**
  * **TypeError** – If the layout is incompatible with the swizzle.
  * **AssertionError** – If the size of the memory range is not greater than zero.

#### *class* \_MemRangeMeta(name, bases, dct)

Bases: `type`

A metaclass for creating MemRange classes.

This metaclass is used to dynamically create MemRange classes with specific
data types and sizes.

* **Variables:**
  * **\_dtype** – The data type of the MemRange.
  * **\_size** – The size of the MemRange.

#### \_dtype *= None*

#### \_size *= None*

#### *property* elem_width

#### *property* size

#### *property* size_in_bytes

#### \_\_init_\_(cls)

Initializes a new struct decorator instance.

* **Parameters:**
  **cls** – The class representing the structured data type.
* **Raises:**
  **TypeError** – If the struct is empty.

#### *static* \_is_scalar_type(dtype)

Checks if the given type is a scalar numeric type.

* **Parameters:**
  **dtype** – The type to check.
* **Returns:**
  True if the type is a subclass of Numeric, False otherwise.

#### *static* align_offset(offset, align)

Return the round-up offset up to the next multiple of align.

#### size_in_bytes() → int

Returns the size of the struct in bytes.

* **Returns:**
  The size of the struct.

### cutlass.cute.tan(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise tangent of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor (in radians)
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the tangent of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = tan(y)  # Compute tangent
```

### cutlass.cute.tanh(a: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, fastmath: bool = False) → [TensorSSA](#cutlass.cute.TensorSSA) | Numeric

Compute element-wise hyperbolic tangent of the input tensor.

* **Parameters:**
  * **a** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Input tensor
  * **fastmath** (*bool* *,* *optional*) – Enable fast math optimizations, defaults to False
* **Returns:**
  Tensor containing the hyperbolic tangent of each element
* **Return type:**
  Union[[TensorSSA](#cutlass.cute.TensorSSA), Numeric]

Example:

```default
x = cute.make_rmem_tensor(layout)  # Create tensor
y = x.load()  # Load values
z = tanh(y)  # Compute hyperbolic tangent
```

### cutlass.cute.tile_to_shape(atom: [Layout](#cutlass.cute.Layout), trg_shape: Shape, order: Shape, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.tile_to_shape(atom: [ComposedLayout](#cutlass.cute.ComposedLayout), trg_shape: Shape, order: Shape, , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.tiled_divide(target: [Layout](#cutlass.cute.Layout), tiler: Tiler, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.tiled_divide(target: [Tensor](#cutlass.cute.Tensor), tiler: Tiler, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.tiled_product(block: [Layout](#cutlass.cute.Layout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.tiled_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)

### cutlass.cute.transform_apply(\*args, f: Callable, g: Callable)

Transform elements of tuple(s) with f, then apply g to all results.

This function applies f to corresponding elements across input tuple(s),
then applies g to all transformed results. It mimics the C++ CuTe implementation.

Supports multiple signatures:
- transform_apply(t, f, g): For single tuple, computes g(f(t[0]), f(t[1]), …)
- transform_apply(t0, t1, f, g): For two tuples, computes g(f(t0[0], t1[0]), f(t0[1], t1[1]), …)
- transform_apply(t0, t1, t2, …, f, g): For multiple tuples of same length

For non-tuple inputs, f is applied to the input(s) and g is applied to that single result.

* **Parameters:**
  * **args** – One or more tuples (or non-tuples) to transform
  * **f** (*Callable*) – The function to apply to each element (or corresponding elements across tuples)
  * **g** (*Callable*) – The function to apply to all transformed elements
  * **loc** (*optional*) – Source location for MLIR, defaults to None
  * **ip** (*optional*) – Insertion point, defaults to None
* **Returns:**
  The result of applying g to all transformed elements
* **Return type:**
  any

**Examples:**

```python
>>> transform_apply((1, 2, 3), f=lambda x: x * 2, g=lambda *args: sum(args))
12  # (1*2 + 2*2 + 3*2) = 12
>>> transform_apply((1, 2), f=lambda x: (x, x+1), g=tuple_cat)
(1, 2, 2, 3)
>>> transform_apply((1, 2), (3, 4), f=lambda x, y: x + y, g=lambda *args: args)
(4, 6)
```

### cutlass.cute.transform_leaf(f, \*args)

Apply a function to the leaf nodes of nested tuple structures.

This function traverses nested tuple structures in parallel and applies the function f
to corresponding leaf nodes. All input tuples must have the same nested structure.

* **Parameters:**
  * **f** (*Callable*) – Function to apply to leaf nodes
  * **args** – One or more nested tuple structures with matching profiles
* **Returns:**
  A new nested tuple with the same structure as the inputs, but with leaf values transformed by f
* **Raises:**
  **TypeError** – If the input tuples have different nested structures

**Example:**

```python
>>> transform_leaf(lambda x: x + 1, (1, 2))
(2, 3)
>>> transform_leaf(lambda x, y: x + y, (1, 2), (3, 4))
(4, 6)
>>> transform_leaf(lambda x: x * 2, ((1, 2), (3, 4)))
((2, 4), (6, 8))
```

### cutlass.cute.tuple_cat(\*tuples)

Concatenate multiple tuples into a single tuple.

This function takes any number of tuples and concatenates them into a single tuple.
Non-tuple arguments are treated as single-element tuples.

* **Parameters:**
  **tuples** (*tuple* *or* *any*) – Variable number of tuples to concatenate
* **Returns:**
  A single concatenated tuple
* **Return type:**
  tuple

**Examples:**

```python
>>> tuple_cat((1, 2), (3, 4))
(1, 2, 3, 4)
>>> tuple_cat((1,), (2, 3), (4,))
(1, 2, 3, 4)
>>> tuple_cat(1, (2, 3))
(1, 2, 3)
```

### cutlass.cute.unflatten(sequence: Tuple[Any, ...] | List[Any] | Iterable[Any], profile: Any | Tuple[Any | Tuple[XTuple, ...], ...]) → Any | Tuple[Any | Tuple[XTuple, ...], ...]

Unflatten a flat tuple into a nested tuple structure according to a profile.

This function transforms a flat sequence of elements into a nested tuple structure
that matches the structure defined by the profile parameter. It traverses the profile
structure and populates it with elements from the sequence.

sequence must be long enough to fill the profile. Raises RuntimeError if it is not.

* **Parameters:**
  * **sequence** (*Union* *[**Tuple* *[**Any* *,*  *...* *]* *,* *List* *[**Any* *]* *,* *Iterable* *[**Any* *]* *]*) – A flat sequence of elements to be restructured
  * **profile** (*XTuple*) – A nested tuple structure that defines the shape of the output
* **Returns:**
  A nested tuple with the same structure as profile but containing elements from sequence
* **Return type:**
  XTuple

**Examples:**

```python
unflatten([1, 2, 3, 4], ((0, 0), (0, 0)))  # Returns ((1, 2), (3, 4))
```

### cutlass.cute.where(cond: [TensorSSA](#cutlass.cute.TensorSSA), x: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, y: [TensorSSA](#cutlass.cute.TensorSSA) | Numeric, , loc=None, ip=None) → [TensorSSA](#cutlass.cute.TensorSSA)

Return elements chosen from x or y depending on condition; will auto broadcast x or y if needed.

* **Parameters:**
  * **cond** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – Where True, yield x, where False, yield y.
  * **x** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Values from which to choose when condition is True.
  * **y** (*Union* *[*[*TensorSSA*](#cutlass.cute.TensorSSA) *,* *Numeric* *]*) – Values from which to choose when condition is False.
* **Returns:**
  A tensor with elements from x where condition is True, and elements from y where condition is False.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.zeros_like(a, dtype=None, , loc=None, ip=None)

Return a TensorSSA of zeros with the same shape and type as a given array.

* **Parameters:**
  * **a** ([*TensorSSA*](#cutlass.cute.TensorSSA)) – The shape and data-type of a define these same attributes of the returned array.
  * **dtype** (*Type* *[**Numeric* *]* *,* *optional*) – Overrides the data type of the result, defaults to None
* **Returns:**
  Tensor of zeros with the same shape and type (unless overridden) as a.
* **Return type:**
  [TensorSSA](#cutlass.cute.TensorSSA)

### cutlass.cute.zipped_divide(target: [Layout](#cutlass.cute.Layout), tiler: Tiler, , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.zipped_divide(target: [Tensor](#cutlass.cute.Tensor), tiler: Tiler, , loc=None, ip=None) → [Tensor](#cutlass.cute.Tensor)

### cutlass.cute.zipped_product(block: [Layout](#cutlass.cute.Layout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [Layout](#cutlass.cute.Layout)

### cutlass.cute.zipped_product(block: [ComposedLayout](#cutlass.cute.ComposedLayout), tiler: [Layout](#cutlass.cute.Layout), , loc=None, ip=None) → [ComposedLayout](#cutlass.cute.ComposedLayout)
