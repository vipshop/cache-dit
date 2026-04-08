<a id="cute-runtime"></a>

# Runtime

Description

## API documentation

### *class* cutlass.cute.runtime.TensorAdapter(arg)

Bases: `object`

Convert a DLPack protocol supported tensor/array to a cute tensor.

#### \_\_init_\_(arg)

### *class* cutlass.cute.runtime.\_FakeCompactTensor(dtype, shape, stride_order, memspace=None, assumed_align=None, use_32bit_stride=False)

Bases: [`Tensor`](cute.md#cutlass.cute.Tensor)

#### \_\_init_\_(dtype, shape, stride_order, memspace=None, assumed_align=None, use_32bit_stride=False)

#### \_abc_impl *= <_abc._abc_data object>*

#### *property* dynamic_shapes_mask

#### *property* dynamic_strides_mask

#### *property* element_type *: Type[Numeric]*

#### fill(value: Numeric)

#### *property* iterator

#### *property* leading_dim

#### *property* memspace

#### *property* mlir_type *: Type*

#### *property* shape

#### *property* stride

### *class* cutlass.cute.runtime.\_FakeStream(, use_tvm_ffi_env_stream: bool = False)

Bases: `object`

A fake stream that can be used as a placeholder for a stream in compilation.

When use_tvm_ffi_env_stream is True and the function is compiled with TVM-FFI,
the argument will be skipped from the function signature and we pass in
this value through the environment stream obtained from caller context
(e.g. torch.cuda.current_stream()).

#### \_\_init_\_(, use_tvm_ffi_env_stream: bool = False)

#### use_tvm_ffi_env_stream *: bool*

### *class* cutlass.cute.runtime.\_FakeTensor(dtype, shape, , stride, memspace=None, assumed_align=None)

Bases: [`Tensor`](cute.md#cutlass.cute.Tensor)

Fake Tensor implementation as a placeholder.
It mimics the interface of Tensor, but does not hold real data or allow indexing.
Used for compilation or testing situations where only shape/type/layout information is needed.
All attempts to access or mutate data will raise errors.

#### \_\_init_\_(dtype, shape, , stride, memspace=None, assumed_align=None)

#### \_abc_impl *= <_abc._abc_data object>*

#### *property* dynamic_shapes_mask

#### *property* dynamic_strides_mask

#### *property* element_type *: Type[Numeric]*

#### fill(value: Numeric)

#### *property* iterator

#### *property* memspace

#### *property* mlir_type *: Type*

#### *property* shape

#### *property* stride

### *class* cutlass.cute.runtime.\_Pointer(pointer, dtype, mem_space: [AddressSpace](cute.md#cutlass.cute.AddressSpace) = AddressSpace.generic, assumed_align=None)

Bases: `Pointer`

Runtime representation of a pointer that can inter-operate with various data structures,
including numpy arrays and device memory.

* **Parameters:**
  * **pointer** (*int* *or* *pointer-like object*) – The pointer to the data
  * **dtype** (*Type*) – Data type of the elements pointed to
  * **mem_space** ( *\_cute_ir.AddressSpace* *,* *optional*) – Memory space where the pointer resides, defaults to generic
  * **assumed_align** (*int* *,* *optional*) – Assumed alignment of input pointer in bytes, defaults to None
* **Variables:**
  * **\_pointer** – The underlying pointer
  * **\_dtype** – Data type of the elements
  * **\_addr_space** – Memory space of the pointer
  * **\_assumed_align** – Alignment of the pointer in bytes
  * **\_desc** – C-type descriptor for the pointer
  * **\_c_pointer** – C-compatible pointer representation

#### \_\_init_\_(pointer, dtype, mem_space: [AddressSpace](cute.md#cutlass.cute.AddressSpace) = AddressSpace.generic, assumed_align=None)

#### \_abc_impl *= <_abc._abc_data object>*

#### align(min_align: int, , loc=None, ip=None) → Pointer

#### *property* dtype *: Type[Numeric]*

#### *property* memspace

#### *property* mlir_type *: Type*

#### size_in_bytes() → int

### *class* cutlass.cute.runtime.\_Tensor(tensor, assumed_align=None, use_32bit_stride=False, , enable_tvm_ffi=False)

Bases: [`Tensor`](cute.md#cutlass.cute.Tensor)

#### \_\_init_\_(tensor, assumed_align=None, use_32bit_stride=False, , enable_tvm_ffi=False)

#### \_abc_impl *= <_abc._abc_data object>*

#### *property* data_ptr

#### *property* dynamic_shapes_mask

Get the mask of dynamic shapes in the tensor.

#### *property* dynamic_strides_mask

Get the mask of dynamic strides in the tensor.

#### *property* element_type *: Type[Numeric]*

#### fill(value: Numeric)

#### *property* iterator

#### *property* layout

#### *property* leading_dim

Get the leading dimension of this Tensor.

* **Returns:**
  The leading dimension index or indices
* **Return type:**
  int or tuple or None

The return value depends on the tensor’s stride pattern:

* If a single leading dimension is found, returns an integer index
* If nested leading dimensions are found, returns a tuple of indices
* If no leading dimension is found, returns None

#### load_dltensor()

Lazily load the DLTensorWrapper.

This function loads the DLTensorWrapper when needed,
avoiding overhead in the critical path of calling JIT functions.

#### mark_compact_shape_dynamic(mode: int, stride_order: tuple[int, ...] | None = None, divisibility: int = 1)

Marks the tensor shape as dynamic and propagates dynamic and divisibility information to the corresponding strides.

* **Parameters:**
  * **mode** (*int*) – The mode of the compact shape, defaults to 0
  * **stride_order** – Consistent with torch.Tensor.dim_order. Defaults to None.

Indicates the order of the modes (dimensions) if the current layout were converted to row-major order.
It starts from the outermost to the innermost dimension.
:type stride_order: tuple[int, …], optional
:param divisibility: The divisibility constraint for the compact shape, defaults to 1
:type divisibility: int, optional
:return: The tensor with dynamic compact shape
:rtype: \_Tensor

If `stride_order` is not provided, the stride ordering will be automatically deduced from the layout.
Automatic deduction is only possible when exactly one dimension has a stride of 1 (compact layout).
An error is raised if automatic deduction fails.

If `stride_order` is explicitly specified, it does the consistency check with the layout.

For example:
- Layout: (4,2):(1,4) has stride_order: (1,0) indicates the innermost dimension is 0(4:1), the outermost dimension is 1(2:4)
- Layout: (5,3,2,4):(3,1,15,30) has stride_order: (3,2,0,1) indicates the innermost dimension is 1(3:1), the outermost dimension is 3(4:30).

Using torch.Tensor.dim_order() to get the stride order of the torch tensor.
.. code-block:: python
a = torch.empty(3, 4)
t = cute.runtime.from_dlpack(a)
t = t.mark_compact_shape_dynamic(mode=0, stride_order=a.dim_order())

#### mark_layout_dynamic(leading_dim: int | None = None)

Marks the tensor layout as dynamic based on the leading dimension.

* **Parameters:**
  **leading_dim** (*int* *,* *optional*) – The leading dimension of the layout, defaults to None

When `leading_dim` is None, automatically deduces the leading dimension from the tensor layout.
The layout can be deduced only when exactly one dimension has a stride of 1. Raises an error
if the layout cannot be automatically deduced.

When `leading_dim` is explicitly specified, marks the layout as dynamic while setting the
stride at `leading_dim` to 1. Also validates that the specified `leading_dim` is consistent
with the existing layout by checking that the corresponding stride of that dimension is 1.

Limitation: only support flat layout for now. Will work on supporting nested layout in the future.

* **Returns:**
  The tensor with dynamic layout
* **Return type:**
  [\_Tensor](#cutlass.cute.runtime._Tensor)

#### *property* memspace

#### *property* mlir_type *: Type*

#### *property* shape

#### *property* size_in_bytes *: int*

#### *property* stride

### cutlass.cute.runtime.\_get_cute_type_str(inp)

### cutlass.cute.runtime.find_runtime_libraries(, enable_tvm_ffi: bool = True) → List[str]

Find the runtime libraries that needs to be available for loading modules.

* **Parameters:**
  **enable_tvm_ffi** (*bool* *,* *optional*) – Whether to enable TVM-FFI.
* **Returns:**
  A list of runtime libraries that needs to be available for loading modules.
* **Return type:**
  list

### cutlass.cute.runtime.from_dlpack(tensor_dlpack, assumed_align=None, use_32bit_stride=False, , enable_tvm_ffi=False, force_tf32=False) → [Tensor](cute.md#cutlass.cute.Tensor)

Convert from tensor object supporting \_\_dlpack_\_() to a CuTe Tensor.

* **Parameters:**
  * **tensor_dlpack** (*object*) – Tensor object that supports the DLPack protocol
  * **assumed_align** (*int* *,* *optional*) – Assumed alignment of the tensor (bytes), defaults to None,
    if None, will use the element size bytes as the assumed alignment.
  * **use_32bit_stride** (*bool* *,* *optional*) – Whether to use 32-bit stride, defaults to False. When True, the dynamic
    stride bitwidth will be set to 32 for small problem size (cosize(layout) <= Int32_max) for better performance.
    This is only applied when the dimension is dynamic.
  * **enable_tvm_ffi** (*bool* *,* *optional*) – Whether to enable TVM-FFI, defaults to False. When True, the tensor will be converted to
    a TVM-FFI function compatible tensor.
  * **force_tf32** (*bool* *,* *optional*) – Whether to force the element type to TFloat32 if the element type is Float32.
* **Returns:**
  A CuTe Tensor object
* **Return type:**
  [Tensor](cute.md#cutlass.cute.Tensor)

**Examples:**

```python
import torch
from cutlass.cute.runtime import from_dlpack
x = torch.randn(100, 100)
y = from_dlpack(x)
y.shape
# (100, 100)
type(y)
# <class 'cutlass.cute.Tensor'>
```

### cutlass.cute.runtime.load_module(file_path: str, , enable_tvm_ffi: bool = False)

Load a module from a file path.

* **Parameters:**
  * **file_path** (*str*) – The path to the module file
  * **enable_tvm_ffi** (*bool* *,* *optional*) – Whether to enable TVM-FFI, defaults to True. When True, the module will be loaded as a TVM-FFI module.
* **Returns:**
  A module object
* **Return type:**
  module

### cutlass.cute.runtime.make_fake_compact_tensor(dtype, shape, , stride_order=None, memspace=None, assumed_align=None, use_32bit_stride=False)

Create a fake tensor with the specified shape, element type, and a compact memory layout.

* **Parameters:**
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the tensor elements.
  * **shape** (*tuple* *[**int* *,*  *...* *]*) – Shape of the tensor.
  * **stride_order** (*tuple* *[**int* *,*  *...* *]* *,* *optional*) – Order in which strides (memory layout) are assigned to the tensor dimensions.
    If None, the default layout is left-to-right order (known as column-major order for flatten layout).
    Otherwise, it should be a permutation order of the dimension indices.
  * **memspace** (*str* *,* *optional*) – Memory space where the fake tensor resides. Optional.
  * **assumed_align** (*int* *,* *optional*) – Assumed byte alignment for the tensor data. If None, the default alignment is used.
  * **use_32bit_stride** (*bool* *,* *optional*) – Whether to use 32-bit stride for dynamic dimensions. If True and the total size of the
    layout (cosize(layout)) fits within int32, then dynamic strides will use 32-bit integers for improved performance.
    Only applies when dimensions are dynamic. Defaults to False.
* **Returns:**
  An instance of a fake tensor with the given properties and compact layout.
* **Return type:**
  [\_FakeCompactTensor](#cutlass.cute.runtime._FakeCompactTensor)

**Examples:**

```python
@cute.jit
def foo(x: cute.Tensor):
    ...

x = make_fake_compact_tensor(
    cutlass.Float32, (100, cute.sym_int32(divisibility=8)), stride_order=(1, 0)
)

# Compiled function will take a tensor with the type:
#   tensor<ptr<f32, generic> o (100,?{div=8}):(?{i32 div=8},1)>
compiled_foo = cute.compile(foo, x)

# Default stride order is left-to-right order: (1, 8)
y = make_fake_compact_tensor(cutlass.Float32, (8, 3))
```

### cutlass.cute.runtime.make_fake_stream(, use_tvm_ffi_env_stream: bool = False)

Create a fake stream that can be used as a placeholder for a stream in compilation.

When use_tvm_ffi_env_stream is True and the function is compiled with TVM-FFI,
the argument will be skipped from the function signature and we pass in
this value through the environment stream obtained from caller context
(e.g. torch.cuda.current_stream()). This can speedup the calling process
since we no longer need to do stream query in python.

* **Parameters:**
  **use_tvm_ffi_env_stream** (*bool*) – Whether to skip this parameter use environment stream instead.

### cutlass.cute.runtime.make_fake_tensor(dtype, shape, stride, , memspace=None, assumed_align=None)

Create a fake tensor with the specified element type, shape, and stride.

* **Parameters:**
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the tensor elements.
  * **shape** (*tuple* *[**int* *,*  *...* *]*) – Shape of the tensor.
  * **stride** (*tuple* *[**int* *,*  *...* *]*) – Stride of the tensor.
  * **assumed_align** (*int* *,* *optional*) – Assumed byte alignment for the tensor data. If None, the default alignment is used. Defaults to None.
* **Returns:**
  An instance of a fake tensor with the given properties.
* **Return type:**
  [\_FakeTensor](#cutlass.cute.runtime._FakeTensor)

### cutlass.cute.runtime.make_ptr(dtype: Type[Numeric], value: int | \_Pointer, mem_space: [AddressSpace](cute.md#cutlass.cute.AddressSpace) = AddressSpace.generic, assumed_align=None) → Pointer

Create a pointer from a memory address

* **Parameters:**
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the pointer elements
  * **value** (*Union* *[**int* *,* *ctypes._Pointer* *]*) – Memory address as integer or ctypes pointer
  * **mem_space** ([*AddressSpace*](cute.md#cutlass.cute.AddressSpace) *,* *optional*) – Memory address space, defaults to AddressSpace.generic
  * **align_bytes** (*int* *,* *optional*) – Alignment in bytes, defaults to None
* **Returns:**
  A pointer object
* **Return type:**
  Pointer

```python
import numpy as np
import ctypes

from cutlass import Float32
from cutlass.cute.runtime import make_ptr

# Create a numpy array
a = np.random.randn(16, 32).astype(np.float32)

# Get pointer address as integer
ptr_address = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

# Create pointer from address
y = make_ptr(cutlass.Float32, ptr_address)

# Check properties
print(y.element_type)
print(type(y))  # <class 'cutlass.cute.Pointer'>
```

### cutlass.cute.runtime.nullptr(dtype: Type[Numeric], mem_space: [AddressSpace](cute.md#cutlass.cute.AddressSpace) = AddressSpace.generic, assumed_align=None) → Pointer

Create a null pointer which is useful for compilation

* **Parameters:**
  * **dtype** (*Type* *[**Numeric* *]*) – Data type of the pointer elements
  * **mem_space** ([*AddressSpace*](cute.md#cutlass.cute.AddressSpace) *,* *optional*) – Memory address space, defaults to AddressSpace.generic
* **Returns:**
  A null pointer object
* **Return type:**
  Pointer
