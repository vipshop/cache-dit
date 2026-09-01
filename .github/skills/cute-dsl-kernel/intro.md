# Introduction

## Overview
CuTe DSL is a Python-based domain-specific language (DSL) designed for dynamic compilation of high-performance GPU kernels. It evolved from the C++ CUTLASS library and is now available as a decorator-based DSL.

Its primary goals are:

- Zero-cost abstraction, DSL is a zero-cost abstraction thanks to Hybrid DSL approach.
- Consistent with CuTe C++, allowing users to express GPU kernels with full control of the hardware.
- JIT compilation for both host and GPU execution.
- DLPack integration, enabling seamless interop with frameworks (e.g., PyTorch, JAX).
- JIT caching, so that repeated calls to the same function benefit from cached IR modules.
- Native types and type inference to reduce boilerplate and improve performance.
- Optional lower-level control, offering direct access to GPU backends or specialized IR dialects.

## Decorators
CuTe DSL provides two main Python decorators for generating optimized code via dynamic compilation:

1. `@jit` — Host-side JIT-compiled functions
2. `@kernel` — GPU kernel functions

Both decorators can optionally use a preprocessor that automatically expands Python control flow (loops, conditionals) into operations consumable by the underlying IR.

### `@jit`
Declares JIT-compiled functions that can be invoked from Python or from other CuTe DSL functions.

Decorator Parameters:

- `preprocessor`:
  - `True` (default) - Automatically translate Python flow control (e.g., loops, if-statements) into IR operations.
  - `False` - No automatic expansion; Python flow control must be handled manually or avoided.
  
Call-site Parameters:

- `no_cache`:
  - `True` — Disables JIT caching, forcing a fresh compilation each call.
  - `False` (default) — Enables caching for faster subsequent calls.

### `@kernel`
Defines GPU kernel functions, compiled as specialized GPU symbols through dynamic compilation.

Decorator Parameters:

- `preprocessor`:
  - `True` (default) — Automatically expands Python loops/ifs into GPU-compatible IR operations.
  - `False` — Expects manual or simplified kernel implementations.

Kernel Launch Parameters:

- `grid` Specifies the grid size as a list of integers.
- `block` Specifies the block size as a list of integers.
- `cluster` Specifies the cluster size as a list of integers.
- `smem` Specifies the size of shared memory in bytes (integer).

## Calling Conventions
| Caller          | Callee          | Allowed | Compilation/Runtime                     |
| --------------- | --------------- | ------- | --------------------------------------  |
| Python function | `@jit`          | ✅       | DSL runtime                            |
| Python function | `@kernel`       | ❌       | N/A (error raised)                     |
| `@jit`          | `@jit`          | ✅       | Compile-time call, inlined             |
| `@jit`          | Python function | ✅       | Compile-time call, inlined             |
| `@jit`          | `@kernel`       | ✅       | Dynamic call via GPU driver or runtime |
| `@kernel`       | `@jit`          | ✅       | Compile-time call, inlined             |
| `@kernel`       | Python function | ✅       | Compile-time call, inlined             |
| `@kernel`       | `@kernel`       | ❌       | N/A (error raised)                     |


# End-to-End Code Generation
## 1. Hybrid DSL: Python Metaprogramming, Structured GPU Code
CuTe DSL is a hybrid DSL that combines two compilation techniques: AST rewrite and tracing. This combination gives you the best of both worlds:

- Program structure is preserved — control flow (loops, branches) is captured via AST rewrite, compiling to proper structured code instead of flattened traces.
- Python stays Python — arithmetic and tensor operations are captured via tracing, so dynamic shapes, metaprogramming, and Python’s rich expression language work naturally.

To understand why this matters, let’s look at each technique.

### 1.1 AST Rewrite
The function’s abstract-syntax tree is analysed before execution. Python control-flow (`for/while`, `if/else`) and built-ins are converted to structured intermediate representation (IR) constructs. Computation inside each region is left untouched at this stage.

Advantages
- Sees the entire program, so every branch and loop is preserved.
- Keeps loop structure intact for optimization such as tiling, vectorisation or GPU thread mapping.

Disadvantages
- Requires a well-defined Python subset that the rewriter understands.


### 1.2 Tracing
The decorated function is executed once with proxy arguments; overloaded operators record every tensor operation that actually runs and produce a flat trace that is lowered to intermediate representation (IR).

Advantages
- Near-zero compile latency, ideal for straight-line arithmetic.
- No need to parse Python source, so it supports many dynamic Python features, and Python has many features.

Disadvantages
- Untaken branches vanish, so the generated kernel may be wrong for other inputs.
- Loops are flattened to the iteration count observed during tracing.
- Data-dependent control-flow freezes to a single execution path.

### 1.3 The Hybrid Solution
As shown above, neither technique alone is sufficient—but together they complement each other perfectly.

**Why this works: GPU kernels are simple at runtime**

High-performance GPU kernels are structurally simple at runtime: they avoid deep call hierarchies, complex branching, and dynamic dispatch. However, authoring such kernels benefits greatly from Python’s abstractions—classes, metaprogramming, and polymorphic patterns improve readability and maintainability

The hybrid approach resolves this tension by evaluating Python abstractions at compile time while emitting simple, optimized code for runtime execution.

**How |DSL| divides the work:**

1. **AST rewrite handles structure** — loops (`for`, `while`) and branches (`if/else`) are converted to structured intermediate representation (IR) before execution. This solves tracing’s control-flow problem.
2. **Tracing handles arithmetic** — inside each structured region, the tracer records tensor operations exactly as they execute. No need to model Python’s complex semantics—just run Python and record what happens. This solves AST rewriting’s complexity problem.

The result:
- Loops compile to real loops, not unrolled traces.
- All branches are preserved, even if not taken during tracing.
- Dynamic shapes, metaprogramming, and Python idioms work naturally.
- The rewriter only needs to understand control flow, not all of Python.


## 2. CuTe DSL Compilation Flow: Meta-Stage to Object-Stage
CuTe DSL bridges Python and GPU hardware through a three-stage pipeline.

**Stage 1: Pre-Staging (Python AST)**
Before any code executes, the AST preprocessor rewrites the decorated function. It inserts callbacks around control-flow constructs—loops, branches, and function boundaries—so that program structure is captured explicitly rather than lost during execution.

**Stage 2: Meta-Stage (Python Interpreter)**
The rewritten function runs in the Python interpreter with proxy tensor arguments. As execution proceeds:
- Callbacks fire at control-flow boundaries, emitting structured intermediate representation (IR) (loops, branches, etc.).
- Tensor operations are traced: each operator invocation records the corresponding operation.
- Compile-time constants are partially evaluated—values known at JIT time fold directly into the intermediate representation (IR), enabling aggressive specialization.

The result is a complete representation of the kernel, with both high-level structure and low-level arithmetic intact.

**Stage 3: Object-Stage (Compiler Backend)**
The internal representation passes through a lowering pipeline:

1. High-level operations are progressively lowered toward hardware-specific representations.
2. Optimization passes (tiling, vectorization, memory promotion) reshape the code for the target architecture.
3. The final code is translated to PTX/SASS (for NVIDIA GPUs) and assembled into a device binary.

At runtime, the compiled kernel is loaded and launched on the accelerator.


## 3. Meta-Programming vs Runtime: Two Worlds in One Function
A key insight for understanding CuTe DSL is that your Python code runs twice, in two very different contexts:

1. **Meta-programming time (compilation)** — Python executes to build the kernel. This happens on the host CPU when you call a `@jit` function.
2. **Runtime (execution)** — The compiled kernel runs on the GPU with actual tensor data.

This distinction determines what you can observe and when.

### print() vs cute.printf(): Meta-Stage vs Object-Stage Output
CuTe DSL provides two ways to print values, each operating at a different stage:
- **Python’s** `print()` — executes during the meta-stage (compilation). Use it to inspect what the compiler sees.
- `cute.printf()` — compiles into the kernel and executes at runtime on the GPU. Use it to observe actual tensor values during execution.

The following examples demonstrate how the same result variable appears differently depending on when and how you print it.

**Example 1: Dynamic variables (both `a` and `b` are runtime values)**

```python
@cute.jit
def add_dynamicexpr(b: cutlass.Float32):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_dynamicexpr(5.0)
```

```shell
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

At meta-stage, `result` is a proxy—its value is unknown until the kernel runs. At runtime, `cute.printf()` prints the actual GPU-computed value.

**Example 2: Compile-time constants (both a and b are Constexpr)**

```python
@cute.jit
def add_constexpr(b: cutlass.Constexpr):
    a = 2.0
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_constexpr(5.0)
```

```shell
$> python myprogram.py
[meta-stage] result = 7.0
[object-stage] result = 7.000000
```

Both values are known at compile time, so Python evaluates `2.0 + 5.0 = 7.0` during tracing. The constant is baked into the compiled kernel.

**Example 3: Hybrid ( a is dynamic, b is Constexpr)**

```python
@cute.jit
def add_hybrid(b: cutlass.Constexpr):
    a = cutlass.Float32(2.0)
    result = a + b
    print("[meta-stage] result =", result)          # runs at compile time
    cute.printf("[object-stage] result = %f\n", result)  # runs on GPU

add_hybrid(5.0)
```

```shell
$> python myprogram.py
[meta-stage] result = <Float32 proxy>
[object-stage] result = 7.000000
```

The constant `b = 5.0` is folded in, but since `a` is dynamic, the result remains a proxy at meta-stage. The GPU computes the final answer at runtime.

### Practical Implications
- **Use `print()` to debug your meta-program**— inspect shapes, strides, tile sizes, and compile-time decisions.
- **Constexpr parameters enable specialization** — the compiler can generate tighter code when values are known at JIT time.
- **Dynamic parameters preserve generality** — a single compiled kernel can handle varying input sizes without recompilation.

## 4. CuTe DSL Code-Generation Modes
CuTe’s Python front-end combines the techniques above into two mutually exclusive modes (see Left: tracing mode records only the path that executed. Right: preprocessor mode emits structured intermediate representation (IR) for every branch and loop before tracing the arithmetic.), selectable with the `preprocessor` flag of the `@jit` decorator:

1. Tracing mode `@jit(preprocess=False)` – tracing only. This results in the fastest compilation path and is recommended only for kernels that are guaranteed to be straight-line arithmetic. It suffers from all tracing limitations listed in the previous section.
2. Preprocessor mode (**default**) `@jit(preprocess=True)` – AST rewrite + tracing. The AST pass captures every loop and branch, eliminating the correctness and optimisation problems of pure tracing; tracing then fills in the arithmetic. This hybrid “preprocessor” pipeline is unique to CuTe DSL and was designed specifically to overcome the disadvantages identified above.

# Control Flow
## Overview
CuTe DSL walks Python’s AST and converts each control-flow construct it finds into structured intermediate representation (IR). You can therefore write ordinary Python loops and branches while the compiler decides—statement by statement—whether to

- evaluate at compile time if it’s a native Python control flow, or
- emit intermediate representation (IR) when the control flow is marked as dynamic.

Passing intermediate representation (IR) values to a native Python control flow will result in an error.

## For Loops
CuTe DSL recognises three kinds of ranges for `for` loops:
- `range` – the Python built-in, always lowered to intermediate representation (IR)
- `cutlass.range` - Same as Python built-in `range`, but supports advanced unrolling and pipelining control
- `cutlass.range_constexpr` – unrolled at compile time

### range(…)/cutlass.range(…)
Use when you always want a loop in the generated intermediate representation (IR), even if the inputs are Python values.

### cutlass.range_constexpr(…)
Runs in the Python interpreter and is fully unrolled before code generation. All loop indices must be Constexpr (compile-time Python value).

Example:

```python
@cute.jit
def control_flow_examples(bound: cutlass.Int32):
    n = 10

    # ✅ This loop is Python loop, evaluated at compile time.
    for i in cutlass.range_constexpr(n):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, even when bound is Python value.
    for i in range(n):
        cute.printf("%d\\n", i)

    # ❌ This loop bound is a dynamic value, not allowed in Python loop.
    # Should use `range` instead.
    for i in cutlass.range_constexpr(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop.
    for i in range(bound):
        cute.printf("%d\\n", i)

    # ✅ This loop is dynamic, emitted IR loop with unrolling
    for i in cutlass.range(bound, unroll=2):
        cute.printf("%d\\n", i)
```

### Software Pipelining
Software pipelining is a technique used to optimize loops. Typically, this involves writing a prefetch loop and a main loop.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    # prefetch loop
    for i in range(prefetch_stages):
        cute.copy(atom, gmem[i], buffer[i], ...)

    # main loop
    for i in range(bound):
        if i + prefetch_stages < bound:
            cute.copy(atom, gmem[i + prefetch_stages], buffer[(i + prefetch_stages) % total_stages], ...)

        use(buffer[i % total_stages])

    ...
```

This can be tedious to write and tune. CuTe DSL provides a loop attribute to ask the compiler to do this.

```python
@cute.jit
def example():
    ...
    # build a circular buffer
    buffer = ...

    for i in cutlass.range(bound, prefetch_stages=prefetch_stages):
        # Compiler automatically handles the pipelining:
        # - Generates prefetch loop for initial stages
        # - In main loop, prefetches future data while using current data
        cute.copy(atom, gmem[i], buffer[i % total_stages], ...)
        use(buffer[i % total_stages])  # Uses data from previous iterations

    ...
```

Compiler will automatically generate the prefetch loop with prefetch_stages iterations and a corresponding main loop.

This feature is experimental and only supported on sm90 and above.

## If-Else Statements

Standard Python `if/elif/else` is supported.

- Predicate without annotation → lowered to intermediate representation (IR).
- Predicate annotated with `cutlass.const_expr` → evaluated at compile time.

Example:

```python
@cute.jit
def main(const_var: cutlass.Constexpr, dynamic_var: cutlass.Int32):
    # ✅ This branch is Python branch, evaluated at compile time.
    if cutlass.const_expr(const_var):
        cute.printf("Const branch\\n")
    else:
        cute.printf("Const else\\n")

    # ✅ This branch is dynamic branch, emitted IR branch.
    if dynamic_var == 10:
        cute.printf("Dynamic True\\n")
    else:
        cute.printf("Dynamic False\\n")

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    if cutlass.const_expr(dynamic_var == 10):
        cute.printf("Bound is 10\\n")

```

## While Loops
Standard Python `while` is supported.
- Condition without annotation → lowered to intermediate representation (IR).
- Condition annotated with `cutlass.const_expr` → evaluated at compile time.

Example:

```python
@cute.jit
def main(dynamic_var: cutlass.Int32):
    n = 0

    # ✅ This is Python while loop, evaluated at compile time.
    while cutlass.const_expr(n < 10):
        cute.printf("Const branch\\n")
        n += 1

    # ✅ This is dynamic while loop, emitted IR while loop.
    while dynamic_var == 10:
        cute.printf("Dynamic True\\n")
        n += 1

    # ❌ Using a dynamic value with `cutlass.const_expr` is not allowed.
    while cutlass.const_expr(n < dynamic_var):
        n += 1
```

## Summary of Control Flow behavior

| Control Flow                                                         | Run time evaluation | Compile time evaluation |
| -------------------------------------------------------------------- | ------------------- | ----------------------- |
| if cutlass.const_expr()                                              | ❌                  | ✅                      |
| if pred                                                              | ✅                  | ❌                      |
| while cutlass.const_expr()                                           | ❌                  | ✅                      |
| while pred                                                           | ✅                  | ❌                      |
| for i in cutlass.range_constexpr()                                   | ❌                  | ✅                      |
| for i in range()                                                     | ✅                  | ❌                      |
| for i in cutlass.range() (support advanced unrolling and pipelining) | ✅                  | ❌                      |

## Compile-Time Metaprogramming
Mix compile-time constructs with normal CuTe DSL code to generate specialised kernels without runtime overhead. A compile-time flag can, for example, toggle an optional ReLU epilogue:

```python
@cute.kernel
def gemm(..., do_relu: cutlass.Constexpr):
    # main GEMM work
    ...
    if cutlass.const_expr(do_relu):    # compile-time guard
        # ReLU code is emitted only when do_relu is True
        ...
```

```shell
gemm(..., False)   # ReLU is omitted from the generated |IR|
gemm(..., True)    # ReLU is included
```

### Limitations of Dynamic Control Flow
- Early-exit `break, continue, pass` or raising exception from control flow body are not yet supported.
- Operations in the control flow body are traced only when tracing is active in that region.
- Values originating in control flow body are not available outside the control flow.
- Changing type of a variable in control flow body is not allowed.

Example:

```python
@cute.jit
def control_flow_negative_examples(predicate: cutlass.Boolean):
    n = 10

    # ❌ This loop is dynamic, early-exit isn't allowed.
    for i in range(n):
        if i == 5:
            break         # Early-exit

    if predicate:
        val = 10
        # ❌ return from control flow body is not allowed.
        return
        # ❌ Raising exception from control flow body is not allowed.
        raise ValueError("This is not allowed")
        # ❌ Using pass in control flow body is not allowed.
        pass

    # ❌ val is not available outside the dynamic if
    cute.printf("%d\\n", val)

    if predicate:
        # ❌ Changing type of a variable in control flow body is not allowed.
        n = 10.0
```

# JIT Function Argument Generation
## In a nutshell
When using the `@jit` or `@kernel` decorators to define a JIT-compiled function, the arguments to the function are traced to determine the JIT function’s signature. CuTe DSL provides a Pythonic way to write the arguments for JIT function as one normally would in Python, and the CuTe DSL will take care of the rest for you.

Specifically, CuTe DSL honors following when generating the JIT function’s arguments:

- JIT function arguments are assumed to be **dynamic arguments** by default.
- If an argument is explicitly type annotated with `cutlass.Constexpr`, it is treated as a **compile-time constant**.
- If type annotation is provided, CuTe DSL validates the argument type at compile time for **type safety**.
- CuTe DSL provides **runtime checkable protocols** (`JitArgument` and `DynamicExpression`) for generating JIT function arguments for customized types.

More details below for each of the above.

## Static argument vs. Dynamic argument
CuTe DSL supports both static and dynamic arguments for JIT functions.

- Static arguments hold values that are known at compile time. It is not included in the generated JIT function signature.
- Dynamic arguments hold values that are only known at runtime.

By default, CuTe DSL assumes dynamic arguments and tries to infer the argument types from the call-site argument types. An explicit type annotation `cutlass.Constexpr` can be used to specify a static argument.

```python
import cutlass
import cutlass.cute as cute

@cute.jit
def foo(x: cutlass.Int32, y: cutlass.Constexpr):
    print("x = ", x)        # Prints x = ?
    print("y = ", y)        # Prints y = 2
    cute.printf("x: {}", x) # Prints x: 2
    cute.printf("y: {}", y) # Prints y: 2

foo(2, 2)
```

In the example above, `x` is a dynamic argument with type `cutlass.Int32` and `y` is a static argument.

With the `cutlass.Constexpr` annotation, a more sophisticated uses case of static argument in the JIT functions can be something like:

```python
import cutlass
import cutlass.cute as cute

@cute.kernel
def kernel(
    self,
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    mA_mkl: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    mB_nkl: cute.Tensor,
    tma_atom_c: Optional[cute.CopyAtom],
    mC_mnl: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    a_smem_layout_staged: cute.ComposedLayout,
    b_smem_layout_staged: cute.ComposedLayout,
    c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout, None],
    epi_tile: cute.Tile,
    epilogue_op: cutlass.Constexpr,
):
    ...

    # Perform epilogue op on accumulator and convert to C type
    acc_vec = tTR_rAcc.load()
    acc_vec = epilogue_op(acc_vec.to(self.c_dtype))
    tTR_rC.store(acc_vec)
```

In this example, `epilogue_op` is a static argument in the JIT kernel where the argument is used for the epilogue fusion. Upon calling the kernel, an elementwise lambda function can be passed in as the `epilogue_op` argument. For example, a ReLU can be applied for epilogue fusion by simply setting the `epilogue_op` to `lambda x: cute.where(x > 0, x, cute.full_like(x, 0))`

Refer to the [Blackwell dense GEMM example](https://raw.githubusercontent.com/NVIDIA/cutlass/refs/heads/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py) for a complete example.


## Type safety
CuTe DSL makes good use of type annotation in JIT function signature and validates the JIT function argument types at compile time for type safety.

```python
import cutlass
import cutlass.cute as cute
import numpy as np

@cute.jit
def foo(x: cute.Tensor, y: cutlass.Float16):
    ...

a = np.random.randn(10, 10).astype(np.float16)
b = 32

foo(a, b)
foo(b, a)  # This will fail at compile time due to type mismatch
```

The type safety check helps catch the type mismatch issue early at the compile time with clear error message to avoid tricky runtime errors which is usually more expensive to debug. In the example above, the second call to foo will fail at compile time due to the type mismatch with a clear error message:

```shell
cutlass.base_dsl.common.DSLRuntimeError: DSLRuntimeError: expects argument #1 (a) to be <class 'cutlass.cute.typing.Tensor'>, but got <class 'int'>
```


## JIT function arguments with customized types
CuTe DSL supports customized types for JIT function arguments by providing two runtime checkable protocols:

- `JitArgument` which is used for host JIT functions to be called from Python.
  - `__c_pointers__`: Generate a list of ctypes pointers for the current object.
  - `__get_mlir_types__`: Generate a list of MLIR types for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.
- `DynamicExpression` which is used for device JIT functions to be called from the host JIT functions.
  - `__extract_mlir_values__`: Generate a dynamic expression for the current object.
  - `__new_from_mlir_values__`: Create a new object from MLIR values.

Refer to [typing.py](https://raw.githubusercontent.com/NVIDIA/cutlass/refs/heads/main/python/CuTeDSL/cutlass/base_dsl/typing.py) for more details on these protocol APIs.

Depending on different cases of the customized types, CuTe DSL provides easy ways to adopt customized types for JIT function arguments.

### 1. Direct protocol implementation in customized types
One way is to implement the protocol methods directly in the customized types to enable the protocol based JIT function argument generation.

```python
import cutlass
import cutlass.cute as cute

# Customized type that implements the DynamicExpression protocol
class MyDynamicExpression:
    def __init__(self, tensor, offset):
        self._tensor = tensor # Dynamic argument
        self._offset = offset # Dynamic argument

    def __extract_mlir_values__(self):
        return [self._tensor.__extract_mlir_values__(), self._offset.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        return MyDynamicExpression(values[0], values[1])

@cute.kernel
def my_kernel(x: MyDynamicExpression):
    ...
```

In the example above, the `MyDynamicExpression` implements the `DynamicExpression` protocol and CuTe DSL will generate the JIT function arguments for the JIT kernel `my_kernel` based on the protocol methods.


### 2. Adaptor based protocol implementation for customized types
For the case where directly changing the customized types to implement the protocol is not feasible, CuTe DSL provides adaptor based approach to adapt the customized types for JIT function argument generation.

The JIT function argument adaptor is a callable object that implements the desired protocol methods for the registered customized types. This way, CuTe DSL automatically queries the JIT argument adaptor registry to generate the JIT function arguments for the given customized types.

```python
@cutlass.register_jit_arg_adapter(MyFrameworkObject)
class MyFrameworkObjectAdapter:
    """
    Convert a 3rd party framework object to a JIT function argument with JitArgument protocol
    """

    def __init__(self, arg):
        self._arg = arg

    def __c_pointers__(self):
        # Convert the framework object to a C-ABI compatible object
        # thru its C-ABI interface
        return [self._arg.get_cabi_pointer()]

    def __get_mlir_types__(self):
        # Return the list of MLIR types the framework object represents
        return [self._arg.get_data().mlir_type]

    def __new_from_mlir_values__(self, values):
        # Convert the MLIR values back to the framework object
        return MyFrameworkObject(values[0])
```

In this example, the `MyFrameworkObjectAdapter` implements an adaptor class which bridges the CuTe DSL and the 3rd party framework type `MyFrameworkObject`. The registration is done by just decorating the adaptor with `cutlass.register_jit_arg_adapter` for the customized type. With the registered adaptor, CuTe DSL will automatically use the adaptor to generate the JIT function arguments for `MyFrameworkObject` typed arguments.


# Static vs Dynamic layouts
## Static Layout
When integrating with popular deep learning frameworks, one question is how to deal with the layout of the converted `cute.Tensor`. For example, when converting a `torch.Tensor` to a `cute.Tensor`, the shape of the `torch.Tensor` is honored for the layout of `cute.Tensor`.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(f"tensor.layout: {tensor.layout}")  # Prints tensor layout at compile time
    cute.printf("tensor: {}", tensor)         # Prints tensor values at runtime
```

In this example, we define a JIT function `foo` that takes a `cute.Tensor` as input and prints its layout. Note that Python print is used to print the layout at compile time. This works fine for static layout whose value is known at compile time.

Now let’s try to run the JIT function foo with different shapes of the input `torch.Tensor`.

```python
a = torch.tensor([1, 2, 3], dtype=torch.uint16)
a_pack = from_dlpack(a)
compiled_func = cute.compile(foo, a_pack)
compiled_func(a_pack)
```

Here we first convert a 1D `torch.Tensor` with 3 elements to a `cute.Tensor` using `from_dlpack`. Then we compile the JIT function foo with the converted cute.Tensor and call the compiled function.

```text
tensor.layout: (3):(1)
tensor: raw_ptr(0x00000000079e5100: i16, generic, align<2>) o (3):(1) = (1, 2, 3)
```

It prints `(3):(1)` for the layout because the converted `cute.Tensor` has a static layout with shape `(3)` which is the shape of the `a`.

Now if we call the compiled function with a different shape of the input `torch.Tensor`, it would result in an unexpected result at runtime due to the mismatch of the type since `compiled_func` expects a `cute.Tensor` with layout `(3):(1)` while `b` has shape `(5)`.

```python
b = torch.tensor([11, 12, 13, 14, 15], dtype=torch.uint16)
b_pack = from_dlpack(b)
compiled_func(b_pack)  # ❌ This results in an unexpected result at runtime due to type mismatch
```

Following is the output which is unexpected due to the type mismatch.

```text
tensor: raw_ptr(0x00000000344804c0: i16, generic, align<2>) o (3):(1) = (11, 12, 13)
```

To fix that, we would have to trigger another code generation and compilation for the new shape for `b`.

```python
compiled_func_2 = cute.compile(foo, b_pack)  # This would trigger another compilation
compiled_func_2(b_pack)                      # ✅ Now this works fine
```

As shown in the example above, with the newly compiled `compiled_func_2`, we can pass in `b_pack` to the compiled JIT function `compiled_func_2`.

```text
tensor.layout: (5):(1)
tensor: raw_ptr(0x0000000034bb2840:: i16, generic, align<2>) o (5):(1) = (11, 12, 13, 14, 15)
```

Now it recompiles and prints the values of `b` correctly.

It’s obvoius that we need distinct codes generated and compiled for different static layout. In this case, one for layout `(3):(1)` and the other for layout `(5):(1)`.

## Dynamic Layout
In order to avoid generating and compiling multiple times for different shapes of the input `torch.Tensor`, CuTe DSL provides a way to generate and compile JIT function with dynamic layout.

To get dyanmic layout of the `cute.Tensor`, a `torch.Tensor` object can be passed into the JIT function directly which instructs CuTe DSL to call `cute.mark_layout_dynamic` automatically on the converted `cute.Tensor` per the leading dimension of the layout.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor):
    print(tensor.layout)  # Prints (?,?):(?,1) for dynamic layout

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint16)
compiled_func = cute.compile(foo, a)
compiled_func(a)

b = torch.tensor([[11, 12], [13, 14], [15, 16]], dtype=torch.uint16)
compiled_func(b)  # Reuse the same compiled function for different shape
```

In the example above, a single compilation of the JIT function `foo` is reused for different shapes of the input `torch.Tensor`. This is possible because the converted `cute.Tensor` has a dynamic layout `(?,?):(?,1)` which is compatible with the shape of the input `torch.Tensor` of both calls.

Alternatively, for compact layout, `cute.mark_compact_shape_dynamic` can be called for a finer-grained control to specify the mode of the layout for dynamic and the divisibility constraint for the dynamic dimension.

## Static Layout vs. Dynamic Layout
Per the previous sections, we have seen that static layout leads to distinct JIT code generations while dynamic layout leads to a single compilation for different shapes.

That said, creating JIT function with static layout is useful when the use cases targeting input data with fixed shapes. Since more information is available at compile time, the compiler would be able to kick in optimizations that otherwise would not be possible for the code generated for dynamic layout.

On the other hand, dynamic layout would be more flexible for the cases where the input data has varying shapes. This provides more scalability of the generated code to deal with varying input data of different shapes.

## Programming with Static and Dynamic Layout
CuTe DSL provides intuitive way to program with static and dynamic layout in the codes.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    print(cute.size(tensor))  # Prints 3 for the 1st call
                              # Prints ? for the 2nd call
    if cute.size(tensor) > x:
        cute.printf("tensor[2]: {}", tensor[2])
    else:
        cute.printf("tensor size <= {}", x)

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

In this example, the JIT function `foo` is compiled with a static layout `(3):(1)` for the first call, which means the size of the tensor is known at compile time. CuTe DSL makes good use of this and automatically handles the if condition at the compile time. Hence the generated codes are efficient without the if condition at all.

For the second call, the JIT function foo is compiled with a dynamic layout `(?):(1)` hence the tensor size is only evaluated at runtime. CuTe DSL automatically generates the code to handle the dynamic layout and the if condition at runtime.

The same applies to loop as well:

```python
@cute.jit
def foo(tensor, x: cutlass.Constexpr[int]):
    for i in range(cute.size(tensor)):
        cute.printf("tensor[{}]: {}", i, tensor[i])

a = torch.tensor([1, 2, 3], dtype=torch.uint16)
foo(from_dlpack(a), 3)   # First call with static layout

b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint16)
foo(b, 3)                # Second call with dynamic layout
```

With the static layout in the first call, CuTe DSL is able to fully unroll the loop at compile time. While in the second call, the generated codes will have the loop executed at runtime based on the dynamic layout.

With the single JIT function implementation, CuTe DSL is able to handle control-flow constructs and automatically generate the optimized codes for different cases. This is all possible because CuTe DSL is able to walk the Python AST and convert each control-flow construct it finds accordingly.

# JIT Caching
## Zero Compile and JIT Executor
Zero Compile is a feature that enables explicit kernel compilation on demand through `cute.compile`. When `cute.compile` is called, it compiles the kernel and returns a JIT Executor instance. This JIT Executor instance can be cached and reused directly for subsequent executions without compiling the kernel again.

The JIT Executor is a component that independently executes compiled code. It can be created either through `cute.compile` or implicit compilation. The JIT Executor instance behaves like a callable object to execute the compiled code. Each JIT Executor instance maintains a single compiled host function.

It encompasses all necessary execution components:

- Host function pointer and its MLIR execution engine
- CUDA modules (optional)
- Argument specifications defining how Python arguments are converted to C ABI-compatible types. Note that arguments with the `cutlass.Constexpr` hint are excluded from argument specifications since they are evaluated at compile time rather than runtime.

For example, in the following code, `print_result` is a `cutlass.Constexpr` value that is NOT evaluated at runtime:

```python
import cutlass.cute as cute

@cute.jit
def add(a, b, print_result: cutlass.Constexpr):
   if print_result:
      cute.printf("Result: %d\n", a + b)
   return a + b

jit_executor = cute.compile(add, 1, 2, True)

jit_executor(1, 2) # output: ``Result: 3``
```

The JIT Executor ensures all components are properly initialized and loaded after compilation.

For example, all CUDA modules are loaded (via `cuModuleLoad`) and kernel function pointers are extracted (via `cuModuleGetFunction`).

When calling a JIT Executor instance, it:

- Parses Python runtime arguments and converts them to C ABI-compatible types according to argument specifications
- Invokes the host function with the converted arguments

### Custom Caching with `cute.compile`
`cute.compile` bypasses caching in CuTe DSL and always performs compilation, returning a fixed JIT Executor instance. This allows implementing custom caching strategies as shown below:

```python
@cute.jit
def add(b):
   return a + b

# Define a custom cache
custom_cache = {}

a = 1
compiled_add_1 = cute.compile(add, 2)
custom_cache[1] = compiled_add_1
compiled_add_1(2) # result = 3

a = 2
compiled_add_2 = cute.compile(add, 2)
custom_cache[2] = compiled_add_2
compiled_add_2(2) # result = 4

# Use the custom cache
custom_cache[1](2) # result = 3
custom_cache[2](2) # result = 4
```

## Cache in CuTe DSL
By default, cache in CuTe DSL is implicitly enabled to avoid recompilation when kernels are called repeatedly without changes.

The cache is implemented as a map storing compiled JIT Executor instances within CuTe DSL.

The cache key combines hashes of:

- MLIR bytecode of the MLIR program generated by CuTe DSL
- All CuTe DSL Python source files
- All CuTe DSL shared libraries
- All CuTe DSL environment variables

The cache value is a compiled JIT Executor instance.

On a cache hit, compilation is skipped and the cached JIT Executor instance is reused.

On a cache miss, the kernel is compiled and the new JIT Executor instance is stored in the cache.

Here is an example demonstrating automatic caching of the `add` kernel:

```python
# Global variable
a = 1

@cute.jit
def add(b):
   return a + b

# Cache is empty at beginning

# First call: cache miss triggers compilation
result = add(2) # result = 3
# Cache now has one instance

# Second call: cache hit reuses cached JIT Executor
result = add(2) # result = 3

a = 2
# Third call: cache miss due to changed IR code triggers recompilation
result = add(2) # result = 4
# Cache now has two instances
```

The cache can be serialized to files for subsequent runs. After serialization, compiled MLIR bytecode is stored in file. The cache directory is `/tmp/{current_user}/cutlass_python_cache`. During compilation, the cache loads the corresponding kernel from file (if it exists) into memory as needed, and after compilation, it saves any newly compiled executables back to file.

Note that for efficiency, the default cache directory is located in a temporary folder. However, this location is not persistent, it may be cleared by the system (for example, during a reboot or disk space cleanup). If you wish to preserve the cache across sessions, set the `CUTE_DSL_CACHE_DIR` environment variable to point to a persistent directory.

The following environment variables control file caching:

```shell
# Disable file caching while keeping in-memory cache available, defaults to False.
export CUTE_DSL_DISABLE_FILE_CACHING=True

# Cache directory, defaults to /tmp/{current_user}/cutlass_python_cache.
export CUTE_DSL_CACHE_DIR=/home/user/local_cutlass_python_cache/dense_gemm_cache/
```

### Limitations
The intention of caching is to reduce the host launch overhead before each execution. As above example shows, the consistency between the original Python code and the MLIR program is hard to maintain because of the impact of dynamic factors such as global variables. Therefore, the MLIR program MUST always be generated to verify that the kernel content matches what was previously built.

For optimal host launch latency, we recommend using above custom caching method with `cute.compile`.

# JIT Compilation Options
## JIT Compilation Options Overview
When compiling a JIT function using CuTe DSL, you may want to control various aspects of the compilation process, such as optimization level, or debugging flags. CuTe DSL provides a flexible interface for specifying these compilation options when invoking `cute.compile`.

Compilation options allow you to customize how your JIT-compiled functions are built and executed. This can be useful for:

- Enabling or disabling specific compiler optimizations
- Generating debug information for troubleshooting

These options can be passed as keyword arguments to `cute.compile` or set globally for all JIT compilations. The available options and their effects are described in the following sections, along with usage examples to help you get started.

The CuTe DSL provides multiple ways to specify compilation options - either by specifying additional arguments to `cute.compile` or by using a more Pythonic approach with separate Python types for `cute.compile`.

## `cute.compile` Compilation Options as strings
You can provide additional compilation options as a string when calling `cute.compile`. The CuTe DSL uses `argparse` to parse these options and will raise an error if any invalid options are specified.


| Option               | Description                                                                                                                   | Default                           | Type |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------- | ---- |
| `opt-level`          | Optimization level of compilation. The higher the level, the more optimizations are applied. The valid value range is [0, 3]. | 3 (highest level of optimization) | int  |
| `enable-assertions`  | Enable host and device code assertions.                                                                                       | False                             | bool |
| `keep-cubin`         | Keep the generated CUBIN file.                                                                                                | False                             | bool |
| `keep-ptx`           | Keep the generated PTX file.                                                                                                  | False                             | bool |
| `ptxas-options`      | The options to pass to the PTX Compiler library.                                                                              | ""                                | str  |
| `generate-line-info` | Generate line information for debugging.                                                                                      | False                             | bool |
| `gpu-arch`           | The GPU architecture to compile for.                                                                                          | ""                                | str  |
| `enable-tvm-ffi`     | Enable Apache TVM FFI.                                                                                                        | False                             | bool |

You can use the following code to specify compilation options:

```python
jit_executor_with_opt_level_2 = cute.compile(add, 1, 2, options="--opt-level 2")
jit_executor_with_opt_level_1 = cute.compile(add, 1, 2, options="--opt-level 1")
jit_executor_with_enable_assertions = cute.compile(add, 1, 2, options="--enable-assertions")
jit_executor_with_keep_cubin = cute.compile(add, 1, 2, options="--keep-cubin")
jit_executor_with_keep_ptx = cute.compile(add, 1, 2, options="--keep-ptx")
jit_executor_with_ptxas_options = cute.compile(add, 1, 2, options="--ptxas-options '--opt-level=2'")
```

## `cute.compile` Compilation Options as separate Python types
Alternatively, you can also use a more Pythonic way to specify compilation options with separate Python types. Compilation options can be programmatically composed using tuple and passed to `cute.compile` separately.

```python
from cutlass.cute import OptLevel, EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX

my_debugging_options = (OptLevel(1), EnableAssertions, GenerateLineInfo, KeepCUBIN, KeepPTX)
compiled_kernel_1 = cute.compile[my_debugging_options](my_kernel_1, ...)
compiled_kernel_2 = cute.compile[my_debugging_options](my_kernel_2, ...)
```

This approach causes invalid options to raise errors immediately, making it much easier to detect typos when specifying multiple options. Notebly, boolean options are automatically converted to True instances of the option type for convenience.

```python
jit_executor_with_opt_level_2 = cute.compile[OptLevel(2)](add, 1, 2)
jit_executor_with_opt_level_1 = cute.compile[OptLevel(1)](add, 1, 2)
jit_executor_with_enable_assertions = cute.compile[EnableAssertions](add, 1, 2)
jit_executor_with_keep_cubin = cute.compile[KeepCUBIN](add, 1, 2)
jit_executor_with_keep_ptx = cute.compile[KeepPTX](add, 1, 2)
jit_executor_with_ptxas_options = cute.compile[PtxasOptions("--opt-level=2")](add, 1, 2)
```

# Integration with Frameworks
In order to facilitate the integration of CUTLASS Python with popular frameworks, we leverage the DLPack protocol and transform tensors originating from these frameworks to CuTe tensors. The present page documents the conventions, the API available to the user, and provide example code snippets for common usage patterns. We also provide a section on how to bypass the DLPack protocol and directly call the JIT function.

## Implicit Conversion
Tensors originating from frameworks supporting the DLPack protocol can be directly provided to a JIT function as a regular parameter. CuTe DSL’s runtime implicitly converts the original tensor to a CuTe tensor with a fully dynamic layout except for the stride element corresponding to the leading dimension. The example below demonstrates this use case.

```python
import torch
import cutlass.cute as cute

@cute.jit
def foo(src):
    """
    The following lines print

    ptr<f32, generic> o (?,?,?):(?,?,1)
    <class 'cutlass.cute.core._Tensor'>
    """
    print(src)
    print(type(src))

a = torch.randn(30, 20, 32, device="cpu")
foo(a)
```

## Explicit conversion using `from_dlpack`
CuTe DSL’s runtime provides an interface for converting DLPack-compatible tensors to CuTe tensors,

```python
b = cute.runtime.from_dlpack(a)
```
where `a` is a tensor supporting the DLPack protocol with the `__dlpack__` and `__dlpack_device__` methods. The resulting CuTe tensor `b` has a fully static layout. This conversion is performed without copying any tensor data, enabling seamless integration with major frameworks. Users can create tensors using NumPy, PyTorch, etc. and directly feed them into JIT functions writtnen using CuTe DSL.

The resulting CuTe tensor shares the same underlying memory buffer as the original tensor. This zero-copy approach maximizes performance by eliminating unnecessary data duplication. However, it is important to note that the CuTe tensor’s validity is tied to the lifetime of the original tensor. If the source tensor is destroyed or goes out of scope, the corresponding CuTe tensor becomes invalid since it references the original memory location.

The full signature of from_dlpack is as follows:

```python
def from_dlpack(tensor, assumed_align=None, use_32bit_stride=False):
```

The `assumed_align` integer parameter specifies the alignment of the tensor in unit of bytes. The tensor’s base address must be divisible by `assumed_align`. When not provided explicitly, the alignment is set to the natural alignment of the tensor’s element type. Note that the alignment information is part of the pointer type in the generated IR. Therefore, programs with different alignments have a different IR and identical IRs are required for hitting the kernel caching mechanism of CuTe DSL.

The `use_32bit_stride` parameter determines whether to use 32-bit stride for the tensor’s dynamic stride values. By default, it is set to False (64bit) to ensure that address calculations do not risk overflow. For smaller problem sizes (where `cosize(layout_of_tensor) <= Int32_MAX`), users may set it to True (32bit) to improve performance by reducing register usage and the number of address calculation instructions. When `use_32bit_stride` is set to True, a runtime check is performed to ensure that the layout does not overflow. Please note that this parameter only has an effect when the tensor’s layout is marked as dynamic.

The following code demonstrates how to convert a PyTorch tensor to a CuTe tensor using the from_dlpack function with default parameters.

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)
```

Once converted, we can access the tensor’s information through various attributes. The following list shows the attributes of the converted tensor:
- `tensor.shape`: the tensor’s shape
- `tensor.stride`: the tensor’s stride
- `tensor.memspace`: the tensor’s memory space
- `tensor.element_type`: the tensor’s element data type

```python
import torch
import cutlass
from cutlass.cute.runtime import from_dlpack

x = torch.randn(30, 20, device="cpu")
y = from_dlpack(x)

print(y.shape)        # (30, 20)
print(y.stride)       # (20, 1)
print(y.memspace)     # generic (if torch tensor in on device memory, memspace will be gmem)
print(y.element_type) # Float32
print(y)              # Tensor<0x000000000875f580@generic o (30, 20):(20, 1)>
```

The string format of the resulting CuTe tensor is

```text
Tensor<0x{tensor.data_ptr:016x}@{tensor.memspace} o {tensor.shape}:{tensor.stride}>
```

As can be seen in the example above, `from_dlpack` first results in a tensor with a static layout. To obtain dynamic or mixed static/dynamic layouts after calling `from_dlpack`, the `mark_layout_dynamic` and `mark_compact_shape_dynamic` functions are used and described in the following sections.

### When to Use Explicit Conversion?
The DLPack protocol is a widely used protocol for interoperability between different frameworks. However, there is some associated overhead. Based on our benchmark, it usually takes between 2 to 3 us per call to `from_dlpack`.

Explicit conversion allows for caching the converted CuTe tensors in order to avoid the overhead of repeated calls to `from_dlpack`.

```python
x = torch.randn(30, 20, device="cpu")
if key not in cached_tensors:
    # Do the conversion only for cache misses
    cached_tensors[key] = cute.runtime.from_dlpack(x)
foo(cached_tensors[key])
```

Another use case for explicit conversion is to gain fine-grain control over which modes of a tensor are considered dynamic from the perspective of the generated program.

## Mark the Tensor’s Layout as Dynamic with `mark_layout_dynamic`
After calling this function, all shape modes become dynamic. The stride modes also become dynamic with the following two exceptions:

- the leading dimension’s stride remains fixed at 1;
- stride elements equal to 0 (which indicates broadcasting) are retained.

The full signature of `mark_layout_dynamic` is as follows:

```python
def mark_layout_dynamic(self, leading_dim: int|None = None):
```

The `leading_dim` parameter specifies the leading dimension of the tensor. The leading dimension’s stride is set to 1 unless inconsistent with the layout of the DLPack tensor. For example,
- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if `leading_dim` is specified to be 1, the layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- If `leading_dim` is specified to be 0, a deduction failure error is raised because the stride of dimension 0 is 2 (not 1).

The default value for `leading_dim` is `None`. In such case, the system automatically deduces it from the tensor’s layout using the following logic:

1. If a dimension’s stride is 1, that dimension is marked as the leading dimension.
2. If multiple dimensions satisfy condition 1, an error is thrown indicating deduction failure. Note that after converting a PyTorch tensor to the DLPack format, the stride for dimensions with size 1 are canonicalized to 1. This canonicalization can increase the likelihood of deduction failures. This behavior is specific to PyTorch and does not occur with NumPy for example.
3. If no dimension satisfies condition 1, all strides are marked as dynamic.

For example:
- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the leading dimension is 1. The layout will be marked as `(?,?,?,?):(?,1,?,?)`.
- For a tensor with layout `(1,5,1):(1,1,1)`, if `leading_dim` is not specified, a deduction failure error is raised.
- For a tensor with layout `(2,2):(8,2)`, since no dimension has stride 1, all dimensions are marked as dynamic: `(?,?):(?,?)`.

The leading dimension accepts negative index which means the dimension is counted from the last dimension. For example,
- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, if leading_dim is specified to be -1, the layout will be marked as `(?,?,?,?):(?,?,?,1)`.

The following example demonstrates how to use `mark_layout_dynamic` to specify dynamic tensor layouts.
- `t0` shows the usage of `mark_layout_dynamic` with unspecified `leading_dim` and the automatic deduction of leading dimension.
- `t1` & `t2` shows the usage of `mark_layout_dynamic` with specified `leading_dim`.
- `t3` shows the usage of `mark_layout_dynamic` with no leading dimension.
- `t4` shows the usage of `mark_layout_dynamic` with broadcasted dimensions.
- `t5` demonstrates the deduction failure when the there’re more than one dimensions with stride equals to 1.
- `t6` & `t7` demonstrates incorrect settings for `leading_dim` and expected errors.

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)
# (2,2):(8,2)
c = torch.empty(3, 4)[::2, ::2]
# (3,1,1,5):(5,0,0,1)
d = torch.empty(3, 1, 1, 5).expand(3, 4, 2, 5)

# auto deduce the leading dimension to be 3
t0 = from_dlpack(a).mark_layout_dynamic()
print(t0)
# (?,?,?,?):(?,?,?,1)

t1 = from_dlpack(b).mark_layout_dynamic(leading_dim=0)
print(t2)
# (?,?,?,?,?):(1,?,?,?,?)

t2 = from_dlpack(b).mark_layout_dynamic(leading_dim=2)
print(t3)
# (?,?,?,?,?):(?,?,1,?,?)

t3 = from_dlpack(c).mark_layout_dynamic()
print(t3)
# (?,?):(?,?)

t4 = from_dlpack(d).mark_layout_dynamic()
print(t4)
# (?,?,?,?):(?,0,0,1)

t5 = from_dlpack(b).mark_layout_dynamic()
# Can't decude the leading dimension from layout, please specify the leading_dim explicitly.

t6 = from_dlpack(a).mark_layout_dynamic(leading_dim=1)
# Expected strides[leading_dim] == 1, but got 16

t7 = from_dlpack(b).mark_layout_dynamic(leading_dim=3)
# Expected strides[leading_dim] == 1, but got 4

c = torch.empty(1000000000, 1000000000)
t8 = from_dlpack(c, use_32bit_stride=True).mark_layout_dynamic()
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```

## Mark the Tensor’s Layout as Dynamic with `mark_compact_shape_dynamic`
The `mark_compact_shape_dynamic` function provides fine-grain control over dynamic shapes for compact layouts. The full signature of `mark_compact_shape_dynamic` is as follows:

```python
def mark_compact_shape_dynamic(self, mode: int, stride_order: tuple[int, ...]|None = None, divisibility: int = 1):
```

The `mode` parameter determines which shape dimension becomes dynamic. After calling this function, the specific shape dimension given by `mode` is marked as dynamic immediately. The stride will be updated accordingly. For modes that have a shape of size 1, their stride are canonicalized to 0.

The `stride_order` parameter specifies the ordering of strides in the tensor. It is consistent with `torch.Tensor.dim_order()` and defaults to `None`. The parameter indicates the order of modes (dimensions) if the current layout were to be converted to row-major order. It starts from the outermost to the innermost dimension when reading it from left to right. This parameter must be explicitly set when the stride order cannot be automatically deduced from the tensor’s layout, such as when multiple dimensions have a stride of 1.

For example:
- Layout `(4,2):(1,4)` has a stride_order of `(1,0)` indicates the innermost dimension is 0 `(4:1)`, the outermost dimension is 1 `(2:4)`.
- Layout `(5,3,2,4):(3,1,15,30)` has a stride_order of `(3,2,0,1)` indicates the innermost dimension is 1 `(3:1)`, the outermost dimension is 3 `(4:30)`.

If `stride_order` is not specified, the system automatically deduces it from the tensor’s layout using the following logic:
1. Sort the strides in descending order.
2. If multiple dimensions have a stride of 1, a deduction failure error is raised.

For example:
- For a tensor with layout `(2,2,3,4):(2,1,4,12)`, the deduced `stride_order` is `[3,2,0,1]`.
- For a tensor with layout `(1,5,1):(1,1,1)`, `stride_order`’s deduction fails because all dimensions have an identical stride of 1, making it impossible to determine the correct ordering.

If `stride_order` is specified, the system validates that the order is consistent with the tensor’s layout.

The `divisibility` parameter specifies the divisibility of the dynamic shape. It could be used to represent the assumption alignment of the input. Defaults to 1.

Note that this API is only available for compact tensors. For non-compact tensors, we can use `cute.assume` to attach divisibility information to a specific shape mode in a host JIT function, as demonstrated in the following example:

```python
@cute.jit
def foo(a: cute.Tensor):
    new_shape = a.shape
    # use cute.assume to set shape of mode=0 with divisibility=16
    new_shape[0] = cute.assume(new_shape[0], 16)
    new_layout = cute.make_layout(new_shape, stride=a.stride)
    new_a = cute.make_tensor(a.iterator, new_layout)
```

The following example demonstrates how to use `mark_compact_shape_dynamic` to specify dynamic tensor layouts.
- `t0` & `t1` show the usage of `mark_compact_shape_dynamic` with unspecified stride_order and different mode and divisibility.
- `t2` shows the usage of consecutive `mark_compact_shape_dynamic` with unspecified stride_order and different mode and divisibility.
- `t3` & `t4` show the usage of `mark_compact_shape_dynamic` with different specified stride_order.
- `t5`, `t6`, `t7`, `t8`, `t9`, `t10`, `t11`, and `t12` demonstrate incorrect settings for parameters and expected errors.

```python
import torch
from cutlass.cute.runtime import from_dlpack

# (8,4,16,2):(2,16,64,1)
a = torch.empty(16, 4, 8, 2).permute(2, 1, 0, 3)
# (1,4,1,32,1):(4,1,4,4,4) => torch tensor when dimension has shape 1, its stride is degenerated to 1,
# resulting in (1,4,1,32,1):(1,1,1,4,1)
# b.dim_order() is (3,2,4,0,1)
b = torch.empty(32, 1, 1, 1, 4).permute(3, 4, 1, 0, 2)

# auto deduce the stride order to be [2,1,0,3]
t0 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=0, divisibility=2
)
# (?{div=2},4,16,2):(2,?{div=4},?{div=16},1)
print(t0)

t1 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
)
# (8,?{div=2},16,2):(2,16,?{div=32},1)
print(t1)

t2 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=1, divisibility=2
).mark_compact_shape_dynamic(
    mode=3, divisibility=2
)
# (8,?{div=2},16,?{div=2}):(?{div=2},?{div=16},?{div=32},1)
print(t2)

t3 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(3, 0, 2, 4, 1)
)
# (1,4,?,32,1):(0,1,4,?{div=4},0)
print(t3)

t4 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=2, divisibility=1, stride_order=(2, 3, 4, 0, 1)
)
# (1,4,?,32,1):(0,1,128,4,0)
print(t4)

t5 = t2.mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# The stride_order is not consistent with the last stride_order

t6 = from_dlpack(a).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3)
)
# The stride_order is not consistent with the deduced stride_order

t7 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4
)
# The layout could not be deduced, please specify the stride_order explicitly

t8 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=30, divisibility=5, stride_order=(3, 0, 2, 4, 1)
)
# Expected mode value to be in range [0, 5), but got 30

t9 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(2, 1, 2, 3, 4)
)
# Expected stride_order to contain all the dimensions of the tensor, but it doesn't contain 0.

t10 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=3, divisibility=5, stride_order=(0, 1, 2, 3, 4, 5)
)
# Expected stride_order to have 5 elements, but got 6.

t11 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=4, stride_order=b.dim_order()
)
# The shape(1) of mode(0) is not divisible by the divisibility(4)

t12 = from_dlpack(b).mark_compact_shape_dynamic(
    mode=0, divisibility=1, stride_order=(2, 1, 3, 0, 4)
)
# The stride_order is not consistent with the layout

c = torch.empty(1000000000, 1000000000)
t13 = from_dlpack(c, use_32bit_stride=True).mark_compact_shape_dynamic(
    mode=0, divisibility=1
)
# Layout in DLTensorWrapper has int32 overflow risk. Please set use_32bit_stride to False.
```

## Leveraging TVM FFI for Faster PyTorch Interop
The latest version of CuTe DSL supports TVM FFI to improve interoperability with PyTorch and other machine learning frameworks. Using TVM FFI provides the following features:

- Faster JIT function invocation.
- Direct acceptance of `torch.Tensor` objects as function arguments.
- Enhanced error handling and kernel validation.
- Seamless integration with multiple programming languages.

## Bypass the DLPack Protocol
In certain scenarios, users may wish to bypass the DLPack protocol and invoke the JIT function directly. This can be accomplished by creating a lightweight JIT wrapper around the existing JIT function, utilizing `cute.ptr` and `cute.make_tensor` to pass pointers and construct tensors directly.

Typical use cases for bypassing DLPack include: 
1. Users want to call the JIT function directly to avoid the overhead introduced by the DLPack protocol. 
2. DLPack canonicalizes the stride of shape-1 dimensions to 1, which may result in incorrect alignment propagation and affect memory access or performance. 
3. DLPack may lack support for some narrow data types.

The following example illustrates how to bypass the DLPack protocol when invoking a JIT function. Assume we have a pre-defined `TensorOpGemm` kernel whose JIT interface expects three arguments of type `cute.Tensor`. To enable direct invocation without DLPack, we first define a JIT wrapper function that accepts `cute.Pointer` types as parameters. Within this wrapper, we use `cute.make_tensor` to construct tensors from the provided pointers, and then call the `TensorOpGemm` kernel as usual.

```python
@cute.jit
def tensor_op_gemm_wrapper(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Int32,
    n: cutlass.Int32,
    k: cutlass.Int32,
    l: cutlass.Int32,
):

    # Assume alignment of shape to call tensorop_gemm example
    m = cute.assume(m, divby=8)
    n = cute.assume(n, divby=8)

    # Torch is row major
    a_layout = cute.make_ordered_layout((m, k, l), order=(0, 1, 2))
    b_layout = cute.make_ordered_layout((n, k, l), order=(0, 1, 2))
    c_layout = cute.make_ordered_layout((m, n, l), order=(1, 0, 2))
    mA = cute.make_tensor(a_ptr, layout=a_layout)
    mB = cute.make_tensor(b_ptr, layout=b_layout)
    mC = cute.make_tensor(c_ptr, layout=c_layout)

    # TensorOpGemm is a pre-defined kernel from our example
    tensor_op_gemm = TensorOpGemm(
        a_ptr.value_type, c_ptr.value_type, cutlass.Float32, (2, 2, 1)
    )

    tensor_op_gemm(mA, mB, mC)
```

To pass a PyTorch tensor to this new JIT wrapper, we retrieve the raw pointer from the PyTorch tensor and create a `cute.Pointer` instance using `cute.make_ptr`. This approach allows us to bypass the DLPack protocol entirely, avoiding its overhead and potential issues with shape-1 dimension handling.

```python
a = torch.randn(
    m, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
b = torch.randn(
    n, k, l, dtype=torch.float16, device="cuda"
).permute(2, 1, 0)
c = torch.randn(
    n, m, l, dtype=torch.float16, device="cuda"
).permute(1, 2, 0)

# from cutlass.cute.runtime import make_ptr
a_ptr = make_ptr(
    cutlass.Float16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
b_ptr = make_ptr(
    cutlass.Float16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
c_ptr = make_ptr(
    cutlass.Float16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32
)
tensor_op_gemm_wrapper(a_ptr, b_ptr, c_ptr, m, n, k, l)
```

# Debugging
## Source Code Correlation
CuTe DSL provides Python code to PTX/SASS correlation to enable the profiling/debugging of generated kernels with debug symbols by generating line info when compiling the kernel.

You can enable that globally via the environment variable CUTE_DSL_LINEINFO=1. Alternative, you can use compilation options to enable that per kernel. Please refer to **JIT Compilation Options** for more details.

## DSL Debugging
CuTe DSL provides built-in logging mechanisms to help you understand the code execution flow and some of the internal state.

### Enabling Logging
CuTe DSL provides environment variables to control logging level:

```shell
# Enable console logging (default: False)
export CUTE_DSL_LOG_TO_CONSOLE=1

# Log to file instead of console (default: False)
export CUTE_DSL_LOG_TO_FILE=my_log.txt

# Control log verbosity (0, 10, 20, 30, 40, 50, default: 10)
export CUTE_DSL_LOG_LEVEL=20
```

### Dump the generated IR
For users familiar with MLIR and compilers, CuTe DSL supports dumping the Intermediate Representation (IR). This helps you verify whether the IR is generated as expected.

```shell
# Dump Generated CuTe IR (default: False)
export CUTE_DSL_PRINT_IR=1

# Keep Generated CuTe IR in a file (default: False)
export CUTE_DSL_KEEP_IR=1
```

### Dump the generated PTX & CUBIN
For users familiar with PTX and SASS, CuTe DSL supports dumping the generated PTX and CUBIN.

```shell
# Dump generated PTX in a .ptx file (default: False)
export CUTE_DSL_KEEP_PTX=1

# Dump generated cubin in a .cubin file (default: False)
export CUTE_DSL_KEEP_CUBIN=1
```

To further get SASS from cubin, users can use nvdisasm (usually installed with CUDA toolkit) to disassemble the cubin.

```shell
nvdisasm your_dsl_code.cubin > your_dsl_code.sass
```

### Access the dumped contents programmatically
For compiled kernels, the generated PTX/CUBIN/IR can be accessed programmatically as well through following attributes:
- `__ptx__`: The generated PTX code of the compiled kernel.
- `__cubin__`: The generated CUBIN data of the compiled kernel.
- `__mlir__`: The generated IR code of the compiled kernel.

```python
compiled_foo = cute.compile(foo, ...)
print(f"PTX: {compiled_foo.__ptx__}")
with open("foo.cubin", "wb") as f:
    f.write(compiled_foo.__cubin__)
```

### Change the dump directory
By default, all dumped files are saved in the current working directory. To specify a different directory for the dumped files, please set the environment variable `CUTE_DSL_DUMP_DIR` accordingly.


## Kernel Functional Debugging
### Using Python’s `print` and CuTe’s `cute.printf`
CuTe DSL programs can use both Python’s native `print()` as well as our own `cute.printf()` to print debug information during kernel generation and execution. They differ in a few key ways:

- Python’s `print()` executes during compile-time only (no effect on the generated kernel) and is typically used for printing static values (e.g. a fully static layouts).
- `cute.printf()` executes at runtime on the GPU itself and changes the PTX being generated. This can be used for printing values of tensors at runtime for diagnostics, but comes at a performance overhead similar to that of printf() in CUDA C.

### Handling Unresponsive/Hung Kernels
When a kernel becomes unresponsive and `SIGINT` (CTRL+C) fails to terminate it, you can follow these steps to forcefully terminate the process:
1. Use CTRL+Z to suspend the unresponsive kernel
2. Execute the following command to terminate the suspended process:

```shell
# Terminate the most recently suspended process
kill -9 $(jobs -p | tail -1)
```

CuTe DSL can also be debugged using standard NVIDIA CUDA tools.

### Using Compute-Sanitizer
For detecting memory errors and race conditions:

```shell
compute-sanitizer --some_options python your_dsl_code.py
```

### Set function name prefix
By default, the function name (host function or kernel function) is automatically generated based on the function name and its parameters. Sometimes you may want to attach some runtime information to the function name to make performance profiling and debugging easier, e.g., the kernel configs or the rank ids. You can assign a name prefix to the name by calling the `set_name_prefix` method on the host function or kernel function.

```python
@cute.kernel
def kernel(arg1, arg2, ...):
    ...
@cute.jit
def launch_kernel():
    kernel.set_name_prefix("your_custom_name_prefix")
    kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
```

For above example, the generated kernel name will be `your_custom_name_prefix_xxx`.

# Guidance for Auto-Tuning
Numerous GEMM kernel code examples are offered within the CuTe DSL codebase. When integrating these kernels into frameworks, auto-tuning becomes essential for achieving optimal performance. This involves selecting the appropriate kernel parameters based on the inputs of real applications. Next, we’ll briefly introduce some tips on how to perform auto-tuning.

The auto-tuning process typically involves the following steps:
1. Define search space
2. Benchmark each configuration and select the kernel with the best performance
3. Enable caching to reduce the tuning cost

The search space defines the valid combinations of kernel parameters that can be used to run the kernels. Different inputs (shapes, data types, etc.) typically require different kernel parameters to achieve optimal performance. The search space is related to the kernel. We take the Blackwell GEMM persistent kernel as an example. The search space is as follows:

- **mma_tiler_mn**: Defines the dimensions of the matrix tile that each Matrix Multiply-Accumulate (MMA) instruction processes in a single operation.
- **cluster_shape_mn**: Specifies the number of CTAs along each dimension within a cluster. Refer Parallel Thread Execution ISA documentation for the possible mma tiler size and cluster shape for different tensor data types.
- **use_2cta_instrs**: Whether to utilize Blackwell’s 2 CTA instructions for MMA/Copy.
- **use_tma_store**: Whether to use Tensor Memory Access (TMA) instructions to store the result back to global memory.

After defining the search space, we could traverse all parameter combinations to find the optimal kernel. The `autotune_gemm` function below demonstrates a simple exhaustive search approach - it iterates through configurations, compiles and benchmarks each kernel, and returns the best performing one. Since kernel compilation incurs overhead, it’s important to cache and reuse compiled kernels to minimize host launch latency. CuTe DSL facilitates this through its separate compilation and execution workflow. As demonstrated in the `autotune_gemm` function (between the begin of cache the compiled GEMM kernel and end of cache the compiled GEMM kernel comments), we can use `cute.compile()` to compile a kernel once, cache the compiled result, and reuse the cached JIT executor for multiple kernel executions. We could maintain a global configuration-to-kernel dictionary (`config_kernel_dict`) to cache the compiled GEMM kernels, where each key (`kernel_cache_key`) uniquely identifies a kernel based on its characteristics. Usually we could use the `{dtype + kernel configs}` as the cached key for GEMM compilation. For example,

```python
kernel_cache_key = f"{ab_dtype}x{c_dtype}x{acc_dtype}x{use_2cta_instrs}x{mma_tiler}x{cluster_shape_mn}x{use_tma_store}"
```

If the input tensor’s layout is static, we should add the shape in the cached key too. Users can customize the `benchmark` function to measure kernel execution time. For stable and reliable performance measurements:

1. Run a few warmup iterations (e.g., 5-10) to stabilize GPU temperature
2. Execute multiple timed iterations (e.g., 100-1000) for statistical significance
3. Use CUDA events and synchronization for precise timing
4. Lock GPU frequencies (SM and memory frequencies) with nvidia-smi
5. Process results by removing outliers and using min/avg statistics as measurements.

This ensures reliable kernel selection through proper benchmarking.

```python
# get the best GEMM kernel for given input tensors
def autotune_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    use_2cta_instrs_list: List[bool] = [True],
    use_tma_store_list: List[bool] = [True],
    mma_tiler_m_list: List[int] = [256],
    mma_tiler_n_list: List[int] = [256],
    cluster_shape_m_list: List[int] = [2],
    cluster_shape_n_list: List[int] = [1],
):
    best_kernel = None
    min_time = float("inf")
    # traverse the search space
    for use_2cta_instrs in use_2cta_instrs_list:
        for use_tma_store in use_tma_store_list:
            for mma_tiler_mn in product(mma_tiler_m_list, mma_tiler_n_list):
                for cluster_shape_mn in product(cluster_shape_m_list, cluster_shape_n_list):
                    acc_dtype = cutlass.Float32
                    hardware_info = cutlass.utils.HardwareInfo()
                    max_active_clusters = hardware_info.get_max_active_clusters(
                        cluster_shape_mn[0] * cluster_shape_mn[1]
                    )
                    # instance a GEMM kernel
                    gemm = PersistentDenseGemmKernel(
                        acc_dtype,
                        use_2cta_instrs,
                        mma_tiler_mn,
                        cluster_shape_mn,
                        use_tma_store,
                    )
                    # begin of cache the compiled GEMM kernel
                    if kernel_cache_key not in config_kernel_dict:
                        # compile gemm kernel
                        compiled_gemm = cute.compile(
                            gemm,
                            a,
                            b,
                            c,
                            max_active_clusters,
                            stream,
                        )
                        config_kernel_dict[kernel_cache_key] = compiled_gemm
                    else:
                        compiled_gemm = config_kernel_dict[kernel_cache_key]
                    # end of cache the compiled GEMM kernel
                    try:
                        # define a benchmark function to measure the execution time of the compiled GEMM kernel
                        cur_time = benchmark(
                            partial(compiled_gemm, a, b, c, stream),
                        )
                    except Exception as e:
                        print(f"Execution error: {e}")
                        cur_time = float("inf")
                    if cur_time < min_time:
                        min_time = cur_time
                        best_kernel = compiled_gemm
    if best_kernel is None:
        raise ValueError("No best kernel found")
    return best_kernel
```

This brute-force approach ensures we could find the optimal parameters, though at the cost of trying every possibilities. For more advanced use cases, users can explore sophisticated optimization techniques like search space pruning and genetic algorithms to reduce tuning overhead and discover better configurations more efficiently.

To further optimize tuning performance, we can utilize caching mechanisms to avoid redundant computations. We could cache the tuning results in a input-to-kernel dictionary (e.g., `input_kernel_dict`). When processing inputs with matching `config_key` values, the cached kernel can be reused directly without re-tuning. The `config_key` is related with the input tensor’s characteristics, such as the shape, data type, etc. The setup of `config_key` is very flexible, users can customize it based on their own application. For instance, if the data type is fixed in users’ application, we could use the input tensor’s shape as the key, i.e., `(m, n, k)`. To further reduce tuning overhead, we could consider using a simplified key like `config_key = (power_of_2(m), power_of_2(n), power_of_2(k))`, where `m`, `n`, and `k` are rounded up to the nearest power of 2. This simplification can significantly reduce the number of unique keys while still maintaining good performance in most cases. However, it’s important to validate that this approximation doesn’t negatively impact performance for your specific use case.

```python
config_key = (m, n, k)
if config_key in input_kernel_dict:
    compiled_gemm = input_kernel_dict[config_key]
else:
    compiled_gemm = autotune_gemm(...)
    input_kernel_dict[config_key] = compiled_gemm
# launch gemm kernel
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

By following the methods above, you can customize your own auto-tuner to find the optimal GEMM kernel configuration for specific matrix dimensions and data types, significantly improving computational performance for models.

# Ahead-of-Time (AOT) Compilation
This guide demonstrates how to use CuTe DSL’s Ahead-of-Time (AOT) compilation features to export compiled kernels for use in production environments.

CuTe DSL Ahead-of-Time (hereinafter referred to as AOT) compilation allows you to:
- Compile once, enable cross-compilation: Write kernels in Python and cross-compile them for multiple GPU architectures.
- Remove JIT overhead: Eliminate compilation delays in production by pre-compiling kernels.
- Flexible integration: Easily integrate compiled kernels into both Python and C/C++ codebases using flexible deployment options.

We provide 2 levels of AOT ABI:
1. Low-Level CuTe ABI: This ABI is expressed using CuTe DSL types and tensors, mirroring the original Python function.
2. High-Level Apache TVM FFI ABI: For interop with various frameworks (e.g., PyTorch, JAX), and offer high-level stable ABI access.

## CuTe ABI AOT Workflow
### Export Interface
The `export_to_c` interface is provided by the `JitCompiledFunction` class. It accepts the following parameters:
- `file_path`: The path to the directory where the header and object files will be saved.
- `file_name`: The base name for the header and object files. The same file name will always overwrite existing files.
- `function_prefix`: The prefix of the function symbol in the generated object file. This should be a unique identifier to avoid symbol conflicts. Users should ensure the function prefix is unique for each exported function. Defaults to the `file_name`.

It generates the following files:
- `{file_path}/{file_name}.h`: A C header file containing API function declarations. This header specifies the runtime function signatures in C, mirroring the original Python function interfaces.
-` {file_path}/{file_name}.o`: A standard object file containing the compiled kernel code. You can link this object file into either a static or shared library. It includes the host entry function, fatbin data, and helper functions such as cuda_init and cuda_load_to_device. Additionally, it embeds metadata for runtime loading and version verification.

Example:

```python
import cutlass.cute as cute
import cutlass.cute.cuda as cuda

@cute.kernel
def print_tensor_kernel(a: cute.Tensor):
    cute.printf("a: {}", a)

@cute.jit
def print_tensor(a: cute.Tensor, stream: cuda.CUstream):
    print_tensor_kernel(a).launch(grid=(1, 1, 1), block=(1, 1, 1), stream=stream)

compiled_func = cute.compile(print_tensor)
# Export compiled functions to object files and headers
compiled_func.export_to_c(file_path="./artifacts", file_name="print_tensor_example", function_prefix="print_tensor")
```

### Loading in Python
Load pre-compiled object files or shared libraries into Python for execution.

```python
import cutlass.cute as cute
import torch
from cutlass.cute import from_dlpack
import cutlass.cute.cuda as cuda

# Load module from object file
module = cute.runtime.load_module("./artifacts/print_tensor_example.o")
# or
module = cute.runtime.load_module("./artifacts/libprint_tensor_example.so")

# Prepare data
a = torch.arange(160, dtype=torch.float32, device="cuda").reshape(16, 10)
a_cute = from_dlpack(a).mark_layout_dynamic()
stream = cuda.CUstream(0)

# Call the function (no JIT compilation needed!)
module.print_tensor(a_cute, stream=stream)

# This will fail because 'non_existing_api' was not exported:
# module.non_existing_api()
```

### C++ Integration with Static Linking
Integrate compiled kernels directly into your C++ executable during the build process. The generated header file supplies the necessary API for loading the module and invoking the function.

```c++
#include "print_tensor_example.h"
#include <cuda_runtime.h>

void run_print_tensor() {
    // Prepare tensor, the tensor declaration is in the header file
    print_tensor_Tensor_a_t tensor_a;
    tensor_a.data = nullptr; // GPU memory is set to nullptr.
    // Set dynamic shapes and strides
    tensor_a.dynamic_shapes[0] = 32;
    tensor_a.dynamic_shapes[1] = 16;
    tensor_a.dynamic_strides[0] = 16;

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Load module before calling the kernel
    print_tensor_Kernel_Module_t module;
    print_tensor_Kernel_Module_Load(&module);

    // Call the kernel; the kernel wrapper function is defined in the header file
    cute_dsl_print_tensor_wrapper(&module, &tensor_a, stream);

    // Cleanup
    print_tensor_Kernel_Module_Unload(&module);
    cudaStreamDestroy(stream);
}
```

The `print_tensor_example.h` header file is generated by the `export_to_c` interface. It includes:
- The `print_tensor_Kernel_Module_t` type: Represents the kernel module.
- The `print_tensor_Tensor_a_t` type: A tensor-specific type that defines the ABI for a particular CuTe tensor.
- The `cute_dsl_print_tensor_wrapper` function: The user-facing entry point to invoke the kernel.

The compilation of the C++ executable requires the `libcuda_dialect_runtime.so` or `libcuda_dialect_runtime_static.a` library which is involved in `<wheel_install_path>/lib`, along with the CUDA driver and runtime libraries, to function properly.

### C++ Integration with Dynamic Loading
Dynamically load pre-compiled object files or shared libraries at runtime. By including the `CuteDSLRuntime.h` header, you can load the module, look up exported functions, and invoke them.

```c++
#include "CuteDSLRuntime.h"
#include <cuda_runtime.h>

void run_print_tensor() {
    // Load module from shared library
    CuteDSLRT_Module_t *module = nullptr;
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/libprint_tensor_example.so"
    );
    // or
    CuteDSLRT_Error_t err = CuteDSLRT_Module_Load(
        &module,
        "./artifacts/print_tensor_example.o"
    );
    check_error(err);

    // Lookup function
    CuteDSLRT_Function_t *func = nullptr;
    err = CuteDSLRT_Module_Get_Function(&func, module, "print_tensor");
    check_error(err);

    // Prepare arguments, matching the argument type defined in the header file
    typedef struct {
        void *data;
        int32_t dynamic_shapes[2];
        int64_t dynamic_strides[1];
    } print_tensor_Tensor_a_t;

    print_tensor_Tensor_a_t tensor_a;
    tensor_a.data = nullptr;
    tensor_a.dynamic_shapes[0] = 32;
    tensor_a.dynamic_shapes[1] = 16;
    tensor_a.dynamic_strides[0] = 16;

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Call the function; the runtime function accepts packed arguments, refer to the wrapper in the header file
    int ret;
    void* args[] = {&tensor_a, &stream, &ret};
    err = CuteDSLRT_Function_Run(func, args, 3);
    check_error(err);
    cudaStreamSynchronize(stream);

    // Cleanup
    CuteDSLRT_Module_Destroy(module);
    cudaStreamDestroy(stream);
}
```

The `CuteDSLRuntime.h` header file can be found in `<wheel_install_path>/include.` It includes:
- The `CuteDSLRT_Error_t` type: Indicates error status.
- The `CuteDSLRT_Module_Load` function: Loads the module.
- The `CuteDSLRT_Module_Get_Function` function: Gets a function from the loaded module. The runtime API will load the CUDA module for kernel execution.
- The `CuteDSLRT_Function_Run` function: Runs the function.
- The `CuteDSLRT_Module_Destroy` function: Destroys the module.

The compilation of the C++ executable requires the `libcute_dsl_runtime.so` library which is involved in `<wheel_install_path>/lib`, along with the CUDA driver and runtime libraries, to function properly.

## Supported Argument Types
CuTe DSL supports the following argument types:

- `cute.Tensor`
- `cute.Shape` / `cute.Coord` / `cute.Tile` / `cute.IntTuple` / `cute.Stride`
- `cuda.CUstream`
- `cutlass.Int8` / `cutlass.Int16` / `cutlass.Int32` / `cutlass.Int64` / `cutlass.Boolean`
- `cutlass.Uint8` / `cutlass.Uint16` / `cutlass.Uint32` / `cutlass.Uint64`
- `cutlass.Float32` / `cutlass.TFloat32` / `cutlass.Float64` / `cutlass.Float16`

Note that:
1. `cute.Tensor` is a dynamic tensor type that only contains dynamic shapes and strides in its ABI representation. As a result, different compilations may produce different tensor ABIs. This is why declarations for each tensor type are included in the generated header file.
2. strides in `cute.Tensor` are determined by the `use_32bit_strides` compile argument. When `use_32bit_strides` is set to `True`, the strides are 32-bit; when set to False, they are 64-bit.
3. Currently, custom types are not supported for AOT compilation.

## Object File Compatibility Issues
The object file generated by CuTe DSL depends on the CUDA runtime library. Therefore, ensure that the version of the CUDA runtime/toolkit library matches the version used by CuTe DSL. Otherwise, ABI compatibility with the CUDA runtime cannot be guaranteed.

When using C++ static linking integration, compatibility is assured because the header and object files are generated together and guaranteed to match.

For C++ dynamic loading integration and Python loading, the binary file is loaded at runtime. To ensure compatibility, version information is embedded in the metadata of the generated binary file. At runtime, this version information is checked, and if it does not match the expected version, the binary file will be rejected.
