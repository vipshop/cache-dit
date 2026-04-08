---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html
---

# 5.5. Floating-Point Computation

## 5.5.1. Floating-Point Introduction

Since the adoption of the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) for Binary Floating-Point Arithmetic in 1985, virtually all mainstream computing systems, including NVIDIA’s CUDA architectures, have implemented the standard. The IEEE-754 standard specifies how the results of floating-point arithmetic should be approximated.

To get accurate results and achieve the highest performance with the required precision, it is important to consider many aspects of floating-point behavior. This is particularly important in a heterogeneous computing environment where operations are performed on different types of hardware.

The following sections review the basic properties of floating-point computation and cover Fused Multiply-Add (FMA) operations and the dot product. These examples illustrate how different implementation choices affect accuracy.

### 5.5.1.1. Floating-Point Format

Floating-point format and functionality are defined in the [IEEE-754 Standard](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229).

The standard mandates that binary floating-point data be encoded on three fields:

  * **Sign** : one bit to indicate a positive or negative number.

  * **Exponent** : encodes the base 2 exponent offset by a numeric bias.

  * **Significand** (also called _mantissa_ or _fraction_): encodes the fractional value of the number.


[![Floating-Point Encoding](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point-encoding.drawio.png) ](../_images/floating-point-encoding.drawio.png)

The latest IEEE-754 standard defines the encodings and properties of the following binary formats:

  * 16-bit, also known as half-precision, corresponding to the `__half` data type in CUDA.

  * 32-bit, also known as single-precision, corresponding to the `float` data type in C, C++, and CUDA.

  * 64-bit, also known as double-precision, corresponding to the `double` data type in C, C++, and CUDA.

  * 128-bit, also known as quad-precision, corresponding to the `__float128` or `_Float128` data types in CUDA.


These types have the following bit lengths:

[![IEEE-754 Floating-Point Encodings](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point-ieee.drawio.png) ](../_images/floating-point-ieee.drawio.png)

The numeric value associated with floating-point encoding for [normal](#normal-subnormal) values is computed as follows:

\\[(-1)^\mathrm{sign} \times 1.\mathrm{mantissa} \times 2^{\mathrm{exponent} - \mathrm{bias}}\\]

For [subnormal](#normal-subnormal) values, the formula is modified to:

\\[(-1)^\mathrm{sign} \times 0.\mathrm{mantissa} \times 2^{1-\mathrm{bias}}\\]

The exponents are biased by \\(127\\) and \\(1023\\) for single- and double-precision, respectively. The integral part of \\(1.\\) is implicit in the fraction.

For example, the value \\(-192 = (-1)^1 \times 2^7 \times 1.5\\), and is encoded as a negative sign, an exponent of \\(7\\), and a fractional part \\(0.5\\). Hence the exponent \\(7\\) is represented by bit strings with values `7 + 127 = 134 = 10000110` for `float` and `7 + 1023 = 1030 = 10000000110` for `double`. The mantissa `0.5 = 2^-1` is represented by a binary value with `1` in the first position. The binary encodings of \\(-192\\) in single-precision and double-precision are shown in the following figure:

[![Floating-Point Representation for ``-192``](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point-192.drawio.png) ](../_images/floating-point-192.drawio.png)

Since the fraction field uses a limited number of bits, not all real numbers can be represented exactly. For instance, the binary representation of the mathematical value of the fraction \\(2 / 3\\) is `0.10101010...`, which has an infinite number of bits after the binary point. Therefore, \\(2 / 3\\) must be rounded before it can be represented as a floating-point number with limited precision. The rounding rules and modes are specified in IEEE-754. The most frequently used mode is _round-to-nearest-ties-to-even_ , abbreviated round-to-nearest.

### 5.5.1.2. Normal and Subnormal Values

Any floating-point value with an exponent field that is neither all zeros nor all ones is called _normal_.

An important aspect of floating-point values is the wide gap between the smallest representable positive normal number, `FLT_MIN`, and zero. This gap is much wider than the gap between `FLT_MIN` and the second-smallest normal number.

Floating-point _subnormal_ numbers, also called _denormals_ , were introduced to address this issue. A subnormal floating-point value is represented with all bits in the exponent set to zero and at least one bit set in the significand. Subnormals are a required part of the IEEE-754 floating-point standard.

Subnormal numbers allow for a gradual loss of precision as an alternative to sudden rounding toward zero. However, subnormal numbers are computationally more expensive. Therefore, applications that don’t require strict accuracy may choose to avoid them to improve performance. The `nvcc` compiler allows disabling subnormal numbers by setting the `-ftz=true` option (flush-to-zero), which is also included in `--use_fast_math`.

A simplified visualization of the encoding of the smallest normal value and subnormal values in single-precision is shown in the following figure:

[![minimum normal value and subnormal values representations](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point-subnormal.drawio.png) ](../_images/floating-point-subnormal.drawio.png)

where `X` represents both `0` and `1`.

### 5.5.1.3. Special Values

The IEEE-754 standard defines three special values for floating-point numbers:

**Zero:**

  * Mathematical zero.

  * Note that there are two possible representations of floating-point zero: `+0` and `-0`. This differs from the representation of integer zero.

  * `+0 == -0` evaluates to `true`.

  * Zero is encoded with all bits set to `0` in the exponent and significand.


**Infinity:**

  * Floating-point numbers behave according to saturation arithmetic, in which operations that overflow the representable range result in `+Infinity` or `-Infinity`.

  * Infinity is encoded with all bits in the exponent set to `1` and all bits in the significand set to `0`. There are exactly two encodings for infinity values.

  * Arithmetic operations involving infinity and finite nonzero values typically result in infinity. Indeterminate forms such as `Inf * 0.0`, `Inf - Inf`, `Inf / Inf`, and `0.0 / 0.0` result in NaN.


**Not-a-Number (NaN):**

  * NaN is a special symbol that represents an undefined or non-representable value. Common examples are `0.0 / 0.0`, `sqrt(-1.0)`, or `+Inf - Inf`.

  * NaN is encoded with all bits in the exponent set to `1` and any bit pattern in the significand, except for all bits set to 0. There are \\(2^{\mathrm{mantissa} + 1} - 2\\) possible encodings.

  * Any arithmetic operation involving a NaN will result in NaN.

  * Any ordered comparison (`<`, `<=`, `>`, `>=`, `==`) involving a NaN will result in `false`, including `NaN == NaN` (non-reflexive). The unordered comparison `NaN != NaN` returns `true`.

  * NaNs are provided in two forms:

    * Quiet NaNs `qNaN` are used to propagate errors resulting from invalid operations or values. Invalid arithmetic operations generally produce a quiet NaN. They are encoded with the most significant bit of the significand set to `1`.

    * Signaling NaNs `sNaN` are designed to raise an invalid-operation exception. Signaling NaNs are generally explicitly created. They are encoded with the most significant bit of the significand set to `0`.

    * The exact bit patterns for Quiet and Signaling NaNs are implementation-defined. CUDA provides the [cuda::std::numeric_limits<T>::quiet_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/quiet_NaN.html) and [cuda::std::numeric_limits<T>::signaling_NaN](https://en.cppreference.com/w/cpp/types/numeric_limits/signaling_NaN.html) constants to get their special values.


A simplified visualization of the encodings of special values is shown in the following figure:

[![Floating-Point Representation for Infinity and NaN](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point-special-values.drawio.png) ](../_images/floating-point-special-values.drawio.png)

where `X` represents both `0` and `1`.

### 5.5.1.4. Associativity

It is important to note that the rules and properties of mathematical arithmetic do not directly apply to floating-point arithmetic due to its limited precision. The example below shows single-precision values `A`, `B`, and `C` and the exact mathematical value of their sum computed using different associativity.

\\[\begin{split}\begin{aligned} A &= 2^{1} \times 1.00000000000000000000001 \\\ B &= 2^{0} \times 1.00000000000000000000001 \\\ C &= 2^{3} \times 1.00000000000000000000001 \\\ (A + B) + C &= 2^{3} \times 1.01100000000000000000001011 \\\ A + (B + C) &= 2^{3} \times 1.01100000000000000000001011 \end{aligned}\end{split}\\]

Mathematically, \\((A + B) + C\\) is equal to \\(A + (B + C)\\).

Let \\(\mathrm{rn}(x)\\) denote one rounding step on \\(x\\). Performing the same computations in single-precision floating-point arithmetic in round-to-nearest mode according to IEEE-754, we obtain:

\\[\begin{split}\begin{aligned} A + B &= 2^{1} \times 1.1000000000000000000000110000\ldots \\\ \mathrm{rn}(A+B) &= 2^{1} \times 1.10000000000000000000010 \\\ B + C &= 2^{3} \times 1.0010000000000000000000100100\ldots \\\ \mathrm{rn}(B+C) &= 2^{3} \times 1.00100000000000000000001 \\\ A + B + C &= 2^{3} \times 1.0110000000000000000000101100\ldots \\\ \mathrm{rn}\big(\mathrm{rn}(A+B) + C\big) &= 2^{3} \times 1.01100000000000000000010 \\\ \mathrm{rn}\big(A + \mathrm{rn}(B+C)\big) &= 2^{3} \times 1.01100000000000000000001 \end{aligned}\end{split}\\]

For reference, the exact mathematical results are also computed above. The results computed according to IEEE-754 differ from the exact mathematical results. Additionally, the results corresponding to the sums \\(\mathrm{rn}(\mathrm{rn}(A + B) + C)\\) and \\(\mathrm{rn}(A + \mathrm{rn}(B + C))\\) differ from each other. In this case, \\(\mathrm{rn}(A + \mathrm{rn}(B + C))\\) is closer to the correct mathematical result than \\(\mathrm{rn}(\mathrm{rn}(A + B) + C)\\).

This example shows that seemingly identical computations can produce different results, even when all basic operations comply with IEEE-754.

### 5.5.1.5. Fused Multiply-Add (FMA)

The Fused Multiply-Add (FMA) operation computes the result with only one rounding step. Without the FMA, the result would require two rounding steps: one for multiplication and one for addition. Because the FMA uses only one rounding step, it produces a more accurate result.

The Fused Multiply-Add operation can affect the propagation of NaNs differently than two separate operations. However, FMA NaN handling is not universally identical across all targets. Different implementations with multiple NaN operands may prefer a quiet NaN or propagate one operand’s payload. Additionally, IEEE-754 does not strictly mandate a deterministic payload selection order when multiple NaN operands are present. NaNs may also occur in intermediate computations, for example, \\(\infty \times 0 + 1\\) or \\(1 \times \infty - \infty\\), resulting in an implementation-defined NaN payload.

* * *

For clarity, first consider an example using decimal arithmetic to illustrate how the FMA operation works. We will compute \\(x^2 - 1\\) using five total digits of precision, with four digits after the decimal point.

  * For \\(x = 1.0008\\), the correct mathematical result is \\(x^2 - 1 = 1.60064 \times 10^{-4}\\). The closest number using only four digits after the decimal point is \\(1.6006 \times 10^{-4}\\).

  * The Fused Multiply-Add operation achieves the correct result using only one rounding step \\(\mathrm{rn}(x \times x - 1) = 1.6006 \times 10^{-4}\\).

  * The alternative is to compute the multiply and add steps separately. \\(x^2 = 1.00160064\\) translates to \\(\mathrm{rn}(x \times x) = 1.0016\\). The final result is \\(\mathrm{rn}(\mathrm{rn}(x \times x) -1) = 1.6000 \times 10^{-4}\\).


Rounding the multiply and add separately yields a result that is off by \\(0.00064\\). The corresponding FMA computation is wrong by only \\(0.00004\\) and its result is closest to the correct mathematical answer. The results are summarized below:

\\[\begin{split}\begin{aligned} x &= 1.0008 \\\ x^{2} &= 1.00160064 \\\ x^{2} - 1 &= 1.60064 \times 10^{-4} && \text{true value} \\\ \mathrm{rn}\big(x^{2} - 1\big) &= 1.6006 \times 10^{-4} && \text{fused multiply-add} \\\ \mathrm{rn}\big(x^{2}\big) &= 1.0016 \\\ \mathrm{rn}\big(\mathrm{rn}(x^{2}) - 1\big) &= 1.6000 \times 10^{-4} && \text{multiply, then add} \end{aligned}\end{split}\\]

* * *

Below is another example, using binary single precision values:

\\[\begin{split}\begin{aligned} A &= 2^{0} \times 1.00000000000000000000001 \\\ B &= -2^{0} \times 1.00000000000000000000010 \\\ \mathrm{rn}\big(A \times A + B\big) &= 2^{-46} \times 1.00000000000000000000000 && \text{fused multiply-add} \\\ \mathrm{rn}\big(\mathrm{rn}(A \times A) + B\big) &= 0 && \text{multiply, then add} \end{aligned}\end{split}\\]

  * Computing multiplication and addition separately results in the loss of all bits of precision, yielding \\(0\\).

  * Computing the FMA, on the other hand, provides a result equal to the mathematical value.


Fused multiply-add helps prevent loss of precision during subtractive cancellation. Subtractive cancellation occurs when quantities of similar magnitude with opposite signs are added. In this case, many of the leading bits cancel out, resulting in fewer meaningful bits. The fused multiply-add computes a double-width product during multiplication. Thus, even if subtractive cancellation occurs during addition, there are enough valid bits remaining in the product to yield a precise result.

* * *

**Fused Multiply-Add Support in CUDA:**

CUDA provides the Fused Multiply-Add operation in several ways for both `float` and `double` data types:

  * `x * y + z` when compiled with the flags `-fmad=true` or `--use_fast_math`.

  * `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).

  * `__fmaf_[rd, rn, ru, rz]`, `__fmaf_ieee_[rd, rn, ru, rz]`, and `__fma_[rd, rn, ru, rz]` [CUDA mathematical intrinsic functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html).

  * `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).


* * *

**Fused Multiply-Add Support on Host Platforms:**

Whether to use the fused operation depends on the availability of the operation on the platform and how the code is compiled. It is important to understand the host platform’s support for Fused Multiply-Add when comparing CPU and GPU results.

  * Compiler flags and Fused Multiply-Add hardware support:

    * `-mfma` with [GCC](https://gcc.gnu.org/onlinedocs/gcc/x86-Options.html#index-mmmx) and [Clang](https://clang.llvm.org/docs/UsersManual.html#cmdoption-ffp-contract), `-Mfma` with [NVC++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and `/fp:contract` with [Microsoft Visual Studio](https://learn.microsoft.com/en-us/cpp/preprocessor/fp-contract).

    * x86 platforms with the AVX2 ISA, for example, code compiled with the `-mavx2` flag using GCC or Clang, and `/arch:AVX2` with Microsoft Visual Studio.

    * Arm64 (AArch64) platforms with Advanced SIMD (Neon) ISA.

  * `fma(x, y, z)` and `fmaf(x, y, z)` [C Standard Library functions](https://en.cppreference.com/w/c/numeric/math/fma).

  * `std::fma(x, y, z)` and `std::fmaf(x, y, z)` [C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).

  * `cuda::std::fma(x, y, z)` and `cuda::std::fmaf(x, y, z)` [CUDA C++ Standard Library functions](https://en.cppreference.com/w/cpp/numeric/math/fma.html).


### 5.5.1.6. Dot Product Example

Consider the problem of finding the dot product of two short vectors \\(\overrightarrow{a}\\) and \\(\overrightarrow{b}\\) both with four elements.

\\[\begin{split}\overrightarrow{a} = \begin{bmatrix} a_{1} \\\ a_{2} \\\ a_{3} \\\ a_{4} \end{bmatrix} \qquad \overrightarrow{b} = \begin{bmatrix} b_{1} \\\ b_{2} \\\ b_{3} \\\ b_{4} \end{bmatrix} \qquad \overrightarrow{a} \cdot \overrightarrow{b} = a_{1}b_{1} + a_{2}b_{2} + a_{3}b_{3} + a_{4}b_{4}\end{split}\\]

Although this operation is easy to write down mathematically, implementing it in software involves several alternatives that could lead to slightly different results. All of the strategies presented here use operations that are fully compliant with IEEE-754.

**Example Algorithm 1:** The simplest way to compute the dot product is to use a sequential sum of products, keeping the multiplications and additions separate.

> The final result can be represented as \\(((((a_1 \times b_1) + (a_2 \times b_2)) + (a_3 \times b_3)) + (a_4 \times b_4))\\).

**Example Algorithm 2:** Compute the dot product sequentially using fused multiply-add.

> The final result can be represented as \\((a_4 \times b_4) + ((a_3 \times b_3) + ((a_2 \times b_2) + (a_1 \times b_1 + 0)))\\).

**Example Algorithm 3:** Compute the dot product using a divide-and-conquer strategy. First, we find the dot products of the first and second halves of the vectors. Then, we combine these results using addition. This algorithm is called the “parallel algorithm” because the two subproblems can be computed in parallel since they are independent of each other. However, the algorithm does not require a parallel implementation; it can be implemented with a single thread.

> The final result can be represented as \\(((a_1 \times b_1) + (a_2 \times b_2)) + ((a_3 \times b_3) + (a_4 \times b_4))\\).

### 5.5.1.7. Rounding

The IEEE-754 standard requires support for several operations. These include arithmetic operations such as addition, subtraction, multiplication, division, square root, fused multiply-add, finding the remainder, conversion, scaling, sign, and comparison operations. The results of these operations are guaranteed to be consistent across all implementations of the standard for a given format and rounding mode.

* * *

**Rounding Modes**

The IEEE-754 standard defines four rounding modes: _round-to-nearest_ , _round towards positive_ , _round towards negative_ , and _round towards zero_. CUDA supports all four modes. By default, operations use _round-to-nearest_. [Intrinsic mathematical functions](#mathematical-functions-appendix-intrinsic-functions) can be used to select other rounding modes for individual operations.

Rounding Mode | Interpretation  
---|---  
`rn` | Round to nearest, ties to even  
`rz` | Round towards zero  
`ru` | Round towards \\(\infty\\)  
`rd` | Round towards \\(-\infty\\)  
  
### 5.5.1.8. Notes on Host/Device Computation Accuracy

The accuracy of a floating-point computation result is affected by several factors. This section summarizes important considerations for achieving reliable results in floating-point computations. Some of these aspects have been described in greater detail in previous sections.

These aspects are also important when comparing the results between CPU and GPU. Differences between host and device execution must be interpreted carefully. The presence of differences does not necessarily mean the GPU’s result is incorrect or that there is a problem with the GPU.

**Associativity** :

> Floating-point addition and multiplication in finite precision are not [associative](#associativity) because they often result in mathematical values that cannot be directly represented in the target format, requiring rounding. The order in which these operations are evaluated affects how rounding errors accumulate and can significantly alter the final result.

**Fused Multiply-Add** :

> [Fused Multiply-Add](#fused-multiply-add) computes \\(a \times b + c\\) in a single operation, resulting in greater accuracy and a faster execution time. The accuracy of the final result can be affected by its use. Fused Multiply-Add relies on hardware support and can be enabled either explicitly by calling the related function or implicitly through compiler optimization flags.

**Precision** :

> Increasing the floating-point precision can potentially improve the accuracy of the results. Higher precision reduces loss of significance and enables the representation of a wider range of values. However, higher precision types have lower throughput and consume more registers. Additionally, using them to explicitly store input and output increases memory usage and data movement.

**Compiler Flags and Optimizations** :

> All major compilers provide a variety of optimization flags to control the behavior of floating-point operations.
> 
>   * The highest optimization level for GCC (`-O3`), Clang (`-O3`), nvcc (`-O3`), and Microsoft Visual Studio (`/O2`) does not affect floating-point semantics. However, inlining, loop unrolling, vectorization, and common subexpression elimination could affect the results. The NVC++ compiler also requires the flags `-Kieee -Mnofma` for IEEE-754-compliant semantics.
> 
>   * Refer to the [GCC](https://gcc.gnu.org/wiki/FloatingPointMath), [Clang](https://clang.llvm.org/docs/UsersManual.html#controlling-floating-point-behavior), [Microsoft Visual Studio Compiler](https://learn.microsoft.com/en-us/cpp/build/reference/fp-specify-floating-point-behavior), [nvc++](https://docs.nvidia.com/hpc-sdk/compilers/hpc-compilers-user-guide/index.html#gpu), and [Arm C/C++ compiler](https://developer.arm.com/documentation/101458/2404/Compiler-options?lang=en) documentation for detailed information about options that affect floating-point behavior.
> 
>   * See also the `nvcc` [User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) for detailed descriptions of compiler flags that specifically affect floating-point behavior in CUDA device code: `-ftz`, `-prec-div`, `-prec-sqrt`, `-fmad`, `--use_fast_math`. Besides these floating-point options, it is also important to verify the effects of other compiler optimizations in the context of the user program. Users are encouraged to verify the correctness of their results with extensive testing and compare results obtained with optimizations enabled versus all device code optimizations disabled; see also the `-G` compiler flag.
> 
> 


**Library Implementations** :

> Functions defined outside the IEEE-754 standard are not guaranteed to be correctly rounded and depend on implementation-defined behavior. Therefore, the results may differ across different platforms, including between host, device, and different device architectures.

**Deterministic Results** :

> A deterministic result refers to computing the same bit-wise numerical outputs every time when run with the same inputs under the same specified conditions. Such conditions include:
> 
>   * Hardware dependencies, such as execution on the same CPU processor or GPU device.
> 
>   * Compiler aspects, such as the version of the compiler and the [Compiler Flags and Optimizations](#compiler-flags-and-optimizations).
> 
>   * Run-time conditions that affect the computation, such as [rounding mode](#floating-point-rounding) or environment variables.
> 
>   * Identical inputs to the computation.
> 
>   * Thread configuration, including the number of threads involved in the computation and their organization, for example block and grid size.
> 
>   * The ordering of [arithmetic atomic operations](cpp-language-extensions.html#atomic-functions) depends on hardware scheduling which can vary between runs.
> 
> 


**Taking Advantage of the CUDA Libraries** :

> The [CUDA Math Libraries](https://developer.nvidia.com/gpu-accelerated-libraries), [C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html), and [C++ Standard Library Mathematical functions](https://nvidia.github.io/cccl/libcudacxx/standard_api.html) are designed to boost developer productivity for common functionalities, particularly for floating-point math and numerics-intensive routines. These functionalities provide a consistent high-level interface, are optimized, and are widely tested across platforms and edge cases. Users are encouraged to take full advantage of these libraries and avoid tedious manual reimplementations.

## 5.5.2. Floating-Point Data Types

CUDA supports the Bfloat16, half-, single-, double-, and quad-precision floating-point data types. The following table summarizes the supported floating-point data types in CUDA and their requirements.

Table 43 Supported Floating-Point Types Precision / Name | Data Type | IEEE-754 | Header / Built-in | Requirements  
---|---|---|---|---  
Bfloat16 | `__nv_bfloat16` | ❌ | `<cuda_bf16.h>` | Compute Capability 8.0 or higher.  
Half Precision | `__half` | ✅ | `<cuda_fp16.h>` |   
Single Precision | `float` | ✅ | Built-in |   
Double Precision | `double` | ✅ | Built-in |   
Quad Precision | `__float128`/`_Float128` | ✅ | Built-in `<crt/device_fp128_functions.h>` for mathematical functions | Host compiler support and Compute Capability 10.0 or higher. The C or C++ spelling, `_Float128` and `__float128` respectively, also depends on the host compiler support.  
  
CUDA also supports [TensorFloat-32](https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/) (`TF32`), [microscaling (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) floating-point types, and other [lower precision numerical formats](https://resources.nvidia.com/en-us-blackwell-architecture) that are not intended for general-purpose computation, but rather for specialized purposes involving tensor cores. These include 4-, 6-, and 8-bit floating-point types. See the [CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/structs.html) for more details.

The following figure reports the mantissa and exponent sizes of the supported floating-point data types.

[![Floating-Point Types: Mantissa and Exponent sizes](https://docs.nvidia.com/cuda/cuda-programming-guide/_images/floating-point.drawio.png) ](../_images/floating-point.drawio.png)

The following table reports the ranges of the supported floating-point data types.

Table 44 Supported Floating-Point Types Properties Precision / Name | Largest Value | Smallest Positive Value | Smallest Positive Denormal | Epsilon  
---|---|---|---|---  
Bfloat16 | \\(\approx 2^{128}\\) | \\(\approx 3.39 \cdot 10^{38}\\) | \\(2^{-126}\\) | \\(\approx 1.18 \cdot 10^{-38}\\) | \\(2^{-133}\\) | \\(2^{-7}\\)  
Half Precision | \\(\approx 2^{16}\\) | \\(65504\\) | \\(2^{-14}\\) | \\(\approx 6.1 \cdot 10^{-5}\\) | \\(2^{-24}\\) | \\(2^{-10}\\)  
Single Precision | \\(\approx 2^{128}\\) | \\(\approx 3.40 \cdot 10^{38}\\) | \\(2^{-126}\\) | \\(\approx 1.18 \cdot 10^{-38}\\) | \\(2^{-149}\\) | \\(2^{-23}\\)  
Double Precision | \\(\approx 2^{1024}\\) | \\(\approx 1.8 \cdot 10^{308}\\) | \\(2^{-1022}\\) | \\(\approx 2.22 \cdot 10^{-308}\\) | \\(2^{-1074}\\) | \\(2^{-52}\\)  
Quad Precision | \\(\approx 2^{16384}\\) | \\(\approx 1.19 \cdot 10^{4932}\\) | \\(2^{-16382}\\) | \\(\approx 3.36 \cdot 10^{-4932}\\) | \\(2^{-16494}\\) | \\(2^{-112}\\)  
  
Hint

The [CUDA C++ Standard Library](cpp-language-support.html#cpp-standard-library) provides `cuda::std::numeric_limits` in the `<cuda/std/limits>` header to query the properties and the ranges of the supported floating-point types, including [microscaling formats (MX)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf). See the [C++ reference](https://en.cppreference.com/w/cpp/types/numeric_limits.html) for the list of queryable properties.

**Complex numbers support:**

  * The [CUDA C++ Standard Library](cpp-language-support.html#cpp-standard-library) supports complex numbers with the [cuda::std::complex](https://en.cppreference.com/w/cpp/numeric/complex) type in the `<cuda/std/complex>` header. See also the [libcu++ documentation](https://nvidia.github.io/cccl/libcudacxx/standard_api/numerics_library/complex.html) for more details.

  * CUDA also provides basic support for complex numbers with the `cuComplex` and `cuDoubleComplex` types in the `cuComplex.h` header.


* * *

## 5.5.3. CUDA and IEEE-754 Compliance

All GPU devices follow the [IEEE 754-2019](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229) standard for binary floating-point arithmetic with the following limitations:

  * There is no dynamically configurable rounding mode; however, most of the operations support multiple constant IEEE rounding modes, selectable via specifically named [device intrinsics functions](#mathematical-functions-appendix-intrinsic-functions).

  * There is no mechanism to detect floating-point exceptions, so all operations behave as if IEEE-754 exceptions are always masked. If there is an exceptional event, the default masked response defined by IEEE-754 is delivered. For this reason, although signaling NaN `SNaN` encodings are supported, they are not signaling and are handled as quiet exceptions.

  * Floating-point operations may alter the bit patterns of input NaN payloads. Operations such as absolute value and negation may also not comply with the IEEE 754 requirement, which could result in the sign of a NaN being updated in an implementation-defined manner.


To maximize the portability of results, users are recommended to use the default settings of the `nvcc` compiler’s floating-point options: `-ftz=false`, `-prec-div=true`, and `-prec-sqrt=true`, and not use the `--use_fast_math` option. Note that floating-point expression re-associations and contractions are allowed by default, similarly to the `--fmad=true` option. See also the `nvcc` [User Manual](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math) for a detailed description of these compilation flags.

The IEEE-754 and C/C++ language standards do not explicitly address the conversion of a floating-point value to an integer value in cases where the rounded-to-integer value falls outside the range of the target integer format. The clamping behavior to the range of GPU devices is delineated in the [PTX ISA conversion instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt) section. However, compiler optimizations may leverage the unspecified behavior clause when out-of-range conversion is not invoked directly via a PTX instruction, consequently resulting in undefined behavior and an invalid CUDA program. The CUDA Math documentation issues warnings to users on a per-function/intrinsic basis. For instance, consider the [__double2int_rz()](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html#_CPPv415__double2int_rzd) instruction. This may differ from how host compilers and library implementations behave.

**Atomic Functions Denormals Behavior** :

Atomic operations have the following behavior regarding floating-point denormals, regardless of the setting of the compiler flag `-ftz`:

  * Atomic single-precision floating-point adds on global memory always operate in flush-to-zero mode, namely behave equivalent to PTX `add.rn.ftz.f32` semantic.

  * Atomic single-precision floating-point adds on shared memory always operate with denormal support, namely behave equivalent to PTX `add.rn.f32` semantic.


## 5.5.4. CUDA and C/C++ Compliance

**Floating-Point Exceptions:**

Unlike the host implementation, the mathematical operators and functions supported in device code do not set the global `errno` variable nor report [floating-point exceptions](https://en.cppreference.com/w/cpp/numeric/fenv/FE_exceptions) to indicate errors. Thus, if error diagnostic mechanisms are required, users should implement additional input and output screening for the functions.

**Undefined Behavior with Floating-Point Operations:**

Common conditions of undefined behavior for mathematical operations include:

  * Invalid arguments to mathematical operators and functions:

    * Using an uninitialized floating-point variable.

    * Using a floating-point variable outside its lifetime.

    * Signed integer overflow.

    * Dereferencing an invalid pointer.

  * Floating-point specific undefined behavior:

    * Converting a floating-point value to an integer type for which the result is not representable is undefined behavior. This also includes NaN and infinity.


Users are responsible for ensuring the validity of a CUDA program. Invalid arguments may result in undefined behavior and be subject to compiler optimizations.

Contrary to integer division by zero, floating-point division by zero is not undefined behavior and not subject to compiler optimizations; rather, it is implementation-specific behavior. C++ implementations that conform to [IEC-60559](https://en.cppreference.com/w/cpp/types/numeric_limits/is_iec559.html) (IEEE-754), including CUDA, produce infinity. Note that invalid floating-point operations produce NaN and should not be misinterpreted as undefined behavior. Examples include zero divided by zero and infinity divided by infinity.

**Floating-Point Literals Portability:**

Both C and C++ allow for the representation of floating-point values in either decimal or hexadecimal notation. Hexadecimal floating-point literals, which are supported in [C99](https://en.cppreference.com/w/c/language/floating_constant.html) and [C++17](https://en.cppreference.com/w/cpp/language/floating_literal.html), denote a real value in scientific notation that can be precisely expressed in base-2. However, this does not guarantee that the literal will map to an actual value stored in a target variable (see the next paragraph). Conversely, a decimal floating-point literal may represent a numeric value that cannot be expressed in base-2.

According to the [C++ standard rules](https://eel.is/c++draft/lex.fcon#3), hexadecimal and decimal floating-point literals are rounded to the nearest representable value, larger or smaller, chosen in an implementation-defined manner. This rounding behavior may differ between the host and the device.
    
    
    float f1 = 0.5f;    // 0.5, '0.5f' is a decimal floating-point literal
    float f2 = 0x1p-1f; // 0.5, '0x1p-1f' is a hexadecimal floating-point literal
    float f3 = 0.1f;
    // f1, f2 are represented as 0 01111110 00000000000000000000000
    // f3     is represented as  0 01111011 10011001100110011001101
    

The run-time and compile-time evaluations of the same floating-point expression are subject to the following portability issues:

  * The run-time evaluation of a floating-point expression may be affected by the selected rounding mode, floating-point contraction (FMA) and reassociation compiler settings, as well as floating-point exceptions. Note that CUDA does not support floating-point exceptions and the [rounding mode](#floating-point-rounding) is set to _round-to-nearest-ties-to-even_ by default. Other rounding modes can be selected using [intrinsic functions](#mathematical-functions-appendix-intrinsic-functions).

  * The compiler may use a higher-precision internal representation for constant expressions.

  * The compiler may perform optimizations, such as constant folding, constant propagation, and common subexpression elimination, which can lead to a different final value or comparison result.


**C Standard Math Library Notes:**

The host implementations of common mathematical functions are mapped to [C Standard Math Library functions](https://en.cppreference.com/w/c/header/math.html) in a platform-specific way. These functions are provided by the host compiler and the respective host `libm`, if available.

  * Functions not available from the host compilers are implemented in the `crt/math_functions.h` header file. For example, `erfinv()` is implemented there.

  * Less common functions, such as `rhypot()` and `cyl_bessel_i0()`, are only available in the device code.


As previously mentioned, the host and device implementations of mathematical functions are independent. For more details on the behavior of these functions, please refer to the host implementation’s documentation.

* * *

## 5.5.5. Floating-Point Functionality Exposure

The mathematical functions supported by CUDA are exposed through the following methods:

[Built-in C/C++ language arithmetic operators](#builtin-math-operators):

  * `x + y`, `x - y`, `x * y`, `x / y`, `x++`, `x--`, `x += y`, `x -= y`, `x *= y`, `x /= y`.

  * Support single-, double-, and quad-precision types, `float`, `double`, and `__float128/_Float128` respectively.

    * `__half` and `__nv_bfloat16` types are also supported by including the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.

    * `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table.

  * They are available in both host and device code.

  * Their behavior is affected by the `nvcc` [optimization flags](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#use-fast-math-use-fast-math).


[CUDA C++ Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions):

  * Expose the full set of C++ `<cmath>` [header functions](https://en.cppreference.com/w/cpp/header/cmath) through the `<cuda/std/cmath>` header and the `cuda::std::` namespace.

  * Support IEEE-754 standard floating-point types, `__half`, `float`, `double`, `__float128`, as well as Bfloat16 `__nv_bfloat16`.

    * `__float128` support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table.

  * They are available in both host and device code.

  * They often rely on the [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html). Therefore, there could be different levels of accuracy between the host and device code.

  * Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).

  * A subset of functionalities is also supported in constant expressions, such as `constexpr` functions, in accordance with the C++23 and C++26 standard specifications.


[CUDA C Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

  * Expose a subset of the C `<math.h>` [header functions](https://en.cppreference.com/w/c/header/math.html).

  * Support single and double-precision types, `float` and `double` respectively.

    * They are available in both host and device code.

    * They don’t require additional headers.

    * Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).

  * A subset of the `<math.h>` header functionalities is also available for `__half`, `__nv_bfloat16`, and `__float128/_Float128` types. These functions have names that resemble those of the C Standard Library.

    * `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively. Their host and device code availability is defined on a per-function basis.

    * `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header and they are only available in device code.

  * They can have a different accuracy between host and device code.


[Non-standard CUDA Mathematical functions](#mathematical-functions-appendix-additional-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

  * Expose mathematical functionalities that are not part of the C/C++ Standard Library.

  * Mainly support single- and double-precision types, `float` and `double` respectively.

    * Their host and device code availability is defined on a per-function basis.

    * They don’t require additional headers.

    * They can have a different accuracy between host and device code.

  * `__nv_bfloat16`, `__half`, `__float128/_Float128` are supported for a limited set of functions.

    * `__half` and `__nv_bfloat16` types require the `<cuda_fp16.h>` and `<cuda_bf16.h>` headers, respectively.

    * `__float128/_Float128` type support relies on the host compiler and device compute capability, see the [Supported Floating-Point Types](#supported-floating-point-types) table. The related functions require the `crt/device_fp128_functions.h` header.

    * They are only available in device code.

  * Their behavior is affected by the `nvcc` [optimization flags](../02-basics/nvcc.html#optimization-options).


[Intrinsic Mathematical functions](#mathematical-functions-appendix-intrinsic-functions) ([CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html)):

  * Support single- and double-precision types, `float` and `double` respectively.

  * They are only available in device code.

  * They are faster but less accurate than the respective [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/index.html).

  * Their behavior is not affected by the `nvcc` [floating-point optimization flags](../02-basics/nvcc.html#optimization-options) `-prec-div=false`, `-prec-sqrt=false`, and `-fmad=true`. The only exception is `-ftz=true`, which is also included in `-use_fast_math`.


Table 45 Summary of Math Functionality Features Functionality | Supported Types | Host | Device | Affected by Floating-Point Optimization Flags   
(only for `float` and `double`)  
---|---|---|---|---  
[Built-in C/C++ language arithmetic operators](#builtin-math-operators) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128/_Float128`, `cuda::std::complex` | ✅ | ✅ | ✅  
[CUDA C++ Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double`, `__half`, `__nv_bfloat16`, `__float128`, `cuda::std::complex` | ✅ | ✅ | ✅  
`__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0`, `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` *****  
[CUDA C Standard Library Mathematical functions](#mathematical-functions-appendix-cxx-standard-functions) | `float`, `double` | ✅ | ✅ | ✅  
`__nv_bfloat16`, `__half` with limited support and similar names | On a per-function basis  
`__float128/_Float128` with limited support and similar names | ❌ | ✅  
[Non-standard CUDA Mathematical functions](#mathematical-functions-appendix-additional-functions) | `float`, `double` | On a per-function basis | ✅  
`__nv_bfloat16`, `__half`, `__float128/_Float128` with limited support | ❌ | ✅  
[Intrinsic functions](#mathematical-functions-appendix-intrinsic-functions) | `float`, `double` | ❌ | ✅ | Only with `-ftz=true`, also included in `-use_fast_math`  
  
***** The [CUDA C++ Standard Library functions](cpp-language-support.html#cpp-standard-library) support queries for small floating-point types, such as [numeric_limits<T>](https://en.cppreference.com/w/cpp/types/numeric_limits.html), [fpclassify()](https://en.cppreference.com/w/cpp/numeric/math/fpclassify), [isfinite()](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html), [isnormal()](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html), [isinf()](https://en.cppreference.com/w/cpp/numeric/math/isinf.html), and [isnan()](https://en.cppreference.com/w/cpp/numeric/math/isnan.html).

The following sections provide accuracy information for some of these functions, when applicable. It uses ULP for quantification. For more information on the definition of the [Unit in the Last Place (ULP)](https://en.wikipedia.org/wiki/Unit_in_the_last_place), please see Jean-Michel Muller’s paper [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf).

* * *

## 5.5.6. Built-In Arithmetic Operators

The built-in C/C++ language operators, such as `x + y`, `x - y`, `x * y`, `x / y`, `x++`, `x--`, and reciprocal `1 / x`, for single-, double-, and quad-precision types comply with the IEEE-754 standard. They guarantee a maximum ULP error of zero using a _round-to-nearest-ties-to-even_ rounding mode. They are available in both host and device code.

The `nvcc` compilation flag `-fmad=true`, also included in `--use_fast_math`, enables contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations and has the following effect on the maximum ULP error for the single-precision type `float`:

  * `x * y + z` → [__fmaf_rn(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff): 0 ULP


The `nvcc` compilation flag `-prec-div=false`, also included in `--use_fast_math`, has the following effect on the maximum ULP error for the division operator `/` for the single-precision type `float`:

  * `x / y` → [__fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__cuda__math__intrinsic__single_1gac996beec34f94f6376d0674a6860e107): 2 ULP

  * `1 / x`: 1 ULP


* * *

## 5.5.7. CUDA C++ Mathematical Standard Library Functions

CUDA provides comprehensive support for [C++ Standard Library mathematical functions](https://en.cppreference.com/w/cpp/header/cmath.html) through the `cuda::std::` namespace. The functionalities are part of the `<cuda/std/cmath>` header. They are available in both host and device code.

The following sections specify the mapping with the [CUDA Math APIs](https://docs.nvidia.com/cuda/cuda-math-api/index.html) and the error bounds of each function when executed on the device.

  * The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the _round-to-nearest ties-to-even_ rounding mode.

  * The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.


### 5.5.7.1. Basic Operations

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for basic operations are available in both host and device code, except for `__float128`.

All the following functions have a maximum ULP error of zero.

Table 46 C++ Mathematical Standard Library Functions   
C Math API Mapping   
**Basic Operations** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
[fabs(x)](https://en.cppreference.com/w/cpp/numeric/math/fabs.html) | \\(|x|\\) | [__habs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__habsK13__nv_bfloat16) | [__habs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__habsK6__half) | [fabsf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fabsff) | [fabs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fabsd) | [__nv_fp128_fabs(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fabsg)  
[fmod(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmod.html) | Remainder of \\(\dfrac{x}{y}\\), computed as \\(x - \mathrm{trunc}\left(\dfrac{x}{y}\right) \cdot y\\) | N/A | N/A | [fmodf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmodfff) | [fmod(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmoddd) | [__nv_fp128_fmod(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmodgg)  
[remainder(x, y)](https://en.cppreference.com/w/cpp/numeric/math/remainder.html) | Remainder of \\(\dfrac{x}{y}\\), computed as \\(x - \mathrm{rint}\left(\dfrac{x}{y}\right) \cdot y\\) | N/A | N/A | [remainderf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410remainderfff) | [remainder(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49remainderdd) | [__nv_fp128_remainder(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv420__nv_fp128_remaindergg)  
[remquo(x, y, iptr)](https://en.cppreference.com/w/cpp/numeric/math/remquo.html) | Remainder and quotient of \\(\dfrac{x}{y}\\) | N/A | N/A | [remquof(x, y, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47remquofffPi) | [remquo(x, y, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46remquoddPi) | N/A  
[fma(x, y, z)](https://en.cppreference.com/w/cpp/numeric/math/fma.html) | \\(x \cdot y + z\\) | [__hfma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__ARITHMETIC.html#_CPPv46__hfmaK13__nv_bfloat16K13__nv_bfloat16K13__nv_bfloat16), device-only | [__hfma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__ARITHMETIC.html#_CPPv46__hfmaK6__halfK6__halfK6__half), device-only | [fmaf(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44fmaffff) | [fma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43fmaddd) | [__nv_fp128_fma(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_fmaggg)  
[fmax(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmax.html) | \\(\max(x, y)\\) | [__hmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hmaxK13__nv_bfloat16K13__nv_bfloat16) | [__hmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hmaxK6__halfK6__half) | [fmaxf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fmaxfff) | [fmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmaxdd) | [__nv_fp128_fmax(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmaxgg)  
[fmin(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fmin.html) | \\(\min(x, y)\\) | [__hmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv46__hminK13__nv_bfloat16K13__nv_bfloat16) | [__hmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv46__hminK6__halfK6__half) | [fminf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fminfff) | [fmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fmindd) | [__nv_fp128_fmin(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fmingg)  
[fdim(x, y)](https://en.cppreference.com/w/cpp/numeric/math/fdim.html) | \\(\max(x-y, 0)\\) | N/A | N/A | [fdimf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45fdimfff) | [fdim(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44fdimdd) | [__nv_fp128_fdim(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_fdimgg)  
[nan(str)](https://en.cppreference.com/w/cpp/numeric/math/nan.html) | NaN value from string representation | N/A | N/A | [nanf(str)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44nanfPKc) | [nan(str)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43nanPKc) | N/A  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.2. Exponential Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for exponential functions are available in both host and device code only for `float` and `double` types.

Table 47 C++ Mathematical Standard Library Functions   
C Math API Mapping and Accuracy (Maximum ULP)   
**Exponential Functions** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
  
[exp(x)](https://en.cppreference.com/w/cpp/numeric/math/exp.html) |   
\\(e^x\\) | [hexp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hexpK13__nv_bfloat16)   
  
0 ULP | [hexp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hexpK6__half)   
  
0 ULP | [expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff)   
  
2 ULP | [exp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43expd)   
  
1 ULP | [__nv_fp128_exp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_expg)   
  
1 ULP  
  
[exp2(x)](https://en.cppreference.com/w/cpp/numeric/math/exp2.html) |   
\\(2^x\\) | [hexp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hexp2K13__nv_bfloat16)   
  
0 ULP | [hexp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hexp2K6__half)   
  
0 ULP | [exp2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45exp2ff)   
  
2 ULP | [exp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44exp2d)   
  
1 ULP | [__nv_fp128_exp2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_exp2g)   
  
1 ULP  
  
[expm1(x)](https://en.cppreference.com/w/cpp/numeric/math/expm1.html) |   
\\(e^x - 1\\) |   
N/A |   
N/A | [expm1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46expm1ff)   
  
1 ULP | [expm1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45expm1d)   
  
1 ULP | [__nv_fp128_expm1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_expm1g)   
  
1 ULP  
  
[log(x)](https://en.cppreference.com/w/cpp/numeric/math/log.html) |   
\\(\ln(x)\\) | [hlog(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hlogK13__nv_bfloat16)   
  
0 ULP | [hlog(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hlogK6__half)   
  
0 ULP | [logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff)   
  
1 ULP | [log(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43logd)   
  
1 ULP | [__nv_fp128_log(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_logg)   
  
1 ULP  
  
[log10(x)](https://en.cppreference.com/w/cpp/numeric/math/log10.html) |   
\\(\log_{10}(x)\\) | [hlog10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hlog10K13__nv_bfloat16)   
  
0 ULP | [hlog10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hlog10K6__half)   
  
0 ULP | [log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff)   
  
2 ULP | [log10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log10d)   
  
1 ULP | [__nv_fp128_log10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log10g)   
  
1 ULP  
  
[log2(x)](https://en.cppreference.com/w/cpp/numeric/math/log2.html) |   
\\(\log_2(x)\\) | [hlog2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hlog2K13__nv_bfloat16)   
  
0 ULP | [hlog2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hlog2K6__half)   
  
0 ULP | [log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff)   
  
1 ULP | [log2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44log2d)   
  
1 ULP | [__nv_fp128_log2(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_log2g)   
  
1 ULP  
  
[log1p(x)](https://en.cppreference.com/w/cpp/numeric/math/log1p.html) |   
\\(\ln(1+x)\\) |   
N/A |   
N/A | [log1pf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log1pff)   
  
1 ULP | [log1p(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45log1pd)   
  
1 ULP | [__nv_fp128_log1p(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_log1pg)   
  
1 ULP  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.3. Power Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for power functions are available in both host and device code only for `float` and `double` types.

Table 48 C++ Mathematical Standard Library Functions   
C Math API Mapping and Accuracy (Maximum ULP)   
**Power Functions** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
  
[pow(x, y)](https://en.cppreference.com/w/cpp/numeric/math/pow.html) |   
\\(x^y\\) |   
N/A |   
N/A | [powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff)   
  
4 ULP | [pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43powdd)   
  
2 ULP | [__nv_fp128_pow(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_powgg)   
  
1 ULP  
  
[sqrt(x)](https://en.cppreference.com/w/cpp/numeric/math/sqrt.html) |   
\\(\sqrt{x}\\) | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hsqrtK13__nv_bfloat16)   
  
0 ULP | [hsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hsqrtK6__half)   
  
0 ULP | [sqrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sqrtff)   
  
▪ 0 ULP   
▪ 1 ULP with `--use_fast_math` | [sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sqrtd)   
  
0 ULP | [__nv_fp128_sqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sqrtg)   
  
0 ULP  
  
[cbrt(x)](https://en.cppreference.com/w/cpp/numeric/math/cbrt.html)   
  
|   
\\(\sqrt[3]{x}\\) |   
N/A |   
N/A | [cbrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45cbrtff)   
  
1 ULP | [cbrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44cbrtd)   
  
1 ULP |   
N/A  
  
[hypot(x, y)](https://en.cppreference.com/w/cpp/numeric/math/hypot.html) |   
\\(\sqrt{x^2 + y^2}\\) |   
N/A |   
N/A | [hypotf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46hypotfff)   
  
3 ULP | [hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45hypotdd)   
  
2 ULP | [__nv_fp128_hypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_hypotgg)   
  
1 ULP  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.4. Trigonometric Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for trigonometric functions are available in both host and device code only for `float` and `double` types.

Table 49 C++ Mathematical Standard Library Functions   
C Math API Mapping and Accuracy (Maximum ULP)   
**Trigonometric Functions** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
  
[sin(x)](https://en.cppreference.com/w/cpp/numeric/math/sin.html) |   
\\(\sin(x)\\) | [hsin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hsinK13__nv_bfloat16)   
  
0 ULP | [hsin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hsinK6__half)   
  
0 ULP | [sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff)   
  
2 ULP | [sin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43sind)   
  
2 ULP | [__nv_fp128_sin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_sing)   
  
1 ULP  
  
[cos(x)](https://en.cppreference.com/w/cpp/numeric/math/cos.html) |   
\\(\cos(x)\\) | [hcos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hcosK13__nv_bfloat16)   
  
0 ULP | [hcos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hcosK6__half)   
  
0 ULP | [cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff)   
  
2 ULP | [cos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43cosd)   
  
2 ULP | [__nv_fp128_cos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_cosg)   
  
1 ULP  
  
[tan(x)](https://en.cppreference.com/w/cpp/numeric/math/tan.html) |   
\\(\tan(x)\\) |   
N/A |   
N/A | [tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff)   
  
4 ULP | [tan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43tand)   
  
2 ULP | [__nv_fp128_tan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv414__nv_fp128_tang)   
  
1 ULP  
  
[asin(x)](https://en.cppreference.com/w/cpp/numeric/math/asin.html) |   
\\(\sin^{-1}(x)\\) |   
N/A |   
N/A | [asinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45asinff)   
  
2 ULP | [asin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44asind)   
  
2 ULP | [__nv_fp128_asin(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_asing)   
  
1 ULP  
  
[acos(x)](https://en.cppreference.com/w/cpp/numeric/math/acos.html) |   
\\(\cos^{-1}(x)\\) |   
N/A |   
N/A | [acosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45acosff)   
  
2 ULP | [acos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44acosd)   
  
2 ULP | [__nv_fp128_acos(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_acosg)   
  
1 ULP  
  
[atan(x)](https://en.cppreference.com/w/cpp/numeric/math/atan.html) |   
\\(\tan^{-1}(x)\\) |   
N/A |   
N/A | [atanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45atanff)   
  
2 ULP | [atan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44atand)   
  
2 ULP | [__nv_fp128_atan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_atang)   
  
1 ULP  
  
[atan2(y, x)](https://en.cppreference.com/w/cpp/numeric/math/atan2.html) |   
\\(\tan^{-1}\left(\dfrac{y}{x}\right)\\) |   
N/A |   
N/A | [atan2f(y, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atan2fff)   
  
3 ULP | [atan2(y, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atan2dd)   
  
2 ULP |   
N/A  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.5. Hyperbolic Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for hyperbolic functions are available in both host and device code only for `float` and `double` types.

Table 50 C++ Mathematical Standard Library Functions   
C Math API Mapping and Accuracy (Maximum ULP)   
**Hyperbolic Functions** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
  
[sinh(x)](https://en.cppreference.com/w/cpp/numeric/math/sinh.html) |   
\\(\sinh(x)\\) |   
N/A |   
N/A | [sinhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45sinhff)   
  
3 ULP | [sinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44sinhd)   
  
2 ULP | [__nv_fp128_sinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_sinhg)   
  
1 ULP  
  
[cosh(x)](https://en.cppreference.com/w/cpp/numeric/math/cosh.html) |   
\\(\cosh(x)\\) |   
N/A |   
N/A | [coshf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45coshff)   
  
2 ULP | [cosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44coshd)   
  
1 ULP | [__nv_fp128_cosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_coshg)   
  
1 ULP  
  
[tanh(x)](https://en.cppreference.com/w/cpp/numeric/math/tanh.html) |   
\\(\tanh(x)\\) | [htanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45htanhK13__nv_bfloat16)   
  
0 ULP | [htanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45htanhK6__half)   
  
0 ULP | [tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff)   
  
2 ULP | [tanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44tanhd)   
  
1 ULP | [__nv_fp128_tanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_tanhg)   
  
1 ULP  
  
[asinh(x)](https://en.cppreference.com/w/cpp/numeric/math/asinh.html) |   
\\(\operatorname{sinh}^{-1}(x)\\) |   
N/A |   
N/A | [asinhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46asinhff)   
  
3 ULP | [asinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45asinhd)   
  
3 ULP | [__nv_fp128_asinh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_asinhg)   
  
1 ULP  
  
[acosh(x)](https://en.cppreference.com/w/cpp/numeric/math/acosh.html) |   
\\(\operatorname{cosh}^{-1}(x)\\) |   
N/A |   
N/A | [acoshf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46acoshff)   
  
4 ULP | [acosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45acoshd)   
  
3 ULP | [__nv_fp128_acosh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_acoshg)   
  
1 ULP  
  
[atanh(x)](https://en.cppreference.com/w/cpp/numeric/math/atanh.html) |   
\\(\operatorname{tanh}^{-1}(x)\\) |   
N/A |   
N/A | [atanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46atanhff)   
  
3 ULP | [atanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45atanhd)   
  
2 ULP | [__nv_fp128_atanh(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_atanhg)   
  
1 ULP  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

### 5.5.7.6. Error and Gamma Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for error and gamma functions are available in both host and device code for `float` and `double` types.

Error and Gamma functions are not natively available for CUDA-extended floating-point types, such as `__half` and `__nv_bfloat16`. In these cases, the functions are emulated by converting to a `float` type and then converting the result back.

Table 51 C++ Mathematical Standard Library Functions   
C Math API Mapping and Accuracy (Maximum ULP)   
**Error and Gamma Functions** `cuda::std` Function | Meaning | `float` | `double`  
---|---|---|---  
  
[erf(x)](https://en.cppreference.com/w/cpp/numeric/math/erf.html) |   
\\(\dfrac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt\\) | [erff(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44erfff)   
  
2 ULP | [erf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv43erfd)   
  
2 ULP  
  
[erfc(x)](https://en.cppreference.com/w/cpp/numeric/math/erfc.html) |   
\\(1 - \mathrm{erf}(x)\\) | [erfcf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45erfcff)   
  
4 ULP | [erfc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44erfcd)   
  
5 ULP  
  
[tgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/tgamma.html) |   
\\(\Gamma(x)\\) | [tgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47tgammaff)   
  
5 ULP | [tgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46tgammad)   
  
10 ULP  
  
[lgamma(x)](https://en.cppreference.com/w/cpp/numeric/math/lgamma.html) |   
\\(\ln |\Gamma(x)|\\) | [lgammaf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lgammaff)   
  
▪ 6 ULP for \\(x \notin [-10.001, -2.264]\\)   
▪ larger otherwise | [lgamma(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lgammad)   
  
▪ 4 ULP for \\(x \notin [-23.0001, -2.2637]\\)   
▪ larger otherwise  
  
### 5.5.7.7. Nearest Integer Floating-Point Operations

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for nearest integer floating-point operations are available in both host and device code only for `float` and `double` types.

All the following functions have a maximum ULP error of zero.

Table 52 C++ Mathematical Standard Library Functions   
C Math API Mapping   
**Nearest Integer Floating-Point Operations** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
[ceil(x)](https://en.cppreference.com/w/cpp/numeric/math/ceil.html) | \\(\lceil x \rceil\\) | [hceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hceilK13__nv_bfloat16) | [hceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hceilK6__half) | [ceilf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45ceilff) | [ceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44ceild) | [__nv_fp128_ceil(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_ceilg)  
[floor(x)](https://en.cppreference.com/w/cpp/numeric/math/floor.html) | \\(\lfloor x \rfloor\\) | [hfloor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hfloorK13__nv_bfloat16) | [hfloor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hfloorK6__half) | [floorf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46floorff) | [floor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45floord) | [__nv_fp128_floor(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_floorg)  
[trunc(x)](https://en.cppreference.com/w/cpp/numeric/math/trunc.html) | Truncate to integer | [htrunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46htruncK13__nv_bfloat16) | [htrunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46htruncK6__half) | [truncf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46truncff) | [trunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45truncd) | [__nv_fp128_trunc(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_truncg)  
[round(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero | N/A | N/A | [roundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46roundff) | [round(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45roundd) | [__nv_fp128_round(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_roundg)  
[nearbyint(x)](https://en.cppreference.com/w/cpp/numeric/math/nearbyint.html) | Round to nearest integer, ties to even | N/A | N/A | [nearbyintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nearbyintff) | [nearbyint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nearbyintd) | N/A  
[rint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even | [hrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv45hrintK13__nv_bfloat16) | [hrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv45hrintK6__half) | [rintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45rintff) | [rint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44rintd) | [__nv_fp128_rint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_rintg)  
[lrint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even (returns `long int`) | N/A | N/A | [lrintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46lrintff) | [lrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45lrintd) | N/A  
[llrint(x)](https://en.cppreference.com/w/cpp/numeric/math/rint.html) | Round to nearest integer, ties to even (returns `long long int`) | N/A | N/A | [llrintf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47llrintff) | [llrint(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46llrintd) | N/A  
[lround(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero (returns `long int`) | N/A | N/A | [lroundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47lroundff) | [lround(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46lroundd) | N/A  
[llround(x)](https://en.cppreference.com/w/cpp/numeric/math/round.html) | Round to nearest integer, ties away from zero (returns `long long int`) | N/A | N/A | [llroundf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48llroundff) | [llround(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47llroundd) | N/A  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16. In these cases, the functions are emulated by converting to a float type and then converting the result back.

**Performance Considerations**

The recommended way to round a single- or double-precision floating-point operand to an integer is to use the functions `rintf()` and `rint()`, not `roundf()` and `round()`. This is because `roundf()` and `round()` map to multiple instructions in device code, whereas `rintf()` and `rint()` map to a single instruction. `truncf()`, `trunc()`, `ceilf()`, `ceil()`, `floorf()`, and `floor()` each map to a single instruction as well.

### 5.5.7.8. Floating-Point Manipulation Functions

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for floating-point manipulation functions are available in both host and device code, except for `__float128`.

Floating-point manipulation functions are not natively available for CUDA-extended floating-point types, such as `__half` and `__nv_bfloat16`. In these cases, the functions are emulated by converting to a `float` type and then converting the result back.

All the following functions have a maximum ULP error of zero.

Table 53 C++ Mathematical Standard Library Functions   
C Math API Mapping   
**Floating-Point Manipulation Functions** `cuda::std` Function | Meaning | `float` | `double` | `__float128`  
---|---|---|---|---  
[frexp(x, exp)](https://en.cppreference.com/w/cpp/numeric/math/frexp.html) | Extract mantissa and exponent | [frexpf(x, exp)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46frexpffPi) | [frexp(x, exp)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45frexpdPi) | [__nv_fp128_frexp(x, nptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_frexpgPi)  
[ldexp(x, n)](https://en.cppreference.com/w/cpp/numeric/math/ldexp.html) | \\(x \cdot 2^{\mathrm{n}}\\) | [ldexpf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ldexpffi) | [ldexp(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ldexpdi) | [__nv_fp128_ldexp(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ldexpgi)  
[modf(x, iptr)](https://en.cppreference.com/w/cpp/numeric/math/modf.html) | Extract integer and fractional parts | [modff(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45modfffPf) | [modf(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44modfdPd) | [__nv_fp128_modf(x, iptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv415__nv_fp128_modfgPg)  
[scalbn(x, n)](https://en.cppreference.com/w/cpp/numeric/math/scalbn.html) | \\(x \cdot 2^n\\) | [scalbnf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47scalbnffi) | [scalbn(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46scalbndi) | N/A  
[scalbln(x, n)](https://en.cppreference.com/w/cpp/numeric/math/scalbn.html) | \\(x \cdot 2^n\\) | [scalblnf(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48scalblnffl) | [scalbln(x, n)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47scalblndl) | N/A  
[ilogb(x)](https://en.cppreference.com/w/cpp/numeric/math/ilogb.html) | \\(\lfloor \log_2(|x|) \rfloor\\) | [ilogbf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46ilogbff) | [ilogb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45ilogbd) | [__nv_fp128_ilogb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_ilogbg)  
[logb(x)](https://en.cppreference.com/w/cpp/numeric/math/logb.html) | \\(\lfloor \log_2(|x|) \rfloor\\) | [logbf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45logbff) | [logb(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44logbd) | N/A  
[nextafter(x, y)](https://en.cppreference.com/w/cpp/numeric/math/nextafter.html) | Next representable value toward \\(y\\) | [nextafterf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv410nextafterfff) | [nextafter(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv49nextafterdd) | N/A  
[copysign(x, y)](https://en.cppreference.com/w/cpp/numeric/math/copysign.html) | Copy sign of \\(y\\) to \\(x\\) | [copysignf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49copysignfff) | [copysign(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48copysigndd) | [__nv_fp128_copysign(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv419__nv_fp128_copysigngg)  
  
### 5.5.7.9. Classification and Comparison

[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/index.html) for classification and comparison functions are available in both host and device code, except for `__float128`.

All the following functions have a maximum ULP error of zero.

Table 54 C++ Mathematical Standard Library Functions   
C Math API Mapping   
**Classification and Comparison Functions** `cuda::std` Function | Meaning | `__nv_bfloat16` | `__half` | `float` | `double` | `__float128`  
---|---|---|---|---|---|---  
[fpclassify(x)](https://en.cppreference.com/w/cpp/numeric/math/fpclassify.html) | Classify \\(x\\) | N/A | N/A | N/A | N/A | N/A  
[isfinite(x)](https://en.cppreference.com/w/cpp/numeric/math/isfinite.html) | Check if \\(x\\) is finite | N/A | N/A | [isfinite(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48isfinitef) | [isfinite(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48isfinited) | N/A  
[isinf(x)](https://en.cppreference.com/w/cpp/numeric/math/isinf.html) | Check if \\(x\\) is infinite | [__hisinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisinfK13__nv_bfloat16) | [__hisinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisinfK6__half) | [isinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isinff) | [isinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isinfd) | N/A  
[isnan(x)](https://en.cppreference.com/w/cpp/numeric/math/isnan.html) | Check if \\(x\\) is NaN | [__hisnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv48__hisnanK13__nv_bfloat16) | [__hisnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv48__hisnanK6__half) | [isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45isnanf) | [isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45isnand) | [__nv_fp128_isnan(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_isnang)  
[isnormal(x)](https://en.cppreference.com/w/cpp/numeric/math/isnormal.html) | Check if \\(x\\) is normal | N/A | N/A | N/A | N/A | N/A  
[signbit(x)](https://en.cppreference.com/w/cpp/numeric/math/signbit.html) | Check if sign bit is set | N/A | N/A | [signbit(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47signbitf) | [signbit(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47signbitd) | N/A  
[isgreater(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isgreater.html) | Check if \\(x > y\\) | [__hgt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgtK13__nv_bfloat16K13__nv_bfloat16) | [__hgt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hgt6__half6__half) | N/A | N/A | N/A  
[isgreaterequal(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isgreaterequal.html) | Check if \\(x \geq y\\) | [__hge(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hgeK13__nv_bfloat16K13__nv_bfloat16) | [__hge(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hge6__half6__half) | N/A | N/A | N/A  
[isless(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isless.html) | Check if \\(x < y\\) | [__hlt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hltK13__nv_bfloat16K13__nv_bfloat16) | [__hlt(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hlt6__half6__half) | N/A | N/A | N/A  
[islessequal(x, y)](https://en.cppreference.com/w/cpp/numeric/math/islessequal.html) | Check if \\(x \leq y\\) | [__hle(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hleK13__nv_bfloat16K13__nv_bfloat16) | [__hle(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hle6__half6__half) | N/A | N/A | N/A  
[islessgreater(x, y)](https://en.cppreference.com/w/cpp/numeric/math/islessgreater.html) | Check if \\(x < y\\) or \\(x > y\\) | [__hne(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__COMPARISON.html#_CPPv45__hneK13__nv_bfloat16K13__nv_bfloat16) | [__hne(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__COMPARISON.html#_CPPv45__hneK6__halfK6__half) | N/A | N/A | N/A  
[isunordered(x, y)](https://en.cppreference.com/w/cpp/numeric/math/isunordered.html) | Check if \\(x\\), \\(y\\), or both are NaN | N/A | N/A | N/A | N/A | [__nv_fp128_isunordered(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv422__nv_fp128_isunorderedgg)  
  
***** Mathematical functions marked with “N/A” are not natively available for CUDA-extended floating-point types, such as __half and __nv_bfloat16.

## 5.5.8. Non-Standard CUDA Mathematical Functions

CUDA provides mathematical functions that are not part of the C/C++ Standard Library and are instead offered as extensions. For single- and double-precision functions, host and device code availability is defined on a per-function basis.

This section specifies the error bounds of each function when executed on the device.

  * The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the _round-to-nearest ties-to-even_ rounding mode.

  * The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.


Table 55 **Non-standard CUDA Mathematical functions**   
`float` and `double`   
Mapping and Accuracy (Maximum ULP) Meaning | `float` | `double`  
---|---|---  
\\(\dfrac{x}{y}\\) | [fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff), device-only   
  
0 ULP, same as `x / y` |   
N/A  
  
\\(10^x\\) | [exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff)   
  
2 ULP | [exp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45exp10d)   
  
1 ULP  
  
\\(\sqrt{x^2 + y^2 + z^2}\\) | [norm3df(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm3dffff), device-only   
  
3 ULP | [norm3d(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm3dddd), device-only   
  
2 ULP  
  
\\(\sqrt{x^2 + y^2 + z^2 + t^2}\\) | [norm4df(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47norm4dfffff), device-only   
  
3 ULP | [norm4d(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46norm4ddddd), device-only   
  
2 ULP  
  
\\(\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}\\) | [normf(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45normfiPKf), device-only   
  
An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off | [norm(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv44normiPKd), device-only   
  
An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off  
\\(\dfrac{1}{\sqrt{x}}\\) | [rsqrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rsqrtff)   
  
2 ULP | [rsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rsqrtd)   
  
1 ULP  
\\(\dfrac{1}{\sqrt[3]{x}}\\) | [rcbrtf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rcbrtff)   
  
1 ULP | [rcbrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rcbrtd)   
  
1 ULP  
\\(\dfrac{1}{\sqrt{x^2 + y^2}}\\) | [rhypotf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47rhypotfff), device-only   
  
2 ULP | [rhypot(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46rhypotdd), device-only   
  
1 ULP  
\\(\dfrac{1}{\sqrt{x^2 + y^2 + z^2}}\\) | [rnorm3df(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm3dffff), device-only   
  
2 ULP | [rnorm3d(x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm3dddd), device-only   
  
1 ULP  
\\(\dfrac{1}{\sqrt{x^2 + y^2 + z^2 + t^2}}\\) | [rnorm4df(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48rnorm4dfffff), device-only   
  
2 ULP | [rnorm4d(x, y, z, t)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47rnorm4ddddd), device-only   
  
1 ULP  
  
\\(\dfrac{1}{\sqrt{\sum_{i=0}^{\mathrm{dim}-1} p_i^{2}}}\\) | [rnormf(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46rnormfiPKf), device-only   
  
An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off | [rnorm(dim, p)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45rnormiPKd), device-only   
  
An error bound cannot be provided because a fast algorithm is used with accuracy loss due to round-off  
  
\\(\cos(\pi x)\\) | [cospif(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46cospiff)   
  
1 ULP | [cospi(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45cospid)   
  
2 ULP  
  
\\(\sin(\pi x)\\) | [sinpif(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46sinpiff)   
  
1 ULP | [sinpi(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45sinpid)   
  
2 ULP  
  
\\(\sin(\pi x), \cos(\pi x)\\) | [sincospif(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv49sincospiffPfPf)   
  
1 ULP | [sincospi(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv48sincospidPdPd)   
  
2 ULP  
  
\\(\Phi(x)\\) | [normcdff(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48normcdfff)   
  
5 ULP | [normcdf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47normcdfd)   
  
5 ULP  
  
\\(\Phi^{-1}(x)\\) | [normcdfinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv411normcdfinvff)   
  
5 ULP | [normcdfinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv410normcdfinvd)   
  
8 ULP  
  
\\(\mathrm{erfc}^{-1}(x)\\) | [erfcinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48erfcinvff)   
  
4 ULP | [erfcinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv47erfcinvd)   
  
6 ULP  
  
\\(e^{x^2}\mathrm{erfc}(x)\\) | [erfcxf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46erfcxff)   
  
4 ULP | [erfcx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv45erfcxd)   
  
4 ULP  
  
\\(\mathrm{erf}^{-1}(x)\\) | [erfinvf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47erfinvff)   
  
2 ULP | [erfinv(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv46erfinvd)   
  
5 ULP  
  
\\(I_0(x)\\) | [cyl_bessel_i0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i0ff), device-only   
  
6 ULP | [cyl_bessel_i0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i0d), device-only   
  
6 ULP  
  
\\(I_1(x)\\) | [cyl_bessel_i1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv414cyl_bessel_i1ff), device-only   
  
6 ULP | [cyl_bessel_i1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv413cyl_bessel_i1d), device-only   
  
6 ULP  
  
\\(J_0(x)\\) | [j0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j0ff)   
  
▪ 9 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\), otherwise | [j0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j0d)   
  
▪ 7 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 5 \cdot 10^{-12}\\), otherwise  
  
\\(J_1(x)\\) | [j1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43j1ff)   
  
▪ 9 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\), otherwise | [j1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42j1d)   
  
▪ 7 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 5 \cdot 10^{-12}\\), otherwise  
  
\\(J_n(x)\\) | [jnf(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43jnfif)   
  
For \\(n = 128\\), the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\) | [jn(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42jnid)   
  
For \\(n = 128\\), the maximum absolute error \\(= 5 \cdot 10^{-12}\\)  
  
\\(Y_0(x)\\) | [y0f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y0ff)   
  
▪ 9 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\), otherwise | [y0(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y0d)   
  
▪ 7 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 5 \cdot 10^{-12}\\), otherwise  
  
\\(Y_1(x)\\) | [y1f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43y1ff)   
  
▪ 9 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\), otherwise | [y1(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42y1d)   
  
▪ 7 ULP for \\(|x| < 8\\)   
▪ the maximum absolute error \\(= 5 \cdot 10^{-12}\\), otherwise  
  
\\(Y_n(x)\\) | [ynf(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv43ynfif)   
  
▪ \\(\lceil 2 + 2.5n \rceil\\) for \\(|x| < n\\)   
▪ the maximum absolute error \\(= 2.2 \cdot 10^{-6}\\), otherwise | [yn(n, x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html#_CPPv42ynid)   
  
For \\(|x| > 1.5n\\), the maximum absolute error \\(= 5 \cdot 10^{-12}\\)  
  
Non-standard CUDA Mathematical functions for `__half`, `__nv_bfloat16`, and `__float128/_Float128` are only available in device code.

Table 56 **Non-standard CUDA Mathematical functions**   
`__nv_bfloat16`, `__half`, `__float128/_Float128`   
Mapping and Accuracy (Maximum ULP) Meaning | `__nv_bfloat16` | `__half` | `__float128/_Float128`  
---|---|---|---  
\\(\dfrac{1}{x}\\) | [hrcp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv44hrcpK13__nv_bfloat16)   
  
0 ULP | [hrcp(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv44hrcpK6__half)   
  
0 ULP |   
N/A  
  
\\(10^x\\) | [hexp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hexp10K13__nv_bfloat16)   
  
0 ULP | [hexp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hexp10K6__half)   
  
0 ULP | [__nv_fp128_exp10(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__QUAD.html#_CPPv416__nv_fp128_exp10g)   
  
1 ULP  
\\(\dfrac{1}{\sqrt{x}}\\) | [hrsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv46hrsqrtK13__nv_bfloat16)   
  
0 ULP | [hrsqrt(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv46hrsqrtK6__half)   
  
0 ULP |   
N/A  
  
\\(\tanh(x)\\) (approximate) | [htanh_approx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____BFLOAT16__FUNCTIONS.html#_CPPv412htanh_approxK13__nv_bfloat16)   
  
1 ULP | [htanh_approx(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH____HALF__FUNCTIONS.html#_CPPv412htanh_approxK6__half)   
  
1 ULP |   
N/A  
  
## 5.5.9. Intrinsic Functions

Intrinsic mathematical functions are faster and less accurate versions of their corresponding [CUDA C Standard Library Mathematical functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html).

  * They have the same name prefixed with `__`, such as `__sinf(x)`.

  * They are only available in device code.

  * They are faster because they map to fewer native instructions.

  * The flag `--use_fast_math` automatically translates the corresponding [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) into intrinsic functions. See the [–use_fast_math Effect](#use-fast-math) section for the full list of affected functions.


### 5.5.9.1. Basic Intrinsic Functions

A subset of mathematical intrinsic functions allow specifying the rounding mode:

  * Functions suffixed with `_rn` operate using the _round to nearest even_ rounding mode.

  * Functions suffixed with `_rz` operate using the _round towards zero_ rounding mode.

  * Functions suffixed with `_ru` operate using the _round up_ (toward positive infinity) rounding mode.

  * Functions suffixed with `_rd` operate using the _round down_ (toward negative infinity) rounding mode.


The `__fadd_[rn,rz,ru,rd]()`, `__dadd_[rn,rz,ru,rd]()`, `__fmul_[rn,rz,ru,rd]()`, and `__dmul_[rn,rz,ru,rd]()` functions map to addition and multiplication operations that the compiler never merges into the `FFMA` or `DFMA` instructions. In contrast, additions and multiplications generated from the `*` and `+` operators are often combined into `FFMA` or `DFMA`.

The following table lists the single- and double-precision floating-point intrinsic functions. All of them have a maximum ULP error of 0 and are IEEE-compliant.

Table 57 Single- and Double-Precision Floating-Point Intrinsic Functions Meaning | `float` | `double`  
---|---|---  
\\(x + y\\) | [__fadd_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fadd_rnff) | [__dadd_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dadd_rndd)  
\\(x - y\\) | [__fsub_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fsub_rnff) | [__dsub_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dsub_rndd)  
\\(x \cdot y\\) | [__fmul_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmul_rnff) | [__dmul_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__dmul_rndd)  
\\(x \cdot y + z\\) | [__fmaf_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fmaf_rnfff) | [__fma_[rn,rz,ru,rd](x, y, z)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv48__fma_rnddd)  
\\(\dfrac{x}{y}\\) | [__fdiv_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__fdiv_rnff) | [__ddiv_[rn,rz,ru,rd](x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__ddiv_rndd)  
\\(\dfrac{1}{x}\\) | [__frcp_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__frcp_rnf) | [__drcp_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv49__drcp_rnd)  
\\(\sqrt{x}\\) | [__fsqrt_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fsqrt_rnf) | [__dsqrt_[rn,rz,ru,rd](x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__DOUBLE.html#_CPPv410__dsqrt_rnd)  
  
### 5.5.9.2. Single-Precision-Only Intrinsic Functions

The following table lists the single-precision floating-point intrinsic functions with their maximum ULP error.

  * The maximum ULP error is stated as the maximum observed absolute value of the difference in ULPs between the value returned by the function and a correctly rounded result of the corresponding precision obtained according to the _round-to-nearest ties-to-even_ rounding mode.

  * The error bounds are derived from extensive, though not exhaustive, testing. Therefore, they are not guaranteed.


Table 58 **Single-Precision Only Floating-Point Intrinsic Functions**   
Mapping and Accuracy (Maximum ULP) Function | Meaning | Maximum ULP Error  
---|---|---  
[__fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff) | \\(\dfrac{x}{y}\\) | \\(2\\) for \\(|y| \in [2^{-126}, 2^{126}]\\)  
[__frsqrt_rn(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv411__frsqrt_rnf) | \\(\dfrac{1}{\sqrt{x}}\\) | 0 ULP  
[__expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff) | \\(e^x\\) | \\(2 + \lfloor |1.173 \cdot x| \rfloor\\)  
[__exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff) | \\(10^x\\) | \\(2 + \lfloor |2.97 \cdot x| \rfloor\\)  
[__powf(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff) | \\(x^y\\) | Derived from `exp2f(y * __log2f(x))`  
[__logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff) | \\(\ln(x)\\) | ▪ \\(2^{-21.41}\\) abs error for \\(x \in [0.5, 2]\\)   
▪ 3 ULP, otherwise  
[__log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff) | \\(\log_2(x)\\) | ▪ \\(2^{-22}\\) abs error for \\(x \in [0.5, 2]\\)   
▪ 2 ULP, otherwise  
[__log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff) | \\(\log_{10}(x)\\) | ▪ \\(2^{-24}\\) abs error for \\(x \in [0.5, 2]\\)   
▪ 3 ULP, otherwise  
[__sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff) | \\(\sin(x)\\) | ▪ \\(2^{-21.41}\\) abs error for \\(x \in [-\pi, \pi]\\)   
▪ larger otherwise  
[__cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff) | \\(\cos(x)\\) | ▪ \\(2^{-21.41}\\) abs error for \\(x \in [-\pi, \pi]\\)   
▪ larger otherwise  
[__sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf) | \\(\sin(x), \cos(x)\\) | Component-wise, the same as `__sinf(x)` and `__cosf(x)`  
[__tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff) | \\(\tan(x)\\) | Derived from `__sinf(x) * (1 / __cosf(x))`  
[__tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff) | \\(\tanh(x)\\) | ▪ Max relative error: \\(2^{-11}\\)   
▪ Subnormal results are not flushed to zero even under `-ftz=true` compiler flag.  
  
### 5.5.9.3. `--use_fast_math` Effect

The `nvcc` compiler flag `--use_fast_math` translates a subset of [CUDA Math API functions](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html) called in device code into their intrinsic counterpart. Note that the [CUDA C++ Standard Library functions](#mathematical-functions-appendix-cxx-standard-functions) are also affected by this flag. See the [Intrinsic Functions](#mathematical-functions-appendix-intrinsic-functions) section for more details on the implications of using intrinsic functions instead of CUDA Math API functions.

> A more robust approach is to selectively replace mathematical function calls with intrinsic versions only where the performance gains justify it and where the changed properties, such as reduced accuracy and different special-case handling, are acceptable.

Table 59 Functions Directly Affected by `--use_fast_math` Device Function | Intrinsic Function  
---|---  
[x/y, fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv48fdividefff) | [__fdividef(x, y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv410__fdividefff)  
[sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44sinff) | [__sinf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__sinff)  
[cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44cosff) | [__cosf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__cosff)  
[tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44tanff) | [__tanf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__tanff)  
[sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv47sincosffPfPf) | [__sincosf(x, sptr, cptr)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv49__sincosffPfPf)  
[logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44logff) | [__logf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__logff)  
[log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45log2ff) | [__log2f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__log2ff)  
[log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46log10ff) | [__log10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__log10ff)  
[expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44expff) | [__expf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__expff)  
[exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv46exp10ff) | [__exp10f(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv48__exp10ff)  
[powf(x,y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv44powfff) | [__powf(x,y)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv46__powfff)  
[tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__SINGLE.html#_CPPv45tanhff) | [__tanhf(x)](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html#_CPPv47__tanhff)  
  
## 5.5.10. References

  1. [IEEE 754-2019 Standard for Floating-Point Arithmetic](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8766229).

  2. Jean-Michel Muller. [On the definition of ulp(x)](https://inria.hal.science/inria-00070503v1/file/RR2005-09.pdf). INRIA/LIP research report, 2005.

  3. Nathan Whitehead, Alex Fit-Florea. [Precision & Performance: Floating Point and IEEE 754 Compliance for NVIDIA GPUs](https://developer.nvidia.com/content/precision-performance-floating-point-and-ieee-754-compliance-nvidia-gpus). Nvidia Report, 2011.

  4. David Goldberg. [What every computer scientist should know about floating-point arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html). ACM Computing Surveys, March 1991.

  5. David Monniaux. [The pitfalls of verifying floating-point computations](https://dl.acm.org/doi/pdf/10.1145/1353445.1353446). ACM Transactions on Programming Languages and Systems, May 2008.

  6. Peter Dinda, Conor Hetland. [Do Developers Understand IEEE Floating Point?](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425212). IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2018.
