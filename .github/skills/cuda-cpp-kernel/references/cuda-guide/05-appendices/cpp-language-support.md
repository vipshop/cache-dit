---
url: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html
---

# 5.3. C++ Language Support

`nvcc` processes CUDA and device code according to the following specifications:

  * **C++03** (ISO/IEC 14882:2003), `--std=c++03` flag.

  * **C++11** (ISO/IEC 14882:2011), `--std=c++11` flag.

  * **C++14** (ISO/IEC 14882:2014), `--std=c++14` flag.

  * **C++17** (ISO/IEC 14882:2017), `--std=c++17` flag.

  * **C++20** (ISO/IEC 14882:2020), `--std=c++20` flag.


Passing `nvcc` `-std=c++<version>` flag turns on all C++ features related to the specified version and also invokes the host preprocessor, compiler and linker with the corresponding C++ dialect option.

The compiler supports all language features of the supported standards, subject to the restrictions reported in the following sections.

## 5.3.1. C++11 Language Features

Table 34 C++11 Language Features Supported by NVCC for device code Language Feature | C++11 Proposal | NVCC/CUDA Toolkit 7.x  
---|---|---  
[Rvalue references](#rvalue-references) | [N2118](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n2118.html) | ✅  
Rvalue references for `*this` | [N2439](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2439.htm) | ✅  
Initialization of class objects by rvalues | [N1610](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1610.html) | ✅  
Non-static data member initializers | [N2756](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2008/n2756.htm) | ✅  
Variadic templates | [N2242](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2242.pdf) | ✅  
Extending variadic template template parameters | [N2555](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2555.pdf) | ✅  
[Initializer lists](#initializer-list) | [N2672](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2672.htm) | ✅  
Static assertions | [N1720](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1720.html) | ✅  
`auto`-typed variables | [N1984](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1984.pdf) | ✅  
Multi-declarator `auto` | [N1737](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1737.pdf) | ✅  
Removal of auto as a storage-class specifier | [N2546](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2546.htm) | ✅  
New function declarator syntax | [N2541](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2541.htm) | ✅  
[Lambda expressions](#lambda-expressions) | [N2927](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2927.pdf) | ✅  
Declared type of an expression | [N2343](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2343.pdf) | ✅  
Incomplete return types | [N3276](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3276.pdf) | ✅  
Right angle brackets | [N1757](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1757.html) | ✅  
Default template arguments for function templates | [DR226](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#226) | ✅  
Solving the SFINAE problem for expressions | [DR339](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2634.html) | ✅  
Alias templates | [N2258](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2258.pdf) | ✅  
Extern templates | [N1987](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1987.htm) | ✅  
Null pointer constant | [N2431](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2431.pdf) | ✅  
Strongly-typed enums | [N2347](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2347.pdf) | ✅  
Forward declarations for enums | [N2764](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2764.pdf)   
[DR1206](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1206) | ✅  
Standardized attribute syntax | [N2761](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2761.pdf) | ✅  
[Generalized constant expressions](#constexpr-functions) | [N2235](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2235.pdf) | ✅  
Alignment support | [N2341](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2341.pdf) | ✅  
Conditionally-supported behavior | [N1627](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1627.pdf) | ✅  
Changing undefined behavior into diagnosable errors | [N1727](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1727.pdf) | ✅  
Delegating constructors | [N1986](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1986.pdf) | ✅  
Inheriting constructors | [N2540](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2540.htm) | ✅  
Explicit conversion operators | [N2437](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2437.pdf) | ✅  
New character types | [N2249](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2249.html) | ✅  
Unicode string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | ✅  
Raw string literals | [N2442](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2442.htm) | ✅  
Universal character names in literals | [N2170](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2170.html) | ✅  
User-defined literals | [N2765](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2765.pdf) | ✅  
Standard Layout Types | [N2342](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2342.htm) | ✅  
[Defaulted functions](#cpp11-defaulted-function) | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | ✅  
Deleted functions | [N2346](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2346.htm) | ✅  
Extended friend declarations | [N1791](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1791.pdf) | ✅  
Extending `sizeof` | [N2253](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2253.html)   
[DR850](http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#850) | ✅  
[Inline namespaces](#inline-namespaces) | [N2535](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2535.htm) | ✅  
Unrestricted unions | [N2544](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2544.pdf) | ✅  
[Local and unnamed types as template arguments](#templates) | [N2657](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2657.htm) | ✅  
Range-based for | [N2930](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2930.html) | ✅  
Explicit `virtual` overrides | [N2928](http://www.open-std.org/JTC1/SC22/WG21/docs/papers/2009/n2928.htm)   
[N3206](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3206.htm)   
[N3272](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2011/n3272.htm) | ✅  
Minimal support for garbage collection and reachability-based leak detection | [N2670](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2670.htm) | ❌  
Allowing move constructors to throw [noexcept] | [N3050](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3050.html) | ✅  
Defining move special member functions | [N3053](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2010/n3053.html) | ✅  
**Concurrency**  
Sequence points | [N2239](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2239.html) | ❌  
Atomic operations | [N2427](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2427.html) | ❌  
Strong Compare and Exchange | [N2748](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2748.html) | ❌  
Bidirectional Fences | [N2752](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2752.htm) | ❌  
Memory model | [N2429](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2429.htm) | ❌  
Data-dependency ordering: atomics and memory model | [N2664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2664.htm) | ❌  
Propagating exceptions | [N2179](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2179.html) | ❌  
Allow atomics use in signal handlers | [N2547](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2547.htm) | ❌  
Thread-local storage | [N2659](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2659.htm) | ❌  
Dynamic initialization and destruction with concurrency | [N2660](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2008/n2660.htm) | ❌  
**C99 Features in C++11**  
`__func__` predefined identifier | [N2340](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2007/n2340.htm) | ✅  
C99 preprocessor | [N1653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1653.htm) | ✅  
`long long` | [N1811](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2005/n1811.pdf) | ✅  
Extended integral types | [N1988](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2006/n1988.pdf) | ❌  
  
## 5.3.2. C++14 Language Features

Table 35 C++14 Language Features Supported by NVCC for device code Language Feature | C++14 Proposal | NVCC/CUDA Toolkit 9.x  
---|---|---  
Tweak to certain C++ contextual conversions | [N3323](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3323.pdf) | ✅  
Binary literals | [N3472](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2012/n3472.pdf) | ✅  
[Functions with deduced return type](#return-type-deduction) | [N3638](https://isocpp.org/files/papers/N3638.html) | ✅  
Generalized lambda capture (init-capture) | [N3648](https://isocpp.org/files/papers/N3648.html) | ✅  
Generic (polymorphic) lambda expressions | [N3649](https://isocpp.org/files/papers/N3649.html) | ✅  
[Variable templates](#variable-templates) | [N3651](https://isocpp.org/files/papers/N3651.pdf) | ✅  
Relaxing requirements on constexpr functions | [N3652](https://isocpp.org/files/papers/N3652.html) | ✅  
Member initializers and aggregates | [N3653](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3653.html) | ✅  
Clarifying memory allocation | [N3664](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3664.html) | ❌  
Sized deallocation | [N3778](https://isocpp.org/files/papers/n3778.html) | ❌  
`[[deprecated]]` attribute | [N3760](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3760.html) | ✅  
Single-quotation-mark as a digit separator | [N3781](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2013/n3781.pdf) | ✅  
  
## 5.3.3. C++17 Language Features

Table 36 C++17 Language Features Supported by NVCC for device code Language Feature | C++17 Proposal | NVCC/CUDA Toolkit 11.x  
---|---|---  
Removing trigraphs | [N4086](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4086.html) | ✅  
`u8` character literals | [N4267](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4267.html) | ✅  
Folding expressions | [N4295](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4295.html) | ✅  
Attributes for namespaces and enumerators | [N4266](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4266.html) | ✅  
Nested namespace definitions | [N4230](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4230.html) | ✅  
Allow constant evaluation for all non-type template arguments | [N4268](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4268.html) | ✅  
Extending `static_assert` | [N3928](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3928.pdf) | ✅  
New Rules for `auto` deduction from braced-init-list | [N3922](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n3922.html) | ✅  
Allow typename in a template template parameter | [N4051](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2014/n4051.html) | ✅  
`[[fallthrough]]` attribute | [P0188R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0188r1.pdf) | ✅  
`[[nodiscard]]` attribute | [P0189R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0189r1.pdf) | ✅  
`[[maybe_unused]]` attribute | [P0212R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0212r1.pdf) | ✅  
Extension to aggregate initialization | [P0017R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html) | ✅  
Wording for `constexpr` lambda | [P0170R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0170r1.pdf) | ✅  
Unary Folds and Empty Parameter Packs | [P0036R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0036r0.pdf) | ✅  
Generalizing the Range-Based For Loop | [P0184R0](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0184r0.html) | ✅  
Lambda capture of `*this` by Value | [P0018R3](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html) | ✅  
Construction Rules for `enum class` variables | [P0138R2](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0138r2.pdf) | ✅  
Hexadecimal floating literals for C++ | [P0245R1](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0245r1.html) | ✅  
Dynamic memory allocation for over-aligned data | [P0035R4](https://wg21.link/p0035) | ✅  
Guaranteed copy elision | [P0135R1](https://wg21.link/p0135) | ✅  
Refining Expression Evaluation Order for Idiomatic C++ | [P0145R3](https://wg21.link/p0145) | ✅  
`constexpr if` | [P0292R2](https://wg21.link/p0292) | ✅  
Selection statements with initializer | [P0305R1](https://wg21.link/p0305) | ✅  
Template argument deduction for class templates | [P0091R3](https://wg21.link/p0091)   
[P0512R0](https://wg21.link/p0512r0) | ✅  
Declaring non-type template parameters with `auto` | [P0127R2](https://wg21.link/p0127) | ✅  
Using attribute namespaces without repetition | [P0028R4](https://wg21.link/p0028) | ✅  
Ignoring unsupported non-standard attributes | [P0283R2](https://wg21.link/p0283) | ✅  
[Structured bindings](#structured-binding) | [P0217R3](https://wg21.link/p0217) | ✅  
Remove Deprecated Use of the `register` Keyword | [P0001R1](https://wg21.link/p0001) | ✅  
Remove Deprecated `operator++(bool)` | [P0002R1](https://wg21.link/p0002) | ✅  
Make exception specifications be part of the type system | [P0012R1](https://wg21.link/p0012) | ✅  
`__has_include` for C++17 | [P0061R1](https://wg21.link/p0061) | ✅  
Rewording inheriting constructors (core issue 1941 et al) | [P0136R1](https://wg21.link/p0136) | ✅  
[Inline variables](#inline-variables) | [P0386R2](https://wg21.link/p0386r2) | ✅  
DR 150, Matching of template template arguments | [P0522R0](https://wg21.link/p0522r0) | ✅  
Removing dynamic exception specifications | [P0003R5](https://wg21.link/p0003r5) | ✅  
Pack expansions in using-declarations | [P0195R2](https://wg21.link/p0195r2) | ✅  
A `byte` type definition | [P0298R0](https://wg21.link/p0298r0) | ✅  
DR 727, In-class explicit instantiations | [CWG727](https://cplusplus.github.io/CWG/issues/727.html) | ✅  
  
## 5.3.4. C++20 Language Features

GCC version ≥ 10.0, Clang version ≥ 10.0, Microsoft Visual Studio ≥ 2022, and nvc++ version ≥ 20.7.

Table 37 C++20 Language Features Supported by NVCC for device code Language Feature | C++20 Proposal | NVCC/CUDA Toolkit 12.x  
---|---|---  
Default member initializers for bit-fields | [P0683R1](https://wg21.link/p0683r1) | ✅  
Fixing `const`-qualified pointers to members | [P0704R1](https://wg21.link/p0704r1) | ✅  
Allow lambda capture `[=, this]` | [P0409R2](https://wg21.link/p0409r2) | ✅  
`__VA_OPT__` for preprocessor comma elision | [P0306R4](https://wg21.link/p0306r4)   
[P1042R1](https://wg21.link/p1042r1) | ✅  
Designated initializers | [P0329R4](https://wg21.link/p0329r4) | ✅  
Familiar template syntax for generic lambdas | [P0428R2](https://wg21.link/p0428r2) | ✅  
List deduction of vector | [P0702R1](https://wg21.link/p0702r1) | ✅  
Concepts | [P0734R0](https://wg21.link/p0734r0)   
[P0857R0](https://wg21.link/p0857r0)   
[P1084R2](https://wg21.link/p1084r2)   
[P1141R2](https://wg21.link/p1141r2)   
[P0848R3](https://wg21.link/p0848r3)   
[P1616R1](https://wg21.link/p1616r1)   
[P1452R2](https://wg21.link/p1452r2)   
[P1972R0](https://wg21.link/p1972r0)   
[P1980R0](https://wg21.link/p1980r0)   
[P2092R0](https://wg21.link/p2092r0)   
[P2103R0](https://wg21.link/p2103r0)   
[P2113R0](https://wg21.link/p2113r0) | ✅  
Range-based for statements with initializer | [P0614R1](https://wg21.link/p0614r1) | ✅  
Simplifying implicit lambda capture | [P0588R1](https://wg21.link/p0588r1) | ✅  
ADL and function templates that are not visible | [P0846R0](https://wg21.link/p0846r0) | ✅  
`const` mismatch with defaulted copy constructor | [P0641R2](https://wg21.link/p0641r2) | ✅  
Less eager instantiation of `constexpr` functions | [P0859R0](https://wg21.link/p0859r0) | ✅  
[Consistent comparison](#cpp20-spaceship) (`operator<=>`) | [P0515R3](https://wg21.link/p0515r3)   
[P0905R1](https://wg21.link/p0905r1)   
[P1120R0](https://wg21.link/p1120r0)   
[P1185R2](https://wg21.link/p1185r2)   
[P1186R3](https://wg21.link/p1186r3)   
[P1630R1](https://wg21.link/p1630r1)   
[P1946R0](https://wg21.link/p1946r0)   
[P1959R0](https://wg21.link/p1959r0)   
[P2002R1](https://wg21.link/p2002r1)   
[P2085R0](https://wg21.link/p2085r0) | ✅  
Access checking on specializations | [P0692R1](https://wg21.link/p0692r1) | ✅  
Default constructible and assignable stateless lambdas | [P0624R2](https://wg21.link/p0624r2) | ✅  
Lambdas in unevaluated contexts | [P0315R4](https://wg21.link/p0315r4) | ✅  
Language support for empty objects | [P0840R2](https://wg21.link/p0840r2) | ✅  
Relaxing the range-for loop customization point finding rules | [P0962R1](https://wg21.link/p0962r1) | ✅  
[Allow structured bindings to accessible members](#structured-binding) | [P0969R0](https://wg21.link/p0969r0) | ✅  
Relaxing the structured bindings customization point finding rules | [P0961R1](https://wg21.link/p0961r1) | ✅  
Down with typename! | [P0634R3](https://wg21.link/p0634r3) | ✅  
Allow pack expansion in lambda init-capture | [P0780R2](https://wg21.link/p0780r2)   
[P2095R0](https://wg21.link/p2095r0) | ✅  
Proposed wording for `likely` and `unlikely` attributes | [P0479R5](https://wg21.link/p0479r5) | ✅  
Deprecate implicit capture of this via `[=]` | [P0806R2](https://wg21.link/p0806r2) | ✅  
Class Types in Non-Type Template Parameters | [P0732R2](https://wg21.link/p0732r2) | ✅  
Inconsistencies with non-type template parameters | [P1907R1](https://wg21.link/p1907r1) | ✅  
Atomic Compare-and-Exchange with Padding Bits | [P0528R3](https://wg21.link/p0528r3) | ✅  
Efficient sized delete for variable sized classes | [P0722R3](https://wg21.link/p0722r3) | ✅  
Allowing Virtual Function Calls in Constant Expressions | [P1064R0](https://wg21.link/p1064r0) | ✅  
Prohibit aggregates with user-declared constructors | [P1008R1](https://wg21.link/p1008r1) | ✅  
`explicit(bool)` | [P0892R2](https://wg21.link/p0892r2) | ✅  
Signed integers are two’s complement | [P1236R1](https://wg21.link/p1236r1) | ✅  
`char8_t` | [P0482R6](https://wg21.link/p0482r6) | ✅  
[Immediate functions](#cpp20-consteval) (`consteval`) | [P1073R3](https://wg21.link/p1073r3)   
[P1937R2](https://wg21.link/p1937r2) | ✅  
`std::is_constant_evaluated` | [P0595R2](https://wg21.link/p0595r2) | ✅  
Nested `inline` namespaces | [P1094R2](https://wg21.link/p1094r2) | ✅  
Relaxations of `constexpr` restrictions | [P1002R1](https://wg21.link/p1002r1)   
[P1327R1](https://wg21.link/p1327r1)   
[P1330R0](https://wg21.link/p1330r0)   
[P1331R2](https://wg21.link/p1331r2)   
[P1668R1](https://wg21.link/p1668r1)   
[P0784R7](https://wg21.link/p0784r7) | ✅  
Feature test macros | [P0941R2](https://wg21.link/p0941r2) | ✅  
Modules | [P1103R3](https://wg21.link/p1103r3)   
[P1766R1](https://wg21.link/p1766r1)   
[P1811R0](https://wg21.link/p1811r0)   
[P1703R1](https://wg21.link/p1703r1)   
[P1874R1](https://wg21.link/p1874r1)   
[P1979R0](https://wg21.link/p1979r0)   
[P1779R3](https://wg21.link/p1779r3)   
[P1857R3](https://wg21.link/p1857r3)   
[P2115R0](https://wg21.link/p2115r0)   
[P1815R2](https://wg21.link/p1815r2) | ❌  
Coroutines | [P0912R5](https://wg21.link/p0912r5) | ❌  
Parenthesized initialization of aggregates | [P0960R3](https://wg21.link/p0960r3)   
[P1975R0](https://wg21.link/p1975r0) | ✅  
DR: array size deduction in new-expression | [P1009R2](https://wg21.link/p1009r2) | ✅  
DR: Converting from `T*` to bool should be considered narrowing | [P1957R2](https://wg21.link/p1957r2) | ✅  
Stronger Unicode requirements | [P1041R4](https://wg21.link/p1041r4)   
[P1139R2](https://wg21.link/p1139r2) | ✅  
Structured binding extensions | [P1091R3](https://wg21.link/p1091r3)   
[P1381R1](https://wg21.link/p1381r1) | ✅  
Deprecate `a[b,c]` | [P1161R3](https://wg21.link/p1161r3) | ✅  
Deprecating some uses of `volatile` | [P1152R4](https://wg21.link/p1152r4) | ✅  
`[[nodiscard("with reason")]]` | [P1301R4](https://wg21.link/p1301r4) | ✅  
`using enum` | [P1099R5](https://wg21.link/p1099r5) | ✅  
Class template argument deduction for aggregates | [P1816R0](https://wg21.link/p1816r0)   
[P2082R1](https://wg21.link/p2082r1) | ✅  
Class template argument deduction for alias templates | [P1814R0](https://wg21.link/p1814r0) | ✅  
Permit conversions to arrays of unknown bound | [P0388R4](https://wg21.link/p0388r4) | ✅  
`constinit` | [P1143R2](https://wg21.link/p1143r2) | ✅  
Layout-compatibility and Pointer-interconvertibility Traits | [P0466R5](https://wg21.link/p0466r5) | ✅  
DR: Checking for abstract class types | [P0929R2](https://wg21.link/p0929r2) | ✅  
DR: More implicit moves | [P1825R0](https://wg21.link/p1825r0) | ✅  
DR: Pseudo-destructors end object lifetimes | [P0593R6](https://wg21.link/p0593r6) | ✅  
  
## 5.3.5. CUDA C++ Standard Library

CUDA provides an implementation of the C++ Standard Library (STL), called [libcu++](https://nvidia.github.io/cccl/libcudacxx/standard_api.html). The library presents the following benefits:

  * The functionalities are available on both host and device.

  * Compatible with all [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#id59) and [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#id2) platforms supported by the CUDA Toolkit.

  * Compatible with all [GPU architectures](https://developer.nvidia.com/cuda-gpus) supported by the last two major versions of the CUDA Toolkit.

  * Compatible with all [CUDA Toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with the current and previous major versions.

  * Provides C++17 backports of C++ Standard Library features available in recent standard versions, including C++20, C++23, and C++26.

  * Supports extended data types, such as 128-bit integers (`__int128`), half-precision floats (`__half`), Bfloat16 (`__nv_bfloat16`), and quad-precision floats (`__float128`).

  * Highly optimized for device code.


In addition, `libcu++` provides [extended features](https://nvidia.github.io/cccl/libcudacxx/extended_api.html) that are not available in the C++ Standard Library to improve productivity and application performance. Such features include mathematical functions, memory operations, synchronization primitives, container extensions, high-level abstractions of CUDA intrinsics, C++ PTX wrappers, and more.

`libcu++` is available as part of the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads), as well as part of the open-source [CCCL](https://nvidia.github.io/cccl/) repository.

## 5.3.6. C Standard Library Functions

### 5.3.6.1. `clock()` and `clock64()`
    
    
    __host__ __device__ clock_t   clock();
    __device__          long long clock64();
    

When executed in device code, it returns the value of a per-multiprocessor counter that increments every clock cycle. Sampling this counter at the beginning and end of a kernel, subtracting the two values, and recording the result for each thread provides an estimate of the number of clock cycles the device spends executing the thread. However, this value does not represent the actual number of clock cycles the device spends executing the thread’s instructions. The former number is greater than the latter because threads are time-sliced.

Hint

  * The corresponding [CUDA C++ function](https://en.cppreference.com/w/cpp/chrono/c/clock.html) `cuda::std::clock()` is provided in the `<cuda/std/ctime>` header.

  * A portable [C++](https://en.cppreference.com/w/cpp/header/chrono) `<chrono>` implementation is also provided in the `<cuda/std/chrono>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/time_library.html#libcudacxx-standard-api-time) for similar purposes.


### 5.3.6.2. `printf()`
    
    
    int printf(const char* format[, arg, ...]);
    

The function prints formatted output from a kernel to a host-side output stream.

The in-kernel `printf()` function behaves similarly to the standard C library `printf()` function. Users should refer to their host system’s manual pages for complete descriptions of `printf()` behavior. Essentially, the string passed in as `format` is output to a stream on the host.

The `printf()` command is executed like any other device-side function: per thread and in the context of the calling thread. In a multi-threaded kernel, a straightforward call to `printf()` will be executed by every thread using the data specified by that thread. Consequently, multiple versions of the output string will appear at the host stream, each corresponding to a thread that encountered the `printf()`.

Unlike the C standard `printf()`, which returns the number of characters printed, CUDA’s `printf()` returns the number of arguments parsed. If no arguments follow the format string, 0 is returned. If the format string is `NULL`, `-1` is returned. If an internal error occurs, -2 is returned.

Internally, `printf()` uses a shared data structure, so it is possible that calling `printf()` may alter the execution order of threads. In particular, a thread that calls `printf()` might take a longer execution path than a thread that does not call `printf()`, and the length of that path depends on the parameters of `printf()`. However, note that CUDA makes no guarantees about the order of thread execution except at explicit `__syncthreads()` barriers. Therefore, it is impossible to tell whether the order of execution has been modified by `printf()` or by other scheduling behaviors in the hardware.

* * *

**Format Specifiers**

As for standard `printf()`, format specifiers take the form: `%[flags][width][.precision][size]type`

The following fields are supported. See the widely available documentation for a complete description of all behaviors.

  * Flags: `#`, `' '`, `0`, `+`, `-`

  * Width: `*`, `0-9`

  * Precision: `0-9`

  * Size: `h`, `l`, `ll`

  * Type: `%cdiouxXpeEfgGaAs`


* * *

**Limitations**

The final formatting of the `printf()` output takes place on the host system. This means that the format string must be understood by the compiler and C library of the host system. While every effort has been made to ensure that the format specifiers supported by CUDA’s `printf()` function are a universal subset of those supported by the most common host compilers, the exact behavior will be dependent on the host operating system.

`printf()` accepts all valid combinations of flags and types. This is because it cannot determine what will and will not be valid on the host system where the final output is formatted. Consequently, output may be undefined if the program emits a format string containing invalid combinations.

The `printf()` function can accept up to 32 arguments, in addition to the format string. Any additional arguments will be ignored, and the format specifier will be output as is.

Due to the different sizes of the `long` type on Windows platforms (32-bit) and Linux platforms (64-bit), a kernel compiled on a Linux machine and then run on a Windows machine will produce corrupted output for all format strings that include `%ld`. To ensure safety, it is recommended that the compilation and execution platforms match.

* * *

**Host-Side Buffer**

The output buffer for `printf()` is set to a fixed size before kernel launch. The buffer is circular, so if more output is produced during kernel execution than can fit in the buffer, older output is overwritten. The buffer is flushed only when one of the following actions is performed:

  * Kernel launch via `<<< >>>` or `cuLaunchKernel()`: at the start of the launch, and if the `CUDA_LAUNCH_BLOCKING` environment variable is set to 1, at the end of the launch as well,

  * Synchronization via `cudaDeviceSynchronize()`, `cuCtxSynchronize()`, `cudaStreamSynchronize()`, `cuStreamSynchronize()`, `cudaEventSynchronize()`, or `cuEventSynchronize()`,

  * Memory copies via any blocking version of `cudaMemcpy*()` or `cuMemcpy*()`,

  * Module loading/unloading via `cuModuleLoad()` or `cuModuleUnload()`,

  * Context destruction via `cudaDeviceReset()` or `cuCtxDestroy()`.

  * Prior to executing a stream callback added by `cudaLaunchHostFunc()` or `cuLaunchHostFunc()`.


Note that the buffer is not automatically flushed when the program exits.

The following API functions set and retrieve the size of the buffer used to transfer `printf()` arguments and internal metadata to the host. The default size is one megabyte.

  * `cudaDeviceGetLimit(size_t* size,cudaLimitPrintfFifoSize)`

  * `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size_t size)`


* * *

**Examples**

The following code sample:
    
    
    #include <stdio.h>
    
    __global__ void helloCUDA(float value) {
        printf("Hello thread %d, value=%f\n", threadIdx.x, value);
    }
    
    int main() {
        helloCUDA<<<1, 5>>>(1.2345f);
        cudaDeviceSynchronize();
        return 0;
    }
    

will output:
    
    
    Hello thread 2, value=1.2345
    Hello thread 1, value=1.2345
    Hello thread 4, value=1.2345
    Hello thread 0, value=1.2345
    Hello thread 3, value=1.2345
    

Notice that each thread encounters the `printf()` command. Therefore, there are as many lines of output as there are threads in the grid.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/d4MPj7qG8).

* * *

The following code sample:
    
    
    #include <stdio.h>
    
    __global__ void helloCUDA(float value) {
        if (threadIdx.x == 0)
            printf("Hello thread %d, value=%f\n", threadIdx.x, value);
    }
    
    int main() {
        helloCUDA<<<1, 5>>>(1.2345f);
        cudaDeviceSynchronize();
        return 0;
    }
    

will output:
    
    
    Hello thread 0, value=1.2345
    

Clearly, the `if()` statement limits which threads call `printf()`, so only one line of output is seen.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/YqEss81sf).

### 5.3.6.3. `memcpy()` and `memset()`
    
    
    __host__ __device__ void* memcpy(void* dest, const void* src, size_t size);
    

The function copies `size` bytes from the memory location pointed by `src` to the memory location pointed by `dest`.
    
    
    __host__ __device__ void* memset(void* ptr, int value, size_t size);
    

The function sets `size` bytes of memory block pointed by `ptr` to `value`, interpreted as an `unsigned char`.

Hint

It is suggested to use the `cuda::std::memcpy()` and `cuda::std::memset()` functions provided in the `<cuda/std/cstring>` [header](https://nvidia.github.io/cccl/libcudacxx/standard_api/c_library/cstring.html#libcudacxx-standard-api-cstring) as safer versions of `memcpy` and `memset`.

### 5.3.6.4. `malloc()` and `free()`
    
    
    __host__ __device__ void* malloc(size_t size);
    // or cuda::std::malloc(), cuda::std::calloc() in the <cuda/std/cstdlib> header
    

The functions `malloc()` (device-side), `cuda::std::malloc()`, and `cuda::std::calloc()` allocate at least `size` bytes from the device heap and return a pointer to the allocated memory. If insufficient memory exists to fulfill the request, it returns `NULL`. The returned pointer is guaranteed to be aligned to a 16-byte boundary.
    
    
    __device__ void* __nv_aligned_device_malloc(size_t size, size_t align);
    // or cuda::std::aligned_alloc() in the <cuda/std/cstdlib> header
    

The functions `__nv_aligned_device_malloc()` and [C++](https://en.cppreference.com/w/cpp/memory/c/aligned_alloc) `cuda::std::aligned_alloc()` allocate at least `size` bytes from the device heap and return a pointer to the allocated memory. If there is insufficient memory to fulfill the requested size or alignment, it returns `NULL`. The address of the allocated memory is a multiple of `align`. `align` must be a non-zero power of two.
    
    
    __host__ __device__ void free(void* ptr);
    // or cuda::std::free() in the <cuda/std/cstdlib> header
    

The device-side functions `free()` and `cuda::std::free()` deallocate the memory pointed to by `ptr`, which must have been returned by a previous call to `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()`. If `ptr` is `NULL`, the call to `free()` or `cuda::std::free()` is ignored. Repeated calls to `free()` or `cuda::std::free()` with the same `ptr` have undefined behavior.

Memory allocated by a given CUDA thread via `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, or `cuda::std::aligned_alloc()` remain allocated for the lifetime of the CUDA context, or until it is explicitly released by a call to `free()` or `cuda::std::free()`. This memory can be used by other CUDA threads, even those from subsequent kernel launches. Any CUDA thread can free memory allocated by another thread; however, care should be taken to ensure that the same pointer is not freed more than once.

* * *

**Heap Memory API**

The size of the device memory heap must be specified before any program that allocates or frees memory in device code, including the `new` and `delete` keywords. If any program uses the device memory heap without explicitly specifying the heap size, a default heap of eight megabytes is allocated.

The following API functions get and set the heap size:

  * `cudaDeviceGetLimit(size_t* size, cudaLimitMallocHeapSize)`

  * `cudaDeviceSetLimit(cudaLimitMallocHeapSize, size_t size)`


The heap size granted will be at least `size` bytes. [cuCtxGetLimit()](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html#group__CUDA__CTX_1g9f2d47d1745752aa16da7ed0d111b6a8) and [cudaDeviceGetLimit()](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g720e159aeb125910c22aa20fe9611ec2) return the currently requested heap size.

The actual memory allocation for the heap occurs when a module is loaded into the context, either explicitly through the CUDA driver API (see [Module](../03-advanced/driver-api.html#driver-api-module)) or implicitly through the CUDA runtime API. If memory allocation fails, the module load generates a `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` error.

The heap size cannot be changed after a module has been loaded, and it does not dynamically resize according to need.

The memory reserved for the device heap is in addition to the memory allocated through host-side CUDA API calls such as `cudaMalloc()`.

* * *

**Interoperability with the Host Memory API**

Memory allocated via the device-side functions `malloc()`, `cuda::std::malloc()`, `cuda::std::calloc()`, `__nv_aligned_device_malloc()`, `cuda::std::aligned_alloc()`, or the `new` keyword cannot be used or freed with runtime or driver API calls such as `cudaMalloc`, `cudaMemcpy`, or `cudaMemset`. Similarly, memory allocated via the host runtime API cannot be freed using the device-side functions `free()`, `cuda::std::free()`, or the `delete` keyword.

* * *

Per-Thread Allocation example:
    
    
    #include <stdlib.h>
    #include <stdio.h>
    
    __global__ void single_thread_allocation_kernel() {
        size_t size = 123;
        char*  ptr  = (char*) malloc(size);
        memset(ptr, 0, size);
        printf("Thread %d got pointer: %p\n", threadIdx.x, ptr);
        free(ptr);
    }
    
    int main() {
        // Set a heap size of 128 megabytes.
        // Note that this must be done before any kernel is launched.
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
        single_thread_allocation_kernel<<<1, 5>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

will output:
    
    
    Thread 0 got pointer: 0x20d5ffe20
    Thread 1 got pointer: 0x20d5ffec0
    Thread 2 got pointer: 0x20d5fff60
    Thread 3 got pointer: 0x20d5f97c0
    Thread 4 got pointer: 0x20d5f9720
    

Notice how each thread encounters the `malloc()` and `memset()` commands and so receives and initializes its own allocation.

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/z7K191z58).

* * *

Per-Thread-Block Allocation example:
    
    
    #include <stdlib.h>
    
    __global__ void block_level_allocation_kernel() {
        __shared__ int* data;
        // The first thread in the block performs the allocation and shares the pointer
        // with all other threads through shared memory, so that access can be coalesced.
        if (threadIdx.x == 0) {
            size_t size = blockDim.x * 64; // 64 bytes per thread are allocated.
            data = (int*) malloc(size);
        }
        __syncthreads();
        // Check for failure
        if (data == nullptr)
            return;
    
        // Threads index into the memory, ensuring coalescence
        for (int i = 0; i < 64; ++i)
            data[i * blockDim.x + threadIdx.x] = threadIdx.x;
        // Ensure all threads complete before freeing
        __syncthreads();
    
        // Only one thread may free the memory!
        if (threadIdx.x == 0)
            free(data);
    }
    
    int main() {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);
        block_level_allocation_kernel<<<10, 128>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/7s8x7oonz).

* * *

Allocation Persisting Between Kernel Launches example:
    
    
    #include <stdlib.h>
    #include <stdio.h>
    
    const int NUM_BLOCKS = 20;
    
    __device__ int* data_ptrs[NUM_BLOCKS]; // Per-block pointer
    
    __global__ void allocate_memory_kernel() {
        // Only the first thread in the block performs the allocation
        // since we need only one allocation per block.
        if (threadIdx.x == 0)
            data_ptrs[blockIdx.x] = (int*) malloc(blockDim.x * 4);
        __syncthreads();
        // Check for failure
        if (data_ptrs[blockIdx.x] == nullptr)
            return;
        // Zero the data with all threads in parallel
        data_ptrs[blockIdx.x][threadIdx.x] = 0;
    }
    
    // Simple example: store the thread ID into each element
    __global__ void use_memory_kernel() {
        int* ptr = data_ptrs[blockIdx.x];
        if (ptr != nullptr)
            ptr[threadIdx.x] += threadIdx.x;
    }
    
    // Print the content of the buffer before freeing it
    __global__ void free_memory_kernel() {
        int* ptr = data_ptrs[blockIdx.x];
        if (ptr != nullptr)
            printf("Block %d, Thread %d: final value = %d\n",
                blockIdx.x, threadIdx.x, ptr[threadIdx.x]);
        // Only free from one thread!
        if (threadIdx.x == 0)
            free(ptr);
    }
    
    int main() {
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
        // Allocate memory
        allocate_memory_kernel<<<NUM_BLOCKS, 10>>>();
    
        // Use memory
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();
        use_memory_kernel<<<NUM_BLOCKS, 10>>>();
    
        // Free memory
        free_memory_kernel<<<NUM_BLOCKS, 10>>>();
        cudaDeviceSynchronize();
        return 0;
    }
    

See the example on [Compiler Explorer](https://cuda.godbolt.org/z/h7r6G3dGP).

### 5.3.6.5. `alloca()`
    
    
    __host__ __device__ void* alloca(size_t size);
    

The `alloca()` function allocates `size` bytes of memory within the caller’s stack frame. The returned value is a pointer to the allocated memory. When the function is invoked from device code, the beginning of the memory is 16-byte aligned. The memory is automatically freed when the caller returns from `alloca()`.

Note

On the Windows platform, the `<malloc.h>` header file must be included before using the `alloca()` function. Calls to `alloca()` may cause the stack to overflow; the user needs to adjust the stack size accordingly.

Example:
    
    
    __device__ void device_function(int num_items) {
        int4* ptr = (int4*) alloca(num_items * sizeof(int4));
        // use of ptr
        ...
    }
    

## 5.3.7. Lambda Expressions

The compiler determines the execution space of a lambda expression or closure type (C++11) by associating it with the execution space of the innermost enclosing function scope. If there is no enclosing function scope, the execution space is specified as `__host__`.

The execution space can also be specified explicitly with the [extended lambda syntax](#extended-lambdas).

Examples:
    
    
    auto global_lambda = [](){ return 0; }; // __host__
    
    void host_function() {
        auto lambda1 = [](){ return 1; };   // __host__
        [](){ return 3; };                  // __host__, closure type (body of a lambda expression)
    }
    
    __device__ void device_function() {
        auto lambda2 = [](){ return 2; };   // __device__
    }
    
    __global__ void kernel_function(void) {
        auto lambda3 = [](){ return 3; };   // __device__
    }
    
    __host__ __device__ void host_device_function() {
        auto lambda4 = [](){ return 4; };   // __host__ __device__
    }
    
    using function_ptr_t = int (*)();
    
    __device__ void device_function(float          value,
                                    function_ptr_t ptr = [](){ return 4; } /* __host__ */) {}
    

See the example on [Compiler Explorer](https://godbolt.org/z/scv4vcczr).

### 5.3.7.1. Lambda Expressions and `__global__` Function Parameters

A lambda expression or a closure type can only be used as an argument to a `__global__` function if its execution space is `__device__` or `__host__ __device__`. Global or namespace scope lambda expressions cannot be used as arguments in a `__global__` function.

Examples:
    
    
    template <typename T>
     __global__ void kernel(T input) {}
    
     __device__ void device_function() {
         // device kernel call requires separate compilation (-rdc=true flag)
         kernel<<<1, 1>>>([](){});
         kernel<<<1, 1>>>([] __device__() {});          // extended lambda
         kernel<<<1, 1>>>([] __host__ __device__() {}); // extended lambda
     }
    
     auto global_lambda = [] __host__ __device__() {};
    
     void host_function() {
         kernel<<<1, 1>>>([] __device__() {});          // CORRECT, extended lambda
         kernel<<<1, 1>>>([] __host__ __device__() {}); // CORRECT, extended lambda
     //  kernel<<<1, 1>>>([](){});                      // ERROR, closure type with host execution space
     //  kernel<<<1, 1>>>(global_lambda);               // ERROR, extended lambda, but at global scope
     }
    

See the example on [Compiler Explorer](https://godbolt.org/z/ajrsn5z5Y).

### 5.3.7.2. Extended Lambdas

The `nvcc` flag `--extended-lambda` allows explicit annotations of execution spaces in a lambda expression. These annotations should appear after the lambda introducer and before the optional lambda declarator.

`nvcc` defines the macro `__CUDACC_EXTENDED_LAMBDA__` when the `--extended-lambda` flag is specified.

  * An _extended lambda_ is defined within the scope of an immediate or nested block of a `__host__` or `__host__ __device__` function.

  * An _extended device lambda_ is a lambda expression annotated with the `__device__` keyword.

  * An _extended host-device lambda_ is a lambda expression annotated with the `__host__ __device__` keywords.


Unlike standard lambda expressions, extended lambdas can be used as type arguments in `__global__` functions.

Example:
    
    
    void host_function() {
        auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
        auto lambda2 = [] __device__ {};           // extended lambda
        auto lambda3 = [] __host__ __device__ {};  // extended lambda
        auto lambda4 = [] __host__ {};             // NOT an extended lambda
    }
    
    __host__ __device__ void host_device_function() {
        auto lambda1 = [] {};                      // NOT an extended lambda: no explicit execution space annotations
        auto lambda2 = [] __device__ {};           // extended lambda
        auto lambda3 = [] __host__ __device__ {};  // extended lambda
        auto lambda4 = [] __host__ {};             // NOT an extended lambda
    }
    
    __device__ void device_function() {
        // none of the lambdas within this function are extended lambdas,
        // because the enclosing function is not a __host__ or __host__ __device__  function.
        auto lambda1 = [] {};
        auto lambda2 = [] __device__ {};
        auto lambda3 = [] __host__ __device__ {};
        auto lambda4 = [] __host__ {};
    }
    
    auto global_lambda = [] __host__ __device__ { }; // NOT an extended lambda because it is not defined
                                                     // within a __host__ or __host__ __device__ function
    

### 5.3.7.3. Extended Lambda Type Traits

The compiler provides type traits to detect closure types for extended lambdas at compile time.
    
    
    bool __nv_is_extended_device_lambda_closure_type(type);
    

The function returns `true` if `type` is the closure class created for an extended `__device__` lambda, `false` otherwise.
    
    
    bool __nv_is_extended_device_lambda_with_preserved_return_type(type);
    

The function returns `true` if `type` is the closure class created for an extended `__device__` lambda and the lambda is defined with trailing return type, `false` otherwise. If the trailing return type definition refers to any lambda parameter name, the return type is not preserved.
    
    
    bool __nv_is_extended_host_device_lambda_closure_type(type);
    

The function returns `true` if `type` is the closure class created for an extended `__host__ __device__` lambda, `false` otherwise.

* * *

The lambda type traits can be used in all compilation modes, regardless of whether lambdas or extended lambdas are enabled. The traits will always return `false` if extended lambda mode is inactive.

Example:
    
    
    auto lambda0 = [] __host__ __device__ { };
    
    void host_function() {
        auto lambda1 = [] { };
        auto lambda2 = [] __device__ { };
        auto lambda3 = [] __host__ __device__ { };
        auto lambda4 = [] __device__ () -> double { return 3.14; }
        auto lambda5 = [] __device__ (int x) -> decltype(&x) { return 0; }
    
        using lambda0_t = decltype(lambda0);
        using lambda1_t = decltype(lambda1);
        using lambda2_t = decltype(lambda2);
        using lambda3_t = decltype(lambda3);
        using lambda4_t = decltype(lambda4);
        using lambda5_t = decltype(lambda5);
    
        // 'lambda0' is not an extended lambda because it is defined outside function scope
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda0_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda0_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda0_t));
    
        // 'lambda1' is not an extended lambda because it has no execution space annotations
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda1_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda1_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda1_t));
    
        // 'lambda2' is an extended device-only lambda
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda2_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda2_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda2_t));
    
        // 'lambda3' is an extended host-device lambda
        static_assert(!__nv_is_extended_device_lambda_closure_type(lambda3_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda3_t));
        static_assert(__nv_is_extended_host_device_lambda_closure_type(lambda3_t));
    
        // 'lambda4' is an extended device-only lambda with preserved return type
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda4_t));
        static_assert(__nv_is_extended_device_lambda_with_preserved_return_type(lambda4_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda4_t));
    
        // 'lambda5' is not an extended device-only lambda with preserved return type
        // because it references the operator()'s parameter types in the trailing return type.
        static_assert(__nv_is_extended_device_lambda_closure_type(lambda5_t));
        static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(lambda5_t));
        static_assert(!__nv_is_extended_host_device_lambda_closure_type(lambda5_t));
    }
    

### 5.3.7.4. Extended Lambda Restrictions

Before invoking the host compiler, the CUDA compiler replaces an extended lambda expression with an instance of a placeholder type defined in namespace scope. The placeholder type’s template argument requires taking the address of a function that encloses the original extended lambda expression. This is necessary for correctly executing any `__global__` function template whose template argument involves the closure type of an extended lambda. The enclosing function is computed as follows.

By definition, an extended lambda is present within the immediate or nested block scope of a `__host__` or `__host__ __device__` function.

  * If the function is not the `operator()` of a lambda expression, it is considered the enclosing function for the extended lambda.

  * Otherwise, the extended lambda is defined within the immediate or nested block scope of the `operator()` of one or more enclosing lambda expressions.

    * If the outermost lambda expression is defined within the immediate or nested block scope of a function `F`, then `F` is the computed enclosing function.

    * Otherwise, the enclosing function does not exist.


Example:
    
    
    void host_function() {
        auto lambda1 = [] __device__ { }; // enclosing function for lambda1 is "host_function()"
        auto lambda2 = [] {
            auto lambda3 = [] {
                auto lambda4 = [] __host__ __device__ { }; // enclosing function for lambda4 is "host_function"
            };
        };
    }
    
    auto global_lambda = [] {
        auto lambda5 = [] __host__ __device__ { }; // enclosing function for lambda5 does not exist
    };
    

* * *

Extended Lambda Restrictions

  1. An extended lambda cannot be defined inside another extended lambda expression. Example:
         
         void host_function() {
             auto lambda1 = [] __host__ __device__  {
                  // ERROR, extended lambda defined within another extended lambda
                 auto lambda2 = [] __host__ __device__ { };
             };
         }
         

  2. An extended lambda cannot be defined inside a generic lambda expression. Example:
         
         void host_function() {
             auto lambda1 = [] (auto) {
                  // ERROR, extended lambda defined within a generic lambda
                 auto lambda2 = [] __host__ __device__ { };
             };
         }
         

  3. If an extended lambda is defined within the immediate or nested block scope of one or more nested lambda expressions, then the outermost lambda expression must be defined within the immediate or nested block scope of a function. Example:
         
         auto lambda1 = []  {
             // ERROR, outer enclosing lambda is not defined within a non-lambda-operator() function
             auto lambda2 = [] __host__ __device__ { };
         };
         

  4. The enclosing function of the extended lambda must be named, and its address must be accessible. If the enclosing function is a class member, the following conditions must be met:

     * All classes enclosing the member function must have a name.

     * The member function must not have private or protected access within its parent class.

     * All enclosing classes must not have private or protected access within their respective parent classes.

Example:
    
    void host_function() {
        auto lambda1 = [] __device__ { return 0; }; // OK
        {
            auto lambda2 = [] __device__          { return 0; }; // OK
            auto lambda3 = [] __device__ __host__ { return 0; }; // OK
        }
    }
    
    struct MyStruct1 {
        MyStruct1() {
            auto lambda4 = [] __device__ { return 0; }; // ERROR, address of the enclosing function is not accessible
        }
    };
    
    class MyStruct2 {
        void foo() {
            auto temp1 = [] __device__ { return 10; }; // ERROR, enclosing function has private access in parent class
        }
    
        struct MyStruct3 {
            void foo() {
                auto temp1 = [] __device__ { return 10; };  // ERROR, enclosing class MyStruct3 has private access in its parent class
            }
        };
    };
    

  5. At the point where the extended lambda has been defined, it must be possible to unambiguously take the address of the enclosing routine. However, this may not always be feasible, for example, when an alias declaration shadows a template type argument with the same name. Example:
         
         template <typename T>
         struct A {
             using Bar = void;
             void test();
         };
         
         template<>
         struct A<void> { };
         
         template <typename Bar>
         void A<Bar>::test() {
             // In code sent to host compiler, nvcc will inject an address expression here, of the form:
             //   (void (A< Bar> ::*)(void))(&A::test))
             //  However, the class typedef 'Bar' (to void) shadows the template argument 'Bar',
             //  causing the address expression in A<int>::test to actually refer to:
             //    (void (A< void> ::*)(void))(&A::test))
             //  which doesn't take the address of the enclosing routine 'A<int>::test' correctly.
             auto lambda1 = [] __host__ __device__ { return 4; };
         }
         
         int main() {
             A<int> var;
             var.test();
         }
         

  6. An extended lambda cannot be defined in a class that is local to a function. Example:
         
         void host_function() {
             struct MyStruct {
                 void bar() {
                     // ERROR, bar() is member of a class that is local to a function
                     auto lambda2 = [] __host__ __device__ { return 0; };
                 }
             };
         }
         

  7. The enclosing function for an extended lambda cannot have deduced return type. Example:
         
         auto host_function() {
             // ERROR, the return type of host_function() is deduced
             auto lambda3 = [] __host__ __device__ { return 0; };
         }
         

  8. A host-device extended lambda cannot be a generic lambda, namely a lambda with an `auto` parameter type. Example:
         
         void host_function() {
             // ERROR, __host__ __device__ extended lambdas cannot be a generic lambda
             auto lambda1 = [] __host__ __device__ (auto i) { return i; };
         
             // ERROR, a host-device extended lambda cannot be a generic lambda
             auto lambda2 = [] __host__ __device__ (auto... i) {
                 return sizeof...(i);
             };
         }
         

  9. If the enclosing function is an instantiation of a function or member template, or if the function is a member of a class template, then the template(s) must satisfy the following constraints:

     * The template must have at most one variadic parameter, and it must be listed last in the template parameter list.

     * The template parameters must be named.

     * The template instantiation argument types cannot involve types that are either local to a function (except for closure types for extended lambdas), or are `private` or `protected` class members.

Example 1:
    
    template <template <typename...> class T,
              typename... P1,
              typename... P2>
    void bar1(const T<P1...>, const T<P2...>) {
        // ERROR, enclosing function has multiple parameter packs
        auto lambda = [] __device__ { return 10; };
    }
    
    template <template <typename...> class T,
              typename... P1,
              typename    T2>
    void bar2(const T<P1...>, T2) {
        // ERROR, for enclosing function, the parameter pack is not last in the template parameter list
        auto lambda = [] __device__ { return 10; };
    }
    
    template <typename T, T>
    void bar3() {
        // ERROR, for enclosing function, the second template parameter is not named
        auto lambda = [] __device__ { return 10; };
    }
    

Example 2:
    
    template <typename T>
    void bar4() {
        auto lambda1 = [] __device__ { return 10; };
    }
    
    class MyStruct {
        struct MyNestedStruct {};
    
        friend int main();
    };
    
    int main() {
        struct MyLocalStruct {};
        // ERROR, enclosing function for device lambda in bar4() is instantiated with a type local to main
        bar4<MyLocalStruct>();
    
        // ERROR, enclosing function for device lambda in bar4 is instantiated with a type
        //        that is a private member of a class
        bar4<MyStruct::MyNestedStruct>();
    }
    

  10. With Microsoft Visual Studio host compilers, the enclosing function must have external linkage. This restriction exists because the host compiler does not support using the addresses of non-extern linkage functions as template arguments. The CUDA compiler transformations require these addresses to support extended lambdas.

  11. With Microsoft Visual Studio host compilers, an extended lambda shall not be defined within the body of an `if constexpr` block.

  12. An extended lambda has the following restrictions on captured variables:

     * The variable may be passed by value to a sequence of helper functions in the code sent to the host compiler before being used to directly initialize the field of the class type representing the closure type for the extended lambda. However, the C++ standard specifies that the captured variable should be used for direct initialization of the closure type’s field.

     * A variable can only be captured by value.

     * A variable of array type cannot be captured if the number of array dimensions is greater than 7.

     * For an array-type variable, the array field of the closure type is first default-initialized and then each array element is copy-assigned from the corresponding element of the captured array variable in the code sent to the host compiler. Therefore, the array element type must be both default-constructible and copy-assignable in the host code.

     * A function parameter that is an element of a variadic argument pack cannot be captured.

     * The captured variable type cannot be local to a function, except for extended lambda closure types, or `private` or `protected` class members.

     * Init-capture is not supported for host-device extended lambdas. However, it is supported for device extended lambdas, except when the initializer is an array or of type `std::initializer_list`.

     * The function call operator for an extended lambda is not a `constexpr`. The closure type of an extended lambda is not a literal type. The `constexpr` and `consteval` specifiers cannot be used when declaring an extended lambda.

     * A variable cannot be implicitly captured inside an `if-constexpr` block that is lexically nested inside an extended lambda unless the variable has been implicitly captured outside the `if-constexpr` block or appears in the extended lambda’s explicit capture list.

Examples:
    
    void host_function() {
        // CORRECT, an init-capture is allowed for an extended device-only lambda
        auto lambda1 = [x = 1] __device__ () { return x; };
    
        // ERROR, an init-capture is not allowed for an extended host-device lambda
        auto lambda2 = [x = 1] __host__ __device__ () { return x; };
    
        int a = 1;
        // ERROR, an extended __device__ lambda cannot capture variables by reference
        auto lambda3 = [&a] __device__ () { return a; };
    
        // ERROR, by-reference capture is not allowed for an extended device-only lambda
        auto lambda4 = [&x = a] __device__ () { return x; };
    
        struct MyStruct {};
        MyStruct s1;
        // ERROR, a type local to a function cannot be used in the type of a captured variable
        auto lambda6 = [s1] __device__ () { };
    
        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda7 = [x = {11}] __device__ () { };
    
        std::initializer_list<int> b = {11,22,33};
        // ERROR, an init-capture cannot be of type std::initializer_list
        auto lambda8 = [x = b] __device__ () { };
    
        int  var     = 4;
        auto lambda9 = [=] __device__ {
            int result = 0;
            if constexpr(false) {
                //ERROR, An extended device-only lambda cannot first-capture 'var' in if-constexpr context
                result += var;
            }
            return result;
        };
    
        auto lambda10 = [var] __device__ {
            int result = 0;
            if constexpr(false) {
                // CORRECT, 'var' already listed in explicit capture list for the extended lambda
                result += var;
            }
            return result;
        };
    
        auto lambda11 = [=] __device__ {
            int result = var;
            if constexpr(false) {
                // CORRECT, 'var' already implicit captured outside the 'if-constexpr' block
                result += var;
            }
            return result;
        };
    }
    

  13. When parsing a function, the CUDA compiler assigns a counter value to each extended lambda in the function. This counter value is used in the substituted named type that is passed to the host compiler. Therefore, the presence or absence of an extended lambda within a function should not depend on a particular value of `__CUDA_ARCH__`, nor on `__CUDA_ARCH__` being undefined. Example:
         
         template <typename T>
         __global__ void kernel(T in) { in(); }
         
         __host__ __device__ void host_device_function() {
             // ERROR, the number and relative declaration order of
             //        extended lambdas depend on __CUDA_ARCH__
         #if defined(__CUDA_ARCH__)
             auto lambda1 = [] __device__ { return 0; };
             auto lambda2 = [] __host__ __device__ { return 10; };
         #endif
             auto lambda3 = [] __device__ { return 4; };
             kernel<<<1, 1>>>(lambda3);
         }
         

  14. As described above, the CUDA compiler replaces a device extended lambda defined in a host function with a placeholder type defined in namespace scope. The placeholder type does not define an `operator()` function equivalent to the original lambda declaration unless the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true` for the closure type of the extended lambda. Therefore, an attempt to determine the return type or parameter types of the `operator()` function of such a lambda may work incorrectly in host code because the code processed by the host compiler is semantically different from the input code processed by the CUDA compiler. However, introspecting the return type or parameter types of the `operator()` function within device code is acceptable. Note that this restriction does not apply to host or device extended lambdas for which the trait `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`. Example:
         
         #include <cuda/std/type_traits>
         
         const char& getRef(const char* p) { return *p; }
         
         void foo() {
             auto lambda1 = [] __device__ { return "10"; };
         
             // ERROR, attempt to extract the return type of a device lambda in host code
             cuda::std::result_of<decltype(lambda1)()>::type xx1 = "abc";
         
             auto lambda2 = [] __host__ __device__ { return "10"; };
         
             // CORRECT, lambda2 represents a host-device extended lambda
             cuda::std::result_of<decltype(lambda2)()>::type xx2 = "abc";
         
             auto lambda3 = [] __device__ () -> const char* { return "10"; };
         
             // CORRECT, lambda3 represents a device extended lambda with preserved return type
             cuda::std::result_of<decltype(lambda3)()>::type xx2 = "abc";
             static_assert(cuda::std::is_same_v<cuda::std::result_of<decltype(lambda3)()>::type, const char*>);
         
             auto lambda4 = [] __device__ (char x) -> decltype(getRef(&x)) { return 0; };
             // lambda4's return type is not preserved because it references the operator()'s
             // parameter types in the trailing return type.
             static_assert(!__nv_is_extended_device_lambda_with_preserved_return_type(decltype(lambda4)));
         }
         

  15. For an extended device-only lambda:

     * Introspection of the parameter type of `operator()` is only supported in device code.

     * Introspection of the return type of `operator()` is supported only in device code, unless the trait function `__nv_is_extended_device_lambda_with_preserved_return_type()` returns `true`.

  16. If an extended lambda is passed from host to device code as an argument to a `__global__` function, for example, then any expression in the lambda’s body that captures variables must remain unchanged, regardless of whether the `__CUDA_ARCH__` macro is defined and what value it has. This restriction arises because the lambda’s closure class layout depends on the order in which the compiler encounters the captured variables when processing the lambda expression. The program may execute incorrectly if the closure class layout differs between device and host compilations. Example:
         
         __device__ int result;
         
         template <typename T>
         __global__ void kernel(T in) { result = in(); }
         
         void foo(void) {
             int x1 = 1;
             // ERROR, "x1" is only captured when __CUDA_ARCH__ is defined.
             auto lambda1 = [=] __host__ __device__ {
         #ifdef __CUDA_ARCH__
                 return x1 + 1;
         #else
                 return 10;
         #endif
             };
             kernel<<<1, 1>>>(lambda1);
         }
         

  17. As previously described, the CUDA compiler replaces an extended device-only lambda expression with a placeholder type instance in the code sent to the host compiler. The placeholder type does not define a pointer-to-function conversion operator in the host code; however, the conversion operator is provided in the device code. Note that this restriction does not apply to host-device extended lambdas. Example:
         
         template <typename T>
         __global__ void kernel(T in) {
             int (*fp)(double) = in;
             fp(0); // CORRECT, conversion in device code is supported
             auto lambda1 = [](double) { return 1; };
         }
         
         void foo() {
             auto lambda_device      = [] __device__ (double) { return 1; };
             auto lambda_host_device = [] __host__ __device__ (double) { return 1; };
             kernel<<<1, 1>>>(lambda_device);
             kernel<<<1, 1>>>(lambda_host_device);
         
             // CORRECT, conversion for a __host__ __device__ lambda is supported in host code
             int (*fp1)(double) = lambda_host_device;
         
             // ERROR, conversion for a device lambda is not supported in host code
             int (*fp2)(double) = lambda_device;
         }
         

  18. As previously described, the CUDA compiler replaces an extended device-only or host-device lambda expression with a placeholder type instance in the code sent to the host compiler. This placeholder type may define C++ special member functions, such as constructors and destructors. Consequently, some standard C++ type traits may yield different results for the closure type of the extended lambda in the CUDA front-end compiler than in the host compiler. The following type traits are affected: : `std::is_trivially_copyable`, `std::is_trivially_constructible`, `std::is_trivially_copy_constructible`, `std::is_trivially_move_constructible`, `std::is_trivially_destructible`. Care must be taken to ensure that the results of these traits are not used in the instantiation of the `__global__`, `__device__`, `__constant__`, or `__managed__` function or variable templates. Example:
         
         #include <cstdio>
         #include <type_traits>
         
         template <bool b>
         void __global__ kernel() { printf("hi"); }
         
         template <typename T>
         void kernel_launch() {
             // ERROR, this kernel launch may fail, because CUDA frontend compiler and host compiler
             //        may disagree on the result of std::is_trivially_copyable_v trait on the
             //        closure type of the extended lambda
             kernel<std::is_trivially_copyable_v<T>><<<1,1>>>();
             cudaDeviceSynchronize();
         }
         
         int main() {
             int  x       = 0;
             auto lambda1 = [=] __host__ __device__ () { return x; };
             kernel_launch<decltype(lambda1)>();
         }
         


The CUDA compiler will generate compiler diagnostics for a subset of cases described in `1-12`; no diagnostic will be generated for cases `13-17`, but the host compiler may fail to compile the generated code.

### 5.3.7.5. Host-Device Lambda Optimization Notes

Unlike device-only lambdas, host-device lambdas can be called from host code. As previously mentioned, the CUDA compiler replaces an extended lambda expression defined in host code with an instance of a named placeholder type. The placeholder type for an extended host-device lambda invokes the original lambda’s `operator()` with an indirect function call. The traits will always return false if extended lambda mode is not active.

The presence of an indirect function call may cause the host compiler to optimize an extended host-device lambda less than lambdas that are implicitly or explicitly `__host__` only. In the latter case, the host compiler can easily inline the lambda body into the calling context. However, when it encounters an extended host-device lambda, the host compiler may not be able to easily inline the original lambda body.

### 5.3.7.6. `*this` Capture By-Value

According to C++11/C++14 rules, when a lambda is defined within a non-`static` class member function and the lambda’s body refers to a class member variable, the `this` pointer of the class must be captured by value rather than the referenced member variable. If the lambda is an extended device-only or host-device lambda defined in a host function and executed on the GPU, accessing the referenced member variable on the GPU will cause a runtime error if the `this` pointer points to host memory.

Example:
    
    
    #include <cstdio>
    
    template <typename T>
    __global__ void foo(T in) { printf("value = %d\n", in()); }
    
    struct MyStruct {
        int var;
    
        __host__ __device__ MyStruct() : var(10) {};
    
        void run() {
            auto lambda1 = [=] __device__ {
                // reference to "var" causes the 'this' pointer (MyStruct*) to be captured by value
                return var + 1;
            };
            // Kernel launch fails at run time because 'this->var' is not accessible from the GPU
            foo<<<1, 1>>>(lambda1);
            cudaDeviceSynchronize();
        }
    };
    
    int main() {
        MyStruct s1;
        s1.run();
    }
    

C++17 solves this problem by introducing a new `*this` capture mode. In this mode, the compiler copies the object denoted by `*this` instead of capturing the `this` pointer by value. The `*this` capture mode is described in more detail in [P0018R3](http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2016/p0018r3.html).

The CUDA compiler supports the `*this` capture mode for lambdas defined within `__device__` and `__global__` functions and for extended device-only lambdas defined in host code, when the `--extended-lambda` flag is used.

Here’s the above example modified to use `*this` capture mode:
    
    
    #include <cstdio>
    
    template <typename T>
    __global__ void foo(T in) { printf("\n value = %d", in()); }
    
    struct MyStruct {
        int var;
        __host__ __device__ MyStruct() : var(10) { };
    
        void run() {
            // note the "*this" capture specification
            auto lambda1 = [=, *this] __device__ {
                // reference to "var" causes the object denoted by '*this' to be captured by
                // value, and the GPU code will access 'copy_of_star_this->var'
                return var + 1;
            };
            // Kernel launch succeeds
            foo<<<1, 1>>>(lambda1);
            cudaDeviceSynchronize();
        }
    };
    
    int main() {
        MyStruct s1;
        s1.run();
    }
    

`*this` capture mode is not allowed for non-annotated lambdas defined in host code, or for extended host-device lambdas, unless `*this` capture is enabled by the selected language dialect. The following are examples of supported and unsupported usage:
    
    
    struct MyStruct {
        int var;
        __host__ __device__ MyStruct() : var(10) { };
    
        void host_function() {
            // CORRECT, use in an extended device-only lambda
            auto lambda1 = [=, *this] __device__ { return var; };
    
            // Use in an extended host-device lambda
            // Error if *this capture not enabled by language dialect
            auto lambda2 = [=, *this] __host__ __device__ { return var; };
    
            // Use in an non-annotated lambda in host function
            // Error if *this capture not enabled by language dialect
            auto lambda3 = [=, *this]  { return var; };
        }
    
        __device__ void device_function() {
            // CORRECT, use in a lambda defined in a device-only function
            auto lambda1 = [=, *this] __device__ { return var; };
    
            // CORRECT, use in a lambda defined in a device-only function
            auto lambda2 = [=, *this] __host__ __device__ { return var; };
    
            // CORRECT, use in a lambda defined in a device-only function
            auto lambda3 = [=, *this]  { return var; };
        }
    
        __host__ __device__ void host_device_function() {
            // CORRECT, use in an extended device-only lambda
            auto lambda1 = [=, *this] __device__ { return var; };
    
            // Use in an extended host-device lambda
            // Error if *this capture not enabled by language dialect
            auto lambda2 = [=, *this] __host__ __device__ { return var; };
    
            // Use in an unannotated lambda in a host-device function
            // Error if *this capture not enabled by language dialect
            auto lambda3 = [=, *this]  { return var; };
        }
    };
    

### 5.3.7.7. Argument Dependent Lookup (ADL)

As previously mentioned, the CUDA compiler replaces an extended lambda expression with a placeholder type before invoking the host compiler. One template argument of the placeholder type uses the address of the function that encloses the original lambda expression. This may cause additional namespaces to participate in [Argument-Dependent Lookup (ADL)](https://en.cppreference.com/w/cpp/language/adl.html) for any host function call whose argument types involve the closure type of the extended lambda expression. Consequently, an incorrect function may be selected by the host compiler.

Example:
    
    
    namespace N1 {
    
    struct MyStruct {};
    
    template <typename T>
    void my_function(T);
    
    }; // namespace N1
    
    namespace N2 {
    
    template <typename T>
    int my_function(T);
    
    template <typename T>
    void run(T in) { my_function(in); }
    
    } // namespace N2
    
    void bar(N1::MyStruct in) {
        // For extended device-only lambda, the code sent to the host compiler is replaced with
        // the placeholder type instantiation expression
        //    ' __nv_dl_wrapper_t< __nv_dl_tag<void (*)(N1::MyStruct in),(&bar),1> > { }'
        //
        // As a result, the namespace 'N1' participates in ADL lookup of the
        // call to "my_function()" in the body of N2::run, causing ambiguity.
        auto lambda1 = [=] __device__ { };
        N2::run(lambda1);
    }
    

In the above example, the CUDA compiler replaced the extended lambda with a placeholder type involving the `N1` namespace. Consequently, the `N1` namespace participates in the ADL lookup for `my_function(in)` in the body of `N2::run()`, resulting in a host compilation failure due to the discovery of multiple overload candidates: `N1::my_function` and `N2::my_function`.

## 5.3.8. Polymorphic Function Wrappers

The `nvfunctional` header provides a polymorphic function wrapper class template, `nvstd::function`. Instances of this class template can store, copy, and invoke any callable target, such as lambda expressions. `nvstd::function` can be used in both host and device code.

Example:
    
    
    #include <nvfunctional>
    
    __host__            int host_function()        { return 1; }
    __device__          int device_function()      { return 2; }
    __host__ __device__ int host_device_function() { return 3; }
    
    __global__ void kernel(int* result) {
        nvstd::function<int()> fn1 = device_function;
        nvstd::function<int()> fn2 = host_device_function;
        nvstd::function<int()> fn3 = [](){ return 10; };
        *result                    = fn1() + fn2() + fn3();
    }
    
    __host__ __device__ void host_device_test(int* result) {
        nvstd::function<int()> fn1 = host_device_function;
        nvstd::function<int()> fn2 = [](){ return 10; };
        *result                    = fn1() + fn2();
    }
    
    __host__ void host_test(int* result) {
        nvstd::function<int()> fn1 = host_function;
        nvstd::function<int()> fn2 = host_device_function;
        nvstd::function<int()> fn3 = [](){ return 10; };
        *result                    = fn1() + fn2() + fn3();
    }
    

* * *

Invalid cases:

  * Instances of `nvstd::function` in host code cannot be initialized with the address of a `__device__` function or with a functor whose `operator()` is a `__device__` function.

  * Similarly, instances of `nvstd::function` in device code cannot be initialized with the address of a `__host__` function or with a functor whose `operator()` is a `__host__` function.

  * `nvstd::function` instances cannot be passed from host code to device code (or vice versa) at runtime.

  * `nvstd::function` cannot be used in the parameter type of a `__global__` function if the `__global__` function is launched from host code.


Examples of invalid cases:
    
    
    #include <nvfunctional>
    
    __device__ int device_function() { return 1; }
    __host__   int host_function() { return 3; }
    auto       lambda_host  = [] { return 0; };
    
    __global__ void k() {
        nvstd::function<int()> fn1 = host_function; // ERROR, initialized with address of __host__ function
        nvstd::function<int()> fn2 = lambda_host;   // ERROR, initialized with address of functor with
                                                    //        __host__ operator() function
    }
    
    __global__ void kernel(nvstd::function<int()> f1) {}
    
    void foo(void) {
        auto lambda_device = [=] __device__ { return 1; };
    
        nvstd::function<int()> fn1 = device_function; // ERROR, initialized with address of __device__ function
        nvstd::function<int()> fn2 = lambda_device;   // ERROR, initialized with address of functor with
                                                      //        __device__ operator() function
        kernel<<<1, 1>>>(fn2);                        // ERROR, passing nvstd::function from host to device
    }
    

* * *

`nvstd::function` is defined in the `nvfunctional` header as follows:
    
    
    namespace nvstd {
    
    template <typename RetType, typename ...ArgTypes>
    class function<RetType(ArgTypes...)> {
    public:
        // constructors
        __device__ __host__ function() noexcept;
        __device__ __host__ function(nullptr_t) noexcept;
        __device__ __host__ function(const function&);
        __device__ __host__ function(function&&);
    
        template<typename F>
        __device__ __host__ function(F);
    
        // destructor
        __device__ __host__ ~function();
    
        // assignment operators
        __device__ __host__ function& operator=(const function&);
        __device__ __host__ function& operator=(function&&);
        __device__ __host__ function& operator=(nullptr_t);
        template<typename F>
        __device__ __host__ function& operator=(F&&);
    
        // swap
        __device__ __host__ void swap(function&) noexcept;
    
        // function capacity
        __device__ __host__ explicit operator bool() const noexcept;
    
        // function invocation
        __device__ RetType operator()(ArgTypes...) const;
    };
    
    // null pointer comparisons
    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator==(const function<R(ArgTypes...)>&, nullptr_t) noexcept;
    
    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator==(nullptr_t, const function<R(ArgTypes...)>&) noexcept;
    
    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator!=(const function<R(ArgTypes...)>&, nullptr_t) noexcept;
    
    template <typename R, typename... ArgTypes>
    __device__ __host__
    bool operator!=(nullptr_t, const function<R(ArgTypes...)>&) noexcept;
    
    // specialized algorithms
    template <typename R, typename... ArgTypes>
    __device__ __host__
    void swap(function<R(ArgTypes...)>&, function<R(ArgTypes...)>&);
    
    } // namespace nvstd
    

## 5.3.9. C/C++ Language Restrictions

### 5.3.9.1. Unsupported Features

  * Run-Time Type Information (RTTI) and exceptions are not supported in device code:

    * `typeid` keyword

    * `dynamic_cast` keyword

    * `try/catch/throw` keywords

  * `long double` is not supported in device code.

  * Trigraphs are not supported on any platform. Digraphs are not supported on Windows.

  * User-defined `operator new`, `operator new[]`, `operator delete`, or `operator delete[]` cannot be used to replace the corresponding built-ins provided by the compiler, and it is considered undefined behavior on both host and device.


### 5.3.9.2. Namespace Reservations

Unless otherwise noted, adding definitions to top-level namespaces `cuda::`, `nv::`, or `cooperative_groups::`, or to any nested namespace within them, is undefined behavior. We allow `cuda::` as a subnamespace as depicted below:

Examples:
    
    
    namespace cuda {   // same for "nv" and "cooperative_groups" namespaces
    
    struct foo;        // ERROR, class declaration in the "cuda" namespace
    
    void bar();        // ERROR, function declaration in the "cuda" namespace
    
    namespace utils {} // ERROR, namespace declaration in the "cuda" namespace
    
    } // namespace cuda
    
    
    
    namespace utils {
    namespace cuda {
    
    // CORRECT, namespace "cuda" may be used nested within a non-reserved namespace
    void bar();
    
    } // namespace cuda
    } // namespace utils
    
    // ERROR, Equivalent to adding symbols to namespace "cuda" at global scope
    using namespace utils;
    

### 5.3.9.3. Pointers and Memory Addresses

Pointer dereferencing (`*pointer`, `pointer->member`, `pointer[0]`) is allowed only in the same execution space where the associated memory resides. The following cases result in undefined behavior, most often a segmentation fault and application termination.

  * Dereferencing a pointer either to [global memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-global-memory), [shared memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-shared-memory), or [constant memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-constant-memory) on the host.

  * Dereferencing a pointer to host memory in device code.


The following restrictions apply to functions:

  * It is not allowed to take the address of a `__device__` function in host code.

  * The address of a `__global__` function taken in host code cannot be used in device code. Similarly, the address of a `__global__` function taken in device code cannot be used in host code.


The address of a `__device__` or `__constant__` variable obtained through `cudaGetSymbolAddress()` as described in the [Memory Space Specifiers](cpp-language-extensions.html#memory-space-specifiers) section can only be used in host code.

### 5.3.9.4. Variables

#### 5.3.9.4.1. Local Variables

The `__device__`, `__shared__`, `__managed__`, and `__constant__` memory space specifiers are not allowed on non-`extern` variable declarations within a function that executes on the host.

Examples:
    
    
    __host__ void host_function() {
        int x;                   // CORRECT, __host__ variable
        __device__   int y;      // ERROR,   __device__ variable declaration within a host function
        __shared__   int z;      // ERROR,   __shared__ variable declaration within a host function
        __managed__  int w;      // ERROR,   __managed__ variable  declaration within a host function
        __constant__ int h;      // ERROR,   __constant__ variable declaration within a host function
        extern __device__ int k; // CORRECT, extern __device__ variable
    }
    

The `__device__`, `__constant__`, and `__managed__` memory space specifiers are not allowed on variable declarations that are neither `extern` nor `static` within a function that executes on the device.
    
    
    __device__ void device_function() {
        int x;                   // CORRECT, __device__ variable
        __constant__      int y; // ERROR,   __constant__ variable declaration within a device function
        __managed__       int z; // ERROR,   __managed__ variable  declaration within a device function
        extern __device__ int k; // CORRECT, extern __device__ variable
    }
    

see also the [static variables](#static-variables) section.

#### 5.3.9.4.2. `const`-qualified Variables

A `const`-qualified variable without memory space annotations (`__device__` or `__constant__`) declared at global, namespace, or class scope is considered to be a host variable. Device code cannot contain a reference or take the address of the variable.

The variable may be directly used in device code, if

  * it has been initialized with a constant expression before the point of use,

  * the type is not `volatile`-qualified, and

  * it has one of the following types:

    * built-in integral type, or

    * built-in floating point type, except when the host compiler is Microsoft Visual Studio.


Starting with C++14, it is recommended to use `constexpr` or `inline constexpr` (C++17) variables instead of `const`-qualified ones. `constexpr` variables are not subject to the same type restrictions and can be utilized directly in device code.

`__managed__` variables don’t support `const`-qualified types.

Examples:
    
    
    const            int   ConstVar          = 10;
    const            float ConstFloatVar     = 5.0f;
    inline constexpr float ConstexprFloatVar = 5.0f; // C++17
    
    struct MyStruct {
        static const            int   ConstVar          = 20;
    //  static const             float ConstFloatVar     = 5.0f; // ERROR, static const variables cannot be float
        static inline constexpr float ConstexprFloatVar = 5.0f; // CORRECT
    };
    
    extern const int ExternVar;
    
    __device__ void foo() {
        int array1[ConstVar];                     // CORRECT
        int array2[MyStruct::ConstVar];           // CORRECT
    
        const     float var1 = ConstFloatVar;     // CORRECT, except when the host compiler is Microsoft Visual Studio.
        constexpr float var2 = ConstexprFloatVar; // CORRECT
    //  int             var3 = ExternVar;          // ERROR, "ExternVar" is not initialized with a constant expression
    //  int&            var4 = ConstVar;           // ERROR, reference to host variable
    //  int*            var5 = &ConstVar;          // ERROR, address of host variable
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/eWG8KxK94).

#### 5.3.9.4.3. `volatile`-qualified Variables

Note

The `volatile` keyword is supported to maintain compatibility with ISO C++. However, few, if any, of its [remaining non-deprecated uses](https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2018/p1152r0.html#prop) apply to GPUs.

Reading and writing to `volatile`-qualified objects are not atomic and are compiled into one or more [volatile instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#volatile-operation) that do not guarantee:

  * ordering of memory operations, or

  * that the number of memory operations performed by the hardware matches the number of PTX instructions.


CUDA C++ `volatile` is NOT suitable for:

  * **Inter-Thread Synchronization** : Use atomic operations via [cuda::atomic_ref](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic_ref.html), [cuda::atomic](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/atomic.html), or [Atomic Functions](cpp-language-extensions.html#atomic-functions) instead.

Atomic memory operations provide inter-thread synchronization guarantees and deliver better performance than `volatile` operations. However, CUDA C++ `volatile` operations do not provide any inter-thread synchronization guarantees and are therefore not suitable for this purpose. The following example shows how to pass a message between two threads using atomic operations.

cuda::atomic_ref
        
        #include <cuda/atomic>
        
        __global__ void kernel(int* flag, int* data) {
            cuda::atomic_ref<int, cuda::thread_scope_device> atomic_ref{*flag};
            if (threadIdx.x == 0) {
                // Consumer: blocks until flag is set by producer, then reads data
                while(atomic_ref.load(cuda::memory_order_acquire) == 0)
                    ;
                if (*data != 42)
                    __trap(); // Errors if wrong data read
            }
            else if (threadIdx.x == 1) {
                // Producer: writes data then sets flag
                *data = 42;
                atomic_ref.store(1, cuda::memory_order_release);
            }
        }
          
  
---  
  
cuda::atomic
        
        #include <cuda/atomic>
        
        __global__ void kernel(cuda::atomic<int, cuda::thread_scope_device>* flag, int* data) {
            if (threadIdx.x == 0) {
                // Consumer: blocks until flag is set by producer, then reads data
                while(flag->load(cuda::memory_order_acquire) == 0)
                    ;
                if (*data != 42)
                    __trap(); // Errors if wrong data read
            }
            else if (threadIdx.x == 1) {
                // Producer: writes data then sets flag
                *data = 42;
                flag->store(1, cuda::memory_order_release);
            }
        }
          
  
---  
  
Atomic Functions (`atomicAdd` and `atomicExch`)
        
        __global__ void kernel(int* flag, int* data) {
            if (threadIdx.x == 0) {
                // Consumer: blocks until flag is set by producer, then reads data
                while(atomicAdd(flag, 0) == 0)
                    ;                // Load with Relaxed Read-Modify-Write
                __threadfence();     // SequentiallyConsistent fence
                if (*data != 42)
                    __trap();        // Errors if wrong data read
            } else if (threadIdx.x == 1) {
                // Producer: writes data then sets flag
                *data = 42;
                __threadfence();     // SequentiallyConsistent fence
                atomicExch(flag, 1); // Store with Relaxed Read-Modify-Write
            }
        }
          
  
---  
  
  * **Memory Mapped IO** (MMIO): Use [PTX MMIO operations](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mmio-operation) via inline PTX instead.

PTX MMIO operations strictly preserve the number of memory accesses performed. However, CUDA C++ `volatile` operations do not preserve the number of memory accesses performed and may perform more or fewer accesses than requested in an undetermined way. This makes them unsuitable for MMIO. The following example shows how to read from and write to a register using PTX MMIO operations.
        
        __global__ void kernel(int* mmio_reg0, int* mmio_reg1) {
            // Write to MMIO register:
            int value = 13;
            asm volatile("st.relaxed.mmio.sys.u32 [%0], %1;"
                :
                : "l"(mmio_reg0), "r"(value) : "memory");
        
            // Read MMIO register:
            asm volatile("ld.relaxed.mmio.sys.u32 %0, [%1];"
                : "=r"(value)
                : "l"(mmio_reg1) : "memory");
        
            if (value != 42)
                __trap(); // Errors if wrong data read
        }
        


#### 5.3.9.4.4. `static` Variables

`static` variables are allowed in device code in the following cases:

  * Within `__global__` or `__device__`-only functions.

  * Within `__host__ __device__` functions:

    * `static` variables without an explicit memory space (automatic deduction).

    * `static` variables with an explicit memory space, such as `static __device__/__constant__/__shared__/__managed__`, are allowed only when `__CUDA_ARCH__` is defined.


A `static` variable within a `__host__ __device__` function holds a different value depending on the execution space.

Examples of legal and illegal uses of function-scope `static` variables are shown below.
    
    
    struct TrivialStruct {
        int x;
    };
    
    struct NonTrivialStruct {
        __device__ NonTrivialStruct(int x) {}
    };
    
    __device__ void device_function(int x) {
        static int v1;              // CORRECT, implicit __device__ memory space specifier
        static int v2 = 11;         // CORRECT, implicit __device__ memory space specifier
    //  static int v3 = x;           // ERROR, dynamic initialization is not allowed
    
        static __managed__  int v4; // CORRECT, explicit
        static __device__   int v5; // CORRECT, explicit
        static __constant__ int v6; // CORRECT, explicit
        static __shared__   int v7; // CORRECT, explicit
    
        static TrivialStruct    s1;     // CORRECT, implicit __device__ memory space specifier
        static TrivialStruct    s2{22}; // CORRECT, implicit __device__ memory space specifier
    //  static TrivialStruct    s3{x};   // ERROR, dynamic initialization is not allowed
    //  static NonTrivialStruct s4{3};   // ERROR, dynamic initialization is not allowed
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/TdYKaTq3f).

* * *
    
    
    __host__ __device__ void host_device_function() {
        static            int v1; // CORRECT, implicit __device__ memory space specifier
    //  static __device__ int v2;  // ERROR, __device__-only variable inside a host-device function
    #ifdef __CUDA_ARCH__
        static __device__ int v3; // CORRECT, declaration is only visible during device compilation
    #else
        static int v4;            // CORRECT, declaration is only visible during host compilation
    #endif
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/18qhjn8P1).

* * *
    
    
    #include <cassert>
    
    __host__ __device__ int host_device_function() {
        static int v = 0;
        v++;
        return v;
    }
    
    __global__ void kernel() {
        int ret = host_device_function(); // v = 1
        assert(ret == 4);                 // FAIL
    }
    
    int main() {
        host_device_function();           // v = 1
        host_device_function();           // v = 2
        int ret = host_device_function(); // v = 3
        assert(ret == 3);                 // OK
        kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/Wqo9WjvYY).

#### 5.3.9.4.5. `extern` Variables

When compiling in the [whole program compilation mode](../02-basics/nvcc.html#nvcc-separate-compilation), `__device__`, `__shared__`, `__managed__`, and `__constant__` variables cannot be defined with external linkage using the `extern` keyword.

The only exception is for dynamically allocated `__shared__` variables as described in the [Dynamic Allocation of Shared Memory](../02-basics/writing-cuda-kernels.html#writing-cuda-kernels-dynamic-allocation-shared-memory) section.
    
    
    __device__        int x; // OK
    extern __device__ int y; // ERROR in whole program compilation mode
    extern __shared__ int z; // OK
    

### 5.3.9.5. Functions

#### 5.3.9.5.1. Recursion

`__global__` functions do not support recursion, while `__device__` and `__host__ __device__` functions do not have such restriction.

#### 5.3.9.5.2. External Linkage

Device variables or functions with external linkage require [separate compilation mode](../02-basics/nvcc.html#nvcc-separate-compilation) across multiple translation units.

In separate compilation mode, if a `__device__` or `__global__` function definition is required to exist in a particular translation unit, then the parameters and return types of the function must be complete in that translation unit. The concept is also known as One Definition Rule-use, or ODR-use.

Example:
    
    
    //first.cu:
    struct S;                   // forward declaration
    __device__ void foo(S);     // ERROR, type 'S' is an incomplete type
    __device__ auto* ptr = foo; // ODR-use, address taken
    
    int main() {}
    
    
    
    //second.cu:
    struct S {};               // struct definition
    __device__ void foo(S) {}  // function definition
    
    
    
    # compiler invocation
    $ nvcc -std=c++14 -rdc=true first.cu second.cu -o prog
    nvlink error   : Prototype doesn't match for '_Z3foo1S' in '/tmp/tmpxft_00005c8c_00000000-18_second.o',
                     first defined in '/tmp/tmpxft_00005c8c_00000000-18_second.o'
    nvlink fatal   : merge_elf failed
    

#### 5.3.9.5.3. Formal Parameters

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on formal parameters.
    
    
    void device_function1(__device__ int x) { } // ERROR, __device__ parameter
    void device_function2(__shared__ int x) { } // ERROR, __shared__ parameter
    

#### 5.3.9.5.4. `__global__` Function Parameters

A `__global__` function has the following restrictions:

  * It cannot have a variable number of arguments, namely the C ellipsis syntax `...` and the `va_list` type. C++11 variadic template is allowed, subject to the restrictions described in the [__global__ Variadic Template](#cpp11-variadic-template) section.

  * Function parameters are passed to the device via [constant memory](device-callable-apis.html#constant-memory) and their total size is limited to 32,764 bytes.

  * Function parameters cannot be pass-by-reference or by pass-by-rvalue reference.

  * Function parameters cannot be of type `std::initializer_list`.

  * Polymorphic class parameters (`virtual`) are considered undefined behavior.

  * Lambda expressions and closure types are allowed, subject to the restrictions described in the [Lambda Expressions and __global__ Function Parameters](#lambda-expressions-global) section.


#### 5.3.9.5.5. `__global__` Function Arguments Passing

When launching a `__global__` function [from device code](../02-basics/intro-to-cuda-cpp.html#intro-cpp-launching-kernels), each argument must be trivially copyable and trivially destructible.

When a `__global__` function is launched from host code, each argument type may be non-trivially copyable or non-trivially destructible. However, the processing of these types does not follow the standard C++ model, as described below. The user code must ensure that this workflow does not affect program correctness. The workflow diverges from standard C++ in two areas:

  1. **Raw memory copy instead of copy constructor invocation**

The CUDA Runtime passes the kernel arguments to the `__global__` function by copying the raw memory content, eventually using `memcpy`. If an argument is non-trivially copyable and provides a user-defined copy constructor, the operations and side effects of the invocation are skipped in the host-to-device copy.

Example:
         
         #include <cassert>
         
         struct MyStruct {
             int  value = 1;
             int* ptr;
         
             MyStruct() = default;
         
             __host__ __device__ MyStruct(const MyStruct&) { ptr = &value; }
         };
         
         __global__ void device_function(MyStruct my_struct) {
             // this assert fails because "my_struct" is obtained by copying
             // the raw memory content and the copy constructor is skipped.
             assert(my_struct.ptr == &my_struct.value); // FAIL
         }
         
         void host_function(MyStruct my_struct) {
             assert(my_struct.ptr == &my_struct.value); // CORRECT
         }
         
         int main() {
             MyStruct my_struct;
             host_function(my_struct);
             device_function<<<1, 1>>>(my_struct); // copy constructor invoked in the host-side only
             cudaDeviceSynchronize();
         }
         

See the example on [Compiler Explorer](https://godbolt.org/z/xhqe16dec).

  2. **Destructor may be invoked before the** `__global__` **function has finished**

Kernel launches are asynchronous with host execution. As a result, if a `__global__` function argument has a non-trivial destructor, the destructor may execute in host code even before the `__global__` function has finished execution. This may break programs where the destructor has side effects.

Example:
         
         #include <cassert>
         
         __managed__ int var = 0;
         
         struct MyStruct {
             __host__ __device__ ~MyStruct() { var = 3; }
         };
         
         __global__ void device_function(MyStruct my_struct) {
             assert(var == 0); // FAIL, MyStruct::~MyStruct() sets the value to 3
         }
         
         int main() {
             MyStruct my_struct;
             // GPU kernel execution is asynchronous with host execution.
             // As a result, MyStruct::~MyStruct() could be executed before
             // the kernel finishes executing.
             device_function<<<1, 1>>>(my_struct);
             cudaDeviceSynchronize();
         }
         

See the example on [Compiler Explorer](https://godbolt.org/z/cn6Y5W6zs).


### 5.3.9.6. Classes

#### 5.3.9.6.1. Class-type Variables

A variable definition with `__device__`, `__constant__`, `__managed__` or `__shared__` memory space cannot have a class type with a non-empty constructor or a non-empty destructor. A constructor for a class type is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

  * The constructor function has been defined.

  * The constructor function has no parameters, an empty initializer list, and an empty compound statement function body.

  * Its class has no `virtual` functions, `virtual` base classes, or non-`static` data member initializers.

  * The default constructors of all of its base classes can be considered empty.

  * For all non-`static` data members of the class that are of a class type (or an array thereof), the default constructors can be considered empty.


A class’s destructor is considered empty if it is either trivial or satisfies all of the following conditions at a point in the translation unit:

  * The destructor function has been defined.

  * The destructor function body is an empty compound statement.

  * Its class has no `virtual` functions or `virtual` base classes.

  * The destructors of all of its base classes can be considered empty.

  * For all non-`static` data members of the class that are of a class type (or an array thereof), the destructor can be considered empty.


#### 5.3.9.6.2. Data Members

The `__device__`, `__shared__`, `__managed__` and `__constant__` memory space specifiers are not allowed on `class`, `struct`, and `union` data members.

Only `static` data members evaluated at compile time are supported, such as [const-qualified](#const-variables) and `constexpr` variables.
    
    
    struct MyStruct {
       static inline constexpr int value1 = 10; // C++17
       static constexpr        int value2 = 10; // C++11
       static const            int value3 = 10;
    // static                  int value4; // ERROR
    };
    

#### 5.3.9.6.3. Function Members

`__global__` functions cannot be members of a `struct`, `class`, or `union`.

A `__global__` function is allowed in a `friend` declaration, but cannot be defined.

Example:
    
    
    struct MyStruct {
        friend __global__ void f();   // CORRECT, friend declaration only
    
    //  friend __global__ void g() {} // ERROR, friend definition
    };
    

See the example on [Compiler Explorer](https://godbolt.org/z/rv6cP3b9j).

#### 5.3.9.6.4. Implicitly-Declared and Non-Virtual Explicitly-Defaulted functions

Implicitly-declared special member functions are those the compiler declares for a class when the user does not declare them; Explicitly-defaulted functions are ones the user declares but marks with `= default`. The special member functions that are implicitly-declared or explicitly-defaulted are default constructor, copy constructor, move constructor, copy assignment operator, move assignment operator, and destructor.

Let `F` denote a non-`virtual` function that is either implicitly declared or explicitly defaulted on its first declaration. The execution space specifiers for `F` are the union of the execution space specifiers of all functions that invoke it. Note that for this analysis, a `__global__` caller will be treated as a `__device__` caller. For example:
    
    
    class Base {
        int x;
    public:
        __host__ __device__ Base() : x(10) {}
    };
    
    class Derived : public Base {
        int y;
    };
    
    class Other: public Base {
        int z;
    };
    
    __device__ void foo() {
        Derived D1;
        Other D2;
    }
    
    __host__ void bar() {
        Other D3;
    }
    

In this case, the implicitly declared constructor function `Derived::Derived()` will be treated as a `__device__` function because it is only invoked from the `__device__` function `foo()`. The implicitly declared constructor function `Other::Other()` will be treated as a `__host__ __device__` function since it is invoked both from both a `__device__` function `foo()` and a `__host__` function `bar()`.

Additionally, if `F` is an implicitly-declared `virtual` function (for example, a `virtual` destructor), the execution spaces of each virtual function `D` that is overridden by `F` are added to the set of execution spaces for `F` if `D` not implicitly-declared.

For example:
    
    
    struct Base1 {
        virtual __host__ __device__ ~Base1() {}
    };
    
    struct Derived1 : Base1 {}; // implicitly-declared virtual destructor
                                // ~Derived1() has __host__ __device__  execution space specifiers
    
    struct Base2 {
        virtual __device__ ~Base2() = default;
    };
    
    struct Derived2 : Base2 {}; // implicitly-declared virtual destructor
                                // ~Derived2() has __device__ execution space specifiers
    

#### 5.3.9.6.5. Polymorphic Classes

Polymorphic classes, namely those with `virtual` functions, derived from other polymorphic classes, or with polymorphic data members, are subject to the following restrictions:

  * Copying polymorphic objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.

  * The execution space of an overridden `virtual` function must match the execution space of the function in the base class.


Example:
    
    
    struct MyClass {
        virtual __host__ __device__ void f() {}
    };
    
    __global__ void kernel(MyClass my_class) {
        my_class.f(); // undefined behavior
    }
    
    int main() {
        MyClass my_class;
        kernel<<<1, 1>>>(my_class);
        cudaDeviceSynchronize();
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/To39sGTrW).

* * *
    
    
    struct BaseClass {
        virtual __host__ __device__ void f() {}
    };
    
    struct DerivedClass : BaseClass {
        __device__ void f() override {} // ERROR
    };
    

See the example on [Compiler Explorer](https://godbolt.org/z/xfKhEGfdG).

#### 5.3.9.6.6. Windows-Specific Class Layout

The CUDA compiler follows the IA64 ABI for class layout, while Microsoft Visual Studio does not. This prevents bitwise copy of special objects between host and device code as described below.

Let `T` denote a pointer to member type, or a class type that satisfies any of the following conditions:

  * `T` is a [polymorphic class](#polymorphic-classes)

  * `T` has multiple inheritance with more than one direct or indirect [empty base class](#class-type-variables).

  * All direct and indirect base classes `B` are [empty](#class-type-variables) and the type of the first field `F` of `T` uses `B` in its definition, such that `B` is laid out at offset 0 in the definition of `F`.


Classes of type `T`, with a base class of type `T`, or with data members of type `T`, may have a different class layout and size between host and device when compiled with Microsoft Visual Studio.

Copying such objects from device to host or from host to device, including `__global__` function arguments is undefined behavior.

### 5.3.9.7. Templates

A type cannot be used as template argument of a `__global__` function or a `__device__/__constant__` variable (C++14) if either:

  * The type is defined within a `__host__` or `__host__ __device__` function scope.

  * The type is unnamed, such as an anonymous struct or a lambda expression, unless the type is local to a `__device__` or `__global__` function.

  * The type is a class member with `private` or `protected`, unless the class is local to a `__device__` or `__global__` function.

  * The type is compounded from any of the types above.


Example:
    
    
    template <typename T>
    __global__ void kernel() {}
    
    template <typename T>
    __device__ int device_var; // C++14
    
    struct {
        int v;
    } unnamed_struct;
    
    void host_function() {
        struct LocalStruct {};
    //  kernel<LocalStruct><<<1, 1>>>(); // ERROR, LocalStruct is defined within a host function
        int data = 4;
    //  cudaMemcpyToSymbol(device_var<LocalStruct>, &data, sizeof(data)); // ERROR, same as above
    
        auto lambda = [](){};
    //  kernel<decltype(lambda)><<<1, 1>>>();         // ERROR, unnamed type
    //  kernel<decltype(unnamed_struct)><<<1, 1>>>(); // ERROR, unnamed type
    }
    
    class MyClass {
    private:
        struct PrivateStruct {};
    public:
        static void launch() {
    //      kernel<PrivateStruct><<<1, 1>>>(); // ERROR, private type
        }
    };
    

See the example on [Compiler Explorer](https://godbolt.org/z/EhTn3GT3z).

## 5.3.10. C++11 Restrictions

### 5.3.10.1. `inline` Namespaces

It is not allowed to define one of the following entities within an `inline` namespace when another entity of the same name and type signature is defined in an enclosing namespace:

  * `__global__` function.

  * `__device__`, `__constant__`, `__managed__`, `__shared__` variables.

  * Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.


Example:
    
    
    __device__ int my_var; // global scope
    
    inline namespace NS {
    
    __device__ int my_var; // namespace scope
    
    } // namespace NS
    

### 5.3.10.2. `inline` Unnamed Namespaces

The following entities cannot be declared in namespace scope within an `inline` unnamed namespace:

  * `__global__` function.

  * `__device__`, `__constant__`, `__managed__`, `__shared__` variables.

  * Variables with surface or texture type, such as `cudaSurfaceObject_t` or `cudaTextureObject_t`.


### 5.3.10.3. `constexpr` Functions

By default, a `constexpr` function cannot be called from a function with incompatible execution space, in the same way as standard functions.

  * Calling a device-only `constexpr` function from a host-function during host code generation phase, namely when `__CUDA_ARCH__` macro is undefined. Example:
        
        constexpr __device__ int device_function() { return 0; }
        
        int main() {
            int x = device_function();  // ERROR, calling a device-only constexpr function from host code
        }
        

  * Calling a host-only `constexpr` function from a `__device__` or `__global__` function, during device code generation phase, namely when `__CUDA_ARCH__` macro is defined. Example:
        
        constexpr int host_function() { return 0; }
        
        __device__ void device_function() {
            int x = host_function();  // ERROR, calling a host-only constexpr function from device code
        }
        


Note that a function template specialization may not be a `constexpr` function even if the corresponding template function is marked with the keyword `constexpr`.

**Relaxed constexpr-Function Support**

The experimental `nvcc` flag `--expt-relaxed-constexpr` can be used to relax this constraint for both `__host__` and `__device__` functions. However, a `__global__` function cannot be declared as `constexpr`. `nvcc` will also define the macro `__CUDACC_RELAXED_CONSTEXPR__`.

When this flag is specified, the compiler will support cross execution space calls described above, as follows:

  1. A call to a `constexpr` function in a cross-execution space is supported if it occurs in a context that requires constant evaluation, such as the initializer of a `constexpr` variable. Example:
         
         constexpr __host__ int host_function(int x) { return x + 1; };
         
         __global__ void kernel() {
             constexpr int val = host_function(1); // CORRECT, the call is in a context that requires constant evaluation.
         }
         
         constexpr __device__ int device_function(int x) { return x + 1; }
         
         int main() {
             constexpr int val = device_function(1); // CORRECT, the call is in a context that requires constant evaluation.
         }
         

  2. Device code is generated during device code generation for the body of a host-only constexpr function, unless it is not used or is only called in a `constexpr` context. Example:
         
         // NOTE: "host_function" is emitted in generated device code because
         //       it is called from device code in a non-constexpr context
         constexpr int host_function(int x) { return x + 1; }
         
         __device__ int device_function(int in) {
             return host_function(in);  // CORRECT, even though argument is not a constant expression
         }
         

  3. All code restrictions that apply to a device function also apply to the `constexpr` host-only function called from the device code. However, the compiler may not emit any build-time diagnostics for restrictions related to the compilation process.

For example, the following code patterns are not supported in the body of the host function. This is similar to any device function; however, no compiler diagnostic may be generated.

     * One-Definition Rule (ODR)-use of a host variable or host-only non-`constexpr` function. Example:
           
           int host_var1, host_var2;
           
           constexpr int* host_function(bool b) { return b ? &host_var1 : &host_var2; };
           
           __device__ int device_function(bool flag) {
               return *host_function(flag); // ERROR, host_function() attempts to refer to the host variables
                                            //        'host_var1' and 'host_var2'.
                                            //        The code will compile, but will NOT execute correctly.
           }
           

     * Use of exceptions `throw/catch` and Run-Time Type Information `typeid/dynamic_cast`. Example:
           
           struct Base { };
           struct Derived : public Base { };
           
           // NOTE: "host_function" is emitted in generated device code
           constexpr int host_function(bool b, Base *ptr) {
               if (b) {
                   return 1;
               }
               else if (typeid(ptr) == typeid(Derived)) { // ERROR, use of typeid in code executing on the GPU
                   return 2;
               }
               else {
                   throw int{4}; // ERROR, use of throw in code executing on the GPU
               }
           }
           
           __device__ void device_function(bool flag) {
               Derived d;
               int val = host_function(flag, &d); //ERROR, host_function() attempts use typeid and throw(),
                                                  //       which are not allowed in code that executes on the GPU
           }
           

  4. During host code generation, the body of a device-only `constexpr` function is preserved in the code sent to the host compiler. However, if the body of a device function attempts to ODR-use a namespace-scope device variable or a non-`constexpr` device function, the call to the device function from host code is not supported. While the code may build without compiler diagnostics, it may behave incorrectly at runtime. Example:
         
         __device__ int device_var1, device_var2;
         
         constexpr __device__ int* device_function(bool b) { return b ? &device_var1 : &device_var2; };
         
         int host_function(bool flag) {
             return *device_function(flag); // ERROR, device_function() attempts to refer to device variables
                                            //        'device_var1' and 'device_var2'
                                            // The code will compile, but will NOT execute correctly.
         }
         


Warning

Due to the above restrictions and the lack of compiler diagnostics for incorrect usage, it is recommended to avoid calling a function in the Standard C++ headers `std::` from device code. The implementation of such functions varies depending on the host platform. Instead, it is strongly suggested to call the equivalent functionality in the CUDA C++ Standard Library [libcu++](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#libcu), in the `cuda::std::` namespace.

### 5.3.10.4. `constexpr` Variables

By default, a `constexpr` variable cannot be used in a function with incompatible execution space, in the same way of standard variables.

A `constexpr` variable can be directly used in device code in the following cases:

  * C++ scalar types, excluding pointer and pointer-to-member types:

    * `nullptr_t`.

    * `bool`.

    * Integral types: `char`, `signed char`, `unsigned`, `long long`, etc.

    * Floating point types: `float`, `double`.

    * Enumerators: `enum` and `enum class`.

  * Class types: `class`, `struct`, and `union` with a `constexpr` constructor.

  * Raw array of the types above, for example `int[]`, only when they are used inside a `constexpr` `__device__` or `__host__ __device__` function.


`constexpr __managed__` and `constexpr __shared__` variables are not allowed.

Examples:
    
    
    constexpr int ConstexprVar = 4; // scalar type
    
    struct MyStruct {
        static constexpr int ConstexprVar = 100;
    };
    
    constexpr MyStruct my_struct = MyStruct{}; // class type
    
    constexpr int array[] = {1, 2, 3};
    
    __device__ constexpr int get_value(int idx) {
        return array[idx];                      // CORRECT
    }
    
    __device__ void foo(int idx) {
        int        v1 = ConstexprVar;           // CORRECT
        int        v2 = MyStruct::ConstexprVar; // CORRECT
    //  const int &v3 = ConstexprVar1;          // ERROR, reference to host constexpr variable
    //  const int *v4 = &ConstexprVar1;         // ERROR, address of host constexpr variable
        int        v5 = get_value(2);           // CORRECT, 'get_value(2)' is a constant expression.
    //  int        v6 = get_value(idx);         // ERROR, 'get_value(idx)' is not a constant expression
    //  int        v7 = array[2];               // ERROR, 'array' is not scalar type.
        MyStruct   v8 = my_struct;              // CORRECT
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/MWa1o3c9z).

### 5.3.10.5. `__global__` Variadic Template

A variadic `__global__` function template has the following restrictions:

  * Only a single pack parameter is allowed.

  * The pack parameter must be listed last in the template parameter list.


Examples:
    
    
    template <typename... Pack>
    __global__ void kernel1(); // CORRECT
    
    // template <typename... Pack, template T>
    // __global__ void kernel2(); // ERROR, parameter pack is not the last parameter
    
    template <typename... TArgs>
    struct MyStruct {};
    
    // template <typename... Pack1, typename... Pack2>
    // __global__ void kernel3(MyStruct<Pack1...>, MyStruct<Pack2...>); // ERROR, more than one parameter pack
    

See the example on [Compiler Explorer](https://godbolt.org/z/x48KnPbbY).

### 5.3.10.6. Defaulted Functions `= default`

The CUDA compiler infers the execution space of explicitly-defaulted member functions as described in [Implicitly-declared and explicitly-defaulted functions](#compiler-generated-functions).

Execution space specifiers on explicitly-defaulted functions are ignored by the compiler, except in the case the function is defined out-of-line or is a `virtual` function.

Examples:
    
    
    struct MyStruct1 {
        MyStruct1() = default;
    };
    
    void host_function() {
        MyStruct1 my_struct; // __host__ __device__ constructor
    }
    
    __device__ void device_function() {
        MyStruct1 my_struct; // __host__ __device__ constructor
    }
    
    struct MyStruct2 {
        __device__ MyStruct2() = default; // WARNING: __device__ annotation is ignored
    };
    
    struct MyStruct3 {
        __host__ MyStruct3();
    };
    MyStruct3::MyStruct3() = default; // out-of-line definition, not ignored
    
    __device__ void device_function2() {
    //  MyStruct3 my_struct; // ERROR, __host__ constructor
    }
    
    struct MyStruct4 {
        //  MyStruct4::~MyStruct4 has host execution space, not ignored because virtual
        virtual __host__ ~MyStruct4() = default;
    };
    
    __device__ void device_function3() {
        MyStruct4 my_struct4;
        // implicit destructor call for 'my_struct4':
        //    ERROR: call from a __device__ function 'device_function3' to a
        //    __host__ function 'MyStruct4::~MyStruct4'
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/q1M4j8YYf).

### 5.3.10.7. `[cuda::]std::initializer_list`

By default, the CUDA compiler implicitly considers the member functions of `[cuda::]std::initializer_list` to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code.

The `nvcc` flag `--no-host-device-initializer-list` disables this behavior; member functions of `[cuda::]std::initializer_list` will then be considered as `__host__` functions and will not be directly invocable from device code.

A `__global__` function cannot have a parameter of type `[cuda::]std::initializer_list`.

Example:
    
    
    #include <initializer_list>
    
    __device__ void foo(std::initializer_list<int> in) {}
    
    __device__ void bar() {
        foo({4,5,6}); // (a) initializer list containing only constant expressions.
        int i = 4;
        foo({i,5,6}); // (b) initializer list with at least one  non-constant element.
                      // This form may have better performance than (a).
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/xeah7r44T).

### 5.3.10.8. `[cuda::]std::move`, `[cuda::]std::forward`

By default, the CUDA compiler implicitly considers `std::move` and `std::forward` function templates to have `__host__ __device__` execution space specifiers, and therefore they can be invoked directly from device code. The `nvcc` flag `--no-host-device-move-forward` disables this behavior; `std::move` and `std::forward` will then be considered as `__host__` functions and will not be directly invocable from device code.

Hint

`cuda::std::move` and `cuda::std::forward` on the contrary always have `__host__ __device__` execution space.

## 5.3.11. C++14 Restrictions

### 5.3.11.1. Functions with Deduced Return Type

A `__global__` function cannot have a deduced return type `auto`.

Introspection of the return type of a `__device__` function with a deduced return type is not allowed in host code.

Note

The CUDA frontend compiler changes the function declaration to have a `void` return type, before invoking the host compiler. This may break introspection of the deduced return type of the `__device__` function in host code. Thus, the CUDA compiler will issue a compile-time error for referencing such a deduced return type outside of device function bodies.

Examples:
    
    
     __device__ auto device_function(int x) { // deduced return type
         return x;                            // decltype(auto) has the same behavior
     }
    
     __global__ void kernel() {
         int x = sizeof(device_function(2));         // CORRECT, device code scope
     }
    
     // const int size = sizeof(device_function(2)); // ERROR, return type deduction on host
    
     void host_function() {
     //  using T = decltype(device_function(2));     // ERROR, return type deduction on host
     }
    
    void host_fn1() {
      // ERROR, referenced outside device function bodies
      int (*p1)(int) = fn1;
    
      struct S_local_t {
        // ERROR, referenced outside device function bodies
        decltype(fn2(10)) m1;
    
        S_local_t() : m1(10) { }
      };
    }
    
    // ERROR, referenced outside device function bodies
    template <typename T = decltype(fn2)>
    void host_fn2() { }
    
    template<typename T> struct MyStruct { };
    
    // ERROR, referenced outside device function bodies
    struct S1_derived_t : MyStruct<decltype(fn1)> { };
    

### 5.3.11.2. Variable Templates

A `__device__` or `__constant__` variable template cannot be `const`-qualified when using the Microsoft compiler.

Examples:
    
    
    // ERROR on Windows (non-portable), const-qualified
    template <typename T>
    __device__ const T var = 0;
    
     // CORRECT, ptr1 is not const-qualified
    template <typename T>
    __device__ const T* ptr1 = nullptr;
    
    // ERROR on Windows (non-portable), ptr2 is const-qualified
    template <typename T>
    __device__ const T* const ptr2 = nullptr;
    

See the example on [Compiler Explorer](https://godbolt.org/z/8hM5Yh7db).

## 5.3.12. C++17 Restrictions

### 5.3.12.1. `inline` Variables

In a single translation unit, using an `inline` variable provides no additional functionality beyond a regular variable and does not provide any practical advantage.

`nvcc` allows `inline` variables with `__device__`, `__constant__`, or `__managed__` memory space only in [Separate Compilation](../02-basics/nvcc.html#nvcc-separate-compilation) mode or for variables with internal linkage.

Note

When using `gcc/g++` host compiler, an `inline` variable declared with `__managed__` memory space specifier may not be visible to the debugger.

Examples:
    
    
    inline        __device__ int device_var1;  // CORRECT, when compiled in Separate Compilation mode (-rdc=true or -dc)
                                               // ERROR, when compiled in Whole Program Compilation mode
    
    static inline __device__ int device_var2;  // CORRECT, internal linkage
    
    namespace {
    
    inline __device__ int device_var3;         // CORRECT, internal linkage
    
    inline __shared__ int shared_var;          // CORRECT, internal linkage
    
    static inline __device__ int device_var4;  // CORRECT, internal linkage
    
    inline __device__ int device_var5;         // CORRECT, internal linkage
    
    } // namespace
    

See the example on [Compiler Explorer](https://godbolt.org/z/oraqeGTzY).

### 5.3.12.2. Structured Binding

A structured binding cannot be declared with a memory space specifier, such as `__device__`, `__shared__`, `__constant__`, or `__managed__`.

Example:
    
    
    struct S {
        int x, y;
    };
    // __device__ auto [a, b] = S{4, 5}; // ERROR
    

## 5.3.13. C++20 Restrictions

### 5.3.13.1. Three-way Comparison Operator

The three-way comparison operator (`<=>`) is supported in device code, but some uses implicitly rely on functionality from the C++ Standard Library, which is provided by the host implementation. Using those operators may require specifying the flag `--expt-relaxed-constexpr` to silence warnings, and the functionality requires the host implementation to satisfy the requirements of the device code.

Examples:
    
    
    #include <compare> // std::strong_ordering implementation
    
    struct S {
        int x, y;
    
        auto operator<=>(const S&) const = default; // (a)
    
        __host__ __device__ bool operator<=>(int rhs) const { return false; } // (b)
    };
    
    __host__ __device__ bool host_device_function(S a, S b) {
        if (a <=> 1)  // CORRECT, calls a user-defined host-device overload (b)
            return true;
        return a < b; // CORRECT, call to an implicitly-declared function (a)
                      // Note: it requires a device-compatible std::strong_ordering
                      //       implementation provided in the header <compare>
                      //       and the flag --expt-relaxed-constexpr
    }
    

See the example on [Compiler Explorer](https://godbolt.org/z/qzs5arfx4).

### 5.3.13.2. `consteval` Functions

`consteval` functions can be called from both host and device code, independently of their execution space.

Examples:
    
    
    consteval int host_consteval() {
        return 10;
    }
    
    __device__ consteval int device_consteval() {
        return 10;
    }
    
    __device__ int device_function() {
        return host_consteval();   // CORRECT, even if called from device code
    }
    
    __host__ __device__ int host_device_function() {
        return device_function();  // CORRECT, even if called from host-device code
    }
