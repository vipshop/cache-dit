# 6. Instruction Operands


##  6.1. [Operand Type Information](#operand-type-information)  
  
All operands in instructions have a known type from their declarations. Each operand type must be compatible with the type determined by the instruction template and instruction type. There is no automatic conversion between types.

The bit-size type is compatible with every type having the same size. Integer types of a common size are compatible with each other. Operands having type different from but compatible with the instruction type are silently cast to the instruction type.


##  6.2. [Source Operands](#source-operands)

The source operands are denoted in the instruction descriptions by the names `a`, `b`, and `c`. PTX describes a load-store machine, so operands for ALU instructions must all be in variables declared in the `.reg` register state space. For most operations, the sizes of the operands must be consistent.

The `cvt` (convert) instruction takes a variety of operand types and sizes, as its job is to convert from nearly any data type to any other data type (and size).

The `ld`, `st`, `mov`, and `cvt` instructions copy data from one location to another. Instructions `ld` and `st` move data from/to addressable state spaces to/from registers. The `mov` instruction copies data between registers.

Most instructions have an optional predicate guard that controls conditional execution, and a few instructions have additional predicate source operands. Predicate operands are denoted by the names `p`, `q`, `r`, `s`.


##  6.3. [Destination Operands](#destination-operands)

PTX instructions that produce a single result store the result in the field denoted by `d` (for destination) in the instruction descriptions. The result operand is a scalar or vector variable in the register state space.


##  6.4. [Using Addresses, Arrays, and Vectors](#using-addresses-arrays-and-vectors)

Using scalar variables as operands is straightforward. The interesting capabilities begin with addresses, arrays, and vectors.

###  6.4.1. [Addresses as Operands](#addresses-as-operands)

All the memory instructions take an address operand that specifies the memory location being accessed. This addressable operand is one of:

`[var]`
    

the name of an addressable variable `var`.

`[reg]`
    

an integer or bit-size type register `reg` containing a byte address.

`[reg+immOff]`
    

a sum of register `reg` containing a byte address plus a constant integer byte offset (signed, 32-bit).

`[var+immOff]`
    

a sum of address of addressable variable `var` containing a byte address plus a constant integer byte offset (signed, 32-bit).

`[immAddr]`
    

an immediate absolute byte address (unsigned, 32-bit).

`var[immOff]`
    

an array element as described in [Arrays as Operands](#arrays-as-operands).

The register containing an address may be declared as a bit-size type or integer type.

The access size of a memory instruction is the total number of bytes accessed in memory. For example, the access size of `ld.v4.b32` is 16 bytes, while the access size of `atom.f16x2` is 4 bytes.

The address must be naturally aligned to a multiple of the access size. If an address is not properly aligned, the resulting behavior is undefined. For example, among other things, the access may proceed by silently masking off low-order address bits to achieve proper rounding, or the instruction may fault.

The address size may be either 32-bit or 64-bit. 128-bit adresses are not supported. Addresses are zero-extended to the specified width as needed, and truncated if the register width exceeds the state space address width for the target architecture.

Address arithmetic is performed using integer arithmetic and logical instructions. Examples include pointer arithmetic and pointer comparisons. All addresses and address computations are byte-based; there is no support for C-style pointer arithmetic.

The `mov` instruction can be used to move the address of a variable into a pointer. The address is an offset in the state space in which the variable is declared. Load and store operations move data between registers and locations in addressable state spaces. The syntax is similar to that used in many assembly languages, where scalar variables are simply named and addresses are de-referenced by enclosing the address expression in square brackets. Address expressions include variable names, address registers, address register plus byte offset, and immediate address expressions which evaluate at compile-time to a constant address.

Here are a few examples:
    
    
    .shared .u16 x;
    .reg    .u16 r0;
    .global .v4 .f32 V;
    .reg    .v4 .f32 W;
    .const  .s32 tbl[256];
    .reg    .b32 p;
    .reg    .s32 q;
    
    ld.shared.u16   r0,[x];
    ld.global.v4.f32 W, [V];
    ld.const.s32    q, [tbl+12];
    mov.u32         p, tbl;
    

####  6.4.1.1. [Generic Addressing](#generic-addressing)

If a memory instruction does not specify a state space, the operation is performed using generic addressing. The state spaces `.const`, [Kernel Function Parameters](#kernel-function-parameters) (`.param`), `.local` and `.shared` are modeled as windows within the generic address space. Each window is defined by a window base and a window size that is equal to the size of the corresponding state space. A generic address maps to `global` memory unless it falls within the window for `const`, `local`, or `shared` memory. The [Kernel Function Parameters](#kernel-function-parameters) (`.param`) window is contained within the `.global` window. Within each window, a generic address maps to an address in the underlying state space by subtracting the window base from the generic address.

###  6.4.2. [Arrays as Operands](#arrays-as-operands)

Arrays of all types can be declared, and the identifier becomes an address constant in the space where the array is declared. The size of the array is a constant in the program.

Array elements can be accessed using an explicitly calculated byte address, or by indexing into the array using square-bracket notation. The expression within square brackets is either a constant integer, a register variable, or a simple _register with constant offset_ expression, where the offset is a constant expression that is either added or subtracted from a register variable. If more complicated indexing is desired, it must be written as an address calculation prior to use. Examples are:
    
    
    ld.global.u32  s, a[0];
    ld.global.u32  s, a[N-1];
    mov.u32        s, a[1];  // move address of a[1] into s
    

###  6.4.3. [Vectors as Operands](#vectors-as-operands)

Vector operands can be specified as source and destination operands for instructions. However, when specified as destination operand, all elements in vector expression must be unique, otherwise behavior is undefined. Vectors may also be passed as arguments to called functions.

Vector elements can be extracted from the vector with the suffixes `.x`, `.y`, `.z` and `.w`, as well as the typical color fields `.r`, `.g`, `.b` and `.a`.

A brace-enclosed list is used for pattern matching to pull apart vectors.
    
    
    .reg .v4 .f32 V;
    .reg .f32     a, b, c, d;
    
    mov.v4.f32 {a,b,c,d}, V;
    

Vector loads and stores can be used to implement wide loads and stores, which may improve memory performance. The registers in the load/store operations can be a vector, or a brace-enclosed list of similarly typed scalars. Here are examples:
    
    
    ld.global.v4.f32  {a,b,c,d}, [addr+16];
    ld.global.v2.u32  V2, [addr+8];
    

Elements in a brace-enclosed vector, say {Ra, Rb, Rc, Rd}, correspond to extracted elements as follows:
    
    
    Ra = V.x = V.r
    Rb = V.y = V.g
    Rc = V.z = V.b
    Rd = V.w = V.a
    

###  6.4.4. [Labels and Function Names as Operands](#labels-and-function-names-as-operands)

Labels and function names can be used only in `bra`/`brx.idx` and `call` instructions respectively. Function names can be used in `mov` instruction to get the address of the function into a register, for use in an indirect call.

Beginning in PTX ISA version 3.1, the `mov` instruction may be used to take the address of kernel functions, to be passed to a system call that initiates a kernel launch from the GPU. This feature is part of the support for CUDA Dynamic Parallelism. See the _CUDA Dynamic Parallelism Programming Guide_ for details.


##  6.5. [Type Conversion](#type-conversion)

All operands to all arithmetic, logic, and data movement instruction must be of the same type and size, except for operations where changing the size and/or type is part of the definition of the instruction. Operands of different sizes or types must be converted prior to the operation.

###  6.5.1. [Scalar Conversions](#scalar-conversions)

[Table 15](#scalar-conversions-convert-instruction-precision-and-format-t1) and [Table 16](#scalar-conversions-convert-instruction-precision-and-format-t2) show what precision and format the cvt instruction uses given operands of differing types. For example, if a `cvt.s32.u16` instruction is given a `u16` source operand and `s32` as a destination operand, the `u16` is zero-extended to `s32`.

Conversions to floating-point that are beyond the range of floating-point numbers are represented with the maximum floating-point value (IEEE 754 Inf for `f32` and `f64`, and ~131,000 for `f16`).

Table 15 Convert Instruction Precision and Format Table 1 | **Destination Format**  
---|---  
**s8** | **s16** | **s32** | **s64** | **u8** | **u16** | **u32** | **u64** | **f16** | **f32** | **f64** | **bf16** | **tf32**  
**Source Format** | **s8** | – | sext | sext | sext | – | sext | sext | sext | s2f | s2f | s2f | s2f | –  
**s16** | chop1 | – | sext | sext | chop1 | – | sext | sext | s2f | s2f | s2f | s2f | –  
**s32** | chop1 | chop1 | – | sext | chop1 | chop1 | – | sext | s2f | s2f | s2f | s2f | –  
**s64** | chop1 | chop1 | chop1 | – | chop1 | chop1 | chop1 | – | s2f | s2f | s2f | s2f | –  
**u8** | – | zext | zext | zext | – | zext | zext | zext | u2f | u2f | u2f | u2f | –  
**u16** | chop1 | – | zext | zext | chop1 | – | zext | zext | u2f | u2f | u2f | u2f | –  
**u32** | chop1 | chop1 | – | zext | chop1 | chop1 | – | zext | u2f | u2f | u2f | u2f | –  
**u64** | chop1 | chop1 | chop1 | – | chop1 | chop1 | chop1 | – | u2f | u2f | u2f | u2f | –  
**f16** | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | – | f2f | f2f | f2f | –  
**f32** | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | – | f2f | f2f | f2f  
**f64** | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | f2f | – | f2f | –  
**bf16** | f2s | f2s | f2s | f2s | f2u | f2u | f2u | f2u | f2f | f2f | f2f | f2f | –  
**tf32** | – | – | – | – | – | – | – | – | – | – | – | – | –  
Table 16 Convert Instruction Precision and Format Table 2 | **Destination Format**  
---|---  
**f16** | **f32** | **bf16** | **e4m3** | **e5m2** | **e2m3** | **e3m2** | **e2m1** | **ue8m0** | **s2f6**  
**Source Format** | **f16** | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f | – | –  
**f32** | f2f | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f | f2f  
**bf16** | f2f | f2f | – | f2f | f2f | f2f | f2f | f2f | f2f | f2f  
**e4m3** | f2f | – | f2f | – | – | – | – | – | – | –  
**e5m2** | f2f | – | f2f | – | – | – | – | – | – | –  
**e2m3** | f2f | – | f2f | – | – | – | – | – | – | –  
**e3m2** | f2f | – | f2f | – | – | – | – | – | – | –  
**e2m1** | f2f | – | f2f | – | – | – | – | – | – | –  
**ue8m0** | – | – | f2f | – | – | – | – | – | – | –  
**s2f6** | – | – | f2f | – | – | – | – | – | – | –  
  
**Notes**

sext = sign-extend; zext = zero-extend; chop = keep only low bits that fit;

s2f = signed-to-float; f2s = float-to-signed; u2f = unsigned-to-float;

f2u = float-to-unsigned; f2f = float-to-float.

1 If the destination register is wider than the destination format, the result is extended to the destination register width after chopping. The type of extension (sign or zero) is based on the destination format. For example, cvt.s16.u32 targeting a 32-bit register first chops to 16-bit, then sign-extends to 32-bit.

###  6.5.2. [Rounding Modifiers](#rounding-modifiers)

Conversion instructions may specify a rounding modifier. In PTX, there are four integer rounding modifiers and six floating-point rounding modifiers. [Table 17](#rounding-modifiers-floating-point-rounding-modifiers) and [Table 18](#rounding-modifiers-integer-rounding-modifiers) summarize the rounding modifiers.

Table 17 Floating-Point Rounding Modifiers Modifier | Description  
---|---  
`.rn` | rounds to nearest even  
`.rna` | rounds to nearest, ties away from zero  
`.rz` | rounds towards zero  
`.rm` | rounds towards negative infinity  
`.rp` | rounds towards positive infinity  
`.rs` | rounds either towards zero or away from zero based on the carry out of the integer addition of random bits and the discarded bits of mantissa  
Table 18 Integer Rounding Modifiers Modifier | Description  
---|---  
`.rni` | round to nearest integer, choosing even integer if source is equidistant between two integers.  
`.rzi` | round to nearest integer in the direction of zero  
`.rmi` | round to nearest integer in direction of negative infinity  
`.rpi` | round to nearest integer in direction of positive infinity


##  6.6. [Operand Costs](#operand-costs)  
  
Operands from different state spaces affect the speed of an operation. Registers are fastest, while global memory is slowest. Much of the delay to memory can be hidden in a number of ways. The first is to have multiple threads of execution so that the hardware can issue a memory operation and then switch to other execution. Another way to hide latency is to issue the load instructions as early as possible, as execution is not blocked until the desired result is used in a subsequent (in time) instruction. The register in a store operation is available much more quickly. [Table 19](#operand-costs-cost-estimates-for-sccessing-state-spaces) gives estimates of the costs of using different kinds of memory.

Table 19 Cost Estimates for Accessing State-Spaces Space | Time | Notes  
---|---|---  
Register | 0 |   
Shared | 0 |   
Constant | 0 | Amortized cost is low, first access is high  
Local | > 100 clocks |   
Parameter | 0 |   
Immediate | 0 |   
Global | > 100 clocks |   
Texture | > 100 clocks |   
Surface | > 100 clocks | 
