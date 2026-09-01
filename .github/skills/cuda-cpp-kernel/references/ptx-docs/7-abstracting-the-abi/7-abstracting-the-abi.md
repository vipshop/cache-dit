# 7. Abstracting the ABI


Rather than expose details of a particular calling convention, stack layout, and Application Binary Interface (ABI), PTX provides a slightly higher-level abstraction and supports multiple ABI implementations. In this section, we describe the features of PTX needed to achieve this hiding of the ABI. These include syntax for function definitions, function calls, parameter passing, and memory allocated on the stack (`alloca`).


Refer to _PTX Writers Guide to Interoperability_ for details on generating PTX compliant with Application Binary Interface (ABI) for the CUDA® architecture.


##  7.1. [Function Declarations and Definitions](#function-declarations-and-definitions)  
  
In PTX, functions are declared and defined using the `.func` directive. A function _declaration_ specifies an optional list of return parameters, the function name, and an optional list of input parameters; together these specify the function’s interface, or prototype. A function _definition_ specifies both the interface and the body of the function. A function must be declared or defined prior to being called.

The simplest function has no parameters or return values, and is represented in PTX as follows:
    
    
    .func foo
    {
        ...
        ret;
    }
    
        ...
        call foo;
        ...
    

Here, execution of the `call` instruction transfers control to `foo`, implicitly saving the return address. Execution of the `ret` instruction within `foo` transfers control to the instruction following the call.

Scalar and vector base-type input and return parameters may be represented simply as register variables. At the call, arguments may be register variables or constants, and return values may be placed directly into register variables. The arguments and return variables at the call must have type and size that match the callee’s corresponding formal parameters.

Example
    
    
    .func (.reg .u32 %res) inc_ptr ( .reg .u32 %ptr, .reg .u32 %inc )
    {
        add.u32 %res, %ptr, %inc;
        ret;
    }
    
        ...
        call (%r1), inc_ptr, (%r1,4);
        ...
    

When using the ABI, `.reg` state space parameters must be at least 32-bits in size. Subword scalar objects in the source language should be promoted to 32-bit registers in PTX, or use `.param` state space byte arrays described next.

Objects such as C structures and unions are flattened into registers or byte arrays in PTX and are represented using `.param` space memory. For example, consider the following C structure, passed by value to a function:
    
    
    struct {
        double dbl;
        char   c[4];
    };
    

In PTX, this structure will be flattened into a byte array. Since memory accesses are required to be aligned to a multiple of the access size, the structure in this example will be a 12 byte array with 8 byte alignment so that accesses to the `.f64` field are aligned. The `.param` state space is used to pass the structure by value:

Example
    
    
    .func (.reg .s32 out) bar (.reg .s32 x, .param .align 8 .b8 y[12])
    {
        .reg .f64 f1;
        .reg .b32 c1, c2, c3, c4;
        ...
        ld.param.f64 f1, [y+0];
        ld.param.b8  c1, [y+8];
        ld.param.b8  c2, [y+9];
        ld.param.b8  c3, [y+10];
        ld.param.b8  c4, [y+11];
        ...
        ... // computation using x,f1,c1,c2,c3,c4;
    }
    
    {
         .param .b8 .align 8 py[12];
         ...
         st.param.b64 [py+ 0], %rd;
         st.param.b8  [py+ 8], %rc1;
         st.param.b8  [py+ 9], %rc2;
         st.param.b8  [py+10], %rc1;
         st.param.b8  [py+11], %rc2;
         // scalar args in .reg space, byte array in .param space
         call (%out), bar, (%x, py);
         ...
    

In this example, note that `.param` space variables are used in two ways. First, a `.param` variable `y` is used in function definition bar to represent a formal parameter. Second, a `.param` variable `py` is declared in the body of the calling function and used to set up the structure being passed to bar.

The following is a conceptual way to think about the `.param` state space use in device functions.

For a caller,

  * The `.param` state space is used to set values that will be passed to a called function and/or to receive return values from a called function. Typically, a `.param` byte array is used to collect together fields of a structure being passed by value.


For a callee,

  * The `.param` state space is used to receive parameter values and/or pass return values back to the caller.


The following restrictions apply to parameter passing.

For a caller,

  * Arguments may be `.param` variables, `.reg` variables, or constants.

  * In the case of `.param` space formal parameters that are byte arrays, the argument must also be a `.param` space byte array with matching type, size, and alignment. A `.param` argument must be declared within the local scope of the caller.

  * In the case of `.param` space formal parameters that are base-type scalar or vector variables, the corresponding argument may be either a `.param` or `.reg` space variable with matching type and size, or a constant that can be represented in the type of the formal parameter.

  * In the case of `.reg` space formal parameters, the corresponding argument may be either a `.param` or `.reg` space variable of matching type and size, or a constant that can be represented in the type of the formal parameter.

  * In the case of `.reg` space formal parameters, the register must be at least 32-bits in size.

  * All `st.param` instructions used for passing arguments to function call must immediately precede the corresponding `call` instruction and `ld.param` instruction used for collecting return value must immediately follow the `call` instruction without any control flow alteration. `st.param` and `ld.param` instructions used for argument passing cannot be predicated. This enables compiler optimization and ensures that the `.param` variable does not consume extra space in the caller’s frame beyond that needed by the ABI. The `.param` variable simply allows a mapping to be made at the call site between data that may be in multiple locations (e.g., structure being manipulated by caller is located in registers and memory) to something that can be passed as a parameter or return value to the callee.


For a callee,

  * Input and return parameters may be `.param` variables or `.reg` variables.

  * Parameters in `.param` memory must be aligned to a multiple of 1, 2, 4, 8, or 16 bytes.

  * Parameters in the `.reg` state space must be at least 32-bits in size.

  * The `.reg` state space can be used to receive and return base-type scalar and vector values, including sub-word size objects when compiling in non-ABI mode. Supporting the `.reg` state space provides legacy support.


Note that the choice of `.reg` or `.param` state space for parameter passing has no impact on whether the parameter is ultimately passed in physical registers or on the stack. The mapping of parameters to physical registers and stack locations depends on the ABI definition and the order, size, and alignment of parameters.

###  7.1.1. [Changes from PTX ISA Version 1.x](#changes-from-ptx-isa-version-1-x)

In PTX ISA version 1.x, formal parameters were restricted to .reg state space, and there was no support for array parameters. Objects such as C structures were flattened and passed or returned using multiple registers. PTX ISA version 1.x supports multiple return values for this purpose.

Beginning with PTX ISA version 2.0, formal parameters may be in either `.reg` or `.param` state space, and `.param` space parameters support arrays. For targets `sm_20` or higher, PTX restricts functions to a single return value, and a `.param` byte array should be used to return objects that do not fit into a register. PTX continues to support multiple return registers for `sm_1x` targets.

Note

PTX implements a stack-based ABI only for targets `sm_20` or higher.

PTX ISA versions prior to 3.0 permitted variables in `.reg` and `.local` state spaces to be defined at module scope. When compiling to use the ABI, PTX ISA version 3.0 and later disallows module-scoped `.reg` and `.local` variables and restricts their use to within function scope. When compiling without use of the ABI, module-scoped `.reg` and `.local` variables are supported as before. When compiling legacy PTX code (ISA versions prior to 3.0) containing module-scoped `.reg` or `.local` variables, the compiler silently disables use of the ABI.


##  7.2. [Variadic Functions](#variadic-functions)

Note

**Support for variadic functions which was unimplemented has been removed from the spec.**

PTX version 6.0 supports passing unsized array parameter to a function which can be used to implement variadic functions.

Refer to [Kernel and Function Directives: .func](#kernel-and-function-directives-func) for details


##  7.3. [Alloca](#alloca)

PTX provides `alloca` instruction for allocating storage at runtime on the per-thread local memory stack. The allocated stack memory can be accessed with `ld.local` and `st.local` instructions using the pointer returned by `alloca`.

In order to facilitate deallocation of memory allocated with `alloca`, PTX provides two additional instructions: `stacksave` which allows reading the value of stack pointer in a local variable, and `stackrestore` which can restore the stack pointer with the saved value.

`alloca`, `stacksave`, and `stackrestore` instructions are described in [Stack Manipulation Instructions](#stack-manipulation-instructions).
