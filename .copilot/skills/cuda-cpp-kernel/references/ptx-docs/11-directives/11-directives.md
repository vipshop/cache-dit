# 11. Directives


##  11.1. [PTX Module Directives](#ptx-module-directives)  
  
The following directives declare the PTX ISA version of the code in the module, the target architecture for which the code was generated, and the size of addresses within the PTX module.

  * `.version`

  * `.target`

  * `.address_size`


###  11.1.1. [PTX Module Directives: `.version`](#ptx-module-directives-version)

`.version`

PTX ISA version number.

Syntax
    
    
    .version  major.minor    // major, minor are integers
    

Description

Specifies the PTX language version number.

The _major_ number is incremented when there are incompatible changes to the PTX language, such as changes to the syntax or semantics. The version major number is used by the PTX compiler to ensure correct execution of legacy PTX code.

The _minor_ number is incremented when new features are added to PTX.

Semantics

Indicates that this module must be compiled with tools that support an equal or greater version number.

Each PTX module must begin with a `.version` directive, and no other `.version` directive is allowed anywhere else within the module.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .version 3.1
    .version 3.0
    .version 2.3
    

###  11.1.2. [PTX Module Directives: `.target`](#ptx-module-directives-target)

`.target`

Architecture and Platform target.

Syntax
    
    
    .target stringlist         // comma separated list of target specifiers
    string = { sm_120a, sm_120f, sm_120,          // sm_12x target architectures
               sm_121a, sm_121f, sm_121,          // sm_12x target architectures
               sm_110a, sm_110f, sm_110,          // sm_11x target architectures
               sm_100a, sm_100f, sm_100,          // sm_10x target architectures
               sm_101a, sm_101f, sm_101,          // sm_10x target architectures
               sm_103a, sm_103f, sm_103           // sm_10x target architectures
               sm_90a, sm_90,                     // sm_9x target architectures
               sm_80, sm_86, sm_87, sm_88, sm_89, // sm_8x target architectures
               sm_70, sm_72, sm_75,               // sm_7x target architectures
               sm_60, sm_61, sm_62,               // sm_6x target architectures
               sm_50, sm_52, sm_53,               // sm_5x target architectures
               sm_30, sm_32, sm_35, sm_37,        // sm_3x target architectures
               sm_20,                             // sm_2x target architectures
               sm_10, sm_11, sm_12, sm_13,        // sm_1x target architectures
               texmode_unified, texmode_independent,   // texturing mode
               debug,                                  // platform option
               map_f64_to_f32 };                       // platform option
    

Description

Specifies the set of features in the target architecture for which the current PTX code was generated. In general, generations of SM architectures follow an _onion layer_ model, where each generation adds new features and retains all features of previous generations. The onion layer model allows the PTX code generated for a given target to be run on later generation devices.

Target architectures with suffix “`a`”, such as `sm_90a`, include architecture-specific features that are supported on the specified architecture only, hence such targets do not follow the onion layer model. Therefore, PTX code generated for such targets cannot be run on later generation devices. Architecture-specific features can only be used with targets that support these features.

Target architectures with suffix “`f`”, such as `sm_100f`, include family-specific features that are supported only within the same architecture family. Therefore, PTX code generated for such targets can run only on later generation devices in the same family. Family-specific features can be used with f-targets as well as a-targets of later generation devices in the same family.

[Table 58](#architecture-family-definition) defines the architecture families.

Table 58 Architecture Families Family | Target SM architectures included  
---|---  
sm_10x family | sm_100f, sm_103f, future targets in sm_10x family  
sm_11x family | sm_110f, sm_101f, future targets in sm_11x family  
sm_12x family | sm_120f, sm_121f, future targets in sm_12x family  
  
Semantics

Each PTX module must begin with a `.version` directive, immediately followed by a `.target` directive containing a target architecture and optional platform options. A `.target` directive specifies a single target architecture, but subsequent `.target` directives can be used to change the set of target features allowed during parsing. A program with multiple `.target` directives will compile and run only on devices that support all features of the highest-numbered architecture listed in the program.

PTX features are checked against the specified target architecture, and an error is generated if an unsupported feature is used. The following table summarizes the features in PTX that vary according to target architecture.

Target | Description  
---|---  
`sm_120` | Baseline feature set for `sm_120` architecture.  
`sm_120f` | Adds support for `sm_120f` family specific features.  
`sm_120a` | Adds support for `sm_120a` architecture-specific features.  
`sm_121` | Baseline feature set for `sm_121` architecture.  
`sm_121f` | Adds support for `sm_121f` family specific features.  
`sm_121a` | Adds support for `sm_121a` architecture-specific features.  
Target | Description  
---|---  
`sm_110` | Baseline feature set for `sm_110` architecture.  
`sm_110f` | Adds support for `sm_110f` family specific features.  
`sm_110a` | Adds support for `sm_110a` architecture-specific features.  
Target | Description  
---|---  
`sm_100` | Baseline feature set for `sm_100` architecture.  
`sm_100f` | Adds support for `sm_100f` family specific features.  
`sm_100a` | Adds support for `sm_100a` architecture-specific features.  
`sm_101` | Baseline feature set for `sm_101` architecture. (Renamed to `sm_110`)  
`sm_101f` | Adds support for `sm_101f` family specific features. (Renamed to `sm_110f`)  
`sm_101a` | Adds support for `sm_101a` architecture-specific features. (Renamed to `sm_110a`)  
`sm_103` | Baseline feature set for `sm_103` architecture.  
`sm_103f` | Adds support for `sm_103f` family specific features.  
`sm_103a` | Adds support for `sm_103a` architecture-specific features.  
Target | Description  
---|---  
`sm_90` | Baseline feature set for `sm_90` architecture.  
`sm_90a` | Adds support for `sm_90a` architecture-specific features.  
Target | Description  
---|---  
`sm_80` | Baseline feature set for `sm_80` architecture.  
`sm_86` | Adds support for `.xorsign` modifier on `min` and `max` instructions.  
`sm_87` | Baseline feature set for `sm_87` architecture.  
`sm_88` | Baseline feature set for `sm_88` architecture.  
`sm_89` | Baseline feature set for `sm_89` architecture.  
Target | Description  
---|---  
`sm_70` | Baseline feature set for `sm_70` architecture.  
`sm_72` |  Adds support for integer multiplicand and accumulator matrices in `wmma` instructions. Adds support for `cvt.pack` instruction.  
`sm_75` |  Adds support for sub-byte integer and single-bit multiplicant matrices in `wmma` instructions. Adds support for `ldmatrix` instruction. Adds support for `movmatrix` instruction. Adds support for `tanh` instruction.  
Target | Description  
---|---  
`sm_60` | Baseline feature set for `sm_60` architecture.  
`sm_61` | Adds support for `dp2a` and `dp4a` instructions.  
`sm_62` | Baseline feature set for `sm_61` architecture.  
Target | Description  
---|---  
`sm_50` | Baseline feature set for `sm_50` architecture.  
`sm_52` | Baseline feature set for `sm_50` architecture.  
`sm_53` | Adds support for arithmetic, comparsion and texture instructions for `.f16` and `.f16x2` types.  
Target | Description  
---|---  
`sm_30` | Baseline feature set for `sm_30` architecture.  
`sm_32` |  Adds 64-bit `{atom,red}.{and,or,xor,min,max}` instructions. Adds `shf` instruction. Adds `ld.global.nc` instruction.  
`sm_35` | Adds support for CUDA Dynamic Parallelism.  
`sm_37` | Baseline feature set for `sm_35` architecture.  
Target | Description  
---|---  
`sm_20` | Baseline feature set for `sm_20` architecture.  
Target | Description  
---|---  
`sm_10` |  Baseline feature set for `sm_10` architecture. Requires `map_f64_to_f32` if any `.f64` instructions used.  
`sm_11` |  Adds 64-bit `{atom,red}.{and,or,xor,min,max}` instructions. Requires `map_f64_to_f32` if any `.f64` instructions used.  
`sm_12` |  Adds `{atom,red}.shared`, 64-bit `{atom,red}.global`, `vote` instructions. Requires `map_f64_to_f32` if any `.f64` instructions used.  
`sm_13` |  Adds double-precision support, including expanded rounding modifiers. Disallows use of `map_f64_to_f32`.  
  
The texturing mode is specified for an entire module and cannot be changed within the module.

The `.target` debug option declares that the PTX file contains DWARF debug information, and subsequent compilation of PTX will retain information needed for source-level debugging. If the debug option is declared, an error message is generated if no DWARF information is found in the file. The debug option requires PTX ISA version 3.0 or later.

`map_f64_to_f32` indicates that all double-precision instructions map to single-precision regardless of the target architecture. This enables high-level language compilers to compile programs containing type double to target device that do not support double-precision operations. Note that `.f64` storage remains as 64-bits, with only half being used by instructions converted from `.f64` to `.f32`.

Notes

Targets of the form `compute_xx` are also accepted as synonyms for `sm_xx` targets.

Targets `sm_{101,101f,101a}` are renamed to targets `sm_{110,110f,110a}` from PTX ISA version 9.0.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target strings `sm_10` and `sm_11` introduced in PTX ISA version 1.0.

Target strings `sm_12` and `sm_13` introduced in PTX ISA version 1.2.

Texturing mode introduced in PTX ISA version 1.5.

Target string `sm_20` introduced in PTX ISA version 2.0.

Target string `sm_30` introduced in PTX ISA version 3.0.

Platform option `debug` introduced in PTX ISA version 3.0.

Target string `sm_35` introduced in PTX ISA version 3.1.

Target strings `sm_32` and `sm_50` introduced in PTX ISA version 4.0.

Target strings `sm_37` and `sm_52` introduced in PTX ISA version 4.1.

Target string `sm_53` introduced in PTX ISA version 4.2.

Target string `sm_60`, `sm_61`, `sm_62` introduced in PTX ISA version 5.0.

Target string `sm_70` introduced in PTX ISA version 6.0.

Target string `sm_72` introduced in PTX ISA version 6.1.

Target string `sm_75` introduced in PTX ISA version 6.3.

Target string `sm_80` introduced in PTX ISA version 7.0.

Target string `sm_86` introduced in PTX ISA version 7.1.

Target string `sm_87` introduced in PTX ISA version 7.4.

Target string `sm_88` introduced in PTX ISA version 9.0.

Target string `sm_89` introduced in PTX ISA version 7.8.

Target string `sm_90` introduced in PTX ISA version 7.8.

Target string `sm_90a` introduced in PTX ISA version 8.0.

Target string `sm_100` introduced in PTX ISA version 8.6.

Target string `sm_100f` introduced in PTX ISA version 8.8.

Target string `sm_100a` introduced in PTX ISA version 8.6.

Target string `sm_101` introduced in PTX ISA version 8.6. (Renamed to `sm_110`)

Target string `sm_101f` introduced in PTX ISA version 8.8. (Renamed to `sm_110f`)

Target string `sm_101a` introduced in PTX ISA version 8.6. (Renamed to `sm_110a`)

Target string `sm_103` introduced in PTX ISA version 8.8.

Target string `sm_103f` introduced in PTX ISA version 8.8.

Target string `sm_103a` introduced in PTX ISA version 8.8.

Target string `sm_110` introduced in PTX ISA version 9.0.

Target string `sm_110f` introduced in PTX ISA version 9.0.

Target string `sm_110a` introduced in PTX ISA version 9.0.

Target string `sm_120` introduced in PTX ISA version 8.7.

Target string `sm_120f` introduced in PTX ISA version 8.8.

Target string `sm_120a` introduced in PTX ISA version 8.7.

Target string `sm_121` introduced in PTX ISA version 8.8.

Target string `sm_121f` introduced in PTX ISA version 8.8.

Target string `sm_121a` introduced in PTX ISA version 8.8.

Target ISA Notes

The `.target` directive is supported on all target architectures.

Examples
    
    
    .target sm_10       // baseline target architecture
    .target sm_13       // supports double-precision
    .target sm_20, texmode_independent
    .target sm_90       // baseline target architecture
    .target sm_90a      // PTX using architecture-specific features
    .target sm_100f     // PTX using family-specific features
    

###  11.1.3. [PTX Module Directives: `.address_size`](#ptx-module-directives-address-size)

`.address_size`

Address size used throughout PTX module.

Syntax
    
    
    .address_size  address-size
    address-size = { 32, 64 };
    

Description

Specifies the address size assumed throughout the module by the PTX code and the binary DWARF information in PTX.

Redefinition of this directive within a module is not allowed. In the presence of separate compilation all modules must specify (or default to) the same address size.

The `.address_size` directive is optional, but it must immediately follow the `.target`directive if present within a module.

Semantics

If the `.address_size` directive is omitted, the address size defaults to 32.

PTX ISA Notes

Introduced in PTX ISA version 2.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    // example directives
       .address_size 32       // addresses are 32 bit
       .address_size 64       // addresses are 64 bit
    
    // example of directive placement within a module
       .version 2.3
       .target sm_20
       .address_size 64
    ...
    .entry foo () {
    ...
    }
    


##  11.2. [Specifying Kernel Entry Points and Functions](#specifying-kernel-entry-points-and-functions)

The following directives specify kernel entry points and functions.

  * `.entry`

  * `.func`


###  11.2.1. [Kernel and Function Directives: `.entry`](#kernel-and-function-directives-entry)

`.entry`

Kernel entry point and body, with optional parameters.

Syntax
    
    
    .entry kernel-name ( param-list )  kernel-body
    .entry kernel-name  kernel-body
    

Description

Defines a kernel entry point name, parameters, and body for the kernel function.

Parameters are passed via `.param` space memory and are listed within an optional parenthesized parameter list. Parameters may be referenced by name within the kernel body and loaded into registers using `ld.param{::entry}` instructions.

In addition to normal parameters, opaque `.texref`, `.samplerref`, and `.surfref` variables may be passed as parameters. These parameters can only be referenced by name within texture and surface load, store, and query instructions and cannot be accessed via `ld.param` instructions.

The shape and size of the CTA executing the kernel are available in special registers.

Semantics

Specify the entry point for a kernel program.

At kernel launch, the kernel dimensions and properties are established and made available via special registers, e.g., `%ntid`, `%nctaid`, etc.

PTX ISA Notes

For PTX ISA version 1.4 and later, parameter variables are declared in the kernel parameter list. For PTX ISA versions 1.0 through 1.3, parameter variables are declared in the kernel body.

The maximum memory size supported by PTX for normal (non-opaque type) parameters is 32764 bytes. Depending upon the PTX ISA version, the parameter size limit varies. The following table shows the allowed parameter size for a PTX ISA version:

PTX ISA Version | Maximum parameter size (In bytes)  
---|---  
PTX ISA version 8.1 and above | 32764  
PTX ISA version 1.5 and above | 4352  
PTX ISA version 1.4 and above | 256  
  
The CUDA and OpenCL drivers support the following limits for parameter memory:

Driver | Parameter memory size  
---|---  
CUDA | 256 bytes for `sm_1x`, 4096 bytes for `sm_2x and higher`, 32764 bytes fo `sm_70` and higher  
OpenCL | 32764 bytes for `sm_70` and higher, 4352 bytes on `sm_6x` and lower  
  
Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry cta_fft
    .entry filter ( .param .b32 x, .param .b32 y, .param .b32 z )
    {
        .reg .b32 %r<99>;
        ld.param.b32  %r1, [x];
        ld.param.b32  %r2, [y];
        ld.param.b32  %r3, [z];
        ...
    }
    
    .entry prefix_sum ( .param .align 4 .s32 pitch[8000] )
    {
        .reg .s32 %t;
        ld.param::entry.s32  %t, [pitch];
        ...
    }
    

###  11.2.2. [Kernel and Function Directives: `.func`](#kernel-and-function-directives-func)

`.func`

Function definition.

Syntax
    
    
    .func {.attribute(attr-list)} fname {.noreturn} {.abi_preserve N} {.abi_preserve_control N} function-body
    .func {.attribute(attr-list)} fname (param-list) {.noreturn} {.abi_preserve N} {.abi_preserve_control N} function-body
    .func {.attribute(attr-list)} (ret-param) fname (param-list) {.abi_preserve N} {.abi_preserve_control N} function-body
    

Description

Defines a function, including input and return parameters and optional function body.

An optional `.noreturn` directive indicates that the function does not return to the caller function. `.noreturn` directive cannot be specified on functions which have return parameters. See the description of `.noreturn` directive in [Performance-Tuning Directives: .noreturn](#performance-tuning-directives-noreturn).

An optional `.attribute` directive specifies additional information associated with the function. See the description of [Variable and Function Attribute Directive: .attribute](#variable-and-function-attribute-directive-attribute) for allowed attributes.

Optional `.abi_preserve` and `.abi_preserve_control` directives are used to specify the number of general purpose registers and control registers. See description of [Performance-Tuning Directives: .abi_preserve](#performance-tuning-directives-abi-preserve) and [Performance-Tuning Directives: .abi_preserve_control](#performance-tuning-directives-abi-preserve-control) for more details.

Directives, if any specified, for a function, e.g. `.noreturn`, must be specified consistently between function declaration and definition.

A `.func` definition with no body provides a function prototype.

The parameter lists define locally-scoped variables in the function body. Parameters must be base types in either the register or parameter state space. Parameters in register state space may be referenced directly within instructions in the function body. Parameters in `.param` space are accessed using `ld.param{::func}` and `st.param{::func}` instructions in the body. Parameter passing is call-by-value.

The last parameter in the parameter list may be a `.param` array of type `.b8` with no size specified. It is used to pass an arbitrary number of parameters to the function packed into a single array object.

When calling a function with such an unsized last argument, the last argument may be omitted from the `call` instruction if no parameter is passed through it. Accesses to this array parameter must be within the bounds of the array. The result of an access is undefined if no array was passed, or if the access was outside the bounds of the actual array being passed.

Semantics

The PTX syntax hides all details of the underlying calling convention and ABI.

The implementation of parameter passing is left to the optimizing translator, which may use a combination of registers and stack locations to pass parameters.

Release Notes

For PTX ISA version 1.x code, parameters must be in the register state space, there is no stack, and recursion is illegal.

PTX ISA versions 2.0 and later with target `sm_20` or higher allow parameters in the `.param` state space, implements an ABI with stack, and supports recursion.

PTX ISA versions 2.0 and later with target `sm_20` or higher support at most one return value.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Support for unsized array parameter introduced in PTX ISA version 6.0.

Support for `.noreturn` directive introduced in PTX ISA version 6.4.

Support for `.attribute` directive introduced in PTX ISA version 8.0.

Support for `.abi_preserve` and `.abi_preserve_control` directives introduced in PTX ISA version 9.0.

Target ISA Notes

Functions without unsized array parameter supported on all target architectures.

Unsized array parameter requires `sm_30` or higher.

`.noreturn` directive requires `sm_30` or higher.

`.attribute` directive requires `sm_90` or higher.

`.abi_preserve` and `.abi_preserve_control` directives require `sm_80` or higher.

Examples
    
    
    .func (.reg .b32 rval) foo (.reg .b32 N, .reg .f64 dbl)
    {
    .reg .b32 localVar;
    
    ... use N, dbl;
    other code;
    
    mov.b32 rval,result;
    ret;
    }
    
    ...
    call (fooval), foo, (val0, val1);  // return value in fooval
    ...
    
    .func foo (.reg .b32 N, .reg .f64 dbl) .noreturn
    {
    .reg .b32 localVar;
    ... use N, dbl;
    other code;
    mov.b32 rval, result;
    ret;
    }
    ...
    call foo, (val0, val1);
    ...
    
    .func (.param .u32 rval) bar(.param .u32 N, .param .align 4 .b8 numbers[])
    {
        .reg .b32 input0, input1;
        ld.param.b32   input0, [numbers + 0];
        ld.param.b32   input1, [numbers + 4];
        ...
        other code;
        ret;
    }
    ...
    
    .param .u32 N;
    .param .align 4 .b8 numbers[8];
    st.param.u32    [N], 2;
    st.param.b32    [numbers + 0], 5;
    st.param.b32    [numbers + 4], 10;
    call (rval), bar, (N, numbers);
    ...
    

###  11.2.3. [Kernel and Function Directives: `.alias`](#kernel-and-function-directives-alias)

`.alias`

Define an alias to existing function symbol.

Syntax
    
    
    .alias fAlias, fAliasee;
    

Description

`.alias` is a module scope directive that defines identifier `fAlias` to be an alias to function specified by `fAliasee`.

Both `fAlias` and `fAliasee` are non-entry function symbols.

Identifier `fAlias` is a function declaration without body.

Identifier `fAliasee` is a function symbol which must be defined in the same module as `.alias` declaration. Function `fAliasee` cannot have `.weak` linkage.

Prototype of `fAlias` and `fAliasee` must match.

Program can use either `fAlias` or `fAlisee` identifiers to reference function defined with `fAliasee`.

PTX ISA Notes

`.alias` directive introduced in PTX ISA 6.3.

Target ISA Notes

`.alias` directive requires `sm_30` or higher.

Examples
    
    
    .visible .func foo(.param .u32 p) {
       ...
    }
    .visible .func bar(.param .u32 p);
    .alias bar, foo;
    .entry test()
    {
          .param .u32 p;
          ...
          call foo, (p);       // call foo directly
           ...
           .param .u32 p;
           call bar, (p);        // call foo through alias
    }
    .entry filter ( .param .b32 x, .param .b32 y, .param .b32 z )
    {
        .reg .b32 %r1, %r2, %r3;
        ld.param.b32  %r1, [x];
        ld.param.b32  %r2, [y];
        ld.param.b32  %r3, [z];
        ...
    }
    


##  11.3. [Control Flow Directives](#control-flow-directives)

PTX provides directives for specifying potential targets for `brx.idx` and `call` instructions. See the descriptions of `brx.idx` and `call` for more information.

  * `.branchtargets`

  * `.calltargets`

  * `.callprototype`


###  11.3.1. [Control Flow Directives: `.branchtargets`](#control-flow-directives-branchtargets)

`.branchtargets`

Declare a list of potential branch targets.

Syntax
    
    
    Label:   .branchtargets  list-of-labels ;
    

Description

Declares a list of potential branch targets for a subsequent `brx.idx`, and associates the list with the label at the start of the line.

All control flow labels in the list must occur within the same function as the declaration.

The list of labels may use the compact, shorthand syntax for enumerating a range of labels having a common prefix, similar to the syntax described in [Parameterized Variable Names](#parameterized-variable-names).

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples
    
    
      .function foo () {
          .reg .u32 %r0;
          ...
          L1:
          ...
          L2:
          ...
          L3:
          ...
          ts: .branchtargets L1, L2, L3;
          @p brx.idx %r0, ts;
          ...
    
    .function bar() {
          .reg .u32 %r0;
          ...
          N0:
          ...
          N1:
          ...
          N2:
          ...
          N3:
          ...
          N4:
          ...
          ts: .branchtargets N<5>;
          @p brx.idx %r0, ts;
          ...
    

###  11.3.2. [Control Flow Directives: `.calltargets`](#control-flow-directives-calltargets)

`.calltargets`

Declare a list of potential call targets.

Syntax
    
    
    Label:   .calltargets  list-of-functions ;
    

Description

Declares a list of potential call targets for a subsequent indirect call, and associates the list with the label at the start of the line.

All functions named in the list must be declared prior to the `.calltargets` directive, and all functions must have the same type signature.

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Target ISA Notes

Requires `sm_20` or higher.

Examples
    
    
    calltgt:  .calltargets  fastsin, fastcos;
    ...
    @p   call  (%f1), %r0, (%x), calltgt;
    ...
    

###  11.3.3. [Control Flow Directives: `.callprototype`](#control-flow-directives-callprototype)

`.callprototype`

Declare a prototype for use in an indirect call.

Syntax
    
    
     // no input or return parameters
    label: .callprototype _ .noreturn {.abi_preserve N} {.abi_preserve_control N};
    // input params, no return params
    label: .callprototype _ (param-list) .noreturn {.abi_preserve N} {.abi_preserve_control N};
    // no input params, // return params
    label: .callprototype (ret-param) _ {.abi_preserve N} {.abi_preserve_control N};
    // input, return parameters
    label: .callprototype (ret-param) _ (param-list) {.abi_preserve N} {.abi_preserve_control N};
    

Description

Defines a prototype with no specific function name, and associates the prototype with a label. The prototype may then be used in indirect call instructions where there is incomplete knowledge of the possible call targets.

Parameters may have either base types in the register or parameter state spaces, or array types in parameter state space. The sink symbol `'_'` may be used to avoid dummy parameter names.

An optional `.noreturn` directive indicates that the function does not return to the caller function. `.noreturn` directive cannot be specified on functions which have return parameters. See the description of .noreturn directive in [Performance-Tuning Directives: .noreturn](#performance-tuning-directives-noreturn).

Optional `.abi_preserve` and `.abi_preserve_control` directives are used to specify the number of general purpose registers and control registers. See description of [Performance-Tuning Directives: .abi_preserve](#performance-tuning-directives-abi-preserve) and [Performance-Tuning Directives: .abi_preserve_control](#performance-tuning-directives-abi-preserve-control) for more details.

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Support for `.noreturn` directive introduced in PTX ISA version 6.4.

Support for `.abi_preserve` and `.abi_preserve_control` directives introduced in PTX ISA version 9.0.

Target ISA Notes

Requires `sm_20` or higher.

`.noreturn` directive requires `sm_30` or higher.

`.abi_preserve` and `.abi_preserve_control` directives require `sm_80` or higher.

Examples
    
    
    Fproto1: .callprototype  _ ;
    Fproto2: .callprototype  _ (.param .f32 _);
    Fproto3: .callprototype  (.param .u32 _) _ ;
    Fproto4: .callprototype  (.param .u32 _) _ (.param .f32 _);
    ...
    @p   call  (%val), %r0, (%f1), Fproto4;
    ...
    
    // example of array parameter
    Fproto5: .callprototype _ (.param .b8 _[12]);
    
    Fproto6: .callprototype  _ (.param .f32 _) .noreturn;
    ...
    @p   call  %r0, (%f1), Fproto6;
    ...
    
    // example of .abi_preserve
    Fproto7: .callprototype _ (.param .b32 _) .abi_preserve 10;
    ...
    @p   call %r0, (%r1), Fproto7;
    ...
    


##  11.4. [Performance-Tuning Directives](#performance-tuning-directives)

To provide a mechanism for low-level performance tuning, PTX supports the following directives, which pass information to the optimizing backend compiler.

  * `.maxnreg`

  * `.maxntid`

  * `.reqntid`

  * `.minnctapersm`

  * `.maxnctapersm` (deprecated)

  * `.pragma`

  * `.abi_preserve`

  * `.abi_preserve_control`


The `.maxnreg` directive specifies the maximum number of registers to be allocated to a single thread; the `.maxntid` directive specifies the maximum number of threads in a thread block (CTA); the `.reqntid` directive specifies the required number of threads in a thread block (CTA); and the `.minnctapersm` directive specifies a minimum number of thread blocks to be scheduled on a single multiprocessor (SM). These can be used, for example, to throttle the resource requirements (e.g., registers) to increase total thread count and provide a greater opportunity to hide memory latency. The `.minnctapersm` directive can be used together with either the `.maxntid` or `.reqntid` directive to trade-off registers-per-thread against multiprocessor utilization without needed to directly specify a maximum number of registers. This may achieve better performance when compiling PTX for multiple devices having different numbers of registers per SM.

Device function directives `.abi_preserve` and `.abi_preserve_control` specify number of data and control registers from callee save registers that a function must preserve for its caller. This can be considered to be the number of general purpose and control registers live in the caller when function is called. Control registers refer to the number of divergent program points that happen in the calltree leading to current function call.

Currently, the `.maxnreg`, `.maxntid`, `.reqntid`, and `.minnctapersm` directives may be applied per-entry and must appear between an `.entry` directive and its body. The directives take precedence over any module-level constraints passed to the optimizing backend. A warning message is generated if the directives’ constraints are inconsistent or cannot be met for the specified target device.

A general `.pragma` directive is supported for passing information to the PTX backend. The directive passes a list of strings to the backend, and the strings have no semantics within the PTX virtual machine model. The interpretation of `.pragma` values is determined by the backend implementation and is beyond the scope of the PTX ISA. Note that `.pragma` directives may appear at module (file) scope, at entry-scope, or as statements within a kernel or device function body.

###  11.4.1. [Performance-Tuning Directives: `.maxnreg`](#performance-tuning-directives-maxnreg)

`.maxnreg`

Maximum number of registers that can be allocated per thread.

Syntax
    
    
    .maxnreg n
    

Description

Declare the maximum number of registers per thread in a CTA.

Semantics

The compiler guarantees that this limit will not be exceeded. The actual number of registers used may be less; for example, the backend may be able to compile to fewer registers, or the maximum number of registers may be further constrained by `.maxntid` and `.maxctapersm`.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo .maxnreg 16 { ... }  // max regs per thread = 16
    

###  11.4.2. [Performance-Tuning Directives: `.maxntid`](#performance-tuning-directives-maxntid)

`.maxntid`

Maximum number of threads in the thread block (CTA).

Syntax
    
    
    .maxntid nx
    .maxntid nx, ny
    .maxntid nx, ny, nz
    

Description

Declare the maximum number of threads in the thread block (CTA). This maximum is specified by giving the maximum extent of each dimension of the 1D, 2D, or 3D CTA. The maximum number of threads is the product of the maximum extent in each dimension.

Semantics

The maximum number of threads in the thread block, computed as the product of the maximum extent specified for each dimension, is guaranteed not to be exceeded in any invocation of the kernel in which this directive appears. Exceeding the maximum number of threads results in a runtime error or kernel launch failure.

Note that this directive guarantees that the _total_ number of threads does not exceed the maximum, but does not guarantee that the limit in any particular dimension is not exceeded.

PTX ISA Notes

Introduced in PTX ISA version 1.3.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo .maxntid 256       { ... }  // max threads = 256
    .entry bar .maxntid 16,16,4   { ... }  // max threads = 1024
    

###  11.4.3. [Performance-Tuning Directives: `.reqntid`](#performance-tuning-directives-reqntid)

`.reqntid`

Number of threads in the thread block (CTA).

Syntax
    
    
    .reqntid nx
    .reqntid nx, ny
    .reqntid nx, ny, nz
    

Description

Declare the number of threads in the thread block (CTA) by specifying the extent of each dimension of the 1D, 2D, or 3D CTA. The total number of threads is the product of the number of threads in each dimension.

Semantics

The size of each CTA dimension specified in any invocation of the kernel is required to be equal to that specified in this directive. Specifying a different CTA dimension at launch will result in a runtime error or kernel launch failure.

Notes

The `.reqntid` directive cannot be used in conjunction with the `.maxntid` directive.

PTX ISA Notes

Introduced in PTX ISA version 2.1.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo .reqntid 256       { ... }  // num threads = 256
    .entry bar .reqntid 16,16,4   { ... }  // num threads = 1024
    

###  11.4.4. [Performance-Tuning Directives: `.minnctapersm`](#performance-tuning-directives-minnctapersm)

`.minnctapersm`

Minimum number of CTAs per SM.

Syntax
    
    
    .minnctapersm ncta
    

Description

Declare the minimum number of CTAs from the kernel’s grid to be mapped to a single multiprocessor (SM).

Notes

Optimizations based on `.minnctapersm` need either `.maxntid` or `.reqntid` to be specified as well.

If the total number of threads on a single SM resulting from `.minnctapersm` and `.maxntid` / `.reqntid` exceed maximum number of threads supported by an SM then directive `.minnctapersm` will be ignored.

In PTX ISA version 2.1 or higher, a warning is generated if `.minnctapersm` is specified without specifying either `.maxntid` or `.reqntid`.

PTX ISA Notes

Introduced in PTX ISA version 2.0 as a replacement for `.maxnctapersm`.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo .maxntid 256 .minnctapersm 4 { ... }
    

###  11.4.5. [Performance-Tuning Directives: `.maxnctapersm` (deprecated)](#performance-tuning-directives-maxnctapersm)

`.maxnctapersm`

Maximum number of CTAs per SM.

Syntax
    
    
    .maxnctapersm ncta
    

Description

Declare the maximum number of CTAs from the kernel’s grid that may be mapped to a single multiprocessor (SM).

Notes

Optimizations based on .maxnctapersm generally need `.maxntid` to be specified as well. The optimizing backend compiler uses `.maxntid` and `.maxnctapersm` to compute an upper-bound on per-thread register usage so that the specified number of CTAs can be mapped to a single multiprocessor. However, if the number of registers used by the backend is sufficiently lower than this bound, additional CTAs may be mapped to a single multiprocessor. For this reason, `.maxnctapersm` has been renamed to .minnctapersm in PTX ISA version 2.0.

PTX ISA Notes

Introduced in PTX ISA version 1.3. Deprecated in PTX ISA version 2.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo .maxntid 256 .maxnctapersm 4 { ... }
    

###  11.4.6. [Performance-Tuning Directives: `.noreturn`](#performance-tuning-directives-noreturn)

`.noreturn`

Indicate that the function does not return to its caller function.

Syntax
    
    
    .noreturn
    

Description

Indicate that the function does not return to its caller function.

Semantics

An optional `.noreturn` directive indicates that the function does not return to caller function. `.noreturn` directive can only be specified on device functions and must appear between a `.func` directive and its body.

The directive cannot be specified on functions which have return parameters.

If a function with `.noreturn` directive returns to the caller function at runtime, then the behavior is undefined.

PTX ISA Notes

Introduced in PTX ISA version 6.4.

Target ISA Notes

Requires `sm_30` or higher.

Examples
    
    
    .func foo .noreturn { ... }
    

###  11.4.7. [Performance-Tuning Directives: `.pragma`](#performance-tuning-directives-pragma)

`.pragma`

Pass directives to PTX backend compiler.

Syntax
    
    
    .pragma list-of-strings ;
    

Description

Pass module-scoped, entry-scoped, or statement-level directives to the PTX backend compiler.

The `.pragma` directive may occur at module-scope, at entry-scope, or at statement-level.

Semantics

The interpretation of `.pragma` directive strings is implementation-specific and has no impact on PTX semantics. See [Descriptions of .pragma Strings](#descriptions-pragma-strings) for descriptions of the pragma strings defined in `ptxas`.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .pragma "nounroll";    // disable unrolling in backend
    
    // disable unrolling for current kernel
    .entry foo .pragma "nounroll"; { ... }
    

###  11.4.8. [Performance-Tuning Directives: `.abi_preserve`](#performance-tuning-directives-abi-preserve)

`.abi_preserve`

Specify number of general purpose registers that should be preserved by the callers of this function.

Syntax
    
    
    .abi_preserve N
    

Description

It is an architecture agnostic value specifying actual number of general purpose registers. Internally ABI defines some general purpose registers as preserved (callee save) registers. Integer N specifies the actual number of general purpose registers that should be preserved by the function.

`.abi_preserve` directive can only be specified on device functions and must appear between a `.func` directive and its body.

Semantics

When this directive is specified compiler backend modifies low level ABI components to ensure that number of live data variables in the callers of this function that are stored in the callee save registers are less than specified value.

PTX ISA Notes

Introduced in PTX ISA version 9.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples
    
    
    .func bar() .abi_preserve 8
    
    // Indirect call via call prototype
    .func (.param .b32 out[30]) foo (.param .b32 in[30]) .abi_preserve 10 { ... }
    ...
    mov.b64 lpfoo, foo;
    prot: .callprototype (.param .b32 out[30]) _ (.param .b32 in[30]) .abi_preserve 10;
    call (out), lpfoo, (in), prot;
    

###  11.4.9. [Performance-Tuning Directives: `.abi_preserve_control`](#performance-tuning-directives-abi-preserve-control)

`.abi_preserve_control`

Specify number of control registers that should be preserved by the callers of this function.

Syntax
    
    
    .abi_preserve_control N
    

Description

It is an architecture agnostic value specifying the number of divergent program points that happen in the calltree leading to current function call. Internally ABI defines some control registers as preserved (callee save) registers. Integer N specifies the actual number of control registers that should be preserved by the function.

`.abi_preserve_control` directive can only be specified on device functions and must appear between a `.func` directive and its body.

Semantics

When this directive is specified compiler backend modifies low level ABI components to ensure that number of live control variables in the callers of this function that are stored in the callee save control registers are less than specified value.

PTX ISA Notes

Introduced in PTX ISA version 9.0.

Target ISA Notes

Requires `sm_80` or higher.

Examples
    
    
    .func foo() .abi_preserve_control 14
    
    // Indirect call via call prototype
    .func (.param .b32 out[30]) bar (.param .b32 in[30]) .abi_preserve_control 10 { ... }
    ...
    mov.b64 lpbar, bar;
    prot: .callprototype (.param .b32 out[30]) _ (.param .b32 in[30]) .abi_preserve_control 10;
    call (out), lpbar, (in), prot;
    


##  11.5. [Debugging Directives](#debugging-directives)

DWARF-format debug information is passed through PTX modules using the following directives:

  * `@@DWARF`

  * `.section`

  * `.file`

  * `.loc`


The `.section` directive was introduced in PTX ISA version 2.0 and replaces the `@@DWARF` syntax. The `@@DWARF` syntax was deprecated in PTX ISA version 2.0 but is supported for legacy PTX ISA version 1.x code.

Beginning with PTX ISA version 3.0, PTX files containing DWARF debug information should include the `.target debug` platform option. This forward declaration directs PTX compilation to retain mappings for source-level debugging.

###  11.5.1. [Debugging Directives: `@@dwarf`](#debugging-directives-atatdwarf)

`@@dwarf`

DWARF-format information.

Syntax
    
    
    @@DWARF dwarf-string
    
    dwarf-string may have one of the
    .byte   byte-list   // comma-separated hexadecimal byte values
    .4byte  int32-list  // comma-separated hexadecimal integers in range [0..2^32-1]
    .quad   int64-list  // comma-separated hexadecimal integers in range [0..2^64-1]
    .4byte  label
    .quad   label
    

PTX ISA Notes

Introduced in PTX ISA version 1.2. Deprecated as of PTX ISA version 2.0, replaced by `.section` directive.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    @@DWARF .section .debug_pubnames, "", @progbits
    @@DWARF .byte   0x2b, 0x00, 0x00, 0x00, 0x02, 0x00
    @@DWARF .4byte  .debug_info
    @@DWARF .4byte  0x000006b5, 0x00000364, 0x61395a5f, 0x5f736f63
    @@DWARF .4byte  0x6e69616d, 0x63613031, 0x6150736f, 0x736d6172
    @@DWARF .byte   0x00, 0x00, 0x00, 0x00, 0x00
    

###  11.5.2. [Debugging Directives: `.section`](#debugging-directives-section)

`.section`

PTX section definition.

Syntax
    
    
    .section section_name { dwarf-lines }
    
    dwarf-lines have the following formats:
      .b8    byte-list       // comma-separated list of integers
                             // in range [-128..255]
      .b16   int16-list      // comma-separated list of integers
                             // in range [-2^15..2^16-1]
      .b32   int32-list      // comma-separated list of integers
                             // in range [-2^31..2^32-1]
      label:                 // Define label inside the debug section
      .b64   int64-list      // comma-separated list of integers
                             // in range [-2^63..2^64-1]
      .b32   label
      .b64   label
      .b32   label+imm       // a sum of label address plus a constant integer byte
                             // offset(signed, 32bit)
      .b64   label+imm       // a sum of label address plus a constant integer byte
                             // offset(signed, 64bit)
      .b32   label1-label2   // a difference in label addresses between labels in
                             // the same dwarf section (32bit)
      .b64   label3-label4   // a difference in label addresses between labels in
                             // the same dwarf section (64bit)
    

PTX ISA Notes

Introduced in PTX ISA version 2.0, replaces `@@DWARF` syntax.

label+imm expression introduced in PTX ISA version 3.2.

Support for `.b16` integers in dwarf-lines introduced in PTX ISA version 6.0.

Support for defining `label` inside the DWARF section is introduced in PTX ISA version 7.2.

`label1-label2` expression introduced in PTX ISA version 7.5.

Negative numbers in dwarf lines introduced in PTX ISA version 7.5.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .section .debug_pubnames
    {
        .b32    LpubNames_end0-LpubNames_begin0
      LpubNames_begin0:
        .b8     0x2b, 0x00, 0x00, 0x00, 0x02, 0x00
        .b32    .debug_info
      info_label1:
        .b32    0x000006b5, 0x00000364, 0x61395a5f, 0x5f736f63
        .b32    0x6e69616d, 0x63613031, 0x6150736f, 0x736d6172
        .b8     0x00, 0x00, 0x00, 0x00, 0x00
      LpubNames_end0:
    }
    
    .section .debug_info
    {
        .b32 11430
        .b8 2, 0
        .b32 .debug_abbrev
        .b8 8, 1, 108, 103, 101, 110, 102, 101, 58, 32, 69, 68, 71, 32, 52, 46, 49
        .b8 0
        .b32 3, 37, 176, -99
        .b32 info_label1
        .b32 .debug_loc+0x4
        .b8 -11, 11, 112, 97
        .b32 info_label1+12
        .b64 -1
        .b16 -5, -65535
    }
    

###  11.5.3. [Debugging Directives: `.file`](#debugging-directives-file)

`.file`

Source file name.

Syntax
    
    
    .file file_index "filename" {, timestamp, file_size}
    

Description

Associates a source filename with an integer index. `.loc` directives reference source files by index.

`.file` directive allows optionally specifying an unsigned number representing time of last modification and an unsigned integer representing size in bytes of source file. `timestamp` and `file_size` value can be 0 to indicate this information is not available.

`timestamp` value is in format of C and C++ data type `time_t`.

`file_size` is an unsigned 64-bit integer.

The `.file` directive is allowed only in the outermost scope, i.e., at the same level as kernel and device function declarations.

Semantics

If timestamp and file size are not specified, they default to 0.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Timestamp and file size introduced in PTX ISA version 3.2.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .file 1 "example.cu"
    .file 2 "kernel.cu"
    .file 1 "kernel.cu", 1339013327, 64118
    

###  11.5.4. [Debugging Directives: `.loc`](#debugging-directives-loc)

`.loc`

Source file location.

Syntax
    
    
    .loc file_index line_number column_position
    .loc file_index line_number column_position,function_name label {+ immediate }, inlined_at file_index2 line_number2 column_position2
    

Description

Declares the source file location (source file, line number, and column position) to be associated with lexically subsequent PTX instructions. `.loc` refers to `file_index` which is defined by a `.file` directive.

To indicate PTX instructions that are generated from a function that got inlined, additional attribute `.inlined_at` can be specified as part of the `.loc` directive. `.inlined_at` attribute specifies source location at which the specified function is inlined. `file_index2`, `line_number2`, and `column_position2` specify the location at which function is inlined. Source location specified as part of `.inlined_at` directive must lexically precede as source location in `.loc` directive.

The `function_name` attribute specifies an offset in the DWARF section named `.debug_str`. Offset is specified as `label` expression or `label + immediate` expression where `label` is defined in `.debug_str` section. DWARF section `.debug_str` contains ASCII null-terminated strings that specify the name of the function that is inlined.

Note that a PTX instruction may have a single associated source location, determined by the nearest lexically preceding .loc directive, or no associated source location if there is no preceding .loc directive. Labels in PTX inherit the location of the closest lexically following instruction. A label with no following PTX instruction has no associated source location.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

`function_name` and `inlined_at` attributes are introduced in PTX ISA version 7.2.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
        .loc 2 4237 0
    L1:                        // line 4237, col 0 of file #2,
                               // inherited from mov
        mov.u32  %r1,%r2;      // line 4237, col 0 of file #2
        add.u32  %r2,%r1,%r3;  // line 4237, col 0 of file #2
    ...
    L2:                        // line 4239, col 5 of file #2,
                               // inherited from sub
        .loc 2 4239 5
        sub.u32  %r2,%r1,%r3;  // line 4239, col 5 of file #2
        .loc 1 21 3
        .loc 1 9 3, function_name info_string0, inlined_at 1 21 3
        ld.global.u32   %r1, [gg]; // Function at line 9
        setp.lt.s32 %p1, %r1, 8;   // inlined at line 21
        .loc 1 27 3
        .loc 1 10 5, function_name info_string1, inlined_at 1 27 3
        .loc 1 15 3, function_name .debug_str+16, inlined_at 1 10 5
        setp.ne.s32 %p2, %r1, 18;
        @%p2 bra    BB2_3;
    
        .section .debug_str {
        info_string0:
         .b8 95  // _
         .b8 90  // z
         .b8 51  // 3
         .b8 102 // f
         .b8 111 // o
         .b8 111 // o
         .b8 118 // v
         .b8 0
    
        info_string1:
         .b8 95  // _
         .b8 90  // z
         .b8 51  // 3
         .b8 98  // b
         .b8 97  // a
         .b8 114 // r
         .b8 118 // v
         .b8 0
         .b8 95  // _
         .b8 90  // z
         .b8 51  // 3
         .b8 99  // c
         .b8 97  // a
         .b8 114 // r
         .b8 118 // v
         .b8 0
        }
    


##  11.6. [Linking Directives](#linking-directives)

  * `.extern`

  * `.visible`

  * `.weak`


###  11.6.1. [Linking Directives: `.extern`](#linking-directives-extern)

`.extern`

External symbol declaration.

Syntax
    
    
    .extern identifier
    

Description

Declares identifier to be defined external to the current module. The module defining such identifier must define it as `.weak` or `.visible` only once in a single object file. Extern declaration of symbol may appear multiple times and references to that get resolved against the single definition of that symbol.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .extern .global .b32 foo;  // foo is defined in another module
    

###  11.6.2. [Linking Directives: `.visible`](#linking-directives-visible)

`.visible`

Visible (externally) symbol declaration.

Syntax
    
    
    .visible identifier
    

Description

Declares identifier to be globally visible. Unlike C, where identifiers are globally visible unless declared static, PTX identifiers are visible only within the current module unless declared `.visible` outside the current.

PTX ISA Notes

Introduced in PTX ISA version 1.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .visible .global .b32 foo;  // foo will be externally visible
    

###  11.6.3. [Linking Directives: `.weak`](#linking-directives-weak)

`.weak`

Visible (externally) symbol declaration.

Syntax
    
    
    .weak identifier
    

Description

Declares identifier to be globally visible but _weak_. Weak symbols are similar to globally visible symbols, except during linking, weak symbols are only chosen after globally visible symbols during symbol resolution. Unlike globally visible symbols, multiple object files may declare the same weak symbol, and references to a symbol get resolved against a weak symbol only if no global symbols have the same name.

PTX ISA Notes

Introduced in PTX ISA version 3.1.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .weak .func (.reg .b32 val) foo;  // foo will be externally visible
    

###  11.6.4. [Linking Directives: `.common`](#linking-directives-common)

`.common`

Visible (externally) symbol declaration.

Syntax
    
    
    .common identifier
    

Description

Declares identifier to be globally visible but “common”.

Common symbols are similar to globally visible symbols. However multiple object files may declare the same common symbol and they may have different types and sizes and references to a symbol get resolved against a common symbol with the largest size.

Only one object file can initialize a common symbol and that must have the largest size among all other definitions of that common symbol from different object files.

`.common` linking directive can be used only on variables with `.global` storage. It cannot be used on function symbols or on symbols with opaque type.

PTX ISA Notes

Introduced in PTX ISA version 5.0.

Target ISA Notes

`.common` directive requires `sm_20` or higher.

Examples
    
    
    .common .global .u32 gbl;
    


##  11.7. [Cluster Dimension Directives](#cluster-dimension-directives)

The following directives specify information about clusters:

  * `.reqnctapercluster`

  * `.explicitcluster`

  * `.maxclusterrank`


The `.reqnctapercluster` directive specifies the number of CTAs in the cluster. The `.explicitcluster` directive specifies that the kernel should be launched with explicit cluster details. The `.maxclusterrank` directive specifies the maximum number of CTAs in the cluster.

The cluster dimension directives can be applied only on kernel functions.

###  11.7.1. [Cluster Dimension Directives: `.reqnctapercluster`](#cluster-dimension-directives-reqnctapercluster)

`.reqnctapercluster`

Declare the number of CTAs in the cluster.

Syntax
    
    
    .reqnctapercluster nx
    .reqnctapercluster nx, ny
    .reqnctapercluster nx, ny, nz
    

Description

Set the number of thread blocks (CTAs) in the cluster by specifying the extent of each dimension of the 1D, 2D, or 3D cluster. The total number of CTAs is the product of the number of CTAs in each dimension. For kernels with `.reqnctapercluster` directive specified, runtime will use the specified values for configuring the launch if the same are not specified at launch time.

Semantics

If cluster dimension is explicitly specified at launch time, it should be equal to the values specified in this directive. Specifying a different cluster dimension at launch will result in a runtime error or kernel launch failure.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .entry foo .reqnctapercluster 2         { . . . }
    .entry bar .reqnctapercluster 2, 2, 1   { . . . }
    .entry ker .reqnctapercluster 3, 2      { . . . }
    

###  11.7.2. [Cluster Dimension Directives: `.explicitcluster`](#cluster-dimension-directives-explicitcluster)

`.explicitcluster`

Declare that Kernel must be launched with cluster dimensions explicitly specified.

Syntax
    
    
    .explicitcluster
    

Description

Declares that this Kernel should be launched with cluster dimension explicitly specified.

Semantics

Kernels with `.explicitcluster` directive must be launched with cluster dimension explicitly specified (either at launch time or via `.reqnctapercluster`), otherwise program will fail with runtime error or kernel launch failure.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .entry foo .explicitcluster         { . . . }
    

###  11.7.3. [Cluster Dimension Directives: `.maxclusterrank`](#cluster-dimension-directives-maxclusterrank)

`.maxclusterrank`

Declare the maximum number of CTAs that can be part of the cluster.

Syntax
    
    
    .maxclusterrank n
    

Description

Declare the maximum number of thread blocks (CTAs) allowed to be part of the cluster.

Semantics

Product of the number of CTAs in each cluster dimension specified in any invocation of the kernel is required to be less or equal to that specified in this directive. Otherwise invocation will result in a runtime error or kernel launch failure.

The `.maxclusterrank` directive cannot be used in conjunction with the `.reqnctapercluster` directive.

PTX ISA Notes

Introduced in PTX ISA version 7.8.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .entry foo ..maxclusterrank 8         { . . . }
    


##  11.8. [Miscellaneous Directives](#miscellaneous-directives)

PTX provides the following miscellaneous directives:

  * `.blocksareclusters`


###  11.8.1. [Miscellaneous Directives: `.blocksareclusters`](#miscellaneous-directives-blocksareclusters)

`.blocksareclusters`

Specify that CUDA thread blocks are mapped to clusters.

Syntax
    
    
    .blocksareclusters
    

Description

Default behavior of CUDA API is to specify the grid launch configuration by specifying the number of thread blocks and the number of threads per block.

When `.blocksareclusters` directive is specified, it implies that the grid launch configuration for the corresponding `.entry` function is specifying the number of clusters, i.e. the launch configuration is specifying number of clusters instead of the number of thread blocks. In this case, the number of thread blocks per cluster is specified by `.reqnctapercluster` directive and the thread block size is specified with the `.reqntid` directive.

`.blocksareclusters` directive is only allowed for `.entry` functions and also needs `.reqntid` and `.reqnctapercluster` directives to be specified.

Refer to _CUDA Programming Guide_ for more details.

PTX ISA Notes

Introduced in PTX ISA version 9.0.

Target ISA Notes

Requires `sm_90` or higher.

Examples
    
    
    .entry foo .reqntid 32, 32, 1 .reqnctapercluster 32, 32, 1 .blocksareclusters { ... }
    
