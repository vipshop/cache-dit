# 4. Syntax


PTX programs are a collection of text source modules (files). PTX source modules have an assembly-language style syntax with instruction operation codes and operands. Pseudo-operations specify symbol and addressing management. The ptxas optimizing backend compiler optimizes and assembles PTX source modules to produce corresponding binary object files.


##  4.1. [Source Format](#source-format)

Source modules are ASCII text. Lines are separated by the newline character (`\n`).

All whitespace characters are equivalent; whitespace is ignored except for its use in separating tokens in the language.

The C preprocessor cpp may be used to process PTX source modules. Lines beginning with `#` are preprocessor directives. The following are common preprocessor directives:

`#include`, `#define`, `#if`, `#ifdef`, `#else`, `#endif`, `#line`, `#file`

_C: A Reference Manual_ by Harbison and Steele provides a good description of the C preprocessor.

PTX is case sensitive and uses lowercase for keywords.

Each PTX module must begin with a `.version` directive specifying the PTX language version, followed by a `.target` directive specifying the target architecture assumed. See [PTX Module Directives](#ptx-module-directives) for a more information on these directives.


##  4.2. [Comments](#comments)

Comments in PTX follow C/C++ syntax, using non-nested `/*` and `*/` for comments that may span multiple lines, and using `//` to begin a comment that extends up to the next newline character, which terminates the current line. Comments cannot occur within character constants, string literals, or within other comments.

Comments in PTX are treated as whitespace.


##  4.3. [Statements](#statements)

A PTX statement is either a directive or an instruction. Statements begin with an optional label and end with a semicolon.

Examples
    
    
            .reg     .b32 r1, r2;
            .global  .f32  array[N];
    
    start:  mov.b32   r1, %tid.x;
            shl.b32   r1, r1, 2;          // shift thread id by 2 bits
            ld.global.b32 r2, array[r1];  // thread[tid] gets array[tid]
            add.f32   r2, r2, 0.5;        // add 1/2
    

###  4.3.1. [Directive Statements](#directive-statements)

Directive keywords begin with a dot, so no conflict is possible with user-defined identifiers. The directives in PTX are listed in [Table 1](#directive-statements-ptx-directives) and described in [State Spaces, Types, and Variables](#state-spaces-types-and-variables) and [Directives](#directives).

Table 1 PTX Directives `.address_size` | `.explicitcluster` | `.maxnreg` | `.section`  
---|---|---|---  
`.alias` | `.extern` | `.maxntid` | `.shared`  
`.align` | `.file` | `.minnctapersm` | `.sreg`  
`.branchtargets` | `.func` | `.noreturn` | `.target`  
`.callprototype` | `.global` | `.param` | `.tex`  
`.calltargets` | `.loc` | `.pragma` | `.version`  
`.common` | `.local` | `.reg` | `.visible`  
`.const` | `.maxclusterrank` | `.reqnctapercluster` | `.weak`  
`.entry` | `.maxnctapersm` | `.reqntid` |   
  
###  4.3.2. [Instruction Statements](#instruction-statements)

Instructions are formed from an instruction opcode followed by a comma-separated list of zero or more operands, and terminated with a semicolon. Operands may be register variables, constant expressions, address expressions, or label names. Instructions have an optional guard predicate which controls conditional execution. The guard predicate follows the optional label and precedes the opcode, and is written as `@p`, where `p` is a predicate register. The guard predicate may be optionally negated, written as `@!p`.

The destination operand is first, followed by source operands.

Instruction keywords are listed in [Table 2](#instruction-statements-reserved-instruction-keywords-new). All instruction keywords are reserved tokens in PTX.

Table 2 Reserved Instruction Keywords `abs` | `cvta` | `membar` | `setp` | `vabsdiff`  
---|---|---|---|---  
`activemask` | `discard` | `min` | `shf` | `vabsdiff2`  
`add` | `div` | `mma` | `shfl` | `vabsdiff4`  
`addc` | `dp2a` | `mov` | `shl` | `vadd`  
`alloca` | `dp4a` | `movmatrix` | `shr` | `vadd2`  
`and` | `elect` | `mul` | `sin` | `vadd4`  
`applypriority` | `ex2` | `mul24` | `slct` | `vavrg2`  
`atom` | `exit` | `multimem` | `sqrt` | `vavrg4`  
`bar` | `fence` | `nanosleep` | `st` | `vmad`  
`barrier` | `fma` | `neg` | `stackrestore` | `vmax`  
`bfe` | `fns` | `not` | `stacksave` | `vmax2`  
`bfi` | `getctarank` | `or` | `stmatrix` | `vmax4`  
`bfind` | `griddepcontrol` | `pmevent` | `sub` | `vmin`  
`bmsk` | `isspacep` | `popc` | `subc` | `vmin2`  
`bra` | `istypep` | `prefetch` | `suld` | `vmin4`  
`brev` | `ld` | `prefetchu` | `suq` | `vote`  
`brkpt` | `ldmatrix` | `prmt` | `sured` | `vset`  
`brx` | `ldu` | `rcp` | `sust` | `vset2`  
`call` | `lg2` | `red` | `szext` | `vset4`  
`clz` | `lop3` | `redux` | `tanh` | `vshl`  
`cnot` | `mad` | `rem` | `tcgen05` | `vshr`  
`copysign` | `mad24` | `ret` | `tensormap` | `vsub`  
`cos` | `madc` | `rsqrt` | `testp` | `vsub2`  
`clusterlaunchcontrol` | `mapa` | `sad` | `tex` | `vsub4`  
`cp` | `match` | `selp` | `tld4` | `wgmma`  
`createpolicy` | `max` | `set` | `trap` | `wmma`  
`cvt` | `mbarrier` | `setmaxnreg` | `txq` | `xor`


##  4.4. [Identifiers](#identifiers)  
  
User-defined identifiers follow extended C++ rules: they either start with a letter followed by zero or more letters, digits, underscore, or dollar characters; or they start with an underscore, dollar, or percentage character followed by one or more letters, digits, underscore, or dollar characters:
    
    
    followsym:   [a-zA-Z0-9_$]
    identifier:  [a-zA-Z]{followsym}* | {[_$%]{followsym}+
    

PTX does not specify a maximum length for identifiers and suggests that all implementations support a minimum length of at least 1024 characters.

Many high-level languages such as C and C++ follow similar rules for identifier names, except that the percentage sign is not allowed. PTX allows the percentage sign as the first character of an identifier. The percentage sign can be used to avoid name conflicts, e.g., between user-defined variable names and compiler-generated names.

PTX predefines one constant and a small number of special registers that begin with the percentage sign, listed in [Table 3](#identifiers-predefined-identifiers).

Table 3 Predefined Identifiers `%aggr_smem_size` | `%dynamic_smem_size` | `%lanemask_gt` | `%reserved_smem_offset_begin`  
---|---|---|---  
`%clock` | `%envreg<32>` | `%lanemask_le` | `%reserved_smem_offset_cap`  
`%clock64` | `%globaltimer` | `%lanemask_lt` | `%reserved_smem_offset_end`  
`%cluster_ctaid` | `%globaltimer_hi` | `%nclusterid` | `%smid`  
`%cluster_ctarank` | `%globaltimer_lo` | `%nctaid` | `%tid`  
`%cluster_nctaid` | `%gridid` | `%nsmid` | `%total_smem_size`  
`%cluster_nctarank` | `%is_explicit_cluster` | `%ntid` | `%warpid`  
`%clusterid` | `%laneid` | `%nwarpid` | `WARP_SZ`  
`%ctaid` | `%lanemask_eq` | `%pm0, ..., %pm7` |   
`%current_graph_exec` | `%lanemask_ge` | `%reserved_smem_offset_<2>` | 


##  4.5. [Constants](#constants)  
  
PTX supports integer and floating-point constants and constant expressions. These constants may be used in data initialization and as operands to instructions. Type checking rules remain the same for integer, floating-point, and bit-size types. For predicate-type data and instructions, integer constants are allowed and are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`.

###  4.5.1. [Integer Constants](#integer-constants)

Integer constants are 64-bits in size and are either signed or unsigned, i.e., every integer constant has type `.s64` or `.u64`. The signed/unsigned nature of an integer constant is needed to correctly evaluate constant expressions containing operations such as division and ordered comparisons, where the behavior of the operation depends on the operand types. When used in an instruction or data initialization, each integer constant is converted to the appropriate size based on the data or instruction type at its use.

Integer literals may be written in decimal, hexadecimal, octal, or binary notation. The syntax follows that of C. Integer literals may be followed immediately by the letter `U` to indicate that the literal is unsigned.
    
    
    hexadecimal literal:  0[xX]{hexdigit}+U?
    octal literal:        0{octal digit}+U?
    binary literal:       0[bB]{bit}+U?
    decimal literal       {nonzero-digit}{digit}*U?
    

Integer literals are non-negative and have a type determined by their magnitude and optional type suffix as follows: literals are signed (`.s64`) unless the value cannot be fully represented in `.s64` or the unsigned suffix is specified, in which case the literal is unsigned (`.u64`).

The predefined integer constant `WARP_SZ` specifies the number of threads per warp for the target platform; to date, all target architectures have a `WARP_SZ` value of 32.

###  4.5.2. [Floating-Point Constants](#floating-point-constants)

Floating-point constants are represented as 64-bit double-precision values, and all floating-point constant expressions are evaluated using 64-bit double precision arithmetic. The only exception is the 32-bit hex notation for expressing an exact single-precision floating-point value; such values retain their exact 32-bit single-precision value and may not be used in constant expressions. Each 64-bit floating-point constant is converted to the appropriate floating-point size based on the data or instruction type at its use.

Floating-point literals may be written with an optional decimal point and an optional signed exponent. Unlike C and C++, there is no suffix letter to specify size; literals are always represented in 64-bit double-precision format.

PTX includes a second representation of floating-point constants for specifying the exact machine representation using a hexadecimal constant. To specify IEEE 754 double-precision floating point values, the constant begins with `0d` or `0D` followed by 16 hex digits. To specify IEEE 754 single-precision floating point values, the constant begins with `0f` or `0F` followed by 8 hex digits.
    
    
    0[fF]{hexdigit}{8}      // single-precision floating point
    0[dD]{hexdigit}{16}     // double-precision floating point
    

Example
    
    
    mov.f32  $f3, 0F3f800000;       //  1.0
    

###  4.5.3. [Predicate Constants](#predicate-constants)

In PTX, integer constants may be used as predicates. For predicate-type data initializers and instruction operands, integer constants are interpreted as in C, i.e., zero values are `False` and non-zero values are `True`.

###  4.5.4. [Constant Expressions](#constant-expressions)

In PTX, constant expressions are formed using operators as in C and are evaluated using rules similar to those in C, but simplified by restricting types and sizes, removing most casts, and defining full semantics to eliminate cases where expression evaluation in C is implementation dependent.

Constant expressions are formed from constant literals, unary plus and minus, basic arithmetic operators (addition, subtraction, multiplication, division), comparison operators, the conditional ternary operator ( `?:` ), and parentheses. Integer constant expressions also allow unary logical negation (`!`), bitwise complement (`~`), remainder (`%`), shift operators (`<<` and `>>`), bit-type operators (`&`, `|`, and `^`), and logical operators (`&&`, `||`).

Constant expressions in PTX do not support casts between integer and floating-point.

Constant expressions are evaluated using the same operator precedence as in C. [Table 4](#constant-expressions-operator-precedence) gives operator precedence and associativity. Operator precedence is highest for unary operators and decreases with each line in the chart. Operators on the same line have the same precedence and are evaluated right-to-left for unary operators and left-to-right for binary operators.

Table 4 Operator Precedence Kind | Operator Symbols | Operator Names | Associates  
---|---|---|---  
Primary | `()` | parenthesis | n/a  
Unary | `+- ! ~` | plus, minus, negation, complement | right  
`(.s64)``(.u64)` | casts | right  
Binary | `*/ %` | multiplication, division, remainder | left  
`+-` | addition, subtraction  
`>> <<` | shifts  
`< > <= >=` | ordered comparisons  
`== !=` | equal, not equal  
`&` | bitwise AND  
`^` | bitwise XOR  
`|` | bitwise OR  
`&&` | logical AND  
`||` | logical OR  
Ternary | `?:` | conditional | right  
  
###  4.5.5. [Integer Constant Expression Evaluation](#integer-constant-expression-evaluation)

Integer constant expressions are evaluated at compile time according to a set of rules that determine the type (signed `.s64` versus unsigned `.u64`) of each sub-expression. These rules are based on the rules in C, but they’ve been simplified to apply only to 64-bit integers, and behavior is fully defined in all cases (specifically, for remainder and shift operators).

  * Literals are signed unless unsigned is needed to prevent overflow, or unless the literal uses a `U` suffix. For example:

    * `42`, `0x1234`, `0123` are signed.

    * `0xfabc123400000000`, `42U`, `0x1234U` are unsigned.

  * Unary plus and minus preserve the type of the input operand. For example:

    * `+123`, `-1`, `-(-42)` are signed.

    * `-1U`, `-0xfabc123400000000` are unsigned.

  * Unary logical negation (`!`) produces a signed result with value `0` or `1`.

  * Unary bitwise complement (`~`) interprets the source operand as unsigned and produces an unsigned result.

  * Some binary operators require normalization of source operands. This normalization is known as _the usual arithmetic conversions_ and simply converts both operands to unsigned type if either operand is unsigned.

  * Addition, subtraction, multiplication, and division perform the usual arithmetic conversions and produce a result with the same type as the converted operands. That is, the operands and result are unsigned if either source operand is unsigned, and is otherwise signed.

  * Remainder (`%`) interprets the operands as unsigned. Note that this differs from C, which allows a negative divisor but defines the behavior to be implementation dependent.

  * Left and right shift interpret the second operand as unsigned and produce a result with the same type as the first operand. Note that the behavior of right-shift is determined by the type of the first operand: right shift of a signed value is arithmetic and preserves the sign, and right shift of an unsigned value is logical and shifts in a zero bit.

  * AND (`&`), OR (`|`), and XOR (`^`) perform the usual arithmetic conversions and produce a result with the same type as the converted operands.

  * AND_OP (`&&`), OR_OP (`||`), Equal (`==`), and Not_Equal (`!=`) produce a signed result. The result value is 0 or 1.

  * Ordered comparisons (`<`, `<=`, `>`, `>=`) perform the usual arithmetic conversions on source operands and produce a signed result. The result value is `0` or `1`.

  * Casting of expressions to signed or unsigned is supported using (`.s64`) and (`.u64`) casts.

  * For the conditional operator ( `? :` ) , the first operand must be an integer, and the second and third operands are either both integers or both floating-point. The usual arithmetic conversions are performed on the second and third operands, and the result type is the same as the converted type.


###  4.5.6. [Summary of Constant Expression Evaluation Rules](#summary-of-constant-expression-evaluation-rules)

[Table 5](#summary-of-constant-expression-evaluation-rules-constant-expression-evaluation-rules) contains a summary of the constant expression evaluation rules.

Table 5 Constant Expression Evaluation Rules Kind | Operator | Operand Types | Operand Interpretation | Result Type  
---|---|---|---|---  
Primary | `()` | any type | same as source | same as source  
constant literal | n/a | n/a | `.u64`, `.s64`, or `.f64`  
Unary | `+-` | any type | same as source | same as source  
`!` | integer | zero or non-zero | `.s64`  
`~` | integer | `.u64` | `.u64`  
Cast | `(.u64)` | integer | `.u64` | `.u64`  
`(.s64)` | integer | `.s64` | `.s64`  
Binary | `+- * /` | `.f64` | `.f64` | `.f64`  
integer | use usual conversions | converted type  
`< > <= >=` | `.f64` | `.f64` | `.s64`  
integer | use usual conversions | `.s64`  
`== !=` | `.f64` | `.f64` | `.s64`  
integer | use usual conversions | `.s64`  
`%` | integer | `.u64` | `.s64`  
`>> <<` | integer | 1st unchanged, 2nd is `.u64` | same as 1st operand  
`& | ^` | integer | `.u64` | `.u64`  
`&& ||` | integer | zero or non-zero | `.s64`  
Ternary | `?:` | `int ? .f64 : .f64` | same as sources | `.f64`  
`int ? int : int` | use usual conversions | converted type
