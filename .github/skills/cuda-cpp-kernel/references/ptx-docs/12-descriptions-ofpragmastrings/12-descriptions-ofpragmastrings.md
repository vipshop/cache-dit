# 12. Descriptions of.pragmaStrings


This section describes the `.pragma` strings defined by ptxas.


##  12.1. [Pragma Strings: `"nounroll"`](#pragma-strings-nounroll)

`"nounroll"`

Disable loop unrolling in the optimizing backend compiler.

Syntax
    
    
    .pragma "nounroll";
    

Description

The `"nounroll" pragma` is a directive to disable loop unrolling in the optimizing backend compiler.

The `"nounroll" pragma` is allowed at module, entry-function, and statement levels, with the following meanings:

module scope
    

disables unrolling for all loops in module, including loops preceding the `.pragma`.

entry-function scope
    

disables unrolling for all loops in the entry function body.

statement-level pragma
    

disables unrolling of the loop for which the current block is the loop header.

Note that in order to have the desired effect at statement level, the `"nounroll"` directive must appear before any instruction statements in the loop header basic block for the desired loop. The loop header block is defined as the block that dominates all blocks in the loop body and is the target of the loop backedge. Statement-level `"nounroll"` directives appearing outside of loop header blocks are silently ignored.

PTX ISA Notes

Introduced in PTX ISA version 2.0.

Target ISA Notes

Requires `sm_20` or higher. Ignored for `sm_1x` targets.

Examples
    
    
    .entry foo (...)
    .pragma "nounroll";  // do not unroll any loop in this function
    {
    ...
    }
    
    .func bar (...)
    {
    ...
    L1_head:
         .pragma "nounroll";  // do not unroll this loop
         ...
    @p   bra L1_end;
    L1_body:
         ...
    L1_continue:
         bra L1_head;
    L1_end:
         ...
    }
    


##  12.2. [Pragma Strings: `"used_bytes_mask"`](#pragma-strings-used-bytes-mask)

`"used_bytes_mask"`

Mask for indicating used bytes in data of ld operation.

Syntax
    
    
    .pragma "used_bytes_mask mask";
    

Description

The `"used_bytes_mask" pragma` is a directive that specifies used bytes in a load operation based on the mask provided.

`"used_bytes_mask" pragma` needs to be specified prior to a load instruction for which information about bytes used from the load operation is needed. Pragma is ignored if instruction following it is not a load instruction.

For a load instruction without this pragma, all bytes from the load operation are assumed to be used.

Operand `mask` is a 32-bit integer with set bits indicating the used bytes in data of load operation.

Semantics
    
    
    Each bit in mask operand corresponds to a byte data where each set bit represents the used byte.
    Most-significant bit corresponds to most-significant byte of data.
    
    // For 4 bytes load with only lower 3 bytes used
    .pragma "used_bytes_mask 0x7";
    ld.global.u32 %r0, [gbl];     // Higher 1 byte from %r0 is unused
    
    // For vector load of 16 bytes with lower 12 bytes used
    .pragma "used_bytes_mask 0xfff";
    ld.global.v4.u32 {%r0, %r1, %r2, %r3}, [gbl];  // %r3 unused
    

PTX ISA Notes

Introduced in PTX ISA version 8.3.

Target ISA Notes

Requires `sm_50` or higher.

Examples
    
    
    .pragma "used_bytes_mask 0xfff";
    ld.global.v4.u32 {%r0, %r1, %r2, %r3}, [gbl]; // Only lower 12 bytes used
    


##  12.3. [Pragma Strings: `"enable_smem_spilling"`](#pragma-strings-enable-smem-spilling)

`"enable_smem_spilling"`

Enable shared memory spilling for CUDA kernels.

Syntax
    
    
    .pragma "enable_smem_spilling";
    

Description

The `"enable_smem_spilling" pragma` is a directive that enables register spilling into shared memory. During the spilling process, registers are first spilled into shared memory, and once the allocated shared memory is full, any additional spills are redirected to local memory. This can enhance performance by reducing memory access latency since shared memory accesses are faster than local memory.

The `"enable_smem_spilling" pragma` is only allowed within the function scope. When applied, it enables shared memory spilling for the specified function.

The usage of pragma is valid only in certain scenarios and specific compilation modes. The usage of pragma is disallowed under following cases and may result in an error:

  * Per-function compilation mode: e.g., Separate Compilation, Device-debug, Whole program with recursive function calls, Extensible-whole-program

  * Kernels utilizing dynamically allocated shared memory

  * Kernels using `setmaxnreg` instruction


Note

If launch bounds are not explicitly specified, the compiler assumes the maximum possible number of threads per CTA to estimate shared memory allocated per CTA and corresponding spill size. However, if the kernel is launched with fewer threads per CTA than estimated, the shared memory allocated per CTA may exceed the compiler estimated size, thereby potentially limiting the number of CTAs that can be launched on an SM. Due to this, using the pragma without launch bounds may lead to performance regressions. Hence it is recommended to use this pragma only when launch bounds are explicitly specified.

PTX ISA Notes

Introduced in PTX ISA version 9.0.

Target ISA Notes

Requires `sm_75` or higher.

Examples
    
    
    .entry foo (...)
    {
        ...
        .pragma "enable_smem_spilling";   // Enable shared memory spilling for this function
        ...
    }
    


##  12.4. [Pragma Strings: `"frequency"`](#pragma-strings-frequency)

`"frequency"`

Specify frequency for basic block execution.

Syntax
    
    
    .pragma "frequency n";
    

Description

The `"frequency" pragma` is a directive that specifies the number of times a basic block is executed by an executing thread. The optimizing compiler backend treats this pragma as a hint which will be used for optimizations.

Operand `n` is a 64-bit non-negative integer constant that specifies the execution frequency.

Note that in order to have the desired effect of this pragma, it should be specified at the start of the basic block. Basic block is defined as a straight-line sequence of instructions with only one entry point and one exit point.

PTX ISA Notes

Introduced in PTX ISA version 9.0.

Target ISA Notes

Supported on all target architectures.

Examples
    
    
    .entry foo (...)
    {
        .pragma "frequency 32";
        ...
    }
    
