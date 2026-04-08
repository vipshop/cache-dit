# Common Performance Traps in CUDA

*Lessons from real-world GPU optimization projects*

Plans help you start, but profiling reveals the real bottlenecks. The problems below are frequently discovered through systematic profiling but often missed in initial designs.

## Bank Conflicts in Shared Memory

### Symptoms
- ncu shows high bank conflict rate (e.g., "16-way bank conflicts")
- Most cycles stalled on shared memory operations
- Low effective bandwidth despite using shared memory

### Common Causes
- Strided access patterns that map multiple threads to the same bank
- Transposing data in shared memory without accounting for bank layout
- Writing in one pattern, reading in another (both can't be conflict-free simultaneously)

### Investigation
```bash
# Check bank conflicts and wavefronts
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    ./program

# Divide conflicts by wavefronts to get average conflicts per operation
# >1 per operation indicates conflicts
```

### Solutions
- Pad shared memory arrays (e.g., `[32][33]` instead of `[32][32]`)
- Change thread-to-data mapping to avoid stride patterns
- Optimize for the more frequent operation if both read and write can't be conflict-free
- For transpose operations, accept conflicts on one dimension

Can give 2-3× speedup when memory-bound.

## Memory Coalescing

### Symptoms
- ncu shows high sector/request ratio (e.g., "32 sectors/request" vs optimal 1-4)
- Low global memory throughput despite high demand
- Memory-bound kernel with poor bandwidth utilization

### Common Causes
- Strided access patterns (every thread reads every Nth element)
- Transposed access patterns (reading column-major when stored row-major)
- Unaligned access or indirection through index arrays

### Investigation
```bash
# Check memory transactions
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./program
# Divide sectors by requests: 1-4 is good, 8-16 is poor, 32+ is essentially random
```

### Solutions
- Use vectorized loads (`float4`, `uint4`) when threads access adjacent memory
- Structure of Arrays (SoA) over Array of Structures (AoS)
- Transpose in shared memory if global access must be strided
- Ensure proper alignment (128-byte for vector loads)

Can give 1.5-2× speedup for severe coalescing issues.

## Scale-Dependent Optimizations

### The Problem
**Optimization techniques that work at large scale can hurt at small scale, and vice versa.**

### Common Examples
- **Warp specialization:** Fixed setup cost only amortizes with large workloads
- **Async operations:** Only hide latency if you have compute to overlap
- **Advanced features (TMA, etc.):** Benefit at high utilization, overhead at low

### Rule
**Always profile at YOUR target scale.** "Best practices" from papers may not apply to your problem size.

### Questions to Ask
- This optimization helped at scale X. What's my scale?
- What's the overhead, and does my workload amortize it?
- Should I verify this applies before implementing?

## Documenting What Doesn't Work

**Document negative results to prevent retrying failed approaches:**

```markdown
## Attempted Optimizations

### Warp Specialization (Stage 9)
- Context: 64×64 tiles
- Result: Slower than baseline
- Reason: Setup overhead > benefit at small scale
- Decision: Don't retry until workload >128×128
```

This prevents loops where you try the same optimization again after losing context. Failed experiments are valuable knowledge if documented.

## Summary

1. **Profile first, always** — Intuition about bottlenecks is usually wrong
2. **Measure at your scale** — Advice from papers may not apply to your problem size
3. **One change at a time** — Compound changes make diagnosis impossible
4. **Document failures** — Prevent retrying what already failed
5. **Verify with metrics** — "Should work" ≠ "does work"

The profile → hypothesis → fix → verify loop is the core optimization methodology.
