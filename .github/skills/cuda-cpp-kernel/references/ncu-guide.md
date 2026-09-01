# Nsight Compute (ncu) Reference

## Table of Contents

- [Overview](#overview) — Detailed kernel analysis and metrics
- [Basic Commands](#basic-commands) — Profile kernels, save reports, metric sets
- [Metric Sets](#metric-sets) — basic, full, memory, compute, launch, occupancy, roofline, source
- [Sections](#sections) — LaunchStatistics, Occupancy, MemoryWorkloadAnalysis, ComputeWorkloadAnalysis, SchedulerStatistics
- [Key Metrics Explained](#key-metrics-explained) — SpeedOfLight, Occupancy, Memory/Compute Throughput
- [Output Formats](#output-formats) — CSV, page format, summary
- [Filtering and Selection](#filtering-and-selection) — By kernel name, invocation, range
- [Analysis Patterns](#analysis-patterns) — Memory vs compute bound, coalescing, bank conflicts, occupancy, source-level
- [Specific Metrics Query](#specific-metrics-query) — Query and use individual metrics
- [Expert System Caveats](#expert-system-caveats) — Use recommendations with caution
- [Troubleshooting](#troubleshooting) — Common issues

## Overview

ncu answers: "Why is this kernel slow?" Use it for detailed kernel analysis: memory throughput, compute utilization, occupancy, roofline analysis.

**Warning:** ncu adds significant overhead. Profile specific kernels, not entire applications.

## Basic Commands

```bash
# Profile all kernels (slow)
ncu ./program

# Profile specific kernel by name
ncu --kernel-name "myKernel" ./program

# Profile kernel by regex
ncu --kernel-name-base "regex:.*Reduce.*" ./program

# Skip first N invocations, profile next M
ncu --kernel-name "myKernel" --launch-skip 10 --launch-count 5 ./program

# Save to file
ncu -o report ./program

# Print to stdout
ncu --print-summary per-kernel ./program
```

## Metric Sets

Pre-defined metric collections:

```bash
--set basic       # Essential metrics (fast)
--set full        # Everything (slow, comprehensive)
--set memory      # Memory subsystem focus
--set compute     # Compute throughput focus
--set launch      # Launch configuration
--set occupancy   # Occupancy analysis
--set roofline    # Roofline model data
--set source      # Per-source-line metrics (needs -lineinfo)
```

**Recommendation:** Start with `--set basic`, deep-dive with specific sections.

## Sections

Fine-grained metric groups:

```bash
ncu --section LaunchStatistics ./program
ncu --section Occupancy ./program
ncu --section MemoryWorkloadAnalysis ./program
ncu --section ComputeWorkloadAnalysis ./program
ncu --section SchedulerStatistics ./program
ncu --section WarpStateStatistics ./program
ncu --section SpeedOfLight ./program
ncu --section SourceCounters ./program
```

Combine multiple:
```bash
ncu --section Occupancy --section MemoryWorkloadAnalysis ./program
```

## Key Metrics Explained

### SpeedOfLight (SOL)

Shows how close kernel is to theoretical peak:

- **SM [%]** — Compute throughput vs peak
- **Memory [%]** — Memory throughput vs peak

Interpretation:
- High SM, Low Memory → Compute bound
- Low SM, High Memory → Memory bound
- Low both → Latency bound (bad)

### Occupancy

- **Theoretical Occupancy** — Max possible given resources
- **Achieved Occupancy** — Actual average active warps
- **Limiting Factor** — What constrains occupancy (registers, shared mem, block size)

Low achieved vs theoretical = scheduling issues, not resource limits.

### Memory Throughput

- **DRAM Throughput** — Global memory bandwidth utilization
- **L2 Throughput** — L2 cache utilization
- **L1 Throughput** — L1/shared memory utilization

Key questions:
- Is memory coalesced? (Check "Memory Workload Analysis")
- Cache hit rates?
- Shared memory bank conflicts?

### Compute Throughput

- **SM Busy** — % time SMs have work
- **Warp Cycles per Issued Instruction** — Lower is better
- **Eligible Warps per Scheduler** — Higher is better

### Launch Statistics

- **Grid Size** — Total blocks
- **Block Size** — Threads per block
- **Registers/Thread** — Register usage
- **Shared Memory/Block** — Static + dynamic shared mem
- **Theoretical Occupancy** — Max warps limited by resources

## Output Formats

```bash
# CSV (for scripting)
ncu --csv ./program > metrics.csv

# CSV with specific metrics
ncu --csv --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed ./program

# Page format (human readable)
ncu --page raw ./program    # Raw metrics
ncu --page details ./program # With descriptions

# Print summary
ncu --print-summary per-kernel ./program
ncu --print-summary per-gpu ./program
```

## Filtering and Selection

```bash
# By kernel name
--kernel-name "exact_name"
--kernel-name-base "regex:pattern"

# By invocation
--launch-skip N      # Skip first N launches
--launch-count M     # Profile M launches

# By kernel ID (from previous runs)
--kernel-id N

# Range
--launch-skip 100 --launch-count 10  # Invocations 100-109
```

## Analysis Patterns

### Pattern: Memory vs Compute Bound

```bash
ncu --section SpeedOfLight --kernel-name "myKernel" ./program
```

Look at SM% vs Memory%:
- SM >> Memory → Compute bound, optimize arithmetic
- Memory >> SM → Memory bound, optimize access patterns
- Both low → Latency bound, improve occupancy

### Pattern: Coalescing Analysis

```bash
ncu --section MemoryWorkloadAnalysis --kernel-name "myKernel" ./program
```

Check MemoryWorkloadAnalysis section for sector/request ratios and theoretical vs actual throughput.

**Specific metrics for coalescing:**
```bash
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./program
# Divide sectors by requests: 1-4 is good, 8-16 is poor, 32+ is very poor
```

Signs of poor coalescing:
- High sector/request ratios
- Low memory throughput despite high memory traffic

### Pattern: Bank Conflicts Analysis

```bash
ncu --section SchedulerStatistics --kernel-name "myKernel" ./program
```

Check "Warp Stall" breakdown to see if stalls are memory-related.

**Specific metrics for bank conflicts:**
```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum \
    ./program
# Divide conflicts by wavefronts to get average conflicts per operation
# >1 per operation indicates significant conflicts
```

### Pattern: Occupancy Investigation

```bash
ncu --section Occupancy --section LaunchStatistics --kernel-name "myKernel" ./program
```

If occupancy is limited by:
- **Registers** → Use `--maxrregcount` or `__launch_bounds__`
- **Shared memory** → Reduce shared mem or use smaller blocks
- **Block size** → Adjust block dimensions

### Pattern: Source-Level Analysis

Compile with `-lineinfo -G`, then:
```bash
ncu --set source --kernel-name "myKernel" ./program
```

Shows metrics per source line. Useful for:
- Finding hot spots within kernel
- Identifying specific memory access issues

### Pattern: Compare Kernel Versions

```bash
# Profile both
ncu -o before --kernel-name "myKernel" ./program_before
ncu -o after --kernel-name "myKernel" ./program_after

# Compare (GUI required for visual diff)
# Or export CSV and diff
```

## Specific Metrics Query

```bash
# List available metrics
ncu --query-metrics

# Profile specific metrics
ncu --metrics sm__throughput.avg_pct_of_peak_sustained_elapsed,\
dram__throughput.avg_pct_of_peak_sustained_elapsed ./program

# Common useful metrics
--metrics sm__throughput.avg_pct_of_peak_sustained_elapsed  # SM utilization
--metrics dram__throughput.avg_pct_of_peak_sustained_elapsed # Memory bandwidth
--metrics sm__warps_active.avg_pct_of_peak_sustained_elapsed # Active warps
--metrics launch__occupancy_limit_registers               # Reg-limited occupancy
--metrics launch__occupancy_limit_shared_mem              # Smem-limited occupancy
--metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum # Bank conflicts (loads)
--metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum # Bank conflicts (stores)
--metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  # Memory sectors (coalescing)
--metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum # Memory requests (coalescing)
```

## Expert System Caveats

ncu includes an "expert system" that provides recommendations. **Use with caution:**

- Recommendations are heuristic-based, not always applicable
- May suggest conflicting optimizations
- Always verify with actual profiling after changes
- Context matters — what works for one kernel may hurt another

Treat recommendations as hypotheses to test, not prescriptions.

## Troubleshooting

**"No kernels profiled"**
- Check kernel name spelling (exact match required)
- Use `--kernel-name-base "regex:.*"` to see all kernels
- Verify kernel actually runs

**Very slow profiling**
- Use `--set basic` instead of `--set full`
- Profile fewer kernel invocations with `--launch-count`
- Skip warmup with `--launch-skip`

**Missing source information**
- Compile with `-lineinfo` (release) or `-G` (debug)
- Ensure binary matches source
