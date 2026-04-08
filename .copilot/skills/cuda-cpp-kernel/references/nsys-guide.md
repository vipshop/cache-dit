# Nsight Systems (nsys) Reference

## Table of Contents

- [Overview](#overview) — System-wide profiling and timeline analysis
- [Basic Commands](#basic-commands) — Profile, trace options, stats reports
- [Stats Reports](#stats-reports) — Key reports (cuda_gpu_kern_sum, cuda_api_sum, cuda_gpu_mem_time_sum, nvtx_sum)
- [Export Formats](#export-formats) — SQLite, CSV, JSON
- [Filtering](#filtering) — Time range, capture control
- [Analysis Patterns](#analysis-patterns) — Finding gaps, memory transfers, kernel overhead, NVTX-guided analysis
- [Troubleshooting](#troubleshooting) — Common issues

## Overview

nsys answers: "Where is time spent?" Use it for system-wide profiling, CPU/GPU interaction, memory transfers, and timeline analysis.

## Basic Commands

```bash
# Profile and save
nsys profile -o report ./program

# Profile with specific traces
nsys profile --trace=cuda,nvtx,osrt -o report ./program

# Stats from existing report
nsys stats report.nsys-rep
```

## Trace Options

```bash
--trace=cuda          # CUDA API and kernels
--trace=nvtx          # NVTX annotations
--trace=osrt          # OS runtime (pthread, etc.)
--trace=cublas        # cuBLAS calls
--trace=cudnn         # cuDNN calls
--trace=opengl        # OpenGL
--trace=none          # Disable all traces

# Common combinations
--trace=cuda,nvtx               # CUDA + custom markers
--trace=cuda,nvtx,osrt          # Full picture
--trace=cuda,cublas,cudnn       # Library-level analysis
```

## Stats Reports

After profiling, extract statistics:

```bash
# All available reports
nsys stats report.nsys-rep --help

# Kernel summary (most useful)
nsys stats report.nsys-rep --report cuda_gpu_kern_sum

# API call summary
nsys stats report.nsys-rep --report cuda_api_sum

# Memory operations
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum
nsys stats report.nsys-rep --report cuda_gpu_mem_size_sum

# NVTX ranges
nsys stats report.nsys-rep --report nvtx_sum

# Multiple reports at once
nsys stats report.nsys-rep --report cuda_gpu_kern_sum --report cuda_api_sum
```

### Key Reports Explained

**cuda_gpu_kern_sum** — Kernel execution summary
- Time: Total GPU time for each kernel
- Instances: Number of launches
- Avg/Min/Max: Per-instance timing
- Look for: Which kernels dominate runtime?

**cuda_api_sum** — Host-side CUDA API calls
- Shows: cudaMalloc, cudaMemcpy, cudaLaunchKernel, etc.
- Look for: Excessive sync calls, allocation overhead

**cuda_gpu_mem_time_sum** — Memory transfer timing
- HtoD, DtoH, DtoD operations
- Look for: Transfer bottlenecks, unnecessary copies

**nvtx_sum** — Custom annotation summary
- Your named regions aggregated
- Look for: Which phases dominate?

## Export Formats

```bash
# Export to SQLite (for custom queries)
nsys export -t sqlite -o report.sqlite report.nsys-rep

# Export stats to file
nsys stats report.nsys-rep --report cuda_gpu_kern_sum --format csv > kernels.csv

# JSON format
nsys stats report.nsys-rep --format json > stats.json
```

## Filtering

```bash
# Time range (seconds from start)
nsys profile --duration 10 ./program           # First 10 seconds
nsys profile --delay 5 --duration 10 ./program # Skip 5s, capture 10s

# Capture control from code (start paused)
nsys profile --capture-range=cudaProfilerApi ./program
# In code: cudaProfilerStart() / cudaProfilerStop()
```

## Analysis Patterns

### Pattern: Finding Gaps

Low GPU utilization often shows as gaps in the timeline. Causes:
1. CPU computation between kernel launches
2. Synchronization barriers (cudaDeviceSynchronize)
3. Memory allocation during runtime
4. Small kernels with high launch overhead

Investigation:
```bash
nsys profile --trace=cuda,osrt -o report ./program
nsys stats report.nsys-rep --report cuda_api_sum
# Look for: cudaDeviceSynchronize count and time
# Look for: cudaMalloc during steady state
```

### Pattern: Memory Transfer Analysis

```bash
nsys stats report.nsys-rep --report cuda_gpu_mem_time_sum
nsys stats report.nsys-rep --report cuda_gpu_mem_size_sum
```

Questions to answer:
- What % of time is memory transfer vs compute?
- Are transfers overlapped with compute?
- Can transfers be batched or eliminated?

### Pattern: Kernel Launch Overhead

Many small kernels = launch overhead dominates.

```bash
nsys stats report.nsys-rep --report cuda_gpu_kern_sum
```

Signs of trouble:
- Many kernels with <10μs execution time
- Large gap between kernel launches
- Solution: Kernel fusion, batching

### Pattern: NVTX-Guided Analysis

Add NVTX markers to code, then:
```bash
nsys profile --trace=cuda,nvtx -o report ./program
nsys stats report.nsys-rep --report nvtx_sum
```

This shows aggregate time per named region across all iterations.

## Common Options

```bash
# Output naming
-o name              # Base name for output files
-f true              # Force overwrite existing

# Sampling
--sample=cpu         # CPU sampling (for CPU profiling)
--sample=none        # Disable CPU sampling (faster)

# GPU metrics (adds overhead)
--gpu-metrics-device=all  # Collect GPU metrics

# Process selection
--trace-fork-before-exec=true  # Follow forks
```

## Troubleshooting

**"No CUDA data collected"**
- Ensure program actually uses CUDA
- Check `--trace=cuda` is specified
- Verify CUDA toolkit is properly installed

**Report too large**
- Use `--duration` to limit capture time
- Use `--trace` to capture only needed data
- Disable CPU sampling with `--sample=none`

**Missing kernel names**
- Compile with `-lineinfo` for source correlation
- Kernel names come from CUDA runtime
