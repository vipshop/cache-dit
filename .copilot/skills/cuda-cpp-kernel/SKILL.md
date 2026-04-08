---
name: cuda-cpp-kernel
description: 'Use when writing, debugging, porting, reviewing, or optimizing CUDA C++ or PTX kernels; investigating CUDA Runtime or Driver API behavior; profiling kernels with Nsight Systems or Nsight Compute; or reasoning about Tensor Core instructions, shared memory, bank conflicts, occupancy, async copy, TMA, WGMMA, and architecture-specific behavior on Ampere, Hopper, or Blackwell.'
argument-hint: 'Describe the kernel or CUDA problem, target GPU architecture, dtype, shapes, current bottleneck, and whether the task is implementation, debugging, optimization, or migration.'
user-invocable: true
---

# CUDA C++ and PTX Kernel Development

## Goal

Use the bundled CUDA, PTX, and profiling references in this skill to implement, debug, and optimize CUDA kernels without relying on agent-specific install paths or ad hoc web searches.

## When to Use

Use this skill when you need to:

- write or review CUDA C++ kernels or supporting host code
- reason about PTX instructions, inline PTX, Tensor Core instructions, or memory model details
- debug CUDA Runtime API or Driver API failures
- profile or optimize a kernel with Nsight Systems, Nsight Compute, compute-sanitizer, or cuda-gdb
- investigate shared memory bank conflicts, memory coalescing, occupancy, register pressure, async copy, TMA, or cluster behavior
- compare a custom operator or kernel against a PyTorch baseline for correctness or performance

Do not use this skill for:

- CUTLASS or CuTe template design as the primary task; use `cutlass-cpp-kernel`
- CuTe DSL Python kernel authoring as the primary task; use `cute-dsl-kernel`
- cache-dit operator registration, packaging, or public API migration work by itself; pair with `operator-migration`

## Reference Style Rule

Use skill-local relative paths for bundled references, for example:

- `references/ptx-docs/`
- `references/cuda-runtime-docs/`
- `references/ncu-docs/ProfilingGuide.md`

Do not write agent-specific install paths into follow-up notes or generated docs.

## Bundled Reference Map

The following directories are bundled under `references/` inside this skill:

- `references/ptx-docs/` — full PTX ISA reference
- `references/ptx-simple/` — condensed PTX quick reference
- `references/cuda-runtime-docs/` — CUDA Runtime API reference
- `references/cuda-driver-docs/` — CUDA Driver API reference
- `references/cuda-guide/` — CUDA Programming Guide
- `references/best-practices-guide/` — CUDA C++ Best Practices Guide
- `references/ncu-docs/` — Nsight Compute docs
- `references/nsys-docs/` — Nsight Systems docs
- `references/debugging-tools.md` — debugging workflow notes
- `references/performance-traps.md` — common optimization traps

The following architecture and kernel reference files are also bundled at the top level of this skill:

- `sm89-optimization-guide.md` — Ada-specific optimization and profiling guidance
- `sm90-optimization-guide.md` — Hopper-specific optimization and profiling guidance
- `sm100-optimization-guide.md` — Blackwell datacenter optimization and profiling guidance
- `sm103-optimization-guide.md` — Blackwell Ultra optimization and profiling guidance
- `sm120-optimization-guide.md` — Blackwell desktop or workstation optimization and profiling guidance
- `kernel-templates.md` — low-level CUDA kernel templates and implementation patterns
- `troubleshooting.md` — debugging, compute-sanitizer, and profiling troubleshooting notes

## How to Search the Bundle

Prefer narrow text search over loading large reference files into context.

Suggested workflow:

1. Identify the exact instruction, API, metric, or concept.
2. Search the narrowest relevant subdirectory first.
3. Read only the matching file or a short relevant range.
4. Translate the documentation into the specific kernel or operator constraint you are implementing.

Typical search targets:

- PTX instruction syntax: `references/ptx-docs/9-instruction-set/`
- Quick PTX lookup: `references/ptx-simple/`
- CUDA Runtime APIs: `references/cuda-runtime-docs/modules/`
- CUDA Driver APIs: `references/cuda-driver-docs/modules/`
- architecture and programming-model behavior: `references/cuda-guide/`
- profiling metrics and sections: `references/ncu-docs/ProfilingGuide.md`
- timeline and launch behavior: `references/nsys-docs/UserGuide.md`

For architecture-specific tuning and profiling, read the matching optimization guide first:

- `sm89-optimization-guide.md`
- `sm90-optimization-guide.md`
- `sm100-optimization-guide.md`
- `sm103-optimization-guide.md`
- `sm120-optimization-guide.md`

## Architecture-Specific Profiling Workflow

When using Nsight Systems or Nsight Compute, interpret the results in the context of the target architecture instead of treating all GPUs the same.

Use the bundled architecture guides as the first reference for cross-architecture analysis:

1. For `sm89`, focus on memory throughput, L2 hit rate, kernel fusion opportunity, and the lack of TMA or cluster features.
2. For `sm90`, focus on TMA overlap, warpgroup behavior, shared-memory staging, and whether the timeline shows good load or compute overlap.
3. For `sm100` and `sm103`, focus on WGMMA or tcgen05 usage, TMEM behavior, TMA v2 overlap, cluster behavior, and whether the kernel actually benefits from Blackwell datacenter features.
4. For `sm120`, treat it closer to Ada than to datacenter Blackwell for profiling purposes: watch memory throughput, L2 hit rate, shared-memory limits, and the absence of TMA, TMEM, or cluster features.

Recommended order:

1. Read the matching `smXX-optimization-guide.md` file.
2. Use `nsys` to identify launch gaps, overlap, copy or compute concurrency, and end-to-end bottlenecks.
3. Use `ncu` to inspect architecture-specific limits such as occupancy, memory throughput, L2 hit rate, tensor core utilization, stall reasons, register pressure, or shared-memory pressure.
4. Compare the observations against the architecture guide before changing tile shapes, pipelines, or memory movement.

## Implementation Checklist

Before changing code, answer these questions:

1. What is the exact shape, dtype, and layout contract?
2. What architectural assumptions exist, such as SM target, shared memory budget, alignment, or Tensor Core mode?
3. Is the bottleneck compute, memory, launch overhead, or synchronization?
4. Which CUDA Runtime, Driver, or PTX rules must be preserved?
5. What verification will prove the kernel is correct and faster enough to justify the change?

If the task is a migration into cache-dit, keep the kernel work separate from registration and packaging decisions and use `operator-migration` for the repository-integration layer.

## Debugging Workflow

1. Reproduce the issue with the smallest input that still fails.
2. Confirm the failure mode: wrong value, launch error, illegal memory access, race, hang, or performance regression.
3. Use compute-sanitizer or cuda-gdb for correctness problems.
4. Use Nsight Systems first for end-to-end bottlenecks, then Nsight Compute for per-kernel root cause.
5. After each change, rerun the focused correctness test before doing broader benchmarks.

## Performance Workflow

Never optimize by intuition alone.

1. Establish a baseline wall-clock measurement.
2. Use Nsight Systems to see where time is spent.
3. Use Nsight Compute to explain why the kernel is slow.
4. Change one dimension at a time: tile shape, memory movement, synchronization, vectorization, or epilogue structure.
5. Re-measure and compare against the previous version, not just the current absolute time.

If the result differs across GPU generations, consult the matching `smXX-optimization-guide.md` file before generalizing the bottleneck diagnosis.

## Validation Requirements

Every operator or kernel task completed under this skill must include validation.

Minimum requirements:

1. Add or update unit tests for correctness.
2. Compare numerical accuracy against a PyTorch baseline or another trusted high-level reference when applicable.
3. Compare performance against that baseline when the task claims a performance benefit or replaces a baseline path.
4. Record the exact benchmark setup: shapes, dtypes, device, warmup, iterations, and timing method.

Additional requirement for rewrites or migrations:

1. If you rewrite an existing operator, such as moving from one CUDA implementation to another, or from a handwritten kernel to a new implementation style, compare the new implementation against the pre-rewrite operator on both accuracy and performance.
2. Treat a PyTorch baseline and the pre-rewrite operator as separate comparisons when both exist.

## Output Expectations

When you finish a task using this skill, report:

- the implementation scope
- the main architectural or API constraints
- the tests added or run
- the PyTorch-baseline accuracy result
- the performance result
- if applicable, the old-versus-new operator comparison
