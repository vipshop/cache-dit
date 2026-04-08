---
name: cutlass-cpp-kernel
description: 'Use when writing, debugging, porting, reviewing, or optimizing CUTLASS or CuTe C++ kernels and templates; navigating CUTLASS examples, collectives, epilogues, pipelines, GEMM schedules, or CuTe headers; or analyzing template configuration, tiling, memory movement, and kernel structure for Hopper or Blackwell GPUs.'
argument-hint: 'Describe the CUTLASS or CuTe C++ task, target architecture, data types, kernel family, and whether the work is source navigation, implementation, optimization, or a rewrite.'
user-invocable: true
---

# CUTLASS and CuTe C++ Kernel Development

## Goal

Use the workspace CUTLASS checkout to understand, implement, and optimize CUTLASS- or CuTe-based C++ kernels while keeping CuTe DSL Python authoring in a separate dedicated workflow.

## When to Use

Use this skill when you need to:

- study or modify CUTLASS GEMM, epilogue, pipeline, or collective-builder code
- navigate CuTe C++ headers such as `layout.hpp`, `tensor.hpp`, `swizzle.hpp`, and `atom/*`
- inspect CUTLASS example kernels for Hopper, Blackwell, FP8, FP4, grouped GEMM, sparse GEMM, or MoE patterns
- reason about template choices such as tile shapes, kernel schedules, epilogue schedules, stage count, and copy or MMA atoms
- debug CUTLASS template errors, layout mismatches, or schedule/configuration issues
- review a rewrite plan from an existing C++ kernel to another CUTLASS- or CuTe-based implementation

Do not use this skill for:

- CuTe DSL Python kernel authoring as the primary task; use `cute-dsl-kernel`
- generic CUDA/PTX debugging with no CUTLASS or CuTe angle; use `cuda-cpp-kernel`
- cache-dit integration plumbing by itself; pair with `operator-migration`

## Scope Split

This skill is the main home for:

- CUTLASS C++ source navigation
- CuTe C++ header semantics
- collective and pipeline configuration
- example-based C++ implementation patterns

For CuTe DSL Python kernels, JIT flows, and generated CuTe DSL API reference files, switch to `cute-dsl-kernel`.

## Workspace Source Location

In this workspace, the CUTLASS checkout is:

- workspace-relative: `vipshop/cutlass`
- shell path: `/workspace/dev/vipshop/cutlass`

Do not rely on skill-local repository mirrors, update scripts, or any agent-local install path.

## Reference Style Rule

When citing CUTLASS sources, prefer workspace-relative paths such as:

- `vipshop/cutlass/include/cutlass/gemm/collective/`
- `vipshop/cutlass/include/cute/layout.hpp`
- `vipshop/cutlass/examples/49_hopper_gemm_with_collective_builder/`

Use absolute shell paths only inside literal command examples when required.

## Bundled CUDA Architecture References

This skill also bundles low-level CUDA architecture profiling references so CUTLASS tuning can be interpreted in the right hardware context.

Use these files when reading Nsight Systems or Nsight Compute results for CUTLASS or CuTe C++ kernels:

- `sm89-optimization-guide.md`
- `sm90-optimization-guide.md`
- `sm100-optimization-guide.md`
- `sm103-optimization-guide.md`
- `sm120-optimization-guide.md`
- `troubleshooting.md`

These guides are especially useful when a CUTLASS kernel behaves differently across Ada, Hopper, Blackwell datacenter, and Blackwell desktop targets.

## Key Source Map

Primary areas to inspect:

- `vipshop/cutlass/include/cutlass/` — CUTLASS library headers
- `vipshop/cutlass/include/cutlass/gemm/collective/` — collective mainloop and epilogue building blocks
- `vipshop/cutlass/include/cutlass/pipeline/` — pipeline abstractions
- `vipshop/cutlass/include/cute/` — CuTe C++ core headers
- `vipshop/cutlass/include/cute/arch/` — architecture-specific copies and MMA helpers
- `vipshop/cutlass/include/cute/atom/` — MMA and copy atoms
- `vipshop/cutlass/examples/` — executable reference implementations
- `vipshop/cutlass/examples/cute/tutorial/` — CuTe C++ tutorial kernels
- `vipshop/cutlass/media/docs/pythonDSL/` — CuTe DSL conceptual and workflow docs useful when reviewing C++ to CuTe DSL rewrites

## Search Workflow

Prefer targeted search rather than reading large header trees end-to-end.

Start from the most likely source family:

1. `examples/` for a runnable pattern close to your target.
2. `include/cutlass/gemm/collective/` for collective-builder and mainloop configuration.
3. `include/cutlass/epilogue/` for fusion and output transforms.
4. `include/cutlass/pipeline/` for stage and async-copy structure.
5. `include/cute/` for layout algebra, tensor partitioning, swizzle, and atom semantics.

For performance diagnosis, pair source study with the bundled architecture guides:

1. On `sm89` and `sm120`, prioritize memory throughput, L2 hit rate, occupancy, and the cost of not having TMA or cluster features.
2. On `sm90`, check whether the design is actually exploiting Hopper-specific staging and overlap opportunities.
3. On `sm100` and `sm103`, verify that the kernel structure aligns with WGMMA, TMEM, TMA v2, and cluster-capable execution rather than only recompiling an older design.

## Architecture-Specific Nsight Guidance

When using this skill for optimization or rewrites, do not read `nsys` or `ncu` output in isolation.

Recommended workflow:

1. Read the relevant `smXX-optimization-guide.md` first.
2. Use `nsys` to determine whether the CUTLASS kernel has launch gaps, poor overlap, or a fusion opportunity at the operator level.
3. Use `ncu` to determine whether the issue is occupancy, memory throughput, L2 reuse, register pressure, shared-memory pressure, or architecture-specific execution features.
4. Only then decide whether to change tile shapes, stage count, epilogue schedule, kernel schedule, or data-movement strategy.

This matters most when comparing the same CUTLASS-style operator across multiple architectures.

## Implementation Workflow

Before editing code, answer these questions:

1. Which existing CUTLASS example is the closest semantic starting point?
2. Which layout, copy, and MMA atoms define the kernel's data movement?
3. Which collective, schedule, and stage decisions matter for the target architecture?
4. What public operator contract or wrapper must remain stable?
5. What tests and benchmarks will prove the rewrite is valid?

If the task becomes repository integration work, move that part to `operator-migration` and keep this skill focused on kernel structure and source study.

For architecture-specific bottlenecks or Nsight interpretation questions, use the bundled optimization guides as supporting reference material instead of assuming the same diagnosis applies across Ada, Hopper, and Blackwell.

## Rewrite Guidance

When the task is a rewrite, such as moving an operator from handwritten C++ to CUTLASS, or replacing one CUTLASS design with another:

1. Preserve the operator contract first.
2. Keep shape, dtype, alignment, and epilogue semantics explicit.
3. Verify the new implementation against the original operator before claiming success.
4. Only optimize after correctness and parity are established.

If the target implementation is CuTe DSL Python rather than C++ templates, use this skill for source study and switch to `cute-dsl-kernel` for the authoring workflow.

## Validation Requirements

Every operator or kernel task completed under this skill must include validation.

Minimum requirements:

1. Add or update unit tests.
2. Compare numerical accuracy against a PyTorch baseline or another trusted eager reference when applicable.
3. Compare performance against that baseline when the work replaces or claims to improve a baseline path.
4. Record benchmark setup details clearly.

Additional requirement for rewrites or ports:

1. Compare the new implementation against the pre-rewrite operator on both accuracy and performance.
2. Treat "PyTorch baseline" and "previous operator implementation" as separate comparisons when both are available.
3. If a rewrite changes schedules, stages, or layouts, isolate whether the change improved throughput, latency, or both.

## Output Expectations

When you finish a task using this skill, report:

- which CUTLASS or CuTe source pattern was used as reference
- what layout, stage, or epilogue choices mattered
- what tests were added or run
- the PyTorch-baseline accuracy and performance result
- the old-versus-new operator comparison when a rewrite was involved
