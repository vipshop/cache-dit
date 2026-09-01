---
name: cute-dsl-kernel
description: 'Use when writing, modifying, porting, or optimizing CuTe DSL GPU kernels in Python; reading CuTe DSL API reference material; integrating a CuTe DSL kernel into a project; or rewriting an existing CUDA or C++ operator into CuTe DSL while preserving correctness and performance expectations.'
argument-hint: 'Describe the target kernel, tensor shapes, dtypes, target GPU architecture, required fusion, integration target, and whether this is a new kernel or a rewrite of an existing implementation.'
user-invocable: true
---

# Write a CuTe DSL GPU Kernel

## Goal

Use the bundled CuTe DSL API snapshots in this skill and the workspace CUTLASS checkout to design, implement, debug, and integrate CuTe DSL GPU kernels in a way that is reusable across projects, including cache-dit.

## When to Use

Use this skill when you need to:

- write or modify a CuTe DSL GPU kernel in Python
- study CuTe DSL types, runtime helpers, architecture APIs, or pipeline abstractions
- port or rewrite an existing CUDA or C++ operator into CuTe DSL
- use CuTe DSL examples from the workspace CUTLASS checkout as reference material
- debug CuTe DSL compilation, runtime behavior, layout issues, or integration problems

Do not use this skill for:

- CUTLASS or CuTe C++ template work as the primary task; use `cutlass-cpp-kernel`
- generic CUDA/PTX documentation lookup with no CuTe DSL angle; use `cuda-cpp-kernel`
- repository integration plumbing by itself; pair with `operator-migration`

## Core Rule

Read the relevant API reference files before writing kernel code.

Do not guess CuTe DSL APIs or architecture helpers from memory when the bundled docs or workspace CUTLASS examples can answer the question precisely.

## Reference Style Rule

Use Copilot-friendly sibling-file references for bundled docs in this skill, for example:

- `cute.md`
- `cute_runtime.md`
- `utils.md`
- `cute_nvgpu_tcgen05.md`
- `pipeline.md`

Use workspace-relative paths for CUTLASS sources, for example:

- `vipshop/cutlass/python/CuTeDSL/`
- `vipshop/cutlass/examples/python/CuTeDSL/`
- `vipshop/cutlass/python/pycute/`
- `vipshop/cutlass/include/cute/`
- `vipshop/cutlass/media/docs/pythonDSL/`

Do not use agent-specific skill paths or placeholder-driven argument text in the final skill content.

## Read These Files First

Core API references:

- `cute.md` — core CuTe DSL types and tensor or layout operations
- `cute_runtime.md` — runtime helpers and data interop
- `utils.md` — helper utilities and hardware info

Architecture-specific references:

- `cute_nvgpu.md` — architecture API index
- `cute_nvgpu_warp.md` — warp-level APIs for SM80 to SM89
- `cute_nvgpu_warpgroup.md` — warpgroup APIs for SM90
- `cute_nvgpu_tcgen05.md` — tcgen05 and SM100+ APIs
- `cute_nvgpu_cpasync.md` — async-copy APIs
- `cute_arch.md` — low-level architecture primitives
- `utils_sm90.md` and `utils_sm100.md` — architecture helpers

Pipeline and overview:

- `pipeline.md`
- `intro.md`

Additional workflow and concept references from the workspace CUTLASS docs:

- `vipshop/cutlass/media/docs/pythonDSL/overview.rst` — high-level positioning of CUTLASS DSLs and how CuTe DSL relates to CUTLASS C++
- `vipshop/cutlass/media/docs/pythonDSL/quick_start.rst` — environment, install, and setup assumptions
- `vipshop/cutlass/media/docs/pythonDSL/functionality.rst` — supported dtypes, architectures, and current feature scope
- `vipshop/cutlass/media/docs/pythonDSL/limitations.rst` — current CuTe DSL limitations and unsupported cases
- `vipshop/cutlass/media/docs/pythonDSL/faqs.rst` — common issues and expected behavior
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl.rst` — CuTe DSL workflow overview
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_api.rst` — API documentation entrypoint
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.rst` — DSL programming model and mental model
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_control_flow.rst` — control-flow semantics and restrictions
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_dynamic_layout.rst` — static vs dynamic layout handling
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.rst` — JIT argument typing and signature generation
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_caching.rst` — JIT cache behavior
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_jit_compilation_options.rst` — compilation flags and debugging options
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/framework_integration.rst` — framework interop patterns
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.rst` — AOT compilation and export flow
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/debugging.rst` — debugging workflow and generated-artifact inspection
- `vipshop/cutlass/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.rst` — autotuning guidance for GEMM kernels

These workspace docs are especially valuable when the bundled API snapshots are too terse for workflow, compilation, debugging, or integration questions.

CUDA architecture and profiling references bundled in this skill:

- `sm89-optimization-guide.md`
- `sm90-optimization-guide.md`
- `sm100-optimization-guide.md`
- `sm103-optimization-guide.md`
- `sm120-optimization-guide.md`
- `troubleshooting.md`

Use these files when interpreting `nsys` and `ncu` results for generated CuTe DSL kernels on different GPU families.

## Workspace Source Map

Use the workspace CUTLASS checkout for source examples and implementation patterns.

Key locations:

- `vipshop/cutlass/python/CuTeDSL/` — CuTe DSL implementation sources
- `vipshop/cutlass/examples/python/CuTeDSL/` — CuTe DSL examples by architecture and topic
- `vipshop/cutlass/python/pycute/` — pycute helpers and layout utilities
- `vipshop/cutlass/include/cute/` — CuTe C++ headers for semantic grounding

Use the shell path `/workspace/dev/vipshop/cutlass` only when you need a literal command path.

## Architecture-Specific Profiling Guidance

CuTe DSL kernels often need architecture-aware profiling because the generated kernel structure can look similar while the best bottleneck diagnosis differs by GPU generation.

Use the bundled optimization guides as follows:

1. On `sm89` and `sm120`, prioritize memory throughput, L2 hit rate, occupancy, and fusion opportunity; these targets do not have TMA, TMEM, or cluster features.
2. On `sm90`, inspect whether TMA-style overlap, warpgroup execution, and shared-memory staging are actually visible in the timeline and counters.
3. On `sm100` and `sm103`, inspect whether tcgen05 or WGMMA, TMEM, TMA v2, and cluster-capable execution are being used effectively.

Recommended profiling order:

1. Read the relevant `smXX-optimization-guide.md` file.
2. Use `nsys` to identify launch gaps, missing overlap, copy or compute imbalance, and end-to-end bottlenecks.
3. Use `ncu` to inspect occupancy, memory throughput, L2 hit rate, register pressure, shared-memory pressure, tensor core utilization, and stall reasons.
4. Only then decide whether to change tiling, pipelining, copy strategy, or fusion structure.

## Implementation Workflow

Before writing code, answer these questions:

1. What are the input and output shapes, dtypes, and memory-layout constraints?
2. What is the target architecture: SM80, SM89, SM90, SM100, or newer?
3. Is this a brand-new kernel or a rewrite of an existing operator?
4. Which CuTe DSL APIs and source examples are the closest match?
5. How will the compiled kernel integrate into the target project?

Then work in this order:

1. Read the relevant bundled API docs.
2. Read the relevant conceptual or workflow docs under `vipshop/cutlass/media/docs/pythonDSL/` when the question is about control flow, JIT behavior, debugging, AOT, integration, or limitations.
3. Read the closest source example in `vipshop/cutlass/examples/python/CuTeDSL/`.
4. Decide the kernel structure: elementwise, reduction, tiled GEMM, fused kernel, or another pattern.
5. Implement the kernel and launch path.
6. Integrate the compiled artifact or launcher into the target repository.
7. Run correctness tests before performance tuning.

When tuning the generated kernel, treat the bundled `smXX-optimization-guide.md` files as first-line references for interpreting profiling output rather than relying only on generic CUDA advice.

## Integration Guidance

Keep integration guidance generic unless the target repository requires a specific loader or manifest format.

For cache-dit or other repositories:

1. preserve the external operator contract first
2. keep the integration layer explicit rather than burying it inside the kernel definition
3. document any generated artifact layout, launcher assumptions, or runtime dependencies
4. if repository-level registration or packaging changes are needed, pair this skill with `operator-migration`

## Rewrite Guidance

When rewriting an existing operator into CuTe DSL:

1. preserve behavior before optimizing
2. preserve shape, dtype, and numerics explicitly
3. keep the old implementation available long enough to benchmark and compare
4. do not claim success based only on compilation or a smoke test

Use `cutlass-cpp-kernel` alongside this skill when you need C++ CUTLASS or CuTe source study to understand the original design.

## Debugging Workflow

1. Use compile-time inspection for layouts, tiling, and static shapes.
2. Use runtime printing sparingly for GPU-side debugging.
3. Save PTX or IR when you need to inspect code generation.
4. Reduce the problem to the smallest shape that still reproduces the failure.
5. If the kernel relies on shared memory, `cp.async`, pipeline stages, or other asynchronous movement, treat synchronization as a primary suspect. When only specific shapes or pipeline configurations produce bad outputs, first inspect barrier placement, shared-stage reuse, and predicate coverage on partial-tile loads or stores.
6. Once correctness is stable, profile before tuning.

## Validation Requirements

Every operator or kernel task completed under this skill must include validation.

Minimum requirements:

1. Add or update unit tests.
2. Compare numerical accuracy against a PyTorch baseline or another trusted eager reference when applicable.
3. Compare performance against that baseline when the kernel is meant to replace or outperform it.
4. Record benchmark setup details clearly.

Additional requirement for rewrites or migrations:

1. If the task rewrites an existing operator, such as replacing a C++ implementation with CuTe DSL, compare the new kernel against the pre-rewrite implementation on both accuracy and performance.
2. Treat the PyTorch baseline and the previous implementation as separate validation targets when both are available.
3. Explain any gap that remains after the rewrite instead of masking it with only a single favorable benchmark.

## Output Expectations

When you finish a task using this skill, report:

- which bundled docs and source examples were used
- what integration assumptions were introduced
- what tests were added or run
- the PyTorch-baseline accuracy and performance result
- the old-versus-new operator comparison when the task was a rewrite
