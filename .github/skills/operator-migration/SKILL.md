---
name: operator-migration
description: 'Use when doing operator migration or kernel migration for CUDA, Triton, or custom ops in cache-dit; porting kernels from nunchaku, deepcompressor, or other repos; designing operator registration and public wrappers; wiring build and packaging for optional extensions; or reviewing an operator migration plan. Guides survey, minimal-closure migration, API design, extension loading, packaging, and layered validation. Do not use for blind copy-paste ports.'
argument-hint: 'Describe the operator family, source repo, target public API, required backends or dtypes, and current migration status.'
user-invocable: true
---

# Operator Migration for cache-dit

## Goal

Migrate one operator or kernel family into cache-dit in a way that is:

- semantically correct
- aligned with cache-dit repository conventions
- safe to import when optional native extensions are absent
- validated at multiple layers instead of by one smoke test

This skill is for migration work that touches native code, Python wrappers, operator registration, build packaging, or quantized module integration.

## When to Use

Use this skill when you need to:

- migrate a CUDA or Triton operator from another repo into cache-dit
- port a nunchaku operator or kernel family into cache-dit
- decide what native files are actually required for a migration
- design cache-dit public wrappers for a newly migrated operator
- register low-level ops through cache-dit's CUDA registry layer
- add optional-extension build logic, submodule checks, or packaging guards
- design layered validation for a migrated operator
- review whether an operator migration plan is thoughtful or mechanical

Do not use this skill for:

- generic model integration with no operator or kernel work
- pure Python feature work unrelated to kernels or extensions
- blind "copy upstream into csrc" execution

## Core Rule

Do not mechanically replay upstream structure.

Treat the source repository as the reference for semantics, not as the required layout.

Before writing code, answer these questions:

1. What behavior is essential to preserve?
2. What is the smallest native and Python closure needed to preserve that behavior?
3. Which names should remain source-compatible, and which should be renamed to match cache-dit conventions?
4. What must be public, and what should remain private implementation detail?
5. Which tests prove the migration works, instead of merely compiling?

If those questions are not answered yet, do not start copying files.

## Reference Style Rule

Use portable references only.

- For cache-dit files, use repo-relative paths such as `src/cache_dit/kernels/ops.py` or `tests/kernels/test_svdquant_runtime.py`.
- For sibling or external repos, use repository-relative or GitHub-searchable paths such as `nunchaku/nunchaku/models/linear.py` or `deepcompressor/deepcompressor/backend/nunchaku/utils.py`.
- Do not write machine-local absolute paths such as `/abs/path/to/workspace/...` into the skill or its supporting documentation.

## Phase 0: Gather Before Coding

Collect the migration inputs first.

### Required inputs

1. Source operator and source repo
   Example: `nunchaku/nunchaku/ops/gemm.py` plus the native files it depends on.
2. Target cache-dit user-facing surface
   Example: a low-level op wrapper, a quantized module, or both.
3. Required backends, dtypes, and scope boundaries
   Example: "INT4 CUDA is required now; FP4 implementation may be retained but not gate current validation."
4. Build and packaging requirements
   Example: optional extension, submodule dependency, or environment gate.
5. Validation target
   Example: import safety, low-level parity, module parity, end-to-end inference, or shape rejection.

### Gather checklist

- Identify the source operator entry points.
- Identify the native files that implement them.
- Identify helper files that are truly required by those implementations.
- Identify existing cache-dit abstractions that should host the migrated behavior.
- Identify the minimum feature slice that must work first.
- Identify what will explicitly not be validated in the current milestone.

## Phase 1: Survey the Existing Design

Inspect both sides before making edits.

### Survey the source implementation

Look for:

- the true call chain from public API to kernel launch
- required helper headers, interop layers, dispatch utilities, and packers
- runtime assumptions such as shape, rank, alignment, architecture, or dtype restrictions
- dependency assumptions such as vendored headers, submodules, or environment variables
- test coverage that already encodes behavior worth preserving

### Survey cache-dit integration points

Common anchor files include:

- `src/cache_dit/kernels/ops.py`
- `src/cache_dit/kernels/cuda/_ops_registery.py`
- `src/cache_dit/kernels/cuda/_<feature>.py`
- `setup.py`
- `pyproject.toml`
- `tests/kernels/...`

Ask these questions while surveying:

1. Where should the public API live?
2. Where should `torch.library` registration live?
3. What should remain a private helper module under `src/cache_dit/kernels/cuda/`?
4. How should optional extension loading fail when the extension is missing?
5. Is there already a naming convention for this operator family?

## Phase 2: Decide the Migration Shape

Make the design decisions before editing files.

### 1. Freeze the public surface first

Define the cache-dit-facing API early.

Examples of questions to settle:

- Which operator names should be exposed publicly?
- Should the public API be low-level only, module-level only, or both?
- Should internal backend toggles be hidden from users?
- Should wrapper functions be explicit rather than `partial(...)` so editors and type tools can see the real signature?

Default rule: keep backend-selection details private unless there is a strong user-facing reason to expose them.

### 2. Migrate the minimal viable closure

Do not import an entire subsystem if only one slice is needed.

Usually migrate:

- the kernel implementation files that are actually on the call path
- the minimum helper headers or Python utilities they require
- the registry and wrapper plumbing needed to call them from cache-dit

Usually do not migrate yet:

- unrelated kernels in the same source repo directory
- optimization branches that are not needed for the current milestone
- extra tooling, benchmark harnesses, or framework abstractions with no direct execution path impact

### 3. Preserve semantics before cleanup

During the first migration pass:

- preserve behavior first
- preserve shape and dtype rules first
- preserve dataflow first

Do not mix the migration with optional cleanups such as naming polish, API reshaping, or algorithmic changes unless they are necessary for repository consistency or import safety.

## Phase 3: Implement the Migration

Apply changes from lowest level to highest level.

### A. Native code and dependency boundary

When migrating native code:

1. Move only the required native closure into cache-dit's `csrc` tree.
2. Rename namespaces and top-level identifiers where needed to match cache-dit ownership.
3. Keep dispatch structure if it is functionally necessary; do not rewrite it just because it looks unfamiliar.
4. Decide dependency strategy explicitly:
   - vendored in-tree
   - git submodule
   - preinstalled system dependency
5. Add build-time validation for missing required dependencies.

### B. Private CUDA helper layer

Use a private helper module under `src/cache_dit/kernels/cuda/` for extension loading and low-level bridging.

Typical responsibilities:

- delayed import of the optional extension
- returning a cached load error
- wrapping direct calls into the extension's `ops` and `utils` submodules
- keeping internal details out of the public operator API

If the extension is optional, `import cache_dit` must remain safe.

### C. Registry layer

Put low-level `torch.library` definitions and implementations in the CUDA registry layer, for example:

- `src/cache_dit/kernels/cuda/_ops_registery.py`

Typical responsibilities:

- define `torch.library` schemas
- implement real CUDA behavior
- add fake registrations where compile or tracing paths need them
- keep the public kernel API separate from raw registration details

Registration and fake-implementation conventions:

- name fake registrations explicitly as `_fake_<operator_name>`; do not use anonymous `def _(...)` helpers
- apply this naming rule consistently across CUDA, Triton, CuTe DSL, and other operator backends in cache-dit
- when adding or migrating operators, add unit tests in the same change
- tests should cover at least one fake shape or dtype path and one runtime correctness or smoke path

### D. Public kernel API layer

Expose user-facing wrappers from `src/cache_dit/kernels/ops.py`.

Default conventions:

- expose explicit functions instead of `partial(...)` aliases when signature discoverability matters
- keep public names repository-aligned
- hide internal backend-selection knobs unless users truly need them
- validate backend support centrally instead of scattering checks

### E. Higher-level modules and state adaptation

If the migration also adds a module abstraction such as a quantized `nn.Module`:

1. keep the module's expected state keys stable
2. adapt upstream raw export keys into cache-dit module keys explicitly
3. do not leak source-repo naming into the public API if cache-dit already has a better convention

## Phase 4: Validate in Layers, Kernels, and Modules

Do not rely on one test.

Validation should usually proceed in this order:

1. **Import safety**
   - importing cache-dit without the optional extension should not crash
2. **Low-level smoke**
   - low-level op runs with expected dtype, device, and shape
3. **Low-level correctness**
   - compare operator output against a dense or reference implementation
4. **Module correctness**
   - verify the higher-level module uses the migrated operator path correctly
5. **Round-trip or end-to-end validation**
   - if serialization, quantization, or pipeline integration exists, test that explicitly
6. **Boundary tests**
   - unsupported geometry, rank, alignment, or build conditions should fail clearly

When scope is intentionally limited, say so explicitly.

Example:

- "INT4 CUDA path is the validation gate."
- "FP4 code is retained but not currently gated by runtime correctness tests."

Do not imply feature maturity beyond what the tests actually cover.

## Phase 5: Packaging and Documentation

Operator migration is incomplete if build and packaging are wrong.

Checklist:

- update `setup.py` for optional extension build gates
- update `pyproject.toml` if packaging metadata or dependencies changed
- enforce submodule or dependency checks where needed
- keep default install/import behavior safe without the optional extension
- document only what is actually usable now

Do not advertise unfinished features in README or user docs ahead of validated capability.

## Anti-Patterns

Avoid these failure modes.

### Do not mechanically mirror upstream layout

Bad:

- copying an entire source repo subtree into `csrc/` because one operator needed two files from it

Better:

- identify the minimum closure and migrate only that set

### Do not expose internal control knobs casually

Bad:

- exposing backend-selection or migration-only tuning arguments to end users because they were convenient during development

Better:

- hardcode them at the internal wrapper layer until a real product need exists

### Do not leak source-repo naming when cache-dit conventions already exist

Bad:

- keeping raw upstream helper names or state keys in the public interface without evaluating cache-dit consistency

Better:

- adapt them to the repository's public naming rules and keep the raw names private if needed

### Do not let optional extensions break base imports

Bad:

- importing the extension eagerly from top-level package import paths

Better:

- delay extension import until the migrated operator is actually needed

### Do not claim correctness from one smoke test

Bad:

- compiling the extension and declaring the migration complete

Better:

- prove import safety, low-level execution, low-level correctness, higher-level module behavior, and boundary failures

### Do not write machine-local reference paths

Bad:

- `/abs/path/to/workspace/...`

Better:

- `src/cache_dit/kernels/ops.py`
- `nunchaku/nunchaku/models/linear.py`
- `deepcompressor/deepcompressor/calib/smooth.py`

## Reference Case: SVDQ / Nunchaku Migration

Use this as an example of the workflow, not as a recipe to replay line by line.

### What the migration demonstrated

- native W4A4 closure can be migrated without importing an entire upstream project
- public operator wrappers should remain explicit and repository-aligned
- `torch.library` schemas and implementations belong in the CUDA registry layer
- optional extension import should be delayed and load errors should be queryable
- packaging may need submodule enforcement instead of hard-coded vendoring
- layered tests should cover runtime, module, and end-to-end behaviors separately

### Useful cache-dit reference files

- `src/cache_dit/kernels/ops.py`
- `src/cache_dit/kernels/cuda/_ops_registery.py`
- `src/cache_dit/kernels/cuda/_svdquant.py`
- `src/cache_dit/quantization/svdquant/linear.py`
- `tests/kernels/test_svdquant_runtime.py`
- `tests/quantization/test_svdquant_quantizer.py`
- `setup.py`

### Useful external reference cases

- `nunchaku/nunchaku/models/linear.py`
- `nunchaku/nunchaku/ops/gemm.py`
- `nunchaku/nunchaku/ops/quantize.py`
- `deepcompressor/deepcompressor/backend/nunchaku/utils.py`
- `deepcompressor/deepcompressor/calib/smooth.py`
- `deepcompressor/deepcompressor/calib/lowrank.py`

### What was specific to that case

These details were important for SVDQ, but are not universal migration rules:

- `svdq_*` naming convention
- W4A4 INT4 and FP4 split in scope and validation
- specific geometry and rank constraints
- quantized module state adaptation rules
- submodule choice for `spdlog`

If your operator family differs, keep the workflow but re-evaluate the decisions.

## Suggested Execution Order

When this skill is invoked for a real migration, follow this order:

1. summarize the target operator and current scope boundary
2. list source files and cache-dit integration points
3. identify the minimum viable closure
4. freeze the public API and naming strategy
5. migrate native code and private helper plumbing
6. add registry entries and public wrappers
7. add or adapt higher-level module code if needed
8. validate in layers or kernel, module, and end-to-end tests
9. document only validated scope

## Exit Criteria

The migration is not done until all of these are true:

- the public API is intentional and repository-aligned
- optional extension behavior is safe
- dependency strategy is explicit
- tests prove correctness at the right layers, kernel and module level as needed
- unsupported cases fail clearly
- documentation matches validated reality
