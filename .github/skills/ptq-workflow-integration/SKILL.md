---
name: ptq-workflow-integration
description: 'Use when integrating a new PTQ workflow into cache-dit; designing quantize/load API shape, backend-specific config validation, save/load manifests, benchmark and regression tests, or reviewing a PTQ integration plan. Uses the SVDQ PTQ integration only as a style and coverage reference. Do not copy the SVDQ implementation mechanically.'
argument-hint: 'Describe the PTQ algorithm or backend, target public API, calibration flow, serialization requirements, target models, and required validation layers.'
user-invocable: true
---

# PTQ Workflow Integration for cache-dit

## Goal

Integrate one PTQ workflow in a way that feels native to cache-dit:

- public API stays simple
- backend-specific logic stays localized
- save/load UX is predictable
- fast tests and slow tests cover different risks
- docs show only the public workflow

This skill is based on lessons from the SVDQ PTQ integration, but SVDQ PTQ is a reference only.

## Core Rule

Do not mechanically copy SVDQ PTQ files.

Use the SVDQ PTQ integration to learn:

- what belongs in public API vs private backend code
- where config validation should live
- where backend serialization and load logic should live
- how tests should be split by cost and purpose
- what documentation shape is acceptable for cache-dit

Do not reuse SVDQ PTQ by blind copy-paste.

Specifically do not copy without redesigning first:

- private helper structure
- class names or helper names
- logging layout
- exact file decomposition
- prompts, thresholds, or benchmark constants
- test bodies that only happen to fit SVDQ

Treat SVDQ as a style reference, not a template to replay.

## When to Use

Use this skill when you need to:

- add a new PTQ backend or algorithm into cache-dit
- extend an existing PTQ backend with save/load support
- decide where PTQ files and tests should live
- review whether a PTQ integration follows cache-dit API style
- plan coverage for a PTQ feature before coding

Do not use this skill for:

- generic quantization work that does not involve PTQ workflow design
- blind upstream porting
- one-off benchmark scripts with no repository integration

## Reference Style Rule

Use repo-relative references only.

- For cache-dit files, use paths like `src/cache_dit/quantization/config.py`.
- For docs, use paths like `docs/user_guide/QUANTIZATION.md`.
- For tests, use paths like `tests/quantization/test_svdquant_ptq.py`.
- Do not write machine-local absolute paths into the skill.

## Design Principles to Keep

### 1. Public API symmetry matters

Prefer a user-facing flow like:

- `cache_dit.quantize(...)`
- `cache_dit.load(...)`
- `QuantizeConfig(...)`

If save/load is part of the workflow, keep `quantize` and `load` at the same API layer unless there is a very strong reason not to.

### 2. Keep backend-specific knobs grouped and validated

Prefer validated backend-specific kwargs or a clearly scoped backend config section over many new top-level config fields.

The pattern to follow is:

- generic config contract in `src/cache_dit/quantization/config.py`
- backend-specific validation still triggered from that shared config layer
- backend math and orchestration remain under the backend package

### 3. Hide internal PTQ machinery

Private PTQ context objects, calibrators, observers, or loaders should stay internal unless users truly need them.

Tests and docs should generally use public APIs only.

### 4. Save/load should be ergonomic and deterministic

If the PTQ workflow serializes checkpoints:

- normalize output to a deterministic file name
- keep a machine-readable manifest next to the checkpoint when directory loading is supported
- resolve config, file path, and directory path through one internal load path resolver
- validate metadata before mutating the target module

### 5. Slow validation must be opt-in

Fast regression coverage should run by default when feasible.

Pipeline-scale validation, compile validation, and large-model comparisons should be environment-gated.

## File Placement Guidelines

Only add new files when they correspond to a real boundary.

### Usually edit existing shared files for

- generic config schema: `src/cache_dit/quantization/config.py`
- public quantize/load routing: `src/cache_dit/quantization/dispatch.py`
- package exports if public API changes: `src/cache_dit/__init__.py`
- optional quantization package exports: `src/cache_dit/quantization/__init__.py`
- user docs: `docs/user_guide/QUANTIZATION.md`

### Usually add backend-local files for

- PTQ orchestration and serialization: `src/cache_dit/quantization/<backend>/ptq.py`
- backend math or accumulation logic: `src/cache_dit/quantization/<backend>/quantizer.py`
- backend module wrappers or load helpers: `src/cache_dit/quantization/<backend>/...`

### Usually add tests in

- backend public-workflow tests: `tests/kernels/test_<backend>_ptq.py`
- backend math or lower-level tests: `tests/kernels/test_<backend>_quantizer.py`

Add a separate shared test utility file only if multiple test files genuinely reuse the same helpers.

## What the SVDQ PTQ Integration Teaches About API Design

Use these as design lessons, not copy targets.

### Shared config layer

`src/cache_dit/quantization/config.py` is the right place for:

- quant type parsing and normalization
- backend auto-resolution
- validation of PTQ-specific unsupported combinations
- normalization of `serialize_to`
- validation of backend-specific kwargs

Do not push these checks into only the backend implementation file if the public config object can reject them earlier.

### Backend PTQ implementation layer

`src/cache_dit/quantization/svdquant/ptq.py` demonstrates the right kind of responsibilities for a backend PTQ file:

- run calibration through the public config callback
- quantize target modules
- serialize checkpoint artifacts
- write a lightweight JSON manifest for directory load UX
- resolve load inputs and validate metadata
- attach runtime metadata back onto the loaded module

This is the right layer for backend save/load orchestration.

### Public docs and tests layer

`docs/user_guide/QUANTIZATION.md` and `tests/quantization/test_svdquant_ptq.py` demonstrate the preferred user story:

- user interacts with `QuantizeConfig`
- user calls `cache_dit.quantize`
- user calls `cache_dit.load`
- user does not need private PTQ classes

Keep new PTQ integrations aligned with that style unless the backend genuinely requires a different experience.

## Serialization and Load Checklist

If the PTQ workflow needs saved checkpoints, check all of the following:

1. The serialized checkpoint file name is deterministic.
2. A colocated manifest exists when directory loading is supported.
3. The load path accepts ergonomic inputs only when they can be resolved unambiguously.
4. Metadata validation happens before module mutation.
5. Quant type mismatches fail clearly.
6. Missing manifest or malformed metadata fail clearly.
7. Round-trip tests prove loaded output matches quantized output.

## Test Strategy

Do not ship a PTQ integration with only one slow end-to-end test.

### Fast tests should cover

- public API quantization replaces the expected layers
- save/load roundtrip restores the quantized module
- config validation rejects unsupported combinations
- directory load, checkpoint load, and config-driven load if all are supported
- invalid metadata and incomplete checkpoint failure cases
- exclusion or filtering behavior
- backend-specific calibration or buffering behavior when applicable

### Slow tests should cover

- at least one real pipeline or model integration path
- serialization plus reload closure
- one primary quality gate
- additional metrics reported separately from the hard gate
- latency, transformer memory, or peak memory when those are part of the PTQ value proposition

### Optional compile tests should cover

- loading an already quantized module
- enabling compile configs if the repo expects them
- `torch.compile(...)`
- one warmup run
- one actual inference run

Compile validation should be behind a separate environment variable, not mixed into the default slow test path.

## Test Style Rules to Preserve

Follow these rules unless there is a strong reason to violate them:

- integration tests should use public APIs, not private PTQ classes
- slow tests should self-skip with clear environment-variable guidance
- deterministic generators or seeds should be used for pipeline tests
- large test artifacts should go under repo-local `.tmp/tests/...`
- benchmark tables and visuals are reports, not pass/fail criteria unless explicitly required
- hard accuracy gates should stay minimal and explainable

## Suggested Coverage Map

When integrating a new PTQ workflow, think in layers:

1. config/schema layer
2. backend quantizer/math layer
3. serialization/load layer
4. public API layer
5. model or pipeline integration layer
6. optional compile layer

If one of these layers is intentionally out of scope, say so explicitly in the PR or plan.

## Recommended Implementation Order

1. Survey existing public quantization API and decide whether the new PTQ flow fits it.
2. Add or update shared config validation in `src/cache_dit/quantization/config.py`.
3. Implement backend PTQ orchestration in `src/cache_dit/quantization/<backend>/ptq.py`.
4. Add backend helper logic only where a real separation exists.
5. Wire dispatch and exports only after backend behavior is stable.
6. Add fast tests for public API, roundtrip, validation, and failure cases.
7. Add env-gated slow tests for real model integration.
8. Add optional compile tests only if compile compatibility matters.
9. Update `docs/user_guide/QUANTIZATION.md` with public API examples only.

## Review Questions

Before merging a PTQ integration, ask:

1. Does the user-facing flow still look like cache-dit?
2. Are backend-specific options validated centrally?
3. Are save/load artifacts deterministic and discoverable?
4. Can the feature be tested quickly without a giant model?
5. Are slow tests opt-in and clearly scoped?
6. Do docs and tests avoid private PTQ symbols?
7. Does this integration borrow SVDQ lessons without copying SVDQ internals?

## Common Mistakes

- exposing private PTQ context classes in docs or integration tests
- adding too many top-level config fields instead of validated backend-specific kwargs
- implementing save/load UX only through one exact file path
- skipping malformed metadata tests
- writing only slow model tests and no fast public-API tests
- making compile validation part of the default slow path
- hardcoding local machine paths into docs or instructions
- copying SVDQ helper structure line-for-line

## Reference Touchpoints

These are the main SVDQ PTQ reference files for style and coverage shape:

- `src/cache_dit/quantization/config.py`
- `src/cache_dit/quantization/svdquant/ptq.py`
- `tests/quantization/test_svdquant_ptq.py`
- `tests/quantization/test_svdquant_quantizer.py`
- `docs/user_guide/QUANTIZATION.md`

Use them to understand cache-dit conventions.

Do not reproduce them mechanically.
