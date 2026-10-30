---
name: cache-dit-operator-migration
description: 'Use when doing operator migration or kernel migration for cache-dit; porting kernels from nunchaku, deepcompressor, or other repos; designing CUDA operator registration and public wrappers; wiring build and packaging for optional CUDA extensions; or reviewing an operator migration plan. cache-dit maintains CUDA operators only, so this skill intentionally excludes Triton and CuTe DSL work. Do not use for blind copy-paste ports.'
argument-hint: 'Describe the operator family, source repo, target public API, required dtypes or architectures, and current migration status.'
user-invocable: true
---

# Operator Migration for cache-dit

## Goal

Migrate one operator or kernel family into cache-dit in a way that is:

- semantically correct
- aligned with cache-dit repository conventions
- safe to import when optional native extensions are absent
- validated at multiple layers instead of by one smoke test

This skill is for migration work that touches native CUDA code, Python wrappers, operator registration, build packaging, or quantized module integration.

## Backend Policy: CUDA Only

This is a hard, long-term constraint, not a per-task preference.

cache-dit maintains **CUDA operators exclusively**. Every operator that lives long term in the
repository is a CUDA kernel under `csrc/kernels/`, reachable through a `torch.library` schema
in the `cache_dit_cuda_ops` namespace.

Therefore:

1. Do not add a Triton or CuTe DSL operator to cache-dit. Not as a fast path, not as a portable
   fallback, not "just for A100", not temporarily.
2. Do not recreate `src/cache_dit/kernels/triton/`, `src/cache_dit/kernels/cutedsl/`, or any
   per-backend subpackage. Those trees were removed and are covered by regression guards, see
   `tests/kernels/test_ops_dispatch.py`.
3. Do not reintroduce a backend selector. `kernels/backend.py`, `kernels/ops.py`, the
   `KernelBackend` enum, and the `_resolve_backend` / `_select_*_backend` dispatch helpers are gone;
   `test_ops_dispatch.py` asserts they stay gone. A single CUDA path with a clear error message
   replaces the dispatch table.
4. If a source repository only has a Triton implementation, read it for the algorithm and rewrite
   the kernel in CUDA. Treat Triton as pseudocode, never as a target language.
5. If someone asks for Triton or CuTe DSL support, say the maintenance policy excludes it, and
   offer the CUDA route instead.

The reason is maintenance cost: three parallel backends with three registration layers, three
fake-registration paths, and three build stories cost more than the wins they deliver, and the
CUDA path is the only one that ships in the release wheel.

Use `triton-kernel` or `cute-dsl-kernel` for standalone kernel work outside cache-dit; do not let
those workflows leak into this repository.

## When to Use

Use this skill when you need to:

- migrate a CUDA operator from another repo into cache-dit
- port a nunchaku operator or kernel family into cache-dit
- add a new CUDA operator domain (a new native directory plus its Python module)
- decide what native files are actually required for a migration
- design cache-dit public wrappers for a newly migrated operator
- register low-level ops through `torch.library` in the `cache_dit_cuda_ops` namespace
- add optional-extension build logic, submodule checks, or packaging guards
- design layered validation for a migrated operator
- review whether an operator migration plan is thoughtful or mechanical

Do not use this skill for:

- generic model integration with no operator or kernel work
- pure Python feature work unrelated to kernels or extensions
- Triton or CuTe DSL operator work, which this repository does not maintain
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

- For cache-dit files, use repo-relative paths such as `src/cache_dit/kernels/_comm.py` or `tests/kernels/test_cuda_fp8_comm_ops.py`.
- For sibling or external repos, use repository-relative or GitHub-searchable paths such as `nunchaku/nunchaku/models/linear.py` or `deepcompressor/deepcompressor/backend/nunchaku/utils.py`.
- Do not write machine-local absolute paths such as `/abs/path/to/workspace/...` into the skill, its supporting documentation, or any error message you add to cache-dit.

## Current Layout

The repository is CUDA-only after the operator refactor, so there is exactly one shape to follow.

### Native tree

```
csrc/kernels/
  attn/attn.{h,cpp}             # pybind TU for cache_dit._C_attn
  attn/merge_states.cu          # fused attention-state merge
  comm/comm_ops.{h,cpp}         # pybind TU for cache_dit._C_comm
  comm/fp8_comm.cu              # FP8 per-token / QKV-permute comm kernels
  svdq/                         # W4A4 GEMM family, split into core/ int4/ fp4/
    pybind.cpp torch.{h,cpp} ops.h zgemm.h dispatch_utils.h Tensor.h common.h
csrc/third_party/spdlog/        # git submodule, SVDQuant logging only
```

Two naming facts that trip people up:

- A native directory is **not** an extension module, but the operator family decides which extension
  hosts its files: `attn/` is compiled into `cache_dit._C_attn`, `comm/` into `cache_dit._C_comm`.
  A kernel never gets registered in another family's pybind TU.
- C++ namespaces follow the extension, not the directory: attention-state merge lives in
  `namespace cache_dit::attn` and talks to Python through `csrc/kernels/attn/attn.cpp`.

### Python tree

```
src/cache_dit/kernels/
  __init__.py        # flat public surface + __all__, re-exported from domain modules
  _attn.py           # attention-state merge: loader for cache_dit._C_attn + registration
  _comm.py           # FP8 comm ops: loader for cache_dit._C_comm + registration
  _svdquant.py       # SVDQ W4A4 ops, loader for cache_dit._C_svdquant (+ utils submodule)
```

There is no `ops.py`, no `cuda/`, no `_ops_registery.py`, no `backend.py`, and no per-backend
package. Each domain module owns its own registrations and its own extension loader.

### Extensions and build flags

| Extension | Built by | Contents |
|---|---|---|
| `cache_dit._C_svdquant` | `CACHE_DIT_BUILD_SVDQUANT=1` | SVDQ W4A4 GEMM + quantizers; exposes `ops` and `utils` submodules |
| `cache_dit._C_comm` | `CACHE_DIT_BUILD_COMM=1`, or any `CACHE_DIT_BUILD_SVDQUANT=1` build | FP8 comm ops; exposes an `ops` submodule |
| `cache_dit._C_attn` | `CACHE_DIT_BUILD_ATTN=1`, `CACHE_DIT_BUILD_COMM=1`, or any `CACHE_DIT_BUILD_SVDQUANT=1` build | Attention-state merge; exposes an `ops` submodule |

`CACHE_DIT_BUILD_SVDQUANT=1` enables all three extensions, because the release wheel must ship them.
The optional operator extensions (`_C_comm`, `_C_attn`) are gated to SM89+; the SVDQuant extension
filters out sm90 targets its INT4 kernels cannot use. Default arch targets are
`("89", "90a", "100", "120a")`, override with `CACHE_DIT_CUDA_ARCH_LIST` or `TORCH_CUDA_ARCH_LIST`.
An explicit `CACHE_DIT_BUILD_ATTN=1` raises when no SM target resolves, so a requested extension is
never dropped silently.

## Phase 0: Gather Before Coding

Collect the migration inputs first.

### Required inputs

1. Source operator and source repo
   Example: `nunchaku/nunchaku/ops/gemm.py` plus the native files it depends on.
2. Target cache-dit user-facing surface
   Example: a low-level op wrapper, a quantized module, or both.
3. Required dtypes, architectures, and scope boundaries
   Example: "INT4 CUDA is required now; FP4 implementation may be retained but not gate current validation."
4. Build and packaging requirements
   Example: which extension hosts the op, which flag enables it, whether a submodule is needed.
5. Validation target
   Example: import safety, low-level parity, module parity, end-to-end inference, or shape rejection.

### Gather checklist

- Identify the source operator entry points.
- Identify the native files that implement them.
- Identify helper files that are truly required by those implementations.
- Identify which existing cache-dit domain module should host the migrated behavior.
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
- whether the source is CUDA already; if not, note that the kernel must be rewritten

### Survey cache-dit integration points

Anchor files:

- `src/cache_dit/kernels/__init__.py` - the public surface
- `src/cache_dit/kernels/_<domain>.py` - the module that will host the op
- `csrc/kernels/<domain>/` - the native side
- `setup.py` - `_required_<ext>_sources()`, `_missing_<ext>_sources()`, `_get_<ext>_extension()`
- `tests/kernels/` - operator tests
- `tools/inspect_release_wheel.py` - release wheel content gates

Ask these questions while surveying:

1. Which existing domain module is the right host, or is a new domain justified?
2. Does the op belong in `cache_dit._C_comm`, `cache_dit._C_attn`, or a new/dedicated extension module?
3. How should extension loading fail when the op is unavailable?
4. Is there already a naming convention in `cache_dit_cuda_ops` for this operator family?
5. Does `test_ops_dispatch.py` already list the public names this migration will extend?

## Phase 2: Decide the Migration Shape

Make the design decisions before editing files.

### 1. Freeze the public surface first

Define the cache-dit-facing API early.

Examples of questions to settle:

- Which operator names should be exposed publicly?
- Should the public API be low-level only, module-level only, or both?
- Should internal backend toggles be hidden from users?
- Should wrapper functions be explicit rather than `partial(...)` so editors and type tools can see the real signature?

Default rule: keep tuning and launch details private unless there is a strong user-facing reason to expose them.

### 2. Migrate the minimal viable closure

Do not import an entire subsystem if only one slice is needed.

Usually migrate:

- the kernel implementation files that are actually on the call path
- the minimum helper headers or Python utilities they require
- the binding TU entry, the loader, the registration, and the wrappers

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

1. Move only the required native closure into `csrc/kernels/<domain>/`.
2. Put new code in `namespace cache_dit::<domain>` and keep upstream helper names internal with an
   anonymous namespace. The migrated SVDQ tree keeps its upstream `svdq::` namespace because it is
   upstream-derived; do not rename it as part of unrelated work.
3. Keep dispatch structure if it is functionally necessary (SVDQ needs its inheritance-based
   launch dispatch); do not rewrite it just because it looks unfamiliar.
4. Decide dependency strategy explicitly:
   - vendored in-tree
   - git submodule (`csrc/third_party/spdlog` is the only one today)
   - preinstalled system dependency
5. Add or extend the binding TU. `csrc/kernels/attn/attn.cpp`, `csrc/kernels/comm/comm_ops.cpp` and
   `csrc/kernels/svdq/pybind.cpp` are the only `PYBIND11_MODULE` entry points; new ops in that
   extension go through the existing entry point, not a new one. A new operator family gets its own
   `<domain>/{<domain>.h,<domain>.cpp}` pair instead of borrowing another family's TU.

### B. Domain Python module

Put everything for one domain in a single private module, `src/cache_dit/kernels/_<domain>.py`.
The sections and their order are consistent across `_attn.py`, `_comm.py`, and `_svdquant.py`, and
a migration should follow the same shape:

```python
_EXTENSION_MODULE_NAME = "cache_dit._C_<ext>"
_cached_extension_module: ModuleType | None = None
_cached_load_error: Exception | None = None


def _load_extension_module() -> ModuleType | None:
  # import once, cache both the module and the failure


def <domain>_extension_is_available() -> bool: ...
def <domain>_get_load_error() -> Exception | None: ...


def _get_required_ops_module() -> ModuleType:
  # raise RuntimeError with the exact build command when unavailable


def _call_<op>(...) -> None:
  # thin bridge: shape/pack differences only, no policy


torch.library.define("cache_dit_cuda_ops::<op>", "(Tensor x) -> Tensor")


@torch.library.impl("cache_dit_cuda_ops::<op>", "CUDA")
def _<op>(x: torch.Tensor) -> torch.Tensor: ...


@torch.library.register_fake("cache_dit_cuda_ops::<op>")
def _fake_<op>(x: torch.Tensor) -> torch.Tensor: ...


def <op>(x: torch.Tensor) -> torch.Tensor:
  """Public wrapper, Sphinx docstring, calls torch.ops.cache_dit_cuda_ops.<op>."""
```

Rules that the existing modules enforce:

- the `torch.library` namespace is always `cache_dit_cuda_ops`
- implementations register on the `"CUDA"` dispatch key
- fake registrations are named `_fake_<operator_name>`; no anonymous `def _(...)`
- allocation and shape math live in the `impl`, the `_call_` bridge only adapts the ABI
- dtype and shape validation raises `ValueError` with the actual value, before allocating
- the public wrapper calls `torch.ops.cache_dit_cuda_ops.<op>`, never the bridge directly, so
  `torch.compile` and fake tensors see one path
- module `__all__` lists public wrappers; private helpers use a leading underscore and are listed
  only when tests or sibling modules need them
- the module holds no mutable global policy state; its only module-level state is the loader cache
  and static lookup tables such as dtype-id maps

Keeping the loader in the same module as its ops means a domain's failure mode, registration, and
public API cannot drift apart. Each domain loads its own extension (`_attn.py` → `cache_dit._C_attn`,
`_comm.py` → `cache_dit._C_comm`, `_svdquant.py` → `cache_dit._C_svdquant`); borrow a sibling loader
only when the ops really ship in that sibling's extension.

### C. Public surface

Expose the operator from `src/cache_dit/kernels/__init__.py` with an explicit re-export and an
`__all__` entry:

```python
from ._comm import fp8_comm_per_token_quant
```

Default conventions:

- one flat namespace, mirroring the domain modules; no subpackage re-export layers
- explicit functions rather than `partial(...)` aliases when signature discoverability matters
- `__all__` must equal the non-underscore public names, which
  `test_ops_dispatch.py::test_public_api_matches_all` enforces
- each name must be the same object as the one in its domain module, which
  `test_public_api_is_reexported_from_domain_module` enforces

Do not create a `kernels/ops.py` or a registry module to "centralize" this. The centralization was
removed on purpose; adding it back is the exact regression the policy forbids.

### D. Higher-level modules and state adaptation

If the migration also adds a module abstraction such as a quantized `nn.Module`:

1. keep the module's expected state keys stable
2. adapt upstream raw export keys into cache-dit module keys explicitly
3. do not leak source-repo naming into the public API if cache-dit already has a better convention
4. route the module through the public wrapper, not through `_call_*` or `torch.ops` directly

## Phase 4: Validate in Layers

Do not rely on one test.

Validation should usually proceed in this order:

1. **Import safety**
   - importing cache-dit without the optional extension should not crash
2. **API surface**
   - public names re-exported identically, `__all__` consistent
3. **Low-level smoke**
   - low-level op runs with expected dtype, device, and shape
4. **Low-level correctness**
   - compare operator output against a dense or reference implementation
5. **Module correctness**
   - verify the higher-level module uses the migrated operator path correctly
6. **Fake / compile path**
   - at least one `register_fake` shape-and-dtype assertion, since every op ships a fake
7. **Round-trip or end-to-end validation**
   - if serialization, quantization, or pipeline integration exists, test that explicitly
8. **Boundary tests**
   - unsupported geometry, rank, alignment, or build conditions should fail clearly

Matching test files:

- `tests/kernels/test_ops_dispatch.py` - re-export identity, `__all__`, and the removed-dispatch guards
- `tests/kernels/test_cuda_fp8_comm_ops.py` - FP8 comm correctness and fake paths
- `tests/kernels/test_cuda_merge_attn_states.py` - attention-state merge correctness
- `tests/kernels/test_svdquant_runtime.py` - SVDQ runtime behavior
- `tests/quantization/` - module-level and quantizer behavior

When scope is intentionally limited, say so explicitly.

Example:

- "INT4 CUDA path is the validation gate."
- "FP4 code is retained but not currently gated by runtime correctness tests."

Do not imply feature maturity beyond what the tests actually cover.

## Phase 5: Packaging and Documentation

Operator migration is incomplete if build and packaging are wrong. All of this happens in
`setup.py`.

Checklist:

1. Add new native files to `_required_svdquant_sources()`, `_required_comm_sources()`, or
   `_required_attn_sources()`.
   `_missing_*_sources()` turns a forgotten entry into a clear build error instead of a link error,
   and a source that is on disk but not in the list is silently not compiled.
2. If the op needs a new extension, add `_get_<ext>_extension()` and wire it into
   `_get_extensions()`; keep the flag-gated, SM-target-gated structure.
3. Keep default install and import behavior safe without any extension: no eager extension import
   outside the lazy loader.
4. Keep `CACHE_DIT_BUILD_SVDQUANT=1` shipping every extension; a cu13 release build asserts
   this. `BuildExtension` only compares source mtimes, so after editing a shared header such as
   `csrc/kernels/attn/attn.h`, wipe `build/` (`CACHE_DIT_CLEAN=1`) instead of trusting the cached
   objects.
5. Update `tools/inspect_release_wheel.py` when a wheel-content expectation changes, e.g. an
   additional `.so` that must be present in the SVDQuant wheel.
6. Update `docs/user_guide/INSTALL.md` for any new flag or host requirement.
7. Document only what is actually usable now.

Do not advertise unfinished features in README or user docs ahead of validated capability.

## Anti-Patterns

Avoid these failure modes.

### Do not add a Triton or CuTe DSL operator

Bad:

- porting a Triton kernel as-is, or reinstating `kernels/triton/` "for portability"

Better:

- rewrite the algorithm as a CUDA kernel under `csrc/kernels/<domain>/`

### Do not reintroduce a backend dispatch layer

Bad:

- adding `kernels/backend.py`, a `KernelBackend` enum, or `_resolve_backend` to pick between
  implementations

Better:

- one CUDA path, one `torch.library` schema, and a `RuntimeError` naming the build command when
  the extension is missing

### Do not mechanically mirror upstream layout

Bad:

- copying an entire source repo subtree into `csrc/` because one operator needed two files from it

Better:

- identify the minimum closure and migrate only that set

### Do not scatter one domain across files

Bad:

- registering the op in one module, loading the extension in another, and wrapping in a third

Better:

- one `_<domain>.py` holding loader, bridge, registration, fake, and wrapper

### Do not expose internal control knobs casually

Bad:

- exposing stage counts, launch tuning, or migration-only switches to end users because they were convenient during development

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

- prove import safety, API surface, low-level execution, low-level correctness, higher-level module behavior, and boundary failures

### Do not write machine-local reference paths

Bad:

- `/abs/path/to/workspace/...` in a docstring, an error message, or a skill reference

Better:

- `src/cache_dit/kernels/_comm.py`
- `nunchaku/nunchaku/models/linear.py`
- `deepcompressor/deepcompressor/calib/smooth.py`

## Reference Case: SVDQ / Nunchaku Migration

Use this as an example of the workflow, not as a recipe to replay line by line.

### What the migration demonstrated

- native W4A4 closure can be migrated without importing an entire upstream project
  (`csrc/kernels/svdq/` was later reorganized into `core/`, `int4/`, `fp4/`, which is a local
  layout choice: upstream nunchaku keeps a flat `csrc/`)
- public operator wrappers should remain explicit and repository-aligned
- `torch.library` schemas and implementations live beside their domain wrappers, in the
  `cache_dit_cuda_ops` namespace
- optional extension import should be delayed and load errors should be queryable
- packaging may need submodule enforcement instead of hard-coded vendoring
- layered tests should cover runtime, module, and end-to-end behaviors separately

### What the later refactor changed

The comm and attention-state-merge operators were CUDA ports of existing Triton and CuTe DSL
kernels. That migration is the template for future work:

- native sources landed in `csrc/kernels/comm/` and `csrc/kernels/attn/`
- each family registers through its own pybind TU: `cache_dit._C_comm` for the FP8 comm ops,
  `cache_dit._C_attn` for the attention-state merge (they shared `_C_comm` until the split)
- their Python modules colocate registration with the wrappers, replacing the old
  `cuda/_ops_registery.py` + `ops.py` pair
- the Triton and CuTe DSL trees, the backend selector, and their tests were deleted, and
  `tests/kernels/test_ops_dispatch.py` now guards against their return

### Useful cache-dit reference files

- `src/cache_dit/kernels/__init__.py`
- `src/cache_dit/kernels/_attn.py`
- `src/cache_dit/kernels/_comm.py`
- `src/cache_dit/kernels/_svdquant.py`
- `src/cache_dit/quantization/svdquant/linear.py`
- `csrc/kernels/attn/attn.cpp`
- `csrc/kernels/attn/attn.h`
- `csrc/kernels/attn/merge_states.cu`
- `csrc/kernels/comm/comm_ops.cpp`
- `csrc/kernels/comm/comm_ops.h`
- `csrc/kernels/svdq/pybind.cpp`
- `tests/kernels/test_ops_dispatch.py`
- `tests/kernels/test_cuda_fp8_comm_ops.py`
- `tests/kernels/test_cuda_merge_attn_states.py`
- `tests/kernels/test_svdquant_runtime.py`
- `tests/quantization/test_svdquant_quantizer.py`
- `setup.py`
- `tools/inspect_release_wheel.py`

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

1. confirm the target is CUDA; if the source is Triton or CuTe DSL, plan a rewrite
2. summarize the target operator and current scope boundary
3. list source files and the cache-dit host module
4. identify the minimum viable closure
5. freeze the public API and naming strategy
6. migrate native code and extend the pybind TU
7. add loader, bridge, registration, fake, and wrapper in one domain module
8. re-export from `kernels/__init__.py`
9. wire `setup.py` source lists and build gates
10. add or adapt higher-level module code if needed
11. validate in layers for kernel, module, and end-to-end behavior
12. document only validated scope

## Exit Criteria

The migration is not done until all of these are true:

- the operator is CUDA, with no Triton or CuTe DSL component anywhere in the change
- the public API is intentional and repository-aligned
- optional extension behavior is safe
- dependency strategy is explicit
- every new native source appears in the `setup.py` source list
- tests prove correctness at the right layers, kernel and module level as needed
- unsupported cases fail clearly
- documentation matches validated reality
