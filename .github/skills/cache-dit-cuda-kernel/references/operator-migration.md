# Operator Migration for cache-dit

> 本文原为独立 skill `cache-dit-operator-migration`，2026-11-05 合并进 `cache-dit-cuda-kernel`。
> 读它之前先读 `kernel-integration-chain.md`（本仓的层级契约、模板与构建触点）；本文只补"从别的仓库搬算子"这一层的判断与流程。

## When to use this document

Use it when you need to:

- migrate a CUDA operator from another repo into cache-dit
- port a nunchaku operator or kernel family into cache-dit
- add a new CUDA operator domain (a new native directory plus its Python module)
- decide what native files are actually required for a migration
- design cache-dit public wrappers for a newly migrated operator
- review whether an operator migration plan is thoughtful or mechanical

Do not use it for:

- generic model integration with no operator or kernel work
- pure Python feature work unrelated to kernels or extensions
- Triton or CuTe DSL operator work, which this repository does not maintain
- blind "copy upstream into csrc" execution

## Goal

Migrate one operator or kernel family into cache-dit in a way that is:

- semantically correct
- aligned with cache-dit repository conventions
- safe to import when optional native extensions are absent
- validated at multiple layers instead of by one smoke test

## 1. Backend policy: CUDA only

This is a hard, long-term constraint, not a per-task preference.

cache-dit maintains **CUDA operators exclusively**. Every operator that lives long term in the
repository is a CUDA kernel under `csrc/kernels/`, reachable through a `torch.library` schema in the
`cache_dit_cuda_ops` namespace.

Therefore:

1. Do not add a Triton or CuTe DSL operator to cache-dit. Not as a fast path, not as a portable
   fallback, not "just for A100", not temporarily.
2. Do not recreate `src/cache_dit/kernels/triton/`, `src/cache_dit/kernels/cutedsl/`, or any
   per-backend subpackage. Those trees were removed and are covered by regression guards, see
   `tests/kernels/test_ops_dispatch.py`.
3. Do not reintroduce a backend selector. `kernels/backend.py`, `kernels/ops.py`, the
   `KernelBackend` enum, and the `_resolve_backend` / `_select_*_backend` dispatch helpers are
   gone; `test_ops_dispatch.py` asserts they stay gone. A single CUDA path with a clear error
   message replaces the dispatch table.
4. If a source repository only has a Triton implementation, read it for the algorithm and rewrite
   the kernel in CUDA. Treat Triton as pseudocode, never as a target language.
5. If someone asks for Triton or CuTe DSL support, say the maintenance policy excludes it, and
   offer the CUDA route instead.

The reason is maintenance cost: three parallel backends with three registration layers, three
fake-registration paths, and three build stories cost more than the wins they deliver, and the CUDA
path is the only one that ships in the release wheel.

Use `triton-kernel` or `cute-dsl-kernel` for standalone kernel work outside cache-dit; do not let
those workflows leak into this repository.

## 2. Core rule

Do not mechanically replay upstream structure. Treat the source repository as the reference for
semantics, not as the required layout.

Before writing code, answer these questions:

1. What behavior is essential to preserve?
2. What is the smallest native and Python closure needed to preserve that behavior?
3. Which names should remain source-compatible, and which should be renamed to match cache-dit
   conventions?
4. What must be public, and what should remain private implementation detail?
5. Which tests prove the migration works, instead of merely compiling?

If those questions are not answered yet, do not start copying files.

### Reference style rule

Use portable references only:

- For cache-dit files, use repo-relative paths such as `src/cache_dit/kernels/_comm.py` or
  `tests/kernels/test_cuda_fp8_comm_ops.py`.
- For sibling or external repos, use repository-relative or GitHub-searchable paths such as
  `nunchaku/nunchaku/models/linear.py` or `deepcompressor/deepcompressor/backend/nunchaku/utils.py`.
- Do not write machine-local absolute paths such as `/abs/path/to/workspace/...` into the skill,
  its supporting documentation, or any error message you add to cache-dit.

## 3. Phase 0: gather before coding

### Required inputs

1. Source operator and source repo. Example: `nunchaku/nunchaku/ops/gemm.py` plus the native files
   it depends on.
2. Target cache-dit user-facing surface. Example: a low-level op wrapper, a quantized module, or
   both.
3. Required dtypes, architectures, and scope boundaries. Example: "INT4 CUDA is required now; FP4
   implementation may be retained but not gate current validation."
4. Build and packaging requirements. Example: which extension hosts the op, which flag enables it,
   whether a submodule is needed.
5. Validation target. Example: import safety, low-level parity, module parity, end-to-end
   inference, or shape rejection.

### Gather checklist

- Identify the source operator entry points.
- Identify the native files that implement them.
- Identify helper files that are truly required by those implementations.
- Identify which existing cache-dit domain module should host the migrated behavior.
- Identify the minimum feature slice that must work first.
- Identify what will explicitly not be validated in the current milestone.

## 4. Phase 1: survey the existing design

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

- `src/cache_dit/kernels/__init__.py` — the public surface
- `src/cache_dit/kernels/_<domain>.py` — the module that will host the op
- `csrc/kernels/<domain>/` — the native side
- `setup.py` — `_required_<ext>_sources()`, `_missing_<ext>_sources()`, `_get_<ext>_extension()`,
  `_get_extensions()`
- `tests/kernels/` — operator tests
- `tools/inspect_release_wheel.py` — release wheel content gates

Ask these questions while surveying:

1. Which existing domain module is the right host, or is a new domain justified?
2. Does the op belong in `cache_dit._C_comm`, `cache_dit._C_attn`, or a new/dedicated extension
   module?
3. How should extension loading fail when the op is unavailable?
4. Is there already a naming convention in `cache_dit_cuda_ops` for this operator family?
5. Does `test_ops_dispatch.py` already list the public names this migration will extend?

## 5. Phase 2: decide the migration shape

### 1. Freeze the public surface first

Settle early:

- Which operator names should be exposed publicly?
- Should the public API be low-level only, module-level only, or both?
- Should internal backend toggles be hidden from users?
- Should wrapper functions be explicit rather than `partial(...)` so editors and type tools can see
  the real signature?

Default rule: keep tuning and launch details private unless there is a strong user-facing reason to
expose them.

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

During the first migration pass, preserve behavior, shape/dtype rules, and dataflow first. Do not mix
the migration with optional cleanups such as naming polish, API reshaping, or algorithmic changes
unless they are necessary for repository consistency or import safety.

## 6. Phase 3: implement the migration

Apply changes from lowest level to highest level. The per-layer file layout, templates and the exact
`setup.py` edit sites are in `kernel-integration-chain.md` §3–§5; this section only records the
migration-specific decisions.

### A. Native code and dependency boundary

1. Move only the required native closure into `csrc/kernels/<domain>/`.
2. Put new code in `namespace cache_dit::<domain>` and keep upstream helper names internal with an
   anonymous namespace. The migrated SVDQ tree keeps its upstream `svdq::` namespace because it is
   upstream-derived; do not rename it as part of unrelated work.
3. Keep dispatch structure if it is functionally necessary (SVDQ needs its inheritance-based launch
   dispatch); do not rewrite it just because it looks unfamiliar.
4. Decide dependency strategy explicitly:
   - vendored in-tree
   - git submodule (`csrc/third_party/spdlog` is the only one today)
   - preinstalled system dependency
5. Extend an existing binding TU when the op belongs to that family; a new operator family gets its
   own `<domain>/{<domain>.h,<domain>.cpp}` pair instead of borrowing another family's TU. The only
   `PYBIND11_MODULE` entry points are `csrc/kernels/attn/attn.cpp`,
   `csrc/kernels/comm/comm_ops.cpp` and `csrc/kernels/svdq/pybind.cpp`.

### B. Domain Python module

Put everything for one domain in a single private module, `src/cache_dit/kernels/_<domain>.py`:
loader, bridge, registration, fake, public wrapper. Keep the loader in the same module as its ops so
a domain's failure mode, registration and public API cannot drift apart. Each domain loads its own
extension (`_attn.py` → `cache_dit._C_attn`, `_comm.py` → `cache_dit._C_comm`, `_svdquant.py` →
`cache_dit._C_svdquant`); borrow a sibling loader only when the ops really ship in that sibling's
extension.

Template and the rules it encodes: `kernel-integration-chain.md` §4.1–§4.5.

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

## 7. Phase 4: validate in layers

Do not rely on one test. Validation should usually proceed in this order:

1. **Import safety** — importing cache-dit without the optional extension should not crash
2. **API surface** — public names re-exported identically, `__all__` consistent
3. **Low-level smoke** — the op runs with the expected dtype, device and shape
4. **Low-level correctness** — compare against a dense or reference implementation
5. **Module correctness** — the higher-level module uses the migrated operator path correctly
6. **Fake / compile path** — at least one `register_fake` shape-and-dtype assertion, plus a
   `fullgraph=True` compile smoke
7. **Round-trip or end-to-end validation** — if serialization, quantization or pipeline integration
   exists, test it explicitly
8. **Boundary tests** — unsupported geometry, rank, alignment or build conditions fail clearly

Matching test files:

- `tests/kernels/test_ops_dispatch.py` — re-export identity, `__all__`, and the removed-dispatch guards
- `tests/kernels/test_cuda_fp8_comm_ops.py` — FP8 comm correctness, fake path, empty input, compile smoke
- `tests/kernels/test_cuda_merge_attn_states.py` — attention-state merge correctness and compile
- `tests/kernels/test_svdquant_runtime.py` — SVDQ runtime behaviour and INT4 v1/v2 equivalence
- `tests/quantization/` — module-level and quantizer behaviour

When scope is intentionally limited, say so explicitly:

- "INT4 CUDA path is the validation gate."
- "FP4 code is retained but not currently gated by runtime correctness tests."

Do not imply feature maturity beyond what the tests actually cover.

## 8. Phase 5: packaging and documentation

Operator migration is incomplete if build and packaging are wrong. The six `setup.py` edit sites and
the three "family list" sync points are enumerated once in `kernel-integration-chain.md` §5.1 and
§5.3 — follow that list. Migration-specific reminders:

1. `_missing_*_sources()` turns a forgotten entry into a clear build error; a source that is on disk
   but not in the list is silently not compiled.
2. Keep default install and import behaviour safe without any extension: no eager extension import
   outside the lazy loader.
3. `BuildExtension` only compares source mtimes, so after editing a shared header such as
   `csrc/kernels/attn/attn.h`, wipe `build/` (`CACHE_DIT_CLEAN=1`) instead of trusting cached
   objects.
4. Update `tools/inspect_release_wheel.py` when a wheel-content expectation changes, e.g. an
   additional `.so` that must be present in the SVDQuant wheel.
5. Update `docs/user_guide/INSTALL.md` for any new flag or host requirement.
6. Document only what is actually usable now; do not advertise unfinished features ahead of
   validated capability.

## 9. Anti-patterns

### Do not add a Triton or CuTe DSL operator

Bad: porting a Triton kernel as-is, or reinstating `kernels/triton/` "for portability".
Better: rewrite the algorithm as a CUDA kernel under `csrc/kernels/<domain>/`.

### Do not reintroduce a backend dispatch layer

Bad: adding `kernels/backend.py`, a `KernelBackend` enum, or `_resolve_backend` to pick between
implementations.
Better: one CUDA path, one `torch.library` schema, and a `RuntimeError` naming the build command
when the extension is missing.

### Do not mechanically mirror upstream layout

Bad: copying an entire source repo subtree into `csrc/` because one operator needed two files.
Better: identify the minimum closure and migrate only that set.

### Do not scatter one domain across files

Bad: registering the op in one module, loading the extension in another, wrapping in a third.
Better: one `_<domain>.py` holding loader, bridge, registration, fake and wrapper.

### Do not expose internal control knobs casually

Bad: exposing stage counts, launch tuning, or migration-only switches because they were convenient
during development.
Better: hardcode them at the internal wrapper layer until a real product need exists.

### Do not leak source-repo naming when cache-dit conventions already exist

Bad: keeping raw upstream helper names or state keys in the public interface.
Better: adapt them to the repository's public naming rules and keep the raw names private.

### Do not let optional extensions break base imports

Bad: importing the extension eagerly from top-level package import paths.
Better: delay extension import until the migrated operator is actually needed (the loader pattern in
`kernel-integration-chain.md` §4.1).

### Do not claim correctness from one smoke test

Bad: compiling the extension and declaring the migration complete.
Better: prove import safety, API surface, low-level execution, low-level correctness, higher-level
module behaviour and boundary failures.

### Do not write machine-local reference paths

Bad: `/abs/path/to/workspace/...` in a docstring, an error message, or a skill reference.
Better: `src/cache_dit/kernels/_comm.py`, `nunchaku/nunchaku/models/linear.py`,
`deepcompressor/deepcompressor/calib/smooth.py`.

## 10. Reference case: SVDQ / Nunchaku migration

Use this as an example of the workflow, not as a recipe to replay line by line.

### What the migration demonstrated

- native W4A4 closure can be migrated without importing an entire upstream project
  (`csrc/kernels/svdq/` was later reorganized into `core/`, `int4/`, `fp4/`, and those now live under
  the `csrc/kernels/svdq/native/` directory with `csrc/kernels/svdq/cute/` as its sibling, which is a
  local layout choice: upstream nunchaku keeps a flat `csrc/`)
- public operator wrappers should remain explicit and repository-aligned
- `torch.library` schemas and implementations live beside their domain wrappers, in the
  `cache_dit_cuda_ops` namespace
- optional extension import should be delayed and load errors should be queryable
- packaging may need submodule enforcement instead of hard-coded vendoring
- layered tests should cover runtime, module and end-to-end behaviours separately

### What the later refactor changed

The comm and attention-state-merge operators were CUDA ports of existing Triton and CuTe DSL kernels.
That migration is the template for future work:

- native sources landed in `csrc/kernels/comm/` and `csrc/kernels/attn/`
- each family registers through its own pybind TU: `cache_dit._C_comm` for the FP8 comm ops,
  `cache_dit._C_attn` for the attention-state merge (they shared `_C_comm` until the split)
- their Python modules colocate registration with the wrappers, replacing the old
  `cuda/_ops_registery.py` + `ops.py` pair
- the Triton and CuTe DSL trees, the backend selector and their tests were deleted, and
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

These details were important for SVDQ but are not universal migration rules:

- `svdq_*` naming convention
- W4A4 INT4 and FP4 split in scope and validation
- specific geometry and rank constraints
- quantized module state adaptation rules
- submodule choice for `spdlog`

If your operator family differs, keep the workflow but re-evaluate the decisions.

## 11. Execution order and exit criteria

Suggested order for a real migration:

1. confirm the target is CUDA; if the source is Triton or CuTe DSL, plan a rewrite
2. summarize the target operator and current scope boundary
3. list source files and the cache-dit host module
4. identify the minimum viable closure
5. freeze the public API and naming strategy
6. migrate native code and extend the pybind TU
7. add loader, bridge, registration, fake and wrapper in one domain module
8. re-export from `kernels/__init__.py`
9. wire `setup.py` source lists and build gates
10. add or adapt higher-level module code if needed
11. validate in layers for kernel, module and end-to-end behaviour
12. document only validated scope

The migration is not done until all of these are true:

- the operator is CUDA, with no Triton or CuTe DSL component anywhere in the change
- the public API is intentional and repository-aligned
- optional extension behaviour is safe
- dependency strategy is explicit
- every new native source appears in the `setup.py` source list
- tests prove correctness at the right layers, kernel and module level as needed
- unsupported cases fail clearly
- documentation matches validated reality
