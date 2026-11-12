"""Guard the single `cache_dit._C_ops` source list and arch policy in `setup.py`.

`_required_cuda_sources()` is what actually gets compiled, and it is assembled from five
hand-maintained lists. A family list that is never consulted, an entry point that is dropped,
or a kernel TU that no list mentions would all compile silently wrong: the extension would
just be missing operators at runtime. `csrc/kernels/ops.cpp` has the mirror-image failure: a
registrar that is never called leaves the operator missing from the `.so`.

The arch policy is the other half: `_is_supported_cuda_sm_target` decides which GPUs can run
the extension at all, and `DEFAULT_CUDA_ARCH_TARGETS` decides how long every build takes. Both
are data, so a silent edit changes nothing until someone builds on that hardware.

This test is CPU-only and does not import torch, the extension or `setup.py` itself; it
extracts the helpers with `ast`, like `test_svdquant_source_manifest.py` and
`test_models_source_manifest.py`.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "setup.py"
KERNELS_DIR = ROOT / "csrc" / "kernels"
ENTRY = "csrc/kernels/ops.cpp"
_HELPERS = (
  "_required_svdquant_sources",
  "_required_comm_sources",
  "_required_fp8_sources",
  "_required_attn_sources",
  "_required_models_sources",
  "_required_cuda_sources",
)
_ARCH_HELPERS = ("_is_supported_cuda_sm_target", "_default_cuda_arch_targets")
_ARCH_CONSTANTS = ("DEFAULT_CUDA_ARCH_TARGETS", )


def _load_helpers() -> dict:
  """Execute the manifest helpers from `setup.py` without importing it.

  :returns: The helpers, keyed by name.
  """

  tree = ast.parse(SETUP.read_text(encoding="utf-8"), filename=str(SETUP))
  wanted = [
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
  ]
  found = frozenset(node.name for node in wanted)
  assert found == frozenset(_HELPERS), (
    f"setup.py is missing manifest helpers: {sorted(frozenset(_HELPERS) - found)}")

  module = ast.Module(body=wanted, type_ignores=[])
  ast.fix_missing_locations(module)
  namespace: dict = {"ROOT_DIR": ROOT}
  exec(compile(module, str(SETUP), "exec"), namespace)
  return {name: namespace[name] for name in _HELPERS}


def _load_arch_helpers() -> dict:
  """Execute the arch-policy helpers and the default target tuple from `setup.py`.

  :returns: The arch helpers plus `DEFAULT_CUDA_ARCH_TARGETS`.
  """

  tree = ast.parse(SETUP.read_text(encoding="utf-8"), filename=str(SETUP))
  functions = [
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _ARCH_HELPERS
  ]
  assert len(functions) == len(_ARCH_HELPERS), f"setup.py must define {_ARCH_HELPERS}"
  constants = [
    node for node in tree.body if isinstance(node, ast.Assign) and any(
      getattr(target, "id", "") in _ARCH_CONSTANTS for target in node.targets)
  ]
  assert constants, f"setup.py must define {_ARCH_CONSTANTS}"

  module = ast.Module(body=[*functions, *constants], type_ignores=[])
  ast.fix_missing_locations(module)
  namespace: dict = {}
  exec(compile(module, str(SETUP), "exec"), namespace)
  return {name: namespace[name] for name in (*_ARCH_HELPERS, *_ARCH_CONSTANTS)}


def test_default_arch_targets_are_ada_and_blackwell_consumer_only():
  helpers = _load_arch_helpers()

  assert helpers["DEFAULT_CUDA_ARCH_TARGETS"] == ("89", "120a")
  assert helpers["_default_cuda_arch_targets"](support_sm120=True) == ["89", "120a"]
  # An nvcc without sm120a support must still build the Ada target instead of nothing.
  assert helpers["_default_cuda_arch_targets"](support_sm120=False) == ["89"]


@pytest.mark.parametrize(
  "sm,expected",
  [
    ("89", True),
    ("90", False),
    ("90a", False),
    ("80", False),
    ("86", False),
    ("100", True),
    ("120", True),
    ("120a", True),
    ("121a", True),
    ("foo", False),
  ],
)
def test_cuda_sm_predicate_is_sm89_plus_without_hopper(sm: str, expected: bool):
  # 100 stays buildable on purpose: it is only out of the *default* set, an explicit
  # CACHE_DIT_CUDA_ARCH_LIST can still add it.
  assert _load_arch_helpers()["_is_supported_cuda_sm_target"](sm) is expected


def test_cuda_sources_are_unique_and_exist():
  sources = _load_helpers()["_required_cuda_sources"]()

  assert len(sources) == len(set(sources)), f"_required_cuda_sources() has duplicates: {sources}"
  missing = [path for path in sources if not (ROOT / path).is_file()]
  assert not missing, f"_required_cuda_sources() lists files that do not exist: {missing}"


def test_cuda_sources_start_with_the_pybind_entry_point():
  sources = _load_helpers()["_required_cuda_sources"]()

  assert sources[0] == ENTRY
  assert (ROOT / ENTRY).is_file()


def test_cuda_sources_are_the_union_of_the_family_lists():
  helpers = _load_helpers()
  families = (
    "_required_svdquant_sources",
    "_required_comm_sources",
    "_required_fp8_sources",
    "_required_attn_sources",
    "_required_models_sources",
  )
  expected = {ENTRY}
  for helper in families:
    expected.update(helpers[helper]())

  assert set(helpers["_required_cuda_sources"]()) == expected


def test_pybind_entry_point_registers_every_family():
  entry = (ROOT / ENTRY).read_text(encoding="utf-8")

  assert entry.count("PYBIND11_MODULE") == 1
  for registrar in (
      "cache_dit::attn::register_attn_ops",
      "cache_dit::comm::register_comm_ops",
      "cache_dit::fp8::register_fp8_ops",
      "cache_dit::models::register_models_ops",
      "cache_dit::svdq::register_svdq_ops",
  ):
    assert entry.count(registrar) == 1, f"{ENTRY} does not hook {registrar} exactly once"


def test_cuda_sources_cover_every_kernel_file():
  listed = {
    path
    for path in _load_helpers()["_required_cuda_sources"]() if path.endswith((".cu", ".cpp"))
  }
  on_disk = {
    path.relative_to(ROOT).as_posix()
    for pattern in ("*.cu", "*.cpp")
    for path in KERNELS_DIR.rglob(pattern)
  }

  assert listed == on_disk, (
    "cache_dit._C_ops source manifest drifted from csrc/kernels: "
    f"not built: {sorted(on_disk - listed)}; not on disk: {sorted(listed - on_disk)}")
