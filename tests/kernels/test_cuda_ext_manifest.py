"""Guard the single `cache_dit._C_ops` source list in `setup.py`.

`_required_cuda_sources()` is what actually gets compiled, and it is assembled from five
hand-maintained lists. A family list that is never consulted, an entry point that is dropped,
or a kernel TU that no list mentions would all compile silently wrong: the extension would
just be missing operators at runtime. This test is CPU-only and does not import torch, the
extension or `setup.py` itself; it extracts the helpers with `ast`, like
`test_svdquant_source_manifest.py` and `test_models_source_manifest.py`.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "setup.py"
KERNELS_DIR = ROOT / "csrc" / "kernels"
ENTRY = "csrc/kernels/ops.cpp"
_HELPERS = (
  "_required_svdquant_sources",
  "_required_comm_sources",
  "_required_attn_sources",
  "_required_models_sources",
  "_required_cuda_sources",
)


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
