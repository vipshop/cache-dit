"""CPU-only guard for the SVDQ CUDA source manifest declared in setup.py.

The extension sources are listed by hand in ``_required_svdquant_sources()``, so a file rename,
a new kernel TU or a stray leftover source can silently change what gets compiled. These tests
compare that manifest against ``csrc/kernels/svdq`` without importing torch, the extension or
setup.py itself.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP_PY = REPO_ROOT / "setup.py"
SVDQ_SOURCE_ROOT = REPO_ROOT / "csrc" / "kernels" / "svdq"
MANIFEST_HELPERS = frozenset({"_required_svdquant_sources"})


def _load_manifest_helpers() -> dict:
  """Evaluate only the SVDQ source-manifest helpers out of setup.py.

  setup.py has no ``__main__`` guard, so importing it would start a build. The helpers are pure
  functions over string literals and can be executed on their own.

  :returns: Namespace holding the manifest helper functions.
  """

  tree = ast.parse(SETUP_PY.read_text(encoding="utf-8"), filename=str(SETUP_PY))
  body = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in MANIFEST_HELPERS
  ]
  found = frozenset(node.name for node in body)
  assert found == MANIFEST_HELPERS, (
    f"setup.py is missing SVDQ manifest helpers: {sorted(MANIFEST_HELPERS - found)}")

  module = ast.Module(body=body, type_ignores=[])
  ast.fix_missing_locations(module)
  namespace: dict = {}
  exec(compile(module, str(SETUP_PY), "exec"), namespace)
  return namespace


def _manifest_sources() -> list[str]:
  return _load_manifest_helpers()["_required_svdquant_sources"]()


def test_svdquant_manifest_paths_exist() -> None:
  sources = _manifest_sources()
  assert len(sources) == len(set(sources)), f"setup.py lists duplicate SVDQ sources: {sources}"

  missing = [path for path in sources if not (REPO_ROOT / path).is_file()]
  assert not missing, f"setup.py lists SVDQ sources that do not exist: {missing}"


def test_svdquant_manifest_covers_every_compiled_source() -> None:
  listed = {path for path in _manifest_sources() if path.endswith((".cu", ".cpp"))}
  on_disk = {
    path.relative_to(REPO_ROOT).as_posix()
    for pattern in ("*.cu", "*.cpp")
    for path in SVDQ_SOURCE_ROOT.rglob(pattern)
  }
  assert listed == on_disk, (
    "SVDQ source manifest drifted from csrc/kernels/svdq: "
    f"not built: {sorted(on_disk - listed)}; not on disk: {sorted(listed - on_disk)}")
