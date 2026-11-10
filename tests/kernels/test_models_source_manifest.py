"""Guard the `cache_dit._C_models` source list in `setup.py`.

`_required_models_sources()` is an explicit list, not a glob, so a new `<model>_<op>.cu`
that is not added there is simply never compiled: `_missing_models_sources()` stays empty
and the failure only shows up at runtime as `AttributeError: ops has no attribute ...`.
This test is CPU-only and does not import torch, the extension or `setup.py` itself; it
extracts the two helper functions with `ast` and executes them in isolation, mirroring
`tests/kernels/test_svdquant_source_manifest.py`.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETUP = ROOT / "setup.py"
MODELS_DIR = ROOT / "csrc" / "kernels" / "models"
_HELPERS = ("_required_models_sources", "_missing_models_sources")


def _load_helpers() -> dict:
  """Execute the manifest helpers from `setup.py` without importing it.

  :returns: The two helpers, keyed by name.
  """

  source = SETUP.read_text()
  tree = ast.parse(source)
  wanted = [
    node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in _HELPERS
  ]
  assert len(wanted) == len(_HELPERS), f"setup.py must define {_HELPERS}"

  module = ast.Module(body=wanted, type_ignores=[])
  namespace = {
    "ROOT_DIR": ROOT,
    "list": list,
    "str": str,
  }
  exec(compile(module, str(SETUP), "exec"), namespace)
  return {name: namespace[name] for name in _HELPERS}


def test_models_sources_cover_every_kernel_file():
  helpers = _load_helpers()
  required = {Path(relative).name for relative in helpers["_required_models_sources"]()}
  on_disk = {path.name
             for path in MODELS_DIR.glob("*.cu")
             } | {path.name
                  for path in MODELS_DIR.glob("*.cpp")}

  missing = on_disk - required
  assert not missing, (
    f"csrc/kernels/models files missing from _required_models_sources(): {sorted(missing)}. "
    "They would never be compiled, because the list is explicit rather than a glob.")


def test_models_sources_do_not_reference_deleted_files():
  helpers = _load_helpers()
  missing = helpers["_missing_models_sources"]()

  assert not missing, f"_required_models_sources() lists files that do not exist: {missing}"
