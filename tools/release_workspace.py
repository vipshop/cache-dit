#!/usr/bin/env python3
"""Prepare the out-of-tree workspace a release wheel is built in.

The distribution name is a static field of ``pyproject.toml``, so the release
build runs in a tree of its own whose project name is rewritten to the release
distribution.  The pure-Python wheel and the CUDA wheel share that distribution
name; what separates them is the local version of the wheel (``+cu13``), which
``build_releases.sh`` hands to setuptools-scm.

Copied: ``setup.py`` and ``src``.  Symlinked: ``csrc`` (the native tree, large and
read-only for this build), ``.git`` (setuptools-scm reads the tags and the commit
id from it) and the two tools the build calls.

Copying ``setup.py`` and ``src`` is what keeps the checkout clean and the wheel
self-consistent:

* setuptools-scm writes the package's ``_version.py`` from the release version
  (it does so for a pretend version too), and the extension build writes its
  objects under ``build/``; both have to land in the workspace, and neither may
  touch the developer's tree.  Letting setuptools-scm write it is also what keeps
  the module's shape -- ``version_tuple`` and the rest -- in step with the tool
  that owns it.
* The checkout's ``src/cache_dit`` may carry build residue -- a ``_version.py``
  from some earlier build, a stale ``_C_svdquant*.so``.  Copying with an ignore
  list keeps that residue out of the wheel, so the metadata, the wheel's own
  ``__version__`` and its payload all describe the same build.
* ``setup.py`` resolves its own root through ``Path(__file__).resolve()``.  Were
  it a symlink, that root would be the checkout, and every path derived from it --
  the spdlog headers, the source list, the include dirs -- would come from the
  wrong tree.

Usage::

    tools/release_workspace.py --source-root . --workspace .tmp/release_build/cu13 \
        --package-name cache_dit
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# What the checkout may hold and the build recreates.  ``_version.py`` is the one
# that matters most: a stale copy would ship a version that contradicts the
# wheel's metadata.
_IGNORE_SRC = shutil.ignore_patterns("*.so", "*.pyd", "*.pyc", "__pycache__", "_version.py",
                                     "*.egg-info")

_COPIED_FILES = ("LICENSE", "MANIFEST.in", "README.md", "setup.py", "setup.cfg")
# Large and read-only for this build; ``.git`` is what setuptools-scm reads.
_LINKED_PATHS = (".git", "csrc")
# Reached through the wrapper the release build calls; both must resolve to the
# workspace, which they do because ``build_fast.sh`` derives its root from ``$0``.
_LINKED_TOOLS = ("build_fast.sh", "nvcc")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Prepare a release workspace for cache-dit wheels.")
  parser.add_argument("--source-root",
                      required=True,
                      help="Repository root containing pyproject.toml")
  parser.add_argument("--workspace", required=True, help="Workspace directory to create")
  parser.add_argument("--package-name",
                      required=True,
                      help="Distribution name to write into pyproject.toml")
  return parser.parse_args()


def _rewrite_pyproject_name(source_pyproject: Path, target_pyproject: Path,
                            package_name: str) -> None:
  """Rename the distribution and the extras that refer to it by name.

  ``all = ["cache-dit[parallelism]", ...]`` has to name this distribution too, or
  installing that extra resolves the PyPI distribution on top of this build --
  two distributions owning the same ``cache_dit/`` package.
  """
  text = source_pyproject.read_text(encoding="utf-8")
  old = 'name = "cache_dit"'
  if text.count(old) != 1:
    raise RuntimeError("expected exactly one project name entry in pyproject.toml")
  text = text.replace(old, f'name = "{package_name}"', 1)
  target_pyproject.write_text(text.replace('"cache-dit[', f'"{package_name}['), encoding="utf-8")


def _symlink(target: Path, source: Path) -> None:
  target.parent.mkdir(parents=True, exist_ok=True)
  target.unlink(missing_ok=True)
  target.symlink_to(source)


def _assert_removable(source_root: Path, workspace: Path) -> None:
  """Refuse to remove something that could be the repository itself.

  Everything else here is repeatable; the rmtree is not, and --workspace takes an arbitrary path.
  """
  if workspace == source_root or source_root.is_relative_to(workspace):
    raise SystemExit(f"release_workspace: refusing to remove {workspace}: it is {source_root} "
                     "or contains it")
  if workspace == Path(workspace.anchor) or len(workspace.parts) < 3:
    raise SystemExit(f"release_workspace: refusing to remove {workspace}: too close to the "
                     "filesystem root")


def main() -> None:
  args = _parse_args()
  source_root = Path(args.source_root).resolve()
  workspace = Path(args.workspace).resolve()

  if not (source_root / "pyproject.toml").is_file():
    raise SystemExit(f"release_workspace: no pyproject.toml under {source_root}")

  _assert_removable(source_root, workspace)
  if workspace.exists():
    shutil.rmtree(workspace)
  workspace.mkdir(parents=True, exist_ok=True)
  (workspace / "tools").mkdir(parents=True, exist_ok=True)

  _rewrite_pyproject_name(source_root / "pyproject.toml", workspace / "pyproject.toml",
                          args.package_name)
  for name in _COPIED_FILES:
    shutil.copyfile(source_root / name, workspace / name)
  for name in _LINKED_PATHS:
    _symlink(workspace / name, source_root / name)
  for name in _LINKED_TOOLS:
    _symlink(workspace / "tools" / name, source_root / "tools" / name)

  shutil.copytree(source_root / "src", workspace / "src", ignore=_IGNORE_SRC)

  entries = sorted(item.name for item in workspace.iterdir())
  print(f"[release_workspace] {workspace}")
  print(f"[release_workspace] distribution: {args.package_name} "
        "(src/cache_dit/_version.py is written by setuptools-scm at build time)")
  print(f"[release_workspace] entries: {' '.join(entries)}")


if __name__ == "__main__":
  main()
