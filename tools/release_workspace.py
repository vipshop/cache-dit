#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Prepare a release workspace for cache-dit wheels.")
  parser.add_argument("--source-root",
                      required=True,
                      help="Repository root containing pyproject.toml")
  parser.add_argument("--workspace", required=True, help="Workspace directory to create or refresh")
  parser.add_argument("--package-name",
                      required=True,
                      help="Distribution name to write into pyproject.toml")
  return parser.parse_args()


def _rewrite_pyproject_name(source_pyproject: Path, target_pyproject: Path,
                            package_name: str) -> None:
  text = source_pyproject.read_text()
  old = 'name = "cache_dit"'
  new = f'name = "{package_name}"'
  if text.count(old) != 1:
    raise RuntimeError("expected exactly one project name entry in pyproject.toml")
  target_pyproject.write_text(text.replace(old, new, 1))


def _symlink(target: Path, source: Path) -> None:
  target.parent.mkdir(parents=True, exist_ok=True)
  target.unlink(missing_ok=True)
  target.symlink_to(source)


def main() -> None:
  args = _parse_args()
  source_root = Path(args.source_root).resolve()
  workspace = Path(args.workspace).resolve()
  package_name = args.package_name

  if workspace.exists():
    shutil.rmtree(workspace)

  workspace.mkdir(parents=True, exist_ok=True)
  (workspace / "src").mkdir(parents=True, exist_ok=True)
  (workspace / "tools").mkdir(parents=True, exist_ok=True)

  _rewrite_pyproject_name(source_root / "pyproject.toml", workspace / "pyproject.toml",
                          package_name)
  _symlink(workspace / ".git", source_root / ".git")
  _symlink(workspace / "LICENSE", source_root / "LICENSE")
  _symlink(workspace / "MANIFEST.in", source_root / "MANIFEST.in")
  _symlink(workspace / "README.md", source_root / "README.md")
  _symlink(workspace / "setup.py", source_root / "setup.py")
  _symlink(workspace / "setup.cfg", source_root / "setup.cfg")
  _symlink(workspace / "csrc", source_root / "csrc")
  _symlink(workspace / "src" / "cache_dit", source_root / "src" / "cache_dit")
  _symlink(workspace / "tools" / "build_fast.sh", source_root / "tools" / "build_fast.sh")
  _symlink(workspace / "tools" / "nvcc", source_root / "tools" / "nvcc")


if __name__ == "__main__":
  main()
