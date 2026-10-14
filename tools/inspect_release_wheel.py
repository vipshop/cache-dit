#!/usr/bin/env python3
"""Assert what a release wheel carries, before it can be published.

Three things a wheel built from a developer checkout can get wrong quietly, and
which only show up on a user's machine:

* A pure-Python wheel that carries a binary.  It installs as ``py3-none-any`` on
  any machine and only fails at import time, where the CUDA runtime is missing.
* A CUDA wheel that carries no extension -- the build silently skipped SVDQuant,
  and the wheel is the pure-Python one under a ``+cu13`` name.
* A ``_version.py`` that disagrees with the wheel's metadata.  ``cache_dit.
  __version__`` is imported from that module, so a stale copy makes the version
  reported at runtime differ from the version that was installed.

Usage::

    tools/inspect_release_wheel.py <wheel> --so none|svdquant --version 1.5.3+cu13
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

# setuptools-scm (via vcs-versioning) writes ``__version__ = version = '...'``,
# while a hand-written file uses a plain ``version = '...'``; accept both.
_VERSION_LINE = re.compile(r"^\s*(?:__version__\s*=\s*)?version\s*=\s*['\"]([^'\"]+)['\"]", re.M)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("wheel", type=Path, help="Wheel to inspect")
  parser.add_argument("--so",
                      choices=("none", "svdquant"),
                      required=True,
                      help="'none' for the pure-Python wheel, 'svdquant' for the CUDA one")
  parser.add_argument("--version",
                      required=True,
                      help="Version the wheel metadata and cache_dit/_version.py must agree on")
  return parser.parse_args()


def _module_version(archive: zipfile.ZipFile) -> str:
  members = [name for name in archive.namelist() if name.endswith("cache_dit/_version.py")]
  if len(members) != 1:
    raise SystemExit(f"expected exactly one cache_dit/_version.py, got {len(members)}")
  match = _VERSION_LINE.search(archive.read(members[0]).decode())
  if match is None:
    raise SystemExit("no 'version = ...' line in cache_dit/_version.py")
  return match.group(1)


def main() -> int:
  args = _parse_args()
  wheel = args.wheel
  if not wheel.is_file():
    raise SystemExit(f"no such wheel: {wheel}")

  with zipfile.ZipFile(wheel) as archive:
    binaries = [
      name for name in archive.namelist() if name.endswith((".so", ".pyd", ".dylib", ".dll", ".a"))
    ]
    if args.so == "none" and binaries:
      raise SystemExit(
        f"{wheel.name} carries a binary, so it is not a pure-Python wheel: {binaries}")
    if args.so == "svdquant" and not any("_C_svdquant" in name for name in binaries):
      raise SystemExit(f"{wheel.name} carries no SVDQuant extension: {binaries}")
    version = _module_version(archive)

  if version != args.version:
    raise SystemExit(
      f"{wheel.name}: cache_dit.__version__ is '{version}', expected '{args.version}'")

  print(f"OK: {wheel.name}: __version__={version}, so={len(binaries)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
