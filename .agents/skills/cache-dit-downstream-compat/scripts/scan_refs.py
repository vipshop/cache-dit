#!/usr/bin/env python
"""Enumerate every cache_dit reference in a downstream tree.

Handles what grep cannot: multi-line `from cache_dit import (...)`, relative module paths, and
`cache_dit.<attr>` reads (which is how `summary` / `refresh_context` / `disable_cache` are usually
called). Comments and docstrings never match because the scan walks the AST.

Usage:
  python scan_refs.py <downstream-dir> [--json] [--max-per-module N]
"""

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path

SKIP_DIRS = {
  ".git",
  "build",
  "dist",
  "node_modules",
  "__pycache__",
  ".venv",
  "venv",
  ".tmp",
  "site-packages",
  ".mypy_cache",
  ".pytest_cache",
  "third_party",
}


def iter_py(root: Path):
  for path in root.rglob("*.py"):
    if any(part in SKIP_DIRS for part in path.parts):
      continue
    yield path


def scan(root: Path):
  imports, attrs = [], []
  for path in iter_py(root):
    try:
      tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
      continue
    for node in ast.walk(tree):
      if isinstance(node,
                    ast.ImportFrom) and node.module and (node.module == "cache_dit"
                                                         or node.module.startswith("cache_dit.")):
        for alias in node.names:
          imports.append([str(path), node.lineno, node.module, alias.name])
      elif isinstance(node, ast.Import):
        for alias in node.names:
          if alias.name == "cache_dit" or alias.name.startswith("cache_dit."):
            imports.append([str(path), node.lineno, alias.name, ""])
      elif isinstance(node, ast.Attribute) and isinstance(
          node.value, ast.Name) and (node.value.id == "cache_dit"):
        attrs.append([str(path), node.lineno, node.attr])
  return imports, attrs


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("root", type=Path, help="downstream package or repo directory")
  ap.add_argument("--json", action="store_true")
  ap.add_argument("--max-per-module", type=int, default=40)
  args = ap.parse_args()

  if not args.root.exists():
    print(f"no such directory: {args.root}", file=sys.stderr)
    return 2

  imports, attrs = scan(args.root)
  if args.json:
    print(json.dumps({"imports": imports, "attrs": attrs}, indent=1))
    return 0

  modules = defaultdict(set)
  for _, _, module, symbol in imports:
    modules[module].add(symbol)
  for module in sorted(modules):
    symbols = sorted(modules[module])
    shown = ", ".join(symbols[:args.max_per_module])
    print(f"{module}  ({len(symbols)})")
    print(f"    {shown}{' …' if len(symbols) > args.max_per_module else ''}")
  print()
  print("attribute calls on `cache_dit`:", ", ".join(sorted({a[2] for a in attrs})) or "(none)")
  print()
  print(f"totals: {len(imports)} import bindings, {len(attrs)} attribute uses")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
