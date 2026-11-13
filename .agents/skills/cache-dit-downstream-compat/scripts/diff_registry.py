#!/usr/bin/env python
"""Compare `BlockAdapterRegister` between two revisions and match downstream class names.

Why this exists: dispatch is by **class-name prefix**, and downstreams keep upstream class names on
vendored copies. A new registration can therefore (a) silently start accepting a model that used to be
rejected, and (b) hand it an adapter written for cache-dit's own pipeline.

  python diff_registry.py --repo <cache-dit> --base v1.5.1 [--target HEAD]
  python diff_registry.py --repo <cache-dit> --target HEAD --classes <downstream>/models

Exit code: 0 always unless the revision cannot be read (the report itself is informational).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ADAPTERS = "src/cache_dit/caching/block_adapters/adapters.py"
REGISTERS = "src/cache_dit/caching/block_adapters/block_registers.py"
TOP_LEVEL = re.compile(r"(Transformer3DModel|Transformer2DModel|DiTModel|DiT|Pipeline)$")


def read(repo: Path, revision: str, path: str) -> str:
  return subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path}"], text=True)


def registrations(source: str):
  """Name -> owned flag, parsed from the decorators (a name can be registered twice, keep any-
  true)."""
  found = {}
  for match in re.finditer(r"@BlockAdapterRegister\.register\(\s*\"([^\"]+)\"\s*(,)?([^)]*)\)",
                           source):
    name, _, rest = match.groups()
    found[name] = found.get(name, False) or bool(re.search(r"owned\s*=\s*True", rest))
  return found


def class_names(directory: Path):
  out = subprocess.run(["grep", "-rhoE", r"^class [A-Za-z0-9_]+",
                        str(directory)],
                       capture_output=True,
                       text=True,
                       check=False).stdout
  names = {line.split()[1] for line in out.splitlines() if line.startswith("class ")}
  return sorted(name for name in names if TOP_LEVEL.search(name))


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--repo", type=Path, required=True, help="cache-dit checkout")
  ap.add_argument("--base", default="v1.5.1")
  ap.add_argument("--target", default="HEAD")
  ap.add_argument("--classes", type=Path, help="downstream models directory to match against")
  args = ap.parse_args()

  if not (args.repo / ".git").exists():
    print(f"not a git checkout: {args.repo}", file=sys.stderr)
    return 2

  try:
    base = registrations(read(args.repo, args.base, ADAPTERS))
    target = registrations(read(args.repo, args.target, ADAPTERS))
  except subprocess.CalledProcessError as exc:
    print(f"cannot read revisions: {exc}", file=sys.stderr)
    return 2

  added, removed = sorted(set(target) - set(base)), sorted(set(base) - set(target))
  newly_owned = sorted(name for name in target if target[name] and not base.get(name, False))
  owned = sorted(name for name in target if target[name])

  print(f"{args.base}: {len(base)} prefixes   {args.target}: {len(target)} prefixes")
  print(f"added   ({len(added)}): {added}")
  print(f"removed ({len(removed)}): {removed}")
  print(f"owned   ({len(owned)}): {owned}")
  print(f"newly owned: {newly_owned}")

  # The prefixed-gate semantics depend on these helpers; flag their presence/shape.
  try:
    regs = read(args.repo, args.target, REGISTERS)
    print()
    print("_owned_adapters table:", "_owned_adapters" in regs)
    print("ownership probe     :", "MRO walk" if "__mro__" in regs else
          ("module check" if "__module__" in regs else "none"))
  except subprocess.CalledProcessError:
    print("\nblock_registers.py not found at the target revision")

  if args.classes:
    if not args.classes.exists():
      print(f"\nno such classes directory: {args.classes}", file=sys.stderr)
      return 2
    names = class_names(args.classes)

    def match(namespace):
      return {cls: pref for cls in names for pref in namespace if cls.startswith(pref)}

    base_hits, target_hits = match(base), match(target)
    print(f"\ndownstream classes: {len(names)}")
    print("newly matched (was 'unsupported'):")
    for cls in sorted(set(target_hits) - set(base_hits)):
      print(f"  {cls:46s} -> {target_hits[cls]}{'  [owned]' if target[target_hits[cls]] else ''}")
    print("no longer matched:")
    for cls in sorted(set(base_hits) - set(target_hits)):
      print(f"  {cls:46s} -> {base_hits[cls]}")
    print("matched in both (check has_separate_cfg accounting):")
    for cls in sorted(set(base_hits) & set(target_hits)):
      print(f"  {cls:46s} -> {target_hits[cls]}{'  [owned]' if target[target_hits[cls]] else ''}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
