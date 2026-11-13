#!/usr/bin/env python
"""Print the cache-dit surface the known downstreams bind to, with signatures.

Run it with `cache_dit` resolving to the **target** tree, e.g.

  PYTHONPATH=<cache-dit>/src $(conda info --base)/envs/<env>/bin/python probe_api.py

and compare the output with the downstream's call sites. Optional sections that fail to import are
reported as `MISSING …` instead of aborting, so a renamed module is visible rather than fatal.
"""

import dataclasses
import inspect


def sig(obj, name=None):
  try:
    return f"{name or obj.__name__}{inspect.signature(obj)}"
  except (TypeError, ValueError) as exc:
    return f"{name}: <unavailable: {exc}>"


def section(title):
  print()
  print(f"== {title}")


def main() -> int:
  import cache_dit

  print("cache_dit:", cache_dit.__version__)
  print("file      :", cache_dit.__file__)

  section("top-level helpers used by downstreams")
  for name in ("enable_cache", "refresh_context", "disable_cache", "steps_mask", "summary",
               "load_configs", "set_compile_configs", "enable_parallelism", "disable_parallelism"):
    fn = getattr(cache_dit, name, None)
    print("  ", sig(fn, name) if fn is not None else f"{name}: MISSING")

  section("configs the downstreams construct")
  for name in ("DBCacheConfig", "CalibratorConfig", "TaylorSeerCalibratorConfig",
               "DMDCalibratorConfig", "ParamsModifier", "QuantizeConfig", "ParallelismConfig"):
    cls = getattr(cache_dit, name, None)
    if cls is None:
      print(f"   {name}: MISSING at top level")
      continue
    try:
      params = list(inspect.signature(cls).parameters)
    except (TypeError, ValueError):
      params = []
    print(f"   {name}({', '.join(params)})")

  section("DBCacheConfig fields (extras beyond the classic set matter)")
  try:
    from cache_dit import DBCacheConfig
    print("   ", [f.name for f in dataclasses.fields(DBCacheConfig)])
  except Exception as exc:  # noqa: BLE001 - report, do not crash the audit
    print(f"   <unavailable: {exc}>")

  section("caching / registry internals that downstreams reach into")
  try:
    from cache_dit.caching.block_adapters import BlockAdapter, BlockAdapterRegister, FakeDiffusionPipeline
    print("  ", sig(BlockAdapter.__init__))
    print("   register      ", sig(BlockAdapterRegister.register))
    print("   get_adapter   ", sig(BlockAdapterRegister.get_adapter))
    print("   is_supported  ", sig(BlockAdapterRegister.is_supported))
    print("   owned prefixes:", sorted(getattr(BlockAdapterRegister, "_owned_adapters", set())))
    print("   FakeDiffusionPipeline", sig(FakeDiffusionPipeline.__init__))
  except Exception as exc:  # noqa: BLE001
    print(f"   <unavailable: {exc}>")

  try:
    from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
    print("   CachedAdapter.apply", sig(CachedAdapter.apply))
  except Exception as exc:  # noqa: BLE001
    print(f"   CachedAdapter: <unavailable: {exc}>")

  try:
    from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    print("   CachedContextManager.similarity", sig(CachedContextManager.similarity))
    print("   CachedContextManager.can_cache ", sig(CachedContextManager.can_cache))
    print("   methods:", [m for m in dir(CachedContextManager) if not m.startswith("_")])
  except Exception as exc:  # noqa: BLE001
    print(f"   CachedContextManager: <unavailable: {exc}>")

  section("attention / kernels availability helpers (renamed once already)")
  try:
    from cache_dit.kernels.cuda_ext import cuda_ext_is_available
    print("   cuda_ext_is_available:", cuda_ext_is_available())
  except Exception as exc:  # noqa: BLE001
    print(f"   cuda_ext_is_available: <unavailable: {exc}>")
  for legacy in ("svdq_is_available", "svdq_get_load_error", "svdq_extension_is_available"):
    print(f"   cache_dit.quantization.{legacy}:",
          "present" if hasattr(getattr(cache_dit, "quantization", None), legacy) else "MISSING")

  section("orchestration submodules that downstreams import directly")
  for module in ("cache_dit.caching.block_adapters", "cache_dit.caching.cache_contexts",
                 "cache_dit.caching.cache_contexts.cache_manager", "cache_dit.distributed",
                 "cache_dit.parallelism"):
    try:
      __import__(module)
      print(f"   {module}: OK")
    except Exception as exc:  # noqa: BLE001
      print(f"   {module}: {type(exc).__name__}: {exc}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
