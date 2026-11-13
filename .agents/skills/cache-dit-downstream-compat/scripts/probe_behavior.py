#!/usr/bin/env python
"""Probe the cache-dit behaviors the downstreams actually depend on.

Everything runs on a synthetic 4-block DiT, so it needs torch but no weights and no GPU. Each check
prints `PASS` / `CHANGED` / `FAIL` so a regression is visible without reading the source.

  PYTHONPATH=<cache-dit>/src python probe_behavior.py [--json]
"""

import argparse
import json
import traceback

import torch.nn as nn

results = []


def record(check, status, detail):
  results.append({"check": check, "status": status, "detail": detail})
  print(f"[{status:7s}] {check}: {detail}")


class _Block(nn.Module):

  def forward(self, hidden_states, encoder_hidden_states, temb):
    return hidden_states + 1, encoder_hidden_states


class _Transformer(nn.Module):

  def __init__(self, n=4):
    super().__init__()
    self.transformer_blocks = nn.ModuleList([_Block() for _ in range(n)])

  def forward(self, hidden_states, encoder_hidden_states=None, **kwargs):
    return (hidden_states, )


def build(enable_separate_cfg=True, num_inference_steps=None):
  from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, ParamsModifier, enable_cache

  transformer = _Transformer()
  config = DBCacheConfig(num_inference_steps=num_inference_steps,
                         Fn_compute_blocks=1,
                         Bn_compute_blocks=0,
                         max_warmup_steps=4,
                         residual_diff_threshold=0.12,
                         enable_separate_cfg=enable_separate_cfg)
  adapter = BlockAdapter(transformer=transformer,
                         blocks=[transformer.transformer_blocks],
                         forward_pattern=[ForwardPattern.Pattern_1],
                         params_modifiers=[ParamsModifier(cache_config=config)],
                         check_forward_pattern=False)
  cached = enable_cache(adapter, cache_config=config)
  return transformer, cached, config


def context_config(transformer):
  name = transformer._context_names[0]
  return transformer._context_manager.get_context(name).cache_config


def check_refresh_merges():
  import cache_dit
  transformer, _, _ = build()
  cache_dit.refresh_context(transformer, num_inference_steps=12, verbose=False)
  cfg = context_config(transformer)
  ok = (cfg.num_inference_steps == 12 and cfg.Fn_compute_blocks == 1
        and abs(cfg.residual_diff_threshold - 0.12) < 1e-9)
  record(
    "refresh_context 只合并不重置", "PASS" if ok else "CHANGED",
    f"num_steps={cfg.num_inference_steps} Fn={cfg.Fn_compute_blocks} R={cfg.residual_diff_threshold}"
  )
  return transformer


def check_refresh_separate_cfg():
  import cache_dit
  transformer, _, _ = build()
  cache_dit.refresh_context(transformer,
                            num_inference_steps=10,
                            enable_separate_cfg=False,
                            verbose=False)
  cfg = context_config(transformer)
  record("refresh_context 接受 enable_separate_cfg",
         "PASS" if cfg.enable_separate_cfg is False else "FAIL",
         f"enable_separate_cfg={cfg.enable_separate_cfg} num_steps={cfg.num_inference_steps}")


def check_refresh_requires_persistent_context():
  import cache_dit
  from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, enable_cache

  class _Pipe(nn.Module):

    def __init__(self):
      super().__init__()
      self.transformer = _Transformer()

  pipe = _Pipe()
  config = DBCacheConfig(Fn_compute_blocks=1,
                         Bn_compute_blocks=0,
                         max_warmup_steps=4,
                         residual_diff_threshold=0.12)
  adapter = BlockAdapter(pipe=pipe,
                         transformer=pipe.transformer,
                         blocks=[pipe.transformer.transformer_blocks],
                         forward_pattern=[ForwardPattern.Pattern_1],
                         check_forward_pattern=False)
  enable_cache(adapter, cache_config=config)
  # With a real pipeline the context manager is non-persistent, so `refresh_context` warns and
  # no-ops. The context itself only exists during a pipeline call, hence the flag is the evidence.
  manager = pipe.transformer._context_manager
  cache_dit.refresh_context(pipe.transformer, num_inference_steps=9, verbose=False)
  record("pipe= 的 adapter 上 refresh_context 是 no-op（非 persistent context）",
         "PASS" if manager.persistent_context is False else "CHANGED",
         f"persistent_context={manager.persistent_context}")


def check_steps_mask_short_schedules():
  import cache_dit
  accepted = []
  for total in (2, 4, 5, 6, 7, 8, 12):
    try:
      cache_dit.steps_mask(mask_policy="fast", total_steps=total)
      accepted.append(total)
    except Exception:  # noqa: BLE001 - the rejection itself is the finding
      pass
  record("steps_mask 短步可用集合", "PASS" if accepted == [4, 6, 8, 12] else "CHANGED",
         f"accepted={accepted}（下游必须自己 guard 2/5/7）")


def check_summary_shape():
  import cache_dit
  transformer, _, _ = build()
  first = len(cache_dit.summary(transformer, details=False))
  cache_dit.refresh_context(transformer, num_inference_steps=8, verbose=False)
  second = len(cache_dit.summary(transformer, details=False))
  record("summary(module) 条数不随 refresh 增长", "PASS" if first == second else "FAIL",
         f"after_enable={first} after_refresh={second}（不要假设 1 条）")


def check_disable():
  import cache_dit
  transformer, cached, _ = build()
  cache_dit.disable_cache(cached)
  ok = getattr(transformer, "_is_cached",
               None) is None and not hasattr(transformer, "_context_manager")
  record(
    "disable_cache(BlockAdapter) 清理干净", "PASS" if ok else "CHANGED",
    f"_is_cached={getattr(transformer, '_is_cached', None)} "
    f"has_manager={hasattr(transformer, '_context_manager')}")


def check_auto_without_pattern():
  from cache_dit import BlockAdapter

  block = BlockAdapter(transformer=_Transformer())
  try:
    BlockAdapter.auto_block_adapter(block)
    record("BlockAdapter(auto=True) 缺 forward_pattern", "CHANGED", "不再断言，需复核派发结果")
  except AssertionError as exc:
    record("BlockAdapter(auto=True) 缺 forward_pattern", "PASS", f"AssertionError: {exc}")


def check_registry_owned():
  from cache_dit.caching.block_adapters import BlockAdapterRegister

  owned = sorted(getattr(BlockAdapterRegister, "_owned_adapters", set()))
  foreign = type("LongLive2Transformer3DModel", (nn.Module, ), {})()
  status = "PASS" if owned and not BlockAdapterRegister.is_supported(foreign) else "CHANGED"
  record(
    "owned 前缀拒绝外来同名类（#1320）", status, f"owned={len(owned)} is_supported(foreign LongLive2)="
    f"{BlockAdapterRegister.is_supported(foreign)}")


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()

  checks = [
    check_refresh_merges,
    check_refresh_separate_cfg,
    check_refresh_requires_persistent_context,
    check_steps_mask_short_schedules,
    check_summary_shape,
    check_disable,
    check_auto_without_pattern,
    check_registry_owned,
  ]
  for check in checks:
    try:
      check()
    except Exception:  # noqa: BLE001 - a probe crash is a finding, keep going
      record(check.__name__, "FAIL", traceback.format_exc(limit=2).strip().splitlines()[-1])

  if args.json:
    print(json.dumps(results, indent=1))
  failures = [r for r in results if r["status"] == "FAIL"]
  print(f"\nsummary: {len(results) - len(failures)} ok, {len(failures)} failed, "
        f"{len([r for r in results if r['status'] == 'CHANGED'])} changed")
  return 1 if failures else 0


if __name__ == "__main__":
  raise SystemExit(main())
