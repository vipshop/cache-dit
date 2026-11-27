# 代码落点与改动清单

> 行号是 `main@7773cda3`（2026-11-16）的快照。代码会动（`main` 已继续前进），动手前用 `rg` 复核一次再改。

## 1. 文件地图

```
src/cache_dit/caching/
  cache_types.py                     # CacheType 枚举 + 别名解析 + block_range()
  cache_interface.py                 # 对外 API：enable_cache / steps_mask / check_min_bn_blocks
  load_configs.py                    # 从 yaml/CLI 还原 config，含 calibrator 互斥校验
  cache_adapters/cache_adapter.py    # patch transformer forward；挂 _context_manager
  cache_blocks/
    __init__.py                      # UnifiedBlocks / CachedBlocks 的 __new__ 分派
    pattern_base.py                  # 0/1/2 类模型的模板：Fn → can_cache → apply/store → Mn → Bn
    pattern_3_4_5.py                 # 3/4/5 类模型：独立完整实现（不是包装！）
    pattern_utils.py                 # 统计挂到 module：cached_steps / residual_diffs
  cache_contexts/
    context_manager.py               # ContextManager.__new__：DBCache 分支 + raise
    cache_manager.py                 # 判定主逻辑：can_cache / similarity / 缓冲读写
    cache_context.py                 # CachedContext 状态（步数 / diff / 连续计数）
    cache_config.py                  # BasicCacheConfig（= DBCacheConfig）+ ExtraCacheConfig
    calibrators/
      base.py                        # CalibratorBase：reset_cache / approximate / mark_step_begin / update
      taylorseer.py  dmd.py  foca.py
      __init__.py                    # CalibratorConfig + 工厂（_supported_calibrators）
```

## 2. 三条硬性边界（改之前必须知道）

1. **`Fn >= 1`**（`cache_manager.py` 的 `Fn_compute_blocks()` 断言）。每步至少算 1 个 block，"整步跳过"在现模板里**无法表达**。
2. **模板固定**：`Fn 必算 → can_cache → 命中则 apply_cache（Mn 整段跳过）→ Bn 必算`。任何"按需改变这个顺序"的算法都不是加 calibrator 能解决的。
3. **两份 pattern 是真实分叉**：`pattern_0_1_2.py` 全是 `...`（纯继承），但 `pattern_3_4_5.py` 是独立实现（`_supported_patterns = [Pattern_3, Pattern_4, Pattern_5]`）。新增 `CacheType` 必须写两遍，否则 `Pattern_3/4/5` 家族的模型直接不可用。

## 3. 方案 1：新增 calibrator（改动最小）

只替换"命中时返回什么张量"，不碰判定逻辑。**4 处改动 + 测试**：

| # | 文件 | 改什么 |
|---|---|---|
| 1 | `calibrators/<name>.py`（新增） | 继承 `CalibratorBase`，实现 `reset_cache/approximate/mark_step_begin/update`（**`reset_cache` 不是 `reset`**，基类是普通类，写错只会延迟到第一次推理才 `NotImplementedError`） |
| 2 | `calibrators/__init__.py` | `_supported_calibrators` 加名字 + 工厂 `if` 分支；参数走 `calibrator_kwargs` 透传（**不需要新 dataclass**） |
| 3 | `load_configs.py` | `enable_<name>` 开关 + 与 taylorseer/dmd/foca 的互斥校验 |
| 4 | `cli/cache.py` | `--calibrator-params` 的 `type` 及参数 key；组装 `CalibratorConfig(...)` |

骨架（2 空格缩进，相对导入；照 `taylorseer.py` 的形状写）：

```python
from typing import Dict

import torch

from .base import CalibratorBase


class XxxState:
  """Per-stream state for one forecast operator."""

  def __init__(self, foo: float = 0.9):
    self.foo = foo

  def reset(self):
    ...

  def mark_step_begin(self):
    ...

  def approximate(self) -> torch.Tensor:
    ...

  def update(self, Y: torch.Tensor):
    ...


class XxxCalibrator(CalibratorBase):
  """One-line description of the forecast operator."""

  def __init__(self, foo: float = 0.9, **kwargs):
    self.foo = foo
    self.states: Dict[str, XxxState] = {}
    self.reset_cache()

  def reset_cache(self):
    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(self, name: str = "default"):
    if name not in self.states:
      self.states[name] = XxxState(foo=self.foo)

  def mark_step_begin(self, *args, **kwargs):
    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def approximate(self, name: str = "default") -> torch.Tensor:
    assert name in self.states, f"State '{name}' not found."
    return self.states[name].approximate()

  def update(self, Y: torch.Tensor, name: str = "default"):
    self.maybe_init_state(name)
    self.states[name].update(Y)
```

注意：框架调用的是 `calibrator.reset_cache()`（无参、遍历所有流），`mark_step_begin()` 也是无参调用；单测模板见 `tests/api/test_foca.py`。

**`to_kwargs()` 必须写成 `dict(self.calibrator_kwargs or {})`**，不能用 `self.calibrator_kwargs.copy()`：`cache_dit.refresh_context(transformer, **shorthand)` 是唯一 `reset=True` 入口，它经 `CalibratorConfig.empty()` 把 `calibrator_kwargs` 置 `None`，随后重建 context 时 `Calibrator(config)` → `to_kwargs()` 会崩（TaylorSeer / DMD / SVD 曾中招，#1359 修掉）。回归用例模板见 `tests/api/test_calibrator_reset.py`。

**已知签名限制**：`approximate()` 只拿到 `name`，拿不到"当前步前缀块的输出"。若算法需要当前步信息（如 DiCache 的 DCTA 需要探针残差），必须改 `cache_manager.get_Bn_buffer()` 与 `pattern_base.py` 的 `apply_cache` 调用点，把张量传进来——这会穿过主路径，成本远高于普通 calibrator。

## 4. 方案 2：扩展现有 calibrator（推荐，维护面零增长）

论文只换"基函数 / 系数来源"时，别新增 `calibrator_type`，直接加开关：

- **HiCache** = `TaylorSeerCalibrator` 的 `basis ∈ {taylor, hermite}` + `sigma`。`derivative()` 不动，只改 `approximate()` 的累加项：`(elapsed ** i)` → `sigma**i * H_i(sigma * elapsed)`。
- **两个锚的线性式** = `FoCaState` 的 `F_bdf2 = 2*F_k - F_km1`（系数 (2, −1)）换成"实测距离比"系数。这类改动同样需要方案 1 末尾提到的签名改动（系数要当前步信息）。

判据：交换后 `calibrator_type` 的数量不变 → 维护面不增。

## 5. 方案 3：换调度（不动预测器）

`cache_interface.py::steps_mask()` 已支持 `compute_bins/cache_bins` 与 `slow/medium/fast/ultra` 四种预设。新增"从曲线生成 mask"：

```python
# mask_policy="magnitude" 时，按 γ 曲线的累积误差生成 0/1
acc, mask = 0.0, []
for gamma in ratios:
  acc += 1.0 - gamma
  mask.append(0 if acc <= delta else 1)
  if mask[-1] == 1:
    acc = 0.0
```

配合 `steps_computation_policy="static"`（mask=0 直接命中缓存，不算 diff）。
改动 2 个文件，**不新增跳过粒度**。代价：只能门控块级缓存，不能整步跳。

## 6. 方案 4：新增 `CacheType`（最后手段，约 10 处）

| 层 | 文件 | 备注 |
|---|---|---|
| enum | `cache_types.py` | 加枚举 + 字符串别名；未识别现在会静默回落 `NONE`，别依赖这个行为 |
| config | `cache_config.py` 或新 `<xxx>_config.py` | 新 dataclass 必须实现 `strify()`（它进日志与产物文件名） |
| config 解析 | `load_configs.py` | `if cache_type == ...` 分支，else 会 raise |
| context 分派 | `context_manager.py` | `__new__` + `_supported_managers` |
| state | `cache_context.py` / 新 manager | 步数、diff、连续计数、CFG 分支同构 |
| block 分派 | `cache_blocks/__init__.py` | `UnifiedBlocks.__new__` / `CachedBlocks.__new__` 两层 `if/elif` |
| block 执行 | `pattern_base.py` **和** `pattern_3_4_5.py` | 两套模板都要写 |
| stats | `pattern_utils.py`、`summary.py` | `apply_stats/remove_stats` + `CacheStats` + `strify` 三处分支 |
| CLI | `cli/optimize.py` | 现在硬编码 `DBCacheConfig(...)`；`--cache-params` 的 `type` 只认 DBCache/SeaCache/SoftCache |
| docs | `docs/user_guide/CACHE_API.md`、`docs/README.md` | 新算法要进文档（根 `README.md` 已最简化，不写算法说明） |

## 7. 与其它特性的交互

| 特性 | 约束 | 位置 |
|---|---|---|
| CP（ulysses/ring） | `Bn >= 1` 被 planner 要求（`_cp_requires_min_bn_blocks`）；Fn 残差要按同一 split hook 切分后再相减。要 CP 就用 `DBCacheConfig(Bn_compute_blocks>=1)` | `cache_interface.py::check_min_bn_blocks`、`pattern_base.py` 的 `_get_Fn_residual` |
| TP | `similarity()` 里 `mean_diff/mean_t1` 走 `all_reduce(AVG)`；batch 分片下是近似 | `cache_manager.py` |
| 量化 | 无显式耦合 | — |
| offload | `cache_blocks/offload_utils.py` 是死代码；group offloading 有 TODO「仍不完全兼容」 | `cache_adapter.py` 的 `new_forward_with_hf_hook` |
| compile | 判定路径全部 `@torch.compiler.disable`（图外 Python 控制流）；快照类状态要 `detach().clone()`，否则被 CUDA-graph buffer 复用踩踏 | `dmd.py` 的 snapshot 注释 |
| CFG | `enable_separate_cfg` 影响计步与计 diff；两分支各有独立计数 | `cache_context.py` |
| 多流（video+audio） | 每条流要独立断言逐比特，不能只测一条 | 参考 MAGI-2 / Kandinsky6 的 adapter 测试 |
| 多 chunk AR | `num_inference_steps` 默认 `None` 会让刷新永不触发；每段起点要 arm `force_refresh_step_hint` | `cache_manager.maybe_refresh` |

## 8. 最小测试矩阵

单测照现成模板抄：新 calibrator → `tests/api/test_foca.py`；config / `load_configs` 解析 → `tests/api/test_dmd_config.py`；`max_cached_steps=0` 逐比特用例放 `tests/api/`。

1. **逐比特**：`max_cached_steps=0` + `residual_diff_threshold=0.0` → `assert_close(atol=0, rtol=0)`。多流模型每条流都断言。
2. **强制命中**：`residual_diff_threshold=1e9`（或 `>=1.0` 短路）→ 断言第二条流不是 `None` 也没被冻结。
3. **真命中**：`cached_steps > 0`，且 `accumulated_executed_steps` 能证明真跳过了计算。**只看 PSNR=inf 会假绿**（零缓存时也是 inf）。
4. **口径**：`calculate_psnr` 的 peak 固定 255（#1335 修的是 uint8 回绕，不是量纲），float[0,1] 仍会 +48.13 dB 虚高——要么自算（`data_range=1`），要么先乘 255；固定 seed。
5. **量化/并行组合**：只在确实要发版时才跑，且必须单独归因（compile 会顺带打开 TF32，VAE 也会变快）。
6. **与 DBCache 同 harness 对照**：新算法必须给出"同 FLOPs / 同 seed / 同 prompt 下 vs DBCache(同档位)"，否则数字无意义。

## 9. 已知坑

- `--warmup` 是 perf 预热，不是 cache warmup；cache warmup 是 `--cache-params ws=...`（默认 8）。少步模型配 `ws=8` 会全程 warmup ⇒ 零缓存。
- `rdt >= 1.0` 是"永远命中"的短路分支，不是"放宽阈值"，不能拿来证明 cache 有效。
- `--compile` 会改变命中序列（数值扰动变了决策），所以 compile 行与非 compile 行不能共用同一工作点。
- **`residual_diffs` 里有 sentinel**：`-0.0`（阈值关闭）/`-1.0`（永远命中）/`-2.0`（shape 不匹配）都会写进同一个 dict。统计 diff 均值/画曲线前必须过滤负值，否则得到无意义的数。
- 行号/默认值有 2 处文档与代码不一致，**一律以代码为准**：`max_warmup_steps` 文档说 0 实际 8；CLI `rdt` 默认 0.12 而 dataclass 默认 0.08。
