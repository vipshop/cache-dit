---
name: cache-dit-cache-integration
description: "Decide whether a new feature-caching / cache-then-forecast paper is worth integrating into cache-dit, and if so which extension seam it lands on (DBCache / CalibratorBase / steps_computation_mask). Use when evaluating a caching paper (TaylorSeer, FoCa, DMD, HiCache, MagCache, SVD-Cache, DiCache, TeaCache, EasyCache, Delta-DiT class), when asked 'is X worth integrating', when triaging a whole cache-paper list, when adding a new skip policy or feature predictor, or when reviewing such a PR for the right seam and the right acceptance gate. 触发词：cache 算法调研、缓存论文对比、要不要集成 cache 算法、新增 calibrator、新增 CacheType、跳过策略、cache 集成优先级、Awesome-DIT-Cache."
user-invocable: true
---

# 新 cache 算法的集成裁决

只回答一个问题：**这篇 cache 论文是否值得进 cache-dit；值得的话落在哪个槽位。**

## ROUTE FIRST

| 你在做什么 | 读哪里 |
|---|---|
| 判"要不要接" | 本文件 §0-§4（交付按 §6/§8） |
| 已知要接，找代码落点 | `references/extension-points.md` |
| 要现有算法 / 已评论文的完整对比与数字 | `references/landscape.md` |
| 查 2026 纯 calibrator / 探针调研 | `references/2026-cache-algorithms.md`（BRACE、BeziCast、SeaCache、RACER 等 30 项） |
| 查 2026 调研原始长报告归档 | `references/recent_dit_cache_algorithms_analysis.md` |
| 手上有新 paper，要逐条过 | `references/paper-triage.md` |
| 要找候选论文 | 本文件 §7 |

## 0. 三条铁律

1. **先定位层次，再谈优劣。** cache-dit 的 cache 只有三层，任何算法必须先落到其中一层，否则讨论没有意义：

   | 层 | 问题 | 现有实现 | 可插拔 |
   |---|---|---|---|
   | ① when | 这一步要不要算 | `CachedContextManager.can_cache()`：warmup → 步掩码 → `max_cached_steps` → `max_continuous_cached_steps` → 累积 diff 上限 → 运行期相对 L1 阈值 | 否 |
   | ② how much | 一次跳过哪些计算 | **只能是块区间** `[Fn, N-Bn)`，且 `Fn >= 1`（`can_cache` 在 `call_Fn_blocks` 之后调用） | 否 |
   | ③ what | 被跳过的部分拿什么填 | 默认复用 `Bn_residual`；可选 `CalibratorBase`（TaylorSeer / DMD / FoCa），**只换左操作数** | 是 |

2. **跨论文数字不可比，必须同 harness 复现。** 硬证据：同一方法同一算力（TaylorSeer N=7,O=2 @ 670.44 TFLOPs）的 ImageReward，在 HiCache 论文是 **0.9572**、SVD-Cache 论文是 **0.9698**、cache-dit 自己的 bench 是 **0.9128**，极差 0.057——而三篇各自宣称的增益只有 +0.02~0.05。凡是"比 X 好 3 个点"的说法，先问是不是同 harness、同 prompt、同 seed、同 FLOPs。

3. **默认答案是"不接"。** cache-dit 单人维护，任何新增都要付线性维护面。只有当收益落在**真短板**（>5x 区间）或能**显著降低**维护面时才做；否则把结论写进记录、不要写代码。

## 1. 五道闸（顺序不可跳）

1. **它打的那一层 cache-dit 有没有？** 对照 §2。若已被覆盖（尤其"判定信号"层），到此为止。
2. **收益是否在 cache-dit 的短板区间？** ≤4x 区间 DBCache 已经是 SOTA（同 harness：2.66x/IR 1.0065、3.23x/1.0221、3.94x/1.0107）。真正弱的是 **>5x**（`DBCache(F)+TS` 5.72x 只有 IR 0.9526）。只在 2–3.5x 上做文章的论文，价值必然低。
3. **是否需要 per-model 离线产物？**（标定曲线、basis、拟合系数文件）→ 需要则降权：cache-dit 目前**零** per-model 产物，这是它相对论文方法的结构性优势。**豁免条件**（三条同时满足才豁免）：① 产物跨 prompt 不变（能在任意一次推理里现算，不必随模型发布）；② 体积小到 O(D·k) 级别；③ 收益确实在 >5x 档。SVD-Cache 的 basis 属可豁免类，MagCache 的 γ 曲线不属（跨模型必须重测）。
4. **能否落在 `CalibratorBase`？** 能 → 2~4 个文件；不能 → 至少 10 处改动 + 两份 pattern 实现（见 `references/extension-points.md`），先问值不值。
5. **验收证据链是否可给？** 必须能出 raw-float 逐比特、`cached_steps > 0`、实际执行 block 数三列。给不出就不可验收，等同不可集成（细则见 §6）。

## 2. 论文常打的东西 → cache-dit 对应物

判"有没有"就查这张表，不要凭印象：

| 论文里的主张 | cache-dit 对应物 | 判读 |
|---|---|---|
| 用前 `m` 层做探针 / 用前缀块算 diff | `Fn_compute_blocks`（默认 8，CLI 默认 1） | **已有**；`Fn >= 1` 是硬约束，`m=0` 不可能 |
| 相对 L1 / 余弦 / 幅度比 度量相似度 | `similarity()`：`mean\|t1−t2\| / mean\|t1\|` | **已有**；`important_condition_threshold` 可做 per-token |
| 累积误差 / 最大连续跳过步数 | `max_accumulated_residual_diff_threshold`、`max_continuous_cached_steps` | **已有**（前者默认关） |
| 固定/预定义跳过调度（LeMiCa / MagCache 曲线 / 前 20% 保护） | `steps_computation_mask` + `steps_computation_policy`（`steps_mask()` 支持显式 bins 与 slow/medium/fast/ultra） | **已有**，缺的只是"从曲线生成 mask" |
| 用历史多步做外推 | `CalibratorBase`：TaylorSeer / DMD / FoCa | **已有三个**，第 4 个要先证明增量 |
| 整步跳过、复用整步输出 | 无（`Fn >= 1`） | **真缺口**，但见 §3 红旗 |
| 缓存 attention states / KV | 无 | 真缺口，但属另一套系统，别混进 calibrator |
| 每层独立缓存特征 | 无（只缓存块组边界张量） | 结构性差异：cache-dit 更省显存，但表达不了逐层策略 |

## 3. 红旗：出现即倾向"不必集成"

- **判定信号是离线先验**（TeaCache 多项式、MagCache 的 γ 曲线）→ cache-dit 是**实测**，信息上严格占优，还要额外背 per-model 产物。
- **只在 ≤4x 档有收益**（DiCache 2.3~3.2x）→ 打的是 cache-dit 的舒适区。
- **整步跳过且丢掉前缀块输出**（DiCache 的 `y = x + R`）→ 相对 DBCache 的 `H(Fn)_t + R_Bn` 是**降级**：每步花同样的前缀算力，却丢掉了当前步信息。
- **要求 per-model 权重 / 曲线 / basis 文件**。
- **声称的增益 ≤0.03 ImageReward 或 ≤0.5 dB PSNR** → 落在跨论文噪声内，除非同 harness 复现。
- **"越加速指标越好"**（加速后 ImageReward 反超未加速基线）→ 通常是指标噪声或选样，不要当证据。
- **只在单一模型单一份额上验证**，且没有和 DBCache 同 harness 对照。

## 4. 已判例（不要重新评估；新增判例按 §8 追加）

| 论文 / 方案 | 裁定 | 一句话理由 |
|---|---|---|
| **BRACE**（barycentric rational forecast，2026） | **已实现（2026-11-19，`calibrators/brace.py`）；完成 FLUX 单点功能验收，仍需高加速/多骨干验证** | C≤3 真实锚点 FIFO、闭式权重、只换 `Bn_residual`；单点 compile 对照中 C2/C3 PSNR 29.57/30.75dB，reuse 29.03、TaylorSeer 31.52，不能据此宣称普遍优于现有预测器。见 `landscape.md` §4.7 |
| **BeziCast**（Tikhonov Bernstein/Bézier 等价权重 + guardrail，2026） | **已实现（2026-11-19，`calibrators/bezicast.py`）；纯预测器完成 FLUX 单点+重跑，guardrail 默认关闭** | 论文默认 K=6/n=4/λ=0.5；cache-dit 单点 guard off/on 相对 compile 为 30.56/30.46dB，重跑 noise floor 39.60/32.58dB，guardrail 尚无稳定收益。见 `landscape.md` §4.8 |
| **SeaCache**（SEA-filtered relative L1，2026） | **已实现（2026-11-19，`cache_contexts/seacache_uitls.py`）；通用 DBCache 可选探针，完成 FLUX 单卡+CP2 验收** | 不改 CacheType/缓存对象，原 similarity 不变；支持 flow/VP/AB、packed/dense、CP/TP/CP+TP。351 项回归与 SGLang 23 项通过；单点不构成普遍收益结论。见 `landscape.md` §4.9 |
| **RACER**（双预测分歧 + trust region，2026） | **探针接做高加速 guard PoC；shrink-only 观察** | 无当前真值，但需短 trace 标定 O(1) 标量与显式只读 probe；per-model 标定仍按闸 3 降权。整套收益不可归因给 shrink/refresh/repay 任一单件。见 `landscape.md` §2.8 |
| **2026 其余候选** | **按完整报告裁定，不重复调研** | GP-Refiner/Spectrum/HybridFF 观察；L2P/LESA/FADE 默认不走通用零产物路线；SoftCache/FeatFix/SpecKCast 属跨层。24 篇深读 + 6 篇初排见 `references/2026-cache-algorithms.md` |
| **HiCache**（Hermite 基） | **已实施（2026-11-18，`calibrators/hicache.py`，#1359）** | 与 `TaylorSeerCalibrator` 数学同构，只差基；落点定为 `taylorseer.py` 的 `_basis_term()` 钩子 + 独立文件（阶梯单份）。FLUX 1.7× 同 harness：HiCache 33.33dB > reuse 31.72 > TS-o2 30.91，σ=1 崩到 24.99（复现论文消融）。σ 仍 per-model 必调；1.7× 档一阶已够。数字见 `references/landscape.md` §4.5 |
| **SVD-Cache**（子空间拆分） | **v1 已实施（2026-11-17，负结果，不建议生产用）** | `SVDCalibrator` 已接入（`calibrators/svd.py`，两种 predictor：`ema` / `taylor`，在线 basis）。FLUX.1-dev 1024² 28 步同 harness 实测：1.9x 档 reuse 23.11dB / `ema` 20.62dB / `taylor` 23.38dB / TaylorSeer 24.21dB；4.9x 档 15.95 / 15.56（o1） / 15.63dB。结论：`ema` 是两锚点凸组合、只能后向平滑；`taylor` 才是论文 Fig 5(b) 那支，能追回大部分但**仍不如全张量 TaylorSeer**，且多 ~0.6s/图；损失来自**冻结残差**（τ 0.9→0.999、k 5→1100 只补回 +0.43dB），不是 predictor 本身。详见 `references/landscape.md` §4.4 |
| **MagCache**（幅度律 + 整步跳） | **不接**；只在"固定调度更好"被验证后考虑给 `steps_mask` 加一个 policy | 两个守卫 cache-dit 已有；同算力整步复用比块级复用差 ~10 dB；需 per-model γ 曲线，且运行期无法自测 |
| **DiCache**（探针 + 整步跳 + DCTA） | **不接**（2026-11-17 裁定） | "when" 与 DBCache 逐项同源；档位在舒适区；整步复用丢当前步信息属降级；DCTA 需改 `approximate()` 签名穿主路径，收益不抵成本 |
| **TaylorSeer / FoCa / DMD** | 已实现 | 三者互斥，`calibrator_type` 三选一 |
| **DBCache** | 已实现 | 主干 |

## 5. 落点速查（细节见 `references/extension-points.md`）

按改动面从小到大，**优先选上面的**：

1. **只换预测器** → 新 `calibrators/<name>.py` + 工厂注册 + CLI flag。`CalibratorConfig.calibrator_kwargs` 能透传参数，**不需要新 dataclass**。
2. **换基 / 换系数** → 扩展现有 calibrator（HiCache = Taylor 的 `basis` 开关；把 FoCa 的常数系数换成数据驱动系数，仍是同一类改动）。不新增 `calibrator_type`，维护面零增长。**例外**：用户明确要求独立文件/独立算法名时按方案 1 走，但基函数仍必须做成基类钩子（`_basis_term`）而不是复制累加循环，见 HiCache 先例。
3. **换调度** → 给 `steps_mask(mask_policy=...)` 加一个预设/生成分支（`cache_interface.py`）+ CLI flag（`cli/utils.py`），运行时配合 `steps_computation_policy="static"`。2 个文件，不动跳过粒度。
4. **换跳过粒度** → 新增 `CacheType`：enum / config / load_configs / ContextManager / context / blocks `__new__` / **两份 pattern** / stats / summary / CLI，约 10 处。**最后手段**。

## 6. 验收纪律（不可协商）

- **PSNR 量纲**：`calculate_psnr` 的 peak 固定为 255（#1335 只修了 uint8 模减法回绕，**没有**改量纲假设），喂 float[0,1] 仍旧白送 +48.13 dB。要么自算（`data_range=1`），要么先把输入乘 255。
- **固定 seed**（默认 0），同配置重复跑必须先得到逐比特一致，才谈跨配置比较。
- **`cached_steps > 0` + executed steps** 两列必给，否则"零缓存"也能全绿。
- **`max_cached_steps=0` → 逐比特**回归用例，每个新预测器/策略都要有。
- **与 `--compile` 组合 run-to-run 不收敛**，只能报区间，不能与其他行共用同一 `rdt` 工作点。
- **不许跨模型搬参照物**：不同模型/分辨率/指标各是各的尺。

## 7. 找候选论文（外部索引）

[`leeguandong/Awesome-DIT-Cache`](https://github.com/leeguandong/Awesome-DIT-Cache)（2026-11-19 实测默认分支为 `main`；README 自述最近核验 2026-11-15）是中文索引，章节结构与本 skill 的三层模型可以对齐：

| 仓库章节 | 对应本 skill 的层 | 用法 |
|---|---|---|
| §2 按缓存/复用粒度（2.1 Step / 2.2 Block / 2.3 Attention / 2.5 Fine-Grained / 2.8 Residual） | ② how much | 判断一篇论文的粒度是不是 cache-dit 表达不了的 |
| §3 按调度策略（3.1 Static / 3.2 Timestep / 3.3 Layer / 3.4 Predictive / 3.6 Frequency / 3.9 Hybrid） | ① when、③ what | `3.4 Predictive` 是 calibrator 候选池；`3.1/3.2` 是 `steps_mask` 候选池 |
| §2.9 粒度 × 调度交叉矩阵 | — | 找"空白格"= 尚未被做过的组合 |
| §1.1 方法全景表 / §1.2 演化时间线 | — | 快速看某个名字是否已被收录、属于第几代 |
| §4 测评（指标/基线/benchmark） | — | 对齐评测口径，避免跨 harness 比较 |
| §5 工程、系统与硬件 | — | 服务级/跨请求复用，**不属于** cache-dit 的范围 |

注意该仓库的收录边界：不含纯量化、纯蒸馏、纯 attention kernel、纯并行推理——正好是 cache-dit 已有的那几块。筛完后剩下的才是 cache 候选。

## 8. 交付与落盘

**结论模板只有一份**：`references/paper-triage.md` §5（A/B/C/D 判据 + 落点 + 裁定 + "若接的第一步证伪实验"）。填不满就说明没查完。

落盘分两处，**都要写**：

| 去向 | 路径 | 性质 |
|---|---|---|
| 草稿 | `cache-dit/.tmp/research/<topic>_analysis.md`（扁平 snake_case，不建日期子目录；`.tmp/` 被 gitignore） | 可丢弃 |
| **版本化** | 本 skill 的 `references/landscape.md`：§2 补算法骨架、§3 补候选裁定、§4 补来源内数字；完整长报告可另存 `references/<year-or-topic>.md` 并加入 ROUTE FIRST | 随 PR 提交，是下次评估的第一手依据 |

只写 `.tmp` 而不回写 `landscape.md`，结论会跟临时目录一起消失——这是最容易漏的一步。
