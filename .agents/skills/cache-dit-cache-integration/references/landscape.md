# 现有 cache 能力与已评论文的全景对照

本文件是"判有没有"的事实基础。数字全部标注来源，跨来源的数字**不可直接比较**（见 §4.3）。

> 论文主张 → cache-dit 对应物的映射表在 `SKILL.md` §2（不在本文件）；本文件 §2 是已评论文的算法骨架，§3 是逐维总表。2026 纯 calibrator/探针定向调研的完整版本见 `references/2026-cache-algorithms.md`。

## 1. cache-dit 现有的 cache 面（截至 2026-11-17）

### 1.1 算法枚举

| `CacheType` | 语义 | 跳过粒度 |
|---|---|---|
| `NONE` | 不缓存 | — |
| `DBCache` | Dual Block Cache：Fn 块算完 → 判 diff → 跳过 `[Fn, N-Bn)` → Bn 块算 | 块区间 |

`TaylorSeer / DMD / FoCa` **不是算法**，是 DBCache 的 forecast 插件（`calibrator_type` 三选一，互斥）。
`SCM` **不是算法**，是外部注入的步调度（`steps_computation_mask`，1=必算 / 0=可缓存）。

### 1.2 判定与状态

| 机制 | 位置 | 语义 |
|---|---|---|
| `can_cache()` | `caching/cache_contexts/cache_manager.py` | warmup → 步掩码 → `max_cached_steps` → `max_continuous_cached_steps` → 累积 diff 上限 → 阈值 |
| `similarity()` | 同上 | `mean\|t1−t2\| / mean\|t1\|`（相对 L1）；`threshold<=0` 永不算，`>=1` 永远算 |
| 累积守卫 | 同上 | `max_accumulated_residual_diff_threshold`（默认 `None`=关） |
| 上下文刷新 | 同上 + `cache_context.py` | 每次 `__call__` 重建；`num_inference_steps` 达标；`force_refresh_step_hint`（`once`/`repeat`） |
| 统计落盘 | `cache_blocks/pattern_utils.py` | `_cached_steps` / `_residual_diffs` 挂到模块上 |

### 1.3 三个 calibrator 的机制

| 预测器 | 数学 | 状态 | 是否随 horizon 演化 | 额外显存 |
|---|---|---|---|---|
| TaylorSeer | 有限差分阶梯 + `Σ (1/i!)·dⁱY·elapsedⁱ` | `dY_prev` / `dY_current` 两条阶梯 | 是（`elapsedⁱ`） | ≈2×order |
| DMD | 窗口快照拟合 `Y_{t+1}≈A Y_t`，`Y_{t+k}=Φ(λ^k⊙b)` | ≤6 份完整快照 | 是（`λ^k`） | **6×** |
| FoCa | 线性外推 `2*F_k − F_km1`（系数 (2, −1)）+ Heun 校正 | 4 张量 | 是 | 4× |

三者共同的边界（`dmd.py` 顶部注释写明的设计）：**只替换 `Bn_residual` 这一个操作数**，`H(t,Fn)` 永远现算，因此不改阈值、不改 diff 追踪、不改累积守卫。

### 1.4 DMD 与 SVD-Cache 的区别（最容易混淆的一对）

两者都叫 SVD，作用的轴完全不同：

| 维度 | `DMDCalibrator` | SVD-Cache |
|---|---|---|
| 本质 | 时间维动态模型（拟合 propagator） | 空间维坐标变换（拆 `F∈R^{N×D}`） |
| SVD 作用对象 | 快照矩阵 `[d, n]`，n≤5 是时间列 | 特征矩阵 `[N, D]`，token × channel |
| 秩的语义 | 时间模态数，被窗口长度卡在 ≤5 | 空间主成分数，τ=0.9 时 FLUX 上 5/3072 |
| 预测式 | `Y_{t+k}=Φ(λ^k⊙b)`，随 horizon 变 | `F̂ = EMA(F_k) + R`，**无 horizon 参数** |
| 类别 | 外推（可过冲/发散，有 `isfinite` 兜底） | order-0 复用 + 平滑（有界，但"停更"） |
| 拟合频率 | 每窗口 SVD+eig+lstsq（fp64/complex128） | 一次 basis，之后每步 2 个瘦 GEMM |
| 生效门槛 | ≥4 个**等间距**快照，否则回退 Taylor | 1 个 anchor + 一套 basis |
| 与调度 cadence | 强耦合（只取最长等间距后缀） | 无耦合 |
| 架构槽位 | 与 Taylor/FoCa 互斥的 predictor | 可**包住**任意 predictor |

## 2. 已评论文的算法骨架

### 2.1 HiCache（arXiv 2508.16984）

```
缓存侧: Δ^{k+1}F = (Δ^k F_new - Δ^k F_prev) / Δt_hist      （与 TaylorSeer 完全同式）
预测侧: F̂ = F₀ + Σ_{k=1..N} (1/k!) · H̃_k(Δt) · Δ^k F
        H̃_k(x) = σ^k H_k(σx)，H 为 physicist Hermite，递推 H_{k+1}=2xH_k-2kH_{k-1}
超参: N_interval（固定）、N_order ≤4、σ≈0.4~0.5
```

与 cache-dit 的差别：**数学上只差一行基函数**；工程上差"逐层 vs 块组"的粒度，与"固定 interval vs 自适应阈值"的调度。

**cache-dit 落地（2026-11-18，#1359）**：`calibrators/hicache.py`（`HiCacheState(TaylorSeerState)` + `HiCacheCalibrator(TaylorSeerCalibrator)`）；`taylorseer.py` 抽出 `_basis_term(i, elapsed)` 钩子、Taylor 路径数值逐比特不变，HiCache 只覆写该钩子 → 差分阶梯单份实现。`hicache_order`（= 论文 N_order，≤4，默认 2）、`hicache_sigma`（∈(0,1]，默认 0.5）。实现用等价递推 `H~_{k+1}=2σ²x·H~_k − 2kσ²·H~_{k-1}`、`H~_1=2σ²x`；**一阶增益是 2σ²**，σ 扫描扫的就是它（`σ=1/√2` 使一阶与 Taylor 等价）。σ=1 是**无阻尼 physicist Hermite**，不是 Taylor 的退化点。

### 2.2 MagCache（arXiv 2506.09045）

```
残差: r_t = v_θ(x_t,t) - x_t                 （x 空间，模型输出级）
幅度律: γ_t = mean_tokens(||r_t||₂ / ||r_{t-1}||₂)   ← 离线校准一次，per-model
误差: ε_skip(t̂,t) = 1 - Π γ_i ;  E_t = E_{t-1} + ε_skip
跳过: E_t ≤ δ 且 (t - t̂) ≤ K，否则重算并清零
保护: 前 20% 强制算
```

**必须离线校准**：决策发生在算之前，而 γ_{t+1} 需要 r_{t+1}；运行期无法自测（末段 γ 骤降正是最需要它的地方）。两个守卫与 cache-dit 的 `max_continuous_cached_steps` / `max_accumulated_residual_diff_threshold` 语义等价，差别只是"预测误差 vs 实测误差"。

### 2.3 SVD-Cache（arXiv 2601.07396）

```
离线（参考 prompt）: 只留 V_C ∈ R^{D×k}, σ_C ∈ R^k（σ 与 V 跨 prompt 不变）
每步: U = F V_C diag(σ_C)^{-1} ;  F_k = U_k diag(σ_C,k) V_{C,k}^T ;  R = F - F_k
      F̂_k = β F̂_k,prev + (1-β) F_k  (β=0.9)   # 主成分 EMA 平滑
      F̂  = F̂_k + R                            # 残差冻结复用
超参: τ(0.85~0.9) → k ; β=0.9 ; 固定 interval N
```

与 HiCache 打同一个靶子的两个方向：HiCache 换更好的基去打外推，SVD-Cache 直接不外推尾巴。可组合（论文 Fig 5(b)：SVD+Taylor > Taylor）。

**cache-dit 落地（2026-11-17）**：`SVDCalibrator` 两种 predictor 都实现了——`ema`（论文原式）与 `taylor`（Fig 5(b) 组合，差商阶梯建在 `[M,k]` 系数上）。实测秩 τ=0.9 → hidden 4–5/3072、encoder 2/512，与论文 k≈5 一致；τ=0.999 → k≈1100。数字与机理见 §4.4。

### 2.4 DiCache（arXiv 2508.17356）

```
首步全跑，R ← y - x
每步: 跑前 m 层 → y^m_t ;  Σerror += L1rel(y^m_t, y^m_{t+1})
      Σerror ≤ δ → y_t = x_t + R          （整步复用，探针结果被丢弃）
      else       → 从第 m+1 层续算，刷新 R，Σerror ← 0
DCTA（how）: γ̂ = L1rel(r^m_t, r^m_tβ) / L1rel(r^m_tα, r^m_tβ)
             r_t = r^{tβ} + γ̂ (r^{tα} - r^{tβ})
超参: m=1（三模型统一）、δ=0.1~0.4
```

与 DBCache 的对应：探针=`Fn`、`L1rel`=`similarity()`、`Σerror`=累积守卫、`resume from probed layer`=Fn 结构。**唯一新件是 DCTA**（双锚 + 实测距离比系数，零 SVD/零拟合/sample-adaptive）。

### 2.5 BRACE（arXiv 2608.07572，2026-08-04）

```
窗口: C≤3 个真实历史锚点 (F_tau, tau)，FIFO 更新
坐标: x(tau) = 2*(tau-tau_min)/(tau_max-tau_min)-1        # Eq.(6)
预测: F_pred = Σ_j [w_j/(x(t)-x(tau_j))] F_tau_j          # Eq.(7)
                 / Σ_j [w_j/(x(t)-x(tau_j))]
权重: adapted Chebyshev，交替符号 + 端点敏感度 γ（默认 0.7）  # Eq.(8), Table 5
```

**裁定：接，首个实验性 calibrator PoC；已实现（2026-11-19，`calibrators/brace.py`）。** 只移植预测式到 `Bn_residual`，保留 DBCache 的 when；无训练、无当前真值、无在线矩阵分解。实现按有效窗口长度取权重，预测不进入 FIFO，标量分母守卫带回退计数。FLUX 单点、12/28 缓存步的功能验收已通过，但尚未覆盖 >5x 与多骨干，不能由该点推断普遍收益。数字见 §4.7。

### 2.6 BeziCast（arXiv 2610.16448，2026-10-12）

```
窗口: 至多 K=6 个真实历史锚点，支持非均匀节点
预测: Bernstein 基矩阵 A；w=b(u)^T(A^T A+lambda I)^-1 A^T，再加权真实锚点
守卫: |sum(w)-1| + (L1(w)-|sum(w)|) → simplex 投影 + 软收缩
参数: K_max=6，n=4，lambda=0.5                              # Eq.(6)-(21)，§3.1 / A.3.2
```

**裁定：已实现（2026-11-19，`calibrators/bezicast.py`），纯预测器为默认。** cache-dit 默认 `guardrail=false,u_max=1.5`；`K<n` 时显式复用最近真实锚点并计数。guardrail 可选 `rho=auto`，但 FLUX 两轮实验未证明收益且 noise floor 更差。论文最高 4.79x latency；本实现尚缺 >5x 与多骨干证据。数字见 §4.8。

### 2.7 SeaCache（arXiv 2602.18993，2026-02-22）

```
滤波: P(G_t, I_t) = iFFT(G_t(f) ⊙ FFT(I_t))             # Eq.(6)
响应: G_t(f) = a_t S_x(f) / (a_t² S_x(f)+b_t²)         # Eq.(5)
      再做均值增益归一化                                  # Eq.(7)
探针: Δ~_t = RelL1(P(G_t,I_t), P(G_t+1,I_t+1))          # Eq.(8)
调度: 保留原有累积 relative L1 与阈值，不改缓存内容
```

**裁定：已实现（2026-11-19，`cache_contexts/seacache_uitls.py`），作为 DBCache 默认关闭的通用探针。** 不新增 CacheType、不改缓存对象，原 `similarity()` 保持不变；支持 flow/VP/显式 AB、dense/packed 几何和 CP/TP/CP+TP。未知 scheduler 或几何明确报错，不做静默退化。FLUX 单点与 CP2 只证明适配和接线，不证明 >5x 或普遍收益。数字见 §4.9。

### 2.8 RACER（arXiv 2608.01740，2026-08-03）

```
预测: direct 与 derivative-aware 两个历史预测器 + anchor forecast
可靠度: r_dir/r_der/r_int → r_max
what: 在 anchor 与 forecast 间收缩（shrink-only 可拆）
when: 高分歧触发 trust-region refresh；refresh-and-repay 不属最小接入
```

**裁定：探针可接做高加速 guard PoC；shrink-only 观察。** 不需当前 Fn 真值，但需短 trace 标定 O(1) 标量；仍须按闸 3 对 per-model 标定降权，不能默认豁免。`can_cache` 先于 `approximate`，所以必须新增显式只读 forecast/probe 路径。论文各表的高加速数字同时包含 shrink/refresh/repay，不可归因给单个组件。

## 3. 逐维总表

| 维度 | DBCache | HiCache | MagCache | SVD-Cache | DiCache |
|---|---|---|---|---|---|
| 跳过粒度 | 块区间 | **整步/逐层**（论文）→ 在 cache-dit 只能是块组 | 整步 | 块区间 | 整步 |
| 判定信号 | 运行期实测相对 L1 | 论文用固定间隔；接进 cache-dit 后复用阈值 | 离线 γ 曲线累积 | 复用阈值 | 运行期实测（浅层探针）累积 |
| 缓存内容 | `Bn_residual` / `hidden_states` | 逐层特征（论文）/ 块组张量（cache-dit 内） | 整步残差 `r_t` | 块边界 `F` 的主成分+残差 | 整步残差 `r_t = y-x` |
| 替代算子 | 直接复用 | Hermite 外推 | 直接复用 | 主成分 EMA + 尾巴冻结 | 直接复用 / DCTA 双锚 |
| 离线产物 | 无 | σ（调参，非文件） | **γ 曲线（per-model）** | **V_C/σ_C（per-model）** | 无 |
| 额外显存 | 1× 块组 | 逐层则数倍 / 块组则 ≈2×order | 1× 步输出 | 1× 块组 + D×k | 1× 步输出 + 1× 探针残差 |
| 额外算力 | 2 次 mean | 0 | 0 | 2 个瘦 GEMM | 2 次 mean（+m 层探针，已算） |
| 时间感知 | 有（elapsed） | 有 | 有 | **无** | 有（累积） |
| 论文覆盖的层 | ①②③ | ①②③（固定 interval + 逐层缓存 + 外推）；在 cache-dit 里落在 ③ | ①② | ③ | ①②（+③ DCTA） |

### 3.1 2026 年新候选裁定（完整报告见 `references/2026-cache-algorithms.md`）

| 候选 | 层与最小落点 | 状态/离线产物 | 裁定 |
|---|---|---|---|
| **BRACE** | ③；新 calibrator，预测 `Bn_residual` | C≤3 锚点；无训练 | **已实现；FLUX 单点功能验收，继续验证高加速档** |
| **BeziCast** | ③；新 calibrator + 预测内部 guardrail | K≤6 锚点 + 小型求解；无训练 | **已实现；纯预测器完成 FLUX 单点+重跑，guardrail 继续观察** |
| **SeaCache** | ①；替换 diff 度量，不改缓存对象 | fp32 FFT、scheduler 系数与几何元数据；无训练 | **已实现；通用 DBCache 可选探针，FLUX 单卡+CP2 验收** |
| **RACER** | ③ shrink；① disagreement guard | 双预测状态 + 短 trace 标定 | **探针接；shrink 观察** |
| GP-Refiner | ③ posterior correction；① posterior variance | h=10 历史对，高维 GP | **观察** |
| Spectrum | ③ 在线 Chebyshev ridge | 全历史/滑窗 + 主路径求解 | **观察** |
| HybridFF Adaptive | ③ 多基自适应融合 | 多预测器 + 协方差求解；Adaptive 无离线权重 | **观察** |
| ChebBooster / BaryCache / TC-Padé | ③ 同族或低阶有理预测 | 不同数量的历史锚点/拟合 | **观察；不与 P0 并列铺开** |
| L2P-Cache / LESA | ③ 学习预测器 | per-model 权重、轨迹或两阶段训练 | **观察离线 / 默认不接** |
| FADE | ③ 分数阶基 + ①阈值 | per-timestep/layer/module LUT | **默认不接** |
| PrediT / MeanCache | 需要当前 latent、额外求值或 JVP | 跨 pipeline/scheduler | **超范围** |
| Z-Cache / PoSTCache | 未来锚点、轨迹修订/重放 | 非 forward-only | **不接** |
| SpecKCast / BeCARE / AlignCache | 新输出状态、输入级控制或 CFG 双分支 | 跨层算法栈 | **观察，独立立项** |
| FeatFix | 付费 exact block 的本地替换 | 固定 layer–timestep 审计点 | **超范围** |
| SoftCache | ①信号的 headroom → scheduler 步长与尾部网格 | 无训练，但改采样网格 | **超范围；高潜力跨层路线** |

统一口径：“接”只表示值得做同 harness 实验性 PoC，不是已验证可合入。2026-11-19 共深读 24 篇原文、摘要级初排 6 篇；报告含逐项 A/B/C/D、首发日期与原始链接。

## 4. 数字

### 4.1 cache-dit 同 harness 基线（`docs/benchmark/HYBRID_CACHE.md`，FLUX.1-dev，DrawBench）

| 配置 | TFLOPs | 加速 | ImageReward | CLIP |
|---|---|---|---|---|
| FLUX 50 步 | 3726.87 | 1.00× | 0.9898 | 32.404 |
| DBCache(S) | 1400.08 | 2.66× | **1.0065** | 32.838 |
| DBCache(S)+TS | 1153.05 | 3.23× | **1.0221** | 32.819 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| DBCache(M) | 944.75 | 3.94× | 0.9997 | 32.849 |
| DBCache(M)+TS | 944.75 | 3.94× | 1.0107 | 32.865 |
| FoCa(N=5) | 893.54 | 4.16× | 1.0029 | 32.948 |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | **0.9128** | 32.128 |
| DBCache(F) | 651.90 | 5.72× | 0.9271 | 32.552 |
| **DBCache(F)+TS** | 651.90 | **5.72×** | **0.9526** | 32.568 |
| FoCa(N=8) | 596.07 | 6.24× | 0.9502 | 32.706 |
| DBCache(U)+TS | 505.47 | 7.37× | 0.8645 | 32.719 |

读法：**≤4x 已达 SOTA，>5x 明显掉队（0.9526 @5.72x）**——这就是"新算法必须有 >5x 增量"的由来。

### 4.2 各论文自述（**只能在自己列内比较**）

| 论文 | 设置 | 关键数字 |
|---|---|---|
| HiCache | FLUX，N=7,O=2 | Taylor IR 0.9572 → HiCache **0.9979**（同 670.44 TFLOPs）；N=9 时 0.8562 → 0.9113；σ 消融：σ=1.0 时 IR 仅 0.759（比 Taylor 更差） |
| SVD-Cache | FLUX | N=5 893.54T IR 1.0123；N=7 670.44T IR 0.9938；N=8 596.07T IR 0.9769。HunyuanVideo N=6 VBench 80.46 @5359T（原模型 80.66） |
| MagCache | Wan2.1 1.3B 50 步 | 2.14×/PSNR 23.42；2.68×/PSNR 21.54。HunyuanVideo 2.25×/34.51；Flux 2.57×/24.46、2.82×/21.35 |
| DiCache | WAN 2.1 / HunyuanVideo / Flux | 2.45×/26.45dB/0.8885；2.34×/32.79dB/0.9396；3.22×/22.39dB/0.8211；DCTA 同加速 +0.81 dB；TaylorSeer 在 HunyuanVideo **OOM** |

### 4.3 为什么这些数字不能横向比

- 同一 TaylorSeer(N=7,O=2)@670.44 TFLOPs 的 ImageReward：**0.9572 / 0.9698 / 0.9128**（三个来源），极差 0.057 > 各论文宣称的增益（+0.02~0.05）。
- MagCache 与 DiCache 在 HunyuanVideo 上互相矛盾：2.25× 报 34.51 dB vs 2.34× 报 32.79 dB（prompt 集 100 vs 300 条）。
- 同算力档（2.66× vs 2.82×，同属 2.7 档）：块级复用（DBCache-S）PSNR 31.53 vs 整步复用（MagCache-Flux）21.35——10 dB 的差距**远超**任何 harness 噪声，反映的是方法学差异（整步复用丢掉全部 block 贡献）。
- 结论：**判定只能靠自己在固定 seed / 固定 prompt / raw float 下跑 A/B。**

### 4.4 SVD-Cache 在 cache-dit 内的同 harness 实测（2026-11-17，FLUX.1-dev 1024²，28 步，seed 0，2 prompt，RTX PRO 5000，torch 2.15.0）

行 = `--cache --ws 1`，psnr 用 `cache-dit-metrics --psnr-type skimage`（uint8 PNG 口径），全部 warmup 后计时：

| 工作点 | 配置 | latency | 加速 | PSNR(dB) | SSIM |
|---|---|---|---|---|---|
| rdt 0.12，14/28 命中 | DBCache（reuse） | 8.85s | 1.94x | 23.11 | 0.8848 |
| 同上 | + TaylorSeer | 8.86s | 1.94x | **24.21** | **0.9123** |
| 同上 | + DMD | 8.91s | 1.93x | 23.49 | 0.9074 |
| 同上 | + SVD `ema` β0.9 | 9.48s | 1.81x | 20.62 | 0.8051 |
| 同上 | + SVD `taylor` o1 | 9.47s | 1.81x | 23.38 | 0.8882 |
| 同上 | + SVD `taylor` o1，τ0.999（k≈1100） | 9.51s | 1.81x | 23.81 | 0.8990 |
| rdt 0.6+mcc6，23/28 命中 | DBCache（reuse） | 3.48s | 4.94x | 15.95 | 0.6023 |
| 同上 | + TaylorSeer | 3.48s | 4.94x | 15.63 | **0.6081** |
| 同上 | + SVD `ema` β0.9 / β0.5 | 4.13s | 4.16x | 15.04 / 15.89 | 0.5446 / 0.6006 |
| 同上 | + SVD `taylor` o1 / o2 | 4.12s | 4.16x | 15.56 / 15.71 | 0.5881 / 0.5832 |

三条可复用的判读：

1. **`ema` 结构上不可能赢 reuse。** `EMA(P)+R` 是「上一个锚点主成分」与「当前锚点主成分」的凸组合，落不到真实锚点之外，`β→0` 精确退化为 reuse（4.94x 档实测 15.89 vs 15.95）。凡是「平滑主成分 + 冻结残差」的变体，都可以先按这条判死，不必跑满矩阵。
2. **`taylor` 是唯一有前向外推的分支**，但也只追回大部分差距（23.38 vs reuse 23.11、TaylorSeer 24.21），代价是 +0.6s/图（basis SVD + 每跳过步一次 `[M,k]@[k,D]` GEMM），而全张量 TaylorSeer 的额外成本是 O(1)。
3. **损失来自冻结残差，不是 predictor。** 把 τ 从 0.9（k=5，残差占 ~10% 能量）抬到 0.999（k≈1100，残差≈0）就补回 +0.43dB（23.38→23.81）；剩下与 TaylorSeer 的 0.4dB 差距来自子空间截断本身。机理：**高频尾巴是逐步变化最快的分量**，把它冻结等于每步都注入一个过期误差——与论文「残差变化慢」的前提相反。

→ 生产建议：`--calibrator taylorseer`（或 `dmd`）。SVD-Cache 保留为复现/实验能力，`svd.py` 顶部注释与 `docs/user_guide/CACHE_API.md` 都写明了这张表与机理。

### 4.5 HiCache 在 cache-dit 内的同 harness 实测（2026-11-18，FLUX.1-dev 1024²，28 步，seed 0，rdt 0.12，RTX PRO 5000，torch 2.15.0，单 prompt）

12/28 步被缓存，且**7 行命中位置完全相同**（`[8,9,11,12,14,15,16,18,19,21,23,25]`），所以 PSNR 是同一调度下的可比对照：

| 配置 | PSNR | SSIM | Latency |
|---|---|---|---|
| base（无 cache） | — | — | 16.91s |
| DBCache reuse | 31.72 | 0.9706 | 9.88s |
| + TaylorSeer o2 | 30.91 | 0.9733 | 9.88s |
| + HiCache o2 σ=0.4 | 33.10 | 0.9765 | 9.89s |
| **+ HiCache o2 σ=0.5** | **33.33** | **0.9781** | 9.88s |
| + HiCache o1 σ=0.5 | 33.36 | 0.9779 | 9.91s |
| + HiCache o2 σ=1/√2 | 32.44 | 0.9770 | 9.88s |
| + HiCache o2 σ=1.0 | 24.99 | 0.9280 | 9.89s |
| + HiCache o2 σ=0.5 + compile | 27.53 / 28.24 | 0.9639 / 0.9668 | 8.69 / 8.66s |
| + TaylorSeer o2（mask 钉死） | 27.77 | 0.9623 | 11.67s |
| + HiCache o2 σ=0.5（mask 钉死） | 29.36 | 0.9738 | 11.68s |

三条可复用判读：

1. **1.7× 档 HiCache 是三者最好的**（+1.61dB vs reuse、+2.42dB vs TS-o2），与「Bn_residual 替换层已饱和」的旧结论冲突——那条结论必须按档位限定（SVD-Cache 的饱和结论是在 1.9×/4.9× 上、且针对子空间拆分那一族）。
2. **σ=1 崩到 24.99dB，精确复现论文 σ 消融的方向**（无阻尼 Hermite 比 Taylor 更差）。这是"基真的接上了"最便宜的证伪点：任何"σ 没接线 / 退化成 Taylor"的实现都不会出现这个断崖。
3. **调度钉死后排序不变**（mask static，9 步：HiCache 29.36 > TS 27.77），说明收益来自基函数本身，不是命中序列差异。1.7× 档一阶已够（o1 33.36 ≈ o2 33.33），更高阶在这个 work point 无增量。

**测试坑**：`tests/api/utils.py` 的 `RandPipeline` 每步喂**同一个** `hidden_states`，特征轨迹恒定 ⇒ 任何 predictor 的预测恒等于锚点，**"输出不同"不能用来证明接线**；应 monkeypatch 计数 `approximate()` 调用，并断言 state 具体类型与已接线的 σ。

### 4.6 2026 候选的论文自述（BRACE 的 cache-dit 单点实测另见 §4.7）

| 论文 | 来源内证据 | 不能据此推出的结论 |
|---|---|---|
| BRACE | FLUX 展示约 5.55x FLOPs；HunyuanVideo N=6 | 不能推出动态 DBCache 下同档增益 |
| BeziCast | 多骨干最高 4.79x latency | 缺 >5x 与 DBCache 对照 |
| SeaCache | FLUX/HunyuanVideo/Wan2.1；约 30%/50% 刷新预算；DiCache+SEA 消融；官方称滤波约 0.2% 总延迟 | 不能推出 Fn residual 或 >5x 已验证；0.2% 不是 cache-dit 实测 |
| RACER | 各表含 5.09x、8.77x、9.49x 等不同口径的高档 paired 实验 | 整套收益不可归因给 shrink 或 disagreement 单件 |
| GP-Refiner | FLUX NFE=12/16 为 6.24x/5.06x FLOPs，PSNR 30.230/30.457，相对 reuse +0.615/+0.489dB | 不能忽略高维 GP 的墙钟、显存与 compile 成本 |
| BeCARE | FLUX 5.94x、Wan2.1 4.80x、HunyuanVideo 5.85x wall-clock | 改到 Fn 输出后不能继承输入级探针结果 |
| L2P-Cache | FLUX 最高 6.03x latency/7.14x FLOPs；Qwen 最高 7.18x FLOPs | 需要 per-model/per-total-steps 权重，不是零配置路径 |

后续 PoC 必须固定 seed、固定 Fn/Bn 与全部 guards；calibrator A/B 保持相同 `Cache Steps`，探针 A/B 固定缓存内容；同时报告 wall-clock、实际 Cache Steps、额外显存和正确量纲的 PSNR/SSIM/LPIPS。最小实验矩阵见 `2026-cache-algorithms.md` §7。

### 4.7 BRACE cache-dit 单点功能验收（2026-11-19）

FLUX.1-dev 1024²、28 步、seed 0、单 prompt、RTX PRO 5000（物理卡 2）、torch 2.15.0、compile+F1B0+rdt 0.12。四臂均为 **12/28 Cache Steps**，BRACE 无分母回退告警；244 项相关回归通过。

| 臂 | 秒 | GiB | PSNR/SSIM vs compile | PSNR/SSIM vs eager base |
|---|---:|---:|---:|---:|
| reuse | 8.65 | 33.87 | 29.03 / 0.9614 | 29.81 / 0.9669 |
| TaylorSeer o1 | 8.66 | 33.95 | 31.52 / 0.9740 | 27.95 / 0.9675 |
| BRACE C2 γ0.7 | 8.65 | 33.90 | 29.57 / 0.9720 | **31.21 / 0.9744** |
| BRACE C3 γ0.7 | 8.66 | 33.92 | **30.75 / 0.9743** | 29.01 / 0.9695 |

结论仅限功能与接线：BRACE 没有可测的额外延迟，两个窗口都正常出图；参考系改变会改变 PSNR 排名，且单 prompt+compile 不构成普遍优劣证据。>5x、多 prompt、多骨干和 noise floor 仍未覆盖。完整记录：`cache-dit/.tmp/research/brace_analysis.md`。

### 4.8 BeziCast cache-dit 单点功能验收（2026-11-19）

FLUX.1-dev 1024²、28 步、seed 0、物理卡 2、compile+F1B0+rdt0.12。所有缓存臂均为 **12/28 Cache Steps**，无数值回退告警；321 项相关回归通过。注意：本节是新一轮独立运行，不能与 §4.7 的跨轮数字直接比较绝对 PSNR。

| 臂 | 秒 | GiB | PSNR/SSIM vs compile | PSNR/SSIM vs eager base |
|---|---:|---:|---:|---:|
| reuse | 8.65 | 33.87 | 29.61 / 0.9634 | 28.11 / 0.9617 |
| TaylorSeer o1 | 8.67 | 33.95 | 30.60 / 0.9698 | 27.60 / 0.9643 |
| BRACE C3 | 8.65 | 33.92 | 29.42 / 0.9699 | 27.97 / 0.9698 |
| BeziCast guard off | 8.67 | 34.00 | 30.56 / 0.9670 | 29.31 / 0.9707 |
| BeziCast guard on，rho=auto | 8.68 | 34.00 | 30.46 / 0.9650 | 29.27 / 0.9663 |

同配置重跑：guard off 为 **39.60dB/0.9932**，guard on 为 **32.58dB/0.9873**；第二轮 vs compile 分别 30.90/28.84dB。纯预测器额外延迟约 0.02s、显存约 0.13GiB；guardrail 未显示一致收益且稳定性更差，因此默认关闭。未覆盖 K=3、>5x、多 prompt/骨干，不能宣称普遍优于 BRACE/Taylor。完整记录：`cache-dit/.tmp/research/bezicast_analysis.md`。

### 4.9 SeaCache cache-dit 通用与分布式验收（2026-11-19）

SeaCache 落在 DBCache 的可选 when 探针：`DBCacheConfig(enable_seacache=True)` 或 `--seacache`，不新增 CacheType。原 `similarity()` 函数体不变，SGLang 23 项集成测试通过；351 项相关回归、8 项行为探针及 pre-commit 通过。CP 等长/非等长、TP、CP+TP 和 CP+CFG 均以全量参照验证。

FLUX.1-dev 1024²、28 步、seed 0、compile+F1B0+warmup8+MCC3：

| 臂 | 秒 / GiB | Cache Steps | PSNR/SSIM vs compile | PSNR/SSIM vs base |
|---|---:|---:|---:|---:|
| reuse，rdt0.12 | 8.67 / 33.87 | 12 | 28.09 / 0.9643 | 27.95 / 0.9620 |
| SeaCache，rdt0.12 | 11.77 / 33.89 | 6 | 29.52 / 0.9749 | 29.03 / 0.9700 |
| SeaCache，rdt0.24 | 8.19 / 33.89 | 13 | 27.80 / 0.9576 | 27.52 / 0.9534 |
| SeaCache，rdt0.24 重跑 | 8.18 / 33.89 | 13 | 28.89 / 0.9575 | 28.56 / 0.9537 |

rdt0.24 同配置重跑为 **33.46dB/0.9887**。同 rdt0.12 时 SeaCache 命中减半且慢于 reuse；rdt0.24 达到相近命中数，但 compile 波动下单 prompt 不构成速度/质量排名。CP2：reuse 5.77s/12 命中，SeaCache 7.81s/6 命中，二者 PSNR/SSIM 31.37/0.9728，单卡与 CP2 的 SeaCache diff 分布一致。所有图片正常。未覆盖 Wan、>5x、多 prompt 或通用最优阈值。完整记录：`cache-dit/.tmp/research/seacache_analysis.md`。

## 5. 复核用的原始出处

- cache-dit：`src/cache_dit/caching/{cache_types.py,cache_contexts/,cache_blocks/,cache_adapters/}`、`cli/utils.py`、`docs/benchmark/HYBRID_CACHE.md`、`docs/user_guide/CACHE_API.md`
- HiCache：Algorithm 1/2/3（p.5、A.2）、Eq.(10)、Table 1（FLUX）、Table 4（σ 消融）、A.7 Listing 1
- MagCache：Eq.(3)(4)(6)(7)(8)(9)(10)(11)、Sec 3.3、Table 1/2/3、A.3/A.4
- SVD-Cache：Eq.(4)-(11)、Sec 3.3-3.5、Table 1/2/3、Fig 5(b)(c)(d)
- DiCache：Eq.(3)(4)(5)(6)、Algorithm 1（p.6）、Eq.(7)-(11)、Fig 3(d)（m=1~3 相关 0.8）、Table 1/2
- 2026 定向调研：`references/2026-cache-algorithms.md`（24 篇原文深读 + 6 篇摘要级初排；含 arXiv v1 日期、式/表号和原始链接）。BRACE Eq.(4)-(8)、BeziCast Eq.(3)-(14)、SeaCache Eq.(5)-(8)/Fig.9、RACER Eq.(9)-(13)/Algorithm 1、GP-Refiner Eq.(2)-(7)/Tables 1-4。
