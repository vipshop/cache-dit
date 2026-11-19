# 2026 年 DiT Cache 算法集成调研：纯 Calibrator 与在线探针

- 调研日期：2026-11-19
- 论文范围：仅纳入首次公开时间在 2026-01-01 至 2026-11-19 的工作；2025 年首发、2026 年更新或录用的论文不作为新候选。
- 代码范围：只读核对 cache-dit 当前接口；调研时未修改或运行 cache-dit 代码，未执行 benchmark、模型推理或 GPU 实验。
- 证据范围：深读 24 篇论文的方法、公式和实验原文；另有 6 篇仅经摘要或官方页面初排。候选来自 Awesome-DIT-Cache 主表及两份 2026 增量记录、arXiv 原文、代表论文引用扩散和官方代码仓库。
- 覆盖声明：本报告是针对“纯预测器”和“在线探针”的定向筛查，不是对 2026 年全部 diffusion cache 论文的穷尽综述。所有深读候选均列在第 8 节；摘要级排除项单列，不参与强推荐。

## 1. 结论摘要

### 1.1 最值得继续验证的候选

| 方向 | 排名 | 候选 | 裁定 | 推荐的最小接入形态 | 主要理由 |
|---|---:|---|---|---|---|
| 纯 calibrator | 1 | **BRACE** | **接：首个实验性 PoC** | 只移植第二 barycentric 形式与 adapted Chebyshev 权重，预测 `Bn_residual`；保留 DBCache 的 when 与全部守卫 | 无需训练、无需当前步真值、通常只存 3 个历史锚点；闭式标量权重，无在线矩阵分解；论文覆盖到 5.55x FLOPs 档 |
| 纯 calibrator | 2 | **BeziCast predictor + guardrail** | **接：第二个实验性 PoC** | 历史锚点的局部 Lagrange 基函数加权和；guardrail 只收缩预测，不改变缓存命中 | 原生支持非均匀历史节点和任意预测距离；无需 per-model 权重；自带不依赖当前真值的过度外推检测 |
| 纯 calibrator | 3 | **RACER shrink-only** | **观察：可做组合式 PoC** | 两个历史预测器 + anchor 预测 + 风险系数收缩；不移植 refresh-and-repay | 可把现有 Taylor/HiCache 与 Spectrum 组合起来；但收益依赖双预测器及离线标定，不能只凭整套 RACER 的高加速数字归因给 shrink |
| 纯 calibrator | 4 | **GP-Refiner** | **观察：高潜力、成本待证** | 在任意基础预测器外做 posterior-mean correction；方差探针另行实验 | 固定调度消融在约 5–6x FLOPs 档相对 reuse 有 0.49–0.62 dB 增益；但高维特征上的 10 点 GP 有显存、kernel 与求解成本 |
| 探针 | 1 | **SeaCache / SEA-filtered relative L1** | **接：优先做探针 PoC** | 只替换 DBCache 的 diff 度量，保留累积阈值和缓存内容；新增 scheduler 系数与空间几何信息 | 与现有探针完全同层，信息增量明确；有官方代码、DiCache 中间特征适配实验和跨图像/视频验证 |
| 探针 | 2 | **RACER forecast-disagreement + trust-region** | **接：高加速 guard PoC** | 在 `can_cache` 前增加只读的双预测可靠度信号；不改变缓存对象和 CacheType | 不要求当前步真值，直接针对长 horizon 的预测可靠性；论文有 5x 以上 paired ablation，但需少量离线标定 |
| 探针 | 3 | **GP-Refiner posterior variance** | **观察** | 复用 calibrator 的 GP 后验方差作为附加 guard | 与预测误差共享状态，语义清晰；但计算和实现复杂度明显高于 RACER |
| 探针 | 4 | **BeCARE budget estimator + PI controller** | **观察：跨层路线** | 单独立项，不包装成 calibrator | 覆盖到 5.94x wall-clock，且直接控制实际 cache 数；但使用首块输入、scheduler sigma、流式更新和较多 per-backbone 标定 |

这里的“接”表示**值得进入同 harness 的实验性 PoC**，不表示可以直接合入或已经证明优于 cache-dit 现有实现。当前没有任何一篇论文的跨论文表格可以替代 cache-dit 自己的 A/B。

### 1.2 不建议投入的方向

1. **不要把 SpecKCast、Z-Cache、PoSTCache、SoftCache 原样称为纯 calibrator。** 它们分别涉及输出级 DCT/Kalman 状态、未来锚点重放、后验平滑、采样网格重构。
2. **不要为等价或弱增量方法增加一组并列 calibrator。** BaryCache、ChebBooster、BRACE、BeziCast 同属历史特征加权/有理或多项式基函数族，先验证 BRACE 与 BeziCast 即可。
3. **默认不接需要 per-model 训练或逐层 LUT 的方案。** L2P-Cache 的高加速结果值得保留为离线路线参考；LESA、FADE、OnlineCache 的模型/层/调度维护成本不符合本次“小改动、通用 API”的优先级。
4. **FeatFix 有价值但不属于本任务的两个缝。** 它在已经付费的 exact verification 点转发精确块输出，既不提出新预测器，也不提出新探针。
5. **SoftCache 值得后续单独评估，但不是新探针。** 它复用现有累积探针的 headroom 来改变采样步长和尾部网格，改动落在 scheduler。

## 2. 术语与判定口径

为避免论文中 `how` 的混用，本报告固定使用三层模型：

| 层 | 问题 | cache-dit 对应物 | 本任务归属 |
|---|---|---|---|
| ① when | 何时允许复用或预测 | `can_cache`、累积 relative L1、warmup、SCM、缓存步数与连续缓存守卫 | 次方向：探针 |
| ② how much | 跳过哪些块或区间 | `[Fn, N-Bn)` | 不属于纯 calibrator |
| ③ what | 缓存什么，以及如何从锚点预测 | 默认 `Bn_residual`，或 `CalibratorBase` 的预测值 | 主方向：纯预测器/calibrator |

用户所说的 calibrator “how” 在本报告中对应 **③what 的预测内容**。②中的缓存粒度不被混入 calibrator。

裁定标签：

- **接**：信息增量、接口适配和潜在收益同时成立，值得做最小实验性 PoC。
- **观察**：有可拆价值，但证据、成本或接口仍有明显缺口。
- **不接**：相对现有能力增量弱，或维护/运行成本与收益不匹配。
- **超范围**：方法本身可能有效，但需要新的执行模式、scheduler、缓存目标或输出路径，不属于本次两个接入方向。

详节中的 A/B/C/D 是对 `paper-triage.md` 模板的集成展开：A=定位/骨架，B=判据/信号，C=替代/状态成本，D=证据/缺口；量化/并行叠加与逐比特、命中证据统一列入 §7.1。

## 3. cache-dit 当前能力边界

### 3.1 纯 calibrator 的真实契约

只读核对结果：

- `src/cache_dit/caching/cache_contexts/calibrators/base.py:11-49`：calibrator 提供 `reset_cache`、`approximate`、`mark_step_begin`、`update`。
- `approximate(name)` 不接收当前 Fn 输出、latent、scheduler sigma 或当前块输入。
- `update(Y, name)` 在真实计算步写入锚点；`elapsed` 表示距最近锚点的步数。
- `src/cache_dit/caching/cache_contexts/calibrators/__init__.py:457-463`：当前已注册 `taylorseer`、`dmd`、`foca`、`svd`、`hicache`。
- `src/cache_dit/caching/cache_blocks/pattern_base.py:263-297`：执行顺序为“计算 Fn → 计算 Fn residual/hidden diff → `can_cache` → `apply_cache` → Bn”。
- `pattern_base.py:307-352`：只有未命中时才经过 middle blocks 并写入新的 Bn/encoder 锚点。
- `src/cache_dit/caching/cache_interface.py:63-166`：对外已有独立 `calibrator_config`，无需改变主入口参数形态。

因此，一个真正的零侵入预测器必须满足：

1. 只消费历史真实锚点、锚点步号和 `elapsed`；不得读取当前 Fn 或 latent。
2. 返回与现有 residual/hidden states 完全同形的张量。
3. 不增加 denoiser/block 前向，不改变缓存命中，不改采样网格。
4. 不以当前预测值冒充真实锚点，除非明确设计并验证闭环预测状态。
5. 若使用绝对时间而非步数，应把 scheduler 依赖列为额外适配成本；不能假定现有接口已经提供。

### 3.2 探针不是现成的可插拔接口

- `cache_manager.py:863-940` 的 `can_cache()` 覆盖 warmup、SCM、缓存步数、连续缓存与累积 diff；`cache_manager.py:553-626` 的 `similarity()` 覆盖相对 L1 与 token 聚合。
- `cache_manager.py:648-651` 与 `cache_manager.py:936-939` 包含 channel 下采样；`cache_manager.py:815-854` 的 `apply_cache()` 负责缓存内容近似，并在 pattern 的 `can_cache` 之后执行。

所以探针候选即使不需要新 CacheType，也通常要改判定层、配置或上下文。其改动面不能按 calibrator 估算。

现模板还有两个结构性约束：

- Fn 至少为 1，并且总会先执行；输入级整步探针可以省掉 Fn，DBCache 不能。
- `can_cache` 先于 `approximate`；使用双预测或后验方差做判定时，需要显式的只读 forecast/probe 路径，不能依赖隐式副作用。

## 4. 主方向：纯 Calibrator 候选排序

### 4.1 总表

| 论文 | arXiv v1 | 可拆出的预测机制 | 离线产物 | 纯 calibrator 适配 | 论文内最高档证据摘要 | 裁定 |
|---|---|---|---|---|---|---|
| BRACE | 2026-08-04 | C≤3 历史特征的 barycentric rational 加权 | 无；一个边界权重超参 | **是**，需将 raw feature 改为 Bn residual，并处理非均匀锚点 | FLUX 展示约 5.55x FLOPs；HunyuanVideo N=6 | **接：P0** |
| BeziCast | 2026-10-12 | Bézier 控制点的正则化 Lagrange 等价权重 + 外推 guardrail | 无 | **是**；需小型稠密求解 | 最高 4.79x latency；多图像/视频骨干 | **接：P0/P1** |
| RACER shrink-only | 2026-08-03 | 双预测分歧驱动的 anchor/forecast 收缩 | 短 trace 标定少量标量 | **可拆**；不含 refresh/repay | 整套论文各表包含 5.09x、8.77x、9.49x 等不同口径的高档，但不能归因给 shrink | **观察（仅③；①见 §5.1）** |
| GP-Refiner | 2026-09-05 | 基础预测器 + GP 后验均值修正 | 无预训练；在线保存 h=10 锚点 | **可拆**；高维 GP 成本需实测 | 固定调度在 5.06/6.24x FLOPs 档比 reuse 高 0.489/0.615 dB | **观察** |
| Spectrum | 2026-03-02 | 全局/滑窗 Chebyshev 基在线 ridge | 无 | **是**；但需全历史和在线求解 | 最高 4.79x latency、5.56x FLOPs；有 DBCache 行但非同档证明 | **观察** |
| HybridFF Adaptive | 2026-10-04 | OLS、Hermite、Chebyshev 多基自适应融合 | Adaptive 无；fixed 模式有 | **是**，但相当于组合多个预测器 | 最高 4.75x latency；相对最佳基线增益多为小幅 | **观察** |
| ChebBooster | 2026-08-21 | 历史特征的全局最小二乘 Chebyshev 系数 | 无 per-model 权重；按调度缓存标量权重 | **可拆**；动态调度需按真实锚点重算权重 | 最高 5.12x FLOPs、4.50x latency；无 DBCache 基线 | **观察** |
| TC-Padé | 2026-03-03 | 低阶 Padé-like 残差 + 分阶段策略 | 无 | 公式可落入 calibrator；张量除法和稳定性需验证 | 重点是 20-step 低步数档，FLUX 2.88x、Wan 1.72x | **观察** |
| BaryCache | 2026-08-26 | H=2 的局部 barycentric 加权 | 无 | 是，但和 BRACE/BeziCast 高度同族 | FLUX 2.45x、Wan 1.87x；动态调度消融不占优 | **不单独接** |
| L2P-Cache | 2026-04-29 | 每步可学习线性权重 | **每模型/总步数 W，50 条轨迹** | 推理公式可落入；训练与产物不属于通用零配置路径 | FLUX 最高 6.03x latency/7.14x FLOPs；Qwen 最高 7.18x FLOPs | **观察：离线路线** |
| LESA | 2026-02-24 | 多阶段 KAN 专家预测器 | **每模型多专家权重和两阶段训练** | 技术上可包装，维护面大 | 多个 5–6.25x FLOPs 档 | **不接默认路线** |
| PrediT | 2026-02-25 | 线性多步预测 + corrector + horizon 控制 | 无 | **否**；公式需要当前 latent，corrector 需要额外模型求值 | FLUX 4.61x、Wan 1.98x | **超范围** |
| MeanCache | 2026-01-27 | JVP 估计的区间平均速度 + 峰值调度 | 每模型/求解器调度表 | **否**；需要当前 latent 和速度/JVP | 主要改善 2–4x 档 | **超范围** |
| FADE | 2026-11-05 | 分数阶基函数系数校正 | **按 t/layer/module 的 LUT + 全局标定** | 只有 predictor 公式可拆，核心证据依赖 LUT | FLUX 最高 7.21x FLOPs、5.76x latency | **不接默认路线** |
| Z-Cache | 2026-11-01 | 双边界 Hermite + 未来锚点重放 | 轨迹段和未来搜索 | **否** | 最快档与同延迟 reuse 相比有明显质量代价，见 4.5 | **不接** |
| SpecKCast | 2026-10-26 | DCT 域 Kalman predictor/corrector | 频带能量标定与 scheduler 时间 | **不是现有残差 calibrator** | 整套到 6.10x latency；组件不可直接继承结论 | **超范围；探针另评** |
| AdaCorrection | 2026-02-13 | cached/fresh 激活自适应混合 | 无 | **否**；算法在缓存层仍计算 fresh block | 主要是中等加速和校正，不是长程预测 | **不接** |
| FeatFix | 2026-07-30 | 付费 exact block 的本地替换 | 固定 layer–timestep 审计点 | **否**；不是预测公式 | 最高 6.70x，但收益属于校正执行路径 | **超范围** |
| SoftCache | 2026-10-15 | 两锚点线性 residual motion | 无 | 预测式近似已有 Taylor 一阶；主创新不在预测器 | 3.5–6.9x，但核心收益来自 headroom jump | **超范围** |

### 4.2 P0：BRACE

**A. 算法骨架与粒度。** BRACE 只替换缓存特征预测式。对窗口内历史锚点 \(F_{\tau_j}\)，先映射到局部时间坐标，再计算

\[
F_{\mathrm{pred}}(t)=\frac{\sum_j w_j F_{\tau_j}/(x(t)-x(\tau_j))}{\sum_j w_j/(x(t)-x(\tau_j))}.
\]

权重使用交替符号和端点调节的 adapted Chebyshev 形式；原论文 Algorithm 1 的固定 interval 属于 when，移植时保留 DBCache 即可。

**B. 信号与离线依赖。** 只使用真实 full-step 特征、对应时间和一个边界权重超参；无训练、无 per-model 系数文件、无额外前向。默认窗口 \(C=3\)，端点敏感度 \(\gamma=0.7\)（Eq. 8、Table 5）。

**C. horizon、对象和成本。** 可对任意目标时间求值；预测对象可平移到 `Bn_residual`，但这属于本报告提出的适配，不是论文已验证对象。每次预测是 \(O(C)\) 次张量乘加，需保存 C 个锚点。没有 Spectrum 的回归求解或 FADE 的 LUT。

**D. 证据与缺口。** 原文覆盖 DiT-XL/2、FLUX.1-dev、HunyuanVideo；FLUX 给出约 5.55x FLOPs 的极端档，视频测试 N=6。没有 DBCache 同 harness 对照，原文部分主表在 HTML 中未转成可读文本，因此本报告不据此编造精确表格数字。固定等间隔锚点与 cache-dit 动态非均匀锚点之间也存在证据缺口。

**集成判断。** 它是当前最便宜的“非 Taylor 新预测族”验证入口。建议只实现 BRACE predictor，不同时引入新的调度器。

**实现更新（2026-11-19）。** 已新增 `calibrators/brace.py` 及 Python/YAML/CLI 接线。使用逻辑步号、真实锚点 FIFO 和有效窗口长度；公开 `gamma≥0.5`（更小值在外推域有解析极点）；预测不入窗，bf16/fp16 分子用 fp32 累加，标量分母相消时回退最近真实锚点并计数。244 项相关回归通过；FLUX 单点结果见 §7.2，完整临时记录为 `cache-dit/.tmp/research/brace_analysis.md`。

### 4.3 P0/P1：BeziCast predictor + guardrail

**A. 算法骨架。** 论文以低阶 Bézier 轨迹和控制点表述；实现上在至多 K 个局部历史锚点构造 Lagrange 基函数，以 Tikhonov 正则求等价预测权重，再加权历史特征。guardrail 使用外推距离、系数 \(l_1\) 范数、权重范数和“最远锚点是否支撑新预测”等信号，以凸包启发向保守 simplex-projected estimate 收缩；这些量全部来自历史节点，不读取当前真值。

**B. 信号与离线依赖。** 在线缓存锚点，默认 \(K_{\max}=6\)、\(d_{\max}=6\)、\(\lambda=10^{-3}\)、\(\sigma_{\max}=4\)；无 per-model 训练或曲线。相比 BRACE 多一个小型稠密线性系统。

**C. horizon 与成本。** 原生支持非均匀历史节点和任意预测距离，较适合 DBCache 的实际锚点分布。最多保存 6 个完整张量；guardrail 只做预测收缩，不强制当前步重算，因此可以留在 calibrator 内。

**D. 证据与缺口。** 覆盖 FLUX、SD3.5、SANA、HunyuanVideo、Wan2.1 等，最高报告 4.79x latency；对比包含 TaylorSeer/HiCache，但缺 DBCache 和同档 >5x 证据。原论文还测试输出近端目标，移植到 Bn residual 后必须重新验收。

**集成判断。** 若目标是“动态 DBCache 下的非均匀锚点预测”，它比固定周期 barycentric 论文更贴近实际。建议作为第二个 PoC，并单独做 guardrail on/off。

### 4.4 第二梯队与同族方法

#### Spectrum

- **A**：把预测写成全局/滑窗基函数回归，默认 Chebyshev 基与在线 ridge。
- **B**：无离线 per-model 权重，但需要从 full-step 历史构造 \(K\times M\) 设计矩阵。
- **C**：可预测任意 t；保存约 \(K\times M\) 个完整状态，带来显存、主路径求解和动态图风险。
- **D**：FLUX/HunyuanVideo 最高约 4.79x latency、5.56x FLOPs。它是 RACER 的 direct predictor，但 RACER 的整体收益不能反向证明 Spectrum 单独优于现有 calibrator。
- **裁定**：观察。若做 RACER，可复用其实现；不作为首个独立 calibrator。

#### HybridFF Adaptive

- **A**：OLS、Hermite、Chebyshev 三个分支，并用真实步上的历史预测误差更新权重。
- **B**：Adaptive 不需要离线 per-model 权重；fixed 模式需要，二者不能混为一谈。
- **C**：`update(Y)` 足以取得真实锚点并计算旧预测误差，接口上可行；但要维护多个预测器和小型协方差求解。
- **D**：最高 4.75x latency；FLUX NFE=10 的 Adaptive ImageReward 相对 Spectrum/HiCache 的增益不足 0.5，且缺 DBCache。
- **裁定**：观察。应在单基预测器证明有增益后再做融合，不宜先引入组合维护面。

#### GP-Refiner

- **A**：保存“基础预测值、精确锚点”对，以 GP 后验均值修正预测，后验方差决定是否刷新。
- **B**：无需预训练，但默认 \(h=10\)，需要高维 RBF kernel、GP 超参和数值正则。
- **C**：校正分支可包装在基础 calibrator 外，且不增加 denoiser；方差分支属于 when，必须另行接入。
- **D**：固定调度下，FLUX 在 NFE=12、16 时分别约 6.24x、5.06x FLOPs，PSNR 30.230、30.457，相比 reuse 的 29.615、29.968 高 0.615、0.489 dB；同论文也报告 TaylorSeer 行。摘要另报与 TaylorSeer 同 NFE 对照的计算量 −19.3%、PSNR +0.9dB、LPIPS 0.46→0.29，属于不同设置，不能与上述高加速档混用。没有 DBCache 对照。
- **裁定**：观察。它比普通多基预测更有新信息，但必须先证明高维 GP 的墙钟和显存开销可忽略。

#### RACER shrink-only

- **A**：分别生成 direct/derivative-aware forecast；用二者相对历史锚点的偏差形成可靠度，再在 anchor 与 forecast 之间收缩。
- **B**：两个预测器均只读历史，但需要短 trace 标定 trust levels 和趋势阈值；不是完全零标定。
- **C**：不增加 denoiser，可作用于现有预测器之外；需要同时保存两套预测状态。
- **D**：整套 RACER 在 FLUX、Wan2.1、HunyuanVideo 的各表包含 5.09x、8.77x、9.49x 等不同口径的高档，并做 matched-pairs；但这些数字同时包含 refresh/repay，不可记成 shrink-only 收益，也不跨表排序。
- **裁定**：观察，适合做第三个组合式 PoC；探针价值高于单独预测价值。

#### ChebBooster、BaryCache、TC-Padé

- **ChebBooster**：N 个 full-step 特征 + 在线 Chebyshev 最小二乘权重，最高 5.12x FLOPs、4.50x latency。无 per-model 训练，但保存更多锚点并执行拟合；与 BRACE/BeziCast 同族，优先级较低。
- **BaryCache**：局部 H=2 barycentric，简单但最高仅约 2.45x，且论文自己的 adaptive schedule 不占优。没有理由在 BRACE 之外再增加一个默认实现。
- **TC-Padé**：残差上有理外推和阶段策略可落入 calibrator，TSI 属于 when；但重点是 20 步低 NFE 场景，最高 2.88x，且逐元素张量分母需要额外数值稳定验证。当前不作为高加速首选。

### 4.5 离线学习、额外求值与未来信息：为何不优先

| 方法 | 关键原文事实 | 本次裁定 |
|---|---|---|
| L2P-Cache | 从 50 条完整轨迹学习 \(49\times49\) 的 per-timestep 权重，约 20 秒；高档结果很强 | 保留为“允许离线产物”的观察项；不能和 BRACE 一样宣称零配置通用接入 |
| LESA | KAN + 多阶段专家 + GT-guided 与 closed-loop 两阶段训练 | 每模型训练与专家维护成本过大，不接默认路线 |
| FADE | 按 timestep、layer、module 存 LUT，并做全局标定；主路径执行分数阶基函数 | 与“少量通用状态”目标冲突；不建议先做逐层 LUT |
| PrediT | AB 形式需要当前 latent；corrector 在 skipped step 评估一次模型并做 full refinement；DSM 改 horizon | 不是纯 calibrator；其 corrector 会改变执行和加速比 |
| MeanCache | 用当前 latent、速度及 JVP 近似区间平均速度，并离线构造调度 | 需要新 pipeline 状态与 JVP 路径，不属于现有 calibrator |
| Z-Cache | 先保留内部草稿，未来锚点出现后做双边界 Hermite，再重放 latent 段 | 使用未来信息且改变状态提交；FLUX N=4 为 9.71s/18.48dB，同表同工作点 reuse(N=4) 为 9.69s/22.82dB：二者延迟基本相等，Z-Cache 的 PSNR 低 4.34dB |
| AdaCorrection | Algorithm 1 在需校正的缓存层仍执行 fresh block，再与 cached block 混合 | 不是跳过路径上的历史预测器，且当前块成本没有被消掉 |
| FeatFix | 只有已支付 exact block 计算时才转发精确结果；不定义 forecast 或在线 acceptance | 是有价值的跨层校正，不属于本任务两个缝 |
| SpecKCast | 对输出近端残差做 DCT、Kalman 预测/校正，并按 scheduler 时间和频带能量建模 | 需要新的缓存对象和状态，不可把整套 6.10x latency 结果继承给普通 Bn residual |
| SoftCache | 用探针 headroom 设置下一步步长、离网落点并重建尾部网格；residual motion 只是两锚点线性外推 | 高潜力 scheduler 路线，但不是新 calibrator 或新探针 |

## 5. 次方向：在线探针候选排序

### 5.1 与现有 DBCache 的信息增量对照

| 候选 | 探针位置与信号 | 是否用当前真值 | 是否需离线产物 | 相对 DBCache 的增量 | 主要改造 | 裁定 |
|---|---|---|---|---|---|---|
| SeaCache | 输入或中间特征；时变 SEA 滤波后的 relative L1 | 否 | 无；需 scheduler \(a_t,b_t\) | **度量空间增量**：抑制噪声、强调内容相关频段 | FFT/iFFT、token 几何 reshape、scheduler 系数、增益归一化 | **接：P0** |
| RACER | 两个历史预测及 anchor forecast；预测分歧和可靠度 | 否 | 短 trace 标定标量 | **预测不确定度增量**，直接面向长 horizon | `can_cache` 前的只读 forecast/probe、双预测状态、trust-region | **接：P0** |
| GP-Refiner | 基础预测与 GP 后验方差 | 否；真实锚点用于在线更新 | 无预训练 | **后验不确定度增量** | 高维 GP 状态、kernel/求解、判定层读取方差 | **观察** |
| BeCARE | 第一块归一化输入；预算观测器 + PI | 否 | 多组 backbone/sampler 标定 | **预算闭环增量**，控制实际 cache 数 | 输入级执行路径、scheduler sigma、流式事件、目标预算控制 | **观察：跨层** |
| SpecKCast | Kalman 预测方差、频带能量、fresh cadence | 否；完整步可做后验更新 | 频带能量标定 | 预测风险与 model-output spectral 状态 | 输出级 DCT/Kalman 与新缓存对象 | **观察：独立算法栈** |
| AlignCache | 精确步上的旧预测与 guided output 的真实误差；条件/无条件分支 | 在精确步使用当前真值 | 无 | **闭环实测误差增量** | CFG 双分支耦合、水平集阈值、整步预测与动量校正 | **观察** |
| ACID | 历史 activation change 的离线敏感度和 per-step 阈值 | 不在线使用当前真值 | **per-model 阈值曲线** | 不是新在线信号；是阈值时间调度 | `can_cache` 的分步阈值与部署产物 | **观察：低优先级** |
| SenCache | 输入/输出 Jacobian 的局部敏感度 | 需要离线 trace | **per-model JVP 曲线** | 输入级敏感度，而非当前 Fn residual | 新输入探针、JVP 离线工具、整步执行路径 | **不接默认路线** |
| SoftCache | 现有累积探针距阈值的 headroom | 否 | 无 | 不新增探针；把 headroom 用于“还能走多远” | scheduler 步长、离网落点、尾部重排 | **超范围** |
| PrediT | 精确步上预测与实测的 \(\delta\)，调整未来 horizon | 在精确步使用 | 无 | 与 DBCache 累积 Fn diff 接近，但测量位置更晚 | horizon scheduler + corrector 执行路径 | **不接为独立探针** |

### 5.2 P0：SeaCache

**A. 算法骨架。** 保持累积 relative L1 和刷新阈值不变，只在距离计算前施加时变 SEA 滤波：

\[
\widetilde{\Delta}_t=\operatorname{RelL1}\left(\mathcal P(G_t^{norm},I_t),\mathcal P(G_{t+1}^{norm},I_{t+1})\right).
\]

滤波器来自线性 denoiser 的 Wiener-like 响应 \(G_t(f)=a_tS_x(f)/(a_t^2S_x(f)+b_t^2)\)（Eq. 5），使用 scheduler 的 \((a_t,b_t)\)、自然图像功率谱假设和均值增益归一化（Eq. 7）。图像在 2D 空间轴、视频在 3D 时空轴执行 FFT/filter/iFFT（Eq. 6）。

**B. 信号与产物。** 在线使用当前和上一时刻的输入/中间特征，不训练、不存 per-model 曲线。官方代码从 scheduler 的 `sigmas` 或 `alphas_cumprod` 得到系数，并将特征转为 fp32 做 FFT。

**C. 成本与落点。** 官方 README 称滤波约占总延迟 0.2%，这是论文/作者口径，不是 cache-dit 实测。论文还在 DiCache 中间块特征上验证 SEA filter，说明它不限于输入级；但这不等于已经验证 cache-dit 的 Fn residual。接入需要明确 token 到 H×W 或 F×H×W 的几何恢复，不能对任意扁平 token 序列直接做空间 FFT。

**D. 证据。** 原文在 FLUX、HunyuanVideo、Wan2.1 上比较 TeaCache/TaylorSeer 等，并做 SEA、反向滤波、无归一化和静态低通消融；Fig. 9 给出 DiCache+SEA 的同刷新率对比。原论文主要覆盖约 30%/50% 刷新预算，并未直接证明 DBCache >5x 档。官方代码可用，且仓库记录 2026-11 已被 Diffusers 和 NVIDIA Cosmos 独立集成。

**推荐形态。** 新增“filtered relative L1”判定选项，不增加 CacheType，不改变 Bn residual。优先验证 FLUX 与 Wan 两类几何明确的骨干；对无法恢复空间/时间轴的模型显式标为不支持，不做静默退化。

### 5.3 P0：RACER forecast-disagreement

**A. 算法骨架。** direct 与 derivative-aware 两个预测器对同一未来步产生不同结果，再与 anchor forecast 比较，得到 \(r_{\text{dir}},r_{\text{der}},r_{\text{int}}\) 和 \(r_{\max}\)。当分歧低时信任预测；升高时收缩；超过高风险阈值时刷新。

**B. 信号与产物。** 全部判定可在当前 denoiser 前由历史锚点计算，无需当前 Fn 真值。需要短离线 trace 标定 trust 阈值、shrink 系数和趋势检测参数。

**C. 成本与落点。** 双预测的主要成本是张量加权和少量标量 norm，无额外 denoiser；但若分别使用 Spectrum 和 Hermite，则需维护两套历史。当前 `can_cache` 在 `approximate` 之前，因此需要显式只读 probe 或把可靠度计算放到独立上下文中；不能只把 RACER 塞进 `approximate` 后还声称能控制 when。

**D. 证据。** 原文在多骨干上给出 5x 以上的 paired ablation，例如 5.09–8.77x 等档；分离了 forecast shrink、trust-region refresh 和 repay。但没有 DBCache 对照，且完整收益来自组件组合。

**推荐形态。** 先做两阶段拆分：

1. shrink-only，DBCache 的命中序列不变，用于验证纯预测收益；
2. disagreement guard，只改 when，固定同一预测器，在相同 Cache Steps 或相同延迟下比较。

完整 refresh-and-repay 不属于最小接入。

### 5.4 第二梯队探针

#### GP-Refiner posterior variance

GP 的 posterior mean 与 variance 共享历史对，不需要当前真值，理论上是非常干净的 when/what 联合概率模型。问题是每个高层特征都要做 h 点 kernel 和反作用，其实现、显存和 compile 成本都高于 RACER。建议只有在 GP correction 被证明有效后再启用 variance guard。

#### BeCARE

它把输入侧相对 L2 经在线观测器转为预算误差，再用 PI 控制缓存预算，并配合最大连缓存和 warmup。优点是直接控制实际 Cache Steps，且论文报告 FLUX 5.94x、Wan2.1 4.80x、HunyuanVideo 5.85x。缺点是探针位于第一块归一化输入，而非 DBCache 的 Fn 后；还需 scheduler sigma、流式更新以及 per-backbone/sampler 标定。若改用 Fn 输出，是新的派生方法，不能继承原论文数字。

#### SpecKCast

Kalman 风险信号在预测状态上生成，不需要当前真值；但状态是 DCT 域输出近端残差，校正和频带能量又与 scheduler 时间轴耦合。适合作为独立 SpecKCast 路线，不适合作为 DBCache 的轻量开关。

#### AlignCache

在精确步测得“旧预测 vs 当前 guided output”的真实误差，再调整下一步预测和双分支阈值。其优点是利用反事实误差而非只看特征位移；缺点是必须同时处理 conditional/unconditional 输出、组合 guided output 和 branch-separate 状态。它更像完整的 closed-loop 控制方案，不是单点探针替换。

#### ACID 与 SenCache

ACID 的核心产物是 per-step 阈值曲线，属于离线 when 策略；cache-dit 虽有 SCM，但 SCM 是布尔步掩码，与分步阈值并不等价。因此 ACID 有小幅适配价值，却不是新的在线探针，且论文最高约 3.97x。

SenCache 的 Jacobian 敏感度可以提供不同信息，但需要离线 JVP 曲线、输入级判定和每模型调度。其成本和维护面不符合本次优先级。

## 6. 复合论文拆件总表

| 论文 | 判定信号 | 粒度 | 替代算子 | 校正回路 | 离线产物 | 可保留残值 |
|---|---|---|---|---|---|---|
| RACER | 双预测分歧 | 整步 horizon | direct + derivative forecast | 预测收缩 | 短 trace 标定 | **预测 shrink 与探针都可拆** |
| GP-Refiner | GP 后验方差 | 整步 | 任意基础预测器 | GP 后验均值 | 无预训练 | **校正和探针均可拆，成本待证** |
| SpecKCast | Kalman 风险 | 整步 | DCT 域 Kalman | 后验校正 | 频带能量 | 仅适合独立算法栈 |
| BeCARE | 输入预算观测 | 整步 | 任意缓存策略 | PI 控制 | 多组标定 | 预算控制器可单独立项 |
| AlignCache | 真实反事实误差 | 整步 | Taylor draft | 动量校正 + 分支学习率 | 无 | 闭环思路可参考，整体跨层 |
| SoftCache | 累积探针 headroom | 整步及时间网格 | 两锚点线性预测 | 无 | 无 | **scheduler 路线；不是新探针** |
| PrediT | 预测误差与条件稳定域 | 整步 horizon | 线性多步 | 额外模型求值 corrector | 无 | 不满足纯 calibrator |
| FeatFix | 固定付费验证点 | 指定 layer–timestep | 任意 draft policy | 转发 exact block | 固定审计点 | 独立局部校正路线 |
| Z-Cache | 最大相对位移 | 轨迹段 | 双边界 Hermite | 未来锚点平滑 | 在线搜索 | 不适合 forward-only cache |
| PoSTCache | 后验方差下降 | 可修订轨迹段 | 任意 forward proposal | 后验平滑 | 离线依赖估计 | 摘要级排除，非在线单向 calibrator |
| IdealCache | 理想轨迹图 | 非均匀调度 | 现有 cache | 目标轨迹选择 | 每输入候选轨迹/图搜索 | 摘要级排除，非轻量在线探针 |

## 7. 建议的最小同 Harness 证伪实验

以下仅为后续实验设计，本次调研没有执行。

### 7.1 共同规则

1. 固定初始 latent/seed、prompt 集、分辨率、sampler、Fn/Bn、warmup、所有 cache guards；预测器 A/B 必须保持相同 Cache Steps。
2. 基线至少包含：无 cache、DBCache+reuse、DBCache+TaylorSeer、DBCache+HiCache；新探针实验则固定同一个缓存内容预测器。
3. 同时覆盖 cache-dit 已有的约 1.7x 档和 >5x 高加速档，并记录完整工作点参数；论文中的 2–4x 结果不能外推为高加速收益。
4. 缓存命中只认 `Cache Steps`。不能用 `Executed Steps` 或缓存残差中的 `-1.0` 代替命中证据。
5. 使用同配置重跑估计 run-to-run noise floor；指标采用原始浮点图像或正确量纲的 uint8，禁止模减和错误 peak 造成的 PSNR 虚高。
6. 分开记录 wall-clock、Cache Steps、预测器额外显存和主路径开销；不能只报 FLOPs。
7. 每个新预测器/策略先过 `max_cached_steps=0` 与无 cache 的逐比特回归；用 `approximate()` 调用计数、state 类型和参数断言证明接线，不能用恒等测试轨迹上的输出差异代替。
8. 至少补一组量化叠加与一组 CP/TP smoke，确认缓存张量替换不改变 quant scale 或并行切分；不支持必须明示。与 compile 组合若不收敛，只报区间，不与其他行共用工作点。

### 7.2 BRACE：先证明“换预测式”本身有收益

| 项 | 设计 |
|---|---|
| 唯一变化 | `Bn_residual` 的预测式；when 完全固定 |
| 臂 | reuse / TaylorSeer / HiCache / BRACE C=2 / BRACE C=3 |
| 关键压力点 | 动态非均匀锚点、长 horizon、分母近零、不同 anchor 间隔 |
| 通过条件 | 相同 Cache Steps 下稳定优于 reuse，并在高档相对最佳现有预测器有超过 noise floor 的精度增益；额外延迟和显存可忽略 |
| 失败即停 | C=3 在多个骨干上均不超过 Taylor/HiCache，或非均匀锚点下出现不稳定 |

**本轮仅完成 FLUX 单点功能子集（2026-11-19），尚不满足上述完整通过条件。** reuse、TaylorSeer、BRACE C2/C3 在 compile+F1B0+rdt0.12 下均为 12/28 Cache Steps，延迟分别 8.65/8.66/8.65/8.66s；相对 compile 的 PSNR 分别 29.03/31.52/29.57/30.75dB，SSIM 0.9614/0.9740/0.9720/0.9743。BRACE 无回退告警，244 项相关回归通过。未跑 HiCache、>5x、多 prompt/骨干、noise floor、量化或 CP/TP，因此该结果只证明实现与通用 E2E 可用，不改变后续证伪要求，也不宣称 BRACE 优于最佳现有预测器。

### 7.3 BeziCast：验证非均匀节点与 guardrail 是否真有增量

| 项 | 设计 |
|---|---|
| 臂 | 无 guardrail / 有 guardrail；K=3 与 K=6；与 BRACE 同调度比较 |
| 额外记录 | 外推距离、权重范数、收缩触发率、预测残差误差 |
| 通过条件 | 在非均匀 DBCache 锚点上优于 BRACE/Taylor，且 guardrail 不是靠额外刷新获得收益 |
| 失败即停 | 小矩阵求解或 6 个锚点带来明显开销，而精度增益低于 noise floor |

### 7.4 SeaCache：只换探针，不换缓存内容

| 项 | 设计 |
|---|---|
| 固定项 | 相同缓存对象和 calibrator；仅 raw relative L1 与 SEA-filtered relative L1 不同 |
| 模型 | 先做几何明确的 FLUX 与 Wan；不为未知 token 几何做默认回退 |
| 对照口径 | 在相同 Cache Steps 下比精度，再在相同精度 gate 下比 Cache Steps/延迟 |
| 额外记录 | fp32 FFT/iFFT 时间、峰值显存、scheduler 系数、compile/offload 行为 |
| 下采样交互 | 分别测试 `downsample_factor=1` 与 `>1`，明确 SEA 滤波位于 channel 下采样之前还是之后 |
| 失败即停 | 滤波收益被 FFT 开销抵消，或 Fn residual 上无法复现论文中间特征结论 |

### 7.5 RACER：把 shrink 与 when 分开验证

| 阶段 | 设计 |
|---|---|
| shrink-only | 固定 DBCache 命中序列，只改变预测收缩；若此臂无收益，不做完整 RACER |
| disagreement guard | 固定预测器，仅加入分歧 guard；与 raw DBCache 在相同 Cache Steps/延迟下比较 |
| 离线纪律 | 校准 prompt 与测试 prompt 严格分离；记录所有 trust 标量，不将每模型调参藏在默认配置中 |
| 失败即停 | 双预测成本明显，或 guard 在 >5x 档不能提高 matched-budget 精度 |

### 7.6 后续可选

- GP-Refiner：先在固定调度下只验证 posterior correction，再单独验证 variance guard。
- SoftCache：若未来允许 scheduler 改造，再单独立项；不要与 calibrator PR 混合。
- L2P-Cache：只有在项目明确接受 per-model artifact 后才考虑；需先验证跨步数、跨 prompt 和动态调度泛化。

## 8. 证据索引

以下 24 篇均阅读了 arXiv HTML 的方法与实验原文，并通过 arXiv abs 的 Submission history 核验 v1 日期。表中数字只是论文来源内陈述，不作跨论文排名。

| 方法 | arXiv / 官方原文 | v1 日期 | 本报告主要依据 |
|---|---|---|---|
| MeanCache | [2601.19961](https://arxiv.org/abs/2601.19961) | 2026-01-27 | 方法 §3.2–3.3；FLUX Table 3 |
| AdaCorrection | [2602.13357](https://arxiv.org/abs/2602.13357) | 2026-02-13 | 方法 Eq. 1–7；Algorithm 1 |
| SeaCache | [2602.18993](https://arxiv.org/abs/2602.18993)；[官方代码](https://github.com/jiwoogit/SeaCache) | 2026-02-22 | Eq. 5–8；Tables 1–4；Fig. 8–9；`FLUX/util_seacache.py` |
| LESA | [2602.20497](https://arxiv.org/abs/2602.20497) | 2026-02-24 | Eq. 4–7；Tables 1–4 |
| SenCache | [2602.24208](https://arxiv.org/abs/2602.24208) | 2026-02-24 | Eq. 3–5；Tables 1–3 |
| PrediT | [2602.18093](https://arxiv.org/abs/2602.18093) | 2026-02-25 | Eq. 5–11；Tables 1–2 |
| Spectrum | [2603.01623](https://arxiv.org/abs/2603.01623) | 2026-03-02 | Eq. 5–9；Tables 1–2 |
| TC-Padé | [2603.02943](https://arxiv.org/abs/2603.02943) | 2026-03-03 | Eq. 1–9；Tables 1–3 |
| L2P-Cache | [2604.26365](https://arxiv.org/abs/2604.26365) | 2026-04-29 | Eq. 12–15；Tables 1–3 |
| BeCARE | [2605.27075](https://arxiv.org/abs/2605.27075) | 2026-05-26 | Eq. 4–13；Table 1；索引旧称 SoftCap，原文现名为 BeCARE |
| ACID | [2607.12358](https://arxiv.org/abs/2607.12358) | 2026-07-13 | Eq. 3–7；Algorithm 1；Tables 1–4 |
| FeatFix | [2607.27842](https://arxiv.org/abs/2607.27842) | 2026-07-30 | Eq. 1–5；matched Verify/FeatFix 设计 |
| RACER | [2608.01740](https://arxiv.org/abs/2608.01740) | 2026-08-03 | Eq. 9–13；Algorithm 1；Tables 2–5 |
| BRACE | [2608.07572](https://arxiv.org/abs/2608.07572)；[项目页](https://youngkinlon.github.io/BRACE-Taming-Sharp-Irregularities-via-Barycentric-Rational-Forecasting-for-Fast-DiT-Inference/) | 2026-08-04 | Eq. 4–8；Algorithm 1；Fig. 5–6 |
| ChebBooster | [2608.23429](https://arxiv.org/abs/2608.23429) | 2026-08-21 | Eq. 3–6；Tables 1–2 |
| BaryCache | [2608.28670](https://arxiv.org/abs/2608.28670) | 2026-08-26 | Eq. 1–3；Tables 1–5 |
| GP-Refiner | [2609.05981](https://arxiv.org/abs/2609.05981) | 2026-09-05 | Eq. 2–7；Tables 1–4 |
| HybridFF | [2610.05254](https://arxiv.org/abs/2610.05254) | 2026-10-04 | Eq. 4–11；Algorithm 1；Tables 1–3 |
| BeziCast | [2610.16448](https://arxiv.org/abs/2610.16448) | 2026-10-12 | Eq. 3–14；Algorithm 1；Tables 1–2 |
| AlignCache | [2610.16545](https://arxiv.org/abs/2610.16545) | 2026-10-10 | Eq. 4–11；Algorithm 1；Tables 1–4 |
| SoftCache | [2610.22369](https://arxiv.org/abs/2610.22369) | 2026-10-15 | Eq. 6–12；Tables 1–2；Fig. 6 |
| SpecKCast | [2610.26837](https://arxiv.org/abs/2610.26837) | 2026-10-26 | Eq. 1–10；Algorithm 1；Tables 1–2 |
| Z-Cache | [2611.01846](https://arxiv.org/abs/2611.01846) | 2026-11-01 | Eq. 1–8；Tables 1–2 |
| FADE | [2611.10514](https://arxiv.org/abs/2611.10514) | 2026-11-05 | Eq. 3–6；Table 1；Algorithm 1 |

### 8.1 摘要或官方页面级排除项

这些条目用于补足近期覆盖，不参与强推荐，也不假装已完成全文 triage。

| 方法 | 首发证据 | 排除原因 |
|---|---|---|
| OnlineCache | [2607.29398](https://arxiv.org/abs/2607.29398)，v1 2026-07-31 | 学习策略网络 + corrector + bilevel optimization，约 3x；不是轻量通用探针 |
| PoSTCache | [2610.25664](https://arxiv.org/abs/2610.25664)，v1 2026-10-19 | 后验平滑修订历史预测，PVR 依赖离线轨迹；非 forward-only 提交语义 |
| IdealCache | [2611.10340](https://arxiv.org/abs/2611.10340)，v1 2026-11-07 | 每输入理想轨迹集合 + 图最短路，不是在线 Fn 探针 |
| TileCache | [2611.20246](https://arxiv.org/abs/2611.20246)，v1 2026-11-15 | 多轮图像生成的跨轮 tile 缓存，任务和缓存对象不同 |
| ScoutCache | [2611.21038](https://arxiv.org/abs/2611.21038)，v1 2026-11-19 | AR KV-cache 的 key 复用与 value 量化，不是 DiT 时间步特征缓存 |
| D2Cache | [CVPR 2026 官方页面](https://openaccess.thecvf.com/content/CVPR2026/html/Liu_D2Cache_Second-Order_Delta_Caching_for_Higher_Video_Diffusion_Acceleration_CVPR_2026_paper.html) | 仅会议/摘要级核验，未找到 arXiv 首发记录；二阶 delta 与现有二阶 Taylor 信息重叠，且 timestep scaling 需额外输入，不作推荐 |

### 8.2 检索来源与去重说明

- [Awesome-DIT-Cache README](https://github.com/leeguandong/Awesome-DIT-Cache/blob/main/README.md)
- [2026-10-08 增量记录](https://github.com/leeguandong/Awesome-DIT-Cache/blob/main/docs/research-update-2026-10-08.md)
- [2026-11-15 增量记录](https://github.com/leeguandong/Awesome-DIT-Cache/blob/main/docs/research-update-2026-11-15.md)
- 补查窗口覆盖到 2026-11-19；新增的 ScoutCache 因任务对象不符被排除。
- `SpectralCache` 在索引中对应不同论文，不能与 HybridFF 的 Spectral Feature World Model 或 SpecKCast 混并。
- `SoftCap` 是索引中的旧名称；arXiv 2605.27075 当前原文标题为 BeCARE。
- BRACE、TC-Padé、D2Cache 通过代表论文引用扩散补入，避免只依赖索引。
- 筛选优先保留①在线判定信号或③历史预测有新信息的候选；主表中的 Phantom、PreSTP、ReelCache、MG-Cache、TGuide、FSC、CAVC、RECAP、SAKURA、FreezePlan 等未进入本次 30 项，也未做原文 triage。未纳入不等于“不接”，后续若转向 token/特征选择、任务专用或离线调度方向需另行评估。

## 9. 最终建议的实施顺序

若后续从调研转入开发，建议严格按以下顺序，每次只引入一个变量：

1. **BRACE calibrator PoC**：最小、无训练、少锚点，先验证动态 DBCache 下的非均匀有理预测。
2. **BeziCast calibrator PoC**：验证局部正则权重与 guardrail 是否比 BRACE 更适合真实锚点分布。
3. **SeaCache 探针 PoC**：固定缓存内容，只换 diff 度量，重点测空间几何和 FFT 开销。
4. **RACER shrink / disagreement 拆分 PoC**：只有前两项显示预测层仍有增益时才做组合；先 shrink，后 guard。
5. GP-Refiner、HybridFF、BeCARE 作为第二梯队；SoftCache、FeatFix、SpecKCast 如需推进，应各自单独立项。

版本化归档：本报告由 `cache-dit/.tmp/research/recent_dit_cache_algorithms_analysis.md` 纳入本 skill；核心裁定同步写入 `references/landscape.md`，导航入口见 `SKILL.md`。
