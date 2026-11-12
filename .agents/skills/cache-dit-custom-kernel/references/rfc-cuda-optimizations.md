# Cache-DiT CUDA Kernel 优化 RFC

> 状态：Active ｜ 日期：2026-11-11
> 关联：[SKILL.md](../SKILL.md)、[SVDQ 性能记录](svdq-nvfp4-cute.md)。

## 1. 验收目标

按依赖顺序验收，不允许跳过下层直接宣称上层完成：

1. `quantize_w4a4_fused_lora_kernel`：nsys kernel-only 稳态追平或略优于 native quantize kernel。
2. `gemm_w4a4_kernel`：nsys kernel-only 稳态相对 native `gemm_w4a4` 延迟降低至少 5%。
3. 完整 NVFP4 CuTe 工作流：相同 FLUX 条件下相对旧 native 工作流 e2e 延迟稳定降低至少 5%。
4. 所有性能优化必须通过数值、确定性、构建和图像回归；PSNR 只作精度比较，不替代性能指标。

“稳定”要求 warmup 后多次迭代，并报告 median/mean/min/max；单次冷启动或单个 profile 离群值不构成验收。

## 2. 总览

| 编号 | 项目 | 状态 | 验收标准 |
|---|---|---|---|
| CDK-1 | 独立 CuTe producer 与 hybrid GEMM 工作流 | 已完成 | producer 输出 native qact/ascales + row-major BF16 D；fused 无 scratch，split-K 使用内部 FP32 scratch/finalize；A/SFA G2R，B/SFB/D/L1 TMA |
| CDK-2 | Producer kernel 性能追平/略优 | 已完成 | PRO 5000 官方 bench 5 shape × rank32/64/128：本次会话回归 geomean 1.1344×，15/15 快（各格均不进入小 M 新分支）；小 M 收益另见 `svdq-nvfp4-cute.md` §31 |
| CDK-3 | Producer 改用 CuTe TiledMMA/TiledCopy | 已完成 | rank32/64/128 拆分 TU，fused + K128 async split-K + K256 split-K；qact/ascales bitwise golden 通过 |
| CDK-4 | GEMM 稳态快于 native 至少 5% | 进行中 | 官方 bench 三轮中位 kernel-only geomean **1.0248×**，12/15；NCU Tensor 84.2%、DRAM 12.9%、local=0，已接近单 CTA 资源几何上限，现实剩余约 1%–2% |
| CDK-5 | 完整 CuTe 工作流 e2e 快至少 5% | 进行中 | 单会话 row-major e2e geomean **1.0553×**（14/15），cute+native-D **1.0462×**（14/15）；尚需跨会话/FLUX 稳定性验收 |
| CDK-6 | 可重复 ccache 构建链路 | 已完成 | `build_fast.sh` editable + nvcc shim 验证通过 |

完成任一项时，同步更新本表、下方清单和 `SKILL.md` 的当前状态摘要。若实验否定某方案，将方案移到性能记录的已证伪表，不要删除证据。

## 3. 完成状态清单

- [x] CDK-1：独立 CuTe producer + A/SFA G2R、B/SFB/D/L1 TMA GEMM；fused 和 split-K 均输出 native qact/ascales 与 row-major BF16 D。
- [x] CDK-2：producer 官方 bench 5 shape × 3 rank 全格快于 native；本次会话回归 geomean 1.1344×，这些格不进入小 M 新分支，不把会话间增量归因于本项改动。
- [x] CDK-3：`quantize_impl.cuh` + `quantize_r32/64/128.cu`，TiledMMA/TiledCopy/swizzle、K128 double-A async、K256 split-K、rank32 大 M fused；K64/192/3136 tail 与 golden 已验证。
- [ ] CDK-4：GEMM 快于 native 至少 5%。kernel-only geomean 1.0248×；Tensor 84.2%、DRAM 12.9%、无 spill，已到当前 TMA/G2R + 单 CTA 结构的局部最优。
- [ ] CDK-5：完整工作流 e2e 快至少 5%。row-major e2e 1.0553×、cute+native-D 1.0462×（单会话）；需重复会话与 FLUX 稳定验收后才能关闭。
- [x] CDK-6：使用 `tools/build_fast.sh` 和 `tools/nvcc` 的 ccache 构建链路。

## 4. CDK-3：Producer CuTe 化设计约束（已完成）

- 保留 native packed qact/ascales 与 row-major BF16 D 输出；不得退回标准 row-major A/SFA 或旧 G2R 反解。
- fused 保持 CTA 内 D 归约；split-K 允许 producer 自有的 fragment-major FP32 scratch 与一次 BF16 finalize，但禁止读取 native producer 的 fragment D。
- 使用 CuTe `TiledMMA/TiledCopy` 与 swizzled shared staging；K128 async 只支持双 A stage，K256 使用四 slice + 单 A stage。
- 每次改动先跑 qact/ascales golden、K64 tail、非单位 rank scale 和高层 Linear parity，再做 nsys/NCU。
- 当前 rank-specialized 实现是新的性能基线；旧手写 fused/split-K 只作为历史参考，不再恢复。

## 5. CDK-4：GEMM 候选方案

2026-11-10 起 CDK-4 是主优先级：producer 已在 15/15 格快于 native（geomean 1.128×），但 kernel-only GEMM 只有 1.013×、e2e 1.038×。先解释 rank64/M8192/N3072 的 0.976× 和 N6144 的 0.988×，再从 shared 装载、TMA/mbarrier、同步和 epilogue 中按 NCU 选一个主瓶颈；不要继续扫 producer warp/M 分桶。

按单个假设逐项验证：

1. **Persistent CTA 调度**：减少小 M/波次尾损和重复 CTA 初始化；需核对每 tile 的 stage phase、descriptor 生命周期和 output overlay 重置。
2. **Stages/smem 复用调优**：已完成 stages 1/2/3/4 与 128x128/128x256/256x128/256x256 扫描。最佳 `128x256/BK128/S2` = 133.76µs，native 115.10µs，仍慢 15.89%；256x128 8-warp 数值错 4.3%，16-warp 编译失败；256x256 被 99KiB 资源门禁拒绝。**结论：stages/smem 不是主因，除非有新 NCU 证据，否则不要再次盲目重复扫描。**
3. **Direct g2r + pack 重排**：已尝试并回退。独立 `gemm_w4a4_direct_kernel` 不用 TMA/smem，但数值始终错误；根因是 native pack 按 MMA fragment 直接排布，而 CuTe producer 输出 standard row-major E2M1，二者不匹配。继续此路线前必须先做 fragment-aligned pack 或把 CuTe fragment reinterpret 成 native `uint4` 布局。
4. **独立 consumer WG**：考虑 1 个 producer WG + 2 个 consumer WG，将 NVFP4 与 BF16 LoRA MMA 分给不同 WG。必须解决共享 FP32 accumulator 或跨 WG 归约成本；先做原型测量，不能直接作为主路径。
5. **指令与装载路径**：结合 SASS/PC sampling 检查 LDSM、SF copy、MMA 依赖链和 epilogue，不凭理论 TFLOPS 推断收益。
6. **Descriptor/调度开销**：使用 nsys 区分 kernel 本体与 host/descriptor 成本；kernel-only 优化不能用 host 缓存收益替代。
7. **K-block/prologue 调度**：提前 release、FlashInfer/CUTLASS 两种 K-block 顺序、232-register budget 和 consumer 初始化后移均已实测；wait 局部改善被 fragment spill 或地址状态开销抵消，最终 nsys 无净收益。详见 `svdq-nvfp4-cute.md` §8，无新 spill 证据时不要重复。
8. **Persistent/light producer/小 tile**：persistent 110 CTAs 为 143.52us，慢约 4.4%（output overlay 串行化）；light producer 288T 路线已停；BN128/S3 为 168.99us 且 occupancy 不变，已回退；BM64 被 128-row SF atom 阻止。详见 `svdq-nvfp4-cute.md` §9。

FlashInfer/CUTLASS 的 K-block 顺序和 producer start state 已完成对照实验；start state 保留为标准协议，K-block 变体均已回退。本节是 2026-11-10 的历史候选；2026-11-11 复核后 A/SFA 全 TMA 与 fragment-aligned pack 已明确关闭，最新优先级见 §6。

GEMM 达标后再讨论默认 runtime 切回 CuTe；最终仍需 CDK-5 独立验收。

## 6. 2026-11-11 复核后的路径

### 6.1 GEMM 判断

- 15 格当前 dispatch 的 NCU：Tensor 84.2%、SM 80.2%、DRAM 12.9%、L2 hit 99.9%、occupancy 17.75%、regs 168、smem 85 KiB、local=0。
- stall：math_pipe_throttle 36.2% + wait 30.8%，long_scoreboard 18.4%，sleeping 7.5%。
- Tensor 指令流已接近持续发射峰值；即使非张量 stall 全消失的粗略上限也只有约 1.19×，实际远小。当前限制是单 CTA / 8 consumer warp / 128 FP32 accumulator 的资源几何，不是 DRAM 或 shared bank conflict。
- **关闭方向**：A/SFA 全 TMA+SMEM（最初实现，慢约14.7%）、8M×1N（0.794×）、S4/独立 LoRA ring（约0.99×）、A/SFA 简单前移/双缓冲、stage 增量回绕/unroll3、persistent/light producer/Non-WS/BN128 小 tile。证据见 `svdq-nvfp4-cute.md` §30。
- 仅存的 GEMM 研究项：以降低 accumulator/寄存器换取 2 CTA，或重做跨 warpgroup/Multi-CTA 流水。没有新的资源与数值证据前，不把 GEMM 列为短期主投入。

### 6.2 Producer 优先项

1. **小 M dispatch（已完成）**：rank64、M≤512、K≤3072 走 fused，NCU 20.58→18.18us、真实形状加权 producer 52.92→51.01ms/step；rank128、M≤256、K≤3072 走 K128 async split-K，NCU 24.80→12.77us、官方 Event 36.74→29.98us。rank128/M512 保持 K256，长 K 和大 M 不变，证据见 `svdq-nvfp4-cute.md` §31。
2. **split-K bank-conflict 路线关闭**：B CACHEGLOBAL 将 LDGSTS 冲突 282.9K→0，但 NCU 57.25→57.66us；64-bit copy 将冲突升至 603.8K 且更慢。不要再做仅以降低冲突计数为目标的 B layout/copy 微调。
3. **uni 路径保持现状**（`quantize_uni.cu`，rank32/K≤3072 四格）：手搓 MMA/ldmatrix + 手写 swizzle，42.9K conflict、occupancy 29.7% 虽可改，但改造与回归成本高，不作为近期项。
4. **大 M fused 不再优化**：rank32/16K 的 DRAM/Mem 已 90.2%，是带宽墙；小 M fused dispatch 不改变这个结论。
5. **剩余小 M 差距**：M=1 的 producer 仍比 native 慢。若继续，先研究低 CTA 并行度与调用开销；完整 FLUX e2e 稳定性仍属 CDK-5，不能用本项替代。

### 6.3 调度与验收

- row-major dispatch 当前为 `M>=16384 || (M>=8192 && N>=6144)` → main-first，其余 LoRA-first；来自同进程 7轮×50次交错 A/B，main-first 仅赢6/15。任何改动继续按 M/N/rank 做逐格 A/B。
- 微优化保留门槛：全矩阵 NSYS 正向约0.1%，且不能出现稳定 >0.2% 回退。
- 数值：GEMM 调度等价要求固定输入 bitwise；producer 改布局要求 qact/ascales bitwise、D rel-L2 <0.01。
- NCU 只做归因，交付性能以 NSYS/官方 Event 为准；跨会话结论必须同轮对照 native。

## 7. 实施纪律

- 动手前写清改动面、注入点、预期指标和验证矩阵。
- 一次只改一个主要因素；profile -> 修改 -> 数值回归 -> 同口径复测。
- 优先保留有实测证据的简单改动，拒绝无收益的复杂布局或调度。
- 性能记录同时写有效、无效和负收益尝试。
- 涉及公开 runtime 默认行为时，同步更新 Python 测试和用户文档；不改变 native checkpoint 格式。
