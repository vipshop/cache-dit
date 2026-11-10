# Cache-DiT CUDA Kernel 优化 RFC

> 状态：Active ｜ 日期：2026-11-05
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
| CDK-2 | Producer kernel 性能追平/略优 | 已完成 | PRO 5000 官方 bench 5 shape × rank32/64/128：producer geomean 1.128× vs native，15/15 快（1.060×–1.212×）；证据见 `svdq-nvfp4-cute.md` §29 |
| CDK-3 | Producer 改用 CuTe TiledMMA/TiledCopy | 已完成 | rank32/64/128 拆分 TU，fused + K128 async split-K + K256 split-K；qact/ascales bitwise golden 通过 |
| CDK-4 | GEMM 稳态快于 native 至少 5% | 进行中 | CUDA Event kernel-only geomean 1.013× vs native；仍未达5%，且 rank64/M8192 两个 e2e 点回退，现为主瓶颈 |
| CDK-5 | 完整 CuTe 工作流 e2e 快至少 5% | 进行中 | 官方 bench e2e geomean 1.038×，13/15 快；rank64/M8192 为 0.976×/0.988×，未达稳定5% |
| CDK-6 | 可重复 ccache 构建链路 | 已完成 | `build_fast.sh` editable + nvcc shim 验证通过 |

完成任一项时，同步更新本表、下方清单和 `SKILL.md` 的当前状态摘要。若实验否定某方案，将方案移到性能记录的已证伪表，不要删除证据。

## 3. 完成状态清单

- [x] CDK-1：独立 CuTe producer + A/SFA G2R、B/SFB/D/L1 TMA GEMM；fused 和 split-K 均输出 native qact/ascales 与 row-major BF16 D。
- [x] CDK-2：producer 官方 bench 5 shape × 3 rank 全格快于 native，geomean 1.128×；rank32/64/128 无回退点。
- [x] CDK-3：`quantize_impl.cuh` + `quantize_r32/64/128.cu`，TiledMMA/TiledCopy/swizzle、K128 double-A async、K256 split-K、rank32 大 M fused；K64/192/3136 tail 与 golden 已验证。
- [ ] CDK-4：GEMM 快于 native 至少 5%。kernel-only geomean 1.013×；producer 收益明显大于 e2e，下一步集中优化 W4A4 GEMM shared/TMA/同步和 epilogue。
- [ ] CDK-5：完整工作流 e2e 快至少 5%。当前 geomean 1.038×；优先消除 rank64/M8192 的 0.976×/0.988× 回退，再评估默认 runtime。
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

FlashInfer/CUTLASS 的 K-block 顺序和 producer start state 已完成对照实验；start state 保留为标准协议，K-block 变体均已回退。最新 NCU 归因显示真正差距是 shared 指令/bank conflict 与同步开销，下一步应针对该路径或 fragment-aligned pack，而不是继续改调度外形。

GEMM 达标后再讨论默认 runtime 切回 CuTe；最终仍需 CDK-5 独立验收。

## 6. 实施纪律

- 动手前写清改动面、注入点、预期指标和验证矩阵。
- 一次只改一个主要因素；profile -> 修改 -> 数值回归 -> 同口径复测。
- 优先保留有实测证据的简单改动，拒绝无收益的复杂布局或调度。
- 性能记录同时写有效、无效和负收益尝试。
- 涉及公开 runtime 默认行为时，同步更新 Python 测试和用户文档；不改变 native checkpoint 格式。
