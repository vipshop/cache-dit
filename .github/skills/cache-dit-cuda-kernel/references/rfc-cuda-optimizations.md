# Cache-DiT CUDA Kernel 优化 RFC

> 状态：Active ｜ 日期：2026-11-04
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
| CDK-1 | 独立 CuTe producer 与工作流解耦 | 已完成 | 无旧 producer、v1 scratch、G2R 反解或 finalize |
| CDK-2 | Producer kernel 性能追平/略优 | 进行中 | 当前 median 约慢 0.8%，需继续稳定复测 |
| CDK-3 | Producer 改用 CuTe TiledMMA/TiledCopy | 待开始 | 保持当前性能与数值，利用 TMA/WS/多级流水 |
| CDK-4 | GEMM 稳态快于 native 至少 5% | 待开始 | 当前 median 约慢 16.3% |
| CDK-5 | 完整 CuTe 工作流 e2e 快至少 5% | 待开始 | 多次 FLUX native/CuTe 对照 |
| CDK-6 | 可重复 ccache 构建链路 | 已完成 | `build_fast.sh` editable + nvcc shim 验证通过 |

完成任一项时，同步更新本表、下方清单和 `SKILL.md` 的当前状态摘要。若实验否定某方案，将方案移到性能记录的已证伪表，不要删除证据。

## 3. 完成状态清单

- [x] CDK-1：独立 CuTe producer + 全 TMA A/B/SFA/SFB GEMM；删除 `lora_act_to_bf16` 和 forward 中旧 native 依赖。
- [ ] CDK-2：producer 稳态性能追平/略优。当前最终 nsys median 为 CuTe 32.192us、native 31.936us。
- [ ] CDK-3：将手写 `mma.sync/ldmatrix` producer 迁移为 CuTe atoms；不得恢复旧 pack 或降低当前性能。
- [ ] CDK-4：GEMM 快于 native 至少 5%。当前最终 nsys median 为 CuTe 133.823us、native 115.135us；按延迟降低 5% 计算，同条件目标约 109.4us 或更低。
- [ ] CDK-5：完整工作流 e2e 快至少 5%。达标前 NVFP4 默认 runtime 保持 native。
- [x] CDK-6：使用 `tools/build_fast.sh` 和 `tools/nvcc` 的 ccache 构建链路。

## 4. CDK-3：Producer CuTe 化设计约束

- 保留标准 row-major E2M1、CUTLASS 128x4 SFA 和 row-major BF16 D 输出。
- 保留 CTA 内 K 切分与 D 归约，避免重新引入 global FP32 scratch 和二次 finalize。
- 评估 CuTe `TiledMMA/TiledCopy`、TMA、多级流水和 warp specialization；不能仅为“使用 CuTe”而牺牲已验证的 coalescing 和 occupancy。
- 每次改动先跑 identity LoRA-down、随机 D、标准 A/SFA 和高层 Linear，再做 nsys/NCU。
- 当前手写实现是性能对照基线，不得在迁移完成前删除。

## 5. CDK-4：GEMM 候选方案

按单个假设逐项验证：

1. **Persistent CTA 调度**：减少小 M/波次尾损和重复 CTA 初始化；需核对每 tile 的 stage phase、descriptor 生命周期和 output overlay 重置。
2. **Stages/smem 复用调优**：扫描 stage 数和 BK/BN 组合；NCU 检查 TMA 供给、consumer wait、smem 占用和 occupancy。不得破坏 A/B 与 D/L1/output overlay 的 empty-stage barrier。
3. **独立 consumer WG**：考虑 1 个 producer WG + 2 个 consumer WG，将 NVFP4 与 BF16 LoRA MMA 分给不同 WG。必须解决共享 FP32 accumulator 或跨 WG 归约成本；先做原型测量，不能直接作为主路径。
4. **指令与装载路径**：结合 SASS/PC sampling 检查 LDSM、SF copy、MMA 依赖链和 epilogue，不凭理论 TFLOPS 推断收益。
5. **Descriptor/调度开销**：使用 nsys 区分 kernel 本体与 host/descriptor 成本；kernel-only 优化不能用 host 缓存收益替代。

GEMM 达标后再讨论默认 runtime 切回 CuTe；最终仍需 CDK-5 独立验收。

## 6. 实施纪律

- 动手前写清改动面、注入点、预期指标和验证矩阵。
- 一次只改一个主要因素；profile -> 修改 -> 数值回归 -> 同口径复测。
- 优先保留有实测证据的简单改动，拒绝无收益的复杂布局或调度。
- 性能记录同时写有效、无效和负收益尝试。
- 涉及公开 runtime 默认行为时，同步更新 Python 测试和用户文档；不改变 native checkpoint 格式。
