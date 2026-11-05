# SVDQ NVFP4 CuTe 重写与性能记录

> 日期：2026-11-05。代码路径均相对 cache-dit 仓库根目录。
> 本文记录当前实现、有效/无效尝试和测量边界；未来工作见 `rfc-cuda-optimizations.md`。

## 1. 最终工作流结构

旧混合路径曾复用 native activation producer，再通过 `lora_act_to_bf16` 将 v1 FP32 LoRA scratch 转成 BF16。该适配层已彻底删除。

当前 CuTe 路径：

1. `csrc/kernels/svdq/cute/fp4/quantize_w4a4.cu` 直接生成：
   - row-major `qact[M,K/2]` E2M1；
   - CUTLASS 128x4、512B atom 的 `ascales`；
   - 完成全 K 归约、折入 LoRA scale 的 row-major BF16 `D[M,R]`。
2. `gemm_w4a4.cuh` 对 A/B/SFA/SFB 使用 TMA，并从 smem 经 LDSM/SF copy 取 fragment。
3. NVFP4 主积和 BF16 LoRA-up 积共享同一 FP32 accumulator，最后只舍入一次。
4. 旧 checkpoint 在 `_prepare_cute_pack()` 中做一次加载时转换；forward 不经过旧 pack/producer/finalize。

数值验收已覆盖 rank 0/32/128、非对齐逻辑 shape、独立 producer、全 TMA GEMM、高层 Linear、INT4 隔离和 FLUX 出图。高层 CuTe/native 输出 relative L2 约 0.003264；独立 GEMM 对 FP32 反量化参考 relative L2 约 0.001662。

## 2. 性能测量条件

除历史 e2e 外，以下 kernel 数据使用：RTX PRO 5000、SM120、M=4096、N=K=3072、rank=32。

- nsys `cuda_gpu_kern_sum`：kernel-only 性能判据。
- CUDA Event：包含 op-level 分配/清零/launch。
- NCU `--set full`：只用于归因；其 duration 不可与 nsys/Event 直接比较。

最终稳态 nsys（warmup=3、iters=15；含 setup 调用）：

| kernel | CuTe mean | CuTe median | native mean | native median | 结论 |
|---|---:|---:|---:|---:|---|
| fused producer | 32.462us | 32.192us | 31.993us | 31.936us | 基本追平，median 仍慢约 0.8% |
| GEMM + LoRA-up | 133.805us | 133.823us | 115.365us | 115.135us | CuTe 慢约 16.3%，未达标 |

最终 CUDA Event（warmup=3、iters=15）：

| 调用 | native | CuTe | 说明 |
|---|---:|---:|---|
| `quantize()` | 34.933us | 33.781us | CuTe op-level 略快 |
| `forward_quant()` | 116.329us | 135.439us | GEMM 侧仍慢 |
| quantize + forward | 148.211us | 167.456us | 整体慢约 13.0% |

不同版本/口径不可拼接成单次收益。producer 的三次冷 profile 曾出现 0.6%-1.1% median 小胜，但最终稳态为约 0.8% 小负，因此记录为“接近持平”，不宣称稳定领先。

## 3. 有效优化尝试

按实施顺序记录：

| 改动 | 证据与效果 | 状态 |
|---|---|---|
| 单 warp 负责 32 行完整 K | 初版 producer 约 204us nsys；grid 仅 128x32，NCU waves 0.07、occupancy 2.44%，local-memory 访问显著 | 已被替代 |
| 8 warps/CTA 切分 K + 编译期 rank tile + shared D 归约 | 消除动态 accumulator 索引；NCU waves 升至 1.16，occupancy 约 16.7%，Event quantize 从约 195.7us 降至 43.3us | 保留 |
| qact 改为每行两个 16B `uint4` store | Event 约 43.3us -> 40.4us；NCU duration 约 53.9us -> 47.6us | 保留 |
| A 先写 shared staging，再 `ldmatrix.x4` | 消除每 lane 4B 的半 sector A load；global excess sectors 从约 47% 降至 32% | 保留 |
| 同一 k64 的 LoRA B fragment 在 mt0/mt1 复用 | lora_down global load 减半；nsys median 从约 34.78us 降至 33.31us | 保留 |
| A staging 改为每 lane 16B、8 lanes/row | 每 k64 的 staging store 从 32 条降至 8 条；解决 PC sampling 中 STS 热点 | 保留 |
| smooth fragment 在 mt0/mt1 复用 | 每个 k64 的 smooth load 减半；最终 Event quantize 约 33.78us | 保留 |

以上数值改动均通过 producer/Linear 回归；性能数值仍须按各自测量轮次理解。

## 4. 无效或负收益尝试

| 尝试 | 结果 | 后续约束 |
|---|---|---|
| 只给 D partial 增加 shared padding | NCU duration 和 bank-conflict 指标基本不变 | 不再把 stride padding 当作主优化方向 |
| rank=32 的 `stride=36 + col^row` XOR shared swizzle | 正确但耗时无改善，约 39.89us vs 39.83us（噪声内） | 不为无实测收益的复杂性继续投入 |
| 在旧 producer 与 CuTe GEMM 之间保留 `lora_act_to_bf16` | 功能正确，但适配 launch/流量抵消 GEMM 收益 | 已删除，不得恢复 |
| 第一版完整独立工作流直接做 FLUX e2e | CuTe 12.80s vs native 6.08s；主因是 producer 初版约 204us | 先修 producer 并行度，再看 e2e |
| 根据单 GEMM NCU 约 5% 的局部结果推断 e2e | 被 layer/e2e 数据否定 | 验收必须回到 nsys kernel-only 和完整 e2e |

这些记录是已证伪清单；重新尝试前必须提出新的瓶颈证据。

## 4.1 GEMM stages/tile 扫描结论（2026-11-04）

条件：SM120、M=4096、N=K=3072、rank=32，三次独立 nsys。

- 最佳配置 `128x256/BK128/S2`：133.76µs，native 115.10µs，仍慢 **15.89%**。
- S3：134.26µs；S1/S4 更慢。**stages 不是主因**。
- 256x128 8-warp：数值错 4.3%，已回退。
- 256x128 16-warp：TiledCopy 编译失败，已移除。
- 256x256：输出 staging 131072B > 99KiB，被资源门禁拒绝。
- light producer 双 buffer：134.8µs，无收益。
- SF copy 向量化、epilogue 无 scale/bias 快路径：数值正确但无性能收益。
- NCU 归因：tensor 指令量与 native 相同；shared 指令 2.65M vs native 0.295M，**9 倍差距**；主 stall 为 math_pipe_throttle 与 wait。

**结论**：不要在无新 NCU 证据时再次盲目扫描 stages/smem 或重复 256x128/256x256；下一轮优先考虑 persistent CTA 或 direct g2r + pack 重排。

## 5. 当前限制与优化线索

### Producer

- 当前仍使用手写 `mma.sync/ldmatrix`，未来用 CuTe `TiledMMA/TiledCopy` 重写时保持数值和布局不变。
- SFA 的 4 条 32-bit store 仍有 excess sectors。一个 32-row CTA 只拥有 128-row SF atom 的一个 row quarter，每行只能写 4B；不能直接扩大成 16B store，否则会覆盖其他 row group。
- 若继续优化 SFA，需要重新设计 CTA 覆盖或协同写出，不是把现有 store 强制改成 `uint4`。
- 当前目标仍是稳定追平或略优 native；最终稳态 median 尚有约 0.8% 小负。

### GEMM

- 当前 GEMM 是比 producer 更大的差距来源。
- stages/tile 扫描已完成，结论见 §4.1；**不要再次盲目扫描 stages/smem**。
- 后续候选按优先级：persistent CTA、direct g2r + pack 重排、NVFP4 与 BF16 LoRA 分属不同 consumer WG。
- 当前为 1 个 TMA producer WG + 8 个 consumer warps。若改为 1 producer WG + 2 个独立 consumer WG，必须处理两套 MMA 的 FP32 累加/归约，不能假定调度拆分天然更快。
- 源码 `gemm_w4a4.cuh` 已记录对应 TODO。

### Direct g2r 实验现状（2026-11-05，已回退）

- 已尝试独立 `gemm_w4a4_direct_kernel`：不用 TMA/smem，A/B/SFA/SFB/LoRA 直接 gmem→register，主循环双 buffer。
- 数值始终错误；`AutoVectorizingCopy`、`UniversalCopy`、自定义 `SubbyteCopy` 均不能修复。
- 根因不是 copy atom，而是 **pack 布局不匹配**：
  - native pack 按 MMA fragment 直接排布，每个 `uint4` 就是一个 m16n8k64 fragment；
  - CuTe producer 输出 standard row-major E2M1，适合 TMA，不适合 direct g2r。
- 256x128 8-warp direct 变体同样数值错误；问题与 tile 无关。
- 代码已回退；实验补丁保存在 `.tmp/svdq-nvfp4-cute/gemm-tune/abandoned/g2r_direct_experiment.patch`。
- 下次继续 direct 路线时，必须先解决 fragment-aligned pack 或把 CuTe fragment reinterpret 成 native `uint4` 布局，否则不要重复同一实验。

### 默认 runtime

在 GEMM 和完整 e2e 达标前，NVFP4 默认使用 native，CuTe 通过 `runtime_kernel="cute"` 显式启用。不得因 producer 接近持平就重新打开默认 CuTe。

## 6. 构建记录

`tools/build_fast.sh` 已验证可用于 editable 构建，nvcc ccache shim 正常生效。首次切换 shim 的完整构建约 273s；该次包含编译路径变化和缓存填充，不能据此宣称加速比例。后续迭代统一复用同一 shim 和 ccache 配置。

## 7. 本地证据

历史研究、日志、nsys/NCU report 和 FLUX 图像主要保存在 `.tmp/svdq-nvfp4-cute/`。这些是本机分析证据，不是可替代正式测试的交付物。关键最终 profile：

- `profile/nsys_quant_bench_final.nsys-rep`
- `profile/nsys_kern_sum_final.csv`
- `profile/quantize_workflow_after_smooth_reuse.json`
- `profile/ncu_cute_quant_v6.ncu-rep`
