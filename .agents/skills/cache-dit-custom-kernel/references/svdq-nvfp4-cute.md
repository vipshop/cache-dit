# SVDQ NVFP4 CuTe 重写与性能记录

> 日期：2026-11-05。代码路径均相对 cache-dit 仓库根目录。
> 本文记录当前实现、有效/无效尝试和测量边界；未来工作见 `rfc-cuda-optimizations.md`。

## 1. 最终工作流结构

2026-11-05 晚已从 IDE 历史快照恢复并落地**混合 G2R 工作流**。早期曾复用 native producer 并通过 `lora_act_to_bf16` 转 BF16；现在当前 CuTe producer 直接输出 v1 A/SFA 与 BF16 D，finalize 仍然不存在。

当前 CuTe 路径：

1. `csrc/kernels/svdq/cute/fp4/quantize_w4a4.cu` 直接生成：
   - v1 256-row MMA-fragment interleaved `qact[M,K/2]` E2M1；
   - 与 v1 qact 匹配的 `ascales`；
   - 完成全 K 归约、折入 LoRA scale 的 row-major BF16 `D[M,R]`。
2. `gemm_w4a4.cuh` 中 A/SFA 从 gmem 直接装入 CuTe fragment（G2R）；B/SFB 与 D/L1 走 TMA+smem。
3. NVFP4 主积和 BF16 LoRA-up 积共享同一 FP32 accumulator，最后只舍入一次。
4. 旧 checkpoint 在 `_prepare_cute_pack()` 中做一次加载时转换；forward 不经过旧 pack producer、FP32 scratch 或 finalize。

数值验收已覆盖 rank 0/32/128、独立 producer（identity smooth 下 v1 qact 与 native 逐字节一致，ascales 值集合一致）、真实量化 Linear 的 native/hybrid parity、INT4 隔离和 FLUX 出图。随机有限量化状态下高层 hybrid/native relative L2 约 0.00215；91 个 SVDQ kernel/runtime 测试通过。

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

## 8. GEMM TMA/sync 实施结果（2026-11-05）

测量条件：SM120、M=4096/N=K=3072/rank=32、BN256/BK128/S3，warmup=3、iters=15，每个 kernel 19 个实例；使用本轮新建的独立 GEMM harness，不能与其他章节的旧 harness 绝对延迟直接拼接。

| 版本 | nsys mean（三次，us） | nsys median（三次，us） | 结论 |
|---|---|---|---|
| 修改前 CuTe | 137.339 / 137.342 / 137.280 | 137.440 / 137.216 / 137.472 | 基线 |
| 最终 CuTe | 137.405 / 137.395 / 136.943 | 137.632 / 137.696 / 136.800 | 中位数变化约 +0.14%，属于噪声 |
| native | 118.321 / 118.403 / 118.220 | 117.504 / 117.600 / 117.440 | 最终 CuTe 仍慢约 17% |

最终保留的协议改动：

- 仅由 `raw_tid == 0` 初始化 full/empty mbarrier，并在 CTA sync 前执行 `fence_barrier_init()`；这是正确性修复。
- producer 使用初始 empty phase=1，删除 consumer 初次 empty arrive；这是标准 start state，稳态收益未超过噪声。
- 删除 producer、NVFP4 consumer 和 LoRA consumer 的纯 TMA load-side proxy fence；output STSM 到 TMA store 的 fence 保留。100 轮 bit-exact、memcheck 和 synccheck 均通过，但不宣称性能收益。

基线 NCU：SM 70.48%、Memory 67.25%、DRAM 8.61%、occupancy 17.77%、168 regs、零 local spill；`math_pipe_throttle=5.898/issue`、`wait=4.454/issue`，shared-load bank conflicts 约 103 万。最终 NCU duration 116.448us，基线 116.864us，仅用于归因，不能替代上述 nsys 结论。

| 尝试 | 实测结果 | 状态与约束 |
|---|---|---|
| 仅提前 empty release | 137.472us；`wait` 4.463 -> 4.464/issue | 无独立收益，已回退 |
| 统一循环的 K-block 流水 | 401.472us；local store/load 16.83M/17.66M sectors，stack 216B | 严重 spill，已回退 |
| steady/tail 拆分 + stage-local views | 144.000us；`wait` 降至 3.684，但 local store/load 0.676M/1.253M | wait 改善被 spill 抵消 |
| 直接 read_stage 索引 | 138.944us；local store/load 36.9K/49.2K | 仍慢于 P0 约 1.4%，未保留 |
| BK128 使用 CUTLASS next-copy-before-MMA | 410.528us；local store/load 16.77M/18.11M | fragment 活跃区间膨胀，禁止直接照搬；BK64 更不能覆盖唯一槽后再 MMA |
| consumer 232 registers | 143.648us；stack/local sectors 与 224 完全不变 | 不改变 ptxas 已生成代码，已回退到 224 |
| consumer 初始化移到 producer 分流后 | 137.792us，start-state-only 为 137.280us | 无收益，已恢复原位置 |

验证：七组独立 GEMM 覆盖 BK128/BK64、rank 0/16/32/128、`alpha != 1` 和 `nKt+nRt < kStages`，最终 100 轮全部与修改前 golden bit-exact；相关 pytest 88 passed；memcheck/synccheck 均为 0 errors。最终默认实例保持 168 regs、STACK=0、LOCAL=0。证据保存在 `.tmp/svdq-nvfp4-cute/gemm-mainloop/`。

**结论**：本轮没有形成可宣称的 GEMM 加速，CDK-4 继续进行中，默认 runtime 保持 native。后续应针对 shared 指令量、fragment 生命周期或新的结构性方案提出新证据，不重复上述 K-block/prologue 变体。

## 9. CuTe vs native NCU 归因与结构实验（2026-11-05）

条件：M=4096/N=K=3072/rank=32，CuTe 为 BN256/BK128/S3/384T，native 为 `GEMMConfig_W4A4<1,0>`/256T；`--set full`、skip 3。NCU 的 duration 受 replay 影响，不能与 nsys 拼接；下表只用于归因。

| 指标 | CuTe | native | 读法 |
|---|---:|---:|---|
| shared 指令 | 2.654M | 0.295M | CuTe 多 9.0 倍 smem→RF 路径 |
| shared load 指令 | 1.130M | 0.098M | A/B/SFA/SFB 都先经 smem |
| shared bank conflict | 1.036M | 0.026M | CuTe 约 39 倍冲突 |
| global load 请求 | 0.197M | 1.923M | native 直接 fragment-aligned gmem load，多 9.8 倍 |
| global load sectors | 0.197M | 27.771M | native 请求更碎，依赖 L2 |
| L1 / L2 hit | 94.55% / 87.26% | 65.22% / 85.54% | native 靠大 L2 吸收重复读 |
| local store / load sectors | 0 / 0 | 1.032M / 0.025M | native 反而有 spill |
| registers / smem | 168 avg / 85.0 KiB | 252 / 18.4 KiB | native 无 3-stage TMA ring |
| waves / CTA threads | 3.49 / 384 | 3.49 / 256 | 两者均为 1 CTA/SM |
| tensor active | 81.59% | 74.25% | 相对各自 profiled elapsed |
| math_pipe_throttle | 6.000/issue | 5.783/issue | 主要 stall 接近 |
| wait | 4.471/issue | 4.102/issue | CuTe 多 barrier/TMA 等待 |
| sleeping / barrier | 1.724 / 0.224 | 0 / 0 | CuTe 独有 mbarrier 等待成本 |
| long_scoreboard | 0.666/issue | 1.458/issue | native 主要等待 global load |

**归因**：CuTe 用 TMA 把 global load 请求减少约 9.8 倍，但把成本转移到 9 倍 shared 指令、39 倍 bank conflict 和额外的 sleeping/barrier；native 没有 smem/TMA 环，付出碎片化 global load 和 spill 的代价，仍靠 L2 与更低同步开销取得约 17% wall-clock 优势。下一步若继续，应直接降低 shared 指令/冲突或做 fragment-aligned pack，不应再围绕 stage 周转微调。

结构实验：

| 实验 | 实测 | 结论 |
|---|---|---|
| persistent CTA（110 CTAs，跨 tile 连续 phase + store-done drain） | 143.520us；168 regs、零 spill；NCU 118.88us | 比协议基线慢约 4.4%。output overlay 迫使同一 CTA 下一 tile 等上一 tile TMA store，丢失原 CTA 间 overlap，已回退 |
| light producer（288T，producer-last warp，dec32/inc224 成对） | 初版 launch failure 的根因是 32 个 producer lane 同时 `arrive_and_expect_tx`；修复为单物理 lane 发射后，verify 仍 hang | 路线已由用户停止，未作为候选；不能把失败归因于 dec/inc 配对 |
| 3WG（2 个 consumer WG 独立子 tile） | 未实施 | 当前 384T 已是物理 3WG；调研判断改 CuTe 对象组织后指令/字节/barrier 拓扑等价，用户决定先停 |
| BN256→BN128，BK128/S3 | 168.992us；regs 仍 168、smem 57.3 KiB、waves 3.49→6.98、bank conflict +42% | 慢约 23%，小 tile 不自动降 ptxas 分配也不提升 occupancy，已回退 |
| BM128→64 | 未实施 | `SfSmemAtom` 的 `kMN/128` 会为 0，且 SF gmem/TMA 按 128-row atom 组织；需先重设计 scale 布局 |

证据：`profile/ncu_compare_{cute,native}.csv`、`compare_ncu.py`、`nsys_persistent_v1_1*`、`nsys_tile_bn128_s3_1*`，均位于 `.tmp/svdq-nvfp4-cute/gemm-mainloop/`。

## 10. 历史 hybrid G2R 恢复（2026-11-05）

历史混合版源码没有 commit，但 IDE 的 `User/History/-74e8d26e/` 保留了完整 `.cuh` 编辑快照；最终回退到全 TMA 前的版本为 `aE1h.cuh`（2026-11-04 11:19:45，56195 B，md5 前缀 7cf1c017）。当前 `gemm_w4a4.cuh` 由该快照恢复，并只做以下适配：

- 命名空间改为 `svdq::cute_kernels`，kernel/launcher 名称统一；
- mbarrier init 改为唯一 `raw_tid == 0` 并补 `fence_barrier_init()`；
- A/SFA 保持 G2R，B/SFB/D/L1 保持 TMA，单 FP32 accumulator、LoRA overlay 和 output store 不变。

当前 CuTe producer 不再写 standard row-major activation，而是在原量化与 BF16 D 计算不变的前提下，把每个 u32 直接写到 native v1 pack 的 MMA-fragment 位置：

```text
qact word base = ((row/256)*(K/64) + kt)*8 + (row%256)/32
local          = 16*(row%8) + ((row%16)/8) + 128*((row%32)/16)
dest_word      = base*256 + local + half*2 + w*4
scale word     = base*32 + 4*(row%8) + (row%32)/8
```

identity smooth 下，当前 producer 的 v1 qact/ascales 与 native quantize 逐字节一致；随机 smooth 的差异来自既有的 BF16 smooth 舍入，不是布局错位。仍然没有 `lora_act_to_bf16` 或任何 finalize。

性能（同输入、同 shape）：

| 口径 | hybrid | native | 全 TMA | 结论 |
|---|---:|---:|---:|---|
| standalone kernel M=1024，nsys median | 30.432us | 31.040us | 36.640us | hybrid 比 native 快约 2.0%，比全 TMA 快 20.4% |
| standalone event M=4096 | 114.752us | 约117.5us（nsys） | 133.546us | hybrid 比 native 约快 2.3%，比全 TMA 快 16.4% |
| 完整 Linear M=1024，event median | 63.024us | 63.888us | - | hybrid 为 native 的 0.986x |
| 完整 Linear M=4096，event median | 166.016us | 168.176us | - | hybrid 为 native 的 0.987x；relative L2 0.00215 |

验证：`tests/kernels/test_ops_dispatch.py`、`test_svdquant_cute.py`、`test_svdquant_runtime.py`、`test_svdquant_source_manifest.py`、`test_quantize_config.py` 共 91 passed。producer 测试新增 native v1 pack parity、非 256 padded batch 负向用例，并保留 0.12 去量化精度门槛；另有一条真实量化 Linear 的 native/hybrid 端到端 parity。恢复依据与 standalone 证据在 `.tmp/svdq-nvfp4-cute/g2r-recovered/`。

**边界**：M 必须是 256 倍数（v1 以 256 行为块）；N 必须为 128 倍数。hybrid 目前约 2% GEMM / 1.3%–1.5% 层收益，尚未达到 e2e 5% 目标。下一步应优化 v1 producer 的分散写回或减少 G2R 重复 load，而不是回到全 TMA。
