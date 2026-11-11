# SVDQ NVFP4 CuTe 重写与性能记录

> 日期：2026-11-05。代码路径均相对 cache-dit 仓库根目录。
> 本文记录当前实现、有效/无效尝试和测量边界；未来工作见 `rfc-cuda-optimizations.md`。

## 1. 最终工作流结构

2026-11-05 晚已从 IDE 历史快照恢复并落地**混合 G2R 工作流**。早期曾复用 native producer 并通过 `lora_act_to_bf16` 转 BF16；现在当前 CuTe producer 直接输出 native A/SFA 与 BF16 D，finalize 仍然不存在。

当前 CuTe 路径：

1. `csrc/kernels/svdq/cute/fp4/quantize_w4a4.cu` 直接生成：
   - native 256-row MMA-fragment interleaved `qact[M,K/2]` E2M1；
   - 与 native qact 匹配的 `ascales`；
   - 完成全 K 归约、折入 LoRA scale 的 row-major BF16 `D[M,R]`。
2. `gemm_w4a4.cuh` 中 A/SFA 从 gmem 直接装入 CuTe fragment（G2R）；B/SFB 与 D/L1 走 TMA+smem。
3. NVFP4 主积和 BF16 LoRA-up 积共享同一 FP32 accumulator，最后只舍入一次。
4. 旧 checkpoint 在 `_prepare_cute_pack()` 中做一次加载时转换；forward 不经过旧 pack producer、FP32 scratch 或 finalize。

数值验收已覆盖 rank 0/32/64/128、独立 producer（identity smooth 下 native qact 与 native 逐字节一致，ascales 值集合一致）、真实量化 Linear 在 rank32/64/128 的 native/hybrid parity、INT4 隔离和 FLUX 出图。随机有限量化状态下高层 hybrid/native relative L2 约 0.00215；94 个 SVDQ kernel/runtime 测试通过。

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

> 2026-11-11 更新：最新结论与优先级见 §30。本章保留 2026-11-05 历史线索；其中 persistent、A/SFA 全 TMA、fragment-aligned pack 等已被后续证据关闭，不要直接照此开工。

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

当前 CuTe producer 不再写 standard row-major activation，而是在原量化与 BF16 D 计算不变的前提下，把每个 u32 直接写到 native native pack 的 MMA-fragment 位置：

```text
qact word base = ((row/256)*(K/64) + kt)*8 + (row%256)/32
local          = 16*(row%8) + ((row%16)/8) + 128*((row%32)/16)
dest_word      = base*256 + local + half*2 + w*4
scale word     = base*32 + 4*(row%8) + (row%32)/8
```

identity smooth 下，当前 producer 的 native qact 与 native quantize 逐字节一致；ascales 在 native 中按 `[K/16,M]` 存储，当前 native pack 为扁平顺序，但两者的 scale 值集合完全一致。随机 smooth 的差异来自既有的 BF16 smooth 舍入，不是布局错位。仍然没有 `lora_act_to_bf16` 或任何 finalize。

性能（同输入、同 shape）：

| 口径 | hybrid | native | 全 TMA | 结论 |
|---|---:|---:|---:|---|
| standalone kernel M=1024，nsys median | 30.432us | 31.040us | 36.640us | hybrid 比 native 快约 2.0%，比全 TMA 快 20.4% |
| committed kernel M=4096，nsys median | 116.80–117.44us（三次 117.440/116.800/117.344） | 117.536us（同轮） | 约137.4us | 比 native 快约 0.3%，比全 TMA 快约 14.7%；不再引用 event 114.75us 的跨工具比较 |
| 完整 Linear M=1024，event median | 63.024us | 63.888us | - | hybrid 为 native 的 0.986x |
| 完整 Linear M=4096，event median | 166.016us | 168.176us | - | hybrid 为 native 的 0.987x；relative L2 0.00215 |
| kernel M4096/N3072/K3072，rank64，event median | 124.14us | 124.34us | - | hybrid 快约0.16% |
| kernel M4096/N3072/K3072，rank128，event median | 130.28us | 132.46us | - | hybrid 快约1.65% |

大 shape 的 rank64/128（GPU4，Event 50 次 median，各3轮）：

| shape / rank | native | hybrid | hybrid/native |
|---|---:|---:|---:|
| 8192x3072 / 64 | 214.33us | 215.22us | 1.0042（慢0.42%） |
| 8192x3072 / 128 | 229.93us | 226.64us | 0.9857（快1.43%） |
| 4096x6144 / 64 | 214.30us | 214.39us | 1.0004（持平） |
| 4096x6144 / 128 | 228.54us | 226.71us | 0.9920（快0.80%） |
| 8192x6144 / 64 | 421.39us | 433.48us | 1.0287（慢2.87%） |
| 8192x6144 / 128 | 456.63us | 455.18us | 0.9968（快0.32%） |
| 16384x8192x8192 / 64 | 2475.42us | 2587.15us | 1.0451（慢4.51%） |
| 16384x8192x8192 / 128 | 2674.07us | 2645.22us | 0.9892（快1.08%） |

rank128 在大 shape 均略优于 native；rank64 在 `8192x6144`/16K 明显落后，rank 对 LoRA stage 数和流水行为的影响仍需单独归因。

验证：`tests/kernels/test_ops_dispatch.py`、`test_svdquant_cute.py`、`test_svdquant_runtime.py`、`test_svdquant_source_manifest.py`、`test_quantize_config.py` 共 94 passed。producer 测试覆盖 rank32/64/128，并包含 native native pack parity、非 256 padded batch 负向用例和 0.12 去量化精度门槛；真实量化 Linear 的 native/hybrid 端到端 parity 覆盖 rank32/64/128。恢复依据与 standalone 证据在 `.tmp/svdq-nvfp4-cute/g2r-recovered/`。

**边界**：M 必须是 256 倍数（native pack 以 256 行为块）；N 必须为 128 倍数。M4096/N3072 下 hybrid kernel-only 只比 native 快约0.3%，完整层约1.3%–1.5%；但大 shape 下 native 反超（16Kx8Kx8K 快6.58%，见 §12），不能宣称 hybrid 普遍快于 native。下一步应优先解释大 shape 退化，而不是继续微调调度。

## 11. 为什么 hybrid G2R 更快，以及旧结论的失效边界（2026-11-05）

同 shape（M=4096/N=K=3072/R32）、同 harness 的 NCU full 三方对比。NCU duration 受 replay 影响，不作为延迟结论，只用于机制归因；wall-clock 使用同轮 nsys。

| 指标 | hybrid G2R | 全 TMA | native | 读法 |
|---|---:|---:|---:|---|
| nsys kernel median（M4096/N3072/K3072） | 117.2us | 约137.4us | 117.5us | hybrid 比全 TMA 快约14.7%，与 native 基本持平/略快；大 shape 下 native 反超，见 §12 |
| shared 指令 | 2.064M | 2.654M | 0.295M | hybrid 比全 TMA 少22.2% |
| shared load | 0.836M | 1.130M | 0.098M | 少26.1% |
| shared wavefront | 7.283M | 8.893M | 0.647M | 少18.1% |
| shared bank conflict | 0.738M | 1.036M | 0.026M | 少28.8% |
| global load 请求 | 0.786M | 0.197M | 1.923M | hybrid 是全 TMA 的4倍，但仅为 native 的41% |
| global sectors | 6.095M | 0.197M | 27.771M | hybrid 仅为 native 的22% |
| L1 hit | 55.10% | 94.55% | 65.22% | hybrid 的 A G2R 降低 L1 命中，但 L2 吸收 |
| L2 hit | 87.34% | 87.26% | 85.54% | 三者相同量级 |
| DRAM throughput | 8.51% | 8.61% | 9.79% | G2R 没增加 DRAM 压力 |
| math_pipe_throttle /issue | 4.614 | 6.000 | 5.783 | hybrid 最低 |
| wait /issue | 4.369 | 4.471 | 4.102 | 略低于全 TMA |
| long_scoreboard /issue | 2.923 | 0.666 | 1.458 | G2R 的直接 global load 代价 |
| tensor active | 79.79% | 81.59% | 74.25% | hybrid 高于 native |
| registers / smem | 168 / 85.0KiB | 168 / 85.0KiB | 252 / 18.4KiB | hybrid 仍 1 CTA/SM，无 spill |
| local store sectors | 0 | 0 | 1.032M | native 有 spill |

机制结论：

1. **hybrid 的收益主要来自拆掉 activation 的 smem 中转，而不是减少总指令数。** 相对全 TMA，shared 指令/wavefront/bank conflict 分别下降约22%/18%/29%，`math_pipe_throttle` 从6.00降到4.61；静态 SASS 的 `FENCE` 从9条降到2条。A/SFA 不再进入 TMA full barrier，也没有每 stage 的 A LDSM 与 SFA 逐字节 shared copy。
2. **代价是 A 的重复 global load。** 每个 CTA、每个 MMA 线程都直接从 native pack 取 A/SFA，跨 N tile 和 ThrN 广播产生重复请求：global 请求/ sectors 相对全 TMA 升4倍/31倍，L1 hit 从94.5%降到55.1%，`long_scoreboard` 从0.67升到2.92。但这些额外流量有87%在 L2 命中，DRAM 吞吐不变，而且总量仍远小于 native（请求41%、sectors22%）。
3. **hybrid 优于 native 的原因不同：它保留了 weight 的 CTA 内共享。** native 每个 warp 直接从 gmem 取 fragment，global 请求是 hybrid 的2.45倍、sectors是4.56倍，并带约103万 local-store sectors 的 spill；hybrid 把 B/SFB 一次 TMA 进 smem 给8个 consumer warp 复用，regs 168 vs 252，tensor active 79.8% vs 74.3%。所以 hybrid 同时避免了“全 TMA 的 smem 税”和“native 的重复 weight load/spill”。
4. 尚未隔离 A 与 SFA 各自的贡献。SFA 是逐字节 shared copy，A 是 LDSM；从指标看 SFA scalar copy 很可能是主要节省项，但需要“A TMA + SFA G2R”的 A/B 才能定论，不能把全部收益归给 A G2R。

旧全 TMA 优化的失效边界：

| 旧结论 | hybrid 下是否仍成立 | 新证据 |
|---|---|---|
| K-block 流水必 spill、不可用 | **失效**：hybrid 上逐 k-block `load A_k -> copy B_k -> MMA_k` 为 168 regs、零 spill，wait/math/sleeping 各降约5% | 大 shape 复测仍未形成明确收益：`8192x3072` -0.23%、`4096x6144` +0.31%、`8192x6144` +0.20%、`16Kx8Kx8K` +0.21%，平均约0.12%，未保留 |
| stages 调优已完成、S3 最优 | **部分仍成立**：S2 被输出 overlay 的64KiB > pipeline55.3KiB 静态拒绝；S4 的110.6KiB超过99KiB。G2R 只释放未使用的 SFA 3KiB，A/B 区仍被 D/L1 per-stage overlay 占用 | 要增加 stages，必须先解耦 D/L1/output overlay，不能只靠 G2R 释放的 smem |
| persistent CTA 慢4.4% | **未重测，旧机制仍在**：`store_done` 让同一 CTA 下一 tile 等上一 tile 的 output TMA store，丢失 CTA 间 epilogue/mainloop overlap；该串行化与 A 走 G2R 无关 | 只有改成不串行 store 的 persistent 设计才值得重测 |
| BN128 小 tile 慢23% | **需在 hybrid 下重测**：旧数据来自全 TMA；hybrid 的 shared 结构已变，且 BN128 可用 S4 | 未做，不能直接沿用旧结论 |
| direct G2R 因 pack 不匹配不可行 | **对 native pack 仍成立，对 native pack 已失效**：native producer + G2R 已落地 | 当前 hybrid 即该路线 |

下一步优先级：

1. A TMA + SFA G2R 的隔离实验，确认收益来自 A 还是 SFA copy；若 SFA 为主，可只针对 SF 设计更低成本的装载。
2. 降低 A G2R 重复 load：跨 N tile 复用、cache hint、或在 cluster 内共享 A。
3. 解耦 D/L1/output overlay，让 S4 或专用 LoRA buffer 成为可能；这是 stages 继续优化的前提。
4. 优化 producer native 的分散 u32 写回（当前为保持零 finalize 的代价）。
5. persistent 只有在设计出去串行 store 的版本后再测。

## 12. 大 shape 标定：K-block 不放大收益，native 随规模反超（2026-11-05）

> 历史口径：本章为 nsys 标定，当前 CUDA Event 统一基线见 §17。


同 harness、同 nsys 口径，每个配置 3 次 `Med (ns)`。baseline 为 committed hybrid；K-block 为 hybrid 上逐 k-block `A G2R -> B copy -> MMA`；native 为 native SVDQ GEMM。

| shape | hybrid baseline (us) | hybrid K-block (us) | native (us) | K-block vs baseline | native vs baseline |
|---|---:|---:|---:|---:|---:|
| 8192x3072x3072 | 210.0 / 209.7 / 210.0 / 210.8* | 210.7 / 210.2 / 210.9 | 205.1 / 204.7 / 204.6 | -0.23% | **+2.58%** |
| 4096x6144x3072 | 208.8 / 209.0 / 209.2 | 208.4 / 208.4 / 208.4 | 205.2 / 205.7 / 205.7 | +0.31% | **+1.70%** |
| 8192x6144x3072 | 423.4 / 422.2 / 422.0 | 421.0 / 421.7 / 422.3 | 404.1 / 404.0 / 404.6 | +0.20% | **+4.52%** |
| 16384x8192x8192 | 2572.6 / 2573.2 / 2573.9 | 2568.9 / 2568.9 / 2566.1 | 2414.0 / 2413.9 / 2415.3 | +0.21% | **+6.58%** |

\* 第一个 shape 有一个额外的 baseline probe run；其余为三次正式 run。

结论：

- **K-block 的收益不会随 shape 放大，反而被噪声淹没。** 四个 shape 平均只有约0.12%，其中 `8192x3072` 还是负的 -0.23%。虽然 K-block 仍保持 168 regs、零 spill，且 wait/math/sleeping 约降5%，但没有可保留的 wall-clock 指标，故回退。
- **native 的优势随输出规模增加。** M4096/N3072 时 hybrid 仅快约0.3%，但 `16Kx8Kx8K` 下 native 快6.58%。这与全 TMA 对比时的结论不同：不能把“hybrid 比 native 快”当成普适结论。
- 一个待验证的解释是：规模增大后 CTA 数和 L2 复用形态变化，hybrid 的 per-CTA TMA/smem 固定开销和 A 的重复 G2R load 占比上升；native 的 warp-direct load 在更多 waves 下由 L2 更好吸收，且没有 barrier/TMA 环。该解释尚未用 16K NCU 证明，不能写成定论。
- 下一步优先级应改为：先在 `16Kx8Kx8K` 上做 native/hybrid 的 NCU 对比，找出规模相关退化点；K-block、persistent、stages 微调在 native 仍领先6.6%时不是首要矛盾。

## 13. 剩余 0.737M shared bank conflict：来源与消除尝试（2026-11-05）

NCU source/SASS 分组（committed hybrid）：

| 指令 | excessive shared wavefront | 说明 |
|---|---:|---|
| `LDSM`（FP4 B） | 0 | TMA `Layout_K_SW32` + LDSM 无冲突 |
| `STSM`（output） | 0 | 无冲突 |
| 标量 `LDS` | 737,900 | 剩余冲突全部在此 |
| `LDG`（A/SFA G2R） | - | 不走 shared |

top SASS 的 `LDS` 紧跟 `HMMA.16816.F32.BF16`，地址落在 D/L1 的 overlay 区，因此冲突源是 **BF16 LoRA D/L1**，不是 SFB。`LoraCopyAtom = AutoVectorizingCopyWithAssumedAlignment<32>`，D/L1 又是无 swizzle 的 row-major overlay；每 lane 4B、行 stride 64B 使 32 lane 聚到相同 bank group，形成 4-way conflict。

尝试与结论：

| 方案 | 结果 | 结论 |
|---|---|---|
| SFB G2R | conflict 仍 737.9k；global 请求 0.79M→10.2M，stack16B，local load 1.18M sectors，167us | SFB 不是冲突源，已回退 |
| `AutoVectorizingCopyWithAssumedAlignment<128>` | SASS/shared 计数完全不变 | 该 CopyAtom 没有把 LoRA copy 提升到 LDS.128 |
| 直接改 `SM75_U32x4_LDSM_N` + row-major | conflict 仍 737.9k，但从4-way变为16-way LDSM；118.3us | 无 swizzle 的 LDSM 更差，已回退 |
| D/L1 G2R（移除 LoRA TMA/smem） | **conflict 737.9k→66**；global requests+31%、sectors+32%，stack8B/local 67k，121.1us | 冲突可消，但重复/非合并 global load 抵消收益，慢3.3%，已回退 |
| 手写 SW64 composed layout + TMA | 编译通过但端到端数值不一致 | TMA descriptor 的 swizzle 与手写 layout 未对齐，已回退 |
| **canonical `tile_to_shape(SW64 atom,(BM,rankTile,stages))` + LDSM** | **conflict 737.9k→483**；shared load -29%；M4096 三次116.704/116.320/116.960us（baseline 117.20），16K 2562.6 vs2573.3us；零 spill | **正确且保留**；BK64 fallback 用 SW32 atom |

核心经验：

1. **bank conflict 的绝对值不等于瓶颈。** 737.9k excessive wavefront 只占 shared wavefront 约10%；把冲突降到66 的版本反而慢3.3%，因为 LoRA G2R 的 global traffic 和 local spill 更贵。
2. TMA 的 swizzle 只服务于匹配的 copy atom；LoRA overlay 为了保证与 A/B 字节级复用，使用了无 swizzle row-major 布局，因此标量 LDS 冲突是布局选择的结果，不是 TMA 失效。
3. 想在保留 TMA 的前提下消冲突，必须使用 **canonical SW64 TMA atom + 匹配的 LDSM/copy 视图**；手写 `composition(swizzle, row-major)` 不足以让 TMA descriptor 与 consumer 视图一致。
4. **canonical SW64 是正确解**：`tile_to_shape(UMMA::Layout_K_SW64_Atom<ElementLora>{}, (BM,rankTile,stages))` 不能加 `Step`；BK64 fallback 因 rankTile=16 必须改用 SW32。TMA descriptor 与 LDSM view 都由同一 canonical layout 派生后，数值正确、conflict 降到483、零 spill，M4096 约快0.45%、16K 约快0.42%。

## 14. 下一步优先级（2026-11-05）

> 历史章节：已被 §30（2026-11-11）取代；当前从 producer bank conflict 与 e2e 上下文开始，GEMM 不再重复 TMA/stage/warp 级实验。

按当前证据排序，前一项没有结论前不要跳到后一项：

1. **解释大 shape 下 native 反超（未完成）。** 在 `16384x8192x8192` 上做 native/hybrid 的 NCU 对比，定位是 CTA 波次、L2 复用、A/SFA 重复 G2R load 还是 per-CTA TMA/barrier 固定成本导致 native 快 6.58%。产出必须给出具体 counter（shared/global/L2/stall/occupancy）和可验证假设。
2. **canonical SW64 bank conflict（已完成）。** canonical `tile_to_shape(SW64 atom,...)` + LDSM 将 conflict 降到483并保留；BK64 fallback 用 SW32。不要重做 conflict 消除。
3. **降低 hybrid 重复流量（部分完成，主路线受阻）。** A-only TMA 与 native pack 不兼容（rel_l2=1.30），producer coalesced 写回无收益（46.81→47.01us）。只有设计“native bytes 经 TMA/共享广播”的新方案才值得继续。
4. **persistent 只在重新设计 output 生命周期后重测。** 当前 `store_done` 让同一 CTA 的下一 tile 等上一 tile 的 TMA store，和 A 走 G2R 无关；没有去串行化设计就不要重复 110-CTA 版本。
5. **FlashInfer CuTeDSL benchmark（已完成，慢）。** 四 shape 比 native 慢9%–15%，不移植；只保留为外部参照。

不要重复的路线：K-block（hybrid 四 shape 平均仅0.12%）、light producer、S2/S4 stages、BN128 小 tile、全 TMA 回退、D/L1 G2R（conflict 可消但慢3.3%）。canonical SW64 已完成，不再重做 conflict 消除。

## 16. A-only TMA 与 producer coalesced 写回（2026-11-05）

- **A-only TMA 不可行**：TMA 只能搬 standard row-major，当前 producer 输出 native fragment pack；TMA 后直接 LDSM 得到 rel_l2=1.30。除非 producer 额外输出一份 standard A（增加写带宽/内存），否则不能用 TMA 去掉 A 的重复 G2R load。
- **producer coalesced 写回无收益**：把每 warp 的 32 行×K64 qact/SFA 先在 smem 重组，再每 lane 写连续32B/4B。M4096 producer op：baseline 三次 median46.81us，候选47.01us；M1024 候选约慢1%。分散4B store 不是瓶颈，未保留。候选在 `.tmp/svdq-nvfp4-cute/g2r-recovered/quantize_w4a4_coalesced_candidate.cu`。


## 15. FlashInfer CuTeDSL 同 shape benchmark（2026-11-05）

> 历史口径：本章只测了 FlashInfer cute-dsl 的 kernel-only（20 iter），不是当前完整 backend 结论。cutlass-sm120 与 cache-dit 全链对比见后续 FlashInfer 章节。


条件：FlashInfer 源码通过 `PYTHONPATH=/workspace/dev/vipshop/flashinfer` 导入，`mm_nvfp4_svdquant(..., backend="cute-dsl")`，与 native/hybrid 使用同一随机输入、rank=32、CUDA Event median（20 iter）。FlashInfer 使用自己的 standard row-major `nvfp4_quantize_smooth` 输出和 CuTe standard weight pack，因此不跨量化口径比较 rel_l2；kernel 输出 finite。JIT 编译已完成，不计入计时。

| shape | native | hybrid | FlashInfer | FlashInfer / native |
|---|---:|---:|---:|---:|
| 8192x3072x3072 | 216.5us | 219.6us | 251.5us | 0.861x |
| 4096x6144x3072 | 216.9us | 218.2us | 252.1us | 0.860x |
| 8192x6144x3072 | 401.6us | 430.8us | 442.9us | 0.907x |
| 16384x8192x8192 | 2354.6us | 2578.6us | 2767.6us | 0.851x |

结论：FlashInfer fused CuTeDSL kernel 在 SM120 上不比当前 native/hybrid 快，慢约 9%–15%；其默认 tile/plan 不是这个 shape 的优势配置。暂不把 kernel 复制进 cache-dit，也不以它作为后续优化目标；保留为外部上限参照。JIT 只约 1s，环境本身不是阻塞项。

## 17. SVDQ 性能基线总表（2026-11-05）

这是后续优化的唯一 CUDA Event 基线；§10/§12/§15 的 nsys/早期 Event 数据只作为历史证据，不与本节拼接。

**测量条件**

- GPU4；`torch.backends.cuda.matmul.allow_tf32=False`。
- 同一 `gemm_harness.py` 的 native/cute 路径；CUDA Event 50 次 median，warmup 10。
- 每配置至少 3 轮；rank32 的 `8192x6144`/`16K` 另做正反序各 3 轮，消除预热顺序偏差。
- `cute/native < 1` 表示 cute 更快；`> 1` 表示 cute 更慢。绝对值不要与 nsys 表比较。
- 当前 CuTe：A/SFA G2R + B/SFB/D/L1 TMA + canonical SW64/SW32 LoRA，producer 输出 native pack + BF16 D。

### 17.1 GEMM kernel-only，rank32

| Shape M×N×K | native (us) | cute (us) | cute/native |
|---|---:|---:|---:|
| 4096×3072×3072 | 120.24 | 120.22 | 0.9999 |
| 8192×3072×3072 | 206.80 | 212.18 | 1.0260 |
| 4096×6144×3072 | 206.26 | 210.34 | 1.0198 |
| 8192×6144×3072 | 404.08 | 422.92 | 1.0466 |
| 16384×8192×8192 | 2406.63 | 2560.39 | 1.0639 |

### 17.2 GEMM kernel-only，rank64/128

| Shape | rank | native (us) | cute (us) | cute/native |
|---|---:|---:|---:|---:|
| 4096×3072×3072 | 64 | 124.34 | 124.14 | 0.9984 |
| 4096×3072×3072 | 128 | 132.46 | 130.28 | 0.9835 |
| 8192×3072×3072 | 64 | 214.33 | 215.22 | 1.0042 |
| 8192×3072×3072 | 128 | 229.93 | 226.64 | 0.9857 |
| 4096×6144×3072 | 64 | 214.30 | 214.39 | 1.0004 |
| 4096×6144×3072 | 128 | 228.54 | 226.71 | 0.9920 |
| 8192×6144×3072 | 64 | 421.39 | 433.48 | 1.0287 |
| 8192×6144×3072 | 128 | 456.63 | 455.18 | 0.9968 |
| 16384×8192×8192 | 64 | 2475.42 | 2587.15 | 1.0451 |
| 16384×8192×8192 | 128 | 2674.07 | 2645.22 | 0.9892 |

### 17.3 完整 Linear 层，rank32

| M | native (us) | cute (us) | cute/native | relative L2 |
|---:|---:|---:|---:|---:|
| 1024 | 63.648 | 62.992 | 0.9897 | 0.0 |
| 4096 | 168.336 | 164.624 | 0.9780 | 0.00215 |

rank64/128 的完整层、以及更多 M 的层数据尚未测，不推算、不填 N/A 数字。

### 17.4 当前结论

1. **M4096 所有已测 rank 基本持平或略优 native**；rank128 在 kernel-only 已稳定快约1%–1.6%。
2. **rank32 大 shape 仍明显落后**：`8192x6144` 慢4.66%、`16K` 慢6.39%，是下一步 NCU 的主目标。
3. **rank128 在大 shape 全部略优于 native**（0.3%–1.4%），说明当前 CuTe 结构并非先天不适合大 shape；rank64 在 `8192x6144`/16K 的退化是 rank-specific 问题。
4. 完整层 M1024/M4096 已快1%–2.2%，但 kernel-only 的大 shape 退化会在更大 M 下吃掉层收益。

## 18. 16K NCU 归因与 load-side fence 清理（2026-11-05）

同轮 NCU full：`16384x8192x8192`、rank32、native/cute 各 launch-skip 3。NCU duration 受 replay 影响，不作为 wall-clock；Event 结论见 §17。

| 指标 | native | cute | 读法 |
|---|---:|---:|---|
| waves/SM | 37.24 | 37.24 | 完全一致，尾部/wave 数量不是差异源 |
| L2 hit | 88.07% | 95.32% | cute 更高，L2 复用不是差异源 |
| DRAM throughput | 31.66% | 11.37% | cute 更低 |
| shared inst | 3.146M | 51.51M | cute 16.4× |
| shared load wavefront % peak | 0.37% | 27.79% | cute 的 B/SFB/D/L1 smem→RF 成本 |
| tensor active | 88.10% | 90.42% | cute 不低 |
| sleeping /issue | 0 | 1.559 | cute 独有 TMA/mbarrier 等待 |
| barrier /issue | 0 | 0.155 | cute 独有 |
| long_scoreboard /issue | 0.853 | 3.102 | cute 的 G2R/共享依赖 |
| wait /issue | 4.715 | 4.925 | cute 略高 |
| math_pipe_throttle /issue | 6.872 | 5.359 | cute 更低 |

归因：**排除 wave 尾部和 L2 复用**；剩余差距集中在 shared/B/SFB/D/L1 装载和 TMA/mbarrier 等待。native 靠更高的 DRAM 流量和 local spill 绕开 smem/TMA，但在大 shape 下更快。

本轮单变量优化：删除 3 条 load-side `fence_view_async_shared()`（producer issue_tma、FP4 consumer 每 stage、LoRA consumer 每 stage），保留 output STSM→TMA store 前必需的 fence。TMA transaction barrier wait 已提供 shared 可见性。

同候选 Event50×3：

| shape | native | cute 候选 | 旧 baseline |
|---|---:|---:|---:|
| 4096x3072 | 120.15 | 120.16 | 120.22 |
| 8192x3072 | 206.12 | 210.34 | 212.18 |
| 4096x6144 | 206.14 | 210.75 | 210.34 |
| 8192x6144 | 404.46 | 423.82 | 422.92 |
| 16384x8192x8192 | 2401.25 | 2563.62 | 2560.39 |

变化均在 ±0.2%（8192x3072 的 -0.87% 是旧 baseline 波动），没有实质性能回归；fence 删除保留为正确性等价的简化。验证：94 tests passed、memcheck 0 errors。

剩余 blocker：删除 fence 只减少了同步指令，不改变 shared 装载和 TMA barrier 等待，因此没有缩小与 native 的大 shape 差距。下一步若继续追赶，需要改 epilogue/output store 生命周期或 shared 装载结构，而不是继续删除同步指令。

## 19. 三方案主循环优化负结果（2026-11-06）

统一条件：GPU4，Event 50 次 median，warmup 10，每配置 3 轮；M=2048/4096/8192/16384，N=K=3072，rank=32/64/128。benchmark 已改为通过 `SVDQW4A4Linear.quantize/forward_quant` 准备和计时，避免手工 pack 造成非有限值与跨 case 非法访问。

### 19.1 当前 baseline（cute/native）

| M | rank32 | rank64 | rank128 |
|---:|---:|---:|---:|
| 2048 | 0.9646 | 0.9590 | 0.9512 |
| 4096 | 0.9715 | 0.9707 | 0.9652 |
| 8192 | 1.0064 | 0.9933 | 0.9846 |
| 16384 | 1.0632 | 1.0468 | 1.0148 |

### 19.2 方案1：A/SFA load 前移到 full wait 之前

只改主循环语句顺序，不加寄存器/smem。12 格中 9 格略慢，最大 +0.41%；16K rank32 +0.09%，16K rank64/128 -0.10%/-0.19%。没有达到 +0.5%，已回退，不进入提交。

### 19.3 方案2：A/SFA 双缓冲跨 stage 预取

SASS 闸门通过：BN256 实际最高 R221（上限224）、STACK=0/LOCAL=0。但 12 格全部退化：

- M2048/4096：+2.4%–3.2%
- M8192：+1.9%–2.6%
- M16384：+5.4%–6.1%

ptxas 在当前 224-reg 预算下为双缓冲牺牲了 B/LDSM 与 MMA 的交织调度，隐藏 A 延迟的收益被抵消。已回退。

### 19.4 方案3：BN256/BK64 S5/S6/S7

host 显式路由验证了四个实例，均 168 regs、零 spill，SW32 fallback 数值正确。但：

| 配置 | M2048 rank32 | 16K rank32 |
|---|---:|---:|
| 256x3x128 baseline | 73.57us | 437.88us |
| 256x5x64 | 76.98us | 551.60us |
| 256x6x64 | 76.96us | 553.33us |
| 256x7x64 | 78.82us | 557.87us |

S5 与 S3 in-flight 字节相同、barrier 翻倍，符合预期；S6/S7 增加预取深度也被 barrier/TMA 交互成本抵消。16K 慢约26%–27%，已回退。

### 19.5 结论

三个方案均未达到收益门槛，全部不保留；kernel 保持 `A/SFA G2R + B/SFB/D/L1 TMA + canonical SW64/SW32 + S3/BK128`。后续不要再重复“单缓冲前移、A/SFA 双缓冲、BK64 深 stage”这三条；若继续，需要改 output/store 生命周期或 shared 装载结构。

## 20. 当前 hybrid 的 K-block 交错重测（2026-11-06）

历史 K-block 结论来自旧 hybrid；在 canonical SW64 + fence 清理后的当前 hybrid 上重新测试 `A load -> B/SFB copy -> MMA` 的逐 k-block 交错。数值 94 tests 通过，BN256 仍 168 regs、零 spill，memcheck 无 Invalid 访问。

12 格三轮均值相对 baseline（cute）：

| M | rank32 | rank64 | rank128 |
|---:|---:|---:|---:|
| 2048 | +0.61% | +0.14% | +0.44% |
| 4096 | +0.01% | +0.01% | -0.15% |
| 8192 | -0.33% | +0.08% | -0.01% |
| 16384 | -0.23% | -0.46% | -0.25% |

结论：12 格加权净变化约 0，2K 的 +0.6% 是唯一超过基线噪声的回退，但仍在既定 1% 验收阈值内；8K/16K 稳定小优。按“无实质性能退化且逻辑更直接”保留 K-block 交错。之前 §12 的“平均0.12%、不保留”是旧 hybrid/nsys 口径，不再适用于当前版本。

## 21. 调度结构实验：persistent/light/non-WS/LoRA-first（2026-11-06）

统一条件：GPU4，Event 50 次 median，warmup 10，3 轮；12 格 = M2048/4096/8192/16384 × rank32/64/128，N=K=3072。baseline 为 §20 的 K-block 版本。

### 21.1 persistent CTA（按 ffpa fp4 persist-D 写法）

实现：grid = min(total_tiles, SM count)，grid-stride work loop，全局 seq 驱动 full/empty phase，consumer-only named barrier，`tile_done`（count=256）保证 output store 完成后 producer 才能复用 output alias。

结果：94 tests 通过，168 regs/零 spill；12 格全部退化 **+3.0%–6.7%**，16K rank32 +4.27%。output alias 串行化成本超过 per-work 启动节省，回退。

### 21.2 light producer：288T（256 consumer + 32 producer）

两个变体：

- consumer 调 `warpgroup_reg_alloc<224>`：288T 不是 128 的整数倍，GEMM parity 测试直接死锁；
- 去掉 `setmaxnreg`：能跑，但 BN256 `STACK=264`，12 格慢 **150%–198%**。

结论：288T 与 warpgroup 寄存器协议不兼容；light producer 回退。

### 21.3 Non-WS：256T 全部 MMA，tid0 内联 TMA

实现：去掉 producer warpgroup，empty count=256，tid0 在 ring 中提前 `kStages-1` 发 TMA，所有线程消费后 arrival。

结果：94 tests 通过，224 regs/零 spill；12 格慢 **1.6%–4.3%**，16K 最差 +4.33%。tid0 的 TMA 发射串行成本大于省掉的 producer warp，回退。

### 21.4 LoRA-first + 首个 FP4 K tile 预取（保留）

实现：D/L1 TMA 先发；LoRA MMA 先累加；LoRA 结果先除以 `alpha*wcscales`，主 FP4 累加后再统一乘回并加 bias；首个 K tile 的 A/SFA 在 LoRA 前预取，主循环 kt=0 跳过重复 load。

12 格相对 K-block baseline：

| M | rank32 | rank64 | rank128 |
|---:|---:|---:|---:|
| 2048 | -0.71% | -1.38% | -2.58% |
| 4096 | -0.48% | -1.37% | -1.96% |
| 8192 | +0.39% | -0.15% | -1.28% |
| 16384 | +1.41% | +1.37% | +0.19% |

结论：≤4K 全部受益，8K rank128/64 受益，16K rank32/64 回退约1.4%。图像模型 token 主要集中在 4K/8K，保留 LoRA-first；长序列（>8K）建议走 native。验证：94 tests，168 regs，STACK=8。

### 21.5 两个后续微调未保留

- `full` wait 移入 `k_block==0`：3 轮 2K/4K/8K 稳定 +0.2%–1.0%，16K 仅 -0.1%–-0.3%，无收益，回退。
- BK64/S3：`kSmemBytes` 下界改为 max(stage ring, 64KB output) 后合法，但 BK64 仍 `STACK=8`（spill 来自 LoRA-first 的逆缩放数组，不是 k_block fragment）；2K 76.9µs、16K 555µs，显著慢于 BK128 的 73.5/443µs，回退。

## 22. 16K 长序列 NCU 复核与 main-first 专用分支（2026-11-06）

对象：`16384x8192x8192 rank32`，native/cute 同轮 NCU full，launch-skip 3。NCU duration 受 replay 影响，不用于 wall-clock；Event 单独记录。

| 指标 | native | LoRA-first cute | main-first cute |
|---|---:|---:|---:|
| Event median | 2409.92us | 2580.64us | 2570.46us |
| regs/thread | 252 | 168 | 168 |
| smem/block | 18.4KB | 85.0KB | 85.0KB |
| tensor active | 86.18% | 87.97% | 88.49% |
| L2 hit | 86.71% | 95.05% | 95.18% |
| DRAM throughput | 35.29% | 11.26% | 11.33% |
| long_scoreboard /issue | 0.945 | 2.629 | 3.011 |
| sleeping /issue | 0 | 1.361 | 1.466 |
| barrier /issue | 0 | 0.108 | 0.145 |
| wait /issue | 4.713 | 4.618 | 4.630 |
| local load sectors | 0.262M | 8.389M | 0 |
| local store sectors | 11.010M | 0.197M | 0 |

结论：

- 当前 16K 不是 wave/L2/DRAM 上限；tensor active 已接近 native。
- native 的 2-stage G2R 流水用更多 global 流量换更低 long_scoreboard；cute 的 TMA/shared 路径仍有同步与延迟隐藏缺口。
- LoRA-first 的逆缩放在长序列产生额外 local sectors；main-first 长序列实例消除了这些 local 访问。

实现：Traits 增加 `kLoRAFirst`；producer/consumer 的 main/LoRA 顺序由编译期开关控制。host 在 `M>=16384` 选择 main-first，M<=8192 保持 LoRA-first。

12 格三轮：2/4/8K 全部在 ±0.5% 内；16K rank32/64/128 相对 LoRA-first 分别 **-1.52%/-1.55%/-0.47%**。94 tests 通过。该分支为后续 16K 专用全 G2R/其他结构保留独立入口。

后续零寄存器实验：main-first 16K 分支对下一 K tile 的 A/SFA 发 `prefetch.global.L1`。三轮结果 rank32/64/128 分别 **+16.18%/+15.98%/+14.99%**，地址计算与预取指令开销远大于收益，已回退；不要重复。

## 23. Helper 微优化逐项实验（2026-11-06）

class 重构后按“scale 融合 -> 地址复用”顺序验证，均保持 94 tests：

1. `apply_inverse_scale` 列主序融合：删除 `inv_scale[]`，逐列计算倒数并立即应用。STACK 仍为 8，说明该数组不是 stack 根因；12 格最大 +0.26%，无收益，回退。
2. `apply_residual_scale` 列主序融合：删除 `col_scale[]/col_bias[]`，保持 alpha 后 fmaf 的舍入顺序。2/4/8K 明显退化，最大 +1.50%（4K rank64）；回退。
3. `load_act_A_g2r` 地址上下文：16K main-first SASS 已用 uniform-register 基址与立即数偏移表达 A/SFA LDG；线程不变量已被 ptxas 复用，没有证据支持手工地址上下文，因此不引入新结构。
4. smem FP32 accumulator：BM128×BN256 的 C 为 32768 floats=128KB，超过 SM120 约99KB可用 smem；且每个 K stage 读写 C 会放大 shared 压力，数学与容量上均不可行。

结论：TMA/LDSM/STSM 已提供硬件向量化；helper 级数组融合和地址复用没有收益。保留 76d9071 的 class 重构与 5040fe7 的长序列 main-first 分支，后续优化应回到流水结构而非这些微改写。

## 24. FLUX e2e：native/cute 出图与计时（2026-11-06）

口径：GPU4、`torch 2.11.0+cu130`、`FLUX.1-dev`、1024x1024、28 steps、seed 0、`--compile`，`python -m cache_dit.generate flux`，每个配置单次运行。命令：

```bash
export FLUX_DIR="<flux-dir>"
python -m cache_dit.generate flux --compile --svdq-nvfp4-r32-dq --save-path native.png
python -m cache_dit.generate flux --compile --svdq-nvfp4-r32-dq --svdq-runtime cute --save-path cute.png
python -m cache_dit.generate flux --save-path base.png
```

计时（同日两次独立跑，取第二次，rebase 到 origin/main 之后）：

| 配置 | Warmup | Inference | 相对 base |
| --- | --- | --- | --- |
| base（BF16, C0） | 17.5s | 16.99s | 1.00x |
| SVDQ native（r32-dq） | 26.0s | 6.06s | **2.80x** |
| SVDQ cute（r32-dq） | 59.2s | 6.19s | **2.74x** |

first-run 对照（rebase 前，同一份 SVDQ 代码）：native 6.05s、cute 6.20s、base 16.96s。cute/native = 1.021，e2e 层面的差距与 kernel-only 的 2/4K 结果一致。

图像（`cache-dit-metrics psnr ssim`）：

| 图像对 | PSNR | SSIM |
| --- | --- | --- |
| native vs cute | 33.88 | 0.9303 |
| native vs base | 28.09 | 0.8081 |
| cute vs base | 29.07 | 0.8144 |

**同配置 run-to-run 抖动**（两个独立进程，相同参数、相同 seed）：

| 图像对 | PSNR | SSIM |
| --- | --- | --- |
| native vs native | 34.08 | 0.9191 |
| cute vs cute | 36.26 | 0.9687 |

结论：

- SVDQ 路径（native 与 cute 都是）**出图不可复现**：同配置两次运行的 PSNR 已在 34–36 dB / SSIM 0.92–0.97，与 native↔cute 的 33.9 dB 同量级。因此**单次 e2e 的 PSNR 不能用来判别 native 与 cute 的数值差异**，只能判别“量化 vs 不量化”。BF16 base 本身可复现（两次 md5 相同），抖动只出现在 SVDQ 路径。
- 抖动来源未定位。已排除 `svd_lowrank` 的随机初始化（`lowrank.py` 用 `fork_rng` + 固定 `_SVD_LOWRANK_SEED`）。剩余候选：量化阶段其他随机源、以及 GEMM 归约顺序。要复现须先用固定 latent/固定权重做 bit-exact A/B。
- kernel 级数值等价性仍由 `tests/kernels/test_svdquant_cute.py` 的 parity 用例保证（rank32/64/128，完整层 relative L2 ~2e-3），不依赖图像 PSNR。
- 视觉上三图同构图、`Hello world` 文案完整可读、无结构伪影；量化版与 BF16 的差异集中在毛色饱和度与条纹对比度（mean |diff| ≈ 18.3/255，SVDQ 版平均亮度高约 8），属 NVFP4 量化的色调偏差，不是 kernel 缺陷。

## 25. PRO 6000 四方对比与 `lora_layout` dispatch（2026-11-07）

口径：RTX PRO 6000 Blackwell（SM120）、`CUDA 13.0.88`、`torch 2.10.0+cu130`、`bench/kernels/bench_svdquant.py`、CUDA Event median、warmup 10 / iters 50、单轮内比较。第四节为 FlashInfer `0.7.3.dev20261106` 的 `cute-dsl` kernel（用它自己的 128x4 pack，只报 `finite`）。kernel-only 与 producer/e2e 来自同一台机器的两轮运行，跨轮单格漂移可达 30%，不要跨轮拼接。

### 25.1 接口现状（PR #1259 之后）

`bool f32d` 已换成 `enum class LoraLayout { kNative, kRowMajor }`。API 统一后，namespace 内叶子名字均为 `gemm_w4a4` / `quantize_w4a4`：`C++` 由 `RuntimeParams` + `LoraLayout` 分派到 `svdq::native_kernels` 或 `svdq::cute_kernels`。Python 只有 `svdq_gemm_w4a4` 与 `svdq_quantize_w4a4` 两个公共入口，用两个正交维度选择：

| runtime | lora_layout | LoRA-down 项 D |
|---|---|---|
| `native` | `native` | packed FP32 fragment（native producer） |
| `cute` | `native` | packed FP32 fragment（native producer） |
| `cute` | `row_major` | 行主序 BF16，rank scale 已折叠（CuTe producer） |

`runtime='native'` 不接受 `row_major`（wrapper + op impl 双层校验）。

**producer 必须与 layout 配对**（混用是静默数值损坏，不报错、结果错 30–54%，曾把 bench 结果误判成 kernel bug）：

- `row_major` → `svdq::cute_kernels::quantize_w4a4`，输入必须 `convert_svdq_nvfp4_pack_to_cute()` 产出的 `_cute_proj_down`（行主序）与 `_cute_smooth`（普通向量）。
- `native` → `svdq::native_kernels::quantize_w4a4`，输入是 checkpoint 里 fragment-packed 的 `proj_down` / `smooth_factor`。

单测因此必须覆盖「producer + GEMM 对 native 的端到端 parity」：只测 GEMM（用解码出来的 native fragment D）或只测 producer（对 `x @ W_down`）都守不住这个洞，`test_row_major_pipeline_matches_native_runtime`（rank 32/64/128）就是补这一层。

### 25.2 GEMM kernel-only（us）

| Shape M×N×K | rank | native | cute (native D) | cute (row_major D) | FlashInfer cute-dsl |
|---|---:|---:|---:|---:|---:|
| 4096×3072×3072 | 32 | 117.81 | 115.10 | **112.19** | 148.78 |
| 4096×3072×3072 | 64 | 121.50 | 118.19 | **114.00** | 149.58 |
| 4096×3072×3072 | 128 | 128.40 | 127.90 | **119.12** | 157.49 |
| 8192×3072×3072 | 32 | 183.38 | 176.11 | **174.62** | 205.39 |
| 8192×3072×3072 | 64 | 188.64 | 183.38 | **178.10** | 207.14 |
| 8192×3072×3072 | 128 | 200.03 | 198.72 | **185.57** | 215.65 |
| 4096×6144×3072 | 32 | 182.75 | 175.98 | **173.49** | 201.49 |
| 4096×6144×3072 | 64 | 189.30 | 184.26 | **177.23** | 207.90 |
| 4096×6144×3072 | 128 | 200.50 | 197.26 | **185.62** | 218.14 |
| 8192×6144×3072 | 32 | 343.41 | 326.91 | **310.40** | 361.18 |
| 8192×6144×3072 | 64 | 337.57 | 321.63 | **305.90** | 327.41 |
| 8192×6144×3072 | 128 | 364.38 | 346.16 | **328.29** | 345.47 |
| 16384×8192×8192 | 32 | 1590.74 | 1634.34 | 1600.50 | **1571.23** |
| 16384×8192×8192 | 64 | 1621.04 | 1654.27 | **1620.72** | 1658.22 |
| 16384×8192×8192 | 128 | 1682.77 | 1715.42 | **1660.46** | 1690.32 |

`row_major` D 在 15 格里 14 格最快（相对 native −0.5%~−10.3%），原因是 D 走 TMA 而不需要 gather FP32 fragment；唯一例外是 16K rank32，FlashInfer 快 1.9%（在噪声内，且其无 e2e 数据）。

### 25.3 producer 与 e2e（us）

| Shape M×N×K | rank | quantize (native D) | quantize-row-major | e2e native | e2e cute | e2e cute+row_major |
|---|---:|---:|---:|---:|---:|---:|
| 4096×3072×3072 | 32 | 41.79 | **36.96** | 143.46 | 140.37 | **129.17** |
| 4096×3072×3072 | 64 | 46.62 | 46.82 | 150.82 | 149.58 | **141.18** |
| 4096×3072×3072 | 128 | **56.51** | 68.80 | **167.25** | 167.54 | 168.62 |
| 8192×3072×3072 | 32 | 58.21 | **53.44** | 229.20 | 219.70 | **208.93** |
| 8192×6144×3072 | 32 | 58.06 | **54.11** | 400.64 | 367.82 | **350.27** |
| 8192×6144×3072 | 64 | **66.62** | 76.59 | 418.24 | 392.32 | **375.44** |
| 8192×6144×3072 | 128 | **82.83** | 119.82 | 459.30 | **437.76** | 450.94 |
| 16384×8192×8192 | 32 | 297.09 | **269.12** | 1895.12 | 1906.40 | **1849.30** |
| 16384×8192×8192 | 64 | 374.62 | **295.87** | 1991.04 | 1993.25 | **1917.74** |
| 16384×8192×8192 | 128 | **411.66** | 535.68 | **2102.94** | 2107.34 | 2196.14 |

规律：CuTe producer 在 rank32 快 7%–12%、rank64 打平到略慢、rank128 明显退化（+22%~+45%）；quantize 占 e2e 的 14%–31%，所以 kernel-only 会高估收益；实测 e2e 略小于 `quantize + gemm` 之和（packs 在 GEMM 期间存活，分配器复用同一批 block）。

### 25.4 dispatch 结论（仅 PRO 6000 数据）

- `rank <= 64` → `cute` + `row_major`：e2e 8/8 格最优（−3.7%~−12.6% vs native），kernel 10 格里 9 格第一。
- `rank == 128` → **不要** row_major：改用 `cute` + native D（8K，437.76us）或 native（16K，2102.94us）。
- **没有做成默认**：PRO 5000 上的实测标定与 PRO 6000 有差异，crossover 不能写成全局常量；`SVDQW4A4Linear` 仍全程 `lora_layout='native'`。接线需要单独 PR + 每 SKU（PRO 5000 / PRO 6000 / H800）复测，并加一条钉住 producer/layout 配对的回归测试。

### 25.5 FlashInfer 在 SM120 上跑通的前提

1. nightly whl 需**两个**包：`flashinfer-python==0.7.3.dev20261106` + 通用 `flashinfer-jit-cache`（提供 `__version__`）+ `flashinfer-jit-cache-sm120f`（SM120f cubins）。0.6.x 的 SVDQuant 只支持 SM100，`cutlass`/`trtllm` backend 对 capability 120 直接拒绝。
2. 需要 `cutlass.cute.FastDivmodDivisorV2` 别名（已发布的 `nvidia-cutlass-dsl` 都没有该符号），只在 bench harness 里打，不进库。
3. `nvidia-cutlass-dsl` 必须**带 deps** 升级：`--no-deps` 只抬 python 包，`nvidia-cutlass-dsl-libs-base` 停在 4.4.0.dev1 时所有 CuTe-DSL kernel 都会 `Loader for '.o' files is not registered`；python 包与 `libs-*` 全部对齐 4.8.1 才正常。

FlashInfer cute-dsl 在 ≤8K 全面落后 cache-dit（+5%~+33%），只在 16K rank32 略胜 1.9%，与本文件 §15 的旧结论（CuTeDSL 慢 9%–15%）方向一致。

## 26. row_major producer rank128 专项优化（2026-11-09，RTX PRO 5000）

### 26.1 范围与代码状态

只修改 `csrc/kernels/svdq/cute/fp4/quantize_w4a4.cu`，不改变 native qact/ascales、row-major BF16 D、公开接口或 GEMM。全部 producer 类型外的 device 辅助函数、量化、主循环、D 归约和 launcher 已收进统一 `W4A4QuantizeKernel`；主 kernel 入口为 `W4A4QuantizeKernel::quantize_w4a4_fused_lora`，外部只保留薄的 `__global__` 启动壳。launcher 使用 `QuantizeLaunchArgs` 参数包，分派处只写 rank/warp/buffer/split 配置。

### 26.2 当前完整矩阵（GPU4，Event median，warmup10/iters50，1 轮复测）

命令口径：`bench/kernels/bench_svdquant.py --rank 32/64/128 --impl quantize,quantize-row-major`。下表左为 native µs，右为 row_major µs，括号为 row/native；<1 表示 CuTe 更快。

| Shape M×N×K | rank32 | rank64 | rank128 |
|---|---:|---:|---:|
| 2048×3072×3072 | 34.38 / 32.43 (0.943) | 38.46 / 40.62 (1.056) | 47.15 / 54.00 (1.145) |
| 4096×3072×3072 | 48.66 / 46.88 (0.964) | 56.62 / 59.82 (1.057) | 73.15 / 80.59 (1.102) |
| 4096×6144×3072 | 48.70 / 47.57 (0.977) | 56.42 / 59.07 (1.047) | 73.28 / 80.86 (1.104) |
| 8192×3072×3072 | 76.45 / 64.11 (0.839) | 91.74 / 92.98 (1.013) | 123.63 / 135.84 (1.099) |
| 8192×6144×3072 | 76.42 / 65.38 (0.856) | 93.10 / 92.96 (0.999) | 123.89 / 137.23 (1.108) |
| 16384×8192×8192 | 380.54 / 347.57 (0.913) | 426.45 / 365.70 (0.858) | 565.94 / 564.64 (0.998) |
| 4608×3072×15360 | 221.70 / 222.75 (1.005) | 252.77 / 258.40 (1.022) | 316.66 / 366.93 (1.159) |

结论：rank32 除长 K 持平外全部更快；rank64 16K 快14%、8K 持平、2K/4K 慢5%–6%；rank128 16K 已持平，2K/4K/8K 仍慢10%–14.5%，长 K 慢约16%。rank128 相比优化前的 1.5×–2.13× 已明显收敛，但“全格≤1.00”目标尚未完成。

### 26.3 保留的优化

1. **D partial shared 归约缓冲可配**：rank128 用 2 个 buffer 分阶段确定性累加，替换每 warp 一份完整 32×R partial，shared 从约67KB降到约34KB；A staging 与 D buffer 显式 phase barrier overlay，所有 warp（包括无 K slice）走同一 barrier。D 仍 FP32 归约、一次 BF16 rounding。
2. **小 M rank split**：`blockIdx.y` 把 rank 分给两个 CTA，grid 翻倍、局部累加器减半；qact/ascales 只由 `blockIdx.y==0` 写，D 各写不重叠列，不引入 scratch/finalize。rank128 在 M≤8192、rank64 在 M≤4096 启用。
3. **rank128 B 装载 `movmatrix.trans`**：四个对齐32-bit load + 寄存器转置，替代八个标量 BF16 fragment load；global load request 约减半，4K/长K/16K有实测收益，qact/ascales/D 保持位级一致。
4. **按 M 的 warp/buffer 分派**：rank128 2K用 `8w/2buf/split2`、4K用 `4w/2buf/split2`、8K用 `8w/2buf/split2`、16K用 `8w/2buf/split1`；rank64 2K用 `8w/8buf/split2`、4K用 `4w/4buf/split2`、8K+用 `8w`。这些阈值是在 RTX PRO 5000 上逐格实测的 crossover，不能直接外推到 PRO 6000/H800。

### 26.4 已证伪并移除的尝试

- **16-row CTA**：累加器减半但两个 M16 half 的 B 复用消失，global load requests/sectors 约翻倍；4K NCU duration 约149µs，明显差于32-row基线，移除。
- **4K×4 rank-group 的16-warp CTA**：512线程触发128-reg上限，local spill约1.2M/2.3M sectors，命名 barrier stall 13.2，4K约138µs，移除。
- **rank64 强制2 CTA/SM + 2归约缓冲**：仍受 shared/寄存器组合限制，引入spill，2K/4K/长K明显回退，移除。
- **B shared staging + XOR swizzle**：请求数下降但 bank conflict不降，延迟无净收益；movmatrix路径更简单，移除staging/swizzle。
- **A staging 128B XOR swizzle**：bank conflict从约0.69M升到1.38M，三档rank均回退，移除。
- **B vector-load + shuffle 伪路径**：早期版本按目标 lane 预选 word，但 `__shfl` 取的是源 lane 的寄存器，映射错误，golden 失败；未保留。
- **rank128 split4**：2K与split2几乎同速；rank64 split4出现全列数值错误且无收益，模板支持已删除。

### 26.5 验证状态

- 修改前后小尺寸 golden：rank0/32/64/128 的 qact/ascales 与 D 逐字节一致（`qact_exact`, `ascales_exact`, `D_exact`）。
- `tests/kernels/test_svdquant_cute.py -k row_major` 当前13 passed（统一类重构前）；统一类重构后重新编译并跑 golden，全部 PASS。提交前仍需跑完整 producer+GEMM parity、pre-commit 和 diff 检查。
- 未完成项：rank128 2K/4K/8K/长K与 rank64 2K/4K仍高于native；下一步应优先解释小M的输入重复读取与D归约开销，而不是继续扫 warp 数或重复已证伪的16-row/K-split-scratch路线。

## 27. split-K rank chunk=1 调参与适用区间（2026-11-09，RTX PRO 5000）

### 27.1 改动

split-K producer 的 rank 流式累加从 `kChunkTiles=2` 改为 `1`：每线程只保留一个 R16 tile 的 FP32 D，降低寄存器压力；K128 CTA 仍覆盖两个 K64 slice。D scratch 使用 fragment-major 连续 lane 布局，`red.add` 按 sector 合并；finalize 合并相邻 j atom，一次 `uint4` 写 8 个 BF16。

### 27.2 chunk=1 vs chunk=2（split-K，µs）

同一口径 GPU4、warmup10/iters50、Event median；比值越小越好。

| Shape | rank64 chunk2 | rank64 chunk1 | rank64 c1/c2 | rank128 chunk2 | rank128 chunk1 | rank128 c1/c2 |
|---|---:|---:|---:|---:|---:|---:|
| 2048×3072×3072 | 41.98 | 41.23 | 0.982 | 52.32 | 50.82 | 0.971 |
| 4096×3072×3072 | 62.54 | 60.80 | 0.972 | 82.69 | 79.42 | 0.960 |
| 8192×3072×3072 | 99.31 | 95.76 | 0.964 | 131.84 | 127.34 | 0.966 |
| 16384×8192×8192 | 578.83 | 577.62 | 0.998 | 591.25 | 617.62 | 1.045 |
| 4608×3072×15360 | 329.92 | 322.48 | 0.977 | 343.28 | 361.89 | 1.054 |

结论：

- rank64：chunk=1 在全部测试格都略优，但 2K/4K/8K 相对 native 仍为 1.06/1.07/1.04 左右，尚未达标。
- rank128：chunk=1 对 2K/4K/8K 有 3%–4% 收益；长K/16K 应保留 chunk=2。
- chunk=1 的适用区间暂定为 **rank64/rank128 的序列长度 ≤8K**；长K/16K 不切换。
- 最终 dispatch 暂不落地，等 fused/split-K 的 cp.async/GEMM 优化收敛后统一按 `rank + M + K` 决定。

### 27.3 状态

当前 split-K 仍由 `CACHE_DIT_SVDQ_PRODUCER_SPLIT_K` 控制，默认生产路径仍是 fused；本节只记录调参结论，不代表已改默认分派。固定 dispatch 已于 §28 落地，取代本节 env 状态。

## 28. 固定 rank/M dispatch 与 rank64 现状（2026-11-09，RTX PRO 5000）

### 28.1 已落地 dispatch

不再依赖环境变量；`svdq.cpp`（原 `ops.h`）在 row_major 路径按 rank 和 M 决定是否分配 FP32 scratch：

| rank | split-K 区间 | fused 区间 |
|---|---|---|
| 32 | M ≤ 8192 | M > 8192（16K） |
| 64 | M ≤ 2048 | M > 2048（4K/8K/16K/长K） |
| 128 | M ≤ 8192 | M > 8192（16K） |

split-K producer：256 行 CTA、8 warps、K128 切片、rank chunk=1；D 写 fragment-major FP32 scratch，`red.add.v2.f32` 相邻两列；finalize 每个线程读 8 个 FP32，一次 `uint4` 写 8 个 BF16。

### 28.2 当前矩阵（GPU4，warmup10/iters50，Event median，µs）

| Shape | rank32 native / row (ratio) | rank64 native / row (ratio) | rank128 native / row (ratio) |
|---|---:|---:|---:|
| 2048×3072×3072 | 35.09 / 33.23 (0.947) | 38.48 / 40.67 (1.057) | 47.44 / 51.55 (1.087) |
| 4096×3072×3072 | 48.64 / 44.70 (0.919) | 57.68 / 60.10 (1.042) | 73.76 / 81.68 (1.107) |
| 4096×6144×3072 | 49.54 / 45.31 (0.915) | 56.94 / 59.74 (1.049) | 73.68 / 81.62 (1.108) |
| 8192×3072×3072 | 77.30 / 63.95 (0.827) | 92.50 / 93.73 (1.013) | 124.46 / 131.39 (1.056) |
| 8192×6144×3072 | 76.53 / 64.26 (0.840) | 92.56 / 93.92 (1.015) | 124.72 / 130.48 (1.046) |
| 16384×8192×8192 | 382.19 / 348.02 (0.911) | 426.66 / 365.92 (0.858) | 572.43 / 569.41 (0.995) |
| 4608×3072×15360 | 223.50 / 197.52 (0.884) | 254.62 / 259.49 (1.019) | 319.81 / 368.82 (1.153) |

### 28.3 rank64 重点结论

- rank64 只有 2K 走 split-K（1.057），4K 保持 fused（1.042/1.049），8K 1.013–1.015，长K 1.019，16K 0.858。
- 4K 是 rank64 的主要差距；NCU：CuTe 4w/4buf/split2 为 52.03µs、182 regs、grid256、waves1.16；native 48.67µs、164 regs、grid384、waves3.49。两者 occupancy 都是 16.7%，差异主要是 block 粒度/尾波和输入重复读取。
- 4K split-K producer 58.18µs，比 fused 52.03 更慢；缓存 scratch 没有收益。
- 已证伪的 4K 配置：8w/4buf/split2 63.3/64.2；8w/8buf/split2 63.4/62.4；8w/split1 65.8；split4 与 K64 split-K 均明显回退。
- qact 按 K64 slice 在 split2 两个 CTA 间分摊：rank128 约快 1%，但 rank64 4K 回退到 77.5µs，并破坏 fused D 位级一致，已撤销。
- 当前最优仍是 4w/4buf/split2；要再进一步需要减少 split2 的输入重复读取（cluster/DSMEM 或共享 A）或匹配 native 的 256-row + K 维 grid/epilogue 结构。

## 29. Rank-specialized CuTe producer 与 GEMM 主瓶颈（2026-11-10，RTX PRO 5000，PR #1271）

### 29.1 实现与 dispatch 状态

- CuTe producer 已拆为 `quantize_impl.cuh` + `quantize_r32/64/128.cu`，每个 TU 只实例化一个 rank；通用路径保留在 `quantize_uni.cu`。GEMM 在 `gemm/`，producer 在 `quantize/`。
- fused：CTA 内 FP32 D 归约；split-K：rank streaming、fragment-major FP32 scratch、`red.relaxed.gpu.add.v2.f32`、后置向量化 BF16 finalize。
- `svdq.cpp` 当前选择：
  - rank32 且 K≤3072：全部走 universal（K3072 的 2K/4K/8K 实测 universal 更快）。
  - rank32 且 K>3072：M≤8192 走 split-K，M>8192 走 fused。
  - rank64/rank128：当前生产矩阵全部走 split-K。
- 特化 split-K：rank32/K>3072 用 K128 double-A async；rank64/K≤3072 用 K256，rank64/K>3072 用 K128 async；rank128 全 K256。
- rank64/128 fused 实例仍编译并做过独立数值验收，但当前 `svdq.cpp` 不会选中；不要根据旧 §26/§28 的 fused 阈值重开，除非给出新的逐格 Event 证据。
- Python 普通 CuTe 路径默认 `lora_layout="row_major"`，显式 native 与 `fused_mlp` 保持 FP32 fragment D。

### 29.2 测量口径

官方 `bench/kernels/bench_svdquant.py`，5 个默认 `MxNxK`，显式扩展 rank32/64/128；warmup=10、iterations=50、median；同时跑 `quantize`、`quantize-row-major`、native/cute GEMM、三组 e2e 和 FlashInfer。共 150 行，cutlass-sm120 不支持 rank128 产生 5 个 skipped。GPU4：NVIDIA RTX PRO 5000 72GB。

### 29.3 Producer 与 e2e 全矩阵

µs 为 CUDA Event median；producer speedup = native quantize / CuTe row-major quantize，e2e speedup = native+e2e / CuTe-row-major+e2e。

| M×N×K | R | native q µs | CuTe q µs | q speedup | native e2e µs | CuTe e2e µs | e2e speedup | e2e rel-L2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096×3072×3072 | 32 | 48.11 | 43.86 | 1.097× | 170.00 | 160.50 | 1.059× | 0.003263 |
| 4096×3072×3072 | 64 | 56.21 | 52.02 | 1.081× | 181.12 | 169.82 | 1.067× | 0.003209 |
| 4096×3072×3072 | 128 | 72.64 | 63.36 | 1.146× | 204.37 | 189.20 | 1.080× | 0.003117 |
| 4096×6144×3072 | 32 | 48.11 | 43.97 | 1.094× | 270.46 | 252.18 | 1.073× | 0.003279 |
| 4096×6144×3072 | 64 | 56.22 | 51.30 | 1.096× | 283.90 | 267.38 | 1.062× | 0.003241 |
| 4096×6144×3072 | 128 | 72.56 | 63.36 | 1.145× | 318.56 | 293.14 | 1.087× | 0.003167 |
| 8192×3072×3072 | 32 | 76.64 | 64.26 | 1.193× | 306.50 | 301.47 | 1.017× | 0.003263 |
| 8192×3072×3072 | 64 | 90.98 | 85.82 | 1.060× | 324.96 | 333.10 | **0.976×** | 0.003209 |
| 8192×3072×3072 | 128 | 123.81 | 106.46 | 1.163× | 368.77 | 366.74 | 1.006× | 0.003117 |
| 8192×6144×3072 | 32 | 76.70 | 63.26 | 1.212× | 500.77 | 494.91 | 1.012× | 0.003279 |
| 8192×6144×3072 | 64 | 91.07 | 84.99 | 1.072× | 528.42 | 534.80 | **0.988×** | 0.003240 |
| 8192×6144×3072 | 128 | 123.90 | 106.56 | 1.163× | 591.44 | 575.57 | 1.028× | 0.003167 |
| 16384×8192×8192 | 32 | 379.95 | 349.49 | 1.087× | 2953.52 | 2939.39 | 1.005× | 0.003297 |
| 16384×8192×8192 | 64 | 428.61 | 358.99 | 1.194× | 3107.42 | 2978.32 | 1.043× | 0.003282 |
| 16384×8192×8192 | 128 | 566.42 | 500.11 | 1.133× | 3438.46 | 3169.07 | 1.085× | 0.003247 |

汇总：

- producer geomean **1.1281×**，15/15 快，范围 **1.060×–1.212×**。
- kernel-only CuTe row-major GEMM geomean **1.0125×**；仅比 native 快 1.25%，是当前主瓶颈。
- e2e geomean **1.0384×**，13/15 快。
- e2e 回退点均为 rank64/M8192：N3072 **0.976×**、N6144 **0.988×**；producer 本身仍为 1.060×/1.072×，说明回退来自 GEMM schedule，而不是 quantize。
- producer 收益大于 e2e 收益，后续优化重点明确转为 **W4A4 GEMM**。

### 29.4 FlashInfer 对照

kernel-only GEMM，FlashInfer 使用独立 activation pack，因此 rel-L2 不可比：

- vs `flashinfer+cute-dsl`：geomean **1.0951×**，15/15 CuTe 胜；单点 +0.36%–+17.84%。
- vs `flashinfer+cutlass-sm120`：geomean **1.1438×**，9/10 CuTe 胜；唯一落后是 `8192×6144×3072, rank32`，CuTe 慢约 0.61%。cutlass-sm120 不支持 rank128。

### 29.5 正确性与构建

- 121 项 SVDQ 回归（dispatch/CuTe/runtime/source manifest/config）通过。
- 12 组 v1 golden：qact、ascales bitwise 一致；D 用误差门槛，因为 split-K 的 relaxed FP32 atomics 不保证逐位顺序。
- 独立 FP32 D 覆盖 K64/K192/K3136 tail、非单位 rank scale、M2304/M8448、rank32/64/128。
- fused 与 split-K 均实际执行并做 golden/独立参考验证；最终恢复生产 dispatch。
- FLUX eager + compile 八组生成成功，e2e rel-L2 ≤0.0033；compile 模式 row-major 为 6.04s，native 6.06s，BF16 base 14.85s。
- `setup.py` 已给 SVDQ extension 加 header `depends`；pre-commit/clang-format/yapf 通过。

### 29.6 后续优先级

> 已被 §30 更新：GEMM 已完成 mainloop 索引与 dispatch 优化并确认进入平台；当前优先 CuTe split-K（`quantize_impl.cuh`）bank conflict 与 producer e2e 上下文，uni 路径保持现状。

1. **W4A4 GEMM shared/TMA/同步与 epilogue**：先按 `svdq-nvfp4-cute-tma-sync-analysis.md` 取 NCU，再定位 rank64/M8192 两个 e2e 回退。
2. 不再重复 producer 的 warp 数、M 分桶、K split chunk 扫描；本轮已把 15/15 producer 做到 native 以下，重复扫参收益不足。
3. 若继续 producer，只接受能解释 NCU 主指标的新假设；同时保持 qact/ascales bitwise golden 与 D rel-L2 验收。
4. GEMM 达到相对 native 至少 5% 后，再讨论 NVFP4 runtime 默认从 native 切到 cute。

## 30. 当前 GEMM 平台、P0/P1 负结果与 producer 优先方向（2026-11-11，RTX PRO 5000）

### 30.1 落地状态

- PR #1284 已 squash 合入 main（`3439fb11`）：mainloop 改为 kt 归纳变量，移除 LoRA-first 循环元数据 spill；固定输入下 15/15 GEMM bitwise。
- dev 上 `3ad684c6` 调整 row-major 调度：`M>=16384 || (M>=8192 && N>=6144)` 用 main-first，其余 LoRA-first。受控同进程 7 轮 × 50 次交错 A/B 中 main-first 只在 6/15 格胜出，不能全局替换。

### 30.2 当前官方 Event 矩阵

同机三轮 median-of-3，5 shape × rank32/64/128：

- producer：**1.1088×**，15/15 快，范围 1.037–1.213。
- row-major kernel-only GEMM：**1.0248×**，12/15 快；三个回退为 8192×6144/r32 0.971、r64 0.985，16K/r32 0.998。
- row-major e2e：**1.0553×**，14/15 快；唯一回退 8192×6144/r64 0.999。
- cute+native-D e2e：**1.0462×**，14/15 快；唯一回退 8192×3072/r64 0.990。

这是单会话结果，不能替代跨会话稳定验收。P2 自身只在 6 个 main-first 格提供约 0.7%–1.4% 纯 GEMM 收益，不能把整矩阵 geomean 全部归给 P2。

### 30.3 当前 GEMM NCU（15 格 cute）

| 指标 | 中位数 | 读法 |
|---|---:|---|
| SM / Tensor / Mem | 80.2% / **84.2%** / 57.8% | Tensor 已接近持续发射峰值 |
| DRAM / L2 hit / L1 hit | 12.9% / 99.9% / 54.7% | 不缺带宽；A G2R 有 L1/L2 延迟 |
| occupancy / issue / eligible | 17.75% / 15.4% / 0.224 | 单 CTA、可发射 warp 少 |
| regs / smem / local ld+st | 168 / 85 KiB / 0 | 无 spill |
| math_pipe_throttle + wait | **36.2% + 30.8%** | MMA 发射与依赖链 |
| long_scoreboard | 18.4% | A/SFA G2R latency |
| sleeping / barrier | 7.5% / 1.0% | TMA 不是主瓶颈 |

即使消掉全部非张量 stall，按当前 tensor active 计算的粗略上限也只有约 `1/0.842=1.19×`，实际远小于此。当前是 G2R latency、MMA dependency 与单 CTA occupancy 的联合局部最优。

### 30.4 已证伪的 GEMM 方向

- **A/SFA 全 TMA+SMEM（最初实现）**：同 shape nsys 约 137.4us，hybrid G2R 117.2us，全 TMA 慢约 14.7%；shared 指令 2.654M vs 2.064M，bank conflict 1.036M vs 0.738M。long_scoreboard 从 2.923 降到 0.666，但被 smem/LDSM/barrier 税抵消。**不要把 A/SFA 重新做回 TMA。**
- **8M×1N**：A global sectors 12.39M→7.08M，但 B/SFB fragment 翻倍并新增 local ld/st 8.63M/6.93M；纯 GEMM geomean **0.794×**，15/15 慢。SFB N=1 映射也未通过 bitwise。
- **main-first S4 + 独立 LoRA ring**：15/15 bitwise，但 16K NSYS 0.990/0.990/0.991；long_scoreboard 降到 14%，serial LoRA 使 sleeping 升到 8.4%。让 L1 复用 B 槽并与尾段重叠后仍为 0.992/0.991/0.997。patch 保存在 `.tmp/.../p1_rejected.patch`。
- **stage 增量回绕 / ring 对齐 unroll3**：分别增加指令 0.3%–1.8% 和 6.1M–12.7M，均回退。
- **简单 A/SFA 前移、双缓冲、K-block prologue、persistent、light producer、Non-WS、BN128 小 tile**：均已在 §8/§9/§19/§21 记录为负结果，无新 NCU 证据不要重复。

### 30.5 当前 producer NCU

| 实现/类型 | CuTe 关键指标 | 结论 |
|---|---|---|
| `quantize_uni.cu` split-K（rank32/K≤3072，4 格） | dur 47.6us，SM 48%，LSU 55%，occ 29.7%，bank conflicts **42.9K**（native 1.5K），MIO/LG 14.1/15.1% | 手搓 MMA/ldmatrix，手写 swizzle 改造成本高，**近期保持现状** |
| `quantize_impl.cuh` fused（rank32/16K，1 格） | dur 326us，DRAM/Mem **90.2%**，long_scoreboard 47.2%，occ 16.4% | 已到 DRAM 带宽墙，空间小 |
| `quantize_impl.cuh` split-K（rank64/128，10 格） | dur 83.7us，SM 31%，Mem 48%，DRAM 0.4%，L2 100%，occ 16.6%，regs 198，smem 91 KiB，bank conflicts **380.9K**（native 1.9K） | CuTe 路径，L2-resident，latency/冲突主导；**后续唯一 producer 重点** |

quantize kernel 相对 native：universal 1.17×、fused 1.30×、split-K 1.08×。split-K finalize 中位约 2us，不是主成本。

但 split-K 的 e2e producer 总时间中位为 native 94.4us / CuTe 101.0us，而 kernel 本体 CuTe 多数更快。下一步必须先拆清 e2e 的 allocator/cache/finalize/上下文，不能只根据 isolated kernel 判断 producer 落后。

### 30.6 后续优先级

1. **CuTe split-K producer**（`quantize_impl.cuh`，rank64/128 十格）：先对 SFA/SFB shared copy 做 SASS/PC 定位，确认 38 万 conflict 的具体指令；只改该拷贝路径，保持 qact/ascales bitwise 与 D 误差门槛。目标先看 producer kernel 3%–8%，不要承诺 e2e 同幅。
2. **uni 路径保持现状**（`quantize_uni.cu`，rank32/K≤3072 四格）：该路径是手搓 MMA/ldmatrix + 手写 swizzle，bank conflict 虽为 4.3 万，但改造成本和回归风险高，不作为近期优化项。
3. **producer e2e 上下文**：对比 isolated producer 与 e2e trace 的 kernel duration、allocator 复用、finalize 与 launch gap；在 `8192×3072/rank64` 上优先解释“kernel 更快但总量更慢”。
4. **GEMM 只做资源几何级实验**：可选方向只有降低 accumulator/寄存器以争取 2 CTA，或重做跨 warpgroup/Multi-CTA 流水；A/SFA TMA、S4、8M×1N、简单 prefetch 全部关闭。没有新资源证据时，GEMM 维持现状。
5. **dispatch 继续用逐格 A/B**：任何 LoRA/main 规则变化必须做同进程交错多轮，不得用单轮跨进程 Event 直接改。

### 30.7 验收门槛

- GEMM/ producer 微优化：全矩阵 NSYS 正向约 0.1% 才保留，且不能有超过约 0.2% 的稳定回退。
- 数值：GEMM 调度等价改动要求固定输入 bitwise；producer 改布局要求 qact/ascales bitwise，D rel-L2 <0.01。
- NCU 只用于归因；最终性能用 NSYS/官方 Event，并记录 session 内 native 对照。
