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

数值验收已覆盖 rank 0/32/64/128、独立 producer（identity smooth 下 v1 qact 与 native 逐字节一致，ascales 值集合一致）、真实量化 Linear 在 rank32/64/128 的 native/hybrid parity、INT4 隔离和 FLUX 出图。随机有限量化状态下高层 hybrid/native relative L2 约 0.00215；94 个 SVDQ kernel/runtime 测试通过。

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

identity smooth 下，当前 producer 的 v1 qact 与 native quantize 逐字节一致；ascales 在 native 中按 `[K/16,M]` 存储，当前 v1 pack 为扁平顺序，但两者的 scale 值集合完全一致。随机 smooth 的差异来自既有的 BF16 smooth 舍入，不是布局错位。仍然没有 `lora_act_to_bf16` 或任何 finalize。

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

验证：`tests/kernels/test_ops_dispatch.py`、`test_svdquant_cute.py`、`test_svdquant_runtime.py`、`test_svdquant_source_manifest.py`、`test_quantize_config.py` 共 94 passed。producer 测试覆盖 rank32/64/128，并包含 native v1 pack parity、非 256 padded batch 负向用例和 0.12 去量化精度门槛；真实量化 Linear 的 native/hybrid 端到端 parity 覆盖 rank32/64/128。恢复依据与 standalone 证据在 `.tmp/svdq-nvfp4-cute/g2r-recovered/`。

**边界**：M 必须是 256 倍数（v1 以 256 行为块）；N 必须为 128 倍数。M4096/N3072 下 hybrid kernel-only 只比 native 快约0.3%，完整层约1.3%–1.5%；但大 shape 下 native 反超（16Kx8Kx8K 快6.58%，见 §12），不能宣称 hybrid 普遍快于 native。下一步应优先解释大 shape 退化，而不是继续微调调度。

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
2. **代价是 A 的重复 global load。** 每个 CTA、每个 MMA 线程都直接从 v1 pack 取 A/SFA，跨 N tile 和 ThrN 广播产生重复请求：global 请求/ sectors 相对全 TMA 升4倍/31倍，L1 hit 从94.5%降到55.1%，`long_scoreboard` 从0.67升到2.92。但这些额外流量有87%在 L2 命中，DRAM 吞吐不变，而且总量仍远小于 native（请求41%、sectors22%）。
3. **hybrid 优于 native 的原因不同：它保留了 weight 的 CTA 内共享。** native 每个 warp 直接从 gmem 取 fragment，global 请求是 hybrid 的2.45倍、sectors是4.56倍，并带约103万 local-store sectors 的 spill；hybrid 把 B/SFB 一次 TMA 进 smem 给8个 consumer warp 复用，regs 168 vs 252，tensor active 79.8% vs 74.3%。所以 hybrid 同时避免了“全 TMA 的 smem 税”和“native 的重复 weight load/spill”。
4. 尚未隔离 A 与 SFA 各自的贡献。SFA 是逐字节 shared copy，A 是 LDSM；从指标看 SFA scalar copy 很可能是主要节省项，但需要“A TMA + SFA G2R”的 A/B 才能定论，不能把全部收益归给 A G2R。

旧全 TMA 优化的失效边界：

| 旧结论 | hybrid 下是否仍成立 | 新证据 |
|---|---|---|
| K-block 流水必 spill、不可用 | **失效**：hybrid 上逐 k-block `load A_k -> copy B_k -> MMA_k` 为 168 regs、零 spill，wait/math/sleeping 各降约5% | 大 shape 复测仍未形成明确收益：`8192x3072` -0.23%、`4096x6144` +0.31%、`8192x6144` +0.20%、`16Kx8Kx8K` +0.21%，平均约0.12%，未保留 |
| stages 调优已完成、S3 最优 | **部分仍成立**：S2 被输出 overlay 的64KiB > pipeline55.3KiB 静态拒绝；S4 的110.6KiB超过99KiB。G2R 只释放未使用的 SFA 3KiB，A/B 区仍被 D/L1 per-stage overlay 占用 | 要增加 stages，必须先解耦 D/L1/output overlay，不能只靠 G2R 释放的 smem |
| persistent CTA 慢4.4% | **未重测，旧机制仍在**：`store_done` 让同一 CTA 下一 tile 等上一 tile 的 output TMA store，丢失 CTA 间 epilogue/mainloop overlap；该串行化与 A 走 G2R 无关 | 只有改成不串行 store 的 persistent 设计才值得重测 |
| BN128 小 tile 慢23% | **需在 hybrid 下重测**：旧数据来自全 TMA；hybrid 的 shared 结构已变，且 BN128 可用 S4 | 未做，不能直接沿用旧结论 |
| direct G2R 因 pack 不匹配不可行 | **对 native pack 仍成立，对 v1 pack 已失效**：v1 producer + G2R 已落地 | 当前 hybrid 即该路线 |

下一步优先级：

1. A TMA + SFA G2R 的隔离实验，确认收益来自 A 还是 SFA copy；若 SFA 为主，可只针对 SF 设计更低成本的装载。
2. 降低 A G2R 重复 load：跨 N tile 复用、cache hint、或在 cluster 内共享 A。
3. 解耦 D/L1/output overlay，让 S4 或专用 LoRA buffer 成为可能；这是 stages 继续优化的前提。
4. 优化 producer v1 的分散 u32 写回（当前为保持零 finalize 的代价）。
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

按当前证据排序，前一项没有结论前不要跳到后一项：

1. **解释大 shape 下 native 反超（未完成）。** 在 `16384x8192x8192` 上做 native/hybrid 的 NCU 对比，定位是 CTA 波次、L2 复用、A/SFA 重复 G2R load 还是 per-CTA TMA/barrier 固定成本导致 native 快 6.58%。产出必须给出具体 counter（shared/global/L2/stall/occupancy）和可验证假设。
2. **canonical SW64 bank conflict（已完成）。** canonical `tile_to_shape(SW64 atom,...)` + LDSM 将 conflict 降到483并保留；BK64 fallback 用 SW32。不要重做 conflict 消除。
3. **降低 hybrid 重复流量（部分完成，主路线受阻）。** A-only TMA 与 v1 pack 不兼容（rel_l2=1.30），producer coalesced 写回无收益（46.81→47.01us）。只有设计“v1 bytes 经 TMA/共享广播”的新方案才值得继续。
4. **persistent 只在重新设计 output 生命周期后重测。** 当前 `store_done` 让同一 CTA 的下一 tile 等上一 tile 的 TMA store，和 A 走 G2R 无关；没有去串行化设计就不要重复 110-CTA 版本。
5. **FlashInfer CuTeDSL benchmark（已完成，慢）。** 四 shape 比 native 慢9%–15%，不移植；只保留为外部参照。

不要重复的路线：K-block（hybrid 四 shape 平均仅0.12%）、light producer、S2/S4 stages、BN128 小 tile、全 TMA 回退、D/L1 G2R（conflict 可消但慢3.3%）。canonical SW64 已完成，不再重做 conflict 消除。

## 16. A-only TMA 与 producer coalesced 写回（2026-11-05）

- **A-only TMA 不可行**：TMA 只能搬 standard row-major，当前 producer 输出 v1 fragment pack；TMA 后直接 LDSM 得到 rel_l2=1.30。除非 producer 额外输出一份 standard A（增加写带宽/内存），否则不能用 TMA 去掉 A 的重复 G2R load。
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
- 当前 CuTe：A/SFA G2R + B/SFB/D/L1 TMA + canonical SW64/SW32 LoRA，producer 输出 v1 pack + BF16 D。

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
