# SVDQ CuTe FP4 采用 CUTLASS Collective API 的可行性与静态性能分析

> 日期：2026-11-16  
> cache-dit：`dev@eced0158b25e597aa629e6ce6551397359d65afe`  
> CUTLASS：v4.8.1，`2c85f497547eb441f0fe80bfb0fb2e6f9add1719`  
> 范围：只读源码、既有测量文档与数量逻辑分析；未改实现、未编译、未运行 cache-dit、未新增 benchmark。  
> 下文将 **measured**（既有实测）、**derived**（本次静态推导）与 **hypothesis**（待验证判断）分开。

## 1. 结论摘要

### 1.1 可行性必须拆成三个问题

| 问题 | 结论 | 原因 |
|---|---|---|
| SM120 NVFP4 主 GEMM 能否使用 Collective | **可以，但现役 pack 不能直通** | 现役 SVDQ 与 stock Collective 使用同一个 `SM120_16x8x64_TN_VS`、E2M1、UE4M3、VS=16、FP32 accumulator；B/SFB 已符合 canonical 布局，A/SFA 则需由 producer 改为 row-major/canonical。 |
| 能否用 stock `CollectiveBuilder + GemmUniversal` 直接替换完整 SVDQ | **rank=0 可以条件性迁移；rank>0 不可以直接替换** | stock mainloop 只有一对 A/B 和 SFA/SFB；stock epilogue/EVT 没有第二个 K=R 的矩阵归约，不能表达 `D @ L1^T` 并与主积共享 accumulator。 |
| 自定义 LoRA Collective 是否可行 | **可行，但已不是“只使用 stock API”** | FlashInfer 已有 `CollectiveMmaLoRaSm120 + GemmUniversalLoRaSm120` 的 C++ 实现，证明可以扩展 Collective/mainloop/kernel driver 来保留 LoRA 融合。 |

### 1.2 性能结论

**现有证据不支持“迁移 Collective 后会比当前 SVDQ CuTe 更快”的普遍判断，也不支持现在做全量替换。**

- 当前手写 CuTe 已经不是未优化的朴素 mainloop：既有 15 格官方矩阵中，row-major GEMM 相对 native 的几何均值加速为 **1.0248×**，12/15 格更快；NCU 中位 Tensor active 为 **84.2%**，无 local spill。
- Collective 仍走 SM120 `mma.sync`、TMA→SMEM→RF，不会换成更高吞吐的另一类指令。它主要改变数据搬运、流水、tile 和 scheduler，而不是改变主 GEMM 的算术指令。
- 现役 SVDQ 的 A/SFA G2R 是针对专用 pack 做的优化。既有同形状 nsys 中，全 TMA 为 **137.4 µs**、hybrid G2R 为 **117.2 µs**：全 TMA 耗时比 hybrid 高 **17.2%**，换成 hybrid 可节省全 TMA 时间的 **14.7%**。这不能证明所有 Collective 都慢，但否定了“标准 TMA mainloop 天然优于当前混合路径”的推断。
- 最直接的 Collective 参照实现是 FlashInfer `cutlass-sm120`。既有 10 格 kernel-only 对照中，cache-dit CuTe 在 9/10 格更快，FlashInfer/cache-dit 延迟比几何均值为 **1.1438×**。该对照同时改变 pack、tile、epilogue 和调度，**只能反驳“直接迁移即可获益”，不能把差距全部归因于 Collective API**。
- 正式汇总没有给出足以支持 shape 选择的 Collective 逐格结论。大 M 是否更有利仍是待验证假设，不能借用 native 或 FlashInfer cute-dsl 的 16K 结果代替 cutlass-sm120 证据。

### 1.3 默认建议

1. **生产路径保留当前手写 CuTe，不做全量 Collective 替换。**
2. 若目标是隔离 stock mainloop 的潜在收益，最小研究对象是 **rank=0、无 LoRA** 路径；否则 LoRA 融合和 pack 改写会污染对照。
3. rank>0 若坚持 Collective 化，合理路线是 FlashInfer 式 **自定义 LoRA Collective + 自定义 kernel driver**，而不是 stock Builder 直接替换，也不是拆成两个 GEMM 后忽略额外流量。
4. 在没有同 pack、同 shape、同进程 A/B 证据前，不应把通用 FP4/FP8 的迁移收益外推到 SVDQ。

---

## 2. FP4/FP8 现有 Collective 用法能复用什么

### 2.1 两族的共同结构

当前 `csrc/kernels/fp4` 与 `csrc/kernels/fp8` 都已删除手写 GEMM mainloop，采用：

```text
epilogue CollectiveBuilder
  -> mainloop CollectiveBuilder(StageCountAutoCarveout)
  -> gemm::kernel::GemmUniversal
  -> device::GemmUniversalAdapter
  -> get_workspace_size / initialize / run
```

关键实现：

| 实现 | 配置与职责 | 证据 |
|---|---|---|
| FP4 | `128x128x256`、cluster `1x1x1`、epilogue tile `64x64`、A/B alignment 32；block-scaled mainloop | `csrc/kernels/fp4/gemm.cuh:214-239` |
| FP4 EVT | FP32 中计算 `g_w[n] * (g_x[m] * P) + bias[n]`，最后一次 BF16 转换 | `csrc/kernels/fp4/gemm.cuh:43-64,122-170` |
| FP8 per-row/per-tensor | `128x128x128`；scale 在 EVT 中广播，per-tensor 通过零 stride 复用同一实例 | `csrc/kernels/fp8/gemm/gemm.cuh:202-227,285-334` |
| FP8 per-block | `Sm120BlockwiseScaleConfig<1,128,128>`，scale 进入 mainloop，epilogue 只加 bias | `csrc/kernels/fp8/gemm/gemm.cuh:229-283,336-387` |
| host | 每次构造 Arguments、查询 workspace、`initialize`、`run` | `csrc/kernels/fp4/gemm.cuh:273-338`；`csrc/kernels/fp8/gemm/gemm.cuh:285-387` |

### 2.2 可直接借鉴的部分

- **Builder/host 编排**：stride、canonical SF layout、workspace、adapter 调用顺序。
- **EVT 的逐元素融合**：SVDQ 的 `alpha`、`wcscales[N]`、`bias[N]` 都是标量或列广播，可以在主积之后用 FP32 EVT 表达。
- **SM120 类型隔离**：FP4/FP8 把 CUTLASS 模板留在 `.cu/.cuh`，pybind 入口只接触普通指针和基础类型。
- **生产 shape 优先于尾部通用性**：FP4 最终只保留 production tile，非生产尾部由 padding 处理；这减少了第二套 mainloop 的维护面。

### 2.3 不能直接外推的部分

通用 FP4/FP8 都是 **单 GEMM + 逐元素 epilogue**。SVDQ 多出一个沿 rank 维收缩的 BF16 GEMM，且要求与主积共用 accumulator。两者不是同一个算子闭包。

通用 FP4 的既有 NCU 对照显示，手写 WS 到 Collective 的 kernel duration 为 **128.67→106.24 µs**，Tensor pipe 为 **60.63%→75.19%**；这证明 Collective 在通用 FP4 的特定上下文能改善 mainloop 结构，但 NCU duration 不是 SVDQ 的交付性能，也不包含 SVDQ LoRA/producer 成本（measured；`.agents/skills/cache-dit-custom-kernel/references/fp4.md:296-311`）。

FP8 迁移后的公开对照是 cache-dit 与 TorchAO，而不是旧手写实现的严格 A/B；FLUX compile 单次结果为 7.11 s vs 7.00 s，不能用来证明 SVDQ 会提速（measured；`.agents/skills/cache-dit-custom-kernel/references/fp8.md:396-450`）。

---

## 3. SVDQ 当前实现的硬契约

### 3.1 数学与舍入顺序

当前算子为：

\[
P=\operatorname{dequant}(A)\operatorname{dequant}(B)^T,\qquad
L=D L1^T
\]

\[
Y=\operatorname{BF16}((\alpha P)\cdot wcscales[N]+bias[N]+L)
\]

`alpha` 与 `wcscales` 只作用于主积 P，不能作用于 LoRA 项 L。两种 MMA 直接累加到同一个 FP32 `tCrC`，最后只做一次输出 BF16 转换（`csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cuh:55-80,296-353,1071-1107,1226-1235`）。

注意：“单次舍入”指最终 Y。row-major D 自身在 producer 中已经由 FP32 归约转为 BF16；native D 则在 GEMM gather 时乘 rank scale 并转 BF16。不能把最终输出单次舍入扩大解释为整个量化链只有一次舍入。

两种调度的实际运算顺序不同：

- **main-first**：`C=P` → `C*=alpha` → `C=fmaf(C,wc,bias)` → `C+=L`。
- **LoRA-first**：`C=L` → `C*=1/(alpha*wc)` → `C+=P` → 最终再乘 `alpha*wc` 并加 bias。

LoRA-first 的逆缩放不是注释层面的可选细节，而是让 LoRA 与首块 A/SFA 预取重叠的调度组成部分（`gemm_w4a4.cuh:763-802,1174-1224`）。它与 main-first 不保证逐位相同。

### 3.2 数据布局不是 stock row-major/canonical 全集

| 张量 | 当前布局与路径 | 对 Collective 的影响 |
|---|---|---|
| A | producer 的 native 256-row interleaved E2M1 pack；直接 G2R 到寄存器 | **不兼容 stock mainloop 的 row-major A + TMA**；需改 producer 或增加 repack。 |
| SFA | 与 native A 配套的 interleaved scale pack；直接 G2R | **不是 canonical 128x4 gmem atom**；需与 A 一起改。 |
| B | row-major packed E2M1，K-major；TMA→SMEM→RF | 已符合 TN 数据方向。 |
| SFB | canonical 128x4、512 B atom；TMA | 已符合 stock Collective 布局。 |
| D | row-major BF16 `[M,R]`，或 native FP32 fragment D | stock 路径只能自然覆盖前者；后者需要独立适配。 |
| L1 | row-major BF16 `[N,R]`；与 D 一起复用 A/B stage storage | 需要自定义 LoRA mainloop 或独立 GEMM。 |
| output | FP32 accumulator → 一次 BF16 → STSM→SMEM→TMA store | stock epilogue 可完成输出，但不能自行完成第二个 GEMM。 |

证据：`gemm_w4a4.cuh:69-76,82-107,375-445,610-705`；host 校验位于 `gemm/gemm_w4a4.cu:75-147`。

**约束来源必须分清**：`M%256` 来自 native activation pack，不是 stock GEMM 的 M 尾部能力；`N%128` 与 SF atom 相关；`K%64` 对应 block-scaled MMA 原子。权重一次性 pack 另有更严的 `K%128` 要求。迁移 GEMM 不会自动消除 producer 约束。

### 3.3 当前调度与资源几何

- 固定 `BM=128`；默认 `BN=256/BK=128/S=3`，回退 `BN=128/BK=64/S=4`。
- 384 线程：128 线程 producer warpgroup，只有物理线程 0 发 TMA；256 线程 consumer，8 个 MMA warp。
- A/SFA 走 G2R；B/SFB/D/L1 走 TMA；output 也走 TMA store。
- main K tile 与 LoRA rank tile 共用连续 `g/stage/phase` 环。
- `kRankTile=BK/4`，使 D/L1 的 BF16 stage 与 A/B 的 packed FP4 stage 字节数精确相等。
- output tile 在 mainloop 结束后覆盖整个 pipeline SMEM，避免独立 64 KiB output buffer。
- host 侧已有 5 个 TMA descriptor 的 `thread_local` cache，key 包含指针、shape、rank；动态 SMEM attribute 每设备只设置一次（`gemm_w4a4.cuh:1241-1249,1285-1328`）。

---

## 4. CUTLASS 4.8.1 SM120 能力矩阵与缺口

### 4.1 已有能力

| 能力 | CUTLASS 4.8.1 SM120 状态 | 证据 |
|---|---|---|
| NVFP4 atom | `SM120_16x8x64_TN_VS`，E2M1×E2M1，UE4M3，VS=16 | `csrc/third_party/cutlass/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:87-148` |
| accumulator | 寄存器 FP32；不是 TMEM | `.../sm120_blockscaled_mma_tma.hpp:714-755` |
| 输入方向 | 仅 TN，即 A row-major、B column-major，两者 K-major | `.../sm120_blockscaled_mma_builder.inl:97-105` |
| cluster | 只能 `1x1x1`，无 programmatic multicast | 同上，`:101-105` |
| SF 布局 | `Sm1xxBlockScaledConfig<16>`，128 行×4 factor 的 512 B atom | `.../cutlass/detail/sm100_blockscaled_layout.hpp:48-114` |
| 默认调度 | Auto→Cooperative block-scaled SM120，带 persistent/CLC scheduler | `.../sm120_blockscaled_mma_builder.inl:239-285` |
| StreamK | 有 SM120 NVFP4 device 测试 | `.../test/unit/gemm/device/sm120_blockscaled_tensorop_gemm/sm120_bs_gemm_nvf4_nvf4_f32_f32_stream_k.cu:105-121` |
| epilogue | SM90 风格 TMA epilogue + EVT，可做广播与逐元素计算 | `.../epilogue/collective/builders/sm120_builder.inl:469-515` |

SM120 没有 SM100 的 `tcgen05/TMEM/2SM UMMA` 路径；不能把 SM100 的 LoRA Collective 直接当作 SM120 实现依据。

Pingpong 需要谨慎表述：builder 源码存在 Pingpong 分支，但仓内 FP4 调优记录写明 F4 block-scaled Pingpong 被拒，具体触发条件未在本次静态分析中定位。因此本报告不把它列为已可用路线（`.agents/skills/cache-dit-custom-kernel/references/fp4.md:348-352`）。

### 4.2 stock mainloop/kernel 的结构性缺口

stock mainloop 的 `mma()` 只有一组 A/B/SFA/SFB，并在入口执行 `clear(accum)`；`mma_tail()` 为空（`sm120_blockscaled_mma_tma.hpp:714-727,903-906`）。

stock cooperative kernel 的生命周期为：

```text
partition_fragment_C
  -> mainloop.mma(accum)
  -> mma_tail
  -> scheduler.fixup
  -> epilogue.store(accum)
```

中间没有第二个 GEMM 的插入点（`gemm/kernel/sm90_gemm_tma_warpspecialized_cooperative.hpp:785-858`）。

EVT 能读取 accumulator、广播、逐元素计算和存储，但不提供沿 R 维归约的 MMA 节点。`Sm90AuxLoad` 读取的是与输出同 MN 的辅助张量，不是 `[M,R]` 与 `[N,R]` 的收缩（`epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp:203-270`）。

因此，在保留 BF16 LoRA、当前 scale 顺序和最终单次输出舍入的前提下：

- `alpha*P*wcscales+bias`：**EVT 可表达**。
- `D@L1^T`：**stock EVT 不可表达**。
- 与主积共享 FP32 accumulator、最终一次 BF16 转换：**需要自定义 mainloop/kernel 或自定义 epilogue 内的 MMA hook**。

---

## 5. 可选迁移路线与可行性分级

| 路线 | 数据流 | 可行性 | 数值/覆盖代价 | 工程判断 |
|---|---|---|---|---|
| A. stock Collective，仅 rank=0 | canonical A/SFA → 主 GEMM → EVT scale/bias | **条件性可行** | 必须改 producer 的 A/SFA 输出；M%256 可随 pack 重设计重新评估 | 最干净的 mainloop 对照，但不能代表 rank>0。 |
| B. stock 主 GEMM + 独立 LoRA GEMM + EVT AuxLoad | 先算并落地 `L[M,N]`；主 GEMM epilogue 读回并相加 | **功能可行** | 多一次 launch 和 `[M,N]` 中间写+读；BF16 aux 多一次舍入，FP32 aux 流量翻倍；只自然对应 main-first | 额外流量大，静态上不具吸引力。 |
| C. 自定义 LoRA Collective + 自定义 kernel driver | 主积与 LoRA 共用 FP32 accumulator，再进 epilogue | **可行，已有 FlashInfer 证明** | 可保留最终单次 BF16；需自行支持 rank、K tail、两种 D 与调度 | 不是 stock-only；维护和升级成本高。 |
| D. 自定义 epilogue 内做 LoRA MMA | stock 主 mainloop，epilogue subtile 前执行 BF16 MMA | **可行，FlashInfer 已采用部分路径** | 需要 fork epilogue 并处理寄存器/SMEM；大 tile 可能 spill | 仍是自定义 Collective 生态，不是配置 Builder 即可完成。 |
| E. 保留手写 CuTe，选择性借鉴 scheduler/流水 | 不改公开数据契约 | **可行但非即插即用** | StreamK 下 LoRA 必须在 fixup 后、且仅由 epilogue-owning workunit 执行一次 | 风险小于全替换；已有 persistent/non-WS 负结果，需新证据。 |

### 5.1 路线 B 的额外舍入不能省略

若独立 LoRA GEMM 输出 BF16 `L`，则 L 先舍入一次，再与主积相加，最终 Y 又舍入一次；这不同于当前 FP32 中直接 `C+=L`。

AuxLoad 也允许 FP32 辅助张量。FP32 aux 可避免这次 BF16 中间舍入，但写+读流量是 BF16 的两倍；独立 GEMM 的 K 归约顺序仍可能不同，不能预先承诺 bitwise。

### 5.2 FlashInfer 是直接的可行性证据，不是 stock 反例

FlashInfer SM120 C++ 实现：

- 用 mainloop Builder 抽取 `DispatchPolicy/StridePair/SmemLayoutAtoms/SmemCopyAtoms/TiledMma` 等类型；
- 真正实例化的是 fork 的 `CollectiveMmaLoRaSm120`；
- kernel 是 fork 的 `GemmUniversalLoRaSm120`；
- 部分 tile 使用带 subtile hook 的 epilogue fork；
- LoRA 与主积共享 FP32 accumulator，并在 StreamK fixup 后执行。

证据：`../flashinfer/include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h:155-172,193-269`；`../flashinfer/include/flashinfer/gemm/nvfp4_svdquant_gemm_collective_sm120.h:50-73,139-152`。

它证明的是“**扩展 Collective 可以实现**”，而不是“**stock Builder 已经支持 SVDQ**”。

其数学契约为 `alpha*(P+L)+bias`，要求调用方预先将 L1 除以 alpha；cache-dit 则是 `alpha*wc*P+bias+L`。cache-dit 适配层还会把 `wcscales` 折进 FP8 block scale，属于数值近似，不是等价替换（`src/cache_dit/kernels/flashinfer/svdquant.py:1-8,145-180`；`src/cache_dit/quantization/svdquant/packing.py:205-241`；`../flashinfer/flashinfer/gemm/gemm_svdquant.py:801-839`）。

FlashInfer 的 cache-dit 适配目前只开放 rank 32/64、SM120（`src/cache_dit/kernels/flashinfer/svdquant.py:19-35,92-110`）；其本地 CUTLASS 基线为 4.5.0，而 cache-dit 为 4.8.1，不能假定 fork 源码可无修改平移。

---

## 6. 静态数量分析

以下均为 **derived**，不是新增实测。

### 6.1 SMEM：当前 overlay 是性能与资源联合设计

每 stage 主输入字节数为：

\[
S_{stage}=\frac{BM\cdot BK}{2}+\frac{BN\cdot BK}{2}
+\frac{BM\cdot BK}{16}+\frac{BN\cdot BK}{16}
\]

其中前两项是 packed E2M1，后两项是每 16 元素一个 UE4M3 scale。

| 配置 | A | B | SFA | SFB | 每 stage | stage 总数 | output |
|---|---:|---:|---:|---:|---:|---:|---:|
| 当前 SVDQ `128x256x128,S3` | 8 KiB | 16 KiB | 1 KiB | 2 KiB | **27 KiB** | **81 KiB** | 64 KiB，覆盖 pipeline |
| 当前 SVDQ `128x128x64,S4` | 4 KiB | 4 KiB | 0.5 KiB | 0.5 KiB | **9 KiB** | **36 KiB** | 32 KiB，覆盖 pipeline |
| 通用 FP4 Collective `128x128x256` | 16 KiB | 16 KiB | 2 KiB | 2 KiB | **36 KiB** | 由 AutoCarveout 决定 | 独立 epilogue storage |

LoRA overlay 的等量关系为：

\[
BM\cdot(BK/4)\cdot2=BM\cdot BK/2
\]

默认配置中 D=8 KiB、L1=16 KiB，恰好覆盖 A/B stage。若 Collective 不能复用同一存储，就必须另加 D/L1 buffer；同时 stock epilogue 与 CLC scheduler 也占 SMEM。即使主输入字节相同，也不能假定仍能得到 S3。

这解释了为什么通用 FP4 已记录：`BN256/BK256` 只剩一个非法 stage，显式 S4 初始化失败，`BN256/BK128` 约 150 µs（measured；`.agents/skills/cache-dit-custom-kernel/references/fp4.md:348-352`）。这些数字不是 SVDQ 的实测，但构成 tile/SMEM 外推的硬警告。

### 6.2 CTA/wave：小 tile 不等于减少尾损

当前 grid 为：

\[
CTA=(M/128)(N/BN),\qquad waves=CTA/110
\]

110 来自既有 RTX PRO 5000 测试口径。

| M×N | BN=256 | BN=128 |
|---|---:|---:|
| 4096×3072 | 384 CTA，3.49 waves | 768 CTA，6.98 waves |
| 1024×3072 | 96 CTA，0.87 waves | 192 CTA，1.75 waves |
| 16384×8192 | 4096 CTA，37.24 waves | 8192 CTA，74.47 waves |

小 M 时 BN=256 确实会留下空闲 SM；BN=128 增加 CTA 数，但也把每 CTA 工作量减半、增加 tile/调度次数。总 wave 数近似翻倍，不能只看末波空闲比例推断净收益。

StreamK 能改变主 K 归约的分配，但 LoRA 必须在主积 fixup 之后恰好执行一次；这也是 FlashInfer 修改 kernel driver 的原因之一。当前手写 kernel 不能只替换 scheduler tag 就获得合法 StreamK 语义。

### 6.3 拆出 LoRA 的 `[M,N]` 中间流量

路线 B 的额外中间张量写一次、由 AuxLoad 读一次：

\[
Traffic_{aux,BF16}=2MN+2MN=4MN\ bytes
\]

\[
Traffic_{aux,FP32}=8MN\ bytes
\]

| M×N | BF16 中间张量 | BF16 写+读 | FP32 写+读 |
|---|---:|---:|---:|
| 4096×3072 | 24 MiB | **48 MiB** | **96 MiB** |
| 4608×12288 | 108 MiB | **216 MiB** | **432 MiB** |
| 16384×8192 | 256 MiB | **512 MiB** | **1024 MiB** |

这些是额外逻辑流量，不等于全部落到 DRAM；L2 能吸收多少取决于调度与生命周期。96 MiB L2 口径下，4608×12288 的 BF16 中间张量本身已为 108 MiB，不能默认可完整驻留。

对比 rank32 的 LoRA 输入总量 `2R(M+N)`：

- 4096×3072：D+L1 为 0.4375 MiB；单个 BF16 `[M,N]` 中间张量是其约 **54.9×**。
- 4608×12288：D+L1 为 1.03125 MiB；单个 BF16 中间张量是其约 **104.7×**。

因此“把 LoRA 拆出去再由 epilogue 加回”不是免费简化；它把一个小 rank 维问题变成全 MN 中间张量问题。

### 6.4 A/SFA 改布局的成本

stock Collective 需要 row-major A 和 canonical SFA。若通过独立 repack 实现，读+写额外流量为：

\[
2\left(\frac{MK}{2}+\frac{MK}{16}\right)=\frac{9MK}{8}\ bytes
\]

| M×K | A pack | SFA | A+SFA 写+读 |
|---|---:|---:|---:|
| 4096×3072 | 6 MiB | 0.75 MiB | **13.5 MiB** |
| 16384×8192 | 64 MiB | 8 MiB | **144 MiB** |

更合理的研究方向是让 producer **直接写 canonical A/SFA**，这样不增加独立 repack 往返；但它会改变 producer 的 store 模式、pack ABI 和现有 bitwise golden，并同时失去当前 G2R 路径。净收益不能只根据 GEMM 推断。

### 6.5 LoRA 算术量小，但不能按 FLOPs 直接估时

主积与 LoRA 的 FLOPs 比为：

\[
\frac{2MNR}{2MNK}=\frac{R}{K}
\]

K=3072 时，rank32/64/128 的 LoRA FLOPs 分别约为主 GEMM 的 **1.04%/2.08%/4.17%**。但 LoRA 使用 BF16 `m16n8k16`，主积使用 NVFP4 block-scaled `m16n8k64`，二者指令吞吐、操作数搬运和流水位置不同；该比值不能当作延迟占比。

### 6.6 Host 固定成本方向不确定

通用 FP4/FP8 当前每次调用执行 `get_workspace_size + initialize + run`，initialize 会构造底层参数/TMA descriptor。当前 SVDQ 则缓存 5 个 descriptor，并按设备缓存动态 SMEM 设置。

因此直接照搬 FP4/FP8 host 包装可能增加每次 Linear 的 host 成本；若未来做原型，应缓存可复用 Params/descriptor，并把 host+launch 与 device kernel 分开计量。本任务没有既有迁移后同口径数据，不能给出 µs 预估。

### 6.7 Amdahl：GEMM 收益不等于整层或 e2e 收益

若整层其他部分时间不变，GEMM 延迟下降 x，整层收益为：

\[
Speedup_{layer}=\frac{1}{1-f_{GEMM}+f_{GEMM}(1-x)}
\]

既有官方矩阵的 4096×3072×3072/rank32 一行中，CuTe quantize=43.86 µs、整层 e2e=160.50 µs（measured）。将余量全部近似归为 GEMM，则 `f_GEMM≈(160.50-43.86)/160.50=72.7%`；GEMM 延迟下降 10% 时，两层内核整链延迟约下降 **7.3%**，等价 speedup 为 **1.078×**，均为近似上界。该近似包含 allocator 等余量，也不是全模型 e2e 收益。

用户文档另记录 quantize 占 e2e 的 14–31%（`docs/user_guide/QUANTIZATION.md:1167-1170`）。无论采用哪个形状，都必须先乘 GEMM 在整链中的占比，不能把 GEMM-only 百分比直接写成 FLUX e2e。

---

## 7. 既有性能证据如何约束判断

| 证据 | 类型与口径 | 能说明什么 | 不能说明什么 |
|---|---|---|---|
| 通用 FP4 手写 WS→Collective：NCU 128.67→106.24 µs，Tensor 60.63→75.19% | measured，NCU | Collective 在单 GEMM+EVT 中能改善主流水 | 不包含 SVDQ LoRA/producer；不能外推 SVDQ wall-clock。 |
| SVDQ 全 TMA 137.4 µs vs hybrid 117.2 µs | measured，同形状 nsys | 当前专用 pack 下，A/SFA 全 TMA 耗时高 17.2% | 不等于所有 Collective 实现都有相同回退。 |
| 当前 SVDQ row-major GEMM vs native：1.0248×，12/15 | measured，15 格 CUDA Event | 当前 CuTe 已小幅领先 native，已进入平台期 | 不是 Collective 对照，也不代表跨会话稳定收益。 |
| 当前 NCU Tensor 84.2%、math throttle 36.2%、wait 30.8%、0 spill | measured，15 格 NCU 中位 | 主要空间在 MMA 依赖/发射与 G2R 联合优化 | `1/0.842≈1.19×` 只是粗上限，不是 Collective 预期收益。 |
| 当前手写 persistent 慢 3.0–6.7%，non-WS 慢 1.6–4.3% | measured | 当前 overlay/WS 结构下这些路线已证伪 | stock Collective 的独立 epilogue 不改变资源结构，不能直接套同一因果。 |
| FlashInfer Collective-fork 对照：cache-dit CuTe 9/10 更快，几何均值 1.1438× | measured，10 格、单会话、CUDA Event | 现成的 LoRA-fused Collective 路线没有显示普遍优势 | pack、数值、tile、调度均不同，不能做纯 API 归因。 |

来源：`.agents/skills/cache-dit-custom-kernel/references/svdq-nvfp4-cute.md:868-907,927-991`；`.agents/skills/cache-dit-custom-kernel/references/fp4.md:285-352`。

### 7.1 分路线性能预判

| 路线 | 静态性能判断 | 置信度 |
|---|---|---|
| rank=0 stock Collective | 有主流水优化可能，但 producer 改布局和 host 成本决定净值；当前无直接证据 | 低 |
| 独立 LoRA + AuxLoad | 额外 MN 往返、launch 与潜在舍入使其很难优于当前单 accumulator 融合 | 中 |
| FlashInfer 式自定义 LoRA Collective | 技术可行；现有直接对照整体落后，大 M 是否更有利仍待逐格验证 | 中 |
| 全量替换当前实现 | 覆盖 rank/tail/双 D 布局和调度所需工作大，未显示收益 | 中高 |
| 保留现状 | 与当前性能平台、已有负结果和维护成本一致 | 高 |

---

## 8. 工程成本与维护面

### 8.1 全量替换至少要同时解决

1. **producer ABI**：A 与 SFA 从 native interleaved pack 改为 row-major/canonical，或引入额外 repack。
2. **LoRA 融合**：自定义 mainloop/kernel/epilogue，保持 alpha/wc/bias 只作用主积及最终单次 BF16。
3. **双 D 布局**：row-major BF16 与 native FP32 fragment D；若只支持前者，必须明确保留旧路径而不是静默回退。
4. **rank 覆盖**：当前常用 rank32/64/128，以及通用路径的其他 rank；FlashInfer cutlass-sm120 只覆盖 32/64。
5. **K/rank 几何**：当前 kernel 接收 K%64；部分 LoRA Collective byte-exact 配置要求一个 stage 容下完整 rank tile（`LoRaStorageK=TileK/4`），不能意外收紧 K/rank 组合。
6. **调度**：main-first 与 LoRA-first 的性能分工；StreamK 下 fixup 与 LoRA 恰好一次。
7. **host 生命周期**：descriptor/Params cache，workspace 与 StreamK workspace。
8. **数值门禁**：pack golden、GEMM parity、producer+GEMM 联合 parity，不能只测其中一个。
9. **构建成本**：更多 tile/rank/scheduler 实例化；FlashInfer 侧以多 kernel、多 tactic 换覆盖，不等于维护量减少。

### 8.2 Collective 化可能减少什么

如果仅限于 rank=0 单 GEMM，stock Collective 确实可以删除自管 mbarrier ring、K-block copy/MMA 和 epilogue 细节，并跟随 CUTLASS 获得调度修复。

但对完整 SVDQ，FlashInfer 路线仍需维护自定义 mainloop、kernel driver 和部分 epilogue fork。它减少的是“重复发明标准组件”，不是消除自定义核心。再考虑 native/cute/collective 并存，运行时与测试矩阵反而扩大。

---

## 9. 最终判断与重新评估条件

### 最终判断

- **指令与类型层面：可行。**
- **stock Collective 直接替换完整 SVDQ：不可行。**
- **自定义 LoRA Collective：工程上可行，FlashInfer 已证明，但属于大型 fork/扩展，不是轻量重构。**
- **性能优势：当前无证据支持普遍优势；4K/8K 有回退风险，大 M 是否更有利仍待同 pack、同后端逐格验证。**
- **当前建议：不迁移完整实现；若只做研究，先用 rank=0 隔离 stock mainloop，再决定是否承担 LoRA Collective 成本。**

### 只有满足以下条件，才值得重新开启实现评估

1. 有同 pack、同数值契约的 rank=0 原型，证明 stock mainloop 在目标 shape 上有稳定 GEMM-only 收益，而非借用通用 FP4 的结论。
2. canonical producer 不引入独立 repack，且 producer+GEMM 总时间不退化。
3. LoRA 方案不靠 BF16 `[M,N]` 中间往返掩盖成本；若使用 FP32 aux，必须把翻倍流量和独立 GEMM 算入。
4. 自定义 LoRA Collective 能覆盖 rank32/64/128、K64 tail、main-first/LoRA-first 或给出明确 dispatch，不静默缩小支持面。
5. 在同一进程、同 pack、同 shape 下做多轮交错 A/B，分别报告 host+launch、GEMM kernel、producer+GEMM 整层和模型 e2e。
6. 数值验证覆盖固定输入 GEMM bitwise（仅限声称调度等价的改动）、qact/ascales golden、D 的既有误差门，以及 producer+GEMM 联合 parity。

在这些证据出现之前，采用 Collective 的合理定位是**研究备选与结构参考**，不是当前 SVDQ CuTe FP4 的升级方向。
