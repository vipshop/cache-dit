# SVDQ NVFP4 CuTe GEMM：TMA、warp/block 同步与 FlashInfer SM120 对照

> 实施回写（2026-11-05）：P0 的唯一物理线程初始化和 `fence_barrier_init()` 已完成。当前 GEMM 已恢复为 **A/SFA G2R + B/SFB/D/L1 TMA**：TMA producer 只跟踪 B+SFB 或 D+L1，A/SFA 不进入 full barrier；consumer 用 native fragment 直接 gmem load。本文关于 B/SFB stage 环、empty arrive、D/L1 overlay 和 output store 的同步约束仍然成立，但“A/B/SFA/SFB 四路 TMA”的段落已过时。提前 release、K-block、persistent（慢约 4.4%）和 light producer 均不采用。实测见 `svdq-nvfp4-cute.md` §8–§10。

## 1. 范围与结论

本文整理 2026-11-04 对以下实现的静态代码调研：

- cache-dit：`cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh` 中的 `gemm_w4a4_kernel`。
- cache-dit 实际编译使用的 vendored CUTLASS SM120 block-scaled C++ 参考 mainloop。
- FlashInfer SM120 SVDQuant：`flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`。
- FlashInfer SM100 SVDQuant C++ 路径仅用于辨认架构边界，不作为 SM120 调度参考。

本轮没有编译或运行代码，没有新增性能、精度数据。文中性能背景来自 `./svdq-nvfp4-cute.md` 的既有记录：SM120、M=4096/N=K=3072/rank=32 时，CuTe GEMM 约比 native 慢 16.3%。

核心结论：

1. cache-dit 的 mainloop/LoRA 共用 stage 环、LoRA tx count 和 output overlay 的总体设计合理。
2. 当前首先应修正 mbarrier 初始化中的两个正确性风险，再讨论性能。
3. steady-state 最值得优化的是 consumer 的 K-block copy/MMA 软件流水和提前 stage release。
4. FlashInfer SM120 在 64/128 行 MMA atom（8 个 MMA warp）配置下采用 1 个 DMA warp，即 288 线程 CTA；其 K-block 软件流水、producer start state 和 output subtile pipeline 是直接相关的调度参考。
5. FlashInfer 的 SVDQuant D/L1 correction 与 cache-dit 一样复用 mainloop stage 环，但 alpha/bias 契约不同，不能交叉套用。
6. 不应把 FlashInfer SM100 的 `tcgen05/UMMA + TMEM` 路线搬进当前 SM120 kernel。

## 2. cache-dit 当前调度结构

### 2.1 CTA 角色与数据流

当前 CTA 共 384 线程：

- producer warpgroup：128 线程，但只有 `raw_tid == 0` 发射 TMA；其余线程在分支后退出。
- consumer：256 线程，即 8 个 MMA warp。
- producer 在 `__syncthreads()` 后执行 `warpgroup_reg_dealloc<32>()`。
- consumer 执行 `warpgroup_reg_alloc<224>()`。

数据流：

```text
quantize_w4a4_fused_lora_kernel
  -> row-major E2M1 A + CUTLASS 128x4 SFA + BF16 D[M,R]
  -> gemm_w4a4_kernel
  -> TMA A/B/SFA/SFB
  -> SM120 block-scaled NVFP4 MMA
  -> TMA D/L1 + BF16 LoRA-up MMA
  -> scale/bias + BF16 output
```

TMA 只负责 gmem→smem 和最终 smem→gmem。NVFP4 MMA 前仍需通过 LDSM/普通 copy 将 A/B/SFA/SFB 搬到寄存器；优化重点因而不是增加 TMA 指令，而是缩短以下闭环：

```text
等待 stage -> smem→RF copy -> MMA -> release stage -> 下一次 TMA 复用
```

### 2.2 mainloop 与 LoRA 共用连续 stage 环

producer 使用统一全局步号：

```text
g = 0 ... nKt-1          : TMA A/B/SFA/SFB，NVFP4 mainloop
g = nKt ... nKt+nRt-1    : TMA D/L1，BF16 LoRA-up

stage = g % kStages
phase = (g / kStages) & 1
```

LoRA 不重置 `g/stage/phase`，因此可在主 K 循环尾部进入现有 ring 预取。consumer 在 mainloop 后先完成 accumulator 的 alpha/wcscale/bias，再消费 LoRA stage；这一顺序保持单 accumulator 数学契约：

```text
P = dequant(A) @ dequant(B)^T
Y = BF16((alpha * P) * wcscales[N] + bias[N] + D @ L1^T)
```

不能在单 accumulator 中把 scale 推迟到 LoRA 之后，否则 LoRA-up 结果也会被再次缩放。

### 2.3 TMA transaction barrier

main stage 使用一个 `full` barrier 跟踪四类 copy，tx bytes 为：

```text
A + SFA + B + SFB
```

LoRA stage 只复用 A/B 存储，tx bytes 改为：

```text
D + L1
```

这一精确 tx count 是必要的，不能沿用 main stage 的 A+B+SFA+SFB 字节数。

`empty[s]` 的 arrive count 为 8，每个 consumer warp 在完成后由 lane 0 arrive 一次。producer 只有等 8 个 warp 都释放后才能复用该 stage。

## 3. 优先修正的 barrier 初始化风险

### 3.1 两个线程会同时执行 mbarrier 初始化

当前 tid 映射：

```cpp
const int raw_tid = threadIdx.x;
const bool is_producer = raw_tid < 128;
const int tid = is_producer ? 0 : raw_tid - 128;
```

因此：

- producer `raw_tid == 0` 时，`tid == 0`；
- consumer `raw_tid == 128` 时，`tid == 0`。

后续初始化却使用 `if (tid == 0)`，导致两个线程并发对同一组 `full/empty` 执行 `mbarrier.init`。应改为唯一物理线程，例如 `raw_tid == 0`。这是源代码可确定的重复初始化风险，不需要性能测试才能确认。

### 3.2 缺少 `fence_barrier_init()`

当前初始化后只有 `__syncthreads()`。CUTLASS pipeline 和 FlashInfer CuTe DSL 的标准流程均包含：

```text
唯一 initializing warp/thread 执行 mbarrier.init
-> mbarrier_init_fence / fence_barrier_init
-> CTA 或 cluster scope sync
```

应补齐与 SM120 TMA/mbarrier 语义相匹配的初始化 fence，不能只依赖普通 CTA shared-memory 同步。

## 4. `__syncwarp()` / `__syncthreads()` 是否必须

当前 kernel 有 2 个 `__syncwarp()` 和 3 个 `__syncthreads()`。按现有协议，它们都有作用，不能直接删除；可优化的是位置和替代协议。

| 同步位置 | 保护对象 | 当前方案下是否必要 | 可选改造 |
|---|---|---|---|
| mbarrier 初始化后的 `__syncthreads()` | 所有 warp 使用已初始化的 full/empty | 必须保留 CTA 级同步 | 同时补 `fence_barrier_init()` |
| NVFP4 每 stage 后的 `__syncwarp()` | lane 0 不能在其余 lane 未消费完时 release | 必要 | 可提前到最后一次 smem 读取后；删除则须改为全 lane arrive 等替代协议 |
| LoRA 每 stage 后的 `__syncwarp()` | D/L1 overlay 的复用 | 必要 | 同上 |
| output STSM 前的 `__syncthreads()` | output overlay 不能覆盖仍被读取的 pipeline smem | overlay 方案下必要 | 独立 output smem 或完整的 empty-stage 协议才可替代 |
| TMA store 前的 `__syncthreads()` | 单线程 TMA store 必须等待所有 STSM 写完成 | 必要 | 可改成 output subtile + NamedBarrier/store pipeline，但同步关系仍在 |

### 4.1 为什么不建议删除 warp 同步、改成全 lane arrive

若删除 `__syncwarp()`，一种直接替代是让 256 个 consumer 线程全部 arrive，并把 empty count 从 8 改为 256。这样每 stage 的 arrive 数从 8 增至 256，通常比 warp leader 方案更差。

### 4.2 为什么 epilogue 两个 CTA barrier 不能合并

第一个 barrier 位于 output 写入之前，保护“pipeline smem 不再有人读”；第二个位于 STSM 之后，保证“完整 output tile 已写完，TMA store 可以读”。两者之间有实际写操作，不能合并。

`full/empty` stage barrier 也不能自然替代第二个 barrier，因为这里需要所有 consumer 到齐，而不是单个 stage 的生产/消费关系。

## 5. 不依赖 FlashInfer 的优化分析

### 5.1 整 stage copy→MMA 过于保守

当前 NVFP4 consumer：

```text
wait full
-> copy A/B/SFA/SFB 的整个 stage
-> 执行整个 stage 的 cute::gemm
-> __syncwarp
-> lane 0 release empty
```

问题：

- copy 与 MMA 没有显式 K-block 级交错。
- 最后一个 MMA 只消费寄存器，但 stage 要等到全部 MMA 后才释放。
- `#pragma unroll 1` 且 fragment 每 stage 复用，编译器难以自动形成跨 stage 软件流水。

CUTLASS `sm120_blockscaled_mma_tma.hpp` 的参考 mainloop 在 stage 内按 k-block 流水，并在最后一个 k-block 释放旧 stage、准备下一 stage。对 BK=128，两个 k64 atom 可形成：

```text
copy current.k0
copy current.k1 + MMA current.k0
release current stage
copy next.k0    + MMA current.k1
```

注意 CUTLASS C++ 参考的 next-copy-before-last-MMA 顺序不能直接套到 BK=64 单 k-block fragment；FlashInfer 的 mainloop 提供了另一种可覆盖 BK=64 的顺序，见第 6 节。

### 5.2 首批 TMA 被 consumer 初始化阻塞

当前顺序：

```text
barrier init / CTA sync
-> 所有线程构造 consumer views/fragments/accumulator
-> clear(tCrC)
-> producer/consumer 分支
-> consumer 初次 arrive 所有 empty
-> producer 首批 TMA
```

因此首批 TMA 必须等 consumer 初始化完成。可以考虑：

- 尽早分流 producer/consumer；
- consumer 寄存器扩配后立即初次 release，再初始化 fragments；
- 或采用标准 producer start state，让初始 stage 天然可生产，删除初次 empty arrive。

consumer 的大 accumulator 当前还位于 `warpgroup_reg_alloc<224>()` 之前的源代码区域。尽早分流可让 consumer 初始化完全位于寄存器扩配之后，并减少 producer 的无效初始化工作。

### 5.3 load-side proxy fence 可能冗余

当前 producer 每次 TMA 前、NVFP4 consumer 每次 full wait 后、LoRA consumer 每次 full wait 后都执行 `fence_view_async_shared()`。

CUTLASS TMA mainloop 和 FlashInfer 的纯 TMA load 路径均不在这些位置逐 stage 加 proxy fence。`full` transaction barrier wait 已承担 TMA 完成和可见性同步。尤其 consumer fence 由 256 线程执行，值得优先审查。

不能全局删除该 fence：output STSM 是 generic-proxy smem 写，随后交给 TMA store，output 前的 proxy fence 是必要场景。FlashInfer 也只在 cp.async/手工 gmem→smem 写和 output store 等路径使用相应 fence，而不是纯 TMA load 路径。

### 5.4 stage/phase 与 TMA partition

每轮在所有 producer/consumer 中重复计算：

```cpp
stage = g % kStages;
phase = (g / kStages) & 1;
```

可改为递增、回绕、翻转，尤其 stages=3 时可避免重复除法/乘法强度削减序列。main 与 LoRA 两个循环必须保持 state 连续。

TMA gmem/smem partition 也可参考 CUTLASS/FlashInfer 提升到循环外，循环内只索引 `g` 和 `stage`，减少重复地址计算。其实际收益需结合 SASS 判断。

### 5.5 TMA store 尾部存在无效工作

当前只有 consumer `tid == 0` 发射 output TMA store，但全部 256 个 consumer 都执行 `tma_store_arrive()` 和 `tma_store_wait<0>()`。非发射线程只是提交/等待空 group，可将 commit/wait 与 copy 放入同一个发射分支。

这是尾部小优化，不应与主循环收益混为一谈。

### 5.6 单 DMA warp 的静态可行性

当前 4 个 producer warp 中只有 1 个线程工作。若改为 32 线程 DMA，并完全取消寄存器重配置：

$$
(256+32)\times224=64512<65536
$$

从 SM120 64K 32-bit registers/SM 的静态总量看存在可行性。288 线程也可减少 CTA 初始化和 warp slot 开销。

但该方案使静态寄存器池占用约 98.4%，且当前 producer warpgroup 形式与 `setmaxnreg` 相关。真正实施前必须检查 launch bounds、ptxas 实际寄存器数、分配粒度、local spills 和 SASS，不能只凭算式替换。

### 5.7 stage 数与 output overlay 强耦合

默认 BM=128/BN=256/BK=128：

| 区域 | 字节数 |
|---|---:|
| A | 8192 |
| B | 16384 |
| SFA | 1024 |
| SFB | 2048 |
| 单 main stage | 27648 |
| 3 stages | 82944，约 81 KiB |
| BF16 output tile | 65536，64 KiB |

因此：

- 4 stages 为 110592 bytes，超过当前 `99 * 1024` 限制。
- 2 stages 为 55296 bytes，放不下 64 KiB output overlay。
- 当前 3 stages 不只是流水深度选择，也是完整 output overlay 的最小可行值。

fallback BN=128/BK=64 时，单 stage 9216 bytes；3 stages 为 27648 bytes，仍小于 32 KiB output，因此当前 4 stages 也与 overlay 容量相关。

所以不能只扫描 stage 数。减少 stage 必须同时做 output subtile、独立 output 存储或其他 epilogue 改造。

### 5.8 不作为首选的方向

- 不为 A/B/SFA/SFB 拆四套 full barrier：consumer 最终仍需等待全部操作数，额外 wait/arrive 通常只增加开销。
- 不把每 stage 的 per-warp release 改成 CTA-wide barrier：会扩大慢 warp 的影响。
- 不优先拆 NVFP4/LoRA 两套 consumer warp group：LoRA 只含输出 rank 归约的一部分，需要第二 accumulator 或额外 FP32 归约。
- persistent CTA 可摊薄初始化，但会引入跨 tile stage phase、output overlay 和 TMA store 生命周期问题，应排在局部 mainloop 优化之后。

## 6. FlashInfer SM120 对照

### 6.1 正确识别参考路径

`flashinfer/gemm/gemm_svdquant.py` 按设备分流：

- SM100：`nvfp4_svdquant_gemm_*_sm100`，CUTLASS C++、2SM block-scaled UMMA、TMEM accumulator。
- SM120：`dense_blockscaled_gemm_sm120_b12x.py`，CuTe DSL、warp-level `m16n8k64`、寄存器 accumulator。

只有后者与 cache-dit 当前 SM120 kernel 属于同一硬件执行模型。SM100 的 TMEM epilogue、2SM UMMA 和 cluster 细节不能直接移植。

### 6.2 结构对比

| 项目 | cache-dit CUDA | FlashInfer SM120 |
|---|---|---|
| MMA | SM120 block-scaled `m16n8k64` | 同路线 |
| CTA | 384 线程，8 MMA + 4 producer warp | 64/128 行 atom 时为 288 线程，8 MMA + 1 DMA warp |
| consumer 寄存器 | 224 | 232 |
| DMA 寄存器 | 4 个 producer warp 各 32 | 1 个 DMA warp 40 |
| tile scheduler | 每 CTA 一个 output tile | 默认 static persistent scheduler |
| SVDQuant D/L1 | 复用 mainloop A/B stage | 同样复用 mainloop A/B stage |
| NVFP4 consumer | 整 stage copy→整 stage MMA→release | K-block copy/MMA 流水，提前 release |
| mainloop outer loop | `#pragma unroll 1` | steady loop `unroll=2` |
| output | 完整 tile overlay pipeline smem | 独立 `sC`，按 epilogue subtile 输出 |
| epilogue 同步 | 两次 CTA `__syncthreads()` | 256 个 MMA 线程 NamedBarrier + store pipeline |

在 64/128 行 atom（8 MMA warp）配置下，寄存器总量为：

$$
\text{cache-dit}:8\times32\times224+4\times32\times32=61440
$$

$$
\text{FlashInfer}:8\times32\times232+1\times32\times40=60672
$$

在这一配置下，FlashInfer 不是简单增加寄存器预算，而是减少 3 个 warp，并给每个 MMA 线程增加 8 个寄存器。16/32 行 atom 则分别使用 2/4 个 MMA warp，加 1 个 DMA warp 后为 96/160 线程，不属于上述 288 线程和 60672 寄存器算式。

FlashInfer 说明 9-warp CTA 是值得参考的 SM120 组织方式，但其 CuTe DSL `setmaxregister_increase/decrease` 不能在未核对 PTX 和生成代码的情况下机械照搬到当前 CUDA C++ warpgroup 实现。

### 6.3 FlashInfer 的 K-block mainloop

FlashInfer prologue：

```text
wait stage s
load A/B[s][k0]
load SFA/SFB[s] 的完整 stage fragment
```

steady-state 每个 k-block：

```text
若当前为最后一个 k-block：
  release 旧 stage
  advance consumer state
  wait 下一 stage

MMA 当前 k-block
copy 下一 k-block 的 A/B

若刚跨过 stage：
  在旧 SF 已被最后一个 MMA 消费后，copy 下一 stage 的 SFA/SFB
```

与 cache-dit 相比，增加了三类重叠：

1. 当前 k-block MMA 与下一 k-block A/B copy。
2. 最后一个 MMA 之前的旧 stage release。
3. 当前 stage 尾部 MMA 与下一 stage 数据准备。

其顺序是先执行当前 MMA，再覆盖相应 A/B/SF 寄存器，因此不仅适用于 BK=128，也适用于 BK=64。它比直接照搬 CUTLASS C++ 的 next-copy-before-last-MMA 顺序更适合作为 cache-dit 两种 BK 配置的统一参考。

SFA/SFB fragment 保存完整 stage，每 stage 只从 smem 读取一次；跨 stage 时必须等旧 stage 的最后一个 MMA 已使用旧 SF，再写入下一 stage 的 SF fragment。

FlashInfer 的手工 M/N atom unroll 包含规避 DSL `hasAuxTensor` address-space 问题的目的，不是 cache-dit C++ 必须复制的调度措施。

### 6.4 producer start state 与初始化

FlashInfer/CuTe DSL 标准状态：

```text
producer: index=0, phase=1
consumer: index=0, phase=0
```

producer 将初始 buffer 视为 empty，可直接开始生产，不需要 consumer 先对所有 empty barrier arrive。这正好消除 cache-dit 的首批 TMA 冷启动等待。

FlashInfer 的 `PipelineTmaAsync.create()` 还明确执行：

```text
warp 0 mbarrier.init
-> mbarrier_init_fence
-> ThreadBlock sync
```

可作为修正 cache-dit 初始化协议的直接参考。

### 6.5 consumer release 没有显式 `__syncwarp()`

FlashInfer kernel 的 SVDQuant mainloop 在 `consumer_release()` 前没有 cache-dit 这样的显式 `__syncwarp()`；pipeline 内部仍只让每个 warp 的 signalling lane arrive。

这可能依赖 warp-collective MMA、DSL 控制流和生成代码的收敛保证。不能据此直接删除 CUDA C++ 中的 warp 同步。建议移植顺序：

1. 先移植 K-block 流水和提前 release，保留必要的同步。
2. 再单独检查 C++/PTX/SASS 中 lane 0 的 empty arrive 是否可能越过其他 lane 的 copy/MMA。
3. 只有在有明确收敛与内存序证据时，才考虑删除 `__syncwarp()`。

### 6.6 SVDQuant ring 一致，但 alpha/bias 契约不同

FlashInfer SVDQuant 与 cache-dit 都采用：

- BK=128 对应 rank tile 32，BK=64 对应 rank tile 16。
- D/L1 分别 alias 到 mainloop A/B smem。
- 继续使用 mainloop 的 stage/index/phase。
- correction TMA 重新使用精确 A+B tx count，不包含 SF。
- LoRA-up 使用 BF16 MMA，加入同一 FP32 accumulator。

因此无需因 FlashInfer 而重新设计 SVDQuant ring。但两侧的缩放时机和输入预处理不同：

| 项目 | cache-dit | FlashInfer SM120 |
|---|---|---|
| 输入 L1 | 原始 LoRA-up 权重 | 调用方必须预先传入 `L1 / alpha` |
| alpha | correction 前只乘 NVFP4 accumulator | correction 后在 epilogue 统一乘 residual + correction |
| bias | correction 前加入 accumulator | correction 后在 output epilogue 加入 |
| per-column wcscales | correction 前只作用于 NVFP4 accumulator | 当前公开 SVDQuant 路径没有对应处理 |

FlashInfer 的等价关系是：

```text
alpha * (P + D @ (L1 / alpha)^T) + bias
  = alpha * P + D @ L1^T + bias
```

cache-dit 则直接计算：

```text
(alpha * P) * wcscales + bias + D @ L1^T
```

两种契约在各自输入预处理下成立，但不能交叉套用。把 FlashInfer 的 epilogue 调度搬到 cache-dit 时，必须保留 wcscales 只作用于 NVFP4 主积、且不缩放 LoRA 项的约束。

### 6.7 epilogue 与 output subtile

FlashInfer：

- mainloop `sAB` 与 output `sC` 使用独立 smem 区域。
- output 按 `epi_tile` 分块。
- 写 `sC` 前先通过 store pipeline acquire + NamedBarrier 等待上一轮 TMA store 不再读取该 epi buffer。
- R2S/STSM 后执行 proxy fence + 第二次 NamedBarrier，再发射 TMA store。
- store pipeline 通常等待较早的 store group，而不是每个 subtile 都 `wait<0>`。
- 在 8 MMA warp 配置下，NamedBarrier 包含 256 个 MMA 线程，不包含 DMA warp。

因此独立 `sC` 并没有消除 epilogue barrier，而是把保护对象从 cache-dit 的“mainloop smem overlay 仍在被读”改为“epi buffer 可能仍被上一笔 TMA store 读取”。output 写后的 fence + barrier 在两者中都不能省略。

独立 `sC` 的代价是额外 output smem。cache-dit 默认 mainloop 已约 81 KiB，不能直接再增加 64 KiB 完整 output tile。对 cache-dit 更现实的方向是保留 overlay，同时改成较小 output subtile，使 stage 数、临时寄存器、barrier 和 store pipeline 能共同调整。该改造复杂度高于 mainloop，应后置。

### 6.8 persistent scheduler

FlashInfer 默认使用 static persistent scheduler，DMA 和 MMA warp 跨 output tile 工作，能够摊薄部分初始化和 descriptor/setup 开销。

cache-dit 若后续引入 persistent scheduler，必须明确：

- mainloop producer/consumer state 是否跨 tile 连续或如何重建；
- LoRA 是否继续使用同一 ring；
- output overlay 是否必须等上一 tile store 的 smem read 完成；
- producer tail 是否只在所有 tile 完成后执行；
- 不重新初始化正在使用的 mbarrier。

不能用 persistent CTA 掩盖当前 steady-state mainloop 的保守调度。

## 7. 综合建议与优先级

### P0：正确性协议

1. mbarrier 初始化改为唯一物理线程。
2. 增加 `fence_barrier_init()`，保持正确的 CTA sync。
3. 修正后先跑 tiny/bit-exact 和现有 FP4 gate，再继续性能优化。

### P1：steady-state mainloop

1. 参考 FlashInfer 实现 K-block copy/MMA 软件流水。
2. 在旧 stage 最后一次 smem 读取完成后提前 release，而不是等整 stage MMA 全部完成。
3. 保留 SFA/SFB 全 stage fragment 的寄存器复用约束。
4. 保持 BK=128/BK=64 和 LoRA rank tile 语义不变。
5. 审查并移除纯 TMA load 路径中冗余的逐 stage proxy fence。

### P2：producer 与 prologue

1. 使用 producer start state，消除初次 empty arrive。
2. 尽早分离 producer/consumer，consumer 初始化放到寄存器扩配后。
3. 评估 288 线程 CTA：8 MMA warp + 1 DMA warp。
4. stage/phase 改为增量回绕，提升 TMA partition，谨慎评估 outer-loop unroll。

### P3：epilogue 与结构性改造

1. 先把 output TMA store commit/wait 限定在实际发射线程。
2. 根据 smem 预算研究 output subtile 和 store pipeline。
3. 最后再评估 persistent scheduler、独立 `sC` 或独立 consumer warp group。

## 8. 后续实施时的验证门

本轮只做静态分析，后续有 CUDA 环境时至少验证：

1. **正确性**：barrier 初始化、无 hang、无 race；tiny tests、native bit-exact、rank=32/128、BK=64 fallback。
2. **alpha/LoRA 契约**：使用 `alpha != 1` 的输入，确认 cache-dit 的 alpha/wcscales 不缩放 LoRA 项；若对照 FlashInfer，必须记住其调用方传入的是 `L1 / alpha`，不能重复预除或漏除。
3. **SASS**：实际 copy/MMA 顺序、寄存器数、local spills、`setmaxnreg` 或替代方案、empty arrive 相对 copy/MMA 的位置。
4. **kernel-only 性能**：固定 SM120、M=4096/N=K=3072/rank=32，与 native 和修改前 CuTe 同口径比较。
5. **独立 GEMM 之外**：高层 SVDQW4A4Linear、完整 producer→GEMM→LoRA 链路和 FLUX e2e。
6. **一次只改一个主假设**：不能把 barrier 修正、K-block 流水、288 线程和 persistent scheduler 混在一次性能归因中。

交付目标仍以 `./rfc-cuda-optimizations.md` 为准：独立 GEMM 与完整 e2e 都需要验收，不能因局部 kernel 改善就将整个 CuTe 工作流标为完成。

## 9. 详细代码参考地图

以下路径均以工作区根为基准，全部使用相对路径。行号只对应 2026-11-04 本次调研快照；代码变化后应优先按函数、类和关键语句重新定位。

### 9.1 cache-dit：NVFP4 CuTe GEMM 主文件

主文件：`cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh`

| 行号 | 符号/区域 | 阅读重点 |
|---|---|---|
| 33–53 | CuTe 术语和 Stage 1–9 阅读指南 | gmem/smem/RF、fragment、layout、partition、mbarrier 的含义 |
| 55–79 | 数学与数据流契约 | `P=A@B^T`、wcscales/bias、D/L1、单 FP32 accumulator |
| 309–457 | `W4A4CuteTraits` | BM/BN/BK、stages、rank tile、MMA、smem layout、D/L1 overlay |
| 432–456 | stage bytes 与 tx bytes | A/B/SFA/SFB 的物理字节、D/L1 overlay、main/LoRA expect_tx |
| 442–448 | static assertions | D=A、L1=B 的 overlay 字节等式、output 容量、99 KiB 上限 |
| 503–543 | `gemm_w4a4_kernel` 签名、线程角色 | 384 线程、producer/consumer 的 `raw_tid/tid` 映射 |
| 546–558 | Stage 1 | TMA descriptors、gmem tensor 和按 CTA tile 的局部视图 |
| 561–603 | Stage 2 前半 | `extern __shared__`、stage 字节偏移、D/L1 overlay 地址 |
| 588–596 | Stage 2 后半 | `full/empty` 初始化、重复初始化风险、初始化后的 CTA sync |
| 598–638 | Stage 3 | smem→RF copy view、MMA fragment、D/L1 fragment、FP32 accumulator |
| 661–688 | `issue_tma` | main stage 四路 TMA、LoRA 两路 TMA、精确 tx bytes、producer proxy fence |
| 691–714 | Stage 5 | producer 主循环、stage/phase、consumer 初次 empty arrive |
| 716–742 | Stage 6 | NVFP4 consumer 的 wait、四次 smem→RF copy、整 stage MMA、syncwarp、release |
| 744–764 | Stage 7 | alpha、wcscales、bias；为什么必须在 LoRA accumulation 前完成 |
| 766–809 | Stage 8 | LoRA consumer 复用同一 stage/phase、D/L1 copy、BF16 MMA、syncwarp、release |
| 811–836 | Stage 9 | output overlay、两次 `__syncthreads()`、STSM、proxy fence、TMA store |
| 852–887 | `launch_gemm_w4a4` 的 gmem views | A/B/SFA/SFB/C/D/L1 tensor 和 rank=0 placeholder |
| 889–911 | `build_params` | 七个 `make_tma_copy` descriptor 的统一构造点 |
| 916–927 | host descriptor cache | cache key 和 cache 生命周期 |
| 929–938 | kernel 配置与 launch | dynamic smem 属性、384 线程和 CTA grid |

TMA 主分支的关键阅读顺序：

```text
W4A4CuteTraits 的字节/layout 契约
-> build_params 构造 descriptors
-> Stage 1 gmem tile 坐标
-> Stage 2 smem stage 地址
-> issue_tma 的 expect_tx/copy
-> Stage 5 producer_acquire 等价逻辑
-> Stage 6/8 consumer_wait/release 等价逻辑
-> Stage 9 output store
```

### 9.2 cache-dit：proxy fence 与 native 性能参照

| 相对路径 | 行号 | 内容 |
|---|---:|---|
| `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/barrier.h` | 726–738 | `fence_view_async_shared()` 实际对应 `fence.proxy.async.shared::cta` |
| `cache-dit/csrc/kernels/svdq/native/gemm_w4a4.cuh` | 15–42 | `GEMM_W4A4` 和不支持 FP4 时的 `trap_no_fp4()` 兜底 |
| 同上 | 64–88 | `load_amscale/load_wmscale` 的 packed scale 读取 |
| 同上 | 249–340 | `gemm_w4a4_fp4_block` 的 native stage 预取与主循环入口 |
| `cache-dit/csrc/kernels/svdq/native/gemm_w4a4_impl.cuh` | 28–64 | native packed FP4 到当前 CuTe 布局转换的性能负结果背景 |

本文写作时该目录名拼写有误；已在后续提交中统一更正为 `native/` 并同步全部 include、构建和文档引用，本文引用路径已按新名更新。native 实现不含这里的 TMA smem pipeline，因此只适合作为性能、数值和数据布局参照，不适合作为 mbarrier 协议参考。

### 9.3 CUTLASS：SM120 block-scaled C++ mainloop

主文件：`cache-dit/csrc/third_party/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp`

| 行号 | 符号/区域 | 与 cache-dit 的关系 |
|---|---|---|
| 266–278 | `SharedStorage`、pipeline storage | full/empty barrier storage 的实际位置 |
| 420–428 | TMA descriptor prefetch | cache-dit 当前未做 device-side descriptor prefetch；属于次要启动优化 |
| 617–661 | producer gmem/smem tensors 与 TMA partition | 在循环前构造 gA/gB/gSFA/gSFB 和 smem views |
| 662–709 | producer `for k_tile` | `producer_acquire`、barrier 指针、四路 TMA、state advance |
| 723–805 | consumer MMA/copy views | consumer 进入 mainloop 前的 fragments 和 SF copy views |
| 818–825 | `K_BLOCK_MAX` 和首 stage views | 确定 k-block 数量及当前读 stage |
| 827–841 | `copy_kblock` lambda | A/B/SFA/SFB 按 k-block 拷入 RF 的核心参考 |
| 843–846 | `gemm_kblock` lambda | 当前 k-block 的 zipped A/SFA、B/SFB MMA |
| 848–850 | consumer prologue | 首个 stage wait、kblock 0 copy |
| 851–878 | steady-state mainloop | NamedBarrier、提前 release、next k-block copy、当前 k-block MMA |
| 880–900 | last k-tile 特化 | 不再预取下一 tile 时的尾部处理 |

CUTLASS C++ 参考在 stage 边界先 copy 下一 stage 的 kblock 0，再执行当前最后一个 k-block 的 MMA。对 BK=128 可复用 kblock 0/1 两个寄存器槽位；对 BK=64 不能直接机械照搬，否则可能覆盖尚未执行 MMA 的唯一 A/B fragment。FlashInfer 的“先当前 MMA、再 next copy”顺序可解决这一差异。

### 9.4 CUTLASS：pipeline、mbarrier、寄存器与 TMA store 原语

以下均指向 cache-dit 默认构建实际使用的 vendored CUTLASS，而不是工作区中另一份 CUTLASS checkout。

| 相对路径 | 行号 | 符号/内容 |
|---|---:|---|
| `cache-dit/csrc/third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp` | 228–250 | `PipelineState::advance()`：index 回绕、phase 翻转、count 递增 |
| 同上 | 252–260 | `make_producer_start_state()`：初始 stage=0、phase=1 |
| 同上 | 301–324 | `init_barriers()`：initializing warp 初始化和 `fence_barrier_init()` |
| 同上 | 419–443 | `producer_acquire/producer_commit` overloads |
| 同上 | 445–454 | `producer_tail()`：等待所有 stage 的 empty phase |
| 同上 | 466–540 | consumer try/wait/release API 区域 |
| `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/barrier.h` | 391–432 | `ClusterBarrier::init/wait` |
| 同上 | 507–523 | `ClusterBarrier` 的 local smem arrive |
| 同上 | 546–602 | `ClusterTransactionBarrier` 和 `arrive_and_expect_tx` |
| 同上 | 709–722 | `fence_barrier_init()` 的 PTX 与所需 scope sync 注释 |
| 同上 | 726–738 | `fence_view_async_shared()` |
| `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/reg_reconfig.h` | 72–86 | `warpgroup_reg_alloc/dealloc` 的 `setmaxnreg.inc/dec.sync.aligned` 包装 |
| `cache-dit/csrc/third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp` | 1212–1221 | `tma_store_fence()` |
| 同上 | 1223–1233 | `tma_store_arrive()` 对应 `cp.async.bulk.commit_group` |
| 同上 | 1245–1259 | `tma_store_wait<N>()` 对应 `.wait_group.read N` |
| `cache-dit/csrc/third_party/cutlass/include/cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp` | 713–760 | `issue_tma_store` 限定发射 warp，commit/acquire 也位于该条件下 |

CUTLASS 参考 epilogue 的 TMA store copy、commit 和 reuse-acquire 都由 `issue_tma_store` 条件保护。因此 cache-dit 可研究将 commit/wait 限定到实际发射线程，但这仍是尾部小优化，收益需与主循环严格分开测量。

### 9.5 FlashInfer：公开 SVDQuant wrapper 与架构分流

文件：`flashinfer/flashinfer/gemm/gemm_svdquant.py`

| 行号 | 内容 | 阅读重点 |
|---|---|---|
| 90–106 | `_sm120_svdquant_kernel_name` | rank、bias、tile、swap_ab、cluster、PDL 等 codegen cache key |
| 109–143 | `_compile_sm120_nvfp4_svdquant` | SM120 fused kernel 的编译与缓存入口 |
| 256–314 | `_mm_nvfp4_svdquant_sm120_fused` | SM120 fused 参数和输入准备 |
| 317–355 | `_mm_nvfp4_svdquant_sm120_unfused` | compositional oracle/unfused 对照路径 |
| 358–408 | SM120 runner 与 tactics | runner 选择和有效 tactic 生成 |
| 750–774 | `mm_nvfp4_svdquant` 公共契约 | 输出公式和调用方必须把 `1/alpha` 折入 L1 |
| 827–860 | backend 分流 | `cute-dsl` fused/unfused 与 auto tuning 路径 |
| 867–879 | CUTLASS runner 区域 | SM100/SM103 CUTLASS 后端，不属于 SM120 DSL kernel |

### 9.6 FlashInfer：SM120 CuTe DSL kernel 主体

主文件：`flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`

| 行号 | 符号/区域 | 阅读重点 |
|---|---|---|
| 230–289 | constructor / scheduler 选项 | `single_work_tile_per_cta`、persistent scheduler 和当前 cluster 约束 |
| 290–334 | atom shape、warp/register 配置 | 16/32/64/128 行 atom 对应 2/4/8 个 MMA warp，再加 1 个 DMA warp |
| 314–378 | SM120 tiled MMA | `m16n8k64`、permutation、M/N/K tile 数 |
| 380–406 | SF layout 与 stage 计算 | SFA/SFB smem layout 和 `_compute_stages()` 调用 |
| 412–453 | SVDQuant 配置 | BK→rank tile、BF16 MMA、D/L1 smem alias mainloop A/B |
| 548–581 | mainloop TMA atoms | A/B/SFA/SFB descriptor 和 tensor 构造 |
| 1048–1080 | kernel 参数与 descriptor prefetch | alpha、SVDQuant 参数和 A/B/SF/C/D/L1 descriptor prefetch |
| 1091–1112 | copy bytes | mainloop A+B+SF 与 correction A+B 字节等式 |
| 1114–1167 | smem/pipeline/storage views | smem allocator、mainloop pipeline、sA/sB/sC/sSFA/sSFB |
| 1184–1223 | TMA producer partition | A/B/SFA/SFB 的 gmem/smem TMA partition |
| 1376–1398 | consumer fragments | A/B/SF fragment 和 FP32 accumulator |
| 1400–1405 | CTA/cluster sync | pipeline 创建后、角色循环前的同步 |
| 1418–1439 | tile scheduler 与 pipeline states | static persistent scheduler、producer/consumer start state |
| 1441–1444 | MMA 分支入口 | 8 MMA warp 配置下寄存器增加到 232 |
| 1516–1524 | `PipelineTmaStore.create` | output subtile store pipeline 的实际创建点 |
| 1544–1583 | SF copy views | full-stage SF fragment 的 copy view 组织 |
| 1634–1697 | mainloop prologue | accumulator 清零、首次 full wait、A/B k0 copy、全 stage SF copy |
| 1699–1791 | steady-state K-block mainloop | 提前 release、当前 MMA、next A/B copy、跨 stage 后 next SF copy |
| 1793–1833 | last k-tile | 最后一个 K tile 的独立尾部处理 |
| 1835–1881 | SVDQuant consumer | `L1/alpha` 契约、同一 pipeline 上的 D/L1 BF16 MMA 和 release |
| 1930–2000 | epilogue copy/TMA partition | STSM/R2S、accumulator retile、output epi tile |
| 2090–2165 | output subtile loop | output 分块、alpha/bias epilogue 和格式转换 |
| 2166–2195 | alpha、epi acquire、两次 NamedBarrier、proxy fence | 保护 epi buffer 轮转，并等待 STSM 完成后发射 store |
| 2241–2261 | store commit/tail 与 tile advance | TMA store pipeline 的收尾和 persistent consumer |
| 2263–2265 | DMA 分支入口 | DMA warp 的寄存器降为 40 |
| 2295–2555 | mainloop TMA producer | producer acquire、四路 TMA、state advance；纯 TMA 路径无逐 stage proxy fence |
| 2458、2530 | cp.async/手工 smem 写 fence | 非纯 TMA 写入路径的 proxy fence |
| 2559–2611 | SVDQuant TMA producer | 同一 mainloop ring、D/L1 alias、精确 A+B tx count |
| 2615–2624 | producer 下一 tile 与 tail | persistent producer advance，全部 tile 后 `producer_tail()` |
| 2631–2758 | `_compute_stages()` | mainloop/epilogue stage 与 smem capacity 的联合计算 |

### 9.7 FlashInfer/CuTe DSL：pipeline 内部实现

| 相对路径 | 行号 | 符号/内容 |
|---|---:|---|
| `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py` | 147–188 | `MbarrierArray`：producer/consumer group 和 arrive count |
| 同上 | 206–225 | `mbarrier_init()`：warp 0 初始化 barrier array |
| 同上 | 555–635 | `PipelineState`：index/count/phase 以及 advance/reverse |
| 同上 | 651–668 | `make_pipeline_state()`：producer phase=1、consumer phase=0 |
| `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py` | 153–212 | 通用 async pipeline 创建和 init fence/CTA sync |
| 同上 | 377–431 | `init_empty_barrier_arrive_signal()`：每 warp signalling lane 和 cluster mask |
| 同上 | 433–516 | `PipelineTmaAsync.create()`：full/empty、tx count、init fence、CTA/cluster sync |
| 同上 | 519–539 | TMA producer acquire：empty wait + full arrive/expect_tx |
| 同上 | 547–557 | consumer release：signalling lane 条件 arrive；kernel 侧没有显式 syncwarp |

这些内部实现说明：FlashInfer 的简洁 kernel 代码并不代表没有同步，而是同步被封装在 pipeline create/wait/release 中。比较 cache-dit 时必须同时读 DSL library，不能只读 `dense_blockscaled_gemm_sm120_b12x.py`。

### 9.8 FlashInfer SM100：只用于确认架构边界

| 相对路径 | 行号 | 内容 |
|---|---:|---|
| `flashinfer/csrc/nvfp4_svdquant_gemm_cutlass_sm100.cu` | 39–77 | SM100/SM103、2SM/1SM tile tag 选择 |
| 同上 | 79–148 | `svdquant_gemm_runner`、arguments、workspace 和 grid 配置 |
| 同上 | 149–245 | tile config dispatch 和 kernel launch |
| `flashinfer/include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm100.h` | 54–109 | SM100 类型、tile、TMEM stage、cluster 和 smem 计算 |
| 同上 | 111–179 | `GemmUniversalAdapterLoRA` / `NVFP4SVDQuantGemmSm100Kernel` 组织 |
| `flashinfer/include/flashinfer/gemm/nvfp4_svdquant_gemm_collective_sm100.h` | 57–66 | 明确说明这是 SM100 block-scaled warp-specialized LoRA-fused mainloop |
| 同上 | 79–129 | SM100 block-scaled layout/atom traits |
| 同上 | 130–220 | Arguments、Params、SharedStorage 和 pipeline storage |

这些代码使用 SM100 `tcgen05`/UMMA/TMEM。可以研究其高层 fused API 和参数组织，但不能把它的 MMA、accumulator 或 barrier 调度直接套到 SM120。

## 10. 按问题交叉对照的具体代码位置

### 10.1 mbarrier 初始化与 init fence

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:537–539,588–596` | 两个 `tid==0`；只有 CTA sync |
| CUTLASS | `cache-dit/csrc/third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:301–324` | 单 warp init + `fence_barrier_init()` |
| CUTLASS 原语 | `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/barrier.h:709–722` | `fence.mbarrier_init.release.cluster` 及 scope sync 注释 |
| FlashInfer kernel | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1114–1152,1400–1405` | pipeline create 封装初始化，之后还有角色循环前同步 |
| FlashInfer DSL | `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:206–225` | warp 0 初始化 |
| FlashInfer DSL | `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:502–507` | `mbarrier_init_fence()` + CTA/cluster sync |

### 10.2 初次 empty release 与 producer start state

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:691–714` | consumer 先 arrive 所有 empty，producer 才能开始 |
| CUTLASS | `cache-dit/csrc/third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:252–260` | producer 初始 phase=1 |
| FlashInfer | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1433–1439` | 分别构造 producer/consumer state |
| FlashInfer DSL | `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:651–668` | producer/consumer 初始相位定义 |

### 10.3 整 stage 消费与 K-block 软件流水

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:716–742` | 整 stage copy→整 stage MMA→release |
| CUTLASS prologue | `cache-dit/csrc/third_party/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:827–850` | `copy_kblock` 和首 stage 预热 |
| CUTLASS steady | 同上 `:851–878` | copy/MMA 流水、提前 release |
| CUTLASS tail | 同上 `:880–900` | 最后一个 K tile 不再预取 |
| FlashInfer prologue | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1634–1697` | A/B k0 + 全 stage SF |
| FlashInfer steady | 同上 `:1699–1791` | 当前 MMA→next A/B copy；跨 stage 后 next SF copy |
| FlashInfer tail | 同上 `:1793–1833` | last k-tile 独立处理 |

### 10.4 consumer release 与 warp 同步

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit NVFP4 | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:724–740` | `__syncwarp()` 后 warp leader arrive |
| cache-dit LoRA | 同上 `:793–806` | 同样的 D/L1 stage 保护 |
| CUTLASS | `cache-dit/csrc/third_party/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:851–878` | NamedBarrier 与 pipeline release 的参考位置 |
| FlashInfer kernel | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1699–1707,1793–1801` | release 前无显式 `__syncwarp()` |
| FlashInfer DSL | `flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:377–431,547–557` | signalling lane 选择和条件 arrive |

### 10.5 SVDQuant D/L1 复用 mainloop ring

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit producer | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:661–705` | LoRA 走同一 `g/stage/phase`，tx count 为 D+L1 |
| cache-dit consumer | 同上 `:744–809` | alpha/wcscales/bias 后消费原始 L1，保持单 accumulator |
| FlashInfer 配置 | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:412–453` | rank tile 和 D/L1 alias mainloop A/B |
| FlashInfer consumer | 同上 `:1835–1881` | 输入为 L1/alpha；correction 先 accumulation，alpha/bias 留给 epilogue |
| FlashInfer producer | 同上 `:2559–2611` | 同一 producer state、精确 `ab_copy_bytes` |

### 10.6 proxy fence 的必要与可能冗余位置

| 场景 | 相对路径与行号 | 判断依据 |
|---|---|---|
| cache-dit TMA load | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:661–664,729,798` | 逐 stage load-side fence，CUTLASS/FlashInfer 纯 TMA 路径没有对应操作 |
| cache-dit output | 同上 `:829–834` | STSM generic 写后、TMA store 前，必须保留相应 proxy ordering |
| CUTLASS fence 定义 | `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/barrier.h:726–738` | 确认实际 PTX |
| FlashInfer 非 TMA 写 | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:2458,2530` | cp.async/手工 smem 写后的 proxy fence |
| FlashInfer output | 同上 `:2176–2195` | epi buffer acquire/barrier、STSM、fence、第二次 barrier |
| FlashInfer 纯 TMA producer | 同上 `:2483–2555` | 四路 TMA 直接发射，无逐 stage proxy fence |

### 10.7 output overlay 与 subtile epilogue

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:811–836` | 完整 output tile overlay、两个 CTA barrier、单次大 TMA store |
| cache-dit 容量约束 | 同上 `:432–448` | stage bytes、output overlay 和 99 KiB static assertions |
| FlashInfer smem | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1114–1167` | smem allocator 和独立 `sC` view |
| FlashInfer store pipeline | 同上 `:1516–1524` | `PipelineTmaStore` 创建 |
| FlashInfer subtile loop | 同上 `:1930–2000,2090–2195` | output 分块、两次 MMA-only NamedBarrier、TMA store |
| CUTLASS store 原语 | `cache-dit/csrc/third_party/cutlass/include/cute/arch/copy_sm90_tma.hpp:1212–1259` | store fence、commit group 和 wait-group.read |

### 10.8 384 线程与 288 线程 CTA

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| cache-dit | `cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:503–543,691–707,929–938` | 384 launch、128 producer、实际单线程发射 TMA |
| cache-dit 寄存器原语 | `cache-dit/csrc/third_party/cutlass/include/cutlass/arch/reg_reconfig.h:72–86` | 当前 `setmaxnreg.inc/dec.sync.aligned` 包装 |
| FlashInfer | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:290–334` | 2/4/8 MMA warp + 1 DMA warp；8 MMA warp 时才为 288 |
| FlashInfer MMA/DMA 分支 | 同上 `:1441–1444,2263–2265` | MMA 232 registers、DMA 40 registers |

### 10.9 persistent scheduler 与 producer tail

| 实现 | 相对路径与行号 | 观察点 |
|---|---|---|
| FlashInfer scheduler | `flashinfer/flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py:1418–1439` | static persistent scheduler 创建 |
| FlashInfer consumer | 同上 `:2241–2261` | store 收尾、consumer 跨 output tile advance |
| FlashInfer producer | 同上 `:2615–2624` | producer 跨 tile advance，最终 `producer_tail()` |
| CUTLASS producer tail | `cache-dit/csrc/third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:445–454` | producer 尾部 empty wait 的通用语义 |

## 11. 推荐的逐段对照阅读顺序

为了后续实施时避免在大量 CuTe 类型代码中迷失，建议按以下顺序读，而不是从三个文件第一页顺序通读：

1. **先读 cache-dit 的角色和 stage 环**：`cache-dit/csrc/kernels/svdq/cute/fp4/gemm_w4a4.cuh:503–543,561–596,661–714`。
2. **再读 CUTLASS 的标准 producer start state 和 init fence**：`cache-dit/csrc/third_party/cutlass/include/cutlass/pipeline/sm90_pipeline.hpp:252–260,301–324`。
3. **并排读 cache-dit 与 FlashInfer 的 consumer 主循环**：cache-dit `gemm_w4a4.cuh:716–742`；FlashInfer `dense_blockscaled_gemm_sm120_b12x.py:1634–1833`。
4. **用 CUTLASS C++ mainloop 复核第三种调度顺序**：`cache-dit/csrc/third_party/cutlass/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:818–900`。特别比较 stage 边界处 copy-next 和 MMA-current 的先后，以及 BK=64 是否安全。
5. **核对 SVDQuant correction 是否始终使用同一 ring**：cache-dit `gemm_w4a4.cuh:661–705,744–809`；FlashInfer SM120 `dense_blockscaled_gemm_sm120_b12x.py:412–453,1835–1881,2559–2611`。
6. **核对两侧不同的 alpha 契约**：cache-dit `gemm_w4a4.cuh:55–67,744–809`；FlashInfer `gemm_svdquant.py:750–774` 和 `dense_blockscaled_gemm_sm120_b12x.py:1835–1881,2160–2174`。
7. **再读 DSL pipeline 内部，避免漏掉被封装的同步**：`flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/helpers.py:147–225,555–668`；`flashinfer/3rdparty/cutlass/python/CuTeDSL/cutlass/pipeline/sm90.py:377–557`。
8. **最后比较 epilogue 和 persistent scheduler**：cache-dit `gemm_w4a4.cuh:811–836`；FlashInfer SM120 `dense_blockscaled_gemm_sm120_b12x.py:1114–1167,1516–1524,1930–2000,2090–2195,2241–2261,2615–2624`。
9. **SM100 文件只在需要确认架构边界时阅读**：不把 SM100 TMEM/UMMA 的调度细节混入 SM120 优化假设。

## 12. 代码引用使用注意

- 上述行号是调研快照定位，不是永久 API；修改文件后应按符号重新搜索。
- `flashinfer/flashinfer/...` 的重复目录分别是仓库根和 Python package，不是笔误。
- `cache-dit/csrc/kernels/svdq/native/...` 目录名此前拼写有误，已在后续提交中统一更正为 `native` 并同步全部引用；本文引用路径均已按新名更新。
- FlashInfer 的 SM120 kernel 是 CuTe DSL Python 源，不代表 cache-dit 应引入 CuTe DSL 实现；这里只借鉴调度、barrier 和数据流，再用现有 CUDA/CuTe C++ 落地。
- 所有性能候选仍需按第 8 节验证门实测；代码结构相似不能直接推出收益幅度。
