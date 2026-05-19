# Triton 算子核优化指南

本文档系统性梳理 Triton kernel 的优化方法论，涵盖从 tile 尺寸调优到工程化诊断的八大类措施。每一类都从原理、手段、要点三个角度展开，作为编写和调优 Triton 算子时的参考手册。

---

## Block 与 Tile 尺寸调优

### 原理

Triton 的编程模型是"每个 program 处理一个 tile"。tile 的尺寸（如 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`）是最核心的性能参数，它直接决定了四件事：

- **并行度**：grid 大小等于问题规模除以 block 大小。block 越大则 grid 越小，SM 占用率可能下降；block 越小则 program 数目越多，launch 和调度开销增加。
- **寄存器压力**：block 内的中间结果驻留在寄存器中，block 过大会导致寄存器 spill 到 local memory（实际位于显存），引发严重性能下降。
- **访存效率**：tile 太小时，每次 load 搬运的数据量不足以饱和显存带宽；tile 太大时，L1/L2 cache 命中率又会下降。
- **算术强度**：对矩阵乘法而言，tile 越大数据复用率越高，算术强度（FLOPs/Bytes）越高，越容易从 memory-bound 转为 compute-bound。

### 调优手段

- **使用 `triton.autotune`**：提供多组候选 `triton.Config`，由 Triton 在首次运行时自动 benchmark 选出最优解，并通过 `key` 参数按输入形状分桶缓存。
- **分形状区间调优**：大矩阵偏好大 tile（如 128×256），小矩阵或 batched small GEMM 偏好小 tile（如 32×32），不能一套配置打天下。
- **联动 `num_warps`**：常用值为 4 或 8，决定每个 program 拥有的线程数，从而影响每线程寄存器数量与并行粒度。
- **联动 `num_stages`**：决定软件流水线深度，一般取 2～5，受限于 shared memory 容量。

### 要点

- Tile 尺寸一般取 2 的幂（16、32、64、128、256），以契合 warp（32 线程）和硬件向量化指令。
- 观察 `ncu` 输出的寄存器使用量和 local memory 使用量，发现 spill 时应立即减小 tile 或增加 `num_warps`。
- Autotune 的候选集不是越多越好，集合太大会让首次运行极慢，应基于经验筛选 5～10 组常用配置。
- 不同硬件（A100、H100、消费级显卡）的最优配置差异显著，最好针对目标硬件单独调优。

---

## 内存访问优化

显存带宽通常是 kernel 的首要瓶颈，内存访问层面的优化往往带来最大的收益。

### 合并访问

GPU 显存控制器一次搬运一个连续的 128 字节 sector。若一个 warp 内 32 个线程访问的是连续地址，则能被合并成少量事务；若跨 stride 访问，则会产生冗余搬运和带宽浪费。

在 Triton 中保证合并访问的关键是：**让最内层维度在内存中连续**。加载二维 tile 时，应把 stride 为 1 的维度放在 `[None, :]` 位置，让同一 warp 内的线程沿着连续地址展开。

### 对齐与向量化提示

Triton 编译器会根据指针属性生成 `ld.global.v4`、`ld.global.v2` 等宽指令。可以通过以下方式帮助编译器识别对齐：

- **`tl.multiple_of(x, n)`**：告知编译器 `x` 是 `n` 的整数倍。
- **`tl.max_contiguous(x, n)`**：告知编译器 `x` 中连续元素的最大长度。
- **函数参数标注**：通过 `tl.constexpr` 约束某些 stride 为 1，可触发更优的代码生成路径。

这些提示在处理非规整形状时尤其重要，它们决定了编译器是否敢于使用 128-bit 宽 load。

### 掩码裁剪处理边界

对非整除的形状，统一使用 `mask=` 参数处理尾部，而不是写成 `if` 分支。好处是主循环路径保持规整，避免 warp divergence；同时通过 `other=` 为越界位置填充中性值（加法用 0，乘法用 1，max 用 -inf）。

### 共享内存复用

Triton 不需要手动管理 shared memory，但通过合理的 tile 形状和循环结构，编译器会自动把复用的数据放在 shared memory 中。例如矩阵乘法沿 K 维度循环时，A 和 B 的 tile 都会被缓存下来以便多次复用，从而提升算术强度。

### 避免 Bank Conflict

shared memory 被分成 32 个 bank，一个 warp 内多个线程同时访问同一 bank 不同地址会造成 serialization。避免 bank conflict 的手段：

- 使用经过验证的 tile 形状（如 matmul 教程里的 128×128×32）。
- 对转置类 kernel，避免 `BLOCK_M == BLOCK_N == 32` 的朴素写法，可通过 padding 到 33 或让编译器自动 swizzle 缓解。

### 减少冗余加载

- 把循环不变的 load（如 bias、scale 系数）提到循环外，避免每轮重复搬运。
- 把可复用的中间结果缓存在寄存器数组中，而不是反复从显存读取。
- 对需要多次访问的只读张量，考虑主动预取到 shared memory。

---

## 流水线与异步化

### 软件流水线（num_stages）

软件流水线的核心思想是：**在计算第 i 轮时，异步发起第 i+1 轮（甚至更远）的 load**，让访存延迟被计算掩盖。

`num_stages=N` 意味着编译器会维护 N 份 shared memory buffer 轮流使用。选择原则：

- **Ampere（A100）**：通常 3～4 级最优。
- **Hopper（H100）**：配合 TMA 可以做到 5～6 级。
- **过深的副作用**：shared memory 不够用，反而导致 occupancy 下降或 spill。

软件流水线只对循环体内有稳定 load-compute 模式的 kernel 生效，单次 kernel（如 elementwise）不需要设置。

### 异步拷贝与 TMA

在较新的硬件上，Triton 会自动生成异步拷贝指令：

- **Ampere 的 `cp.async`**：实现 global → shared 的异步搬运，不阻塞计算单元。
- **Hopper 的 TMA（Tensor Memory Accelerator）**：专门的硬件单元负责张量搬运，延迟更低、效率更高，还支持多维寻址。

用户侧需要做的是：

- 保证 tile 大小与对齐满足异步拷贝的要求（通常要求 4/8/16 字节对齐）。
- 使用最新版 Triton，让其识别到硬件并走最优 lowering 路径。

### Load / Compute / Store 顺序

手写代码的顺序会影响编译器生成的依赖图。一般建议：

1. 先发起所有需要的 load。
2. 对已到达的数据做计算。
3. 最后一次性 store 结果。

这样可以让编译器构造尽可能深的流水线，把访存和计算最大程度重叠。

---

## 计算层面的优化

### 使用 `tl.dot` 走 Tensor Core

矩阵乘的核心操作务必使用 `tl.dot`，它会被 lowering 到 MMA / WMMA / WGMMA 指令，吞吐是标量 FMA 的十倍以上。

要点：

- 形状需要满足硬件 MMA 的最小粒度（通常 16×16×16）。
- 累加器一般保持 fp32，以避免误差累积导致的精度损失。
- 应严格沿用上层算子规定的数据类型，不得擅自降精度；是否能使用 fp16/bf16/fp8 是框架或模型设计层面的决策，kernel 作者不应越权。

### 算子融合

融合是 Triton 相对调库方案的最大优势之一。把多个逐元素或归约操作合并到一个 kernel 中，可以大幅减少显存往返次数。典型例子：

- **Fused LayerNorm / RMSNorm**：一个 kernel 完成均值、方差、归一化、affine。
- **Fused Softmax**：一个 kernel 完成 max、exp、sum、除法。
- **Fused Attention（FlashAttention）**：将 QK^T、softmax、×V 全部融合，配合 online softmax 避免物化 N×N 的注意力矩阵。
- **Matmul + Epilogue**：把 bias、激活、dropout、residual add 直接接在 `tl.dot` 之后完成。

融合的代价是 kernel 复杂度上升、寄存器压力增加，需要在调度层面做权衡。

### 指令级技巧

- **`tl.exp2` 代替 `tl.exp`**：硬件对 `exp2` 有专用快速指令，softmax 中可预先将 `x/ln 2` 合并到上游计算。
- **`rsqrt` 代替 `1 / sqrt`**：LayerNorm、RMSNorm 中常用，少一次除法。
- **显式使用 FMA**：编译器通常会自动融合乘加，但在复杂表达式里显式写更稳。
- **避免整数除法和取模**：改用位运算或预先计算好的 stride，尤其是在内层循环中。

### 数值稳定性与性能结合

- **Softmax 减最大值**：保持数值稳定，同时允许使用 `exp2` 等快路径。
- **Welford 单趟统计**：LayerNorm 用 Welford 算法一趟计算均值方差，省一次扫描。
- **Online Softmax**：FlashAttention 的关键思想，用递推式在一次扫描内完成 softmax，避免保存中间矩阵。

---

## 并行与 Grid 策略

### Grid 划分

Grid 决定了 program 总数以及它们与数据块的映射关系。常见选择：

- **1D grid**：适合 reduction、elementwise。
- **2D grid**：适合 matmul，两个维度分别对应 M 和 N。
- **1D + 内部解码**：把 `(pid_m, pid_n)` 编码成单个 `pid`，便于实现 swizzle。

原则是：grid 总数至少要让所有 SM 吃饱，一般 grid 数 ≥ SM 数 × 2～4 以提供调度余量。

### Swizzle 提升 L2 命中率

在大矩阵 matmul 中，线性的 `(pid_m, pid_n)` 顺序会让多个 program 同时访问 B 的不同列，L2 cache 命中率下降。采用 group-major swizzle 的思路是让相邻的 program 集中在一个 L 形区域内，共享对 A 和 B 的访问，从而显著提升 L2 命中率。这是 Triton matmul 教程里的经典技巧，在大尺寸 GEMM 中常能带来 10%～30% 的加速。

### Persistent Kernel

当 grid 数远大于 SM 数时，每个 program 的 launch 和上下文切换都是开销。Persistent kernel 的做法是让 program 数等于 SM 数（或其整数倍），在 kernel 内部循环处理多个 tile。优势：

- 减少 launch 开销和调度抖动。
- 跨 tile 复用寄存器中的常量和预加载数据。
- 在 H100 上配合 warp specialization 可进一步拉开差距。

### Split-K

当 K 很大而 M、N 较小时，正常 grid 只有几十个 program，无法占满 SM。Split-K 的做法是把 K 维再切分为 S 份，由 S 个 program 并行计算部分和，最后通过原子加或二级 reduction kernel 合并结果。

代价是多次原子加或额外 kernel 开销，但对"小 M/N 大 K"的形状（如某些 attention 投影、embedding 反传）通常非常值得。

### 负载均衡

- 对变长输入（如 NLP 的 batch），用 `cu_seqlens` 记录每个样本的偏移，每个 program 处理一个 token block，避免 padding 带来的无效计算。
- 对稀疏或不规则的问题，使用分桶或 bin-packing 策略，让每个 program 的工作量尽量接近，避免"长尾 program"拖慢整个 kernel。

---

## 寄存器与占用率管理

### 寄存器压力

每个 SM 的寄存器总数固定（例如 A100 有 64K 个 32-bit 寄存器）。单线程使用的寄存器越多，能同时驻留的 warp 就越少，occupancy 随之下降。

**诊断手段**：

- 使用 `ncu --set full` 查看 `registers per thread` 和 `achieved occupancy`。
- 观察 local memory 使用量，非 0 通常意味着 spill。
- 设置 `TRITON_DEBUG=1` 或查看 PTX 输出确认寄存器分配。

**缓解手段**：

- 减小 `BLOCK_M / BLOCK_N` 等 tile 维度。
- 减小 `num_stages`（流水线 buffer 也会占用寄存器）。
- 增大 `num_warps`，把工作分摊到更多线程。
- 拆分 kernel，把不相关的计算分开。
- 避免持有过多不必要的中间张量。

### Occupancy 与延迟隐藏

Occupancy 不是越高越好。对计算密集型算子（如 matmul），低 occupancy + 大 tile 反而更快，因为流水线和 Tensor Core 已经把延迟吃完；对访存密集型算子，则需要高 occupancy 来切换 warp 隐藏访存延迟。

经验法则：

- **Matmul / Attention**：目标 occupancy 25%～50%，tile 尽量大。
- **Elementwise / Reduction**：目标 occupancy 50%～100%，追求访存并行度。

### 循环不变量外提与强度削减

把循环不变的计算（如归一化系数、常量 reciprocal）提到循环外，减少内层循环的指令数。编译器通常会自动优化，但复杂表达式中显式提取更保险。

---

## 特殊数据路径

本章讨论的是在**不改变原始数据类型**的前提下，利用硬件特殊通路获得加速的手段。能否降精度（fp32 → fp16/bf16/fp8）属于模型和框架层面的决策，kernel 作者应当严格沿用上层传入的 dtype，不得擅自改动。

### 结构化稀疏与块稀疏

- **2:4 稀疏**：Ampere 起支持，每 4 个权重中 2 个为 0，可获得约 2x 吞吐，需要 `cusparseLt` 或自定义 Triton kernel 配合。
- **Block-sparse**：按 block 稀疏存储（如 blocksparse attention），通过 index 数组跳过空 block。Triton 写块稀疏 kernel 相对容易，只需在外层循环中按 index 跳转。

### Masked 与变长路径

对需要 mask 的场景（attention、padding），避免使用 `if` 分支造成 warp divergence，改用 `tl.where` 和带 mask 的 load/store。对因果 mask 可以直接跳过整个被完全 mask 掉的 block，减少无效计算。对变长 batch，使用 `cu_seqlens` 索引每个样本的起止位置，避免对 padding token 做无意义的计算。

---

## 工程化与诊断手段

优化一半靠写，一半靠量。没有 profiler 就是盲调。

### Autotune 诊断

通过环境变量 `TRITON_PRINT_AUTOTUNING=1` 可以打印每个 config 的耗时和被选中的最优 config。据此可以：

- 缩小搜索空间，剔除明显劣势的配置。
- 固化常用形状的最优配置，避免首跑抖动。
- 发现"某些形状没有好配置"的信号，提示需要重新设计 kernel 结构。

### 查看中间代码

设置 `TRITON_CACHE_DIR` 指向某个目录，Triton 会把编译过程中的中间产物保留下来，包括：

- **TTIR（Triton IR）**：高层 IR，看整体结构。
- **TTGIR（Triton GPU IR）**：能看到 layout、pipeline、num_stages 是否生效。
- **LLIR（LLVM IR）**：用于深度调试。
- **PTX**：检查是否生成了 `ld.global.v4`、`mma.sync`、`cp.async` 等关键指令。
- **cubin / 元信息**：查看寄存器数量、shared memory 使用量。

其中 TTGIR 和 PTX 是性能调优时最重要的两个产物。

### Nsight Compute

使用 `ncu` 进行细粒度 profiling 时，重点关注以下指标：

- **SM 利用率**（`sm__throughput.avg.pct_of_peak_sustained_elapsed`）：反映计算单元是否饱和。
- **显存带宽利用率**（`dram__throughput...`）：反映是否 memory-bound。
- **全局 load 事务数**：判断是否存在访存冗余或 uncoalesced。
- **Tensor Core 指令数**：确认是否真正走到了 MMA 路径。
- **每线程寄存器数与每 block shared memory 量**：判断 occupancy 瓶颈。

### Roofline 分析

Roofline 模型是判断优化方向的基本工具。计算硬件理论算术强度：

- 若 kernel 的算术强度低于硬件拐点：**memory-bound**，优化方向是访存合并、tile 放大、融合、减少冗余搬运。
- 若 kernel 的算术强度高于硬件拐点：**compute-bound**，优化方向是使用 Tensor Core、低精度、指令选择。

盲目优化前先做一次 roofline，可以避免走弯路。

### 基准对照

- 与 cuBLAS / cuDNN / CUTLASS / FlashAttention 等官方实现对齐相同形状，看差距。
- 与 `torch.compile` 生成的 Triton kernel 对比，后者常常是一个已经 autotune 过的 baseline。
- 测试覆盖多种形状（大方阵、瘦长矩阵、batched small GEMM、极端 K 长度），避免只对单一 case 调优。

### 正确性与稳定性验证

- 用 fp64 参考值做 allclose 检查，`atol/rtol` 根据精度类型合理设置。
- 边界 case 必测：M/N/K 不是 block 整数倍、K = 0、全 mask、极大极小值、NaN 传播。
- 多架构交叉验证（T4、A100、H100、消费级卡），避免只在单一硬件上能跑。

### 性能回归与版本管理

- Autotune 结果应缓存固化，发版时一并打包，避免用户首跑慢。
- 把关键 kernel 的 benchmark 纳入 CI，Triton 升级时第一时间发现性能回归。
- 记录真实业务的形状分布，做 profile-guided 优化，而不是凭直觉选配置。

---

## 附：优化顺序建议

这八类措施彼此耦合，实际调优时推荐的顺序是：

1. **先正确，再优化**：写出功能正确的 naive 版本作为 baseline 和对照。
2. **调 tile 和 num_warps / num_stages**：用 autotune 快速拿到可用性能。
3. **打磨访存**：合并访问、向量化、掩码处理、swizzle。
4. **走 Tensor Core 路径**：若是计算密集型算子，核心运算使用 `tl.dot`，让计算映射到 MMA 指令；数据类型严格沿用上层给定的 dtype，不擅自降精度。
5. **做融合**：把 epilogue、前后处理一并塞进 kernel，减少显存往返。
6. **改 grid 策略**：swizzle、persistent、split-K，按形状和硬件选择。
7. **查 profiler**：用 `ncu` 定位最后 10%～20% 的瓶颈。
8. **固化与回归**：缓存 autotune 配置，建立 benchmark 和 CI 监控。

不同算子的侧重点也不同：

- **Matmul**：重 tile 与 Tensor Core，重 swizzle 与 split-K。
- **Attention**：重融合与 online softmax，重 IO 复杂度。
- **Elementwise / 归一化**：几乎完全是访存优化，重融合与向量化。
- **变长 / 稀疏**：重 grid 策略与负载均衡。

把以上维度都纳入考量并循序渐进地打磨，Triton kernel 通常可以做到接近甚至持平手写 CUDA 的水平。

---

## 官方文档

https://triton-lang.org/main/index.html
