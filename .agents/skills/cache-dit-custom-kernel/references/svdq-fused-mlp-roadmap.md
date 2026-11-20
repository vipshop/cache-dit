# SVDQ NVFP4 Fused MLP 覆盖与后续路线

状态日期：2026-11-20。范围是 cache-dit SVDQ NVFP4 的 Python fused-MLP pass 与 native/CuTe 融合 epilogue，不包含 FP8/FP4 native backend。结论来自对本地 `diffusers`（0.44.0.dev0）和 cache-dit 已集成模型的结构调研；新增 pattern 尚未做全部真实模型 e2e，文中标为"待验收"的项不能当作已支持。

## 1. 一句话结论

融合 kernel 的主体能力已经具备：tanh-GELU qout、独立 gate 张量、`value * silu(gate)` producer、bias 和 native/cute 两 runtime 都有实现。后续主要矛盾不是继续优化 GEMM 微架构，而是：

1. 修掉会让 exact GELU 被静默替换成 tanh GELU 的正确性问题，并补齐 pass 护栏；
2. 用 Python 检测器覆盖更多等价的 Linear/Gate 组合，这是当前覆盖率和收益的主要来源；
3. 最后才扩 kernel 的 activation 模式（GEGLU、exact GELU 等）。

不要把 e2e 的量化收益算成融合收益。已有实测中融合本身在 FLUX 上约 +3.1%、Qwen21 约 +4.4%，链级 native 约 1.25×；大头来自量化。

## 2. 当前精确支持面

只有 3 个公开 pass：

- `FusedGeluMlpPass`：`src/cache_dit/quantization/svdquant/passes.py`
- `FusedGeluProjPass`：同文件
- `FusedSwiGluMlpPass`：同文件

对应 runtime helper 在 `src/cache_dit/quantization/svdquant/fused.py`：

- `fused_gelu_mlp`：fc1 fused qout+D → fc2，`write_dense=False`。
- `fused_gelu_proj`：只融合单流投影的 GELU，必须返回 dense 激活。
- `fused_swiglu_mlp`：value 先 GEMM，`gate` 进入下一层 activation producer，producer 内执行 `value * silu(gate)`。

当前硬约束：

- fused GEMM 实例只覆盖 BN=256/BK=128 的 tile，因此 fused 链实际需要 `N % 256 == 0`、`K % 128 == 0`，并且 M 有 tile 对齐要求；CuTe 路径的 `padded_out_features` 目前只按 128 取整，宽度 1920/2688 等可能在推理期直接抛错。pass 必须在装配前预检并跳过，不能等 kernel 报错。
- 独立 gate 张量要 pad 到 256 对齐；half-split 的列序必须由模型语义确认，不能只看宽度。
- `row_major` 融合 D 只允许 identity rank scale；TP/DTensor 层当前全部被跳过，因此 TP 下 fused 覆盖率为 0。
- native 支持 INT4 与 NVFP4；CuTe 只支持 NVFP4，且当前重点是 NVFP4。

## 3. 模型 FFN/MLP pattern 盘点

这里按共享结构聚合，不按模型逐条列流水账。路径相对 `/workspace/dev/vipshop/diffusers`。

| Pattern | 结构要点 | 代表模型 | 当前状态 |
|---|---|---|---|
| A. 标准 `FeedForward + tanh GELU` | `net=[GELU(proj), Dropout, Linear]`，bias 默认 True，常见 mult=4 | FLUX.1、SD3/3.5、PixArt、Qwen-Image、GLM、Wan、HunyuanVideo、CogVideoX、LTX、Mochi 等 | ✅ `fused_gelu_mlp`；exact GELU 有误融风险 |
| B. 单流 concat GELU | `proj_mlp -> GELU -> cat(attn, mlp) -> proj_out` | FLUX.1 single block、Chroma | ⚠️ 只有 `fused_gelu_proj`，fc2 输入含 attn 半边，不能全融 |
| C. 独立 gate SwiGLU（diffusers 命名） | `out(silu(gate_layer(x)) * proj(x))` | Qwen-Image-2.1 等 | ✅ 只认 `proj/gate_layer/out + nn.SiLU` |
| D. half-split SwiGLU（diffusers 标准类） | 单 Linear 出 `2H`，按列 chunk 后 value/silu(gate) | Mochi、MiniMax-H3、StableAudio(v1) | ✅ `fused_swiglu_mlp(gate=None)`，列序需与 kernel 一致 |
| E. Llama 式 w1/w3/w2 或 gate/up/down | `down(silu(w1(x)) * w3(x))`，常用函数式 `F.silu`，无 `nn.SiLU` 模块 | Z-Image、HiDream dense、Lumina2、Krea2、LongCat-Video-Avatar、StableAudio3 | ❌ 属性命名不匹配，纯 Python 可覆盖 |
| F. 自建/Sequential GELU MLP | `Sequential(Linear, GELU, [Dropout], Linear)` 或分离的 `fc1/act/fc2`；GELU 不是 block 直接 `net[0]` | Wan-Animate-2、LongCat-AudioDiT、SanaWM、Kandinsky5 | ❌ 类名/嵌套不匹配，纯 Python 可覆盖 |
| G. GEGLU | `linear_fc2(up_proj(x) * gelu(gate_proj(x)))` | Ernie-Image | ❌ kernel 的 gate activation 目前固定 SiLU |
| H. packed QKV+MLP | QKV 与 MLP up/gate 在同一个大 GEMM 输出里按列切 | FLUX.2/3 | ❌ 结构不是三个独立 Linear；单独架构专项 |
| I. Conv MLP / MoE | Conv2d + depthwise + temporal GLU；shared/router experts | SanaVideo、HiDream MoE | ❌ 不应强行套 GEMM 融合 |

补充：

- 视频侧多数标准 DiT（Wan、HunyuanVideo、CogVideoX、LTX、Mochi 等）使用的是 pattern A/D，现有 pass 已能命中；新增工作主要集中在自建类、Llama 式命名和图像模型的特殊结构。
- 现有 pass docstring 里列出的部分类名已过期（diffusers 中没有对应类），新增 pass 时一并修正，不要按 docstring 猜结构。
- 当前 `FusedGeluProjPass` 用"out>in 且 in2==in1+out1"的宽度启发式；对同一 block 含多个投影时可能误配，必须加唯一 activation 和精确数据流校验。

## 4. P0：正确性修复（先做，阻断后续扩展）

### P0-1. GELU `approximate` 语义必须显式处理

现状：SVDQ qout epilogue 固定 tanh 近似，但 pass 不读取 `nn.GELU.approximate`；因此 exact GELU 模型可能被静默按 tanh 融合。

落点：

- `src/cache_dit/quantization/svdquant/passes.py` 的 GELU 检测；
- `csrc/kernels/svdq/native/gemm_utils.cuh` 的 GELU 公式；
- `csrc/kernels/svdq/cute/fp4/gemm/gemm_w4a4.cuh` 的融合 qout epilogue。

先做：

- 读取 `approximate`；
- `none`/exact 暂时跳过融合并 warning；
- 只有 `tanh` 才走当前 qout；
- `fused_gelu_proj` 同样处理。

验收：

- exact GELU 结构 fusion 命中数为 0，输出与 unfused 等价；
- tanh 结构仍命中；
- Kandinsky6、Wan/SkyReels 的 exact embedder 加回归用例；GLM glyph 结构需在真实模型上确认。

### P0-2. fused 几何预检

在 pass `apply()` 前检查 fused 实例的 N/K/M 对齐和 `padded_out_features % 256`，不满足则 skip + 明确 reason。重点保护 CuTe `runtime_kernel="cute"`，但 common preflight 不应只放在 runtime helper 中。

验收：`hidden=1920/2688` 等非 256 对齐宽度在装配期跳过，不出现首次推理 throw；统计中记录 skip 原因。

### P0-3. 对齐 FP4 的 pass 护栏

SVDQ 当前缺少：

- `parent.training` / active Dropout 跳过；
- GELU 子模块唯一性；
- 幂等标记；
- patched forward 接收 `*args/**kwargs`；
- 不改写 model checkpoint 的保证。

参考 `src/cache_dit/quantization/fp4/passes.py`。验收：训练态、`p>0` Dropout、双 GELU、连续 apply 两次均不融合/不重复包装。

### P0-4. 命中统计与 fail-fast

当前零命中只写 `debug`，用户打开 `--svdq-fused-mlp` 后可能完全没生效。应：

- 给 root module 记录 `_svdq_fused_mlp_targets`（按 gelu_mlp/gelu_proj/swiglu 分类）；
- 每个 pass 记录 applied/skipped reason；
- 零命中至少 warning；与 FP4 对齐时可考虑显式 ValueError（需评估旧用户兼容性）；
- 日志不得输出 tensor。

验收：命中数与 profiler/公开 helper 调用次数一致，零命中用例有可读错误或 warning。

### P0-5. 补 pass 自身测试

现有测试主要覆盖 kernel 和 SwiGLU 单块，覆盖最多模型的 `FusedGeluMlpPass` / `FusedGeluProjPass` 没有 detect/apply 测试。必须补：

- 两个 GELU pass 的 detect/apply；
- `fused_mlp=True` 从 quantize 到 forward、serialize/load 再 forward；
- `fused_gelu_proj` 的 rel-L2 / cosine 数值断言；
- 非 128/256 对齐宽度与 padding；
- TP 降级；
- compile / CUDA Graph；
- training/Dropout/双 GELU/幂等负例。

## 5. P1：passes 覆盖扩展（主要 ROI）

原则：优先用 Python 检测器表达"等价数据流"，不为每种命名写一个模型专用分支；只有语义不明确或需要新 producer 数学时才进 kernel。

### P1-1. 泛化独立 gate 检测

从固定 `proj/gate_layer/out + nn.SiLU` 扩为：

- `w1/w3/w2`；
- `gate_proj/up_proj/down_proj`；
- `gate/up/down`；
- `proj_in/proj_out`；
- 函数式 `F.silu`，不要求 `nn.SiLU` 模块。

约束：

- 两个同宽 value/gate 候选 + 一个 down 投影；
- down.in_features 与 hidden 宽度一致，必要时使用显式 gate，不猜 half-split 列序；
- 不匹配时跳过，不能凭模块名硬改；
- 覆盖 Z-Image、HiDream dense、Lumina2、Krea2、LongCat-Video-Avatar、StableAudio3。

FLUX.2/3 的 `linear_in/linear_out` 或 packed projection 需要单独确认列布局；先不要把它们合并进默认 half-split 规则。

### P1-2. 支持自建 Sequential/嵌套 GELU MLP

检测：

- `nn.Sequential(Linear, GELU, [Dropout], Linear)`；
- block 上分离的 `fc1/act/fc2`、`linear1/act/linear2`。

要求：

- GELU 是唯一、无歧义的激活；
- 中间无 residual、norm、额外乘法；
- 推理态且 Dropout p==0；
- 复用 `fused_gelu_mlp`，不改 kernel。

覆盖 Wan-Animate-2、LongCat-AudioDiT、SanaWM、Kandinsky5。

### P1-3. 增加 `fused_silu_proj`（可选低优先）

kernel 的 dense SiLU 已存在（`fuse_silu`），但没有公开 pass。若真实模型存在 `Linear -> SiLU -> concat` 且不接 qout，可新增只返回 dense 的 projection pass。它不解决 GEMM 链融合，收益有限，排在 P1-1/P1-2 之后。

### P1-4. 每个新 pattern 的真实模型 acceptance

至少做：

- 模型级 fused 目标数 > 0；
- qact 与 unfused 逐字节（允许 D 的既有误差门槛）；
- chain rel-L2 < 5e-3、cosine > 0.999；
- eager + compile；
- 固定 seed 的 PSNR/SSIM 和视觉图；
- CUDA Event 的 chain latency。

建议顺序：

1. Wan-Animate-2（Sequential GELU）；
2. Z-Image 或 Krea2（w1/w3/w2）；
3. Qwen21（现有 SwiGLU 回归）；
4. FLUX.2（只在列序确认后）。

## 6. P2：kernel activation 模式扩展

只在前两项完成、覆盖率统计稳定后投入。优先做与真实模型明确相关的模式，不为"通用性"预造变体。

### P2-1. Producer 的 gate activation 枚举

现状：

- `quantize_glu.cuh` 的成对数学固定 `value * silu(gate)`；
- `quantize_w4a4` 只有 `gate`/`fuse_glu` 语义；
- native/CuTe 两族 producer 都受影响。

建议引入明确的 `gate_act` 或 `act_mode`，至少支持：

- SiLU（现有，保持 bitwise）；
- tanh GELU（可用于近似 GEGLU）；
- exact GELU（Ernie-Image 需要，和 PyTorch `F.gelu` 保持一致）。

数学契约：

- 先计算 gate activation，再按模型原有 BF16 舍入点相乘；
- 保持 qact、ascales、D 的既有输出布局；
- 不能把 `fuse_glu` 与 `qout` 的既有互斥关系打乱；
- native/CuTe 的 activation 结果必须逐字节或明确误差门槛对拍。

验收：GEGLU 单层、独立 gate、half-split、两 runtime、rank32/64/128、M tail；Ernie-Image 模型级 e2e。

### P2-2. Fused qout 的 exact/tanh GELU 模式

如果 P0 发现 exact-GELU 模型占比显著，再给 GEMM epilogue 增加 activation mode：

- native 的 `EpilogueGelu` 从固定 tanh 改为显式 exact/tanh；
- CuTe 的融合 qout 增加同样分派；
- pass 把 `approximate` 传入 GEMM；
- unfused 与 fused 的舍入顺序保持一致。

若 exact 模型很少，长期保持"exact 跳过融合"更划算，不要仅为了统一接口新增 kernel。

### P2-3. 后置架构专项（不属于当前 activation 核心）

以下方向潜在收益高但实现成本/风险显著，单独立项：

- FLUX.2/3 packed QKV+MLP：同一 GEMM 的列切片和 gate/up 布局，需要新的 epilogue/部分 K 写回；
- FLUX.1 single-stream concat 整链融合：目前只能 `fused_gelu_proj`；整链需要 attn/mlp 分片协同，或量化期拆分 `proj_out`；
- MoE 多专家：routing/gather/scatter 与批量 GEMM，当前收益不明确。

## 7. 不建议的方向

- 不重跑已证伪的 CuTe D 归约微优化（LDSM A、累加器寄存器 A、双缓冲、BK64/S6、BN128 等）；当前 GEMM 已接近局部最优。
- 不为 INT4 优化 fused dense store；INT4 GEMM 过慢，融合收益占比仅约 0.8%。
- 不将 `fuse_silu` 与 qout 组合；数学上需要第二操作数，kernel 已拒绝。
- 不用 e2e PSNR 判定 fused 等价性；SVDQ 跨进程原子顺序造成 23–28 dB 噪声。等价性看 qact 逐字节、qscale 集合/误差和链 rel-L2。
- 不把 Conv MLP 或 MoE 强行转成 Linear pass。

## 8. 推荐落地顺序

1. P0-1 exact GELU guard + P0-2 fused geometry preflight + P0-3 pass guards；
2. P0-4 命中统计/零目标提示 + P0-5 pass 测试；
3. P1-1 泛化 gated 命名；
4. P1-2 Sequential/嵌套 GELU；
5. 真实模型 acceptance + 覆盖率/性能统计；
6. 再评估 P2-1 GEGLU 是否值得进 kernel，P2-2 exact epilogue 仅按实际模型需求决定。

每个新 pattern 合并前必须回答：命中了多少层、是否真实执行了 fused helper、融合是否改变 qact、收益是否超过噪声、未命中是否明确可见。

## 9. 证据边界与待确认项

- 本报告来自源码结构调研，新增 pattern 的 detect/apply 需要小 config 实例化验证。
- 视频模型 FFN 参数占比约 57%–70%（按代码默认宽度估算），不等于 e2e 收益；必须实测 chain/e2e。
- MiniMax-H3 等部分实际 checkpoint 的 `ffn_dim`/`hidden_size` 未从 config 读取，不能写死阈值。
- 现有 pass 的 docstring 模型类名有过期项；实现时以本地 diffusers 源码为准。
- TP、offload、compile、CUDA Graph 的组合没有完整回归，新增 pass 必须至少覆盖 compile 和 TP skip。
