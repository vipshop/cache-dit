---
name: cache-dit-quantization
description: 'Use when choosing, using, extending, or debugging Cache-DiT quantization: TorchAO online quantization (fp8/int8/int4/nvfp4), SVDQuant W4A4 PTQ (calibration + checkpoint) and DQ (no calibration), QuantizeConfig / regional quantize / precision plan / per_tensor_fallback, per-layer accuracy observers, the SVDQ converter CLI, and the user docs in docs/user_guide/QUANTIZATION.md.'
argument-hint: 'Describe the target model and GPU, the quantization type or backend, the accuracy/memory/latency goal, and whether serialization, observers, or a new backend is involved.'
user-invocable: true
---

# Cache-DiT 量化与后端接入

## 0. 任务路由

| 任务 | 判断依据 | 读哪里 |
|---|---|---|
| 选一种量化 / 跑通量化 | 只是调用 `enable_cache` / `quantize` | §2、§3；用户文档以 `docs/user_guide/QUANTIZATION.md` 为唯一权威 |
| 调精度与显存 | 要动 regional / exclude / precision plan / fallback / observer | §4–§6 |
| 接入新量化后端或新 PTQ 工作流 | 要新增 `src/cache_dit/quantization/<backend>/` 或改共享 config/dispatch | §7–§11 |
| 改 SVDQ 算子或 GEMM 本身 | 动 `csrc/kernels/svdq/` | 走 `cache-dit-cuda-kernel`，本 skill 只管工作流与配置 |

代码、文档、CLI 三处描述同一个量化能力时，**先改 `docs/user_guide/QUANTIZATION.md`，再同步本 skill**；本 skill 只做索引与判据，不复制整段用户文档。

## 1. 代码地图

| 位置 | 职责 |
|---|---|
| `src/cache_dit/quantization/config.py` | `QuantizeConfig` 全部字段 + `svdq_kwargs` 的集中校验（`_resolve_svdq_*`）、`DEFAULT_EXCLUDE_LAYERS` |
| `src/cache_dit/quantization/backend.py` | `QuantizeBackend` 枚举（`AUTO`/`TORCHAO`/`CACHE_DIT`/`NONE`）与字符串别名解析 |
| `src/cache_dit/quantization/dispatch.py` | 公共 `quantize()` / `load()` / `remove_quantization_stats()`，按 backend 分派 |
| `src/cache_dit/quantization/torchao/` | TorchAO online 路径：`quantize_ao.py`（quant_type → AO config、过滤与 fallback）、`_scaled_mm.py` |
| `src/cache_dit/quantization/svdquant/` | SVDQuant：`ptq.py`（校准 + 序列化 + load）、`quantizer.py`（DQ/PTQ 量化数学）、`linear.py`/`fused.py`（运行模块）、`packing.py`/`lowrank.py`/`passes.py`/`dtensor.py` |
| `src/cache_dit/quantization/observers.py` | 逐层精度观察器（默认关闭） |
| `src/cache_dit/quantization/converter.py` | 独立转换 CLI（`cache-dit-convert`，只支持 SVDQ `_dq`） |
| `src/cache_dit/_utils/utils.py` | CLI 参数与 `maybe_quantize_transformer`（`--quantize-*` → `QuantizeConfig`） |
| `docs/user_guide/QUANTIZATION.md` | 用户文档：类型总表、各路径示例、observer 与 converter |
| `tests/quantization/` | 配置、observer、torchao、SVDQ（ptq/dq/quantizer/compile/padding/imports/offload）测试 |

## 2. 支持面（quant_type）

两条互不相干的路径，**不要混用**：

- **TorchAO online quantization**（`quant_type` 直接是 torchao 类型）：`float8_per_row`（推荐）、`float8_per_tensor`、`float8_per_block`、`float8_weight_only`、`int8_per_row`、`int8_per_tensor`、`int8_weight_only`、`int4_weight_only`、`nvfp4`、`nvfp4_weight_only`。
- **SVDQuant W4A4**（`quant_type` 形如 `svdq_{int4,nvfp4}_r{rank}[_dq]`）：有 `_dq` 后缀是 DQ（无校准），无后缀是 PTQ（需要校准 + 可序列化）。

门槛与依赖（细节与最新版本要求见 `QUANTIZATION.md` 的 Overview）：

| 类型 | GPU | 额外依赖 |
|---|---|---|
| float8_* | >=sm89 | torchao>=0.17 |
| int8_* | >=sm80 | torchao（int8 kernel 走 mslk，需另装） |
| int4_weight_only | >=sm90（w4a16 需要 TMA） | torchao（int4 kernel 走 mslk，需另装） |
| nvfp4 / nvfp4_weight_only | >=sm100 | torchao（nvfp4 kernel 走 mslk，需另装；Triton 激活量化 + 动态 per-tensor scale） |
| svdq_int4_* | >=sm80 且排除 sm90（无 INT4 MMA） | cache-dit CUDA 扩展 |
| svdq_nvfp4_* | >=sm120 | cache-dit CUDA 扩展 |

分布式（CP/TP）场景优先 float8，避免 int4/nvfp4 的运行时不兼容；float8 内部还要避开 **`float8_per_block`**——文档明确它对 distributed inference 暂不支持。

## 3. 公共 API 表面

```python
from cache_dit import QuantizeConfig, QuantizationObserverConfig

cache_dit.enable_cache(pipe, quantize_config=QuantizeConfig(quant_type="float8_per_row"))
quantized = cache_dit.quantize(module, quantize_config)      # 底层入口
restored = cache_dit.load(module, "path/to/checkpoint")      # 仅 CACHE_DIT（SVDQ）支持
```

`QuantizeConfig` 主要字段：

| 字段 | 作用 |
|---|---|
| `backend` | `AUTO`（`quant_type` 以 `svdq` 开头 → `CACHE_DIT`，其余 → `TORCHAO`）/`TORCHAO`/`CACHE_DIT`/`NONE` |
| `quant_type` | 见 §2 |
| `exclude_layers` | 按 state_dict 名排除（默认带 `DEFAULT_EXCLUDE_LAYERS`） |
| `regional_quantize` | 只量化 repeated blocks（默认 True） |
| `repeated_blocks` | 非 diffusers 模型手动指定块名 |
| `filter_fn` | `(module, name) -> bool`，指定后忽略 `exclude_layers` |
| `components_to_quantize` | `list[str]` 或 `{组件: {quant_type: ...}}`，跨 transformer/text_encoder/controlnet |
| `per_tensor_fallback` | fp8 per-row/per-block 不受支持时回退 per-tensor（TP 的 RowwiseParallel 场景需要，默认 True） |
| `precision_plan` | `{层名片段: quant_type}`，粒度最细，覆盖上面两项 |
| `calibrate_fn` / `serialize_to` | SVDQ PTQ 的校准回调与落盘路径 |
| `svdq_kwargs` | SVDQ 后端专用参数（经 `_resolve_svdq_*` 统一校验，含 `streaming`/`calibrate_precision`/`runtime_kernel`/`quantize_device`/`smooth_strategy` 等） |
| `observer` | `QuantizationObserverConfig`，默认 `None` = 关闭 |
| `verbose` | 打印逐层量化信息 |

CLI：`--quantize` / `--quantize-type` / 预设（如 `--svdq-int4-r32-dq`、`--svdq-nvfp4-r32-dq`）/ `--quantize-exclude-layers` / `--quantize-text-encoder` / `--quantize-text-type` / `--quantize-controlnet` / `--quantize-controlnet-type` / `--quantize-verbose`。转换独立模型用 `cache-dit-convert`（§5.4）。

## 4. TorchAO online 路径

- 入口：`quantize_ao()` + `QuantizeAOContext`（`torchao/quantize_ao.py`），负责 quant_type → AO config、`filter_fn`/fallback 过滤、模块替换。
- `int8/int4/nvfp4` 的 kernel 由 mslk 提供（torchao 侧依赖，本仓不 import）；`nvfp4` 固定 `use_triton_kernel=True`、`use_dynamic_per_tensor_scale=True`，与 `svdq_nvfp4_*` 是两条不同实现，精度/性能结论不能互相套用。
- `per_tensor_fallback` 只对 fp8 生效（int8 未实现回退）；TP 下 `RowwiseParallel` 层常需要它。
- 只支持 online quantization，**没有** TorchAO 的序列化/加载路径：`cache_dit.load()` 遇到非 CACHE_DIT backend 会直接 `ValueError`。

## 5. SVDQuant W4A4：PTQ 与 DQ

### 5.1 选 PTQ 还是 DQ

| | PTQ（`svdq_*_r{rank}`） | DQ（`svdq_*_r{rank}_dq`） |
|---|---|---|
| 需要校准数据 | 是（`calibrate_fn`） | 否 |
| 产物 | 可 `serialize_to` 落盘并 `load` | 运行时量化，不落盘 |
| 适用 | 想要更高精度、可复现交付 | 想零数据、直接开箱 |

两者共享 runtime：`runtime_kernel` 只接受 `None/native/cute`；`None` 一律解析为 `native`（`svdquant/linear.py` 实际实现；`config.py` 里 "NVFP4 -> CuTe" 的注释是过期描述，别照抄），`cute` 是显式 opt-in 验证路径，且要求 `precision="nvfp4"` + `torch_dtype=bfloat16`。

### 5.2 PTQ 流程要点

1. 用 float 模型 + 校准回调收集统计（回调签名见 `QUANTIZATION.md` 的 PTQ 示例）。
2. `QuantizeConfig(quant_type="svdq_int4_r128", calibrate_fn=..., serialize_to=...)` → `cache_dit.quantize`。
3. 落盘是确定性文件名 + colocated manifest（`quant_config.json`）；`cache_dit.load(module, path)` 支持 config / 文件 / 目录三种输入，**先校验元数据再改模块**。
4. 量化完成后释放校准产物（`remove_quantization_stats`）。

### 5.3 smooth 策略（PTQ 与 DQ 不同）

`smooth_strategy` 总体合法值：`activation`、`identity`、`weight`、`weight_inv`、`few_shot`，但两侧可用集是互斥的：

- **PTQ 只接受 `activation`**（校准派生平滑）；传其他四个值会在 config 校验阶段直接 `ValueError`。
- **DQ 不接受 `activation`**（`quantizer.py` 会拒绝），只能用 `identity`（DQ 的默认）/`weight`/`weight_inv`/`few_shot`。

`few_shot` 是"先跑若干次真实 forward 再量化"，其 `few_shot_steps` 计数是**累积在同一 armed 模块上的**（同一 pipeline 跑两次 50 步 ≈ 100 次 forward），relax 策略见 `_SVDQ_FEW_SHOT_RELAX_STRATEGIES`。`few_shot` 与 `--compile` 组合时有专门的延迟 compile 顺序，改动前先看 `QUANTIZATION.md` 对应小节。

### 5.4 converter CLI

`cache-dit-convert`（或 `python3 -m cache_dit.quantization.converter`）一步完成"加载 float pipeline → SVDQ DQ → 落盘"，**只支持 `_dq`**，无需校准数据；参数与示例见 `QUANTIZATION.md` 的 "SVDQ Converter CLI"。

## 6. 逐层精度观察器（默认关闭）

`QuantizeConfig(observer=QuantizationObserverConfig(...))` 会把每个量化 linear 用 bf16 重算并统计误差：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `enabled` | `False` | 唯一开关，关闭时零开销（不做模块遍历、不挂 hook） |
| `json_path` | `None`（落到 `./quant_observer.json`） | 聚合结果落盘位置 |
| `max_steps` | `1` | 每 N 次 transformer forward 刷新一次，`<=0` 只在退出时 dump |
| `keep_refs_on_gpu` | `False` | 参考权重常驻显存（快，但约翻倍权重显存） |
| `layer_name_patterns` | `None`（观察全部量化 linear） | 正则（`re.search`）过滤层名，过滤发生在快照之前 |

用途是定位"哪个层掉精度"，不是质量门禁；报告产物不要写进 pass/fail 判据。

## 7. 接入新量化后端 / 新 PTQ 工作流：设计原则

`torchao`（online，无序列化）与 `svdquant`（PTQ + 序列化 + DQ）是两个不同形态的参照，**按形态选参照，不要机械照搬**。

1. **公共 API 对称**：`quantize` / `load` / `QuantizeConfig` 同一层；有 save 就必须有对等的 load。
2. **后端参数集中校验**：新增字段优先进 `svdq_kwargs` 式的作用域字典（配 `_resolve_*` 校验），不要往顶层堆字段。
3. **内部机制私有**：calibrator、observer、context、packer 不进 docs 与集成测试。
4. **序列化确定**：确定性文件名 + manifest + 元数据先校验再改模块；load 路径解析收敛到一处。
5. **慢测必须 opt-in**：默认跑快的公共 API 测试；pipeline/compile/大模型对比用环境变量门控。

同时明确：cache-dit 的量化算子只维护 CUDA（详见 `cache-dit-cuda-kernel` 的后端政策），不要引入 Triton/CuTe DSL 后端。

## 8. 文件落点

| 改动 | 位置 |
|---|---|
| 通用配置字段与校验 | `src/cache_dit/quantization/config.py` |
| 后端解析 | `src/cache_dit/quantization/backend.py` |
| 公共分派 | `src/cache_dit/quantization/dispatch.py` |
| 包导出（公共 API 变化时） | `src/cache_dit/quantization/__init__.py`、`src/cache_dit/__init__.py` |
| 后端实现 | `src/cache_dit/quantization/<backend>/{ptq.py,quantizer.py,...}` |
| 用户文档 | `docs/user_guide/QUANTIZATION.md` |
| 测试 | `tests/quantization/test_<backend>_*.py` |

只有跨多个测试文件复用时才新建共享测试工具（现状：`tests/quantization/_svdq_test_utils.py`、TP 保存/加载子进程 worker `_svdq_tp_save_load_worker.py`）。

## 9. 序列化与 load checklist

1. checkpoint 文件名确定性；
2. 目录加载时同目录有 manifest；
3. load 接受 config / 文件 / 目录，但只有能无歧义解析时才接受；
4. 元数据校验先于模块改写；
5. quant_type 不匹配报清晰错误；
6. manifest 缺失/损坏报清晰错误；
7. 往返测试证明 load 后输出与量化后一致。

## 10. 测试策略

现有文件（按用途）：

| 文件 | 覆盖 |
|---|---|
| `test_quantize_config.py` | `QuantizeConfig` 字段与校验、`svdq_kwargs` 拒绝非法值 |
| `test_quantize_observers.py` | observer 开关、正则过滤、dump 内容 |
| `test_torchao_int8.py` / `test_torchao_nvfp4.py` | TorchAO int8 / nvfp4 路径 |
| `test_svdquant_ptq.py` | PTQ 公共 API、序列化与 load 往返、元数据失败 |
| `test_svdquant_dq.py` | DQ 公共 API、CLI 映射、smooth 策略、layerwise offload 相关 |
| `test_svdquant_quantizer.py` / `test_svdquant_padding.py` | 量化数学、padding 行为 |
| `test_svdquant_compile.py` | 量化 + `torch.compile` 组合 |
| `test_svdquant_imports.py` | 扩展缺失时的 import/错误路径 |
| `test_layerwise_offload.py` | layerwise offload 交互 |
| `_svdq_test_utils.py` / `_svdq_tp_save_load_worker.py` | 共享测试工具；TP 保存/加载用的子进程 worker |

要求：

- 快测默认跑：公共 API 替换了预期层、save/load 往返、配置拒绝非法组合、目录/文件/config 三种 load、坏元数据、exclude/filter 行为。
- 慢测 env-gated 且分开：真实 pipeline 集成、序列化+重载闭环、一个主质量门、性能/显存单列报告。
- compile 测单独门控，不与慢测混跑。
- 大产物落 `.tmp/tests/...`；确定性 seed；benchmark 表只是报告，不做 pass/fail。

## 11. 常见坑

1. **`QuantizeConfig` 与旧 kwargs 混用**：同时传 `quantize_config` 和 kwargs 时 kwargs 被忽略（只 warn），会被误当成生效。
2. **`load` 只支持 CACHE_DIT/SVDQ**：TorchAO 量化没有序列化产物，离线交付只能用 SVDQ PTQ。
3. **`per_tensor_fallback` 只管 fp8**，且默认开启；关掉后排并行层可能直接报布局错误。
4. **int4/nvfp4 对分布式不友好**，CP/TP 组合先降到 float8 验证框架再谈低比特；float8 里再避开 `float8_per_block`（distributed 暂不支持）。
5. **`few_shot` 的 forward 计数是累积的**，同一模块多次 pipeline 调用会提前触发物化。
6. **`svdq_int4_*` 在 Hopper（sm90）不可用**，构建期就会被 SKIP。
7. **observer 默认关闭且不在质量门禁里**，别把它的 JSON 当成验收数据。
8. **fp8 × compile 组合要单独验证语义**（历史上有过数值/图像语义异常），不要只看 latency。
9. **PTQ 与 DQ 的 `smooth_strategy` 互斥**：PTQ 只认 `activation`，DQ 只认 identity/weight/weight_inv/few_shot。
10. 文档与 skill 的 quant_type 表会漂移：发现不一致以 `QUANTIZATION.md` 为准并回来修本 skill。

## 12. 参考触点

- `docs/user_guide/QUANTIZATION.md`（用户视角的唯一权威）
- `src/cache_dit/quantization/config.py`、`backend.py`、`dispatch.py`
- `src/cache_dit/quantization/torchao/quantize_ao.py`
- `src/cache_dit/quantization/svdquant/{ptq.py,quantizer.py,linear.py,fused.py}`
- `src/cache_dit/quantization/observers.py`
- `src/cache_dit/quantization/converter.py`
- `tests/quantization/`
