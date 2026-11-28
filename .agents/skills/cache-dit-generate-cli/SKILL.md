---
name: cache-dit-generate-cli
description: "Use when running, composing or debugging `cache_dit.generate` / `cache-dit-generate`: selecting an example or model path, cache (DBCache/SeaCache/SoftCache) and calibrators, TorchAO or SVDQuant quantization, attention backends, parallel degrees (Ulysses/Ring/TP, TE-P/VAE-P/ControlNet-P), CPU offload, torch.compile/CUDA Graph, steps masks, profiling and the multi-GPU launcher. Covers the `--xxx-params` grammar, `--xxx-config` aliases, quoting, validation errors and verified FLUX.1 recipes. 触发词：generate CLI、命令行推理、--cache-params、--parallel-params、--quantize-params、--svdq-params、FLUX e2e、多卡推理。"
argument-hint: "Describe the model or example, the GPU count and memory budget, and the goal (speed, memory, accuracy, multi-GPU, quantized, profiling)."
user-invocable: true
---

# Cache-DiT Generate CLI

## 0. 适用范围

用于 `python -m cache_dit.generate` 与同义的 `cache-dit-generate`：

- 选例子、模型路径或 task，拼一条可运行的命令；
- 组合 cache、calibrator、quantization、attention、parallel、offload、compile；
- 排查参数解析、默认值与参数之间的冲突。

不覆盖：新模型接入（`cache-dit-model-integration`）、量化算法与后端实现（`cache-dit-quantization`）、并行 planner 与通信原语（`cache-dit-distributed`）、attention 后端实现（`cache-dit-attention`）。

## 1. 硬规则

1. **入口只有两个**：`python3 -m cache_dit.generate` 或 `cache-dit-generate`，参数完全一致。
2. **所有 `--xxx-params` 都是一个逗号分隔的 `k=v` 列表**，整体必须加引号；列表首项可以是裸 `type`，等价于 `type=<值>`：
   ```bash
   --cache-params 'DBCache,rdt=0.2'
   --cache-config 'type=DBCache,rdt=0.2'      # 同义
   ```
3. **每个 `--xxx-params` 都有 `--xxx-config` 别名**（`--quant-comm-params` / `--quant-comm-config` 是 `--quant-communicate-*` 的短名）。不要混用新旧逐项开关，它们已删除。
4. `--quantize-params` **本身就是量化开关**：除 `text_encoder` 外的任意键都会开启 transformer 量化；不传 `quant_type` 时默认 `float8_per_row`。
5. `--calibrator-params` **依赖** `--cache-params`；没有 cache 会直接报错。
6. `--parallel-params` 的轴度数是真实度数：`ulysses × ring × tp` 的乘积必须等于进程数。度数为 1 的轴不建组；只写 `vae=true` / `text=true` / `controlnet=true` / `ulysses_anything=true` 而不带一个 >1 的轴，是 **silent no-op**。
7. `--quant-communicate-params` 需要 `ulysses > 1`，否则报 `requires Ulysses parallelism`。
8. `--svdq-params` 只能配 `svdq_*` 类型的 `quant_type`；`lora_layout=row_major` 必须同时给 `runtime_kernel=cute`。
9. 进度条只在 **stdout/stderr 都是真实终端且 rank 0** 时显示；管道、日志文件和 notebook 输出不会再有逐行 `\r` 刷屏。
10. 未设置 `WORLD_SIZE` 时，CLI 根据 `--parallel-params` 或 `--parallel-config` 提供的多卡度数启动 worker，默认使用 `--launcher auto`。自启动要求可见 GPU 数等于度数乘积，可用 `CUDA_VISIBLE_DEVICES` 限定。多卡度数只写在 YAML 的 `parallelism_config` 中时，用 `torchrun` 等启动 worker。

## 2. 最小可用命令

```bash
# 列出所有已注册例子
python3 -m cache_dit.generate list

# 按注册名跑（FLUX.1-dev）
export FLUX_DIR=/path/to/FLUX.1-dev
python3 -m cache_dit.generate flux --save-path flux.base.png

# 按模型路径跑（自动匹配例子；歧义时加 --task）
python3 -m cache_dit.generate /path/to/FLUX.1-dev --task T2I --save-path flux.base.png
```

常用输出参数：`--steps/--num_inference_steps`、`--height`、`--width`、`--num-frames`、`--seed`、`--save-path`、`--repeat`、`--warmup`。

### 2.1 模型文件查找顺序（本机开发）

在本工作区开发时，按以下顺序查找模型权重，找到即停止：

1. `/workspace/dev/vipdev/hf_models/`
2. 当前工作区下的 `hf_models/` 目录
3. `$HF_MODELS` 指向的目录
4. `$HF_MODEL` 指向的目录

本地模型目录名通常与 Hugging Face repo 名一致，例如：

```bash
export FLUX_DIR=/workspace/dev/vipdev/hf_models/FLUX.1-dev
python3 -m cache_dit.generate flux --height 1024 --width 1024
```

不是所有模型都在本地；上述位置都没有目标模型时，先确认是否已下载，或改用 `--model-path` 指向实际目录，不要用不存在的路径硬跑。

## 3. 参数家族速查

| 目标 | flag / 别名 | 主要键 |
|---|---|---|
| 缓存 | `--cache-params` / `--cache-config` | `type=DBCache|SeaCache|SoftCache`、`Fn`、`Bn`、`rdt`、`ws`、`wi`、`mc`、`mcc`、`scm`、`steps_mask`、`seacache_*`、`softcache_*` |
| 校准器 | `--calibrator-params` / `--calibrator-config` | `type=taylorseer|dmd|foca|svd|hicache|brace|bezicast|residual_motion` + 各自键 |
| 量化 | `--quantize-params` / `--quantize-config` | `quant_type`、`backend`、`exclude_layers`、`text_encoder`、`regional_quantize`、`per_tensor_fallback`、`verbose`、`fp8_fused_mlp`、`fp4_fused_mlp` |
| SVDQuant | `--svdq-params` / `--svdq-config` | `runtime_kernel`、`lora_layout`、`smooth_strategy`、`few_shot_*`、`fused_mlp`、`layerwise_offload`、`packed_a4_*` |
| 注意力 | `--attn-params` / `--attn-config` | `type=native|flash|flash_varlen|sage|sage3|ffpa*|sol_attn|...` + 后端专属键 |
| 并行 | `--parallel-params` / `--parallel-config` | `ulysses`、`ring`、`tp`、`vae`、`text`、`controlnet`、`ulysses_anything`、`ulysses_async`、`ring_rotate_method`、`ring_convert_to_fp32` |
| 通信量化 | `--quant-communicate-params` / `--quant-communicate-config` | `bits=8|6|4`、`quant_type=per_token|per_group_16|per_group_32`、`quantize_k` |
| CPU offload | `--cpu-offload` / `--sequential-cpu-offload` / `--module-layerwise-cpu-offload`（`--layerwise-offload`）/ `--adaln-host-residency` / `--adaln-online` | 五者互斥 |
| 编译 | `--compile`、`--disable-compile-repeated-blocks`（`--no-regional-compile`）、`--cuda-graph`、`--compile-full-graph`、`--force-compile-dynamic`、`--max-autotune` | 另有 `--compile-vae`、`--compile-text-encoder`、`--compile-controlnet` |
| 输出/诊断 | `--cache-summary`、`--track-memory`、`--profile`、`--profile-activities`、`--profile-dir`、`--saved-fps` | `--profile-activities CPU GPU MEM` |

完整选项、默认值与别名见 [`references/options.md`](./references/options.md)；参数键的权威定义在 `src/cache_dit/cli/params.py` 与 `src/cache_dit/cli/cache.py`。

## 4. 已验证 recipe（FLUX.1-dev）

### 4.1 基线 / cache / compile

```bash
export FLUX_DIR=/path/to/FLUX.1-dev

python3 -m cache_dit.generate flux --save-path flux.base.png
python3 -m cache_dit.generate flux --compile --save-path flux.compile.png
python3 -m cache_dit.generate flux --cache-params 'DBCache,rdt=0.12' --compile \
  --save-path flux.cache.png
```

`rdt` 是 `residual_diff_threshold`，默认 0.12；蒸馏模型通常需要调高。`--compile` 默认只编译 repeated blocks。

### 4.2 Cache + 校准器

```bash
python3 -m cache_dit.generate flux \
  --cache-params 'DBCache' \
  --calibrator-params 'taylorseer,order=1' \
  --save-path flux.taylorseer.png
```

`--calibrator-params` 只能和 `--cache-params` 一起用。TaylorSeer 的 `order` 默认 1。

### 4.3 FP8 + compile

```bash
python3 -m cache_dit.generate flux \
  --quantize-params 'float8_per_row' \
  --compile \
  --save-path flux.fp8.png
```

`--quantize-params` 省略 `quant_type` 时就是 `float8_per_row`；`backend` 省略时 AUTO。FP8 per-row + compile 是单卡首选加速档。

### 4.4 Cache + 校准器 + FP8 + attention + compile（组合档）

```bash
python3 -m cache_dit.generate flux \
  --cache-config 'DBCache,rdt=0.12' \
  --calibrator-config 'taylorseer,order=1' \
  --quantize-config 'float8_per_row' \
  --attn-config 'native' \
  --compile \
  --save-path flux.unified.png
```

### 4.5 SVDQuant

```bash
# native GEMM + native layout（默认）
python3 -m cache_dit.generate flux \
  --quantize-params 'svdq_nvfp4_r32_dq' \
  --svdq-params 'lora_layout=native' \
  --save-path flux.svdq.native.png

# CuTe + native layout
python3 -m cache_dit.generate flux \
  --quantize-params 'svdq_nvfp4_r32_dq' \
  --svdq-params 'runtime_kernel=cute,lora_layout=native' \
  --save-path flux.svdq.cute.native.png

# CuTe + row_major layout（row_major 必须配 cute）
python3 -m cache_dit.generate flux \
  --quantize-params 'svdq_nvfp4_r32_dq' \
  --svdq-params 'runtime_kernel=cute,lora_layout=row_major' \
  --compile \
  --save-path flux.svdq.cute.row_major.png
```

三种合法 `runtime × layout` 组合：`native+native`、`cute+native`、`cute+row_major`；`flashinfer*` 固定 `native`。

### 4.6 双卡 Ulysses + FP8 通信量化

```bash
CUDA_VISIBLE_DEVICES=4,5 python3 -m cache_dit.generate flux \
  --parallel-params 'ulysses=2' \
  --quant-communicate-params 'bits=8' \
  --cache-params 'DBCache,rdt=0.12' \
  --calibrator-params 'taylorseer,order=1' \
  --quantize-params 'float8_per_row' \
  --attn-params 'native' \
  --compile \
  --save-path flux.ulysses2.png
```

`bits=8` 默认 `quant_type=per_group_16`、`quantize_k=true`。不需要 `torchrun`。

### 4.7 TP / TE-P / VAE-P / ControlNet-P

```bash
# TP2
python3 -m cache_dit.generate flux --parallel-params 'tp=2' --compile
# TP2 + 文本编码器并行
python3 -m cache_dit.generate flux --parallel-params 'tp=2,text=true'
# TP2 + VAE 并行
python3 -m cache_dit.generate flux --parallel-params 'tp=2,vae=true'
# 加上 ControlNet 并行（模型带 controlnet 时）
python3 -m cache_dit.generate flux --parallel-params 'tp=2,controlnet=true'
```

`vae/text/controlnet` 是组件修饰键，必须和 `tp`/`ulysses`/`ring` 中的一个 >1 轴写在同一个 `--parallel-params` 里。

MiniMax-H3 冷启动可用 `local_weight_shards=true`。四卡 TP2×CP2 加 TE-P 的命令见 [多卡示例](../../../docs/EXAMPLES.md#multi-gpu)。

### 4.8 Offload / CUDA Graph / steps mask

```bash
# 通用逐层 offload
python3 -m cache_dit.generate flux --module-layerwise-cpu-offload --compile
# CUDA Graph（和 compile 一起用）
python3 -m cache_dit.generate flux --compile --cuda-graph
# 跳步 mask：fast 策略 + static 执行
python3 -m cache_dit.generate flux --cache-params 'DBCache,steps_mask=true,scm=fast'
```

## 5. 关键交互与常见错误

| 现象 / 组合 | 规则 |
|---|---|
| `--calibrator-params` without `--cache-params` | 报错；先加 cache |
| `--parallel-params 'vae=true'` / `'text=true'` 单独使用 | no-op；补一个 `tp=N` / `ulysses=N` |
| `--parallel-params 'ulysses_anything=true'` 单独使用 | no-op；写成 `'ulysses=2,ulysses_anything=true'` |
| `--parallel-params 'tp=1,ulysses=1,ring=1'` | 只打印 "parallelism is not enabled"，不建组 |
| `--parallel-params` 度数乘积 ≠ 进程数 | 断言失败；度数必须等于 world size |
| `--quant-communicate-params` 无 Ulysses >1 | 报错 `requires Ulysses parallelism` |
| `--svdq-params` 但 `quant_type` 不是 `svdq_*` | 报错；先设 SVDQ 类型 |
| `lora_layout=row_major` 无 `runtime_kernel=cute` | 报错；补 `cute` |
| 同时给两个 offload 开关 | argparse 互斥，直接拒绝 |
| `--compile-full-graph` + cache/offload | 跳过/流式逻辑会引入 graph break，不要组合；用默认 `--compile` |
| `--adaln-host-residency` + `--cuda-graph`/`--compile-full-graph` | 未量化或未 TP 时被拒绝（AdaLN 每块拷贝是 graph break） |
| `--attn-params type=...` 取值非法 | 报 `invalid choice` / 未知 backend；查 `references/options.md` |
| 日志里没有进度条 | 非 TTY 环境（管道/日志/notebook）自动关闭；真实终端仍是一行刷新 |

## 6. 调试流程

```bash
python3 -m cache_dit.generate --help
python3 -m cache_dit.generate list
python3 -m cache_dit.generate flux --help 2>/dev/null || true   # 例子级参数以全局 help 为准
```

先用少步数做参数冒烟（不验证质量，只验证接线）：

```bash
python3 -m cache_dit.generate flux \
  --cache-params 'DBCache' \
  --calibrator-params 'taylorseer,order=1' \
  --quantize-params 'float8_per_row' \
  --attn-params 'native' \
  --steps 2 --height 1024 --width 1024 --warmup 0 \
  --save-path /tmp/generate-smoke.png
```

看输出末尾的 `Optimization:` 标签核对实际生效项，例如：

```text
C1_DBCache_F1B0_W8I1M0MC3_R0.2_CFG0_T1O1_Ulysses2_UF8K_G16_float8_per_row_native
```

标签依次编码 compile、cache、warmup/cached 步数、阈值、calibrator、并行、通信量化、量化类型和 attention；和命令行不一致时优先信标签。

## 7. 源码与维护

| 位置 | 职责 |
|---|---|
| `src/cache_dit/cli/entrypoint.py` | 全部 flag、别名、默认值；`--help` 的唯一来源 |
| `src/cache_dit/cli/params.py` | `--parallel-params` / `--attn-params` / `--cache-params` 的键与取值校验 |
| `src/cache_dit/cli/cache.py` | cache 与 calibrator 参数解析、`DBCacheConfig` 构建 |
| `src/cache_dit/cli/quantize.py` | `--quantize-params` / `--svdq-params` → `QuantizeConfig` |
| `src/cache_dit/cli/parallel.py` | 并行参数 → `ParallelismConfig`、组件并行模块 |
| `src/cache_dit/cli/attention.py` | attention backend 与 `set_attention_backend` |
| `src/cache_dit/cli/optimize.py` | cache/parallel/quantize 的应用顺序与 steps mask |
| `src/cache_dit/cli/compile.py` | compile / CUDA Graph / fullgraph |
| `src/cache_dit/cli/offload.py` | CPU offload 与 AdaLN 相关约束 |
| `src/cache_dit/cli/registers.py` | `Example.run`、进度条、计时与 summary |
| `docs/EXAMPLES.md` | 最小单卡/多卡命令；完整选项、默认值与别名的权威来源是 `--help` |

修改 CLI 后：

1. 同步更新 `docs/EXAMPLES.md` 的快速示例；
2. 同步本 skill 的 recipe 和 `references/options.md`；
3. 跑 `python3 -m cache_dit.generate --help` 和一条 tiny smoke；
4. 跑 `pytest tests/api/test_parallel_params.py tests/quantization/test_quantize_params.py tests/api/test_attn_params.py`。
