# planner 清单与新增流程

注册表在 `transformers/register.py`；所有内置 planner 的 import 集中在 `transformers/planners.py::_activate_cp_planners` / `_activate_tp_planners`（懒激活，首次 `parallelize_transformer` 时触发）。

## 1. 查某个模型用哪个 planner

```python
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister, TensorParallelismPlannerRegister,
)

_activate_cp_planners()
ContextParallelismPlannerRegister.get_planner("WanS2VTransformer3DModel")   # → 类
ContextParallelismPlannerRegister.supported_planners()                       # → (数量, 名字列表)
```

匹配规则：`class_name.startswith(注册名)`，**遍历顺序 = 注册顺序**，先命中者胜。因此短前缀必须后注册（`Wan` 在 `WanS2V` / `WanAnimate2` 之后，`QwenImage` 在 `QwenImage21` 之后）。

## 2. 清单（按文件；CP / TP 可能注册在别的文件里）

| 文件 | CP 注册名 | TP 注册名 |
|---|---|---|
| `anyflow.py` | `AnyFlowTransformer3D` | `AnyFlow` |
| `boogu_image.py` | `BooguImageTransformer` | `BooguImageTransformer` |
| `bria_fibo.py` | `BriaFiboTransformer2DModel` | 同名 |
| `bria_fibo2.py` | `BriaFibo2Transformer2DModel` | 同名 |
| `causvid.py` | `CausVidTransformer3D` | 同名 |
| `chroma.py` | `Chroma` | —（TP 在 `flux.py`） |
| `chrono_edit.py` | `ChronoEditTransformer3D` | —（TP 在 `wan.py`：`ChronoEdit`） |
| `cogvideox.py` | `CogVideoX` | —（TP 在 `cogview.py`） |
| `cogview.py` | `CogView3Plus`、`CogView4` | `ConsisID`、`CogView3Plus`、`CogView4`、`CogVideoX` |
| `cosisid.py` | `ConsisID` | — |
| `cosmos.py` | `CosmosTransformer3DModel` | 同名 |
| `dit.py` | `DiT` | — |
| `ernie_image.py` | `ErnieImage` | 同名 |
| `fast_h3.py` | `FastH3Transformer3DModel` | 同名 |
| `flux.py` | `FluxTransformer2DModel` | `Chroma`、`HunyuanImage`、`HunyuanVideo`、`FluxTransformer` |
| `flux2.py` | `Flux2Transformer2DModel`、`_Flux2KleinTransformer2DModel` | 同名 |
| `flux3.py` | `Flux3Transformer2DModel`、`_Flux3KleinTransformer2DModel` | 同名 |
| `glm_image.py` | — | `GlmImageTransformer2DModel` |
| `helios.py` | `HeliosTransformer3DModel` | 同名 |
| `hidream_o1_image.py` | `HiDreamO1ImageTransformer2DModel` | 同名 |
| `hunyuan.py` | `HunyuanImage`、`HunyuanVideo` | — |
| `hunyuan_dit.py` | — | `HunyuanDiT` |
| `hy_nomas.py` | `Nomas` | `Nomas` |
| `ideogram4.py` | `Ideogram4Transformer2DModel` | 同名 |
| `joy_image.py` | `JoyImageEditTransformer` | 同名 |
| `kandinsky.py` | `Kandinsky5` | —（TP 在 `kandinsky5.py`） |
| `kandinsky5.py` | — | `Kandinsky5` |
| `kandinsky6.py` | `Kandinsky6Transformer3DModel` | 同名 |
| `krea2.py` | `Krea2Transformer2DModel` | `Krea2` |
| `lingbot_video.py` | `LingBotVideoTransformer` | 同名 |
| `longcat_audio_dit.py` | `LongCatAudioDiTTransformer` | 同名 |
| `longcat_image.py` | `LongCatImageTransformer2DModel` | 同名 |
| `longcat_video_avatar.py` | `LongCatVideoAvatarTransformer3DModel` | 同名 |
| `longlive2.py` | `LongLive2` | `LongLive2` |
| `ltx2.py` | `LTX2` | —（TP 在 `ltx2_video.py`） |
| `ltx2_video.py` | — | `LTX2` |
| `ltx_video.py` | — | `LTXVideo` |
| `ltxvideo.py` | `LTXVideo` | — |
| `magi2.py` | `Magi2TransformerBlock` | 同名 |
| `minimax_h3.py` | `MiniMaxH3Transformer3DModel` | 同名 |
| `minimax_h3_image.py` | `MiniMaxH3ImageTransformer2DModel` | 同名 |
| `mochi.py` | — | `Mochi` |
| `moss_image.py` | `MOSSImageTransformer` | 同名 |
| `motif_video.py` | `MotifVideoTransformer3DModel` | 同名 |
| `mova.py` | `MOVAStack` | `MOVAStack` |
| `nucleus_moe.py` | — | `NucleusMoEImageTransformer2DModel` |
| `nunchaku.py` | `NunchakuFlux`、`NunchakuQwenImage`、`NunchakuZImageTransformer2DModel` | — |
| `ovis_image.py` | `OvisImageTransformer2DModel` | `OvisImage` |
| `pixart.py` | `PixArt` | `PixArt` |
| `prx.py` | `PRXTransformer2DModel` | 同名 |
| `qwen_image.py` | `QwenImage` | `QwenImageTransformer2DModel` |
| `qwen_image21.py` | `QwenImage21`、`_QwenImage21` | `QwenImage21Transformer2DModel`、`_QwenImage21Transformer2DModel` |
| `sana_video.py` | `SanaVideoTransformer3DModel` | 同名 |
| `sana_video2.py` | `SanaVideo2Transformer3DModel` | 同名 |
| `sana_wm.py` | `SanaWM` | `SanaWM` |
| `skyreels.py` | `SkyReelsV2` | `SkyReelsV2` |
| `stable_audio_3.py` | `StableAudio3DiTModel` | 同名 |
| `wan.py` | `WanTransformer3D`、`WanVACETransformer3D` | `ChronoEdit`、`Wan` |
| `wan_animate_2.py` | `WanAnimate2` | `WanAnimate2` |
| `wan_s2v.py` | `WanS2V` | `WanS2V` |
| `zimage.py` | `ZImageTransformer2DModel`、`MingImageTransformer2DModel` | `MingImage`、`Lumina2`、`ZImage` |

只有 CP、没有 TP 的模型（`—` 列）：chroma、chrono_edit、cogvideox、cosisid、dit、hunyuan、kandinsky、ltx2、ltxvideo、nunchaku。只有 TP、没有 CP：glm_image、hunyuan_dit、kandinsky5、ltx2_video、ltx_video、mochi、nucleus_moe。

async Ulysses（`transformers/async_ulysses/`）：flux、flux2、qwen_image、zimage、zimage_controlnet、longcat_image、ovis_image、controlnets。

TE-P（`text_encoders/`）：T5、UMT5、Qwen2.5-VL、Qwen3 / Qwen3-VL / Qwen3.5、Llama、Mistral、Gemma 系列、Glm / Glm4 / GlmImage、SmolLM3、MingImageMLLM。

VAE-P（`autoencoders/`）：AutoencoderKL、KLWan、KLHunyuanVideo、KLFlux2、KLLTX2Video、KLQwenImage（含 21）、KLMiniMaxH3（含 image）、KLPH3Image。

## 3. plan 的五种形态（决定新模型该抄谁）

| 形态 | 例子 | 特征 |
|---|---|---|
| 单流图像 DiT | FLUX（`flux.py`） | 根 `""` 切 `hidden_states`/`encoder_hidden_states`（有时含 `img_ids`/`txt_ids`），`proj_out` gather |
| 视频 3D DiT | Wan（`wan.py`） | 切 `hidden_states`（dim=1 帧序列）+ `time`/`rope`，注意 rope 是 **split_output**；`blocks.0` 才切隐状态 |
| 双流 / 多流 | Boogu、BriaFibo2、Ideogram4 | 每个流单独切，输出用 `None` 占位跳过；`_cp_requires_min_bn_blocks` 常见 |
| 音频 / 音视频 | StableAudio3、LongCatAudioDiT、MOVA、Kandinsky6 | 非 3D 张量按 2D 切（dim=1 是时间/帧），注意 audio 流与 video 流的 dim 不同 |
| 带外部注入栈 | WanS2V（audio inject）、Ming（direct tokens） | 注入发生在 `rope` 之后 → 必须切 `rope` 的输入而非输出，否则会被重新物化 |

## 4. 新增 planner 的 runbook

1. 新建 `transformers/<model>.py`，继承 `ContextParallelismPlanner` / `TensorParallelismPlanner`，用 `@...Register.register("类名前缀")` 注册。
2. 在 `transformers/planners.py` 的 `_activate_cp_planners` / `_activate_tp_planners` 里加 `_safe_import`，**注意插入位置**（前缀遮蔽）。
3. 需要的话在 `_apply` 里设 `transformer._cp_requires_min_bn_blocks`（尾部 gather 的模型）、调 `AsyncUlyssesRegistry.apply(transformer)`。
4. TP planner 记得 `shard_div_attr(block.attn, "heads", tp_mesh.size())`，并把 `layer_plan` 返回给基类 `record_plans`。
5. 测试两件套：
   - `tests/parallelism/test_<model>_cp_plan.py`：CPU-only，断言 plan 字典的 split/gather 标记与 TP 层 plan（"字符串契约"，形状对不代表切点对）。
   - `tests/parallelism/test_<model>_cp_tp_tiny.py`：tiny 模型 + `mp.spawn`，与单卡比数值；端口用 `socket.bind(("127.0.0.1", 0))`，`try/finally: destroy_process_group()`。
6. VAE/TE 有新结构时分别加 `autoencoders/autoencoder_kl_<name>.py` / `text_encoders/<name>.py`，并在对应 `planners.py` 激活；VAE 多 tile decode 用 `TileSafeP2PComm`。
7. 文档：`docs/user_guide/CONTEXT_PARALLEL.md` 的能力表，以及 `cache-dit-model-integration` skill 的矩阵里对应行的 C/P 列。

## 5. 现有测试地图

| 测试 | 覆盖 |
|---|---|
| `test_context_parallel_core.py` / `test_context_parallel_entrypoint.py` / `test_cp_hook_robustness.py` | plan→hook、split/gather 语义、异常路径 |
| `test_distributed_primitives.py` / `test_fN_fused_qkv_a2a.py` / `test_bf16_a2a_alignment.py` / `test_fp8_a2a_alignment.py` / `test_ulysses_gqa_multirank.py` | all-to-all 与量化通信 |
| `test_templated_usp.py` / `test_ring_buffer_reuse.py` | USP / Ring 的合并与双缓冲 |
| `test_*_cp_plan.py` | planner 逐模型字符串契约（CPU） |
| `test_*_cp_tp_tiny.py` / `test_*_cp_tiny.py` / `test_*_tp_tiny.py` | 多卡数值等价 |
| `test_*_vae_dp*.py` / `test_vae_p2p_order.py` | VAE-P 与 tile 协议 |
| `test_ming_image_te_p_tiny.py` | TE-P |
| `test_topology.py` / `test_tp_utils_divisible_attr.py` | NUMA 解析、`shard_div_attr` 报错信息 |
| `test_async_ulysses_registry.py` | async Ulysses 的补丁注册与撤销 |
