# Examples for Cache-DiT

|Z-Image-ControlNet| Context Parallel: Ulysses 2 |  Context Parallel: Ulysses 4 | + ControlNet Parallel |
|:---:|:---:|:---:|:---:|
|Base L20x1: 22s|15.7s|12.7s|**🚀7.71s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses2.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4_CNP.png" width=200px> |
| **+ Hybrid Cache** | **+ Torch Compile** | **+ Async Ulyess CP** | **+ FP8 All2All + CUDNN ATTN** | 
|**🚀6.85s**|6.45s|6.38s|**🚀6.19s, 5.47s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> |<img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_async_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_CNP_sdpa_cudnn.png" width=200px> 

## Installation

```bash
pip install -U uv # use uv for faster installation
uv pip install torch==2.11.0 torchvision torchaudio triton \
  transformers diffusers accelerate torchao opencv-python-headless \
  einops imageio-ffmpeg ftfy numpy
uv pip install -U cache-dit # stable release from PyPI.
```

## Available Examples

```bash
python3 -m cache_dit.generate list  # list all available examples

Available examples:
- ✅ flux_nunchaku                  - Default: nunchaku-tech/nunchaku-flux.1-dev
- ✅ flux                           - Default: black-forest-labs/FLUX.1-dev
- ✅ flux_fill                      - Default: black-forest-labs/FLUX.1-Fill-dev
- ✅ flux2                          - Default: black-forest-labs/FLUX.2-dev
- ✅ flux2_klein_base_9b            - Default: black-forest-labs/FLUX.2-klein-base-9B
- ✅ flux2_klein_base_4b            - Default: black-forest-labs/FLUX.2-klein-base-4B
- ✅ flux2_klein_9b                 - Default: black-forest-labs/FLUX.2-klein-9B
- ✅ flux2_klein_4b                 - Default: black-forest-labs/FLUX.2-klein-4B
- ✅ flux2_klein_base_9b_edit       - Default: black-forest-labs/FLUX.2-klein-base-9B
- ✅ flux2_klein_base_4b_edit       - Default: black-forest-labs/FLUX.2-klein-base-4B
- ✅ flux2_klein_9b_edit            - Default: black-forest-labs/FLUX.2-klein-9B
- ✅ flux2_klein_4b_edit            - Default: black-forest-labs/FLUX.2-klein-4B
- ✅ flux2_klein_9b_kv_edit         - Default: black-forest-labs/FLUX.2-klein-9b-kv
- ✅ qwen_image_lightning           - Default: lightx2v/Qwen-Image-Lightning
- ✅ qwen_image_2512                - Default: Qwen/Qwen-Image-2512
- ✅ qwen_image                     - Default: Qwen/Qwen-Image
- ✅ qwen_image_edit_2511_lightning - Default: lightx2v/Qwen-Image-Edit-2511-Lightning
- ✅ qwen_image_edit_2511           - Default: Qwen/Qwen-Image-Edit-2511
- ✅ qwen_image_edit_lightning      - Default: lightx2v/Qwen-Image-Lightning
- ✅ qwen_image_edit                - Default: Qwen/Qwen-Image-Edit-2509
- ✅ qwen_image_controlnet          - Default: InstantX/Qwen-Image-ControlNet-Inpainting
- ✅ qwen_image_layered             - Default: Qwen/Qwen-Image-Layered
- ✅ skyreels_v2                    - Default: Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
- ✅ ltx2_t2v                       - Default: Lightricks/LTX-2
- ✅ ltx2_i2v                       - Default: Lightricks/LTX-2
- ✅ wan_animate_2                  - Default: Wan-AI/Wan2.2-Animate-2-14B-Diffusers
- ✅ wan_animate_2_distilled        - Default: Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers
- ✅ wan2.2_t2v                     - Default: Wan-AI/Wan2.2-T2V-A14B-Diffusers
- ✅ wan2.1_t2v                     - Default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- ✅ wan2.2_i2v                     - Default: Wan-AI/Wan2.2-I2V-A14B-Diffusers
- ✅ wan2.1_i2v                     - Default: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
- ✅ wan2.2_vace                    - Default: linoyts/Wan2.2-VACE-Fun-14B-diffusers
- ✅ wan2.1_vace                    - Default: Wan-AI/Wan2.1-VACE-1.3B-diffusers
- ✅ ovis_image                     - Default: AIDC-AI/Ovis-Image-7B
- ✅ zimage_turbo_nunchaku          - Default: nunchaku/nunchaku-z-image-turbo
- ✅ zimage_turbo                   - Default: Tongyi-MAI/Z-Image-Turbo
- ✅ zimage                         - Default: Tongyi-MAI/Z-Image
- ✅ zimage_turbo_controlnet_2.0    - Default: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0
- ✅ zimage_turbo_controlnet_2.1    - Default: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- ✅ longcat_image                  - Default: meituan-longcat/LongCat-Image
- ✅ longcat_image_edit             - Default: meituan-longcat/LongCat-Image-Edit
- ✅ glm_image                      - Default: zai-org/GLM-Image
- ✅ glm_image_edit                 - Default: zai-org/GLM-Image
- ✅ firered_image_edit_1.0         - Default: FireRedTeam/FireRed-Image-Edit-1.0
- ✅ firered_image_edit_1.1         - Default: FireRedTeam/FireRed-Image-Edit-1.1
- ✅ helios_t2v                     - Default: BestWishYsh/Helios-Base
- ✅ heliost_t2v_distill            - Default: BestWishYsh/Helios-Distilled
```

## Single GPU Inference

The easiest way to enable hybrid cache acceleration for DiTs with cache-dit is to start with single GPU inference. For examples:  

```bash
# baseline
# use default model path, e.g, "black-forest-labs/FLUX.1-dev"
python3 -m cache_dit.generate flux 
python3 -m cache_dit.generate flux_nunchaku # need nunchaku library
python3 -m cache_dit.generate flux2
python3 -m cache_dit.generate ovis_image
python3 -m cache_dit.generate qwen_image_edit_lightning
python3 -m cache_dit.generate qwen_image
python3 -m cache_dit.generate ltx2_t2v --cache --cpu-offload
python3 -m cache_dit.generate ltx2_i2v --cache --cpu-offload
python3 -m cache_dit.generate skyreels_v2
python3 -m cache_dit.generate wan2.2
python3 -m cache_dit.generate zimage_turbo 
python3 -m cache_dit.generate zimage_turbo_nunchaku 
python3 -m cache_dit.generate zimage_turbo_controlnet_2.1 
python3 -m cache_dit.generate firered_image_edit_1.0
python3 -m cache_dit.generate generate longcat_image
python3 -m cache_dit.generate generate longcat_image_edit
# w/ cache acceleration
python3 -m cache_dit.generate flux --cache
python3 -m cache_dit.generate flux --cache --taylorseer
python3 -m cache_dit.generate flux_nunchaku --cache
python3 -m cache_dit.generate qwen_image --cache
python3 -m cache_dit.generate zimage_turbo --cache --rdt 0.6 --scm fast
python3 -m cache_dit.generate zimage_turbo_controlnet_2.1 --cache --rdt 0.6 --scm fast
# enable cpu offload or vae tiling if your encounter an OOM error
python3 -m cache_dit.generate qwen_image --cache --cpu-offload
python3 -m cache_dit.generate qwen_image --cache --cpu-offload --vae-tiling
python3 -m cache_dit.generate qwen_image_edit_lightning --cpu-offload --steps 4
python3 -m cache_dit.generate qwen_image_edit_lightning --cpu-offload --steps 8
# or, enable sequential cpu offload for extremly low VRAM device
python3 -m cache_dit.generate flux2 --sequential-cpu-offload # FLUX2 56B total
# use `--summary` option to show the cache acceleration stats
python3 -m cache_dit.generate zimage_turbo --cache --rdt 0.6 --scm fast --summary
```

## Custom Model Path

The default model path are the official model names on HuggingFace Hub. Users can set custom local model path by settig `--model-path`. For examples: 

```bash
python3 -m cache_dit.generate flux --model-path /PATH/TO/FLUX.1-dev
python3 -m cache_dit.generate zimage_turbo --model-path /PATH/TO/Z-Image-Turbo
python3 -m cache_dit.generate qwem_image --model-path /PATH/TO/Qwen-Image
```

## Distributed Inference 

cache-dit is designed to work seamlessly with CPU or Sequential Offloading, 🔥Context Parallelism, 🔥Tensor Parallelism. For examples:

```bash
# context parallelism or tensor parallelism
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses 
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ring 
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel usp # USP: Ulysses + Ring 
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel tp
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel ulysses_tp # Ulysses + TP
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel ring_tp  # Ring + TP
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel usp_tp # USP + TP
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo --parallel ulysses 
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo_controlnet_2.1 --parallel ulysses 
# ulysses anything attention
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo --parallel ulysses --ulysses-anything
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning --parallel ulysses --ulysses-anything
# text encoder parallelism: `--parallel-text-encoder` or `parallel-text`
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel tp --parallel-text
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning --parallel ulysses --ulysses-anything --parallel-text
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses 
torchrun --nproc_per_node=4 -m cache_dit.generate ltx2_t2v --parallel ulysses --parallel-vae --parallel-text --cache --ulysses-anything
torchrun --nproc_per_node=4 -m cache_dit.generate ltx2_t2v --parallel tp --parallel-vae --parallel-text --cache
torchrun --nproc_per_node=4 -m cache_dit.generate ltx2_i2v --parallel ulysses --parallel-vae --parallel-text --cache --ulysses-anything
torchrun --nproc_per_node=4 -m cache_dit.generate ltx2_i2v --parallel tp --parallel-vae --parallel-text --cache
```

> **MOSS-Image** ships its own pipeline, so it is vendored inside cache-dit
> (`cache_dit/models/pipelines/moss_image.py`) and loads with
> `--model-path <checkpoint root>`. Caching assumes one DiT forward per CFG branch: keep batch 1,
> `guidance_scale != 1` (unless `enable_separate_cfg=False`) and leave `slg_*` unset — the pipeline
> raises otherwise. Its joint image+text sequence is short, so Ulysses generally needs
> `--ulysses-anything` unless the token count divides by the CP size; Ring is not supported.

```bash
# single GPU: baseline + DBCache
python3 -m cache_dit.generate moss_image_sft --steps 25 --cache --rdt 0.3

# context parallelism (2 GPUs) / tensor parallelism (2 GPUs)
torchrun --nproc_per_node=2 -m cache_dit.generate moss_image_sft --parallel ulysses --ulysses-anything --cache
torchrun --nproc_per_node=2 -m cache_dit.generate moss_image_sft --parallel tp --parallel-text --cache

# VAE parallelism needs a latent larger than the VAE tile size (>=1536px output)
torchrun --nproc_per_node=2 -m cache_dit.generate moss_image_sft --height 1536 --width 1536 --parallel tp --parallel-vae

# hybrid CP + TP (4 GPUs, 2+2)
torchrun --nproc_per_node=4 -m cache_dit.generate moss_image_sft --parallel ulysses_tp --ulysses-anything --cache
```

> **LingBot-Video** (Dense 1.3B) also ships its own pipeline, vendored at
> `cache_dit/models/pipelines/lingbot_video/`; load it with `--model-path <checkpoint root>`. The
> DiT concatenates the video and text tokens into one joint sequence that all 24 blocks attend over,
> so caching pairs two forwards per step (cond + uncond, i.e. keep `guidance_scale > 1`) and
> `--ulysses-anything` is normally required because the joint length rarely divides by the CP size;
> Ring is not supported. Only the dense checkpoint is supported (MoE raises), and the pipeline
> conditions on its structured JSON caption form — `--num-frames 1` renders a single image. The
> native frame rate is 24 fps, so pass `--saved-fps 24` to write the clip at its real speed
> (`--saved-fps` defaults to 8). fp8-per-row runs, but the released setting (81 frames / 40 steps)
> is quantization sensitive: measured SSIM drops to ~0.69 against 0.93 for caching alone.

```bash
# single GPU: baseline + DBCache (official recipe: 480x832, 81 frames, 40 steps, CFG 3.0)
python3 -m cache_dit.generate lingbot_video --cache --rdt 0.3 --Fn 1 --Bn 0 --saved-fps 24

# context parallelism (2 GPUs) / tensor parallelism (2 GPUs)
torchrun --nproc_per_node=2 -m cache_dit.generate lingbot_video --parallel ulysses --ulysses-anything --cache
torchrun --nproc_per_node=2 -m cache_dit.generate lingbot_video --parallel tp --parallel-text --cache

# VAE parallelism + quantization
torchrun --nproc_per_node=2 -m cache_dit.generate lingbot_video --parallel tp --parallel-vae
python3 -m cache_dit.generate lingbot_video --float8-per-row --compile --cache

# cheaper functional check (short clip, small canvas)
python3 -m cache_dit.generate lingbot_video --num-frames 9 --height 256 --width 256 --steps 20 --cache
```

> **CausVid-Wan** (`tianweiy/CausVid`, a Wan2.1-T2V-1.3B causal few-step student) is vendored at
> `cache_dit/models/pipelines/causvid/`. Point `WAN_DIR` at the `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
> tree (text encoder / tokenizer / VAE) and `CAUSVID_DIR` at the `tianweiy/CausVid` clone (the
> `autoregressive_checkpoint/model.pt` student is converted on load, `strict=True`). The model is
> chunk-autoregressive: 81 frames = 7 chunks of 3 latent frames, each chunk runs the fixed DMD table
> `1000/757/522` and then a clean-latent pass that writes the KV cache every later chunk attends to,
> so `--steps` is ignored and there is no CFG. `--cache` only becomes effective above
> `--rdt 0.95` because consecutive DMD steps are far apart (measured residual diffs 0.96-1.20), and
> at that point the clip changes; TP, Ulysses CP (head-sharded KV, Ring refused), TE-P, VAE-P,
> fp8/SVDQ, layerwise offload and compile all run — measured 1.04-1.27x, see the PR for the matrix.

```bash
export WAN_DIR=/path/to/Wan2.1-T2V-1.3B-Diffusers
export CAUSVID_DIR=/path/to/CausVid

# single GPU baseline (480x832, 81 frames, 3 DMD steps)
python3 -m cache_dit.generate causvid_wan --compile

# attention is ~58% of the FLOPs, so fp8 attention pays off before weight quantization does
python3 -m cache_dit.generate causvid_wan --compile --attn ffpa_fp8

# Ulysses context parallelism (2 GPUs): the KV cache is partitioned by head
torchrun --nproc_per_node=2 -m cache_dit.generate causvid_wan --parallel ulysses --compile

# tensor parallelism / TE-P / VAE-P
torchrun --nproc_per_node=2 -m cache_dit.generate causvid_wan --parallel tp --parallel-text --parallel-vae
```

## Low-bits Quantization 

cache-dit is designed to work seamlessly with torch.compile, Quantization (🔥torchao, 🔥nunchaku), For examples:

```bash
# please also enable torch.compile if the quantation is using.
python3 -m cache_dit.generate flux --cache --quantize-type float8 --compile
python3 -m cache_dit.generate flux --cache --quantize-type int8 --compile
python3 -m cache_dit.generate flux --cache --quantize-type float8_weight_only --compile
python3 -m cache_dit.generate flux --cache --quantize-type int8_weight_only --compile
python3 -m cache_dit.generate flux --cache --quantize-type bnb_4bit --compile # w4a16
python3 -m cache_dit.generate flux_nunchaku --cache --compile # w4a4 SVDQ
```

## Hybrid Acceleration 

Here are some examples for `hybrid cache acceleration + parallelism` for popular DiTs with cache-dit.

```bash
# DBCache + SCM + Taylorseer
python3 -m cache_dit.generate flux --cache --scm fast --taylorsees --taylorseer-order 1
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + CUDNN Attention (--attn _sdpa_cudnn)
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --ulysses-float8 \
         --attn _sdpa_cudnn --parallel-text --cache --scm fast --taylorseer \
         --taylorseer-order 1 --quantize-type float8 --warmup 2 --repeat 5 --compile 
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + FP8 SageAttention (--attn sage)
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --ulysses-float8 \
         --attn sage --parallel-text --cache --scm fast --taylorseer \
         --taylorseer-order 1 --quantize-type float8 --warmup 2 --repeat 5 --compile 
# Case: Hybrid Acceleration for Qwen-Image-Edit-Lightning, tracking memory usage.
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning \
         --parallel ulysses --ulysses-anything --parallel-text \
         --quantize-type float8_weight_only --steps 4 --track-memory --compile
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning \
         --parallel tp --parallel-text --quantize-type float8_weight_only \
         --steps 4 --track-memory --compile
# Case: Hybrid Acceleration + Context Parallelism + ControlNet Parallelism, e.g, Z-Image-ControlNet
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --rdt 0.6 --scm fast
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --scm fast --rdt 0.6 --compile \
         --compile-controlnet --ulysses-float8 --attn _sdpa_cudnn \
         --warmup 2 --repeat 4     
```

## End2End Examples

```bash
# NO Cache Acceleration: 8.27s
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses

INFO 12-17 09:02:31 [base.py:151] Example Input Summary:
INFO 12-17 09:02:31 [base.py:151] - prompt: A cat holding a sign that says hello world
INFO 12-17 09:02:31 [base.py:151] - height: 1024
INFO 12-17 09:02:31 [base.py:151] - width: 1024
INFO 12-17 09:02:31 [base.py:151] - num_inference_steps: 28
INFO 12-17 09:02:31 [base.py:214] Example Output Summary:
INFO 12-17 09:02:31 [base.py:225] - Model: flux
INFO 12-17 09:02:31 [base.py:225] - Optimization: C0_Q0_NONE_Ulysses4
INFO 12-17 09:02:31 [base.py:225] - Load Time: 0.79s
INFO 12-17 09:02:31 [base.py:225] - Warmup Time: 21.09s
INFO 12-17 09:02:31 [base.py:225] - Inference Time: 8.27s
INFO 12-17 09:02:32 [base.py:182] Image saved to flux.1024x1024.C0_Q0_NONE_Ulysses4.png

# Enabled Cache Acceleration: 4.23s
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --cache --scm fast

INFO 12-17 09:10:09 [base.py:151] Example Input Summary:
INFO 12-17 09:10:09 [base.py:151] - prompt: A cat holding a sign that says hello world
INFO 12-17 09:10:09 [base.py:151] - height: 1024
INFO 12-17 09:10:09 [base.py:151] - width: 1024
INFO 12-17 09:10:09 [base.py:151] - num_inference_steps: 28
INFO 12-17 09:10:09 [base.py:214] Example Output Summary:
INFO 12-17 09:10:09 [base.py:225] - Model: flux
INFO 12-17 09:10:09 [base.py:225] - Optimization: C0_Q0_DBCache_F1B0_W8I1M0MC3_R0.24_CFG0_T0O0_Ulysses4_S15
INFO 12-17 09:10:09 [base.py:225] - Load Time: 0.78s
INFO 12-17 09:10:09 [base.py:225] - Warmup Time: 18.49s
INFO 12-17 09:10:09 [base.py:225] - Inference Time: 4.23s
INFO 12-17 09:10:09 [base.py:182] Image saved to flux.1024x1024.C0_Q0_DBCache_F1B0_W8I1M0MC3_R0.24_CFG0_T0O0_Ulysses4_S15.png
```

|NO Cache Acceleration: 8.27s| w/ Cache Acceleration: 4.23s|
|:---:|:---:|
|![](https://github.com/vipshop/cache-dit/raw/main/examples/assets/flux.1024x1024.C0_Q0_NONE_Ulysses4.png)|![](https://github.com/vipshop/cache-dit/raw/main/examples/assets/flux.1024x1024.C0_Q0_DBCache_F1B0_W8I1M0MC3_R0.24_CFG0_T0O0_Ulysses4_S15.png)|

## How to Add New Example

It is very easy to add a new example. Please refer to the specific implementation in [examples.py](https://github.com/vipshop/cache-dit/raw/main/src/_utils/examples.py). For example:

```python
@ExampleRegister.register("flux")
def flux_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import FluxPipeline

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=_path("black-forest-labs/FLUX.1-dev"),
            pipeline_class=FluxPipeline,
            # `text_encoder_2` will be quantized when `--quantize-type` 
            # is set to `bnb_4bit`.
            bnb_4bit_components=["text_encoder_2"],
        ),
        input_data=ExampleInputData(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=28,
        ),
    )

# NOTE: DON'T forget to import this `flux_example` into __init__.py
```

## More Usages about Examples

```bash
python3 -m cache_dit.generate --help

positional arguments:
  {generate,list,flux_nunchaku,flux,flux_fill,flux2,flux2_klein_base_9b,flux2_klein_base_4b,flux2_klein_9b,flux2_klein_4b,qwen_image_lightning,qwen_image_2512,qwen_image,qwen_image_edit_2511_lightning,qwen_image_edit_2511,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,qwen_image_layered,skyreels_v2,ltx2_t2v,ltx2_i2v,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage_nunchaku,zimage,zimage_controlnet_2.0,zimage_controlnet_2.1,longcat_image,longcat_image_edit}
                        The task to perform or example name to run. Use 'list' to list all available examples, or specify an example name directly (defaults to 'generate' task).
  {None,flux_nunchaku,flux,flux_fill,flux2,flux2_klein_base_9b,flux2_klein_base_4b,flux2_klein_9b,flux2_klein_4b,qwen_image_lightning,qwen_image_2512,qwen_image,qwen_image_edit_2511_lightning,qwen_image_edit_2511,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,qwen_image_layered,skyreels_v2,ltx2_t2v,ltx2_i2v,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage_nunchaku,zimage,zimage_controlnet_2.0,zimage_controlnet_2.1,longcat_image,longcat_image_edit}
                        Names of the examples to run. If not specified, skip running example.

options:
  -h, --help            show this help message and exit
  --model-path MODEL_PATH
                        Override model path if provided
  --controlnet-path CONTROLNET_PATH
                        Override controlnet model path if provided
  --lora-path LORA_PATH
                        Override lora model path if provided
  --transformer-path TRANSFORMER_PATH
                        Override transformer model path if provided
  --image-path IMAGE_PATH
                        Override image path if provided
  --mask-image-path MASK_IMAGE_PATH
                        Override mask image path if provided
  --config-path CONFIG_PATH, --config CONFIG_PATH
                        Path to CacheDiT configuration YAML file
  --prompt PROMPT       Override default prompt if provided
  --negative-prompt NEGATIVE_PROMPT
                        Override default negative prompt if provided
  --skip-negative_prompt, --skip-neg
                        Force skip negative prompt even if negative prompt is provided.
  --num_inference_steps NUM_INFERENCE_STEPS, --steps NUM_INFERENCE_STEPS
                        Number of inference steps
  --warmup WARMUP       Number of warmup steps before measuring performance
  --warmup-num-inference-steps WARMUP_NUM_INFERENCE_STEPS, --warmup-steps WARMUP_NUM_INFERENCE_STEPS
                        Number of warmup inference steps per warmup before measuring performance
    --warmup-seed WARMUP_SEED
                                                Optional seed used only for warmup forwards. When set, warmup uses this
                                                seed while formal repeated inference still uses --seed.
    --warmup-prompt WARMUP_PROMPT
                                                Optional prompt used only for warmup forwards. When set, warmup uses this
                                                prompt while formal repeated inference still uses --prompt.
  --repeat REPEAT       Number of times to repeat the inference for performance measurement
  --height HEIGHT       Height of the generated image
  --width WIDTH         Width of the generated image
  --input-height INPUT_HEIGHT
                        Height of the input image
  --input-width INPUT_WIDTH
                        Width of the input image
  --seed SEED           Random seed for reproducibility
  --num-frames NUM_FRAMES, --frames NUM_FRAMES
                        Number of frames to generate for video
  --save-path SAVE_PATH
                        Path to save the generated output, e.g., output.png or output.mp4
  --cache               Enable Cache Acceleration
  --cache-summary, --summary
                        Enable Cache Summary logging
  --Fn-compute-blocks FN_COMPUTE_BLOCKS, --Fn FN_COMPUTE_BLOCKS
                        CacheDiT Fn_compute_blocks parameter
  --Bn-compute-blocks BN_COMPUTE_BLOCKS, --Bn BN_COMPUTE_BLOCKS
                        CacheDiT Bn_compute_blocks parameter
  --residual-diff-threshold RESIDUAL_DIFF_THRESHOLD, --rdt RESIDUAL_DIFF_THRESHOLD
                        CacheDiT residual diff threshold
  --max-warmup-steps MAX_WARMUP_STEPS, --ws MAX_WARMUP_STEPS
                        Maximum warmup steps for CacheDiT
  --warmup-interval WARMUP_INTERVAL, --wi WARMUP_INTERVAL
                        Warmup interval for CacheDiT
  --max-cached-steps MAX_CACHED_STEPS, --mc MAX_CACHED_STEPS
                        Maximum cached steps for CacheDiT
  --max-continuous-cached-steps MAX_CONTINUOUS_CACHED_STEPS, --mcc MAX_CONTINUOUS_CACHED_STEPS
                        Maximum continuous cached steps for CacheDiT
  --taylorseer          Enable TaylorSeer for CacheDiT
  --taylorseer-order TAYLORSEER_ORDER, -order TAYLORSEER_ORDER
                        TaylorSeer order
  --steps-mask          Enable steps mask for CacheDiT
  --mask-policy {None,slow,s,medium,m,fast,f,ultra,u}, --scm {None,slow,s,medium,m,fast,f,ultra,u}
                        Pre-defined steps computation mask policy
  --quantize, --q       Enable quantization for transformer
  --disable-per-row, --no-per-row
                        Disable per row quantization for transformer
  --quantize-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}, --q-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}
  --disable-regional-quantize, --disable-regional, --no-regional
                        Disable quantization for repeated blocks in transformer
  --disable-per-tensor-fallback, --no-per-tensor-fallback
                        Disable (float8 only) per-tensor fallback quantization for transformer
  --float8-per-row, --float8
                        Enable float8 per-row quantization for transformer
  --float8-per-tensor   Enable float8 per-tensor quantization for transformer
  --float8-per-block    Enable float8 per-block quantization for transformer
  --float8-weight-only, --float8-wo
                        Enable float8 weight-only quantization for transformer
  --float8-blockwise, --float8-bw
                        Enable float8 blockwise quantization for transformer
  --int8-per-row, --int8
                        Enable int8 per-row quantization for transformer
  --int8-per-tensor     Enable int8 per-tensor quantization for transformer
  --int8-weight-only, --int8-wo
                        Enable int8 weight-only quantization for transformer
  --int4-weight-only, --int4-wo
                        Enable int4 weight-only quantization for transformer
  --quantize-text-encoder, --q-text
                        Enable quantization for text encoder
  --quantize-text-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}, --q-text-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}
  --quantize-controlnet, --q-controlnet
                        Enable quantization for ControlNet
  --quantize-controlnet-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}, --q-controlnet-type {None,float8_per_row,float8_per_tensor,float8_per_block,float8_weight_only,int8_per_row,int8_per_tensor,int8_weight_only,int4_weight_only,bitsandbytes_4bit}
  --quantize-verbose, --q-verbose
                        Print the verbose logs of the quantization process
  --parallel-type {None,tp,ulysses,ring,usp,ulysses_tp,ring_tp,tp_ulysses,tp_ring,usp_tp}, --parallel {None,tp,ulysses,ring,usp,ulysses_tp,ring_tp,tp_ulysses,tp_ring,usp_tp}
  --parallel-vae        Enable VAE parallelism if applicable.
  --parallel-text-encoder, --parallel-text
                        Enable text encoder parallelism if applicable.
  --parallel-controlnet
                        Enable ControlNet parallelism if applicable.
  --attn {None,flash,_flash_3,native,_native_cudnn,_sdpa_cudnn,sage,_native_npu,_npu_fia}
  --ulysses-anything, --uaa
                        Enable Ulysses Anything Attention for context parallelism
  --ulysses-float8, --ufp8
                        Enable Ulysses Attention/UAA Float8 for context parallelism
  --ulysses-async, --uaqkv
                        Enabled experimental Async QKV Projection with Ulysses for context parallelism
  --ring-rotate-method {allgather,p2p}, --rotate {allgather,p2p}
                        Ring Attention rotation method for context parallelism
  --ring-no-convert-to-fp32, --ring-no-fp32, --no-fp32
                        Disable convert Ring Attention output and lse to fp32 for context parallelism
  --cpu-offload, --cpu-offload-model
                        Enable CPU offload for model if applicable.
  --sequential-cpu-offload
                        Enable sequential GPU offload for model if applicable.
  --device-map-balance, --device-map
                        Enable automatic device map balancing model if multiple GPUs are available.
  --vae-tiling          Enable VAE tiling for low memory device.
  --vae-slicing         Enable VAE slicing for low memory device.
  --compile             Enable compile for transformer, only compile the repeated blocks by default.
  --disable-compile-repeated-blocks, --disable-compile-blocks
                        Disable compile for repeated blocks in transformer
  --force-compile-dynamic
                        Force set the compiled transformer to dynamic mode.
  --cuda-graph          Enable compile with CUDA Graph for transformer if applicable.
  --compile-vae         Enable compile for VAE
  --compile-text-encoder, --compile-text
                        Enable compile for text encoder
  --compile-controlnet  Enable compile for ControlNet
  --max-autotune, --tune
                        Enable max-autotune mode for torch.compile
  --track-memory, --mem
                        Track and report peak GPU memory usage
  --profile             Enable profiling with torch.profiler
  --profile-name PROFILE_NAME
                        Name for the profiling session
  --profile-dir PROFILE_DIR
                        Directory to save profiling results
  --profile-activities {CPU,GPU,MEM} [{CPU,GPU,MEM} ...]
                        Activities to profile (CPU, GPU, MEM)
  --profile-with-stack  profile with stack for better traceability
  --profile-record-shapes
                        profile record shapes for better analysis
  --disable-fuse-lora DISABLE_FUSE_LORA
                        Disable fuse_lora even if lora weights are provided.
  --generator-device GENERATOR_DEVICE, --gen-device GENERATOR_DEVICE
                        Device for torch.Generator, e.g., 'cuda' or 'cpu'. Defaults to the current rank's accelerator, falling back to CPU when there is none. CPU sampling is not reproducible across torch versions: torch 2.12 changed the bf16 `randn` kernel.
  --saved-fps SAVED_FPS, --fps SAVED_FPS
                        Export generated video with specified fps
  --example-summary, --esummary
                        Enable example summary logging
```
