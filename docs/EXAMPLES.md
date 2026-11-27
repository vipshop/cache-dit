# Examples for Cache-DiT

|Z-Image-ControlNet| Context Parallel: Ulysses 2 |  Context Parallel: Ulysses 4 | + ControlNet Parallel |
|:---:|:---:|:---:|:---:|
|Base L20x1: 22s|15.7s|12.7s|**🚀7.71s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses2.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4_CNP.png" width=200px> |
| **+ Hybrid Cache** | **+ Torch Compile** | **+ Async Ulyess CP** | **+ FP8 All2All + CUDNN ATTN** | 
|**🚀6.85s**|6.45s|6.38s|**🚀6.19s, 5.47s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> |<img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_async_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_CNP_sdpa_cudnn.png" width=200px> 

## Table of Contents

- [Installation](#-installation)
- [Available Examples](#-available-examples)
- [Single GPU Inference](#-single-gpu-inference)  
- [Custom Model Path](#-custom-model-path)  
- [Distributed Inference](#-multi-gpu-inference)  
- [Low-bits Quantization](#-low-bits-quantization)  
- [Hybrid Acceleration](#-hybrid-acceleration) 
- [End2End Examples](#-end2end-examples) 
- [How to Add New Example](#-how-to-add-new-example) 
- [More Usages about Examples](#-more-usages-about-examples) 

## Installation

```bash
pip install torch==2.11.0 torchvision torchaudio triton \
  transformers diffusers accelerate torchao opencv-python-headless \
  einops imageio-ffmpeg ftfy numpy
pip install -U cache-dit # stable release from PyPI.
```

## Available Examples

```bash
python3 -m cache_dit.generate list  # list all available examples

Available examples:
- ✅ flux_nunchaku                    - Default: nunchaku-tech/nunchaku-flux.1-dev
- ✅ flux                             - Default: black-forest-labs/FLUX.1-dev
- ✅ flux_fill                        - Default: black-forest-labs/FLUX.1-Fill-dev
- ✅ flux2                            - Default: black-forest-labs/FLUX.2-dev
- ✅ flux2_klein_base_9b              - Default: black-forest-labs/FLUX.2-klein-base-9B
- ✅ flux2_klein_base_4b              - Default: black-forest-labs/FLUX.2-klein-base-4B
- ✅ flux2_klein_9b                   - Default: black-forest-labs/FLUX.2-klein-9B
- ✅ flux2_klein_4b                   - Default: black-forest-labs/FLUX.2-klein-4B
- ✅ flux2_klein_base_9b_edit         - Default: black-forest-labs/FLUX.2-klein-base-9B
- ✅ flux2_klein_base_4b_edit         - Default: black-forest-labs/FLUX.2-klein-base-4B
- ✅ flux2_klein_9b_edit              - Default: black-forest-labs/FLUX.2-klein-9B
- ✅ flux2_klein_4b_edit              - Default: black-forest-labs/FLUX.2-klein-4B
- ✅ flux3                            - Default: black-forest-labs/flux-3-image-klein
- ✅ flux3_fused                      - Default: black-forest-labs/flux-3-image-klein
- ✅ flux3_edit                       - Default: black-forest-labs/flux-3-image-klein
- ✅ flux2_klein_9b_kv_edit           - Default: black-forest-labs/FLUX.2-klein-9b-kv
- ✅ qwen_image_lightning             - Default: lightx2v/Qwen-Image-Lightning
- ✅ qwen_image_2512                  - Default: Qwen/Qwen-Image-2512
- ✅ qwen_image                       - Default: Qwen/Qwen-Image
- ✅ qwen_image_edit_2511_lightning   - Default: lightx2v/Qwen-Image-Edit-2511-Lightning
- ✅ qwen_image_edit_2511             - Default: Qwen/Qwen-Image-Edit-2511
- ✅ qwen_image_edit_lightning        - Default: lightx2v/Qwen-Image-Lightning
- ✅ qwen_image_edit                  - Default: Qwen/Qwen-Image-Edit-2509
- ✅ qwen_image_21                    - Default: Qwen/Qwen-Image-2.1
- ✅ qwen_image_21_fused              - Default: Qwen/Qwen-Image-2.1
- ✅ qwen_image_21_turbo_fused        - Default: Qwen/Qwen-Image-2.1-Turbo
- ✅ qwen_image_21_turbo              - Default: Qwen/Qwen-Image-2.1-Turbo
- ✅ qwen_image_21_edit               - Default: Qwen/Qwen-Image-2.1
- ✅ qwen_image_21_turbo_edit         - Default: Qwen/Qwen-Image-2.1-Turbo
- ✅ qwen_image_21_modular            - Default: Qwen/Qwen-Image-2.1
- ✅ qwen_image_21_turbo_modular      - Default: Qwen/Qwen-Image-2.1-Turbo
- ✅ qwen_image_21_modular_edit       - Default: Qwen/Qwen-Image-2.1
- ✅ qwen_image_21_turbo_modular_edit - Default: Qwen/Qwen-Image-2.1-Turbo
- ✅ qwen_image_controlnet            - Default: InstantX/Qwen-Image-ControlNet-Inpainting
- ✅ qwen_image_layered               - Default: Qwen/Qwen-Image-Layered
- ✅ skyreels_v2                      - Default: Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
- ✅ ltx2_t2v                         - Default: Lightricks/LTX-2
- ✅ ltx2_i2v                         - Default: Lightricks/LTX-2
- ✅ ltx25_t2v                        - Default: Lightricks/LTX-2.5-Diffusers
- ✅ ltx25_i2v                        - Default: Lightricks/LTX-2.5-Diffusers
- ✅ ltx25_dfr                        - Default: Lightricks/LTX-2.5-Diffusers
- ✅ wan2.2_t2v                       - Default: Wan-AI/Wan2.2-T2V-A14B-Diffusers
- ✅ wan2.1_t2v                       - Default: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
- ✅ wan2.2_i2v                       - Default: Wan-AI/Wan2.2-I2V-A14B-Diffusers
- ✅ wan2.1_i2v                       - Default: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
- ✅ wan2.2_vace                      - Default: linoyts/Wan2.2-VACE-Fun-14B-diffusers
- ✅ wan2.1_vace                      - Default: Wan-AI/Wan2.1-VACE-1.3B-diffusers
- ✅ ovis_image                       - Default: AIDC-AI/Ovis-Image-7B
- ✅ zimage_turbo_nunchaku            - Default: nunchaku/nunchaku-z-image-turbo
- ✅ zimage_turbo                     - Default: Tongyi-MAI/Z-Image-Turbo
- ✅ zimage                           - Default: Tongyi-MAI/Z-Image
- ✅ zimage_turbo_controlnet_2.0      - Default: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0
- ✅ zimage_turbo_controlnet_2.1      - Default: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
- ✅ longcat_image                    - Default: meituan-longcat/LongCat-Image
- ✅ longcat_image_edit               - Default: meituan-longcat/LongCat-Image-Edit
- ✅ glm_image                        - Default: zai-org/GLM-Image
- ✅ glm_image_edit                   - Default: zai-org/GLM-Image
- ✅ firered_image_edit_1.1           - Default: FireRedTeam/FireRed-Image-Edit-1.1
- ✅ firered_image_edit_1.0           - Default: FireRedTeam/FireRed-Image-Edit-1.0
- ✅ helios_t2v                       - Default: BestWishYsh/Helios-Base
- ✅ helios_t2v_distill               - Default: BestWishYsh/Helios-Distilled
- ✅ cosmos3_omni_t2i                 - Default: nvidia/Cosmos3-Nano
- ✅ cosmos3_omni_t2v                 - Default: nvidia/Cosmos3-Nano
- ✅ ernie_image                      - Default: baidu/ERNIE-Image
- ✅ ernie_image_turbo                - Default: baidu/ERNIE-Image-Turbo
- ✅ krea2                            - Default: krea/Krea-2-Turbo
- ✅ joy_image_edit                   - Default: jdopensource/JoyAI-Image-Edit-Diffusers
- ✅ bria_fibo                        - Default: briaai/FIBO
- ✅ bria_fibo2                       - Default: briaai/Fibo-2.0-dev
- ✅ bria_fibo2_turbo                 - Default: briaai/Fibo-2.0-edit-turbo
- ✅ echo_t2va                        - Default: jdopensource/JoyAI-Echo
- ✅ motif_video_t2v                  - Default: Motif-Technologies/Motif-Video-2B
- ✅ motif_video_i2v                  - Default: Motif-Technologies/Motif-Video-2B
- ✅ anyflow_t2v                      - Default: nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers
- ✅ minimax_h3                       - Default: MiniMaxAI/MiniMax-H3
- ✅ minimax_h3_fl2va                 - Default: MiniMaxAI/MiniMax-H3
- ✅ minimax_h3_ref2va                - Default: MiniMaxAI/MiniMax-H3
- ✅ fasth3                           - Default: FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
- ✅ fasth3_v2                        - Default: FastVideo/FastVideo-FastH3-8-Step-V2
- ✅ anyflow_far                      - Default: nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers
- ✅ prx_pixel                        - Default: Photoroom/prxpixel-t2i
- ✅ anima                            - Default: circlestone-labs/Anima-Base-v1.0-Diffusers
- ✅ ideogram4                        - Default: ideogram-ai/ideogram-4-fp8
- ✅ moss_image_sft                   - Default: OpenMOSS-Team/MOSS-Image-SFT
- ✅ moss_image_base                  - Default: OpenMOSS-Team/MOSS-Image-Base
- ✅ ming_image_design                - Default: inclusionAI/Ming-Image-0.1-Design
- ✅ ming_image_design_layer          - Default: inclusionAI/Ming-Image-0.1-Design-Layer
- ✅ hidream_o1_image                 - Default: HiDream-ai/HiDream-O1-Image
- ✅ nucleus_image                    - Default: NucleusAI/Nucleus-Image
- ✅ kandinsky6_ti2va                 - Default: kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers
- ✅ longcat_audio_dit                - Default: ruixiangma/LongCat-AudioDiT-1B-Diffusers
- ✅ stable_audio_3                   - Default: stabilityai/stable-audio-3-medium
- ✅ sana_wm                          - Default: Efficient-Large-Model/SANA-WM_bidirectional-diffusers
- ✅ sana_video2                      - Default: Efficient-Large-Model/SANA-Video_2.0_5B_720p
- ✅ sana_video2_ti2v                 - Default: Efficient-Large-Model/SANA-Video_2.0_5B_720p
- ✅ wan_animate_2                    - Default: Wan-AI/Wan2.2-Animate-2-14B-Diffusers
- ✅ wan_animate_2_distilled          - Default: Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers
- ✅ lingbot_video                    - Default: robbyant/lingbot-video-dense-1.3b
- ✅ causvid_wan                      - Default: tianweiy/CausVid
- ✅ longlive2                        - Default: Efficient-Large-Model/LongLive-2.0-5B
- ✅ wan2.2_s2v                       - Default: Wan-AI/Wan2.2-S2V-14B-Diffusers
  (speech-to-video: `--audio-path` + `--image-path`. The cache skips the audio injection on
  cached steps, so keep `--rdt` tighter than the Wan-T2V presets and prefer a small
  `--max-cached-steps`; `rdt=0.08` reaches 39.6 dB, `rdt=0.20` 38.7 dB at 480x832.)
- ✅ sana_video                       - Default: Efficient-Large-Model/SANA-Video_2B_480p_diffusers
- ✅ sana_video_i2v                   - Default: Efficient-Large-Model/SANA-Video_2B_480p_diffusers
- ✅ longcat_video_avatar_a2v         - Default: meituan-longcat/LongCat-Video-Avatar-1.5
- ✅ longcat_video_avatar_ai2v        - Default: meituan-longcat/LongCat-Video-Avatar-1.5
- ✅ magi2_t2va                       - Default: sand-ai/MAGI-2-preview
- ✅ magi2_i2va                       - Default: sand-ai/MAGI-2-preview
- ✅ mova_360p                        - Default: openmoss/MOVA-360p
- ✅ hy_nomas                         - Default: tencent/Hy-Nomas
- ✅ pika_h3_flex                     - Default: Pika-Labs/Pika-H3-Flex
- ✅ pika_h3_flex_vendored            - Default: Pika-Labs/Pika-H3-Flex
- ✅ pika_h3_flex_fl2va               - Default: Pika-Labs/Pika-H3-Flex
- ✅ pika_h3_flex_ref2va              - Default: Pika-Labs/Pika-H3-Flex
- ✅ pika_h3_flex_fl2va_vendored      - Default: Pika-Labs/Pika-H3-Flex
- ✅ pika_h3_flex_ref2va_vendored     - Default: Pika-Labs/Pika-H3-Flex
- ✅ flux2_klein_fused                - Default: black-forest-labs/FLUX.2-klein-4B
- ✅ minimax_h3_image                 - Default: MiniMaxAI/MiniMax-H3-Image
- ✅ minimax_h3_image_edit            - Default: MiniMaxAI/MiniMax-H3-Image
- ✅ ph3_image                        - Default: PrunaAI/P-H3-Image
- ✅ ph3_image_edit                   - Default: PrunaAI/P-H3-Image
```

## Single GPU Inference

The easiest way to enable hybrid cache acceleration for DiTs with cache-dit is to start with single GPU inference. For examples:  

```bash
# baseline
# use default model path, e.g, "black-forest-labs/FLUX.1-dev"
python3 -m cache_dit.generate flux 
python3 -m cache_dit.generate flux_nunchaku # need nunchaku library
python3 -m cache_dit.generate flux2
python3 -m cache_dit.generate flux3 # diffusers modular pipeline
python3 -m cache_dit.generate flux3_edit # diffusers modular pipeline, image editing
python3 -m cache_dit.generate ovis_image
python3 -m cache_dit.generate qwen_image_edit_lightning
python3 -m cache_dit.generate qwen_image
python3 -m cache_dit.generate ltx2_t2v --cache --cpu-offload
python3 -m cache_dit.generate ltx2_i2v --cache --cpu-offload
python3 -m cache_dit.generate ltx25_dfr --cache --cpu-offload   # DFR, schedules its own 8 sigmas
python3 -m cache_dit.generate ltx25_t2v --cache --cpu-offload
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
python3 -m cache_dit.generate flux --cache --seacache --seacache-feature-shape 64 64
python3 -m cache_dit.generate flux --cache --seacache --seacache-feature-shape 64 64 --softcache --softcache-tau 0.5 --calibrator residual_motion --max-warmup-steps 1 --max-cached-steps -1 --max-continuous-cached-steps -1
python3 -m cache_dit.generate flux --cache --calibrator taylorseer
python3 -m cache_dit.generate flux --cache --calibrator svd --calibrator-params "tau=0.9,beta=0.9"
python3 -m cache_dit.generate flux --cache --calibrator hicache --calibrator-params "order=2,sigma=0.5"
python3 -m cache_dit.generate flux --cache --calibrator brace --calibrator-params "capacity=3,gamma=0.7"
python3 -m cache_dit.generate flux --cache --calibrator bezicast --calibrator-params "history=6,control_points=4,ridge=0.5,guardrail=false"
python3 -m cache_dit.generate flux --cache --calibrator residual_motion --calibrator-params "beta=1.0"
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

# Qwen-Image 2.1: the pipeline samples without guidance by default, so the example disables
# separate-CFG bookkeeping (pass `--true-cfg-scale 4.0` and a negative prompt only in your own
# script, and set `enable_separate_cfg=True` there). Turbo keeps the 8-step sigma grid that ships
# with the checkpoint, `--steps` does not override it.
python3 -m cache_dit.generate qwen_image_21 --cache
python3 -m cache_dit.generate qwen_image_21_turbo --cache
python3 -m cache_dit.generate qwen_image_21_edit --cache # 40 steps, needs cfg bookkeeping
python3 -m cache_dit.generate qwen_image_21_turbo_edit --cache
python3 -m cache_dit.generate qwen_image_21_modular --cache # diffusers modular pipeline
python3 -m cache_dit.generate qwen_image_21_turbo_modular --cache
python3 -m cache_dit.generate qwen_image_21_modular_edit --cache
python3 -m cache_dit.generate qwen_image_21_turbo_modular_edit --cache
```

### Ming-Image 0.1

`Ming-Image` uses a built-in pipeline and does not require vLLM, vllm-omni, or remote model code.
Both variants default to 12 inference steps. `Design` returns one RGBA image. `Design-Layer`
returns the composite followed by every RGBA layer, saves each image separately, and preserves the
reference image aspect ratio with 16-pixel alignment.

```bash
python3 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design --cache --rdt 0.20 --ws 8
python3 -m cache_dit.generate ming_image_design_layer \
  --model-path /PATH/TO/Ming-Image-0.1-Design-Layer \
  --image-path /PATH/TO/reference.png --cache --rdt 0.08 --ws 8

torchrun --nproc_per_node=2 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design --parallel ulysses --parallel-vae
torchrun --nproc_per_node=2 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design --parallel tp --parallel-text

python3 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design --quantize --compile
python3 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design --quantize \
  --quantize-params quant_type=svdq_nvfp4_r64_dq --compile
python3 -m cache_dit.generate ming_image_design \
  --model-path /PATH/TO/Ming-Image-0.1-Design \
  --layerwise-offload --layerwise-async-transfer --no-layerwise-text-async-transfer --compile
```

TE-P shards fused-QKV and every dense/routed/shared expert `gate_proj`/`up_proj` with replicated
outputs. Each expert's `down_proj`, the attention output projection, routers and the vision tower
stay replicated so every consumer of a sharded linear still receives a complete activation. Ring
and `--ulysses-anything` are rejected for the zero-masked Design variant because it emits a full
padding mask; the learned-padding Design-Layer variant keeps the Z-Image behavior. TE-P is
memory-oriented and can be slower than plain TP. For offload, keep async transfer on for the
DiT/VAE but disable it for the 15k-submodule MLLM (`--no-layerwise-text-async-transfer`); global
text async is much slower.

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
python3 -m cache_dit.generate lingbot_video --quantize --compile --cache

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

# Qwen-Image 2.1: Ulysses only. Prefix and target tokens are sharded as one sequence with tail
# padding, and prefix KV caches are head-sharded. Ring and `--ulysses-anything` are rejected.
# Sequence lengths need not divide by cp_size; the local head count after TP must divide by it.
torchrun --nproc_per_node=2 -m cache_dit.generate qwen_image_21_turbo --parallel ulysses --cache
torchrun --nproc_per_node=2 -m cache_dit.generate qwen_image_21_turbo --parallel tp --parallel-text --parallel-vae --cache
```

## Low-bits Quantization 

cache-dit is designed to work seamlessly with torch.compile, Quantization (🔥torchao, 🔥nunchaku), For examples:

```bash
# please also enable torch.compile if the quantation is using.
python3 -m cache_dit.generate flux --cache --quantize --compile
# cache-dit's own FP8 kernels (sm100+), instead of the default TorchAO backend
python3 -m cache_dit.generate flux --cache --quantize \
  --quantize-params quant_type=float8_per_row,backend=cache_dit --compile
python3 -m cache_dit.generate flux --cache --quantize --quantize-params quant_type=int8_per_row --compile
python3 -m cache_dit.generate flux --cache --quantize --quantize-params quant_type=float8_weight_only --compile
python3 -m cache_dit.generate flux --cache --quantize --quantize-params quant_type=int8_weight_only --compile
python3 -m cache_dit.generate flux --cache --quantize --quantize-params quant_type=bitsandbytes_4bit --compile # w4a16
python3 -m cache_dit.generate flux_nunchaku --cache --compile # w4a4 SVDQ
```

## Hybrid Acceleration 

Here are some examples for `hybrid cache acceleration + parallelism` for popular DiTs with cache-dit.

```bash
# DBCache + SCM + Taylorseer
python3 -m cache_dit.generate flux --cache --scm fast --calibrator taylorseer --calibrator-params order=1
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + CUDNN Attention (--attn _sdpa_cudnn)
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --quant-communicate-params bits=8 \
         --attn _sdpa_cudnn --parallel-text --cache --scm fast --calibrator taylorseer \
         --calibrator-params order=1 --quantize --warmup 2 --repeat 5 --compile
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + FP8 SageAttention (--attn sage)
torchrun --nproc_per_node=4 -m cache_dit.generate flux --parallel ulysses --quant-communicate-params bits=8 \
         --attn sage --parallel-text --cache --scm fast --calibrator taylorseer \
         --calibrator-params order=1 --quantize --warmup 2 --repeat 5 --compile
# Case: Hybrid Acceleration for Qwen-Image-Edit-Lightning, tracking memory usage.
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning \
         --parallel ulysses --ulysses-anything --parallel-text \
         --quantize --quantize-params quant_type=float8_weight_only --steps 4 --track-memory --compile
torchrun --nproc_per_node=4 -m cache_dit.generate qwen_image_edit_lightning \
         --parallel tp --parallel-text --quantize --quantize-params quant_type=float8_weight_only \
         --steps 4 --track-memory --compile
# Case: Hybrid Acceleration + Context Parallelism + ControlNet Parallelism, e.g, Z-Image-ControlNet
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --rdt 0.6 --scm fast
torchrun --nproc_per_node=4 -m cache_dit.generate zimage_turbo_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --scm fast --rdt 0.6 --compile \
         --compile-controlnet --quant-communicate-params bits=8 --attn _sdpa_cudnn \
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

It is very easy to add a new example. Please refer to the specific implementation in [examples.py](https://github.com/vipshop/cache-dit/raw/main/src/cache_dit/cli/examples.py). For example:

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
            # `text_encoder_2` will be quantized when `--quantize-params`
            # `quant_type` is set to `bitsandbytes_4bit`.
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
  {generate,list,flux_nunchaku,flux,flux_fill,flux2,flux2_klein_base_9b,flux2_klein_base_4b,flux2_klein_9b,flux2_klein_4b,flux2_klein_base_9b_edit,flux2_klein_base_4b_edit,flux2_klein_9b_edit,flux2_klein_4b_edit,flux3,flux3_fused,flux3_edit,flux2_klein_9b_kv_edit,qwen_image_lightning,qwen_image_2512,qwen_image,qwen_image_edit_2511_lightning,qwen_image_edit_2511,qwen_image_edit_lightning,qwen_image_edit,qwen_image_21,qwen_image_21_fused,qwen_image_21_turbo_fused,qwen_image_21_turbo,qwen_image_21_edit,qwen_image_21_turbo_edit,qwen_image_21_modular,qwen_image_21_turbo_modular,qwen_image_21_modular_edit,qwen_image_21_turbo_modular_edit,qwen_image_controlnet,qwen_image_layered,skyreels_v2,ltx2_t2v,ltx2_i2v,ltx25_t2v,ltx25_i2v,ltx25_dfr,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage_turbo_nunchaku,zimage_turbo,zimage,zimage_turbo_controlnet_2.0,zimage_turbo_controlnet_2.1,longcat_image,longcat_image_edit,glm_image,glm_image_edit,firered_image_edit_1.1,firered_image_edit_1.0,helios_t2v,helios_t2v_distill,cosmos3_omni_t2i,cosmos3_omni_t2v,boogu_image_edit,ernie_image,ernie_image_turbo,krea2,joy_image_edit,bria_fibo,bria_fibo2,bria_fibo2_turbo,echo_t2va,motif_video_t2v,motif_video_i2v,anyflow_t2v,minimax_h3,minimax_h3_fl2va,minimax_h3_ref2va,fasth3,fasth3_v2,anyflow_far,prx_pixel,anima,ideogram4,moss_image_sft,moss_image_base,ming_image_design,ming_image_design_layer,hidream_o1_image,nucleus_image,kandinsky6_ti2va,longcat_audio_dit,stable_audio_3,sana_wm,sana_video2,sana_video2_ti2v,wan_animate_2,wan_animate_2_distilled,lingbot_video,causvid_wan,longlive2,wan2.2_s2v,sana_video,sana_video_i2v,longcat_video_avatar_a2v,longcat_video_avatar_ai2v,magi2_t2va,magi2_i2va,mova_360p,hy_nomas,pika_h3_flex,pika_h3_flex_vendored,pika_h3_flex_fl2va,pika_h3_flex_ref2va,pika_h3_flex_fl2va_vendored,pika_h3_flex_ref2va_vendored,flux2_klein_fused,minimax_h3_image,minimax_h3_image_edit,ph3_image,ph3_image_edit}
                        The task to perform or example name to run. Use 'list'
                        to list all available examples, or specify an example
                        name directly (defaults to 'generate' task).
  {None,flux_nunchaku,flux,flux_fill,flux2,flux2_klein_base_9b,flux2_klein_base_4b,flux2_klein_9b,flux2_klein_4b,flux2_klein_base_9b_edit,flux2_klein_base_4b_edit,flux2_klein_9b_edit,flux2_klein_4b_edit,flux3,flux3_fused,flux3_edit,flux2_klein_9b_kv_edit,qwen_image_lightning,qwen_image_2512,qwen_image,qwen_image_edit_2511_lightning,qwen_image_edit_2511,qwen_image_edit_lightning,qwen_image_edit,qwen_image_21,qwen_image_21_fused,qwen_image_21_turbo_fused,qwen_image_21_turbo,qwen_image_21_edit,qwen_image_21_turbo_edit,qwen_image_21_modular,qwen_image_21_turbo_modular,qwen_image_21_modular_edit,qwen_image_21_turbo_modular_edit,qwen_image_controlnet,qwen_image_layered,skyreels_v2,ltx2_t2v,ltx2_i2v,ltx25_t2v,ltx25_i2v,ltx25_dfr,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage_turbo_nunchaku,zimage_turbo,zimage,zimage_turbo_controlnet_2.0,zimage_turbo_controlnet_2.1,longcat_image,longcat_image_edit,glm_image,glm_image_edit,firered_image_edit_1.1,firered_image_edit_1.0,helios_t2v,helios_t2v_distill,cosmos3_omni_t2i,cosmos3_omni_t2v,boogu_image_edit,ernie_image,ernie_image_turbo,krea2,joy_image_edit,bria_fibo,bria_fibo2,bria_fibo2_turbo,echo_t2va,motif_video_t2v,motif_video_i2v,anyflow_t2v,minimax_h3,minimax_h3_fl2va,minimax_h3_ref2va,fasth3,fasth3_v2,anyflow_far,prx_pixel,anima,ideogram4,moss_image_sft,moss_image_base,ming_image_design,ming_image_design_layer,hidream_o1_image,nucleus_image,kandinsky6_ti2va,longcat_audio_dit,stable_audio_3,sana_wm,sana_video2,sana_video2_ti2v,wan_animate_2,wan_animate_2_distilled,lingbot_video,causvid_wan,longlive2,wan2.2_s2v,sana_video,sana_video_i2v,longcat_video_avatar_a2v,longcat_video_avatar_ai2v,magi2_t2va,magi2_i2va,mova_360p,hy_nomas,pika_h3_flex,pika_h3_flex_vendored,pika_h3_flex_fl2va,pika_h3_flex_ref2va,pika_h3_flex_fl2va_vendored,pika_h3_flex_ref2va_vendored,flux2_klein_fused,minimax_h3_image,minimax_h3_image_edit,ph3_image,ph3_image_edit}
                        Names of the examples to run. If not specified, skip
                        running example.

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
  --prompt-enhancer PROMPT_ENHANCER
                        Local path of an optional prompt-enhancer repository.
                        When set, the examples that own such a step expand the
                        prompt with it instead of rendering the prompt as
                        given; the per-example docstring says where the
                        weights come from and what else changes. Off by
                        default: the expanded caption is a different input
                        distribution, and the enhancer is 18-19 GiB that no
                        `--parallel*` option covers.
  --mask-image-path MASK_IMAGE_PATH
                        Override mask image path if provided
  --last-image-path LAST_IMAGE_PATH
                        Path of the *last* frame for first/last-frame
                        conditioning (MiniMax-H3 `fl2va`, which takes `image`
                        and `last_image` together)
  --driving-video-path DRIVING_VIDEO_PATH
                        Path of the driving video for a character-animation
                        model (Wan-Animate-2). Falls back to
                        `$WAN_ANIMATE_2_DEMO_DIR/template.mp4`, then to frames
                        synthesized from the reference image
  --driving-video-fps DRIVING_VIDEO_FPS
                        Frame rate of the driving video, used to resample it
                        to the model fps
  --audio-path AUDIO_PATH
                        Path of the driving waveform for a speech-to-video
                        model (Wan2.2-S2V). Falls back to
                        `$WAN_2_2_S2V_DEMO_DIR/*.wav`, then to the example wav
                        next to the model
  --reference-image-path REFERENCE_IMAGE_PATH [REFERENCE_IMAGE_PATH ...]
                        Path(s) of image reference(s) for a reference-
                        conditioned model (MiniMax-H3 `ref2va`). References
                        are consumed in image -> video -> audio order
  --reference-video-path REFERENCE_VIDEO_PATH [REFERENCE_VIDEO_PATH ...]
                        Path(s) of video reference(s) for a reference-
                        conditioned model (MiniMax-H3 `ref2va`). References
                        are consumed in image -> video -> audio order
  --reference-audio-path REFERENCE_AUDIO_PATH [REFERENCE_AUDIO_PATH ...]
                        Path(s) of audio reference(s) for a reference-
                        conditioned model (MiniMax-H3 `ref2va`). References
                        are consumed in image -> video -> audio order
  --config-path CONFIG_PATH, --config CONFIG_PATH
                        Path to CacheDiT configuration YAML file
  --prompt PROMPT       Override default prompt if provided
  --negative-prompt NEGATIVE_PROMPT
                        Override default negative prompt if provided
  --skip-negative_prompt, --skip-neg
                        Force skip negative prompt even if negative prompt is
                        provided.
  --num_inference_steps NUM_INFERENCE_STEPS, --steps NUM_INFERENCE_STEPS
                        Number of inference steps
  --warmup WARMUP       Number of warmup steps before measuring performance
  --warmup-num-inference-steps WARMUP_NUM_INFERENCE_STEPS, --warmup-steps WARMUP_NUM_INFERENCE_STEPS
                        Number of warmup inference steps per warmup before
                        measuring performance
  --warmup-seed WARMUP_SEED
                        Optional random seed used only for warmup forwards.
                        When set, warmup uses this seed while formal repeated
                        inference continues to use --seed. This is especially
                        useful for few-shot SVDQ runs where runtime
                        quantization usually happens during warmup.
  --warmup-prompt WARMUP_PROMPT
                        Optional prompt used only for warmup forwards. When
                        set, warmup uses this prompt while formal repeated
                        inference continues to use --prompt or the example
                        default prompt.
  --repeat REPEAT       Number of times to repeat the inference for
                        performance measurement
  --height HEIGHT       Height of the generated image
  --width WIDTH         Width of the generated image
  --input-height INPUT_HEIGHT
                        Height of the input image
  --input-width INPUT_WIDTH
                        Width of the input image
  --seed SEED           Random seed for reproducibility
  --extra-input-kwargs EXTRA_INPUT_KWARGS
                        Extra input kwargs passed to the pipeline as a Python
                        dict literal. Keys override both default and
                        programmatic extra_input_kwargs. Example: --extra-
                        input-kwargs "{'cfg_normalization': False,
                        'max_sequence_length': 1536}"
  --num-frames NUM_FRAMES, --frames NUM_FRAMES
                        Number of frames to generate for video
  --save-path SAVE_PATH
                        Path to save the generated output, e.g., output.png or
                        output.mp4
  --cache               Enable Cache Acceleration
  --cache-summary, --summary
                        Enable Cache Summary logging
  --Fn-compute-blocks FN_COMPUTE_BLOCKS, --Fn FN_COMPUTE_BLOCKS
                        CacheDiT Fn_compute_blocks parameter
  --Bn-compute-blocks BN_COMPUTE_BLOCKS, --Bn BN_COMPUTE_BLOCKS
                        CacheDiT Bn_compute_blocks parameter
  --residual-diff-threshold RESIDUAL_DIFF_THRESHOLD, --rdt RESIDUAL_DIFF_THRESHOLD
                        CacheDiT residual diff threshold. A model with per-
                        workflow cache presets treats this default as 'not
                        set' and uses its own starting point; any other value
                        wins
  --max-warmup-steps MAX_WARMUP_STEPS, --ws MAX_WARMUP_STEPS
                        Maximum warmup steps for CacheDiT
  --warmup-interval WARMUP_INTERVAL, --wi WARMUP_INTERVAL
                        Warmup interval for CacheDiT
  --max-cached-steps MAX_CACHED_STEPS, --mc MAX_CACHED_STEPS
                        Maximum cached steps for CacheDiT
  --max-continuous-cached-steps MAX_CONTINUOUS_CACHED_STEPS, --mcc MAX_CONTINUOUS_CACHED_STEPS
                        Maximum continuous cached steps for CacheDiT
  --seacache            Enable the optional SEA-filtered relative-L1 probe
                        inside DBCache.
  --seacache-mode {flow,vp}
                        SeaCache scheduler coefficient convention.
  --seacache-feature-shape DIM [DIM ...]
                        Global packed-token geometry: H W for images or F H W
                        for video.
  --seacache-power-exp SEACACHE_POWER_EXP
                        SeaCache clean-signal power-spectrum exponent (2 for
                        images, commonly 3 for video).
  --softcache           Enable SoftCache on top of --seacache: SeaCache's own
                        refresh rule plus headroom sampler jumps (FlowMatch
                        Euler schedulers).
  --softcache-tau SOFTCACHE_TAU
                        SoftCache staleness budget; ~0.3-0.5 for ~3-4x and
                        ~1.1 for ~6.5x on FLUX.1-dev.
  --softcache-jf-max SOFTCACHE_JF_MAX
                        Cap on the SoftCache jump factor.
  --calibrator CALIBRATOR
                        DBCache calibrator to enable: taylorseer, dmd, foca,
                        svd (SVD-Cache), hicache (damped-Hermite), brace
                        (barycentric rational), bezicast (Bernstein/Bezier),
                        or residual_motion (sigma secant, SoftCache). Long
                        forms such as 'svd-calibrator' are accepted. Requires
                        --cache.
  --calibrator-params CALIBRATOR_PARAMS
                        Calibrator options as a comma separated key=value
                        list, e.g. 'tau=0.9,beta=0.9' or
                        'predictor=taylor,order=1' for svd,
                        'order=2,sigma=0.5' for hicache,
                        'capacity=3,gamma=0.7' for brace,
                        'history=6,control_points=4,ridge=0.5,guardrail=false'
                        for bezicast, 'beta=0.5' for residual_motion,
                        'history=6,svd_precision=low' for dmd, or
                        'calibrator_cache_type=hidden_states,encoder=false'
                        for any calibrator.
  --steps-mask          Enable steps mask for CacheDiT
  --mask-policy {None,slow,s,medium,m,fast,f,ultra,u}, --scm {None,slow,s,medium,m,fast,f,ultra,u}
                        Pre-defined steps computation mask policy
  --quantize, --q       Enable quantization for transformer
  --quantize-params QUANTIZE_PARAMS
                        Quantization options as a comma separated key=value
                        list, e.g. 'quant_type=float8_per_row' (default when
                        omitted), 'backend=cache_dit',
                        'exclude_layers=adaln_proj+norm_out',
                        'text_encoder=float8_weight_only' (or 'true' to reuse
                        quant_type; usable without --quantize),
                        'regional_quantize=false',
                        'per_tensor_fallback=false', 'verbose=true',
                        'fp8_fused_mlp=true', 'fp4_fused_mlp=true',
                        'mxfp4_quantize_device=cpu'.
  --svdq-params SVDQ_PARAMS
                        SVDQuant options as a comma separated key=value list,
                        requires an svdq_* quant_type. Keys are the
                        svdq_kwargs fields: smooth_strategy,
                        calibrate_precision, runtime_kernel, lora_layout,
                        quantize_device, offload_quantized_layers_to_cpu,
                        layerwise_offload, async_transfer, transfer_buckets,
                        prefetch_limit, max_copy_streams,
                        max_inflight_prefetch_bytes, persistent_buckets,
                        persistent_bins, defer_move_to_execution_device,
                        few_shot_steps, few_shot_relax_factor,
                        few_shot_relax_top_ratio, few_shot_relax_strategy,
                        few_shot_auto_compile, fused_mlp, svd_refine_iters,
                        packed_a4_stages, packed_a4_inner_group, streaming,
                        activation_buffer_flush_sample_count,
                        activation_buffer_flush_cpu_bytes. The layerwise keys
                        inherit --layerwise-* when omitted.
  --disable-per-row, --no-per-row
                        Disable per row quantization for transformer
  --parallel-type {None,tp,ulysses,ring,usp,ulysses_tp,ring_tp,tp_ulysses,tp_ring,usp_tp}, --parallel {None,tp,ulysses,ring,usp,ulysses_tp,ring_tp,tp_ulysses,tp_ring,usp_tp}
  --parallel-vae        Enable VAE parallelism if applicable.
  --parallel-text-encoder, --parallel-text
                        Enable text encoder parallelism if applicable.
  --parallel-controlnet
                        Enable ControlNet parallelism if applicable.
  --attn {None,flash,flash_varlen,_flash_3,native,_native_cudnn,_sdpa_cudnn,sage,sage3,sol_attn,ffpa,ffpa_fp8,ffpa_fp4,ffpa_fp8_per_block,_native_npu,_npu_fia,_mindiesd_laser}
  --sol-attn-tau SOL_ATTN_TAU
                        Sol-Attn routing threshold multiplier; larger values
                        select fewer exact KV blocks.
  --sol-attn-thresh-type {diag,exact}
                        Sol-Attn threshold estimator: diagonal or full
                        covariance.
  --sol-attn-strict     Raise on Sol-Attn-ineligible calls instead of using
                        observable native fallback.
  --sol-attn-sink-tokens SOL_ATTN_SINK_TOKENS
                        Exact prefix-KV token budget (default: 1024; 0
                        disables; capped at sequence length).
  --ffpa-hybrid         Enable the FFPA hybrid fp16 early-rows stage (--attn
                        ffpa_fp8 / ffpa_fp4 / ffpa_fp8_per_block). Off by
                        default for every backend, including causal attention;
                        opt-in only. Reflected in the saved filename as
                        hybrid.
  --ffpa-hybrid-n-early FFPA_HYBRID_N_EARLY, --ffpa-hybird-n-early FFPA_HYBRID_N_EARLY
                        Number of fp16 early rows for the FFPA hybrid mode
                        (multiple of 128). Requires --attn ffpa_fp8/ffpa_fp4
                        and --ffpa-hybrid. Reflected in the saved filename as
                        hybrid_n_early_<N>.
  --ffpa-fp4-hadamard   Enable the Hadamard Q/K pre-rotation for the FFPA fp4
                        backend (flattens per-16-group outliers, exact in fp32
                        math, small perf overhead). Requires --attn ffpa_fp4.
                        Reflected in the saved filename as fp4_hadamard.
  --ffpa-fp8-hadamard   Enable the Hadamard Q/K pre-rotation for the FFPA fp8
                        backends (incoherent processing as in
                        FlashAttention-3: spreads per-dim outliers so per-
                        block/per-group quant amax drops, exact in fp32 math).
                        Requires an ffpa_fp8* attention backend. Reflected in
                        the saved filename as fp8_hadamard.
  --ffpa-fp4-pv-mm-type {fp4,fp8}
                        PV MMA dtype for the FFPA fp4 backend: fp4 (NVFP4 e2m1
                        + ue4m3/16, default) or fp8 (MXFP8 e4m3 + ue8m0/32,
                        higher PV precision, persist_d head_dim <= 192 only).
                        Requires --attn ffpa_fp4. Reflected in the saved
                        filename as fp4_pv_mm_type_<T>.
  --ffpa-fp4-smooth-v   Subtract the per-(b,hkv) V column mean before V
                        quantize for the FFPA fp4 backend (persist_d head_dim
                        <= 256 only). Requires --attn ffpa_fp4. Reflected in
                        the saved filename as fp4_smooth_v.
  --ffpa-fp8-smooth-v   Subtract the per-(b,h) V dim mean before V quantize
                        for the FFPA fp8 backends (requires per-channel V, the
                        default). Off by default; requires --attn ffpa_fp8 or
                        ffpa_fp8_per_block. Reflected in the saved filename as
                        fp8_smooth_v.
  --ulysses-anything, --uaa
                        Enable Ulysses Anything Attention for context
                        parallelism
  --quant-communicate-params QUANT_COMMUNICATE_PARAMS
                        Ulysses low-precision communication options as k=v
                        pairs, e.g.
                        'bits=6,quant_type=per_group_32,quantize_k=false'
  --ulysses-async, --uaqkv
                        Enabled experimental Async QKV Projection with Ulysses
                        for context parallelism
  --ring-rotate-method {allgather,p2p}, --rotate {allgather,p2p}
                        Ring Attention rotation method for context parallelism
  --ring-no-convert-to-fp32, --ring-no-fp32, --no-fp32
                        Disable convert Ring Attention output and lse to fp32
                        for context parallelism
  --cpu-offload, --cpu-offload-model
                        Enable diffusers model CPU offload for pipeline
                        components when supported.
  --sequential-cpu-offload
                        Enable diffusers sequential CPU offload for pipeline
                        components when supported.
  --module-layerwise-cpu-offload, --layerwise-offload
                        Enable cache-dit generic sequential CPU offload for
                        nn.Module components. This is intended for custom non-
                        diffusers modules and is mutually exclusive with the
                        diffusers pipeline offload switches.
  --adaln-host-residency
                        Keep the transformer resident on the device but stream
                        its AdaLN modulation tables (`adaln_proj`) from a
                        pinned host mirror, and stream the text encoder as
                        usual. Intended for joint video + audio models whose
                        AdaLN tables are a large share of the transformer
                        weights (e.g. MiniMax-H3: 24 GiB of 62 GiB), where it
                        cuts both the resident footprint and the per-step
                        host-to-device traffic. While the AdaLN tables are
                        actually streamed (i.e. without `--quantize` or
                        `--parallel tp`) each per-block copy is a graph break,
                        so `--compile` still works but pays a smaller speedup;
                        `--compile-full-graph` and `--cuda-graph` are rejected
                        in that case.
  --adaln-online        Precompute the AdaLN modulation tables of a MiniMax-H3
                        transformer for the checkpoint's fixed sampling grid,
                        keep the small results on the device and park the 24
                        GiB of `adaln_proj` weights on the host. Exact, not an
                        approximation: the tables are a pure function of the
                        timestep embedding, and a timestep outside the
                        announced grid falls back to recomputing with the
                        weights. Removes the per-step host-to-device traffic
                        `--adaln-host-residency` pays, so it also keeps
                        `--compile` free of that graph break.
  --layerwise-async-transfer
                        Enable CUDA-stream-based async_transfer for cache-dit
                        layerwise CPU offload. This shared setting applies to
                        both generic module layerwise offload and SVDQ
                        layerwise collection when --svdq-params
                        layerwise_offload=true is active.
  --layerwise-text-async-transfer
                        Override text encoder layerwise async_transfer. When
                        omitted, text encoder layerwise offload uses
                        --layerwise-async-transfer.
  --no-layerwise-text-async-transfer
                        Disable text encoder layerwise async_transfer even if
                        --layerwise-async-transfer is enabled.
  --layerwise-transfer-buckets LAYERWISE_TRANSFER_BUCKETS
                        Base async prefetch depth hint used to size layerwise
                        copy-lane concurrency. Default is 1.
  --layerwise-text-transfer-buckets LAYERWISE_TEXT_TRANSFER_BUCKETS
                        Override text encoder layerwise transfer_buckets. When
                        omitted, text encoder layerwise offload uses
                        --layerwise-transfer-buckets.
  --layerwise-prefetch-limit
                        Enable the conservative future-prefetch target-count
                        limit for layerwise async transfer. When set, runtime
                        caps pending/ready future targets to min(4 *
                        --layerwise-transfer-buckets, 8). By default this
                        limit is disabled.
  --layerwise-text-prefetch-limit
                        Override text encoder layerwise prefetch_limit. When
                        omitted, text encoder layerwise offload uses
                        --layerwise-prefetch-limit.
  --no-layerwise-text-prefetch-limit
                        Disable text encoder layerwise prefetch_limit even if
                        --layerwise-prefetch-limit is enabled.
  --layerwise-max-copy-streams LAYERWISE_MAX_COPY_STREAMS
                        Maximum number of async CUDA copy streams used by
                        cache-dit layerwise offload. This shared setting
                        applies to both generic module layerwise offload and
                        SVDQ layerwise collection when --svdq-params
                        layerwise_offload=true is active. When omitted,
                        runtime derives it from --layerwise-transfer-buckets
                        and applies its internal safety cap.
  --layerwise-text-max-copy-streams LAYERWISE_TEXT_MAX_COPY_STREAMS
                        Override text encoder layerwise max_copy_streams. When
                        omitted, text encoder layerwise offload uses
                        --layerwise-max-copy-streams.
  --layerwise-max-inflight-prefetch-bytes LAYERWISE_MAX_INFLIGHT_PREFETCH_BYTES
                        Maximum total CUDA residency budget, in bytes, for in-
                        flight layerwise future-target prefetch. This shared
                        setting applies to both generic module layerwise
                        offload and SVDQ layerwise collection when --svdq-
                        params layerwise_offload=true is active. Accepts raw
                        bytes or suffixes like 512MiB and 4GiB. When omitted,
                        runtime leaves the byte-budget limit disabled.
  --layerwise-text-max-inflight-prefetch-bytes LAYERWISE_TEXT_MAX_INFLIGHT_PREFETCH_BYTES
                        Override text encoder layerwise
                        max_inflight_prefetch_bytes. Accepts raw bytes or
                        suffixes like 512MiB and 4GiB. When omitted, text
                        encoder layerwise offload uses --layerwise-max-
                        inflight-prefetch-bytes.
  --layerwise-persistent-buckets LAYERWISE_PERSISTENT_BUCKETS
                        How many selected layerwise offload targets should
                        stay resident on CUDA for the full handle lifetime.
                        This shared setting applies to both generic module
                        layerwise offload and SVDQ layerwise collection when
                        --svdq-params layerwise_offload=true is active.
                        Default is 0.
  --layerwise-text-persistent-buckets LAYERWISE_TEXT_PERSISTENT_BUCKETS
                        Override text encoder layerwise persistent_buckets.
                        When omitted, text encoder layerwise offload uses
                        --layerwise-persistent-buckets.
  --layerwise-persistent-bins LAYERWISE_PERSISTENT_BINS
                        How many evenly distributed bins should be used when
                        placing persistent layerwise offload targets across
                        the selected target list. A value of 1 keeps the
                        original prefix behavior. For example, with 32
                        targets, --layerwise-persistent-buckets 16 and
                        --layerwise-persistent-bins 4, runtime keeps [0:4),
                        [8:12), [16:20), and [24:28) resident. This shared
                        setting applies to both generic module layerwise
                        offload and SVDQ layerwise collection when --svdq-
                        params layerwise_offload=true is active. Default is 1.
  --layerwise-text-persistent-bins LAYERWISE_TEXT_PERSISTENT_BINS
                        Override text encoder layerwise persistent_bins. When
                        omitted, text encoder layerwise offload uses
                        --layerwise-persistent-bins.
  --device-map-balance, --device-map
                        Enable automatic device map balancing model if
                        multiple GPUs are available.
  --vae-tiling          Enable VAE tiling for low memory device.
  --vae-slicing         Enable VAE slicing for low memory device.
  --compile             Enable compile for transformer, only compile the
                        repeated blocks by default.
  --disable-compile-repeated-blocks, --disable-compile-blocks, --no-regional-compile, --no-rc
                        Disable compile for repeated blocks in transformer
  --force-compile-dynamic
                        Force set the compiled transformer to dynamic mode.
  --cuda-graph          Enable compile with CUDA Graph for transformer if
                        applicable.
  --compile-full-graph, --full-graph
                        Enable compile with full graph requirement for
                        transformer.
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
                        Device for torch.Generator, e.g., 'cuda' or 'cpu'.
                        Defaults to the current rank's accelerator, falling
                        back to CPU when there is none. CPU sampling is not
                        reproducible across torch versions: torch 2.12 changed
                        the bf16 `randn` kernel.
  --saved-fps SAVED_FPS, --fps SAVED_FPS
                        Export generated video with specified fps
  --example-summary, --esummary
                        Enable example summary logging
```
