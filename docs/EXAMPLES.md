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
pip3 install torch==2.9.1 transformers accelerate torchao bitsandbytes torchvision 
pip3 install opencv-python-headless einops imageio-ffmpeg ftfy 
pip3 install git+https://github.com/huggingface/diffusers.git # latest or >= 0.36.0
pip3 install git+https://github.com/vipshop/cache-dit.git # latest
```

## Available Examples

```bash
python3 generate.py list  # list all available examples

[generate.py:47] Available examples:
[generate.py:53] - ✅ flux_nunchaku                  - Defalut: nunchaku-tech/nunchaku-flux.1-dev
[generate.py:53] - ✅ flux                           - Defalut: black-forest-labs/FLUX.1-dev
[generate.py:53] - ✅ flux2                          - Defalut: black-forest-labs/FLUX.2-dev
[generate.py:53] - ✅ qwen_image_lightning           - Defalut: lightx2v/Qwen-Image-Lightning
[generate.py:53] - ✅ qwen_image_2512                - Defalut: Qwen/Qwen-Image-2512
[generate.py:53] - ✅ qwen_image                     - Defalut: Qwen/Qwen-Image
[generate.py:53] - ✅ qwen_image_edit_2511_lightning - Defalut: lightx2v/Qwen-Image-Edit-2511-Lightning
[generate.py:53] - ✅ qwen_image_edit_2511           - Defalut: Qwen/Qwen-Image-Edit-2511
[generate.py:53] - ✅ qwen_image_edit_lightning      - Defalut: lightx2v/Qwen-Image-Lightning
[generate.py:53] - ✅ qwen_image_edit                - Defalut: Qwen/Qwen-Image-Edit-2509
[generate.py:53] - ✅ qwen_image_controlnet          - Defalut: InstantX/Qwen-Image-ControlNet-Inpainting
[generate.py:53] - ✅ qwen_image_layered             - Defalut: Qwen/Qwen-Image-Layered
[generate.py:53] - ✅ ltx2_t2v                       - Defalut: Lightricks/LTX-2
[generate.py:53] - ✅ ltx2_i2v                       - Defalut: Lightricks/LTX-2
[generate.py:53] - ✅ skyreels_v2                    - Defalut: Skywork/SkyReels-V2-T2V-14B-720P-Diffusers
[generate.py:53] - ✅ wan2.2_t2v                     - Defalut: Wan-AI/Wan2.2-T2V-A14B-Diffusers
[generate.py:53] - ✅ wan2.1_t2v                     - Defalut: Wan-AI/Wan2.1-T2V-1.3B-Diffusers
[generate.py:53] - ✅ wan2.2_i2v                     - Defalut: Wan-AI/Wan2.2-I2V-A14B-Diffusers
[generate.py:53] - ✅ wan2.1_i2v                     - Defalut: Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
[generate.py:53] - ✅ wan2.2_vace                    - Defalut: linoyts/Wan2.2-VACE-Fun-14B-diffusers
[generate.py:53] - ✅ wan2.1_vace                    - Defalut: Wan-AI/Wan2.1-VACE-1.3B-diffusers
[generate.py:53] - ✅ ovis_image                     - Defalut: AIDC-AI/Ovis-Image-7B
[generate.py:53] - ✅ zimage_nunchaku                - Defalut: nunchaku/nunchaku-z-image-turbo
[generate.py:53] - ✅ zimage                         - Defalut: Tongyi-MAI/Z-Image-Turbo
[generate.py:53] - ✅ zimage_controlnet_2.0          - Defalut: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0
[generate.py:53] - ✅ zimage_controlnet_2.1          - Defalut: alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1
[generate.py:53] - ✅ longcat_image                  - Defalut: meituan-longcat/LongCat-Image
[generate.py:53] - ✅ longcat_image_edit             - Defalut: meituan-longcat/LongCat-Image-Edit
```

## Single GPU Inference

The easiest way to enable hybrid cache acceleration for DiTs with cache-dit is to start with single GPU inference. For examples:  

```bash
# baseline
# use default model path, e.g, "black-forest-labs/FLUX.1-dev"
python3 generate.py flux 
python3 generate.py flux_nunchaku # need nunchaku library
python3 generate.py flux2
python3 generate.py ovis_image
python3 generate.py qwen_image_edit_lightning
python3 generate.py qwen_image
python3 generate.py ltx2_t2v --parallel-vae --parallel-text-encoder --cache
python3 generate.py ltx2_i2v --parallel-vae --parallel-text-encoder --cache
torchrun --nproc_per_node=4 generate.py ltx2_t2v --parallel ulysses --parallel-vae --parallel-text-encoder --cache
torchrun --nproc_per_node=4 generate.py ltx2_t2v --parallel tp --parallel-vae --parallel-text-encoder --cache
torchrun --nproc_per_node=4 generate.py ltx2_i2v --parallel ulysses --parallel-vae --parallel-text-encoder --cache
torchrun --nproc_per_node=4 generate.py ltx2_i2v --parallel tp --parallel-vae --parallel-text-encoder --cache
python3 generate.py skyreels_v2
python3 generate.py wan2.2
python3 generate.py zimage 
python3 generate.py zimage_nunchaku 
python3 generate.py zimage_controlnet_2.1 
python3 generate.py generate longcat_image
python3 generate.py generate longcat_image_edit
# w/ cache acceleration
python3 generate.py flux --cache
python3 generate.py flux --cache --taylorseer
python3 generate.py flux_nunchaku --cache
python3 generate.py qwen_image --cache
python3 generate.py zimage --cache --rdt 0.6 --scm fast
python3 generate.py zimage_controlnet_2.1 --cache --rdt 0.6 --scm fast
# enable cpu offload or vae tiling if your encounter an OOM error
python3 generate.py qwen_image --cache --cpu-offload
python3 generate.py qwen_image --cache --cpu-offload --vae-tiling
python3 generate.py qwen_image_edit_lightning --cpu-offload --steps 4
python3 generate.py qwen_image_edit_lightning --cpu-offload --steps 8
# or, enable sequential cpu offload for extremly low VRAM device
python3 generate.py flux2 --sequential-cpu-offload # FLUX2 56B total
# use `--summary` option to show the cache acceleration stats
python3 generate.py zimage --cache --rdt 0.6 --scm fast --summary
```

## Custom Model Path

The default model path are the official model names on HuggingFace Hub. Users can set custom local model path by settig `--model-path`. For examples: 

```bash
python3 generate.py flux --model-path /PATH/TO/FLUX.1-dev
python3 generate.py zimage --model-path /PATH/TO/Z-Image-Turbo
python3 generate.py qwem_image --model-path /PATH/TO/Qwen-Image
```

## Multi-GPU Inference 

cache-dit is designed to work seamlessly with CPU or Sequential Offloading, 🔥Context Parallelism, 🔥Tensor Parallelism. For examples:

```bash
# context parallelism or tensor parallelism
torchrun --nproc_per_node=4 generate.py flux --parallel ulysses 
torchrun --nproc_per_node=4 generate.py flux --parallel ring 
torchrun --nproc_per_node=4 generate.py flux --parallel tp
torchrun --nproc_per_node=4 generate.py zimage --parallel ulysses 
torchrun --nproc_per_node=4 generate.py zimage_controlnet_2.1 --parallel ulysses 
# ulysses anything attention
torchrun --nproc_per_node=4 generate.py zimage --parallel ulysses --ulysses-anything
torchrun --nproc_per_node=4 generate.py qwen_image_edit_lightning --parallel ulysses --ulysses-anything
# text encoder parallelism, enable it by add: `--parallel-text-encoder`
torchrun --nproc_per_node=4 generate.py flux --parallel tp --parallel-text-encoder
torchrun --nproc_per_node=4 generate.py qwen_image_edit_lightning --parallel ulysses --ulysses-anything --parallel-text-encoder
# Hint: set `--local-ranks-filter=0` to torchrun -> only show logs on rank 0
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py flux --parallel ulysses 
```

## Low-bits Quantization 

cache-dit is designed to work seamlessly with torch.compile, Quantization (🔥torchao, 🔥nunchaku), For examples:

```bash
# please also enable torch.compile if the quantation is using.
python3 generate.py flux --cache --quantize-type float8 --compile
python3 generate.py flux --cache --quantize-type int8 --compile
python3 generate.py flux --cache --quantize-type float8_weight_only --compile
python3 generate.py flux --cache --quantize-type int8_weight_only --compile
python3 generate.py flux --cache --quantize-type bnb_4bit --compile # w4a16
python3 generate.py flux_nunchaku --cache --compile # w4a16 SVDQ
```

## Hybrid Acceleration 

Here are some examples for `hybrid cache acceleration + parallelism` for popular DiTs with cache-dit.

```bash
# DBCache + SCM + Taylorseer
python3 generate.py flux --cache --scm fast --taylorsees --taylorseer-order 1
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + CUDNN Attention (--attn _sdpa_cudnn)
torchrun --nproc_per_node=4 generate.py flux --parallel ulysses --ulysses-float8 \
         --attn _sdpa_cudnn --parallel-text-encoder --cache --scm fast --taylorseer \
         --taylorseer-order 1 --quantize-type float8 --warmup 2 --repeat 5 --compile 
# DBCache + SCM + Taylorseer + Context Parallelism + Text Encoder Parallelism + Compile 
# + FP8 quantization + FP8 All2All comm + FP8 SageAttention (--attn sage)
torchrun --nproc_per_node=4 generate.py flux --parallel ulysses --ulysses-float8 \
         --attn sage --parallel-text-encoder --cache --scm fast --taylorseer \
         --taylorseer-order 1 --quantize-type float8 --warmup 2 --repeat 5 --compile 
# Case: Hybrid Acceleration for Qwen-Image-Edit-Lightning, tracking memory usage.
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py qwen_image_edit_lightning \
         --parallel ulysses --ulysses-anything --parallel-text-encoder \
         --quantize-type float8_weight_only --steps 4 --track-memory --compile
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py qwen_image_edit_lightning \
         --parallel tp --parallel-text-encoder --quantize-type float8_weight_only \
         --steps 4 --track-memory --compile
# Case: Hybrid Acceleration + Context Parallelism + ControlNet Parallelism, e.g, Z-Image-ControlNet
torchrun --nproc_per_node=4 generate.py zimage_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --rdt 0.6 --scm fast
torchrun --nproc_per_node=4 generate.py zimage_controlnet_2.1 --parallel ulysses \
         --parallel-controlnet --cache --scm fast --rdt 0.6 --compile \
         --compile-controlnet --ulysses-float8 --attn _sdpa_cudnn \
         --warmup 2 --repeat 4     
```

## End2End Examples

```bash
# NO Cache Acceleration: 8.27s
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py flux --parallel ulysses

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
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py flux --parallel ulysses --cache --scm fast

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

It is very easy to add a new example. Please refer to the specific implementation in [registers.py](https://github.com/vipshop/cache-dit/raw/main/examples/registers.py). For example:

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

# NOTE: DON'T forget to add `flux_example` into helpers.py
```

## More Usages about Examples

```bash
python3 generate.py --help

usage: generate.py [-h] [--model-path MODEL_PATH] [--controlnet-path CONTROLNET_PATH] [--lora-path LORA_PATH] [--transformer-path TRANSFORMER_PATH] [--image-path IMAGE_PATH] [--mask-image-path MASK_IMAGE_PATH] [--prompt PROMPT]
                   [--negative-prompt NEGATIVE_PROMPT] [--num_inference_steps NUM_INFERENCE_STEPS] [--warmup WARMUP] [--repeat REPEAT] [--height HEIGHT] [--width WIDTH] [--seed SEED] [--num-frames NUM_FRAMES] [--save-path SAVE_PATH] [--cache]
                   [--cache-summary] [--Fn-compute-blocks FN_COMPUTE_BLOCKS] [--Bn-compute-blocks BN_COMPUTE_BLOCKS] [--residual-diff-threshold RESIDUAL_DIFF_THRESHOLD] [--max-warmup-steps MAX_WARMUP_STEPS] [--warmup-interval WARMUP_INTERVAL]
                   [--max-cached-steps MAX_CACHED_STEPS] [--max-continuous-cached-steps MAX_CONTINUOUS_CACHED_STEPS] [--taylorseer] [--taylorseer-order TAYLORSEER_ORDER] [--steps-mask] [--mask-policy {None,slow,s,medium,m,fast,f,ultra,u}]
                   [--quantize] [--quantize-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}] [--quantize-text-encoder]
                   [--quantize-text-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}] [--quantize-controlnet]
                   [--quantize-controlnet-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}] [--parallel-type {None,tp,ulysses,ring}] [--parallel-vae]
                   [--parallel-text-encoder] [--parallel-controlnet] [--attn {None,flash,_flash_3,native,_native_cudnn,_sdpa_cudnn,sage}] [--ulysses-anything] [--ulysses-float8] [--ulysses-async] [--cpu-offload]
                   [--sequential-cpu-offload] [--device-map-balance] [--vae-tiling] [--vae-slicing] [--compile] [--compile-repeated-blocks] [--compile-vae] [--compile-text-encoder] [--compile-controlnet] [--max-autotune] [--track-memory]
                   [--profile] [--profile-name PROFILE_NAME] [--profile-dir PROFILE_DIR] [--profile-activities {CPU,GPU,MEM} [{CPU,GPU,MEM} ...]] [--profile-with-stack] [--profile-record-shapes] [--disable-fuse-lora DISABLE_FUSE_LORA]
                   [{generate,list,flux_nunchaku,flux,flux2,qwen_image_lightning,qwen_image,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,skyreels_v2,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage,zimage_controlnet,longcat_image,longcat_image_edit}]
                   [{None,flux_nunchaku,flux,flux2,qwen_image_lightning,qwen_image,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,skyreels_v2,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage,zimage_controlnet,longcat_image,longcat_image_edit}]

positional arguments:
  {generate,list,flux_nunchaku,flux,flux2,qwen_image_lightning,qwen_image,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,skyreels_v2,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage,zimage_controlnet,longcat_image,longcat_image_edit}
                        The task to perform or example name to run. Use 'list' to list all available examples, or specify an example name directly (defaults to 'generate' task).
  {None,flux_nunchaku,flux,flux2,qwen_image_lightning,qwen_image,qwen_image_edit_lightning,qwen_image_edit,qwen_image_controlnet,skyreels_v2,wan2.2_t2v,wan2.1_t2v,wan2.2_i2v,wan2.1_i2v,wan2.2_vace,wan2.1_vace,ovis_image,zimage,zimage_controlnet,longcat_image,longcat_image_edit}
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
  --prompt PROMPT       Override default prompt if provided
  --negative-prompt NEGATIVE_PROMPT
                        Override default negative prompt if provided
  --num_inference_steps NUM_INFERENCE_STEPS, --steps NUM_INFERENCE_STEPS
                        Number of inference steps
  --warmup WARMUP       Number of warmup steps before measuring performance
  --repeat REPEAT       Number of times to repeat the inference for performance measurement
  --height HEIGHT       Height of the generated image
  --width WIDTH         Width of the generated image
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
  --quantize-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}, --q-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}
  --quantize-text-encoder, --q-text
                        Enable quantization for text encoder
  --quantize-text-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}, --q-text-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}
  --quantize-controlnet, --q-controlnet
                        Enable quantization for text encoder
  --quantize-controlnet-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}, --q-controlnet-type {None,float8,float8_weight_only,float8_wo,int8,int8_weight_only,int8_wo,int4,int4_weight_only,int4_wo,bitsandbytes_4bit,bnb_4bit}
  --parallel-type {None,tp,ulysses,ring}, --parallel {None,tp,ulysses,ring}
  --parallel-vae        Enable VAE parallelism if applicable.
  --parallel-text-encoder, --parallel-text
                        Enable text encoder parallelism if applicable.
  --parallel-controlnet
                        Enable ControlNet parallelism if applicable.
  --attn {None,flash,_flash_3,native,_native_cudnn,_sdpa_cudnn,sage}
  --ulysses-anything, --uaa
                        Enable Ulysses Anything Attention for context parallelism
  --ulysses-float8, --ufp8
                        Enable Ulysses Attention/UAA Float8 for context parallelism
  --ulysses-async, --uaqkv
                        Enabled experimental Async QKV Projection with Ulysses for context parallelism
  --cpu-offload, --cpu-offload-model
                        Enable CPU offload for model if applicable.
  --sequential-cpu-offload
                        Enable sequential GPU offload for model if applicable.
  --device-map-balance, --device-map
                        Enable automatic device map balancing model if multiple GPUs are available.
  --vae-tiling          Enable VAE tiling for low memory device.
  --vae-slicing         Enable VAE slicing for low memory device.
  --compile             Enable compile for transformer
  --compile-repeated-blocks
                        Enable compile for repeated blocks in transformer
  --compile-vae         Enable compile for VAE
  --compile-text-encoder, --compile-text
                        Enable compile for text encoder
  --compile-controlnet  Enable compile for ControlNet
  --max-autotune        Enable max-autotune mode for torch.compile
  --track-memory        Track and report peak GPU memory usage
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
```
