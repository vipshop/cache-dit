import os
import sys
import gc

sys.path.append("..")
sys.path.append(os.environ.get("HYIMAGE_PKG_DIR", "."))

import time
import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import (
    HunyuanImagePipeline,
)
from hyimage.models.hunyuan.modules.hunyuanimage_dit import (
    HYImageDiffusionTransformer,
)
from utils import get_args, strify, cachify, GiB
import cache_dit

args = get_args()
print(args)

torch.set_grad_enabled(False)

# Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled

# NOTE: This example based on PR:
# https://github.com/Tencent-Hunyuan/HunyuanImage-2.1/pull/12

# export HYIMAGE_PKG_DIR=/path/to/Tencent-Hunyuan/HunyuanImage-2.1
# export HUNYUANIMAGE_V2_1_MODEL_ROOT=/path/to/HunyuanImage-2.1
# cd $HUNYUANIMAGE_V2_1_MODEL_ROOT
# modelscope download --model AI-ModelScope/Glyph-SDXL-v2 --local_dir ./text_encoder/Glyph-SDXL-v2
# modelscope download Qwen/Qwen2.5-VL-7B-Instruct --local_dir ./text_encoder/llm
# modelscope download google/byt5-small --local_dir ./text_encoder/byt5-small
model_name = "hunyuanimage-v2.1"
pipe = HunyuanImagePipeline.from_pretrained(
    model_name=model_name,
    torch_dtype="bf16",
    # NOTE: load in CPU first, this will enable HunyuanImage run
    # on many device with low GPU VRAM (<96 GiB):
    # CPU -> GPU VRAM < 96GiB ? -> FP8 weight only on CPU -> GPU
    device="cpu" if GiB() < 96 else "cuda",
    use_compile=False,
)

if GiB() < 96:
    assert args.quantize, "Please enable quantize for low GPU memory device."

# FP8 weight only
if args.quantize:
    # Minimum VRAM required: 38 GiB
    print("Apply FP8 Weight Only Quantize ...")
    args.quantize_type = "fp8_w8a16_wo"  # force
    pipe.dit = cache_dit.quantize(
        pipe.dit,
        quant_type=args.quantize_type,
        exclude_layers=[
            "img_in",
            "txt_in",
            "time_in",
            "time_r_in",
            "guidance_in",
            "final_layer",
        ],
    )
    pipe.text_encoder = cache_dit.quantize(
        pipe.text_encoder,
        quant_type=args.quantize_type,
    )
    time.sleep(0.5)
    torch.cuda.empty_cache()
    gc.collect()


pipe.to("cuda")

if args.cache:
    from cache_dit import BlockAdapter, ForwardPattern, ParamsModifier

    assert isinstance(pipe.dit, HYImageDiffusionTransformer)

    cachify(
        args,
        BlockAdapter(
            pipe=pipe,
            transformer=pipe.dit,
            blocks=[
                pipe.dit.double_blocks,  # 20
                pipe.dit.single_blocks,  # 40
            ],
            forward_pattern=[
                ForwardPattern.Pattern_0,
                ForwardPattern.Pattern_3,
            ],
            params_modifiers=[
                ParamsModifier(Fn_compute_blocks=args.Fn),
                ParamsModifier(Fn_compute_blocks=1),
            ],
            # block forward contains 'img' and 'txt' as varible names,
            # not 'hidden_states' and 'encoder_hidden_states'.
            check_forward_pattern=False,
            check_num_outputs=False,
        ),
    )

prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."

# 1024, 1024 for low GPU memory device. Generating images with 1K
# resolution will result in artifacts.
height, width = 2048, 2048
if args.compile:
    cache_dit.set_compile_configs()
    if not getattr(pipe.config.dit_config, "use_compile", False):
        pipe.dit = torch.compile(pipe.dit)
    pipe.text_encoder = torch.compile(pipe.text_encoder)

    # warmup
    image = pipe(
        prompt=prompt,
        # Examples of supported resolutions and aspect ratios for HunyuanImage-2.1:
        # 16:9  -> width=2560, height=1536
        # 4:3   -> width=2304, height=1792
        # 1:1   -> width=2048, height=2048
        # 3:4   -> width=1792, height=2304
        # 9:16  -> width=1536, height=2560
        # Please use one of the above width/height pairs for best results.
        width=width,
        height=height,
        use_reprompt=False if GiB() < 96 else True,  # Enable prompt enhancement
        use_refiner=False if GiB() < 96 else True,  # Enable refiner model
        # For the distilled model, use 8 steps for faster inference.
        # For the non-distilled model, use 50 steps for better quality.
        num_inference_steps=8 if "distilled" in model_name else 50,
        guidance_scale=3.5,
        shift=5,
        seed=649151,
    )

start = time.time()
image = pipe(
    prompt=prompt,
    # Examples of supported resolutions and aspect ratios for HunyuanImage-2.1:
    # 16:9  -> width=2560, height=1536
    # 4:3   -> width=2304, height=1792
    # 1:1   -> width=2048, height=2048
    # 3:4   -> width=1792, height=2304
    # 9:16  -> width=1536, height=2560
    # Please use one of the above width/height pairs for best results.
    width=width,
    height=height,
    use_reprompt=False if GiB() < 96 else True,  # Enable prompt enhancement
    use_refiner=False if GiB() < 96 else True,  # Enable refiner model
    # For the distilled model, use 8 steps for faster inference.
    # For the non-distilled model, use 50 steps for better quality.
    num_inference_steps=8 if "distilled" in model_name else 50,
    guidance_scale=3.5,
    shift=5,
    seed=649151,
)
end = time.time()

stats = cache_dit.summary(pipe.dit)

time_cost = end - start
save_path = f"hunyuan-image-2.1.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
