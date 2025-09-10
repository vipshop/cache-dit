import os
import sys
import gc

sys.path.append("..")

import time
import torch
from hyimage.diffusion.pipelines.hunyuanimage_pipeline import (
    HunyuanImagePipeline,
)
from hyimage.models.hunyuan.modules.hunyuanimage_dit import (
    HYImageDiffusionTransformer,
)
from utils import get_args, strify
import cache_dit

args = get_args()
print(args)


# Supported model_name: hunyuanimage-v2.1, hunyuanimage-v2.1-distilled
model_name = os.environ.get("HUNYUAN_IMAGE_DIR", "tencent/HunyuanImage-2.1")
pipe = HunyuanImagePipeline.from_pretrained(
    model_name=model_name, torch_dtype="bf16"
)

# FP8 weight only
if args.quantize:
    pipe.dit = cache_dit.quantize(
        pipe.dit,
        quant_type="fp8_w8a16_wo",
        exclude_layers=[
            "img_in",
            "txt_in",
            "time_in",
            "time_r_in",
            "guidance_in",
            "final_layer",
        ],
    )
    pipe.refiner_pipeline.dit = cache_dit.quantize(
        pipe.refiner_pipeline.dit,
        quant_type="fp8_w8a16_wo",
        exclude_layers=[
            "img_in",
            "txt_in",
            "time_in",
            "time_r_in",
            "guidance_in",
            "final_layer",
        ],
    )
    time.sleep(0.5)
    torch.cuda.empty_cache()
    gc.collect()

# pipe = pipe.to("cuda")

if args.cache:
    from cache_dit import BlockAdapter, ForwardPattern

    assert isinstance(pipe.dit, HYImageDiffusionTransformer)

    cache_dit.enable_cache(
        BlockAdapter(
            pipe=pipe,
            transformer=pipe.dit,
            blocks=[
                pipe.dit.double_blocks,
                pipe.dit.single_blocks,
            ],
            forward_pattern=[
                ForwardPattern.Pattern_0,
                ForwardPattern.Pattern_3,
            ],
        ),
        # Cache context kwargs
        Fn_compute_blocks=args.Fn,
        Bn_compute_blocks=args.Bn,
        max_warmup_steps=args.max_warmup_steps,
        max_cached_steps=args.max_cached_steps,
        max_continuous_cached_steps=args.max_continuous_cached_steps,
        enable_taylorseer=args.taylorseer,
        enable_encoder_taylorseer=args.taylorseer,
        taylorseer_order=args.taylorseer_order,
        residual_diff_threshold=args.rdt,
    )

prompt = "A cute, cartoon-style anthropomorphic penguin plush toy with fluffy fur, standing in a painting studio, wearing a red knitted scarf and a red beret with the word “Tencent” on it, holding a paintbrush with a focused expression as it paints an oil painting of the Mona Lisa, rendered in a photorealistic photographic style."

if args.compile:
    cache_dit.set_compile_configs()
    pipe.dit = torch.compile(pipe.dit)
    pipe.refiner_pipeline.dit = torch.compile(pipe.refiner_pipeline.dit)

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
        width=2048,
        height=2048,
        use_reprompt=True,  # Enable prompt enhancement
        use_refiner=True,  # Enable refiner model
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
    width=2048,
    height=2048,
    use_reprompt=True,  # Enable prompt enhancement
    use_refiner=True,  # Enable refiner model
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
