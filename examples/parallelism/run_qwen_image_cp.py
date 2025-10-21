import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel

from utils import (
    GiB,
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

if GiB() < 96:
    print("Apply FP8 Weight Only Quantize ...")
    args.quantize_type = "fp8_w8a16_wo"  # force
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
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

pipe.to(device)

pipe.vae.enable_tiling()

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# using an empty string if you do not have specific concept to remove
negative_prompt = " "


# Generate with different aspect ratios
aspect_ratios = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

width, height = aspect_ratios["16:9"]

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)


def run_pipe():
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

stats = cache_dit.summary(pipe)

if rank == 0:
    time_cost = end - start
    save_path = f"qwen-image.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed(args)
