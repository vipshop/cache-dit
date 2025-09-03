import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from utils import GiB, get_args
import cache_dit


args = get_args()
print(args)


pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)


if args.cache:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    from cache_dit import BlockAdapter, ForwardPattern

    cache_dit.enable_cache(
        BlockAdapter(
            # Any DiffusionPipeline, Qwen-Image, etc.
            pipe=pipe,
            auto=True,
            # Check `📚Forward Pattern Matching` documentation and hack the code of
            # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
            forward_pattern=ForwardPattern.Pattern_1,
        ),
        # Cache context kwargs
        enable_spearate_cfg=True,
        enable_taylorseer=True,
        enable_encoder_taylorseer=True,
        taylorseer_order=4,
        residual_diff_threshold=0.12,
    )


if torch.cuda.device_count() <= 1:
    # Enable memory savings
    pipe.enable_model_cpu_offload()


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

start = time.time()

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

end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image.adapter.{cache_dit.strify(stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
