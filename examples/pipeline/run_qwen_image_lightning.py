import os
import sys

sys.path.append("..")

import time
import torch
import math
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
    FlowMatchEulerDiscreteScheduler,
)
from utils import GiB, get_args, strify
import cache_dit


args = get_args()
print(args)


# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    scheduler=scheduler,
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

steps = 8 if args.steps is None else args.steps
assert steps in [8, 4]

pipe.load_lora_weights(
    os.environ.get(
        "QWEN_IMAGE_LIGHT_DIR",
        "lightx2v/Qwen-Image-Lightning",
    ),
    weight_name=(
        "Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors"
        if steps > 4
        else "Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors"
    ),
)

if args.fuse_lora:
    pipe.fuse_lora()
    pipe.unload_lora_weights()


if args.cache:
    cache_dit.enable_cache(
        pipe,
        # Cache context kwargs
        Fn_compute_blocks=16,
        Bn_compute_blocks=16,
        max_warmup_steps=4 if steps > 4 else 2,
        max_cached_steps=2 if steps > 4 else 1,
        max_continuous_cached_steps=1,
        enable_spearate_cfg=False,  # true_cfg_scale=1.0
        residual_diff_threshold=0.50 if steps > 4 else 0.8,
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

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

if args.quantize:
    # Apply Quantization (default: FP8 DQ) to Transformer
    pipe.transformer = cache_dit.quantize(
        pipe.transformer,
        quant_type=args.quantize_type,
        per_row=False,
        exclude_layers=[
            "img_in",
            "txt_in",
            "embedder",
            "embed",
            "norm_out",
            "proj_out",
        ],
    )

if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer.compile_repeated_blocks(fullgraph=True)

    # warmup
    image = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=1.0,  # means no separate cfg
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]


start = time.time()
# do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=steps,
    true_cfg_scale=1.0,  # means no separate cfg
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"qwen-image-lightning.{steps}steps.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
