import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from utils import GiB, get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


pipe = QwenImagePipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "QWEN_IMAGE_DIR",
            "Qwen/Qwen-Image",
        )
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=("balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None),
)

if args.cache:
    cachify(args, pipe)

# When device_map is None, we need to explicitly move the model to GPU
# or enable CPU offload to avoid running on CPU
if torch.cuda.device_count() <= 1:
    # Single GPU: use CPU offload for memory efficiency
    pipe.enable_model_cpu_offload()
elif torch.cuda.device_count() > 1 and pipe.device.type == "cpu":
    # Multi-GPU but model is on CPU (device_map was None): move to default GPU
    pipe.to("cuda")


positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

if args.prompt is not None:
    prompt = args.prompt
# using an empty string if you do not have specific concept to remove
negative_prompt = " "
if args.negative_prompt is not None:
    negative_prompt = args.negative_prompt


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

# Use command line args if provided, otherwise default to 16:9
if args.width is not None and args.height is not None:
    width, height = args.width, args.height
else:
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
        num_inference_steps=50 if args.steps is None else args.steps,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]


memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
# do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
image = pipe(
    prompt=prompt + positive_magic["en"],
    negative_prompt=negative_prompt,
    width=width,
    height=height,
    num_inference_steps=50 if args.steps is None else args.steps,
    true_cfg_scale=4.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
