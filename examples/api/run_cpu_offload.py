import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from utils import get_args, strify, cachify
import cache_dit


parser = get_args(parse=False)
parser.add_argument(
    "--offload-type",
    type=str,
    choices=["model", "sequential", "group"],
    default="model",
)
parser.add_argument(
    "--cache-after-offload",
    action="store_true",
    default=False,
)
args = parser.parse_args()
print(args)


pipe = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache and not args.cache_after_offload:
    print("Enabled Cache before offload")
    cachify(args, pipe)

if torch.cuda.device_count() <= 1:
    # Enable memory savings
    if args.offload_type == "model":
        print("Enabled Model CPU Offload")
        pipe.enable_model_cpu_offload()
    elif args.offload_type == "sequential":
        print("Enabled Sequential CPU Offload")
        pipe.enable_sequential_cpu_offload()
    elif args.offload_type == "group":
        print("Enabled Group Offload")
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="leaf_level",
            use_stream=True,
        )

if args.cache and args.cache_after_offload:
    print("Enabled Cache after offload")
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
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
    ).images[0]


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
save_path = f"qwen-image.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
