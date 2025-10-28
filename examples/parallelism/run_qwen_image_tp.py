import os
import sys

sys.path.append("..")

import time

import torch
from diffusers import QwenImagePipeline, QwenImageTransformer2DModel
from utils import (
    cachify,
    get_args,
    maybe_destroy_distributed,
    maybe_init_distributed,
    strify,
)

import cache_dit

args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

pipe: QwenImagePipeline = QwenImagePipeline.from_pretrained(
    os.environ.get(
        "QWEN_IMAGE_DIR",
        "Qwen/Qwen-Image",
    ),
    torch_dtype=torch.bfloat16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

torch.cuda.empty_cache()
pipe.enable_model_cpu_offload(device=device)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
}

# Generate image
prompt = """A coffee shop entrance features a chalkboard sign reading "Qwen Coffee 😊 $2 per cup," with a neon light beside it displaying "通义千问". Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written "π≈3.1415926-53589793-23846264-33832795-02384197". Ultra HD, 4K, cinematic composition"""

# using an empty string if you do not have specific concept to remove
negative_prompt = " "

pipe.set_progress_bar_config(disable=rank != 0)


def run_pipe():
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    output = pipe(
        prompt=prompt + positive_magic["en"],
        negative_prompt=negative_prompt,
        width=1024 if args.width is None else args.width,
        height=1024 if args.height is None else args.height,
        num_inference_steps=50 if args.steps is None else args.steps,
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(42),
        output_type="latent" if args.perf else "pil",
    )
    image = output.images[0] if not args.perf else None
    return image


if args.compile:
    cache_dit.set_compile_configs()
    pipe.transformer = torch.compile(pipe.transformer)

# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

if rank == 0:
    time_cost = end - start
    save_path = f"qwen-image.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    if not args.perf:
        print(f"Saving image to {save_path}")
        image.save(save_path)

maybe_destroy_distributed()
