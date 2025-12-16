import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)

from utils import (
    get_args,
    strify,
    build_cache_dit_optimization,
    maybe_init_distributed,
    maybe_destroy_distributed,
    pipe_quant_bnb_4bit_config,
    is_optimization_flags_enabled,
    MemoryTracker,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

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
    quantization_config=pipe_quant_bnb_4bit_config(args),
)

assert isinstance(pipe.transformer, QwenImageTransformer2DModel)

# Apply cache and context parallelism here
if is_optimization_flags_enabled(args):
    build_cache_dit_optimization(args, pipe)


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

pipe.set_progress_bar_config(disable=rank != 0)

height = 1024 if args.height is None else args.height
width = 1024 if args.width is None else args.width


def run_pipe(warmup: bool = False):
    # do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    input_prompt = prompt + positive_magic["en"]
    output = pipe(
        prompt=input_prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=((50 if args.steps is None else args.steps) if not warmup else 5),
        true_cfg_scale=4.0,
        generator=torch.Generator(device="cpu").manual_seed(0),
        output_type="latent" if args.perf else "pil",
    )
    image = output.images[0] if not args.perf else None
    return image


# warmup
_ = run_pipe(warmup=True)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()


if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"qwen-image.{height}x{width}.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    if not args.perf:
        print(f"Saving image to {save_path}")
        image.save(save_path)

maybe_destroy_distributed()
