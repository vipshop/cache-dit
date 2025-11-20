import os
import sys

sys.path.append("..")

import time
import torch

from PIL import Image
from diffusers import QwenImageEditPipeline, QwenImageTransformer2DModel
from utils import GiB, get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

pipe = QwenImageEditPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "QWEN_IMAGE_EDIT_DIR",
            "Qwen/Qwen-Image-Edit",
        )
    ),
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=("balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None),
)

if args.cache:
    cachify(args, pipe)

if torch.cuda.device_count() <= 1:
    # Enable memory savings
    pipe.enable_model_cpu_offload()


image = Image.open("../data/bear.png").convert("RGB")
prompt = "Only change the bear's color to purple"
if args.prompt is not None:
    prompt = args.prompt

if args.compile:
    assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
    torch._dynamo.config.recompile_limit = 1024
    torch._dynamo.config.accumulated_recompile_limit = 8192
    pipe.transformer.compile_repeated_blocks(mode="default")

    # Warmup
    image = pipe(
        image=image,
        prompt=prompt,
        negative_prompt=" ",
        generator=torch.Generator(device="cpu").manual_seed(0),
        true_cfg_scale=4.0,
        num_inference_steps=50,
    ).images[0]

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()

image = pipe(
    image=image,
    prompt=prompt,
    negative_prompt=" ",
    generator=torch.Generator(device="cpu").manual_seed(0),
    true_cfg_scale=4.0,
    num_inference_steps=50,
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"qwen-image-edit.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
