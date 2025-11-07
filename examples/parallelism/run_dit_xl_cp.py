import os
import sys

sys.path.append("..")

import time
import torch

from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from utils import (
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

pipe = DiTPipeline.from_pretrained(
    os.environ.get(
        "DIT_XL_DIR",
        "facebook/DiT-XL-2-256",
    ),
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

words = ["white shark"]

class_ids = pipe.get_label_ids(words)


def run_pipe():
    image = pipe(
        class_labels=class_ids,
        num_inference_steps=25,
        generator=torch.Generator("cpu").manual_seed(33),
    ).images[0]
    return image


# warmup
_ = run_pipe()

start = time.time()
image = run_pipe()
end = time.time()

if rank == 0:
    cache_dit.summary(pipe)

    time_cost = end - start
    save_path = f"dit-xl.{strify(args, pipe)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving image to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
