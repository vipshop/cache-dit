import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import HunyuanDiTPipeline
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    MemoryTracker,
)
import cache_dit


args = get_args()
print(args)


model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get(
        "HUNYUAN_DIT_DIR",
        "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers",
        # "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers",
    )
)

# Initialize distributed for tensor parallelism
rank, device = maybe_init_distributed(args)

pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)

if args.cache or args.parallel_type is not None:
    cachify(args, pipe)

torch.cuda.empty_cache()
pipe.enable_model_cpu_offload(device=device)

# You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt = "An astronaut riding a horse on Mars"

if args.prompt is not None:
    prompt = args.prompt
memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt,
    num_inference_steps=args.steps if args.steps else 50,
    generator=torch.Generator(device).manual_seed(0),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

if rank == 0:
    time_cost = end - start
    version = "11" if "1.1" in model_id else "12"
    save_path = f"hunyuan_dit_{version}.{strify(args, stats)}.png"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving to {save_path}")
    image.save(save_path)

maybe_destroy_distributed()
