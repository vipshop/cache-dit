import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

pipe = MochiPipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "MOCHI_DIR",
            "genmo/mochi-1-preview",
        )
    ),
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer", "text_encoder"],
    ),
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

pipe.enable_vae_tiling()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)


if args.prompt is not None:

    prompt = args.prompt
memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = pipe(
    prompt,
    num_frames=49,
    num_inference_steps=64,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"mochi.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=10)
