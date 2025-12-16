import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AutoModel, SkyReelsV2Pipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from utils import (
    get_args,
    strify,
    cachify,
    maybe_init_distributed,
    maybe_destroy_distributed,
    pipe_quant_bnb_4bit_config,
    is_optimzation_flags_enabled,
    MemoryTracker,
)
import cache_dit


args = get_args()
print(args)

rank, device = maybe_init_distributed(args)

model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("SKYREELS_V2_DIR", "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers")
)

vae = AutoModel.from_pretrained(
    model_id,
    subfolder="vae",
    torch_dtype=torch.float32,
).to("cuda")

pipe = SkyReelsV2Pipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
    quantization_config=pipe_quant_bnb_4bit_config(
        args,
        components_to_quantize=["text_encoder", "transformer"],
    ),
).to("cuda")

flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

if is_optimzation_flags_enabled(args):
    cachify(args, pipe)

pipe.set_progress_bar_config(disable=rank != 0)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

if args.prompt is not None:
    prompt = args.prompt


def run_pipe(pipe: SkyReelsV2Pipeline):
    video = pipe(
        prompt=prompt,
        num_inference_steps=50 if args.steps is None else args.steps,
        height=720,  # 720 for 720P
        width=1280,  # 1280 for 720P
        num_frames=21,
        generator=torch.Generator("cpu").manual_seed(0),
    ).frames[0]
    return video


# warmup
_ = run_pipe(pipe)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = run_pipe(pipe)
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

if rank == 0:
    cache_dit.summary(pipe, details=True)

    time_cost = end - start
    save_path = f"skyreels_v2.{strify(args, pipe)}.mp4"
    print(f"Time cost: {time_cost:.2f}s")
    print(f"Saving video to {save_path}")
    export_to_video(video, save_path, fps=8, quality=8)

maybe_destroy_distributed()
