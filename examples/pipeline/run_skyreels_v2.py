import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AutoModel, SkyReelsV2Pipeline, UniPCMultistepScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


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

flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)

if args.cache:
    cachify(args, pipe)

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."


if args.prompt is not None:

    prompt = args.prompt
memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
video = pipe(
    prompt=prompt,
    num_inference_steps=50,
    height=720,  # 720 for 720P
    width=1280,  # 1280 for 720P
    num_frames=21,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"skyreels_v2.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8, quality=8)
