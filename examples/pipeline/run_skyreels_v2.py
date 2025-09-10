import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AutoModel, SkyReelsV2Pipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from utils import get_args, GiB, strify
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get(
    "SKYREELS_V2_DIR", "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers"
)
vae = AutoModel.from_pretrained(
    model_id, subfolder="vae", torch_dtype=torch.float32
)

pipe = SkyReelsV2Pipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)

flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
pipe.scheduler = UniPCMultistepScheduler.from_config(
    pipe.scheduler.config, flow_shift=flow_shift
)

if args.cache:
    cache_dit.enable_cache(
        pipe,
        # Cache context kwargs
        Fn_compute_blocks=args.Fn,
        Bn_compute_blocks=args.Bn,
        max_warmup_steps=args.max_warmup_steps,
        max_cached_steps=args.max_cached_steps,
        max_continuous_cached_steps=args.max_continuous_cached_steps,
        enable_taylorseer=args.taylorseer,
        enable_encoder_taylorseer=args.taylorseer,
        taylorseer_order=args.taylorseer_order,
        residual_diff_threshold=args.rdt,
    )

prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."

start = time.time()
video = pipe(
    prompt=prompt,
    num_inference_steps=50,
    height=544,  # 720 for 720P
    width=960,  # 1280 for 720P
    num_frames=97,
).frames[0]
end = time.time()

cache_dit.summary(pipe, details=True)

time_cost = end - start
save_path = f"skyreels_v2.{strify(args, pipe)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=24, quality=8)
