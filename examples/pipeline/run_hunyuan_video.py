import os
import sys

sys.path.append("..")

import time
import torch
from diffusers.utils import export_to_video
from diffusers import HunyuanVideoPipeline, AutoencoderKLHunyuanVideo
from utils import GiB, get_args, strify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get(
    "HUNYUAN_VIDEO_DIR", "hunyuanvideo-community/HunyuanVideo"
)
pipe = HunyuanVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    # https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement
    device_map=(
        "balanced" if (torch.cuda.device_count() > 1 and GiB() <= 48) else None
    ),
)


if args.cache:
    cache_dit.enable_cache(
        pipe,
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

assert isinstance(pipe.vae, AutoencoderKLHunyuanVideo)

# Enable memory savings
if GiB() <= 48:
    pipe.vae.enable_tiling(
        # Make it runnable on GPUs with 48GB memory
        tile_sample_min_height=128,
        tile_sample_stride_height=96,
        tile_sample_min_width=128,
        tile_sample_stride_width=96,
        tile_sample_min_num_frames=32,
        tile_sample_stride_num_frames=24,
    )
else:
    pipe.vae.enable_tiling()

prompt = "A fluffy teddy bear sits on a bed of soft pillows surrounded by children's toys."

start = time.time()
output = pipe(
    prompt=prompt,
    num_frames=18,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hunyuan_video.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(output, save_path, fps=9)
