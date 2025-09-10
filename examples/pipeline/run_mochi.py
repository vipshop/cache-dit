import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

pipe = MochiPipeline.from_pretrained(
    os.environ.get(
        "MOCHI_DIR",
        "genmo/mochi-1-preview",
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

pipe.enable_vae_tiling()

prompt = (
    "Close-up of a chameleon's eye, with its scaly skin "
    "changing color. Ultra high resolution 4k."
)

start = time.time()
video = pipe(
    prompt,
    num_frames=49,
    num_inference_steps=64,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"mochi.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=10)
