import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import AllegroPipeline
from diffusers.utils import export_to_video
from diffusers.quantizers import PipelineQuantizationConfig
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get("ALLEGRO_DIR", "rhymes-ai/Allegro")

pipe = AllegroPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["text_encoder"],
    ),
)

pipe.to("cuda")

pipe.vae.enable_tiling()

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

prompt = (
    "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, "
    "the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this "
    "location might be a popular spot for docking fishing boats."
)


start = time.time()
video = pipe(
    prompt,
    num_frames=49,
    guidance_scale=7.5,
    max_sequence_length=512,
    num_inference_steps=100,
    generator=torch.Generator("cpu").manual_seed(0),
).frames[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"allegro.{strify(args, stats)}.mp4"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving video to {save_path}")
export_to_video(video, save_path, fps=8)
