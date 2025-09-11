import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import (
    BitsAndBytesConfig as DiffusersBitsAndBytesConfig,
    AllegroTransformer3DModel,
    AllegroPipeline,
)
from diffusers.utils import export_to_video
from transformers import (
    BitsAndBytesConfig as BitsAndBytesConfig,
    T5EncoderModel,
)
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get("ALLEGRO_DIR", "rhymes-ai/Allegro")

quant_config = BitsAndBytesConfig(load_in_8bit=True)
text_encoder_8bit = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

quant_config = DiffusersBitsAndBytesConfig(load_in_8bit=True)
transformer_8bit = AllegroTransformer3DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)

pipe = AllegroPipeline.from_pretrained(
    model_id,
    text_encoder=text_encoder_8bit,
    transformer=transformer_8bit,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")


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
