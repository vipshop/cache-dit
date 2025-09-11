import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Lumina2Transformer2DModel, Lumina2Pipeline
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get("LUMINA_DIR", "Alpha-VLLM/Lumina-Image-2.0")

ckpt_path = os.path.join(model_id, "consolidated.00-of-01.pth")
transformer = Lumina2Transformer2DModel.from_single_file(
    ckpt_path, torch_dtype=torch.bfloat16
)

pipe = Lumina2Pipeline.from_pretrained(
    model_id, transformer=transformer, torch_dtype=torch.bfloat16
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


start = time.time()
image = pipe(
    "a cute cat holding a sign that says hello 'Lumina2'",
    height=1024,
    width=1024,
    num_inference_steps=30,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"lumina2.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
