import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

model_id = os.environ.get(
    "PIXART_SIGMA_DIR",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
)
transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
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

start = time.time()
prompt = "A small cactus with a happy face in the Sahara desert."
image = pipe(
    prompt,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)
time_cost = end - start
save_path = f"pixart-sigma.{strify(args, stats)}.png"

print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
