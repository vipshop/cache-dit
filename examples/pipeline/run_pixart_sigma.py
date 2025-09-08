import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import Transformer2DModel, PixArtSigmaPipeline
from utils import get_args
import cache_dit


args = get_args()
print(args)

weight_dtype = torch.float16

model_id = os.environ.get(
    "PIXART_SIGMA_DIR",
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
)
transformer = Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe = PixArtSigmaPipeline.from_pretrained(
    model_id,
    transformer=transformer,
    torch_dtype=weight_dtype,
    use_safetensors=True,
)
pipe.to("cuda")

if args.cache:
    cache_dit.enable_cache(pipe)

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
save_path = (
    f"pixart-sigma.C{int(args.compile)}_Q{int(args.quantize)}"
    f"{'' if not args.quantize else ('_' + args.quantize_type)}_"
    f"{cache_dit.strify(stats)}.png"
)
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
