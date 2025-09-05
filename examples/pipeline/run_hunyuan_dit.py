import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import HunyuanDiTPipeline
from utils import get_args
import cache_dit


args = get_args()
print(args)


model_id = os.environ.get(
    "HUNYUAN_DIT_DIR", "Tencent-Hunyuan/HunyuanDiT-Diffusers"
)

pipe = HunyuanDiTPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

if args.cache:
    cache_dit.enable_cache(pipe)

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"

start = time.time()
image = pipe(
    "一个宇航员在骑马",
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hunyuan_dit.{cache_dit.strify(stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
