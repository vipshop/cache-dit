import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import OmniGenPipeline
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)


model_id = (
    args.model_path
    if args.model_path is not None
    else os.environ.get("OMNIGEN_DIR", "Shitao/OmniGen-v1-diffusers")
)

pipe = OmniGenPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
prompt = "Realistic photo. A young woman sits on a sofa, holding a book and facing the camera. She wears delicate silver hoop earrings adorned with tiny, sparkling diamonds that catch the light, with her long chestnut hair cascading over her shoulders. Her eyes are focused and gentle, framed by long, dark lashes. She is dressed in a cozy cream sweater, which complements her warm, inviting smile. Behind her, there is a table with a cup of water in a sleek, minimalist blue mug. The background is a serene indoor setting with soft natural light filtering through a window, adorned with tasteful art and flowers, creating a cozy and peaceful ambiance. 4K, HD."
if args.prompt is not None:
    prompt = args.prompt
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=3,
    num_inference_steps=50,
    generator=torch.Generator(device="cpu").manual_seed(111),
).images[0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"omingen-v1.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
