import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image
from utils import get_args, strify, cachify, MemoryTracker
import cache_dit


args = get_args()
print(args)

# Load the VisualClozePipeline
pipe = VisualClozePipeline.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "VISUAL_CLOZE_DIR",
            "VisualCloze/VisualClozePipeline-512",
        )
    ),
    resolution=512,
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

# Load in-context images (make sure the paths are correct and accessible)
# The images are from the VITON-HD dataset at https://github.com/shadow2496/VITON-HD
image_paths = [
    # in-context examples
    [
        load_image("../data/visualcloze/00700_00.jpg"),
        load_image("../data/visualcloze/03673_00.jpg"),
        load_image("../data/visualcloze/00700_00_tryon_catvton_0.jpg"),
    ],
    # query with the target image
    [
        load_image("../data/visualcloze/00555_00.jpg"),
        load_image("../data/visualcloze/12265_00.jpg"),
        None,
    ],
]

# Task and content prompt
task_prompt = "Each row shows a virtual try-on process that aims to put [IMAGE2] the clothing onto [IMAGE1] the person, producing [IMAGE3] the person wearing the new clothing."
content_prompt = None

# Run the pipeline
memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()

image = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_height=1632,
    upsampling_width=1232,
    upsampling_strength=0.3,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0][0]

end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"visualcloze-512.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
