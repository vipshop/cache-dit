import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import VisualClozePipeline
from diffusers.utils import load_image
from utils import get_args, strify
import cache_dit


args = get_args()
print(args)

# Load the VisualClozePipeline
pipe = VisualClozePipeline.from_pretrained(
    os.environ.get(
        "VISUAL_CLOZE_DIR",
        "VisualCloze/VisualClozePipeline-512",
    ),
    resolution=512,
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
start = time.time()

image = pipe(
    task_prompt=task_prompt,
    content_prompt=content_prompt,
    image=image_paths,
    upsampling_width=1024,
    upsampling_height=1024,
    upsampling_strength=0.4,
    guidance_scale=30,
    num_inference_steps=30,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0][0]

end = time.time()

cache_dit.summary(pipe)

time_cost = end - start
save_path = f"visualcloze-512.{strify(args, pipe)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving image to {save_path}")
image.save(save_path)
