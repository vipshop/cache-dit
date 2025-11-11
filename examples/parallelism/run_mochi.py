import os
import sys

sys.path.append("..")

import torch
from diffusers import MochiPipeline

pipe = MochiPipeline.from_pretrained(
    os.environ.get(
        "MOCHI_DIR",
        "genmo/mochi-1-preview",
    ),
    torch_dtype=torch.bfloat16,
)

for m in pipe.transformer.named_modules():
    if isinstance(m[1], torch.nn.Linear):
        print(m[0], m[1])
