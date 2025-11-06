import os

import torch
from diffusers import ChromaPipeline


pipe = ChromaPipeline.from_pretrained(
    os.environ.get(
        "CHROMA1_DIR",
        "lodestones/Chroma1-HD",
    ),
    torch_dtype=torch.bfloat16,
)

pipe.to("cuda")

for m in pipe.transformer.name_modules():
    if isinstance(m[1], torch.nn.Linear):
        print(m[0], m[1])
