import os
import sys

sys.path.append("..")

import time
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers import HiDreamImagePipeline
from utils import get_args
import cache_dit

args = get_args()
print(args)


tokenizer_4 = AutoTokenizer.from_pretrained(
    os.environ.get(
        "LLAMA_DIR",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
)
text_encoder_4 = LlamaForCausalLM.from_pretrained(
    os.environ.get(
        "LLAMA_DIR",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ),
    output_hidden_states=True,
    output_attentions=True,
    torch_dtype=torch.bfloat16,
)

pipe = HiDreamImagePipeline.from_pretrained(
    os.environ.get(
        "HIDREAM_DIR",
        "HiDream-ai/HiDream-I1-Full",
    ),
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16,
)

pipe.enable_model_cpu_offload()


if args.cache:
    cache_dit.enable_cache(pipe)

start = time.time()
image = pipe(
    'A cat holding a sign that says "Hi-Dreams.ai".',
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hidream.{cache_dit.strify(stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
