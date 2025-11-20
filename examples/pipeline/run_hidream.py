import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import HiDreamImagePipeline
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from utils import get_args, strify, cachify, MemoryTracker
import cache_dit

args = get_args()
print(args)

tokenizer_4 = AutoTokenizer.from_pretrained(
    (
        args.model_path
        if args.model_path is not None
        else os.environ.get(
            "LLAMA_DIR",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
        )
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
    quantization_config=TransformersBitsAndBytesConfig(
        load_in_4bit=True,
    ),
)

pipe = HiDreamImagePipeline.from_pretrained(
    os.environ.get(
        "HIDREAM_DIR",
        "HiDream-ai/HiDream-I1-Full",
    ),
    tokenizer_4=tokenizer_4,
    text_encoder_4=text_encoder_4,
    torch_dtype=torch.bfloat16,
    quantization_config=PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["transformer"],
    ),
)

pipe.to("cuda")

if args.cache:
    cachify(args, pipe)

# Set default prompt
prompt = 'A cute girl holding a sign that says "Hi-Dreams.ai".'
if args.prompt is not None:
    prompt = args.prompt

memory_tracker = MemoryTracker() if args.track_memory else None
if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
image = pipe(
    prompt,
    height=1024 if args.height is None else args.height,
    width=1024 if args.width is None else args.width,
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hidream.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
