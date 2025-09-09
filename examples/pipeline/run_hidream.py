import os
import sys

sys.path.append("..")

import time
import torch
from diffusers import HiDreamImagePipeline
from transformers import AutoTokenizer, LlamaForCausalLM
from diffusers.quantizers import PipelineQuantizationConfig
from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig

from utils import get_args, strify
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

start = time.time()
image = pipe(
    'A cute girl holding a sign that says "Hi-Dreams.ai".',
    height=1024,
    width=1024,
    guidance_scale=5.0,
    num_inference_steps=50,
    generator=torch.Generator("cpu").manual_seed(0),
).images[0]
end = time.time()

stats = cache_dit.summary(pipe)

time_cost = end - start
save_path = f"hidream.{strify(args, stats)}.png"
print(f"Time cost: {time_cost:.2f}s")
print(f"Saving to {save_path}")
image.save(save_path)
