# Low-bits Quantization

<div id="quantization"></div>

## TorchAo

Currently, torchao has been integrated into cache-dit as the backend for **online** model quantization (with more backends to be supported in the future). You can implement model quantization by calling `cache_dit.quantize(...)`. At present, cache-dit supports the `Hybrid Cache + Low-bits Quantization` scheme. For GPUs with low memory capacity, we recommend using `float8_weight_only` or `int8_weight_only`, as these two schemes cause almost no loss in precision.

```python
# pip3 install "cache-dit[quantization]"
import cache_dit

cache_dit.enable_cache(pipe_or_adapter)

# float8, float8_weight_only, int8, int8_weight_only, int4, int4_weight_only
# int4_weight_only requires fbgemm-gpu-genai>=1.2.0, which only supports
# Compute Architectures >= Hopper (and does not support Ada, ..., etc.)
pipe.transformer = cache_dit.quantize(
    pipe.transformer, quant_type="float8_weight_only"
)
pipe.text_encoder = cache_dit.quantize(
    pipe.text_encoder, quant_type="float8_weight_only"
)
```

## bitsandbytes  

For **4-bits W4A16 (weight only)** quantization, we recommend `nf4` from **bitsandbytes** due to its better compatibility for many devices. Users can directly use it via the `quantization_config` of diffusers. For example:

```python
from diffusers import QwenImagePipeline
from diffusers.quantizers import PipelineQuantizationConfig

pipe = QwenImagePipeline.from_pretrained(
    "Qwen/Qwen-Image",
    torch_dtype=torch.bfloat16,
    quantization_config=(
        PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder", "transformer"],
        )
    ),
).to("cuda")

# Then, apply cache acceleration using cache-dit
cache_dit.enable_cache(pipe, cache_config=...)
```

## Nunchaku 

cache-dit natively supports the `Hybrid Cache + 🔥Nunchaku SVDQ INT4/FP4 + Context Parallelism` scheme. Users can leverage caching and context parallelism to speed up Nunchaku **4-bit** models. 

```python
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"path-to/svdq-int4_r32-qwen-image.safetensors"
)
pipe = QwenImagePipeline.from_pretrained(
   "Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16,
).to("cuda")

cache_dit.enable_cache(pipe, cache_config=..., parallelism_config=...)
```
