# Low-bits Quantization

<div id="quantization"></div>

## TorchAo

Currently, torchao has been integrated into cache-dit as the backend for **online** model quantization (with more backends to be supported in the future). You can implement model quantization by calling `cache_dit.quantize(...)` or pass a `QuantizeConfig` to `cache_dit.enable_cache(...)`. At present, cache-dit supports the `Hybrid Cache + Low-bits Quantization` scheme. For GPUs with low memory capacity, we recommend using `float8`, `float8_weight_only`, `int8_weight_only`, as these methods cause almost no loss in precision. Supported quantization types including:  
  - `float8`: quantize both weights and activations to float8 (dynamic quantization).
  - `float8_weight_only`: quantize only weights to float8, keep activations in full precision (weight-only quantization).
  - `int8`: quantize both weights and activations to int8 (dynamic quantization).
  - `int8_weight_only`: quantize only weights to int8, keep activations in full precision (weight-only quantization).
  - `float8_blockwise`: block-wise quantization to float8, which can provide better performance on some hardware.

Here are some examples for quick start:

```python
# pip3 install "cache-dit[quantization]"
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

# quant_type: float8, float8_weight_only, int8, int8_weight_only, etc.
# Pass a QuantizeConfig to the `enable_cache` API.
cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(quant_type="float8"),
)

# Or, directly call the `quantize` API for more fine-grained control.
cache_dit.quantize(
    pipe.transformer, 
    quantize_config=QuantizeConfig(quant_type="float8"),
)
cache_dit.quantize(
    pipe.text_encoder, 
    quantize_config=QuantizeConfig(quant_type="float8_weight_only"),
)

# Please also enable torch.compile for better performance with quantization.
cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)
pipe.text_encoder = torch.compile(pipe.text_encoder)
```

Users can set `exclude_layers` in `QuantizeConfig` to exclude some sensitive layers that are not robust to quantization, e.g., embedding layers. Layers that contain any of the keywords in the `exclude_layers` list will be excluded from quantization. For example: 

```python
cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(
        quant_type="float8",
        per_row=True, # default, True.
        exclude_layers=["embedder", "embed"],
    ),
)
```
The `per_row` flag indicates whether to use per-row quantization (for float8 dynamic quantization), default to `True` for better precision. Users can set it to False to use per-tensor quantization for better performance on some hardware.

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
