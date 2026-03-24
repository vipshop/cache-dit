# Low-bits Quantization

<div id="quantization"></div>

## Cache-DiT w/ TorchAo

Currently, TorchAo has been integrated into Cache-DiT as the backend for <span style="color:hotpink;">online</span> model quantization (recommended). You can implement model quantization by calling <span style="color:hotpink;">quantize</span> or pass a <span style="color:hotpink;">QuantizeConfig</span> to <span style="color:hotpink;">enable_cache</span> API.

For GPUs with low memory capacity, we recommend using <span style="color:hotpink;">float8</span>, <span style="color:hotpink;">float8_weight_only</span>, <span style="color:hotpink;">int8_weight_only</span>, as these methods cause almost no loss in precision. Supported quantization types including:  

  - <span style="color:hotpink;">float8</span>: quantize both weights and activations to float8 (dynamic quantization).  
  - <span style="color:hotpink;">float8_weight_only</span>: quantize only weights to float8, keep activations in full precision.  
  - <span style="color:hotpink;">int8</span>: quantize both weights and activations to int8 (dynamic quantization).  
  - <span style="color:hotpink;">int8_weight_only</span>: quantize only weights to int8, keep activations in full precision.  
  - <span style="color:hotpink;">float8_blockwise</span>: block-wise quantization to float8, which can provide better precision.

Here are some examples of how to use quantization with cache-dit. You can directly specify the quantization config in the <span style="color:hotpink;">enable_cache</span> API.

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

# quant_type: float8, float8_weight_only, int8, int8_weight_only, etc.
# Pass a QuantizeConfig to the `enable_cache` API.
cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(quant_type="float8"),
)
```

Users can also specify different quantization configs for different components. For example, quantize the transformer to float8 and the text encoder to float8 weight only.

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(
        components_to_quantize={
            "transformer": {
                "quant_type": "float8",
                "exclude_layers": ["embedder", "embed"],
            },
            "text_encoder": {
                "quant_type": "float8_weight_only",
                "exclude_layers": ["lm_head"],
            }
        }
    ),
)
```

Or, directly call the <span style="color:hotpink;">quantize</span> API for more fine-grained control.

```python
import cache_dit
from cache_dit import QuantizeConfig

cache_dit.quantize(
    pipe.transformer, 
    quantize_config=QuantizeConfig(quant_type="float8"),
)
cache_dit.quantize(
    pipe.text_encoder, 
    quantize_config=QuantizeConfig(quant_type="float8_weight_only"),
)
```

Please also enable <span style="color:hotpink;">torch.compile</span> for better performance with quantization.

```python
import cache_dit

cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)
pipe.text_encoder = torch.compile(pipe.text_encoder)
```

Users can set <span style="color:hotpink;">exclude_layers</span> in <span style="color:hotpink;">QuantizeConfig</span> to exclude some sensitive layers that are not robust to quantization, e.g., embedding layers. Layers that contain any of the keywords in the <span style="color:hotpink;">exclude_layers</span> list will be excluded from quantization. For example: 

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

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
The <span style="color:hotpink;">per_row</span> flag indicates whether to use per-row quantization (for float8 dynamic quantization), default to <span style="color:hotpink;">True</span> for better precision. Users can set it to False to use per-tensor quantization for better performance on some hardware.

## Regional Quantization

Cache-DiT also supports <span style="color:hotpink;">regional quantization</span>, which allows users to quantize only part of the transformer blocks. This can be useful for better balancing the <span style="color:hotpink;">precision</span> and efficiency. Users can specify the blocks to be quantized via the <span style="color:hotpink;">quantize_repeated_blocks</span> and <span style="color:hotpink;">repeated_blocks</span> arguments in <span style="color:hotpink;">QuantizeConfig</span>. For example, to quantize repeated blocks of the Flux2's transformer:

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(
        quant_type="float8",
        # Default (True), only quantize the repeated blocks in transformer if the repeated_blocks is 
        # specified. If set to False, the whole transformer will be quantized.
        quantize_repeated_blocks=True, 
        # Specify the block names for the transformer, cache-dit will automatically find the repeated 
        # blocks and quantize it inplace. The block names can be found in the model architecture, e.g., 
        # for FLUX.2, the block name is "Flux2TransformerBlock" and "Flux2SingleTransformerBlock".
        repeated_blocks=['Flux2TransformerBlock', 'Flux2SingleTransformerBlock'],
        # repeated_blocks will be detected automatically from diffusers' transformer class, namely:
        # default repeated_blocks = transformer._repeated_blocks if exists, else None (quantize 
        # the whole transformer.
    ),
)
```

## Bitsandbytes (W4A16)

For <span style="color:hotpink;">4-bits W4A16</span> (weight only) quantization, we recommend `nf4` from **bitsandbytes** due to its better compatibility for many devices. Users can directly use it via the `quantization_config` of diffusers. For example:

```python
import cache_dit
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

## Nunchaku (SVDQ INT4/FP4, W4A4)

cache-dit natively supports the `Hybrid Cache + 🔥Nunchaku SVDQ INT4/FP4 + Context Parallelism` scheme. Users can leverage caching and context parallelism to speed up Nunchaku <span style="color:hotpink;">4-bit</span> models. 

```python
import cache_dit
from diffusers import QwenImagePipeline
from nunchaku import NunchakuQwenImageTransformer2DModel

transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"path-to/svdq-int4_r32-qwen-image.safetensors"
)
pipe = QwenImagePipeline.from_pretrained(
   "Qwen/Qwen-Image", transformer=transformer, torch_dtype=torch.bfloat16,
).to("cuda")

cache_dit.enable_cache(pipe, cache_config=..., parallelism_config=...)
```
