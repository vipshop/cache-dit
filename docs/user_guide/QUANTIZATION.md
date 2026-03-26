# Low-bits Quantization

<div id="quantization"></div>

## Cache-DiT w/ TorchAo

Currently, TorchAo has been integrated into Cache-DiT as the backend for <span style="color:#c77dff;">online</span> quantization. You can implement model quantization by calling <span style="color:#c77dff;">quantize</span> or pass a <span style="color:#c77dff;">QuantizeConfig</span> to <span style="color:#c77dff;">enable_cache</span> API. (recommended)

For GPUs with low memory capacity, we recommend using <span style="color:#c77dff;">float8</span>, <span style="color:#c77dff;">float8_weight_only</span>, <span style="color:#c77dff;">int8_weight_only</span>, as these methods cause almost no loss in precision. Supported quantization types including:  

  - <span style="color:#c77dff;">float8</span>: quantize both weights and activations to float8 (dynamic quantization).  
  - <span style="color:#c77dff;">float8_weight_only</span>: quantize only weights to float8, keep activations in full precision.  
  - <span style="color:#c77dff;">int8</span>: quantize both weights and activations to int8 (dynamic quantization).  
  - <span style="color:#c77dff;">int8_weight_only</span>: quantize only weights to int8, keep activations in full precision.  
  - <span style="color:#c77dff;">float8_blockwise</span>: block-wise quantization to float8, which can provide better precision.

Here are some examples of how to use quantization with cache-dit. You can directly specify the quantization config in the <span style="color:#c77dff;">enable_cache</span> API.

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

Users can also specify different quantization configs for different components. For example, quantize the transformer to float8 and the text encoder to float8_weight_only.

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

Or, directly call the <span style="color:#c77dff;">quantize</span> API for more fine-grained control.

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

Please also enable <span style="color:#c77dff;">torch.compile</span> for better performance with quantization.

```python
import cache_dit

cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)
pipe.text_encoder = torch.compile(pipe.text_encoder)
```

Users can set <span style="color:#c77dff;">exclude_layers</span> in <span style="color:#c77dff;">QuantizeConfig</span> to exclude some sensitive layers that are not robust to quantization, e.g., embedding layers. Layers that contain any of the keywords in the <span style="color:#c77dff;">exclude_layers</span> list will be excluded from quantization. For example: 

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
The <span style="color:#c77dff;">per_row</span> flag indicates whether to use per-row quantization (for float8 dynamic quantization), default to <span style="color:#c77dff;">True</span> for better precision. Users can set it to False to use per-tensor quantization for better performance on some hardware.

## Regional Quantization

Cache-DiT also supports <span style="color:#c77dff;">regional quantization</span>, which allows users to quantize only the repeated blocks in a transformer. This can be useful for better balancing the <span style="color:#c77dff;">precision</span> and efficiency. Users can specify the blocks to be quantized via the <span style="color:#c77dff;">regional_quantize</span> and <span style="color:#c77dff;">repeated_blocks</span> arguments in <span style="color:#c77dff;">QuantizeConfig</span>. For example, to quantize repeated blocks of the Flux2's transformer:

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
        regional_quantize=True, 
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

## FP8 Per-Tensor Fallback

The <span style="color:#c77dff;">float8_per_tensor_fallback</span> option in Cache-DiT's quantization configuration allows users to enable a fallback mechanism for layers that do not support float8 per-row or per-block quantization. This is particularly useful in scenarios where tensor parallelism is applied, and certain layers (e.g., those applied with RowwiseParallel) may encounter memory layout mismatch errors when quantized to float8 per-row.

When <span style="color:#c77dff;">float8_per_tensor_fallback</span> is set to True, if a layer cannot be quantized to float8 per-row or per-block, it will automatically fall back to float8 per-tensor quantization instead of raising an error. This ensures that the quantization process can continue smoothly without interruption, while still providing the benefits of reduced precision for supported layers.  

To enable this feature, simply set the <span style="color:#c77dff;">float8_per_tensor_fallback</span> flag to <span style="color:#c77dff;">True (default)</span> in the <span style="color:#c77dff;">QuantizeConfig</span> when calling the <span style="color:#c77dff;">enable_cache</span> API. For example:

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig
cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(tp_size=2),
    quantize_config=QuantizeConfig(
        quant_type="float8",
        per_row=True, # default, True.
        # Enable fallback to float8 per-tensor quantization, default to True
        # for better compatibility for layers that do not support float8 per-row 
        # quantization, e.g., layers with RowwiseParallel applied in tensor parallelism.
        float8_per_tensor_fallback=True, 
    ),
)
```
Quick examples of fp8 per-tensor fallback in action:

```bash
# w/o fp8 per-tensor fallback, the cache-dit will auto skip the layers that do not support float8 per-row quantization, and raise warning for those layers. The performance will be worse due to less layers being quantized. (quantize 88 layers, skip 56 layers)

torchrun --nproc_per_node=2 -m cache_dit.generate flux2_klein_9b_kv_edit \
   --parallel tp --compile --float8 --q-verbose \
   --disable-float8-per-tensor-fallback

--------------------------------------------------------------------------------------------
Quantized                 Method: float8                                                    |
Quantized                 Region: ['Flux2TransformerBlock', 'Flux2SingleTransformerBlock']  |
Quantized    Basic Linear Layers: 88                                                        |
Quantized Fallback Linear Layers: 0                                                         |
Total    Quantized Linear Layers: 88                                                        |
Skipped      Basic Linear Layers: 56                                                        |
Skipped   Fallback Linear Layers: 0                                                         |
Total      Skipped Linear Layers: 56                                                        |
Total              Linear Layers: 144                                                       |
--------------------------------------------------------------------------------------------
-----------------------------------------------------------------------------
Skip: attn.to_out.0        : pattern<Rowwise(Tensor Parallel)>: 8    layers  |
Skip: attn.to_add_out      : pattern<Rowwise(Tensor Parallel)>: 8    layers  |
Skip: ff.linear_out        : pattern<Rowwise(Tensor Parallel)>: 8    layers  |
Skip: ff_context.linear_out: pattern<Rowwise(Tensor Parallel)>: 8    layers  |
Skip: attn.to_out          : pattern<Rowwise(Tensor Parallel)>: 24   layers  |
-----------------------------------------------------------------------------

# w/ fp8 per-tensor fallback enabled, those layers that do not support float8 per-row quantization will be quantized to float8 per-tensor instead, and the performance will be better due to more layers being quantized. (quantize 144 layers, skip 0 layer)

torchrun --nproc_per_node=2 -m cache_dit.generate flux2_klein_9b_kv_edit \
   --parallel tp --compile --float8 --q-verbose # default, enabled fp8 per-tensor fallback

--------------------------------------------------------------------------------------------
Quantized                 Method: float8                                                    |
Quantized                 Region: ['Flux2TransformerBlock', 'Flux2SingleTransformerBlock']  |
Quantized    Basic Linear Layers: 88                                                        |
Quantized Fallback Linear Layers: 56                                                        |
Total    Quantized Linear Layers: 144                                                       |
Skipped      Basic Linear Layers: 0                                                         |
Skipped   Fallback Linear Layers: 0                                                         |
Total      Skipped Linear Layers: 0                                                         |
Total              Linear Layers: 144                                                       |
--------------------------------------------------------------------------------------------
```


## Bitsandbytes (W4A16)

For <span style="color:#c77dff;">4-bits W4A16</span> (weight only) quantization, we recommend `nf4` from **bitsandbytes** due to its better compatibility for many devices. Users can directly use it via the `quantization_config` of diffusers. For example:

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

## Nunchaku (W4A4)

cache-dit natively supports the `Hybrid Cache + 🔥Nunchaku SVDQ INT4/FP4 + Context Parallelism` scheme. Users can leverage caching and context parallelism to speed up Nunchaku <span style="color:#c77dff;">4-bit W4A4</span> models. 

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
