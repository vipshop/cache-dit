# Low-bits Quantization

<div id="quantization"></div>

## Overview

Quantization is a powerful technique to reduce the memory footprint and computational cost of deep learning models by representing weights and activations with lower precision data types. Cache-DiT supports various quantization methods, including FP8, INT8, and INT4 quantization, to help users achieve faster inference and lower memory usage while maintaining acceptable model performance.

|quantization type| description|devices|
|:---|:---|:---| 
|<span style="color:#c77dff;">float8_per_row</span> |quantize weights and activations to float8 (dynamic quantization) with rowwise method. (**<span style="color:#c77dff;">recommended</span>**)|<span style="color:#c77dff;">>=sm89</span>, Ada, Hopper or newer|
|<span style="color:#c77dff;">float8_per_tensor</span>|quantize weights and activations to float8 (dynamic quantization) with tensorwise method.|<span style="color:#c77dff;">>=sm89</span>, Ada, Hopper or newer|
|<span style="color:#c77dff;">float8_per_block</span>|block-wise quantization (dynamic quantization) to float8, which can provide better precision, activations's blocksize: (1, 128), weight's blocksize: (128, 128) |<span style="color:#c77dff;">>=sm89</span>, Ada, Hopper or newer|
|<span style="color:#c77dff;">float8_weight_only</span>|quantize only weights to float8, keep activations in full precision|<span style="color:#c77dff;">>=sm89</span>, Ada, Hopper or newer|
|<span style="color:#c77dff;">int8_per_row</span>|quantize weights and activations to int8 (dynamic quantization) with rowwise method.|<span style="color:#c77dff;">>=sm80</span>, Ampere or newer|
|<span style="color:#c77dff;">int8_per_tensor</span>|quantize weights and activations to int8 (dynamic quantization) with tensorwise method.|<span style="color:#c77dff;">>=sm80</span>, Ampere or newer|
|<span style="color:#c77dff;">int8_weight_only</span>|quantize only weights to int8, keep activations in full precision|<span style="color:#c77dff;">>=sm80</span>, Ampere or newer|
|<span style="color:#c77dff;">int4_weight_only</span>|quantize only weights to int4, keep activations in full precision|<span style="color:#c77dff;">>=sm90</span>, Hopper or newer, TMA required|


## FP8 Quantization

Currently, TorchAo has been integrated into Cache-DiT as the backend for <span style="color:#c77dff;">online</span> quantization. You can implement model quantization by calling <span style="color:#c77dff;">quantize</span> or pass a <span style="color:#c77dff;">QuantizeConfig</span> to <span style="color:#c77dff;">enable_cache</span> API. (recommended)

For GPUs with low memory capacity, we recommend using <span style="color:#c77dff;">float8</span>, <span style="color:#c77dff;">float8_weight_only</span>, as these methods cause almost no loss in precision. Supported quantization types including:  

  - <span style="color:#c77dff;">float8_per_row</span>: quantize both weights and activations to float8 (dynamic quantization) with rowwise method.  
  - <span style="color:#c77dff;">float8_per_tensor</span>: quantize both weights and activations to float8 (dynamic quantization) with tensorwise method.  
  - <span style="color:#c77dff;">float8_per_block</span>: block-wise quantization (dynamic quantization) to float8, which can provide better precision, activations's blocksize: (1, 128), weight's blocksize: (128, 128).  
  - <span style="color:#c77dff;">float8_weight_only</span>: quantize only weights to float8, keep activations in full precision.  

Here are some examples of how to use quantization with cache-dit. You can directly specify the quantization config in the <span style="color:#c77dff;">enable_cache</span> API.

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

# quant_type: float8, float8_weight_only, int8, int8_weight_only, etc.
# Pass a QuantizeConfig to the `enable_cache` API.
cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(quant_type="float8_per_row"),
)
```

Users can also specify different quantization configs for different components. For example, quantize the <span style="color:#c77dff;">transformer</span> to <span style="color:#c77dff;">float8_per_row</span> and the <span style="color:#c77dff;">text encoder</span> to <span style="color:#c77dff;">float8_weight_only</span>.

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(
        components_to_quantize={
            "transformer": {
                "quant_type": "float8_per_row",
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
    quantize_config=QuantizeConfig(quant_type="float8_per_row"),
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
        quant_type="float8_per_row",
        exclude_layers=["embedder", "embed"],
    ),
)
```
By default, quant_type="float8_per_row" for better precision. Users can set it to "float8_per_tensor" to use per-tensor quantization for better performance on some hardware.

## Regional Quantization

Cache-DiT also supports <span style="color:#c77dff;">regional quantization</span>, which allows users to quantize only the repeated blocks in a transformer. This can be useful for better balancing the <span style="color:#c77dff;">precision</span> and efficiency. Users can specify the blocks to be quantized via the <span style="color:#c77dff;">regional_quantize</span> and <span style="color:#c77dff;">repeated_blocks</span> arguments in <span style="color:#c77dff;">QuantizeConfig</span>. For example, to quantize repeated blocks of the Flux2's transformer:

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(ulysses_size=2),
    quantize_config=QuantizeConfig(
        quant_type="float8_per_row",
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

The <span style="color:#c77dff;">per_tensor_fallback</span> option in Cache-DiT's quantization configuration allows users to enable a fallback mechanism for layers that do not support float8 per-row or per-block quantization. This is particularly useful in scenarios where tensor parallelism is applied, and certain layers (e.g., those applied with RowwiseParallel) may encounter memory layout mismatch errors when quantized to float8 per-row.

When <span style="color:#c77dff;">per_tensor_fallback</span> is set to True, if a layer cannot be quantized to float8 per-row or per-block, it will automatically fall back to float8 per-tensor quantization instead of raising an error. This ensures that the quantization process can continue smoothly without interruption, while still providing the benefits of reduced precision for supported layers.  

To enable this feature, simply set the <span style="color:#c77dff;">per_tensor_fallback</span> flag to <span style="color:#c77dff;">True (default)</span> in the <span style="color:#c77dff;">QuantizeConfig</span> when calling the <span style="color:#c77dff;">enable_cache</span> API. Only support for float8 quantization for now. For example:

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig

cache_dit.enable_cache( 
    pipe, cache_config=DBCacheConfig(), # w/ default
    parallelism_config=ParallelismConfig(tp_size=2),
    quantize_config=QuantizeConfig(
        quant_type="float8_per_row",
        # Must be True to enable fp8 per-tensor fallback.
        regional_quantize=True, # default, True.
        repeated_blocks=['Flux2TransformerBlock', 'Flux2SingleTransformerBlock'],
        # Enable fallback to float8 per-tensor quantization, default to True
        # for better compatibility for layers that do not support float8 per-row 
        # quantization, e.g., layers with RowwiseParallel applied in tensor parallelism.
        per_tensor_fallback=True, 
    ),
)
```

For examples, without fp8 per-tensor fallback, the cache-dit will auto skip the layers that do not support float8 per-row quantization, and raise warning for those layers. The performance will be worse due to less layers being quantized. (<span style="color:#c77dff;">quantize 88 layers, skip 56 layers</span>)

```bash
# w/o fp8 per-tensor fallback, quantize 88 layers, skip 56 layers, performance downgrade.
torchrun --nproc_per_node=2 -m cache_dit.generate flux2_klein_9b_kv_edit \
   --parallel tp --compile --float8-per-row --q-verbose \
   --disable-per-tensor-fallback

--------------------------------------------------------------------------------------------
Quantized                 Method: float8_per_row                                            |
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
```

With fp8 per-tensor fallback enabled, those layers that do not support float8 per-row quantization will be quantized to float8 per-tensor instead, and the performance will be better due to more layers being quantized. (<span style="color:#c77dff;">quantize 144 layers, skip 0 layer</span>)

```bash
# w/ fp8 per-tensor fallback enabled, quantize 144 layers, skip 0 layer, better performance.
torchrun --nproc_per_node=2 -m cache_dit.generate flux2_klein_9b_kv_edit \
   --parallel tp --compile --float8-per-row --q-verbose  

# default, enabled fp8 per-tensor fallback
--------------------------------------------------------------------------------------------
Quantized                 Method: float8_per_row                                            |
Quantized                 Region: ['Flux2TransformerBlock', 'Flux2SingleTransformerBlock']  |
Quantized    Basic Linear Layers: 88                                                        |
Quantized Fallback Linear Layers: 56 (per_tensor)                                           |
Total    Quantized Linear Layers: 144                                                       |
Skipped      Basic Linear Layers: 0                                                         |
Skipped   Fallback Linear Layers: 0                                                         |
Total      Skipped Linear Layers: 0                                                         |
Total              Linear Layers: 144                                                       |
--------------------------------------------------------------------------------------------
```

## INT8/INT4 Quantization

In addition to FP8 quantization, Cache-DiT also supports INT8 and INT4 quantization for weights, which can further reduce the memory footprint of the model. Users can specify <span style="color:#c77dff;">int8_per_row</span>, <span style="color:#c77dff;">int8_per_tensor</span>, <span style="color:#c77dff;">int8_weight_only</span>, or <span style="color:#c77dff;">int4_weight_only</span> as the quantization type in the <span style="color:#c77dff;">QuantizeConfig</span> when calling the <span style="color:#c77dff;">enable_cache</span> API. For example:

```python
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig  

cache_dit.enable_cache( 
    # Or "int8_per_tensor", "int8_weight_only", "int4_weight_only", etc.
    pipe, quantize_config=QuantizeConfig(quant_type="int8_per_row"), 
)
```
INT4 quantization can provide even better memory reduction compared to FP8 or INT8, but it may cause more precision loss. We recommend users to try different quantization types and choose the one that best fits their needs in terms of the trade-off between performance and precision. In most cases, <span style="color:#c77dff;">float8 per-row</span> can be a good choice for better memory reduction while maintaining acceptable precision.

Please note that users should also install <span style="color:#c77dff;">mslk</span> kernel library to enable INT8/INT4 quantization features. The <span style="color:#c77dff;">int4_weight_only</span> w4a16 compute kennel requires architectures >= <span style="color:#c77dff;">sm90</span> (Hopper or newer, TMA required). For older architectures, users can use <span style="color:#c77dff;">int8_weight_only</span> quantization for better compatibility. 

```bash
# stable: mslk, torch and torchao (change cu130 to cu129 if using CUDA 12.9)
uv pip install torch==2.11.0 torchvision torchao triton mslk --index-url https://download.pytorch.org/whl/cu130 --upgrade
# nightly: mslk, torch and torchao (change cu130 to cu129 if using CUDA 12.9)
uv pip install --pre torch torchvision torchao triton mslk --index-url https://download.pytorch.org/whl/nightly/cu130 --upgrade
```
In the case of <span style="color:#c77dff;">distributed inference</span> (context parallelism or tensor parallelism), we recommend users to use <span style="color:#c77dff;">float8 quantization</span> to avoid potential compatibility issues.

## Nunchaku (W4A4)

Cache-DiT natively supports the <span style="color:#c77dff;">Hybrid Cache + Nunchaku + Context Parallelism</span> scheme. Users can leverage caching and context parallelism to speed up Nunchaku <span style="color:#c77dff;">4-bits W4A4</span> models. 

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
