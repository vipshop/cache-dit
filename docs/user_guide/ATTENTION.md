# Attention Backend

## Available backend

Cache-DiT supports multiple Attention backends for better performance. The supported list is as follows:

|backend|details|parallelism|attn_mask|
|:---|:---|:---|:---|  
|<span style="color:#c77dff;">native</span>| Native SDPA Attention, w/ cache-dit optimized|✅|✅|  
|<span style="color:#c77dff;">_sdpa_cudnn</span>| CUDNN Attention via SDPA API, w/ cache-dit optimized|✅|✅|
|<span style="color:#c77dff;">_native_cudnn</span>| CUDNN Attention via SDPA API, w/o cache-dit optimized|✅|✖️|
|<span style="color:#c77dff;">flash</span>| official FlashAttention-2|✅|✖️| 
|<span style="color:#c77dff;">_flash_3</span>| official FlashAttention-3|✅|✖️|
|<span style="color:#c77dff;">sage</span>| FP8 SageAttention|✅|✖️|
|<span style="color:#c77dff;">ffpa</span>| FFPA CUDA fp16/bf16 (sm_120 / sm_89)|✅|✅|
|<span style="color:#c77dff;">ffpa_fp8</span>| FFPA CUDA FP8 (sm_120 / sm_89)|✅|✅|
|<span style="color:#c77dff;">ffpa_fp8_per_block</span>| FFPA CUDA FP8 per-block quant (sm_120 / sm_89)|✅|✅|
|<span style="color:#c77dff;">ffpa_fp4</span>| FFPA CUDA NVFP4 (sm_120 only)|✅|✅|
|<span style="color:#c77dff;">_native_npu</span>| Optimized Ascend NPU Attention|✅|✅|
|<span style="color:#c77dff;">_npu_fia</span>| NPU Attention for Ring Parallelism|✅|✅|


## Single GPU Inference

Users can specify Attention backend by setting the <span style="color:#c77dff;">attention_backend</span> parameter of <span style="color:#c77dff;">enable_cache</span> API or use <span style="color:#c77dff;">set_attn_backend</span> interface directly.  

```python
import cache_dit

# Setting the `attention_backend` parameter of `enable_cache` API
cache_dit.enable_cache(pipe_or_adapter, ..., attention_backend="_sdpa_cudnn")
# Or, use `set_attn_backend` interface directly.  
cache_dit.set_attn_backend(pipe_or_adapter, attention_backend="_sdpa_cudnn")
```


## Distributed inference

Users also can specify Attention backend by setting the <span style="color:#c77dff;">attention_backend</span> parameter of <span style="color:#c77dff;">parallelism_config</span> in the cases of distributed inference:

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
  pipe_or_adapter, 
  cache_config=DBCacheConfig(...),
  parallelism_config=ParallelismConfig(
    ulysses_size=2, # or, tp_size=2
    # flash, native(sdpa), _native_cudnn, _sdpa_cudnn, sage
    attention_backend="_sdpa_cudnn",
  ),
)
```

## FP8 Attention

<div id="fp8-attention"></div>

For FP8 Attention, users must install `sage-attention`. Then, pass the <span style="color:#c77dff;">sage</span> attention backend to the <span style="color:#c77dff;">parallelism_config</span> as an extra parameter. Please note that <span style="color:#c77dff;">attention mask</span> is not currently supported for FP8 sage attention.

```python
# pip3 install git+https://github.com/thu-ml/SageAttention.git 
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
  pipe_or_adapter, 
  cache_config=DBCacheConfig(...),
  parallelism_config=ParallelismConfig(
    ulysses_size=2, # or, tp_size=2
    attention_backend="sage",
  ),
)
```

## FFPA Attention

<div id="ffpa-attention"></div>

[ffpa-attn](https://github.com/vipshop/ffpa-attn) provides hand-written CUDA (CUTE_TMA / cp.async) prefill attention kernels for large head dims. Install its CUDA wheel (the plain PyPI package ships no CUDA kernels):

```bash
pip install ffpa-attn --index-url https://vipshop.github.io/ffpa-attn/whl/cu13/ \
  --extra-index-url https://pypi.org/simple
```

Then pass one of the ffpa backends:

```python
cache_dit.enable_cache(
  pipe_or_adapter,
  cache_config=DBCacheConfig(...),
  attention_backend="ffpa_fp8",  # ffpa, ffpa_fp8, ffpa_fp8_per_block, ffpa_fp4
)
```

Hardware scope:

|backend|precision|hardware|
|:---|:---|:---|
|<span style="color:#c77dff;">ffpa</span>| fp16 / bf16|sm_120, sm_89 (Ada)|
|<span style="color:#c77dff;">ffpa_fp8</span>| FP8 quantized (int8 QK, per-thread / per-channel)|sm_120, sm_89 (Ada)|
|<span style="color:#c77dff;">ffpa_fp8_per_block</span>| FP8 per-block quant (fastest, lowest precision)|sm_120, sm_89 (Ada)|
|<span style="color:#c77dff;">ffpa_fp4</span>| NVFP4 quantized|sm_120 only|

sm_90 / sm_100 GPUs are not supported yet. On sm_89 the fp16 family runs the cp.async native kernels and the fp8 family the TMA-free persist-D kernels (compiled head dims `64 / 128 / 192`, f16 PV accumulator enforced by ffpa-attn); sm_120 runs the CUTE_TMA families.

> [!WARNING]
> Upstream fp8 attention is known to return wrong results at `D=192` (sm_120 and sm_89 share the persist-D code); `D=64` / `D=128` are unaffected. Prefer fp16/bf16, or the fp4 family, at `D=192` until the upstream fix lands.

### Attention mask contract

FFPA backends accept an attention mask on the single-GPU path. The backend only adapts unambiguous shapes and forwards the mask untouched — which positions or sequence shards are masked is the **caller's responsibility**: different models build masks in different ways, and cache-dit never infers pad or sharding semantics.

- Shapes broadcastable to `[B, H, Nq, Nkv]`: `[Nkv]` and `[B, Nkv]` are expanded to 4-D automatically; 3-D / 4-D masks pass through as-is. The last dimension must equal the key sequence length.
- `bool` masks follow the SDPA semantics (`True` = the element participates); float masks are additive biases (ffpa-attn downcasts fp32 masks to the activation dtype).
- `attn_mask` and `is_causal=True` are mutually exclusive.
- Context parallelism (Ulysses / Ring) does not support `attn_mask` for FFPA backends.
