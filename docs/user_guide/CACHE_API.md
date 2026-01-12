# Unified Cache APIs

<div id="unified"></div>  

## Forward Pattern Matching 

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

## Cache Acceleration with One-line Code

In most cases, you only need to call **one-line** of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples/run_qwen_image.py) as an example. 

```python
import cache_dit
from diffusers import DiffusionPipeline 

# Can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
# One-line code with default cache options.
cache_dit.enable_cache(pipe) 
# Just call the pipe as normal.
output = pipe(...)
# Disable cache and run original pipe.
cache_dit.disable_cache(pipe)
```

## Automatic Block Adapter

But in some cases, you may have a **modified** Diffusion Pipeline or Transformer that is not located in the diffusers library or not officially supported by **cache-dit** at this time. The **BlockAdapter** can help you solve this problems. Please refer to [🔥Qwen-Image w/ BlockAdapter](https://github.com/vipshop/cache-dit/blob/main/examples/adapter/run_qwen_image_adapter.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter

# Use 🔥BlockAdapter with `auto` mode.
cache_dit.enable_cache(
    BlockAdapter(
        # Any DiffusionPipeline, Qwen-Image, etc.  
        pipe=pipe, auto=True,
        # Check `📚Forward Pattern Matching` documentation and hack the code of
        # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
        forward_pattern=ForwardPattern.Pattern_1,
    ),   
)

# Or, manually setup transformer configurations.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # Qwen-Image, etc.
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
    ), 
)
```
For such situations, **BlockAdapter** can help you quickly apply various cache acceleration features to your own Diffusion Pipelines and Transformers. 

## Hybrid Forward Pattern

Sometimes, a Transformer class will contain more than one transformer `blocks`. For example, **FLUX.1** (HiDream, Chroma, etc) contains transformer_blocks and single_transformer_blocks (with different forward patterns). The **BlockAdapter** can also help you solve this problem. 
```python
# For diffusers <= 0.34.0, FLUX.1 transformer_blocks and 
# single_transformer_blocks have different forward patterns.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # FLUX.1, etc.
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
)
```

Even sometimes you have more complex cases, such as **Wan 2.2 MoE**, which has more than one Transformer (namely `transformer` and `transformer_2`) in its structure. Fortunately, **cache-dit** can also handle this situation very well. Please refer to [📚Wan 2.2 MoE](https://github.com/vipshop/cache-dit/blob/main/examples) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter, ParamsModifier, DBCacheConfig

cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        transformer=[
            pipe.transformer,
            pipe.transformer_2,
        ],
        blocks=[
            pipe.transformer.blocks,
            pipe.transformer_2.blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_2,
            ForwardPattern.Pattern_2,
        ],
        # Setup different cache params for each 'blocks'. You can 
        # pass any specific cache params to ParamModifier, the old 
        # value will be overwrite by the new one.
        params_modifiers=[
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=4,
                    max_cached_steps=8,
                ),
            ),
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=2,
                    max_cached_steps=20,
                ),
            ),
        ],
        has_separate_cfg=True,
    ),
)
```

## Implement Patch Functor

For any PATTERN not in {0...5}, we introduced the simple abstract concept of **Patch Functor**. Users can implement a subclass of Patch Functor to convert an unknown Pattern into a known PATTERN, and for some models, users may also need to fuse the operations within the blocks for loop into block forward. 

![](https://github.com/vipshop/cache-dit/raw/main/assets/patch-functor.png)

Some Patch functors have already been provided in cache-dit: [📚HiDreamPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/patch_functors/functor_hidream.py), [📚ChromaPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/patch_functors/functor_chroma.py), etc. After implementing Patch Functor, users need to set the `patch_functor` property of **BlockAdapter**.

```python
@BlockAdapterRegister.register("HiDream")
def hidream_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import HiDreamImageTransformer2DModel
    from cache_dit.caching.patch_functors import HiDreamPatchFunctor

    assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.double_stream_blocks,
            pipe.transformer.single_stream_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_0,
            ForwardPattern.Pattern_3,
        ],
        # NOTE: Setup your custom patch functor here.
        patch_functor=HiDreamPatchFunctor(),
        **kwargs,
    )
```

## Transformer-Only Interface

In some cases, users may **not use Diffusers or DiffusionPipeline** at all, and may not even have the concept of a "pipeline"—for instance, **ComfyUI** (which breaks down the pipeline into individual components while still retaining transformer components). cache-dit also supports such scenarios; it only needs to be configured via **BlockAdapter**. The pipeline is not mandatory, and you can simply keep it at the default value of None. In this case, the `num_inference_steps` parameter in cache_config **must be set**, as cache-dit relies on this parameter to refresh the cache context at the appropriate time. Please refer to [📚run_transformer_only.py](https://github.com/vipshop/cache-dit/blob/main/examples/api/run_transformer_only.py) as an example.

```python
cache_dit.enable_cache(
    BlockAdapter( 
        # NO `pipe` required
        transformer=transformer,
        blocks=transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
    ), 
    cache_config=DBCacheConfig(
        num_inference_steps=50  # required
    ),
)
```

If you need to use a **different** num_inference_steps for each user request instead of a fixed value, you should use it in conjunction with `refresh_context` API. Before performing inference for each user request, update the cache context based on the actual number of steps. Please refer to [📚run_cache_refresh](https://github.com/vipshop/cache-dit/blob/main/examples/api) as an example.

```python
import cache_dit
from cache_dit import DBCacheConfig
from diffusers import DiffusionPipeline

# Init cache context with num_inference_steps=None (default)
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
pipe = cache_dit.enable_cache(pipe.transformer, cache_config=DBCacheConfig(num_inference_steps=None))

# Assume num_inference_steps is 28, and we want to refresh the context
cache_dit.refresh_context(pipe.transformer, num_inference_steps=28, verbose=True)
output = pipe(...) # Just call the pipe as normal.
stats = cache_dit.summary(pipe.transformer) # Then, get the summary

# Update the cache context with new num_inference_steps=50.
cache_dit.refresh_context(pipe.transformer, num_inference_steps=50, verbose=True)
output = pipe(...) # Just call the pipe as normal.
stats = cache_dit.summary(pipe.transformer) # Then, get the summary

# Update the cache context with new cache_config.
cache_dit.refresh_context(
    pipe.transformer,
    cache_config=DBCacheConfig(
        residual_diff_threshold=0.1,
        max_warmup_steps=10,
        max_cached_steps=20,
        max_continuous_cached_steps=4,
        # The cache settings should all be located in the cache config 
        # if cache config is provided. Otherwise, we will skip it.
        num_inference_steps=50,
    ),
    verbose=True,
)
output = pipe(...) # Just call the pipe as normal.
stats = cache_dit.summary(pipe.transformer) # Then, get the summary
```

## How to use ParamsModifier

Sometimes you may encounter more complex cases, such as **Wan 2.2 MoE**, which has more than one Transformer (namely `transformer` and `transformer_2`), or FLUX.1, which has multiple transformer blocks (namely `single_transformer_blocks` and `transformer_blocks`). cache-dit will assign separate cache contexts for different `blocks` instances but share the same `cache_config` by default. Users who want to achieve fine-grained control over different cache contexts can consider using `ParamsModifier`. Just pass the `ParamsModifier` per `blocks` to the `BlockAdapter` or `enable_cache(...)` API. Then, the shared `cache_config` will be overwritten by the new configurations from the `ParamsModifier`. For example:

```python
from cache_dit import ParamsModifier 

cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # FLUX.1, etc.
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
    # Basic shared cache config 
    cache_config=DBCacheConfig(...),
    params_modifiers=[
        ParamsModifier(
            # Modified config only for transformer_blocks
            # Must call the `reset` method of DBCacheConfig.
            cache_config=DBCacheConfig().reset(
                Fn_compute_blocks=8,
                residual_diff_threshold=0.08,
            ),
        ),
        ParamsModifier(
            # Modified config only for single_transformer_blocks
            # NOTE: FLUX.1, single_transformer_blocks should have `higher` 
            # residual_diff_threshold because of the precision error 
            # accumulation from previous transformer_blocks
            cache_config=DBCacheConfig().reset(
                Fn_compute_blocks=1,
                residual_diff_threshold=0.16,
            ),
        ),
    ],
)
```

## Cache Stats Summary

After finishing each inference of `pipe(...)`, you can call the `cache_dit.summary()` API on pipe to get the details of the **Cache Acceleration Stats** for the current inference. 
```python
stats = cache_dit.summary(pipe)
```

You can set `details` param as `True` to show more details of cache stats. (markdown table format) Sometimes, this may help you analyze what values of the residual diff threshold would be better.

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs Min | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.045     | 0.084     | 0.114     | 0.147     | 0.241     | 0.297     |
```

## Disable Cache  

Users can call `cache_dit.disable_cache` API to disable and delete all acceleration hooks from the optimized pipeline or adapter. 

```python
cache_dit.disable_cache(pipe_or_adapter)
```

## DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

**DBCache**: **Dual Block Caching** for Diffusion Transformers. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache, enabling a balanced trade-off between performance and precision. Moreover, it can be entirely **training**-**free**. Please check [DBCache Design](./DBCACHE_DESIGN.md) docs for more design details.

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-fnbn-v1.png)

```python
import cache_dit
from diffusers import FluxPipeline

pipe_or_adapter = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe_or_adapter)

# Custom options, F8B8, higher precision
from cache_dit import DBCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=DBCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
)
```  

<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>


|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=140px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=140px>|
|**Baseline(L20x1)**|**F1B0 (0.08)**|**F8B8 (0.12)**|**F8B12 (0.12)**|**F8B16 (0.20)**|**F8B20 (0.20)**|
|27.85s|6.04s|5.88s|5.77s|6.01s|6.20s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_NONE_R0.08.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F1B0_R0.08.png width=140px> |<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B8_R0.12.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B12_R0.12.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B16_R0.2.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B20_R0.2.png width=140px>|


<div align="center">
  <p align="center">
    DBCache, <b> L20x4 </b>, Steps: 20, case to show the texture recovery ability of DBCache
  </p>
</div>

These case studies demonstrate that even with relatively high thresholds (such as 0.12, 0.15, 0.2, etc.) under the DBCache **F12B12** or **F8B16** configuration, the detailed texture of the kitten's fur, colored cloth, and the clarity of text can still be preserved. This suggests that users can leverage DBCache to effectively balance performance and precision in their workflows! 

## DBPrune: Dynamic Block Prune

<div id="dbprune"></div>  

![](https://github.com/vipshop/cache-dit/raw/main/docs/assets/dbprune.png)


We have further implemented a new **Dynamic Block Prune** algorithm based on **Residual Caching** for Diffusion Transformers, which is referred to as **DBPrune**. DBPrune caches each block's hidden states and residuals, then dynamically prunes blocks during inference by computing the L1 distance between previous hidden states. When a block is pruned, its output is approximated using the cached residuals. DBPrune is currently in the experimental phase, and we kindly invite you to stay tuned for upcoming updates.

```python
from cache_dit import DBPruneConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=DBPruneConfig(
        max_warmup_steps=8,  # steps do not apply prune
        residual_diff_threshold=0.12,
        enable_dynamic_prune_threshold=True,
    ),
)
```
We have also brought the designs from DBCache to DBPrune to make it a more general and customizable block prune algorithm. You can specify the values of **Fn** and **Bn** for higher precision, or set up the non-prune blocks list **non_prune_block_ids** to avoid aggressive pruning. For example:

```python
cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=DBPruneConfig(
        max_warmup_steps=8,  # steps do not apply prune
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc
        residual_diff_threshold=0.12,
        enable_dynamic_prune_threshold=True,
        non_prune_block_ids=list(range(16,24)),
    ),
)
```
<div align="center">
  <p align="center">
    DBPrune, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|Pruned(24%)|Pruned(35%)|Pruned(38%)|Pruned(45%)|Pruned(60%)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|19.43s|16.82s|15.95s|14.24s|10.66s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.03_P24.0_T19.43s.png width=140px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.04_P34.6_T16.82s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.05_P38.3_T15.95s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.06_P45.2_T14.24s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.2_P59.5_T10.66s.png width=140px>|

## Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `enable_separate_cfg` param to **False (default, None)**. Otherwise, set it to True. For examples:

```python
from cache_dit import DBCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(
        ...,
        # CFG: classifier free guidance or not
        # For model that fused CFG and non-CFG into single forward step,
        # should set enable_separate_cfg as False. For example, set it as True 
        # for Wan 2.1/Qwen-Image and set it as False for FLUX.1, HunyuanVideo, 
        # CogVideoX, Mochi, LTXVideo, Allegro, CogView3Plus, EasyAnimate, SD3, etc.
        enable_separate_cfg=True, # Wan 2.1, Qwen-Image, CogView4, Cosmos, SkyReelsV2, etc.
        # Compute cfg forward first or not, default False, namely, 
        # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
        cfg_compute_first=False,
        # Compute separate diff values for CFG and non-CFG step, 
        # default True. If False, we will use the computed diff from 
        # current non-CFG transformer step for current CFG step.
        cfg_diff_compute_separate=True,
    ),
)
```

## Hybrid TaylorSeer Calibrator

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That F_pred can be a residual cache or a hidden-state cache.

```python
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    # Basic DBCache w/ FnBn configurations
    cache_config=DBCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
    # Then, you can use the TaylorSeer Calibrator to approximate 
    # the values in cached steps, taylorseer_order default is 1.
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)
``` 

Please note that if you have used TaylorSeer as the calibrator for approximate hidden states, the **Bn** param of DBCache can be set to **0**. In essence, DBCache's Bn is also act as a calibrator, so you can choose either Bn > 0 or TaylorSeer. We recommend using the configuration scheme of **TaylorSeer** + **DBCache FnB0**.

<div align="center">
  <p align="center">
    <b>DBCache F1B0 + TaylorSeer</b>, L20x1, Steps: 28, <br>"A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.12)|+TaylorSeer|F1B0 (0.15)|+TaylorSeer|+compile|  
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|12.85s|12.86s|10.27s|10.28s|8.48s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.12_S14_T12.85s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.12_S14_T12.86s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.15_S17_T10.27s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T10.28s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T8.48s.png width=140px>|

## SCM: Steps Computation Masking

<div id="steps-mask"></div>


The `steps_computation_mask` parameter adopts a step-wise computation masking approach inspired by [LeMiCa](https://github.com/UnicomAI/LeMiCa) and [EasyCache](https://github.com/H-EmbodVis/EasyCache). Its key insight is that **early caching induces amplified downstream errors, whereas later caching is less disruptive**, resulting in a **non-uniform** distribution of cached steps. 

|LeMiCa: Non-Uniform Cache Steps|LeMiCa: Cache Errors|EasyCache: Transformation rate Analysis|
|:---:|:---:|:---:|
|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/lemica.png" width=383px>|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/lemica_0.png" width=235px>|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/easy_cache_0.png" width=343px>|

It is a list of length num_inference_steps indicating whether to compute each step or not. 1 means must compute, 0 means use dynamic/static cache. If provided, will override other settings to decide whether to compute each step. Please check the [📚examples/steps_mask](https://github.com/vipshop/cache-dit/blob/main/examples/api/run_steps_mask.py) for more details.


```python
from cache_dit import DBCacheConfig, TaylorSeerCalibratorConfig

# Scheme: Hybrid DBCache + SCM + TaylorSeer
cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=DBCacheConfig(
        # Basic DBCache configs
        Fn_compute_blocks=8,
        Bn_compute_blocks=0,
        # NOTE: warmup steps is not required now!
        residual_diff_threshold=0.12,
        # LeMiCa or EasyCache style Mask for 28 steps, e.g, 
        # SCM=111111010010000010000100001, 1: compute, 0: cache.
        steps_computation_mask=cache_dit.steps_mask(
            # e.g: slow, medium, fast, ultra.
            mask_policy="fast", total_steps=28,
            # Or, you can use bins setting to get custom mask.
            # compute_bins=[6, 1, 1, 1, 1], # 10
            # cache_bins=[1, 2, 5, 5, 5], # 18
        ),
        # The policy for cache steps can be 'dynamic' or 'static'
        steps_computation_policy="dynamic",
    ),
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)

```

As we can observe, in the case of **static cache**, the image of `SCM Slow S*` (please click to enlarge) has shown **obvious blurriness**. However, the **Ultra** version under **dynamic cache** (`SCM Ultra D*`) still maintains excellent clarity. Therefore, we prioritize recommending the use of dynamic cache while using `SCM: steps_computation_mask`.


|Baseline|SCM S S*|SCM S D*|SCM F D*|SCM U D*|+TS|+compile|+FP8 +Sage|  
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.4s|17.1s|11.4s|8.2s|8.2s|7.1s|4.5s|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.NONE.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/static.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.15_SCM1111111101110011100110011000_dynamic_T0O0_S8.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.2_SCM1111110100010000100000100000_dynamic_T0O0_S15.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.3_SCM111101000010000010000001000000_dynamic_T0O0_S19.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=95px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.C1_Q1_float8_DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=95px>|

<p align="center">
  Scheme: <b>DBCache + SCM(steps_computation_mask) + TaylorSeer</b>, L20x1, S*: static cache, <b>D*: dynamic cache</b>, <br><b>S</b>: Slow, <b>F</b>: Fast, <b>U</b>: Ultra Fast, <b>TS</b>: TaylorSeer, FP8: FP8 DQ, Sage: SageAttention, <b>FLUX.1-Dev</b>, <br>Steps: 28, HxW=1024x1024, Prompt: "A cat holding a sign that says hello world"
</p>

|DBCache + SCM Slow S*|DBCache + SCM Ultra D* + TaylorSeer + compile| 
|:---:|:---:|
|15.4s|7.1s|
|<img src="https://github.com/vipshop/cache-dit/raw/main//assets/steps_mask/static.png" width=460px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=460px>|

<p align="center">
<b>Dynamic Caching is all you need!</b> The <b>Ultra</b> fast version under dynamic cache (<b>SCM Ultra D*</b>) <br>maintains <b>better clarity</b> than the slower static cache one (<b>SCM Slow S*</b>).
</p>
