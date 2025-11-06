<div align="center">
  <p align="center">
    <h2 align="center">
        <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="90" align="left">
        A Unified and Flexible Inference Engine with 🤗🎉<br>Hybrid Cache Acceleration and Parallelism for DiTs<br>
        <a href="https://pypi.org/project/cache-dit/"><img src=https://img.shields.io/pypi/dm/cache-dit.svg ></a> 
        <img src=https://img.shields.io/github/stars/vipshop/cache-dit.svg?style=dark >
        <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a> 
        <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
        <img src=https://img.shields.io/badge/Models-30+-hotpink.svg >
    </h2>
  </p>
</div>

## 📖Table of Contents

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Supported DiTs](#supported)
- [🔥Benchmarks](#benchmarks)
- [🎉Unified Cache APIs](#unified)
  - [📚Forward Pattern Matching](#forward-pattern-matching)
  - [📚Cache with One-line Code](#%EF%B8%8Fcache-acceleration-with-one-line-code)
  - [🔥Automatic Block Adapter](#automatic-block-adapter)
  - [📚Hybrid Forward Pattern](#automatic-block-adapter)
  - [📚Implement Patch Functor](#implement-patch-functor)
  - [📚Transformer-Only Interface](#transformer-only-interface)
  - [📚How to use ParamsModifier](#how-to-use-paramsmodifier)
  - [🤖Cache Acceleration Stats](#cache-acceleration-stats-summary)
- [⚡️DBCache: Dual Block Cache](#dbcache)
- [⚡️DBPrune: Dynamic Block Prune](#dbprune)
- [⚡️Hybrid Cache CFG](#cfg)
- [🔥Hybrid TaylorSeer Calibrator](#taylorseer)
- [⚡️Hybrid Context Parallelism](#context-parallelism)
- [⚡️Hybrid Tensor Parallelism](#tensor-parallelism)
- [🤖Low-bits Quantization](#quantization)
- [🛠Metrics Command Line](#metrics)
- [⚙️Torch Compile](#compile)
- [📚API Documents](#api-docs)

## ⚙️Installation  

<div id="installation"></div>

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit # or, pip3 install -U "cache-dit[all]" for all features
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

## 🔥Supported DiTs  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Examples](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline) for more details. Here are just some of the tested models listed.

```python
>>> import cache_dit
>>> cache_dit.supported_pipelines()
(32, ['Flux*', 'Mochi*', 'CogVideoX*', 'Wan*', 'HunyuanVideo*', 'QwenImage*', 'LTX*', 'Allegro*',
'CogView3Plus*', 'CogView4*', 'Cosmos*', 'EasyAnimate*', 'SkyReelsV2*', 'StableDiffusion3*',
'ConsisID*', 'DiT*', 'Amused*', 'Bria*', 'Lumina*', 'OmniGen*', 'PixArt*', 'Sana*', 'StableAudio*',
'VisualCloze*', 'AuraFlow*', 'Chroma*', 'ShapE*', 'HiDream*', 'HunyuanDiT*', 'HunyuanDiTPAG*',
'Kandinsky5*', 'PRX*'])
>>> cache_dit.supported_matrix()
```

> [!Tip] 
> One **Model Series** may contain **many** pipelines. cache-dit applies optimizations at the **Transformer** level; thus, any pipelines that include the supported transformer are already supported by cache-dit. ✅: known work and official supported now; ✖️: unofficial supported now, but maybe support in the future.

<div align="left">

| 📚Model | Cache  | CP | TP | 📚Model | Cache  | CP | TP |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **🎉[FLUX.1](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[FLUX.1 nunchaku](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[Mochi](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[Lumina](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[CogVideoX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[OmniGen](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Wan 2.1](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[PixArt](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Wan 2.2](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[CogVideoX 1.5](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[HunyuanVideo](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[Sana](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[StableAudio](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Qwen...Lightning](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[Bria](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[LTX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[VisualCloze](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ |
| **🎉[Allegro](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[AuraFlow](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[CogView4](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[ShapE](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[CogView3Plus](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[Chroma](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ️✅ |
| **🎉[Cosmos](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HiDream](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[EasyAnimate](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HunyuanDiT](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[SkyReelsV2](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HunyuanDiTPAG](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[StableDiffusion3](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[Kandinsky5](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[ConsisID](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[PRX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[DiT](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HunyuanImage](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ |
| **🎉[Amused](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[LongCatVideo](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |

</div>

## 🔥Benchmarks

<div id="benchmarks"></div>

cache-dit will support more mainstream Cache acceleration algorithms in the future. More benchmarks will be released, please stay tuned for update. Here, only the results of some precision and performance benchmarks are presented. The test dataset is **DrawBench**. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/raw/main/bench/).

### 📚Text2Image DrawBench: FLUX.1-dev

Comparisons between different FnBn compute block configurations show that **more compute blocks result in higher precision**. For example, the F8B0_W8MC0 configuration achieves the best Clip Score (33.007) and ImageReward (1.0333). **Device**: NVIDIA L20. **F**: Fn_compute_blocks, **B**: Bn_compute_blocks, 50 steps.


| Config | Clip Score(↑) | ImageReward(↑) | PSNR(↑) | TFLOPs(↓) | SpeedUp(↑) |
| --- | --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 32.9217 | 1.0412 | INF | 3726.87 | 1.00x |
| F8B0_W4MC0_R0.08 | 32.9871 | 1.0370 | 33.8317 | 2064.81 | 1.80x |
| F8B0_W4MC2_R0.12 | 32.9535 | 1.0185 | 32.7346 | 1935.73 | 1.93x |
| F8B0_W4MC3_R0.12 | 32.9234 | 1.0085 | 32.5385 | 1816.58 | 2.05x |
| F4B0_W4MC3_R0.12 | 32.8981 | 1.0130 | 31.8031 | 1507.83 | 2.47x |
| F4B0_W4MC4_R0.12 | 32.8384 | 1.0065 | 31.5292 | 1400.08 | 2.66x |

### 📚Compare with Other Methods: Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa

![image-reward-bench](https://github.com/vipshop/cache-dit/raw/main/assets/image-reward-bench.png)

![clip-score-bench](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)

The comparison between **cache-dit: DBCache** and algorithms such as Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa is as follows. Now, in the comparison with a speedup ratio less than **4x**, cache-dit achieved the best accuracy. Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/raw/main/bench/). 

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(S)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(S)+TS](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(M)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache(M)+TS](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | 1.0029 | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[DBCache(F)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.9271 | 32.552 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | 6.24× | 0.9502 | 32.706 |
| **[DBCache(F)+TS](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | **0.9526** | 32.568 |
| **[DBCache(U)+TS](https://github.com/vipshop/cache-dit)** | 505.47 | **7.37x** | 0.8645 | **32.719** |

NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211).

### 📚Text2Image Distillation DrawBench: Qwen-Image-Lightning

Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For example,  **Qwen-Image-Lightning w/ 4 steps**, with the F16B16 configuration, the PSNR is 34.8163, the Clip Score is 35.6109, and the ImageReward is 1.2614. It maintained a relatively high precision.

| Config                     |  PSNR(↑)      | Clip Score(↑) | ImageReward(↑) | TFLOPs(↓)   | SpeedUp(↑) |
|----------------------------|-----------|------------|--------------|----------|------------|
| [**Lightning**]: 4 steps   | INF       | 35.5797    | 1.2630       | 274.33   | 1.00x       |
| F24B24_W2MC1_R0.8          | 36.3242   | 35.6224    | 1.2630       | 264.74   | 1.04x       |
| F16B16_W2MC1_R0.8          | 34.8163   | 35.6109    | 1.2614       | 244.25   | 1.12x       |
| F12B12_W2MC1_R0.8          | 33.8953   | 35.6535    | 1.2549       | 234.63   | 1.17x       |
| F8B8_W2MC1_R0.8            | 33.1374   | 35.7284    | 1.2517       | 224.29   | 1.22x       |
| F1B0_W2MC1_R0.8            | 31.8317   | 35.6651    | 1.2397       | 206.90   | 1.33x       |

## 🎉Unified Cache APIs

<div id="unified"></div>  

### 📚Forward Pattern Matching 

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

### ♥️Cache Acceleration with One-line Code

In most cases, you only need to call **one-line** of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py) as an example. 

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

### 🔥Automatic Block Adapter

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

### 📚Hybrid Forward Pattern

Sometimes, a Transformer class will contain more than one transformer `blocks`. For example, **FLUX.1** (HiDream, Chroma, etc) contains transformer_blocks and single_transformer_blocks (with different forward patterns). The **BlockAdapter** can also help you solve this problem. Please refer to [📚FLUX.1](https://github.com/vipshop/cache-dit/blob/main/examples/adapter/run_flux_adapter.py) as an example.

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

Even sometimes you have more complex cases, such as **Wan 2.2 MoE**, which has more than one Transformer (namely `transformer` and `transformer_2`) in its structure. Fortunately, **cache-dit** can also handle this situation very well. Please refer to [📚Wan 2.2 MoE](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_wan_2.2.py) as an example.

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

### 📚Implement Patch Functor

For any PATTERN not in {0...5}, we introduced the simple abstract concept of **Patch Functor**. Users can implement a subclass of Patch Functor to convert an unknown Pattern into a known PATTERN, and for some models, users may also need to fuse the operations within the blocks for loop into block forward. 

![](https://github.com/vipshop/cache-dit/raw/main/assets/patch-functor.png)

Some Patch functors have already been provided in cache-dit: [📚HiDreamPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/patch_functors/functor_hidream.py), [📚ChromaPatchFunctor](https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/patch_functors/functor_chroma.py), etc. After implementing Patch Functor, users need to set the `patch_functor` property of **BlockAdapter**.

```python
@BlockAdapterRegistry.register("HiDream")
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

### 📚Transformer-Only Interface

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

### 📚How to use ParamsModifier

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

### 🤖Cache Acceleration Stats Summary

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

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

**DBCache**: **Dual Block Caching** for Diffusion Transformers. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache, enabling a balanced trade-off between performance and precision. Moreover, it can be entirely **training**-**free**. Please check [DBCache.md](https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md) docs for more design details.

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

## ⚡️DBPrune: Dynamic Block Prune

<div id="dbprune"></div>  

![](https://github.com/user-attachments/assets/932b6360-9533-4352-b176-4c4d84bd4695)


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

## ⚡️Hybrid Cache CFG

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

## 🔥Hybrid TaylorSeer Calibrator

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

$$
\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)=\mathcal{F}\left(x_t^l\right)+\sum_{i=1}^m \frac{\Delta^i \mathcal{F}\left(x_t^l\right)}{i!\cdot N^i}(-k)^i
$$

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That is $\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)$ can be a residual cache or a hidden-state cache.

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

> [!Important]
> Please note that if you have used TaylorSeer as the calibrator for approximate hidden states, the **Bn** param of DBCache can be set to **0**. In essence, DBCache's Bn is also act as a calibrator, so you can choose either Bn > 0 or TaylorSeer. We recommend using the configuration scheme of **TaylorSeer** + **DBCache FnB0**.

<div align="center">
  <p align="center">
    <b>DBCache F1B0 + TaylorSeer</b>, L20x1, Steps: 28, <br>"A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.12)|+TaylorSeer|F1B0 (0.15)|+TaylorSeer|+compile| 
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|12.85s|12.86s|10.27s|10.28s|8.48s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.12_S14_T12.85s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.12_S14_T12.86s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.15_S17_T10.27s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T10.28s.png width=140px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T8.48s.png width=140px>|


## ⚡️Hybrid Context Parallelism

<div id="context-parallelism"></div>

cache-dit is compatible with context parallelism. Currently, we support the use of `Hybrid Cache` + `Context Parallelism` scheme (via NATIVE_DIFFUSER parallelism backend) in cache-dit. Users can use Context Parallelism to further accelerate the speed of inference! For more details, please refer to [📚examples/parallelism](https://github.com/vipshop/cache-dit/tree/main/examples/parallelism). Currently, cache-dit supported context parallelism for [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [LTXVideo](https://huggingface.co/Lightricks/LTX-Video), [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1), [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo), [CogVideoX 1.0](https://github.com/zai-org/CogVideo), [CogVideoX 1.5](https://github.com/zai-org/CogVideo), [CogView 3/4](https://github.com/zai-org/CogView4) and [VisualCloze](https://github.com/lzyhha/VisualCloze), etc. cache-dit will support more models in the future.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set ulysses_size > 1 to enable ulysses style context parallelism.
    parallelism_config=ParallelismConfig(ulysses_size=2),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

## ⚡️Hybrid Tensor Parallelism

<div id="tensor-parallelism"></div>

cache-dit is also compatible with tensor parallelism. Currently, we support the use of `Hybrid Cache` + `Tensor Parallelism` scheme (via NATIVE_PYTORCH parallelism backend) in cache-dit. Users can use Tensor Parallelism to further accelerate the speed of inference and **reduce the VRAM usage per GPU**! For more details, please refer to [📚examples/parallelism](https://github.com/vipshop/cache-dit/tree/main/examples/parallelism). Now, cache-dit supported tensor parallelism for [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1), [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) and [VisualCloze](https://github.com/lzyhha/VisualCloze), etc. cache-dit will support more models in the future.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set tp_size > 1 to enable tensor parallelism.
    parallelism_config=ParallelismConfig(tp_size=2),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

> [!Important] 
> Please note that in the short term, we have no plans to support Hybrid Parallelism. Please choose to use either Context Parallelism or Tensor Parallelism based on your actual scenario.

## 🤖Low-bits Quantization

<div id="quantization"></div>

Currently, torchao has been integrated into cache-dit as the backend for **online** model quantization (with more backends to be supported in the future). You can implement model quantization by calling `cache_dit.quantize(...)`. At present, cache-dit supports the `Hybrid Cache + Low-bits Quantization` scheme. For GPUs with low memory capacity, we recommend using `float8_weight_only` or `int8_weight_only`, as these two schemes cause almost no loss in precision. For more details, please refer to [📚examples/quantize](https://github.com/vipshop/cache-dit/tree/main/examples/quantize).

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


## 🛠Metrics Command Line

<div id="metrics"></div>    

You can utilize the APIs provided by cache-dit to quickly evaluate the accuracy losses caused by different cache configurations. For example:

```python
# pip3 install "cache-dit[metrics]"
from cache_dit.metrics import compute_psnr
from cache_dit.metrics import compute_ssim
from cache_dit.metrics import compute_fid
from cache_dit.metrics import compute_lpips
from cache_dit.metrics import compute_clip_score
from cache_dit.metrics import compute_image_reward

psnr,   n = compute_psnr("true.png", "test.png") # Num: n
psnr,   n = compute_psnr("true_dir", "test_dir")
ssim,   n = compute_ssim("true_dir", "test_dir")
fid,    n = compute_fid("true_dir", "test_dir")
lpips,  n = compute_lpips("true_dir", "test_dir")
clip,   n = compute_clip_score("DrawBench200.txt", "test_dir")
reward, n = compute_image_reward("DrawBench200.txt", "test_dir")
```

Or, you can use `cache-dit-metrics-cli` tool. For examples: 

```bash
cache-dit-metrics-cli -h  # show usage
# all: PSNR, FID, SSIM, MSE, ..., etc.
cache-dit-metrics-cli all  -i1 true.png -i2 test.png  # image
cache-dit-metrics-cli all  -i1 true_dir -i2 test_dir  # image dir
```

## ⚙️Torch Compile

<div id="compile"></div>  

By the way, **cache-dit** is designed to work compatibly with **torch.compile.** You can easily use cache-dit with torch.compile to further achieve a better performance. For example:

```python
cache_dit.enable_cache(pipe)

# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```
However, users intending to use **cache-dit** for DiT with **dynamic input shapes** should consider increasing the **recompile** **limit** of `torch._dynamo`. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode. 
```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

Please check [perf.py](https://github.com/vipshop/cache-dit/blob/main/bench/perf.py) for more details.

---

## 📚API Documentation

<div id="api-docs"></div>  

Unified Cache API for almost Any Diffusion Transformers (with Transformer Blocks that match the specific Input and Output patterns). For a good balance between performance and precision, DBCache is configured by default with F8B0, 8 warmup steps, and unlimited cached steps. All the configurable params are listed beflows.

### 👏API: enable_cache

```python
def enable_cache(...) -> Union[DiffusionPipeline, BlockAdapter, Transformer]
```

### 🌟Function Description

The `enable_cache` function serves as a unified caching interface designed to optimize the performance of diffusion transformer models by implementing an intelligent caching mechanism known as `DBCache`. This API is engineered to be compatible with nearly `all` diffusion transformer architectures that feature transformer blocks adhering to standard input-output patterns, eliminating the need for architecture-specific modifications.  

By strategically caching intermediate outputs of transformer blocks during the diffusion process, `DBCache` significantly reduces redundant computations without compromising generation quality. The caching mechanism works by tracking residual differences between consecutive steps, allowing the model to reuse previously computed features when these differences fall below a configurable threshold. This approach maintains a balance between computational efficiency and output precision.  

The default configuration (`F8B0, 8 warmup steps, unlimited cached steps`) is carefully tuned to provide an optimal tradeoff for most common use cases. The "F8B0" configuration indicates that the first 8 transformer blocks are used to compute stable feature differences, while no final blocks are employed for additional fusion. The warmup phase ensures the model establishes sufficient feature representation before caching begins, preventing potential degradation of output quality.  

This function seamlessly integrates with both standard diffusion pipelines and custom block adapters, making it versatile for various deployment scenarios—from research prototyping to production environments where inference speed is critical. By abstracting the complexity of caching logic behind a simple interface, it enables developers to enhance model performance with minimal code changes.

### 👇Quick Start

```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
>>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
>>> output = pipe(...) # Just call the pipe as normal.
>>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
>>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
```

### 👇Parameter Description

- **pipe_or_adapter**(`DiffusionPipeline`, `BlockAdapter` or `Transformer`, *required*):  
  The standard Diffusion Pipeline or custom BlockAdapter (from cache-dit or user-defined).
  For example: `cache_dit.enable_cache(FluxPipeline(...))`.

- **cache_config**(`DBCacheConfig`, *required*, defaults to DBCacheConfig()):  
  Basic DBCache config for cache context, defaults to DBCacheConfig(). The configurable parameters are listed below:
  - `Fn_compute_blocks`: (`int`, *required*, defaults to 8):  
    Specifies that `DBCache` uses the**first n**Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 difference and delivering more accurate information to subsequent blocks.
    Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md for more details of DBCache.
  - `Bn_compute_blocks`: (`int`, *required*, defaults to 0):  
    Further fuses approximate information in the**last n**Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.
  - `residual_diff_threshold`: (`float`, *required*, defaults to 0.08):  
    The value of residual difference threshold, a higher value leads to faster performance at the cost of lower precision.
  - `max_warmup_steps`: (`int`, *required*, defaults to 8):  
    DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
  - `warmup_interval`: (`int`, *required*, defaults to 1):    
    Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
    in warmup steps will be computed, others will use dynamic cache.
  - `max_cached_steps`: (`int`, *required*, defaults to -1):  
    DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
  - `max_continuous_cached_steps`: (`int`, *required*, defaults to -1):  
    DBCache disables the caching strategy when the previous continuous cached steps exceed this value to prevent precision degradation.
  - `enable_separate_cfg`: (`bool`, *required*, defaults to None):  
    Whether to use separate cfg or not, such as in Wan 2.1, Qwen-Image. For models that fuse CFG and non-CFG into a single forward step, set enable_separate_cfg as False. Examples include: CogVideoX, HunyuanVideo, Mochi, etc.
  - `cfg_compute_first`: (`bool`, *required*, defaults to False):    
    Whether to compute cfg forward first, default is False, meaning:  
    0, 2, 4, ... -> non-CFG step; 1, 3, 5, ... -> CFG step.
  - `cfg_diff_compute_separate`: (`bool`, *required*, defaults to True):    
    Whether to compute separate difference values for CFG and non-CFG steps, default is True. If False, we will use the computed difference from the current non-CFG transformer step for the current CFG step.
  - `num_inference_steps` (`int`, *optional*, defaults to None):  
    num_inference_steps for DiffusionPipeline, used to adjust some internal settings
    for better caching performance. For example, we will refresh the cache once the
    executed steps exceed num_inference_steps if num_inference_steps is provided.

- **calibrator_config** (`CalibratorConfig`, *optional*, defaults to None):  
  Config for calibrator. If calibrator_config is not None, it means the user wants to use DBCache with a specific calibrator, such as taylorseer, foca, and so on.

- **params_modifiers** ('ParamsModifier', *optional*, defaults to None):  
  Modify cache context parameters for specific blocks. The configurable parameters are listed below:
  - `cache_config`: (`DBCacheConfig`, *required*, defaults to DBCacheConfig()):  
    The same as the 'cache_config' parameter in the cache_dit.enable_cache() interface.
  - `calibrator_config`: (`CalibratorConfig`, *optional*, defaults to None):  
    The same as the 'calibrator_config' parameter in the cache_dit.enable_cache() interface.
  - `**kwargs`: (`dict`, *optional*, defaults to {}):  
    The same as the 'kwargs' parameter in the cache_dit.enable_cache() interface.

- **parallelism_config** (`ParallelismConfig`, *optional*, defaults to None):  
    Config for Parallelism. If parallelism_config is not None, it means the user wants to enable
    parallelism for cache-dit.
    - `backend`: (`ParallelismBackend`, *required*, defaults to "ParallelismBackend.NATIVE_DIFFUSER"):  
        Parallelism backend, currently only NATIVE_DIFFUSER and NVTIVE_PYTORCH are supported.
        For context parallelism, only NATIVE_DIFFUSER backend is supported, for tensor parallelism,
        only NATIVE_PYTORCH backend is supported.
    - `ulysses_size`: (`int`, *optional*, defaults to None):  
        The size of Ulysses cluster. If ulysses_size is not None, enable Ulysses style parallelism.
        This setting is only valid when backend is NATIVE_DIFFUSER.
    - `ring_size`: (`int`, *optional*, defaults to None):  
        The size of ring for ring parallelism. If ring_size is not None, enable ring attention.
        This setting is only valid when backend is NATIVE_DIFFUSER.
    - `tp_size`: (`int`, *optional*, defaults to None):  
        The size of tensor parallelism. If tp_size is not None, enable tensor parallelism.
        This setting is only valid when backend is NATIVE_PYTORCH.
    - `parallel_kwargs`: (`dict`, *optional*, defaults to {}):  
        Additional kwargs for parallelism backends. For example, for NATIVE_DIFFUSER backend,
        it can include `cp_plan` and `attention_backend` arguments for `Context Parallelism`.

- **kwargs** (`dict`, *optional*, defaults to {}):   
  Other cache context keyword arguments. Please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_context.py for more details.
