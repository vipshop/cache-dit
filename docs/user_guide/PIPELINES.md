# Built-in Pipelines

<div id="builtin-pipelines"></div>

Some checkpoints cannot be run with `DiffusionPipeline.from_pretrained`: their `model_index.json` points at a `pipeline_<model>.py` that lives in a third-party package or a remote-code module. Cache-DiT vendors those pipelines (and their transformers, schedulers and attention processors) under `cache_dit.models.pipelines`, so the only difference from a 🤗 Diffusers model is the **import line** — the class has to come from cache-dit:

```python
from cache_dit.models.pipelines.moss_image import MOSSImagePipeline
```

Cache-DiT also ships fused variants of a few models that diffusers *does* have (FLUX.2-Klein, FLUX.3-Klein, Qwen-Image 2.1). Importing cache-dit's `_`-prefixed class only swaps in the fused operators; the loading path is the stock diffusers one.

Nothing is re-exported from `cache_dit` or from `cache_dit.models.pipelines/__init__.py`, so the import always names the submodule. Everything is lazy: importing `cache_dit` never pulls model code.

## Quick Start

Import the pipeline, load it, then enable whatever optimizations you need — exactly as in the [main Quick Start](../README.md) for a diffusers pipeline:

```python
import torch
import cache_dit
from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig
from cache_dit.models.pipelines.moss_image import MOSSImagePipeline

pipe = MOSSImagePipeline.from_pretrained("/PATH/TO/MOSS-Image-SFT", torch_dtype=torch.bfloat16)
pipe.to("cuda")

# Cache only, with default DBCache settings.
cache_dit.enable_cache(pipe)

# Or hybrid cache + parallelism.
cache_dit.enable_cache(
  pipe,
  cache_config=DBCacheConfig(),
  parallelism_config=ParallelismConfig(ulysses_size=2),
)

# Or hybrid cache + parallelism + quantization.
cache_dit.enable_cache(
  pipe,
  cache_config=DBCacheConfig(),
  parallelism_config=ParallelismConfig(ulysses_size=2),
  quantize_config=QuantizeConfig(quant_type="float8_per_row"),
)

output = pipe(  # Then call the pipeline as normal.
  prompt="a red fox sitting in fresh snow, golden hour",
  height=1024,
  width=1024,
  num_inference_steps=50,
  guidance_scale=3.0,
  output_type="pil",
)
output.images[0].save("moss.png")
```

Every argument is optional: `cache_dit.enable_cache(pipe)` alone turns on DBCache with defaults. `layerwise_offload`, `set_compile_configs` and `--parallel` (via `torchrun`) are separate entry points and apply to built-in pipelines unchanged.

## Available Pipelines

```python
# Image
from cache_dit.models.pipelines.moss_image import MOSSImagePipeline
from cache_dit.models.pipelines.boogu_image import BooguImagePipeline
from cache_dit.models.pipelines.ming_image import MingImagePipeline
from cache_dit.models.pipelines.hidream_o1_image import HiDreamO1ImagePipeline

# Video and audio
from cache_dit.models.pipelines.sana_video2 import SanaVideo2Pipeline
from cache_dit.models.pipelines.lingbot_video import LingBotVideoPipeline
from cache_dit.models.pipelines.hy_nomas import NomasPipeline
from cache_dit.models.pipelines.longlive2 import LongLive2Pipeline
from cache_dit.models.pipelines.causvid import CausVidPipeline
from cache_dit.models.pipelines.wan_s2v import WanSpeechToVideoPipeline
from cache_dit.models.pipelines.longcat_video_avatar import LongCatVideoAvatarPipeline
from cache_dit.models.pipelines.mova import MOVAPipeline
from cache_dit.models.pipelines.magi2.pipeline import Magi2PreviewPipeline
from cache_dit.models.pipelines.fast_h3 import FastH3ModularPipeline
from cache_dit.models.pipelines.pika_h3_flex import PikaH3FlexModularPipeline

# Fused variants of diffusers pipelines
from cache_dit.models.pipelines.flux2_klein import _Flux2KleinPipeline
from cache_dit.models.pipelines.qwen_image21 import _QwenImage21Pipeline
# FLUX.3-Klein has no pipeline subclass: upgrade the diffusers pipeline in place.
from cache_dit.models.pipelines.flux3_klein.transformer_flux3_klein import upgrade_transformer
```

Each of them is a normal pipeline object: `from_pretrained(...)`, `enable_cache(...)`, then call it with the model's own arguments. Cache, parallelism, quantization, offload and compile are supported for all of them — see the [supported matrix](../supported_matrix/NVIDIA_GPU.md) for per-model coverage, and `docs/EXAMPLES.md` for the model-specific cache presets.

`python3 -m cache_dit.generate list` prints the CLI example name and default checkpoint of every supported model, e.g.

```bash
python3 -m cache_dit.generate moss_image_sft --cache --rdt 0.20
python3 -m cache_dit.generate sana_video2 --cache
python3 -m cache_dit.generate qwen_image_21_turbo_fused --cache
torchrun --nproc_per_node=2 -m cache_dit.generate lingbot_video --parallel ulysses --cache
```

## Notes

- **Checkpoint path.** Most vendored loaders resolve components as `<root>/<component>`, so pass a **local checkpoint directory**: MOSS-Image, Boogu-Image, Ming-Image, HiDream-O1-Image, MOVA, HY-Nomas and MAGI-2 Preview fail on a Hub repo id (`... is not the path to a directory containing a config.json file`). SANA-Video 2.0, LingBot-Video, CausVid-Wan, LongLive 2.0, Wan2.2-S2V, LongCat-Video Avatar, Fast-H3, Pika H3 Flex and the two fused variants also accept a Hub id. `SANA_VIDEO2_VAE_DIR` / `SANA_VIDEO2_TEXT_ENCODER_DIR` (and `LONGCAT_VIDEO_DIR` for the avatar pipeline) point at the components a checkpoint does not ship.
- **Return values.** Image models return `.images`. Video models return `.frames` (SANA-Video 2.0 adds `.fps`; MOVA and MAGI-2 Preview add `.audio` and `.sampling_rate`; MAGI-2 returns a plain `dict`). LongCat-Video Avatar returns a bare `np.ndarray`. The two modular pipelines (Fast-H3, Pika H3 Flex) return the whole `PipelineState` unless you pass `output=` — `output="videos"` or `output=["videos", "audio", "sampling_rate"]`.
- **CFG bookkeeping.** Caching assumes a fixed number of transformer forwards per step. Pipelines that run one forward per CFG branch need `guidance_scale > 1` (MOSS-Image, Boogu-Image, HiDream-O1-Image, LingBot-Video, MOVA, Wan2.2-S2V); distilled ones need exactly `1.0` (LongLive 2.0, HY-Nomas, CausVid-Wan). The adapter is inferred from the pipeline class, so do not set `enable_separate_cfg` by hand — a mismatch raises instead of silently halving the step count.
- **Fused variants.** `_Flux2KleinPipeline` / `_QwenImage21Pipeline` upgrade the transformer inside `from_pretrained`. FLUX.3-Klein is a diffusers modular pipeline with no cache-dit subclass: load `Flux3DistilledModularPipeline` and call `upgrade_transformer(pipe.transformer)` **before** `enable_cache` / compile / offload. `CACHE_DIT_FUSED_OPS` (see [Environment Variables](./ENV.md)) decides whether the fused operators are installed.
- **Extra assembly.** A few models need one preparation step before `from_pretrained`: CausVid-Wan takes a `transformer=` built by `build_causvid_transformer(...)`, LongLive 2.0 wants its `model_bf16.pt` converted with `python -m cache_dit.models.pipelines.longlive2.convert`, Pika H3 Flex needs `load_components(workflow=...)` plus the released adapter fused, and MAGI-2 Preview only supports text conditioning (`image=` raises `NotImplementedError`). The CLI example of each model is the working recipe.

`enable_cache` matches a built-in pipeline by class-name prefix, and all of the classes above are pre-registered, so no custom `BlockAdapter` is needed. See [Unified Cache APIs](./CACHE_API.md), [Context Parallelism](./CONTEXT_PARALLEL.md), [Tensor Parallelism](./TENSOR_PARALLEL.md) and [Low-Bits Quantization](./QUANTIZATION.md) for the optimization knobs themselves.
