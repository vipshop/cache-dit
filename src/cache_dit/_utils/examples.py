import os
import math
import json
import torch
import argparse
import PIL.Image
import numpy as np
from typing import Any, Dict, Tuple, List, Optional
from diffusers.utils import load_image
from diffusers import FlowMatchEulerDiscreteScheduler
from ..caching import DBCacheConfig, ParamsModifier, steps_mask
from ..logger import init_logger

from .registers import (
  Example,
  ExampleType,
  ExampleInputData,
  ExampleInitConfig,
  ExampleRegister,
)
from .utils import DEFAULT_RESIDUAL_DIFF_THRESHOLD

logger = init_logger(__name__)

__all__ = [
  "flux_example",
  "flux_fill_example",
  "flux2_example",
  "flux2_klein_example",
  "flux2_klein_edit_example",
  "qwen_image_example",
  "qwen_image_controlnet_example",
  "qwen_image_edit_example",
  "qwen_image_layered_example",
  "qwen_image_21_example",
  "qwen_image_21_edit_example",
  "qwen_image_21_turbo_example",
  "qwen_image_21_turbo_edit_example",
  "qwen_image_21_modular_example",
  "qwen_image_21_turbo_modular_example",
  "qwen_image_21_modular_edit_example",
  "qwen_image_21_turbo_modular_edit_example",
  "skyreels_v2_example",
  "ltx2_t2v_example",
  "ltx2_i2v_example",
  "ltx25_t2v_example",
  "ltx25_i2v_example",
  "ltx25_dfr_example",
  "wan_example",
  "wan_i2v_example",
  "wan_vace_example",
  "ovis_image_example",
  "zimage_example",
  "zimage_controlnet_example",
  "longcat_image_example",
  "longcat_image_edit_example",
  "glm_image_example",
  "glm_image_edit_example",
  "firered_image_edit_example",
  "helios_t2v_example",
  "helios_t2v_distill_example",
  "cosmos3_omni_example",
  "flux2_klein_kv_edit_example",
  "flux3_example",
  "flux3_edit_example",
  "boogu_image_edit_example",
  "ernie_image_example",
  "ernie_image_turbo_example",
  "krea2_example",
  "joy_image_edit_example",
  "bria_fibo_example",
  "anyflow_t2v_example",
  "anyflow_far_example",
  "minimax_h3_example",
  "minimax_h3_fl2va_example",
  "minimax_h3_ref2va_example",
  "fasth3_example",
  "fasth3_v2_example",
  "prx_pixel_example",
  "anima_example",
  "bria_fibo2_example",
  "bria_fibo2_turbo_example",
  "echo_t2va_example",
  "motif_video_t2v_example",
  "motif_video_i2v_example",
  "ideogram4_example",
  "moss_image_sft_example",
  "moss_image_base_example",
  "ming_image_design_example",
  "ming_image_design_layer_example",
  "nucleus_image_example",
  "kandinsky6_ti2va_example",
  "longcat_audio_dit_example",
  "stable_audio_3_example",
  "hidream_o1_image_example",
  "sana_wm_example",
  "wan_animate_2_example",
  "wan_animate_2_distilled_example",
  "wan_s2v_example",
  "lingbot_video_example",
]

# Please note that the following environment variables is only for debugging and
# development purpose. In practice, users should directly provide the model names
# or paths. The default values are the official model names on HuggingFace Hub.
_env_path_mapping = {
  "FLUX_DIR": "black-forest-labs/FLUX.1-dev",
  "FLUX_FILL_DIR": "black-forest-labs/FLUX.1-Fill-dev",
  "NUNCHAKU_FLUX_DIR": "nunchaku-tech/nunchaku-flux.1-dev",
  "FLUX_2_DIR": "black-forest-labs/FLUX.2-dev",
  "FLUX_2_KLEIN_4B_DIR": "black-forest-labs/FLUX.2-klein-4B",
  "FLUX_2_KLEIN_BASE_4B_DIR": "black-forest-labs/FLUX.2-klein-base-4B",
  "FLUX_2_KLEIN_9B_DIR": "black-forest-labs/FLUX.2-klein-9B",
  "FLUX_2_KLEIN_BASE_9B_DIR": "black-forest-labs/FLUX.2-klein-base-9B",
  "FLUX_2_KLEIN_9B_KV_DIR": "black-forest-labs/FLUX.2-klein-9b-kv",
  "FLUX_3_DIR": "black-forest-labs/flux-3-image-klein",
  "OVIS_IMAGE_DIR": "AIDC-AI/Ovis-Image-7B",
  "LTX2_DIR": "Lightricks/LTX-2",
  "LTX25_DIR": "Lightricks/LTX-2.5-Diffusers",
  "QWEN_IMAGE_DIR": "Qwen/Qwen-Image",
  "QWEN_IMAGE_2512_DIR": "Qwen/Qwen-Image-2512",
  "QWEN_IMAGE_21_DIR": "Qwen/Qwen-Image-2.1",
  "QWEN_IMAGE_21_TURBO_DIR": "Qwen/Qwen-Image-2.1-Turbo",
  "QWEN_IMAGE_LIGHT_DIR": "lightx2v/Qwen-Image-Lightning",
  "QWEN_IMAGE_EDIT_2509_DIR": "Qwen/Qwen-Image-Edit-2509",
  "QWEN_IMAGE_EDIT_2511_DIR": "Qwen/Qwen-Image-Edit-2511",
  "QWEN_IMAGE_EDIT_2511_LIGHT_DIR": "lightx2v/Qwen-Image-Edit-2511-Lightning",
  "QWEN_IMAGE_CONTROLNET_DIR": "InstantX/Qwen-Image-ControlNet-Inpainting",
  "QWEN_IMAGE_LAYERED_DIR": "Qwen/Qwen-Image-Layered",
  "SKYREELS_V2_DIR": "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
  "WAN_DIR": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
  "WAN_2_2_DIR": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
  "WAN_I2V_DIR": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
  "WAN_2_2_I2V_DIR": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
  "WAN_2_2_S2V_DIR": "Wan-AI/Wan2.2-S2V-14B-Diffusers",
  "WAN_VACE_DIR": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
  "WAN_2_2_VACE_DIR": "linoyts/Wan2.2-VACE-Fun-14B-diffusers",
  "ZIMAGE_DIR": "Tongyi-MAI/Z-Image",
  "ZIMAGE_TURBO_DIR": "Tongyi-MAI/Z-Image-Turbo",
  "NUNCHAKU_ZIMAGE_TURBO_DIR": "nunchaku-tech/nunchaku-z-image-turbo",
  "Z_IMAGE_TURBO_CONTROLNET_2_1_DIR": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
  "Z_IMAGE_TURBO_CONTROLNET_2_0_DIR": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0",
  "LONGCAT_IMAGE_DIR": "meituan-longcat/LongCat-Image",
  "LONGCAT_IMAGE_EDIT_DIR": "meituan-longcat/LongCat-Image-Edit",
  "LONGCAT_AUDIO_DIT_1B_DIR": "ruixiangma/LongCat-AudioDiT-1B-Diffusers",
  "LONGCAT_AUDIO_DIT_35B_DIR": "saddam213/LongCat-AudioDiT-3.5B-diffusers",
  "STABLE_AUDIO_3_DIR": "stabilityai/stable-audio-3-medium",
  "GLM_IMAGE_DIR": "zai-org/GLM-Image",
  "FIRERED_IMAGE_EDIT_1_DIR": "FireRedTeam/FireRed-Image-Edit-1.0",
  "FIRERED_IMAGE_EDIT_1_1_DIR": "FireRedTeam/FireRed-Image-Edit-1.1",
  "HELIOS_BASE_DIR": "BestWishYsh/Helios-Base",
  "HELIOS_DISTILLED_DIR": "BestWishYsh/Helios-Distilled",
  "COSMOS3_OMNI_DIR": "nvidia/Cosmos3-Nano",
  "BOOGU_IMAGE_EDIT_DIR": "BOOGU-IMAGE/Boogu-Image-0.1-Edit",
  "ERNIE_IMAGE_DIR": "baidu/ERNIE-Image",
  "ERNIE_IMAGE_TURBO_DIR": "baidu/ERNIE-Image-Turbo",
  "KREA2_DIR": "krea/Krea-2-Turbo",
  "FIBO_DIR": "briaai/FIBO",
  "JOY_EDIT_IMAGE_DIR": "jdopensource/JoyAI-Image-Edit-Diffusers",
  "ANY_FLOW_DIR": "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers",
  "ANY_FLOW_FAR_DIR": "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
  "MINIMAX_H3_DIR": "MiniMaxAI/MiniMax-H3",
  "FASTH3_DIR": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree",
  "FASTH3_V2_DIR": "FastVideo/FastVideo-FastH3-8-Step-V2",
  "PRX_PIXEL_DIR": "Photoroom/prxpixel-t2i",
  "ANIMA_DIR": "circlestone-labs/Anima-Base-v1.0-Diffusers",
  "FIBO_2_DIR": "briaai/Fibo-2.0-dev",
  "FIBO_2_TURBO_DIR": "briaai/Fibo-2.0-edit-turbo",
  "ECHO_DIR": "jdopensource/JoyAI-Echo",
  "GEMMA_3_12B_DIR": "google/gemma-3-12b-it",
  "MOTIF_VIDEO_DIR": "Motif-Technologies/Motif-Video-2B",
  "IDEOGRAM4_DIR": "ideogram-ai/ideogram-4-fp8",
  "MOSS_IMAGE_SFT_DIR": "OpenMOSS-Team/MOSS-Image-SFT",
  "MOSS_IMAGE_BASE_DIR": "OpenMOSS-Team/MOSS-Image-Base",
  "HIDREAM_O1_IMAGE_DIR": "HiDream-ai/HiDream-O1-Image",
  "MING_IMAGE_DESIGN_DIR": "inclusionAI/Ming-Image-0.1-Design",
  "MING_IMAGE_DESIGN_LAYER_DIR": "inclusionAI/Ming-Image-0.1-Design-Layer",
  "NUCLEUS_IMAGE_DIR": "NucleusAI/Nucleus-Image",
  "KANDINSKY_6_TI2VA_DIR": "kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
  "SANA_WM_DIR": "Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
  "SANA_WM_REFINER_DIR": "Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner",
  "WAN_ANIMATE_2_DIR": "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
  "WAN_ANIMATE_2_DISTILLED_DIR": "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
  "LINGBOT_VIDEO_DIR": "robbyant/lingbot-video-dense-1.3b",
}
_path_env_mapping = {v: k for k, v in _env_path_mapping.items()}
FLUX2_SKIP_INPUT_IMAGE2 = os.environ.get("FLUX2_SKIP_INPUT_IMAGE2", "0").lower() == "1"
CACHE_DIT_DIR = os.environ.get("CACHE_DIT_DIR", None)
CACHE_DIT_FORCE_USE_LOCAL_DATA = os.environ.get("CACHE_DIT_FORCE_USE_LOCAL_DATA",
                                                "0").lower() == "1"


def _path(
  default: str,
  args: Optional[argparse.Namespace] = None,
  ENV: Optional[str] = None,
  lora: bool = False,
  controlnet: bool = False,
  transformer: bool = False,
) -> str:
  # Prefer command line argument if provided
  if args is not None:
    model_path_arg = args.model_path
    if lora:
      model_path_arg = args.lora_path
    if controlnet:
      model_path_arg = args.controlnet_path
    if transformer:
      model_path_arg = args.transformer_path
    if model_path_arg is not None:
      return model_path_arg
  # Next, check environment variable
  if ENV is None:
    ENV = _path_env_mapping.get(default, None)
    if ENV is None:
      return default
  return os.environ.get(ENV, default)


def _force_use_local_data() -> bool:
  return CACHE_DIT_FORCE_USE_LOCAL_DATA and CACHE_DIT_DIR is not None and os.path.exists(
    CACHE_DIT_DIR)


@ExampleRegister.register("flux", default="black-forest-labs/FLUX.1-dev")
@ExampleRegister.register("flux_nunchaku", default="nunchaku-tech/nunchaku-flux.1-dev")
def flux_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import FluxPipeline

  if "nunchaku" in args.example.lower():
    from nunchaku.models.transformers.transformer_flux_v2 import (
      NunchakuFluxTransformer2DModelV2, )

    nunchaku_flux_dir = _path(
      "nunchaku-tech/nunchaku-flux.1-dev",
      args=args,
      transformer=True,
    )
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
      f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors", )
  else:
    transformer = None

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("black-forest-labs/FLUX.1-dev"),
      pipeline_class=FluxPipeline,
      transformer=transformer,  # maybe use Nunchaku Flux transformer
      # `text_encoder_2` will be quantized when `--quantize-type`
      # is set to `bnb_4bit`. Only hints for quantization.
      bnb_4bit_components=["text_encoder_2"],
    ),
    input_data=ExampleInputData(
      prompt="A cat holding a sign that says hello world",
      height=1024,
      width=1024,
      num_inference_steps=28,
    ),
  )


@ExampleRegister.register("flux_fill", default="black-forest-labs/FLUX.1-Fill-dev")
def flux_fill_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import FluxFillPipeline

  if _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/cup.png"))
    mask_image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/cup_mask.png"))
  else:
    image = load_image("https://github.com/vipshop/cache-dit/raw/main/examples/data/cup.png")
    mask_image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/cup_mask.png")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=_path("black-forest-labs/FLUX.1-Fill-dev"),
      pipeline_class=FluxFillPipeline,
      # `text_encoder_2` will be quantized when `--quantize-type`
      # is set to `bnb_4bit`. Only hints for quantization.
      bnb_4bit_components=["text_encoder_2"],
    ),
    input_data=ExampleInputData(
      prompt="a white paper cup",
      image=image,
      mask_image=mask_image,
      guidance_scale=30,
      height=1024,
      width=1024,
      num_inference_steps=28,
    ),
  )


def _flux2_params_modifiers(args: argparse.Namespace) -> List[ParamsModifier]:
  return [
    ParamsModifier(
      # Modified config only for transformer_blocks
      # Must call the `reset` method of DBCacheConfig.
      cache_config=DBCacheConfig().reset(residual_diff_threshold=args.residual_diff_threshold, ), ),
    ParamsModifier(
      # Modified config only for single_transformer_blocks
      # NOTE: FLUX.2, single_transformer_blocks should have `higher`
      # residual_diff_threshold because of the precision error
      # accumulation from previous transformer_blocks
      cache_config=DBCacheConfig().reset(residual_diff_threshold=args.residual_diff_threshold *
                                         3, ), ),
  ]


@ExampleRegister.register("flux2", default="black-forest-labs/FLUX.2-dev")
def flux2_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux2Pipeline

  params_modifiers = _flux2_params_modifiers(args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("black-forest-labs/FLUX.2-dev"),
      pipeline_class=Flux2Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      # Extra init args for DBCacheConfig, ParamsModifier, etc.
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
      },
    ),
    input_data=ExampleInputData(
      prompt=(
        "Realistic macro photograph of a hermit crab using a soda can as its shell, "
        "partially emerging from the can, captured with sharp detail and natural colors, "
        "on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean "
        "waves in the background. The can has the text `BFL Diffusers` on it and it has a color "
        "gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."),
      height=1024,
      width=1024,
      num_inference_steps=28,
      guidance_scale=4,
    ),
  )


@ExampleRegister.register("flux2_klein_4b", default="black-forest-labs/FLUX.2-klein-4B")
@ExampleRegister.register("flux2_klein_9b", default="black-forest-labs/FLUX.2-klein-9B")
@ExampleRegister.register("flux2_klein_base_4b", default="black-forest-labs/FLUX.2-klein-base-4B")
@ExampleRegister.register("flux2_klein_base_9b", default="black-forest-labs/FLUX.2-klein-base-9B")
def flux2_klein_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux2KleinPipeline

  # cfg: guidance_scale > 1 and not is_distilled
  if "base" in args.example.lower():
    num_inference_steps = 50
    guidance_scale = 4.0  # typical cfg for base model
    enable_separate_cfg = True
    if "4b" in args.example.lower():
      model_path = _path("black-forest-labs/FLUX.2-klein-base-4B")
    else:
      model_path = _path("black-forest-labs/FLUX.2-klein-base-9B")
  else:
    num_inference_steps = 4
    guidance_scale = 1.0  # no cfg for klein
    enable_separate_cfg = False
    if "4b" in args.example.lower():
      model_path = _path("black-forest-labs/FLUX.2-klein-4B")
    else:
      model_path = _path("black-forest-labs/FLUX.2-klein-9B")

  params_modifiers = _flux2_params_modifiers(args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=model_path,
      pipeline_class=Flux2KleinPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      # Extra init args for DBCacheConfig, ParamsModifier, etc.
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
        "enable_separate_cfg": enable_separate_cfg,
      },
    ),
    input_data=ExampleInputData(
      prompt="A cute cat sitting on the beach, watching the sunset.",
      height=1024,
      width=1024,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
    ),
  )


@ExampleRegister.register("flux2_klein_4b_edit", default="black-forest-labs/FLUX.2-klein-4B")
@ExampleRegister.register("flux2_klein_9b_edit", default="black-forest-labs/FLUX.2-klein-9B")
@ExampleRegister.register("flux2_klein_base_4b_edit",
                          default="black-forest-labs/FLUX.2-klein-base-4B")
@ExampleRegister.register("flux2_klein_base_9b_edit",
                          default="black-forest-labs/FLUX.2-klein-base-9B")
def flux2_klein_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux2KleinPipeline

  # cfg: guidance_scale > 1 and not is_distilled
  if "base" in args.example.lower():
    num_inference_steps = 50
    guidance_scale = 4.0  # typical cfg for base model
    enable_separate_cfg = True
    if "4b" in args.example.lower():
      model_path = _path("black-forest-labs/FLUX.2-klein-base-4B")
    else:
      model_path = _path("black-forest-labs/FLUX.2-klein-base-9B")
  else:
    num_inference_steps = 4
    guidance_scale = 1.0  # no cfg for klein
    enable_separate_cfg = False
    if "4b" in args.example.lower():
      model_path = _path("black-forest-labs/FLUX.2-klein-4B")
    else:
      model_path = _path("black-forest-labs/FLUX.2-klein-9B")

  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  if _force_use_local_data():
    image1 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/edit2509_2.jpg"))
    image2 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/visualcloze/12265_00.jpg"))
  else:
    image1 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg")  # bear
    image2 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/visualcloze/12265_00.jpg"
    )  # cloth
  # resize images to desired size
  image1 = image1.resize((width, height))
  image2 = image2.resize((width, height))

  params_modifiers = _flux2_params_modifiers(args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path,
      pipeline_class=Flux2KleinPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      # Extra init args for DBCacheConfig, ParamsModifier, etc.
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
        "enable_separate_cfg": enable_separate_cfg,
      },
    ),
    input_data=ExampleInputData(
      prompt="A cute bear wearing this clothing, sitting on the beach, watching the sunset.",
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      image=[image1] if FLUX2_SKIP_INPUT_IMAGE2 else [image1, image2],
    ),
  )


def _flux3_drop_prompt_upsampler(pipe) -> None:
  """Drop the (separate) prompt upsampler step so `prompt` is encoded as given.

  The upsampler is a different checkpoint (`black-forest-labs/flux-3-image-klein-pup`, a ~9B VLM),
  not part of the FLUX.3 model itself, so it is unloaded as well instead of taking device memory.
  `pipe.blocks` returns a deepcopy, so pop from the live `_blocks` tree.

  :param pipe: FLUX.3 modular pipeline.
  """
  pipe._blocks.sub_blocks.pop("prompt_upsampler", None)
  pipe.unload_components(["prompt_upsampler", "prompt_upsampler_processor"])


def _flux3_cache_config(
  args: argparse.Namespace,
  max_warmup_steps: int = 4,
) -> Optional[DBCacheConfig]:
  """DBCache config tuned for the 16-step distilled FLUX.3 transformer.

  Only `single_transformer_blocks` (the joint stack) is cached, so `Fn_compute_blocks` is small and
  the residual threshold is higher than the generic default: consecutive steps of a 16-step
  flow-map model move further apart than those of a 50-step model. Image editing reuses the
  reference tokens and moves less per step, so it caches earlier steps and needs a longer warmup to
  keep the high-frequency detail (SSIM) intact.

  :param args: CLI arguments.
  :param max_warmup_steps: Steps always computed before caching is allowed.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  steps = 16 if args.num_inference_steps is None else args.num_inference_steps
  return DBCacheConfig(
    Fn_compute_blocks=4,
    Bn_compute_blocks=0,
    max_warmup_steps=min(max_warmup_steps, max(1, steps // 2)),
    num_inference_steps=steps,
    # no classifier-free guidance for the step-distilled (flow-map) checkpoint
    enable_separate_cfg=False,
    residual_diff_threshold=0.30,
  )


@ExampleRegister.register("flux3", default="black-forest-labs/flux-3-image-klein")
def flux3_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux3DistilledModularPipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("black-forest-labs/flux-3-image-klein"),
      pipeline_class=Flux3DistilledModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      post_init_hook=_flux3_drop_prompt_upsampler,
      extra_optimize_kwargs={
        "cache_config": _flux3_cache_config(args),
      },
    ),
    input_data=ExampleInputData(
      prompt=(
        'A chunky knitted diorama: three white cable-knit icebergs with tiny pale blue houses, '
        'a cream banner embroidered "KNITTED ICEBERG VILLAGE", a giant magenta knitted whale tail '
        "bursting from a yarn sea"),
      height=1024,
      width=1024,
      num_inference_steps=16,
    ),
  )


@ExampleRegister.register("flux3_edit", default="black-forest-labs/flux-3-image-klein")
def flux3_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux3DistilledModularPipeline

  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  if _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/edit2509_2.jpg"))
  else:
    image = load_image("https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg")
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=_path("black-forest-labs/flux-3-image-klein"),
      pipeline_class=Flux3DistilledModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      post_init_hook=_flux3_drop_prompt_upsampler,
      extra_optimize_kwargs={
        "cache_config": _flux3_cache_config(args, max_warmup_steps=8),
      },
    ),
    input_data=ExampleInputData(
      prompt=("The same bear, now sitting in a chunky knitted diorama of a snowy village at night, "
              "tiny pale blue houses behind it and embroidered stars above the peaks"),
      height=height,
      width=width,
      num_inference_steps=16,
      image=[image],
    ),
  )


@ExampleRegister.register("flux2_klein_9b_kv_edit", default="black-forest-labs/FLUX.2-klein-9b-kv")
def flux2_klein_kv_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Flux2KleinKVPipeline

  model_path = _path("black-forest-labs/FLUX.2-klein-9b-kv")
  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  if _force_use_local_data():
    image1 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/edit2509_2.jpg"))
    image2 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/visualcloze/12265_00.jpg"))
  else:
    image1 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg")  # bear
    image2 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/visualcloze/12265_00.jpg"
    )  # cloth
  # resize images to desired size
  image1 = image1.resize((width, height))
  image2 = image2.resize((width, height))

  params_modifiers = _flux2_params_modifiers(args)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path,
      pipeline_class=Flux2KleinKVPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      # Extra init args for DBCacheConfig, ParamsModifier, etc.
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
        "enable_separate_cfg": False,  # no separate cfg for klein kv model
      },
    ),
    input_data=ExampleInputData(
      prompt="A cute bear wearing this clothing, sitting on the beach, watching the sunset.",
      height=height,
      width=width,
      num_inference_steps=4,
      image=[image1] if FLUX2_SKIP_INPUT_IMAGE2 else [image1, image2],
    ),
  )


def _qwen_light_scheduler() -> FlowMatchEulerDiscreteScheduler:
  # From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
  scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),  # We use shift=3 in distillation
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),  # We use shift=3 in distillation
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,  # set shift_terminal to None
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
  }
  return FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)


def _qwen_light_cache_config(args: argparse.Namespace) -> Optional[DBCacheConfig]:
  if not args.cache:
    return None
  steps = 8 if args.num_inference_steps is None else args.num_inference_steps
  return DBCacheConfig(
    Fn_compute_blocks=16,
    Bn_compute_blocks=16,
    max_warmup_steps=4 if steps > 4 else 2,
    max_cached_steps=2 if steps > 4 else 1,
    max_continuous_cached_steps=1,
    enable_separate_cfg=False,  # true_cfg_scale=1.0
    residual_diff_threshold=0.50 if steps > 4 else 0.8,
  )


@ExampleRegister.register("qwen_image", default="Qwen/Qwen-Image")
@ExampleRegister.register("qwen_image_2512", default="Qwen/Qwen-Image-2512")
@ExampleRegister.register("qwen_image_lightning", default="lightx2v/Qwen-Image-Lightning")
def qwen_image_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImagePipeline

  if "lightning" in args.example.lower():
    scheduler = _qwen_light_scheduler()
  else:
    scheduler = None

  if "lightning" in args.example.lower():
    # For lightning model, only 8 or 4 inference steps are supported
    steps = 8 if args.num_inference_steps is None else args.num_inference_steps
    assert steps in [8, 4]
    lora_weights_path = _path("lightx2v/Qwen-Image-Lightning", args=args, lora=True)
    lora_weight_name = f"Qwen-Image-Lightning-{steps}steps-V1.0-bf16.safetensors"
    cache_config = _qwen_light_cache_config(args)
    true_cfg_scale = 1.0  # means no separate cfg for lightning models
  else:
    steps = 50 if args.num_inference_steps is None else args.num_inference_steps
    lora_weights_path = None
    lora_weight_name = None
    cache_config = None
    true_cfg_scale = 4.0

  positive_magic = {
    "en": ", Ultra HD, 4K, cinematic composition.",  # for english prompt
    "zh": ", 超清，4K，电影级构图.",  # for chinese prompt
  }
  prompt = ("A coffee shop entrance features a chalkboard sign reading "
            '"Qwen Coffee 😊 $2 per cup," with a neon light beside it '
            'displaying "通义千问". Next to it hangs a poster showing a '
            "beautiful Chinese woman, and beneath the poster is written "
            '"π≈3.1415926-53589793-23846264-33832795-02384197". '
            "Ultra HD, 4K, cinematic composition")
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image"),
      pipeline_class=QwenImagePipeline,
      scheduler=scheduler,
      bnb_4bit_components=["text_encoder", "transformer"],
      lora_weights_path=lora_weights_path,
      lora_weights_name=lora_weight_name,
      force_fuse_lora=True,  # For parallelism compatibility
      extra_optimize_kwargs={
        "cache_config": cache_config,
      },
    ),
    input_data=ExampleInputData(
      prompt=prompt + positive_magic["en"],
      negative_prompt=" ",
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=true_cfg_scale,
    ),
  )


@ExampleRegister.register("qwen_image_edit", default="Qwen/Qwen-Image-Edit-2509")
@ExampleRegister.register("qwen_image_edit_lightning", default="lightx2v/Qwen-Image-Lightning")
@ExampleRegister.register("qwen_image_edit_2511", default="Qwen/Qwen-Image-Edit-2511")
@ExampleRegister.register("qwen_image_edit_2511_lightning",
                          default="lightx2v/Qwen-Image-Edit-2511-Lightning")
def qwen_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImageEditPlusPipeline

  if "lightning" in args.example.lower():
    scheduler = _qwen_light_scheduler()
  else:
    scheduler = None

  if "lightning" in args.example.lower():
    # For lightning model, only 8 or 4 inference steps are supported
    steps = 8 if args.num_inference_steps is None else args.num_inference_steps
    assert steps in [8, 4]
    if "2511" in args.example.lower():
      lora_weights_path = _path("lightx2v/Qwen-Image-Edit-2511-Lightning", args, lora=True)
      lora_weight_name = f"Qwen-Image-Edit-2511-Lightning-{steps}steps-V1.0-bf16.safetensors"
    else:
      lora_weights_path = os.path.join(
        _path("lightx2v/Qwen-Image-Lightning", args, lora=True),
        "Qwen-Image-Edit-2509",
      )
      lora_weight_name = f"Qwen-Image-Edit-2509-Lightning-{steps}steps-V1.0-bf16.safetensors"
    cache_config = _qwen_light_cache_config(args)
    true_cfg_scale = 1.0  # means no separate cfg for lightning models
  else:
    steps = 50 if args.num_inference_steps is None else args.num_inference_steps
    lora_weights_path = None
    lora_weight_name = None
    cache_config = None
    true_cfg_scale = 4.0

  if "2511" in args.example.lower():
    model_path_or_name = _path("Qwen/Qwen-Image-Edit-2511", args)
  else:
    model_path_or_name = _path("Qwen/Qwen-Image-Edit-2509", args)

  if _force_use_local_data():
    image1 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/edit2509_1.jpg"))
    image2 = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/edit2509_2.jpg"))
  else:
    image1 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_1.jpg")
    image2 = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path_or_name,
      pipeline_class=QwenImageEditPlusPipeline,
      scheduler=scheduler,
      bnb_4bit_components=["text_encoder", "transformer"],
      lora_weights_path=lora_weights_path,
      lora_weights_name=lora_weight_name,
      force_fuse_lora=True,  # For parallelism compatibility
      extra_optimize_kwargs={
        "cache_config": cache_config,
      },
    ),
    input_data=ExampleInputData(
      prompt=("The magician bear is on the left, the alchemist bear is on the right, "
              "facing each other in the central park square."),
      negative_prompt=" ",
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=true_cfg_scale,  # 1.0 means no separate cfg for lightning models
      image=[image1, image2],
    ),
  )


def _qwen_image_21_cache_config(
  args: argparse.Namespace,
  steps: int,
  *,
  separate_cfg: bool,
  distilled: bool,
) -> Optional[DBCacheConfig]:
  """DBCache config tuned for the Qwen-Image 2.1 family.

  The transformer runs a single forward per step unless classifier-free guidance is enabled with
  `true_cfg_scale > 1`, so `enable_separate_cfg` has to follow the actual number of forwards or
  cache-dit counts every step as half a step. `Fn_compute_blocks=1` keeps the cached prefix short
  and both checkpoints are flow-matching, where consecutive steps of the 8-step distilled one move
  further apart than those of the 40-step base model.

  :param args: CLI arguments.
  :param steps: Denoising steps the pipeline will actually run.
  :param separate_cfg: Whether the pipeline runs a negative branch as a second forward.
  :param distilled: Whether the checkpoint is the few-step Turbo variant.
  :returns: The DBCache config, or None when caching is disabled.
  """

  if not args.cache:
    return None
  # A model-provided cache config normally overrides the CLI knobs, so `--ws`/`--rdt` would be
  # silently ignored. Honour an explicit value and fall back to the tuned default otherwise.
  warmup = args.max_warmup_steps if args.max_warmup_steps != 8 else min(4, max(1, steps // 4))
  threshold = (args.residual_diff_threshold if args.residual_diff_threshold != 0.12 else
               (0.24 if distilled else 0.12))
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=warmup,
    num_inference_steps=steps,
    enable_separate_cfg=separate_cfg,
    residual_diff_threshold=threshold,
  )


def _qwen_image_21_edit_image(
  args: argparse.Namespace,
  model_path_or_name: str,
) -> "PIL.Image.Image":
  """Load the yacht sketch shipped with the Qwen-Image 2.1 Turbo assets.

  :param args: CLI arguments.
  :param model_path_or_name: Model id or local directory of the current example.
  :returns: The condition image used by the edit examples.
  """

  # "Qwen/Qwen-Image-2.1-Turbo" resolves to QWEN_IMAGE_21_TURBO_DIR when it is set
  candidates = [model_path_or_name, _path("Qwen/Qwen-Image-2.1-Turbo")]
  for base in candidates:
    if not os.path.isdir(base):
      continue
    path = os.path.join(base, "assets", "turbo-boat-input.png")
    if os.path.exists(path):
      return load_image(path)
  return load_image(
    "https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/raw/main/assets/turbo-boat-input.png")


def _qwen_image_21_modular_cfg_guider(pipe) -> None:
  """Replace the modular pipeline's guider with a CFG one.

  `ModularPipeline.__call__` does not route `guidance_scale` into a component config, so the
  default guider of the modular pipeline always samples without guidance. The image-conditioned
  workflow needs the replacement to actually run the negative branch.

  :param pipe: The freshly constructed modular pipeline.
  """

  from diffusers import ClassifierFreeGuidance

  pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=4.0))


_QWEN_IMAGE_21_PROMPT = (
  "A capybara wearing a wizard hat, reading a book by candlelight, oil painting")
_QWEN_IMAGE_21_EDIT_PROMPT = (
  "Turn this yacht design sketch into a painted side view of the finished yacht, broadside on a calm "
  "deep-blue sea, black hull, white superstructures and mast, yellow funnel, sunlight from the front "
  "left with a bright reflection path on the water.")


@ExampleRegister.register("qwen_image_21", default="Qwen/Qwen-Image-2.1")
def qwen_image_21_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21Pipeline

  steps = 40 if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image-2.1", args),
      pipeline_class=QwenImage21Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args,
                                                    steps,
                                                    separate_cfg=False,
                                                    distilled=False),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=1.0,  # sampled without guidance by default
    ),
  )


@ExampleRegister.register("qwen_image_21_turbo", default="Qwen/Qwen-Image-2.1-Turbo")
def qwen_image_21_turbo_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21Pipeline

  # The checkpoint ships its own 8-point sigma grid; `num_inference_steps` does not override it.
  steps = 8
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image-2.1-Turbo", args),
      pipeline_class=QwenImage21Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=False,
                                                    distilled=True),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=1.0,
    ),
  )


@ExampleRegister.register("qwen_image_21_edit", default="Qwen/Qwen-Image-2.1")
def qwen_image_21_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21Pipeline

  steps = 40 if args.num_inference_steps is None else args.num_inference_steps
  model_path_or_name = _path("Qwen/Qwen-Image-2.1", args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path_or_name,
      pipeline_class=QwenImage21Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=True,
                                                    distilled=False),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_EDIT_PROMPT,
      negative_prompt="blurry, low quality",
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=4.0,  # separate cfg: one negative branch per step
      image=[_qwen_image_21_edit_image(args, model_path_or_name)],
    ),
  )


@ExampleRegister.register("qwen_image_21_turbo_edit", default="Qwen/Qwen-Image-2.1-Turbo")
def qwen_image_21_turbo_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21Pipeline

  steps = 8
  model_path_or_name = _path("Qwen/Qwen-Image-2.1-Turbo", args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path_or_name,
      pipeline_class=QwenImage21Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=False,
                                                    distilled=True),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_EDIT_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=1.0,
      image=[_qwen_image_21_edit_image(args, model_path_or_name)],
    ),
  )


@ExampleRegister.register("qwen_image_21_modular", default="Qwen/Qwen-Image-2.1")
def qwen_image_21_modular_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21AutoBlocks, QwenImage21ModularPipeline

  steps = 40 if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image-2.1", args),
      pipeline_class=QwenImage21ModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      # the Qwen-Image 2.1 repo has no `modular_model_index.json`, so the blocks are explicit
      modular_blocks_factory=QwenImage21AutoBlocks,
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args,
                                                    steps,
                                                    separate_cfg=False,
                                                    distilled=False),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
    ),
  )


@ExampleRegister.register("qwen_image_21_turbo_modular", default="Qwen/Qwen-Image-2.1-Turbo")
def qwen_image_21_turbo_modular_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21TurboAutoBlocks, QwenImage21TurboModularPipeline

  steps = 8
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image-2.1-Turbo", args),
      pipeline_class=QwenImage21TurboModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      # the Qwen-Image 2.1 repo has no `modular_model_index.json`, so the blocks are explicit
      modular_blocks_factory=QwenImage21TurboAutoBlocks,
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=False,
                                                    distilled=True),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
    ),
  )


@ExampleRegister.register("qwen_image_21_modular_edit", default="Qwen/Qwen-Image-2.1")
def qwen_image_21_modular_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21AutoBlocks, QwenImage21ModularPipeline

  steps = 40 if args.num_inference_steps is None else args.num_inference_steps
  model_path_or_name = _path("Qwen/Qwen-Image-2.1", args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path_or_name,
      pipeline_class=QwenImage21ModularPipeline,
      bnb_4bit_components=[],
      modular_blocks_factory=QwenImage21AutoBlocks,
      # the modular guider defaults to guidance_scale=1.0, i.e. no negative branch
      post_init_hook=_qwen_image_21_modular_cfg_guider,
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=True,
                                                    distilled=False),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_EDIT_PROMPT,
      negative_prompt="blurry, low quality",
      height=1024,
      width=1024,
      num_inference_steps=steps,
      image=[_qwen_image_21_edit_image(args, model_path_or_name)],
    ),
  )


@ExampleRegister.register("qwen_image_21_turbo_modular_edit", default="Qwen/Qwen-Image-2.1-Turbo")
def qwen_image_21_turbo_modular_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImage21TurboAutoBlocks, QwenImage21TurboModularPipeline

  steps = 8
  model_path_or_name = _path("Qwen/Qwen-Image-2.1-Turbo", args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_path_or_name,
      pipeline_class=QwenImage21TurboModularPipeline,
      bnb_4bit_components=[],
      modular_blocks_factory=QwenImage21TurboAutoBlocks,
      extra_optimize_kwargs={
        "cache_config": _qwen_image_21_cache_config(args, steps, separate_cfg=False,
                                                    distilled=True),
      },
    ),
    input_data=ExampleInputData(
      prompt=_QWEN_IMAGE_21_EDIT_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
      true_cfg_scale=1.0,
      image=[_qwen_image_21_edit_image(args, model_path_or_name)],
    ),
  )


@ExampleRegister.register("qwen_image_controlnet",
                          default="InstantX/Qwen-Image-ControlNet-Inpainting")
def qwen_image_controlnet_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImageControlNetModel, QwenImageControlNetInpaintPipeline

  # make sure controlnet is on cuda to avoid device mismatch while using cpu offload
  controlnet = QwenImageControlNetModel.from_pretrained(
    _path(
      "InstantX/Qwen-Image-ControlNet-Inpainting",
      args=args,
      controlnet=True,
    ),
    torch_dtype=torch.bfloat16,
  )

  base_image_url = (
    "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets")
  control_image = load_image(f"{base_image_url}/images/image1.png").convert("RGB")
  control_mask = load_image(f"{base_image_url}/masks/mask1.png")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Qwen/Qwen-Image"),
      pipeline_class=QwenImageControlNetInpaintPipeline,
      controlnet=controlnet,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt="一辆绿色的出租车行驶在路上",
      negative_prompt="worst quality, low quality, blurry, text, watermark, logo",
      control_image=control_image,
      control_mask=control_mask,
      controlnet_conditioning_scale=1.0,
      height=control_mask.size[1] if args.height is None else args.height,
      width=control_mask.size[0] if args.width is None else args.width,
      num_inference_steps=50,
      true_cfg_scale=4.0,
    ),
  )


@ExampleRegister.register("qwen_image_layered", default="Qwen/Qwen-Image-Layered")
def qwen_image_layered_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImageLayeredPipeline

  model_name_or_path = _path("Qwen/Qwen-Image-Layered", args=args)
  if _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR,
                                    "examples/data/yarn-art-pikachu.png")).convert("RGBA")
  else:
    image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/yarn-art-pikachu.png").convert(
        "RGBA")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=model_name_or_path,
      pipeline_class=QwenImageLayeredPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "enable_separate_cfg": False,  # negative prompt is not used in example
      },
    ),
    input_data=ExampleInputData(
      image=image,
      prompt="",
      num_inference_steps=50,
      true_cfg_scale=4.0,
      layers=4,
      resolution=640,
      cfg_normalize=False,
      use_en_prompt=True,
    ),
  )


@ExampleRegister.register("skyreels_v2", default="Skywork/SkyReels-V2-T2V-14B-720P-Diffusers")
def skyreels_v2_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import AutoModel, SkyReelsV2Pipeline, UniPCMultistepScheduler

  model_name_or_path = _path(
    "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
    args=args,
  )
  vae = AutoModel.from_pretrained(
    model_name_or_path if args.model_path is None else args.model_path,
    subfolder="vae",
    torch_dtype=torch.float32,
  )  # Use float32 VAE to reduce video generation artifacts

  def post_init_hook(pipe: SkyReelsV2Pipeline, **kwargs):
    flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config,
                                                         flow_shift=flow_shift)
    logger.info(f"Set UniPCMultistepScheduler with flow_shift={flow_shift} "
                f"for {pipe.__class__.__name__}.")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,  # Text to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=SkyReelsV2Pipeline,
      vae=vae,
      post_init_hook=post_init_hook,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("A cat and a dog baking a cake together in a kitchen. The cat is "
              "carefully measuring flour, while the dog is stirring the batter "
              "with a wooden spoon. The kitchen is cozy, with sunlight streaming "
              "through the window."),
      height=720,
      width=1280,
      num_frames=21,
      num_inference_steps=50,
    ),
  )


@ExampleRegister.register("ltx2_t2v", default="Lightricks/LTX-2")
def ltx2_t2v_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import LTX2Pipeline

  model_name_or_path = _path(
    "Lightricks/LTX-2",
    args=args,
  )
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,  # Text to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=LTX2Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=(
        "A cinematic tracking shot through a neon-lit rainy cyberpunk street at night. "
        "Reflections shimmer on wet asphalt, holographic signs flicker, and steam rises from vents. "
        "Smooth camera motion, natural parallax, ultra-realistic detail, cinematic lighting."),
      negative_prompt=(
        "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion artifacts, "
        "bad anatomy, ugly, transition, static, text, watermark"),
      height=512,
      width=768,
      num_frames=121,
      num_inference_steps=40,
      guidance_scale=4.0,
      extra_input_kwargs={
        "frame_rate": 24.0,
      },
    ),
  )


@ExampleRegister.register("ltx2_i2v", default="Lightricks/LTX-2")
def ltx2_i2v_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import LTX2ImageToVideoPipeline

  model_name_or_path = _path(
    "Lightricks/LTX-2",
    args=args,
  )

  height = 512 if args.height is None else args.height
  width = 768 if args.width is None else args.width
  if args.image_path is not None:
    image = load_image(args.image_path)
  else:
    image = load_image(
      "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
    )
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.I2V,  # Image to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=LTX2ImageToVideoPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("A young girl stands calmly in the foreground, looking directly at the camera, "
              "as a house fire rages in the background."),
      negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
      image=image,
      height=height,
      width=width,
      num_frames=121,
      num_inference_steps=40,
      guidance_scale=4.0,
      extra_input_kwargs={
        "frame_rate": 24.0,
      },
    ),
  )


def _ltx25_cache_config(args: argparse.Namespace, steps: int) -> Optional[DBCacheConfig]:
  """Build the DBCache config shared by the LTX-2.5 pipelines.

  LTX-2.5 is distilled: the guided `LTX2Pipeline` still emits a conditional, an STG and a modality
  pass per denoise step, but only the conditional pass goes through the cache (the other two change
  the block semantics or the token geometry), and `LTX2DFRPipeline` emits exactly one pass per step
  with no CFG, so `enable_separate_cfg` stays `False`. `num_inference_steps` is stated explicitly
  because these pipelines schedule their own passes (`sigmas` for DFR) and the CLI cannot infer the
  count; without it the cache context never refreshes when several passes are chained.

  :param args: CLI arguments.
  :param steps: Number of cached transformer calls in one pass.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  max_warmup_steps = args.max_warmup_steps
  if steps <= 8:
    # A distilled schedule is short; the default warmup would leave nothing to reuse.
    max_warmup_steps = min(max_warmup_steps, 2)
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=max_warmup_steps,
    num_inference_steps=steps,
    enable_separate_cfg=False,
    residual_diff_threshold=args.residual_diff_threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=args.max_continuous_cached_steps,
  )


@ExampleRegister.register("ltx25_t2v", default="Lightricks/LTX-2.5-Diffusers")
def ltx25_t2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """T2VA (joint text to video + audio) with LTX-2.5.

  `LTX2Pipeline` is the guided entry point: it runs classifier-free guidance, spatio-temporal
  guidance and modality guidance, and returns both streams plus a soundtrack. The cache only
  replays the conditional pass; the STG pass (per-block perturbation) and the modality pass
  (isolated cross attention) are computed in full, which is what keeps the output identical to the
  uncached run.

  :param args: CLI arguments.
  :returns: The LTX-2.5 t2v example.
  """
  from diffusers import LTX2Pipeline

  model_name_or_path = _path("Lightricks/LTX-2.5-Diffusers", args=args)
  steps = 30 if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,  # Text to video + audio
      model_name_or_path=model_name_or_path,
      pipeline_class=LTX2Pipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      skip_components=["prompt_enhancer"],
      extra_optimize_kwargs={"cache_config": _ltx25_cache_config(args, steps)},
    ),
    input_data=ExampleInputData(
      prompt=("A cinematic tracking shot through a neon-lit rainy street at night. Reflections "
              "shimmer on wet asphalt, holographic signs flicker, and steam rises from vents."),
      negative_prompt=(
        "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, "
        "motion smear, motion artifacts, fused fingers, bad anatomy, ugly, static."),
      height=512,
      width=768,
      num_frames=121,
      num_inference_steps=30,
      guidance_scale=3.0,
      extra_input_kwargs={
        "frame_rate": 24.0,
      },
    ),
  )


@ExampleRegister.register("ltx25_i2v", default="Lightricks/LTX-2.5-Diffusers")
def ltx25_i2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """I2VA (image + text to video + audio) with LTX-2.5.

  :param args: CLI arguments.
  :returns: The LTX-2.5 i2v example.
  """
  from diffusers import LTX2ImageToVideoPipeline

  model_name_or_path = _path("Lightricks/LTX-2.5-Diffusers", args=args)
  steps = 30 if args.num_inference_steps is None else args.num_inference_steps

  height = 512 if args.height is None else args.height
  width = 768 if args.width is None else args.width
  if args.image_path is not None:
    image = load_image(args.image_path)
  else:
    image = load_image(
      "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/images/8.png"
    )
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.I2V,  # the soundtrack is dropped for the I2V task type
      model_name_or_path=model_name_or_path,
      pipeline_class=LTX2ImageToVideoPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      skip_components=["prompt_enhancer"],
      extra_optimize_kwargs={"cache_config": _ltx25_cache_config(args, steps)},
    ),
    input_data=ExampleInputData(
      prompt=("A young girl stands calmly in the foreground, looking directly at the camera, "
              "as a house fire rages in the background."),
      negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
      image=image,
      height=height,
      width=width,
      num_frames=121,
      num_inference_steps=30,
      guidance_scale=3.0,
      extra_input_kwargs={
        "frame_rate": 24.0,
      },
    ),
  )


@ExampleRegister.register("ltx25_dfr", default="Lightricks/LTX-2.5-Diffusers")
def ltx25_dfr_example(args: argparse.Namespace, **kwargs) -> Example:
  """One DFR (Diffusion Fidelity Rendering) pass with LTX-2.5.

  `LTX2DFRPipeline` takes an explicit distilled sigma schedule instead of `num_inference_steps`
  (8 steps, one transformer call per step, no CFG), so `--num-inference-steps` is not forwarded and
  the cache config derives its step count from the schedule. DFR's `video_tiles` epilogue path
  computes a token *subset* per call and is therefore not cacheable; keep `video_tiles` unset when
  caching is enabled.

  :param args: CLI arguments.
  :returns: The LTX-2.5 DFR example.
  """
  from diffusers import LTX2DFRPipeline
  from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

  model_name_or_path = _path("Lightricks/LTX-2.5-Diffusers", args=args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,  # DFR renders video + audio
      model_name_or_path=model_name_or_path,
      pipeline_class=LTX2DFRPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      skip_components=["prompt_enhancer"],
      extra_optimize_kwargs={
        "cache_config": _ltx25_cache_config(args, len(DISTILLED_SIGMA_VALUES))
      },
    ),
    input_data=ExampleInputData(
      prompt=("A tabby cat stretching in a sunlit window, dust motes drifting in the light"),
      height=512,
      width=768,
      num_frames=121,
      use_num_inference_steps=False,
      extra_input_kwargs={
        "sigmas": DISTILLED_SIGMA_VALUES,
        "frame_rate": 24.0,
      },
    ),
  )


def _wan_2_2_params_modifiers(args: argparse.Namespace) -> List[ParamsModifier]:
  if not args.cache:
    return None
  return [
    ParamsModifier(
      # high-noise transformer only have 30% steps
      cache_config=DBCacheConfig().reset(
        max_warmup_steps=4,
        max_cached_steps=8,
      ), ),
    ParamsModifier(cache_config=DBCacheConfig().reset(
      max_warmup_steps=2,
      max_cached_steps=20,
    ), ),
  ]


@ExampleRegister.register("wan2.1_t2v", default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
@ExampleRegister.register("wan2.2_t2v", default="Wan-AI/Wan2.2-T2V-A14B-Diffusers")
def wan_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import WanPipeline

  if "wan2.2" in args.example.lower():
    model_name_or_path = _path(
      "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
      args=args,
    )
  else:
    model_name_or_path = _path(
      "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
      args=args,
    )

  if "wan2.2" in args.example.lower():
    params_modifiers = _wan_2_2_params_modifiers(args)
  else:
    params_modifiers = None

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,  # Text to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=WanPipeline,
      bnb_4bit_components=(["text_encoder", "transformer", "transformer_2"] if "wan2.2"
                           in args.example.lower() else ["text_encoder", "transformer"]),
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
      },
    ),
    input_data=ExampleInputData(
      prompt="A cat walks on the grass, realistic",
      negative_prompt=("Bright tones, overexposed, static, blurred details, subtitles, "
                       "style, works, paintings, images, static, overall gray, worst quality, "
                       "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                       "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
                       "limbs, fused fingers, still picture, messy background, three legs, many "
                       "people in the background, walking backwards"),
      height=480,
      width=832,
      num_frames=49,
      guidance_scale=5.0,
      num_inference_steps=30,
    ),
  )


@ExampleRegister.register("wan2.1_i2v", default="Wan-AI/Wan2.1-I2V-14B-480P-Diffusers")
@ExampleRegister.register("wan2.2_i2v", default="Wan-AI/Wan2.2-I2V-A14B-Diffusers")
def wan_i2v_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import WanImageToVideoPipeline

  if "wan2.2" in args.example.lower():
    model_name_or_path = _path(
      "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
      args=args,
    )
  else:
    model_name_or_path = _path(
      "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
      args=args,
    )

  if "wan2.2" in args.example.lower():
    params_modifiers = _wan_2_2_params_modifiers(args)
  else:
    params_modifiers = None

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  else:
    image = load_image(
      "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG")

  max_area = 480 * 832
  aspect_ratio = image.height / image.width
  vae_scale_factor_spatial = 8  # for Wan VAE
  patch_size = 2  # for Wan transformer, [1, 2, 2]
  mod_value = vae_scale_factor_spatial * patch_size
  height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
  width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.I2V,  # Image to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=WanImageToVideoPipeline,
      bnb_4bit_components=(["text_encoder", "transformer", "transformer_2"] if "wan2.2"
                           in args.example.lower() else ["text_encoder", "transformer"]),
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
      },
    ),
    input_data=ExampleInputData(
      prompt=("Summer beach vacation style, a white cat wearing sunglasses sits on a "
              "surfboard. The fluffy-furred feline gazes directly at the camera with "
              "a relaxed expression. Blurred beach scenery forms the background featuring "
              "crystal-clear waters, distant green hills, and a blue sky dotted with white "
              "clouds. The cat assumes a naturally relaxed posture, as if savoring the sea "
              "breeze and warm sunlight. A close-up shot highlights the feline's intricate "
              "details and the refreshing atmosphere of the seaside."),
      negative_prompt=("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                       "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                       "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"),
      image=image,
      height=height,
      width=width,
      num_frames=49,
      guidance_scale=3.5,
      num_inference_steps=50,
    ),
  )


@ExampleRegister.register("wan2.1_vace", default="Wan-AI/Wan2.1-VACE-1.3B-diffusers")
@ExampleRegister.register("wan2.2_vace", default="linoyts/Wan2.2-VACE-Fun-14B-diffusers")
def wan_vace_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import WanVACEPipeline, AutoencoderKLWan, UniPCMultistepScheduler

  if "wan2.2" in args.example.lower():
    model_name_or_path = _path(
      "linoyts/Wan2.2-VACE-Fun-14B-diffusers",
      args=args,
    )
  else:
    model_name_or_path = _path(
      "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
      args=args,
    )

  vae = AutoencoderKLWan.from_pretrained(
    model_name_or_path,
    subfolder="vae",
    torch_dtype=torch.float32,
  )

  def post_init_hook(pipe: WanVACEPipeline, **kwargs):
    flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
    pipe.scheduler = UniPCMultistepScheduler.from_config(
      pipe.scheduler.config,
      flow_shift=flow_shift,
    )
    logger.info(f"Set UniPCMultistepScheduler with flow_shift={flow_shift} "
                f"for {pipe.__class__.__name__}.")

  if "wan2.2" in args.example.lower():
    params_modifiers = _wan_2_2_params_modifiers(args)
  else:
    params_modifiers = None

  def _video_and_mask(
    first_img: PIL.Image.Image,
    last_img: PIL.Image.Image,
    height: int,
    width: int,
    num_frames: int,
  ) -> Tuple[List[PIL.Image.Image], List[PIL.Image.Image]]:
    first_img = first_img.resize((width, height))
    last_img = last_img.resize((width, height))
    frames = []
    frames.append(first_img)
    # Ideally, this should be 127.5 to match original code, but they perform
    # computation on numpy arrays whereas we are passing PIL images. If you
    # choose to pass numpy arrays, you can set it to 127.5 to match the original code.
    frames.extend([PIL.Image.new("RGB", (width, height), (128, 128, 128))] * (num_frames - 2))
    frames.append(last_img)
    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    mask = [mask_black, *[mask_white] * (num_frames - 2), mask_black]
    return frames, mask

  if _force_use_local_data():
    first_frame = load_image(
      os.path.join(CACHE_DIT_DIR, "examples/data/flf2v_input_first_frame.png"))
    last_frame = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/flf2v_input_last_frame.png"))
  else:
    first_frame = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/flf2v_input_first_frame.png")
    last_frame = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/flf2v_input_last_frame.png")

  height = 512 if args.height is None else args.height
  width = 512 if args.width is None else args.width
  num_frames = 81 if args.num_frames is None else args.num_frames
  video, mask = _video_and_mask(first_frame, last_frame, height, width, num_frames)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.VACE,  # Video All-in-one Creation and Editing
      model_name_or_path=model_name_or_path,
      pipeline_class=WanVACEPipeline,
      vae=vae,
      post_init_hook=post_init_hook,
      bnb_4bit_components=(["text_encoder", "transformer", "transformer_2"] if "wan2.2"
                           in args.example.lower() else ["text_encoder", "transformer"]),
      extra_optimize_kwargs={
        "params_modifiers": params_modifiers,
      },
    ),
    input_data=ExampleInputData(
      prompt=("CG animation style, a small blue bird takes off from the ground, "
              "flapping its wings. The bird's feathers are delicate, with a unique "
              "pattern on its chest. The background shows a blue sky with white "
              "clouds under bright sunshine. The camera follows the bird upward, "
              "capturing its flight and the vastness of the sky from a close-up, "
              "low-angle perspective."),
      negative_prompt=("Bright tones, overexposed, static, blurred details, subtitles, "
                       "style, works, paintings, images, static, overall gray, worst "
                       "quality, low quality, JPEG compression residue, ugly, incomplete, "
                       "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                       "disfigured, misshapen limbs, fused fingers, still picture, messy "
                       "background, three legs, many people in the background, walking "
                       "backwards"),
      video=video,
      mask=mask,
      height=height,
      width=width,
      num_frames=num_frames,
      guidance_scale=5.0,
      num_inference_steps=30,
    ),
  )


@ExampleRegister.register("ovis_image", default="AIDC-AI/Ovis-Image-7B")
def ovis_image_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import OvisImagePipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("AIDC-AI/Ovis-Image-7B"),
      pipeline_class=OvisImagePipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=(
        'A creative 3D artistic render where the text "OVIS-IMAGE" is written in a bold, '
        "expressive handwritten brush style using thick, wet oil paint. The paint is a mix "
        "of vibrant rainbow colors (red, blue, yellow) swirling together like toothpaste "
        "or impasto art. You can see the ridges of the brush bristles and the glossy, wet "
        "texture of the paint. The background is a clean artist's canvas. Dynamic lighting "
        "creates soft shadows behind the floating paint strokes. Colorful, expressive, tactile "
        "texture, 4k detail."),
      height=1024,
      width=1024,
      num_inference_steps=25,
      guidance_scale=5.0,  # has separate cfg for ovis image
    ),
  )


def _zimage_turbo_steps_mask(args: argparse.Namespace, ) -> Optional[List[int]]:
  if not args.cache:
    return None
  return (steps_mask(
    # slow, medium, fast, ultra.
    mask_policy=args.mask_policy,
    total_steps=9 if args.num_inference_steps is None else args.num_inference_steps,
  ) if args.mask_policy is not None else (steps_mask(
    compute_bins=[5, 1, 1],  # = 7 (compute steps)
    cache_bins=[1, 1],  # = 2 (dynamic cache steps)
  ) if args.steps_mask else None))


@ExampleRegister.register("zimage", default="Tongyi-MAI/Z-Image")
@ExampleRegister.register("zimage_turbo", default="Tongyi-MAI/Z-Image-Turbo")
@ExampleRegister.register("zimage_turbo_nunchaku", default="nunchaku/nunchaku-z-image-turbo")
def zimage_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import ZImagePipeline

  if args.cache:
    # Only warmup 4 steps (total 9 steps) for distilled models
    args.max_warmup_steps = min(4, args.max_warmup_steps)

  if "nunchaku" in args.example.lower():
    from nunchaku import NunchakuZImageTransformer2DModel

    nunchaku_zimage_dir = _path(
      "nunchaku-tech/nunchaku-z-image-turbo",
      args=args,
      transformer=True,
    )
    transformer = NunchakuZImageTransformer2DModel.from_pretrained(
      f"{nunchaku_zimage_dir}/svdq-int4_r128-z-image-turbo.safetensors")
  else:
    transformer = None

  if "turbo" in args.example.lower():
    model_name_or_path = _path("Tongyi-MAI/Z-Image-Turbo", args=args)
    prompt = ("Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
              "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
              "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
              "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
              "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
              "blurred colorful distant lights.")
    negative_prompt = None
    height = 1024
    width = 1024
    guidance_scale = 0.0  # Guidance should be 0 for the Turbo models
    num_inference_steps = 9
    steps_computation_mask = _zimage_turbo_steps_mask(args)
  else:
    model_name_or_path = _path("Tongyi-MAI/Z-Image", args=args)
    prompt = ("两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，可能是室内地毯地面。"
              "左侧女性留着长卷发，身穿藏青色毛衣，左袖有奶油色褶皱装饰，内搭白色立领衬衫，"
              "下身白色裤子；佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，身穿奶油色卫衣，"
              "胸前印有“Tun the tables”字样，下方为“New ideas”，搭配白色裤子；佩戴银色小环耳环，"
              "双臂交叉于胸前。两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，以藏青、"
              "奶油白为主的中性色调，休闲时尚摄影，中等景深，面部和上半身对焦清晰，姿态放松，"
              "表情友好，室内环境，地毯地面，纯色背景。")
    negative_prompt = ""
    height = 1280
    width = 720
    num_inference_steps = 50
    guidance_scale = 4.0
    steps_computation_mask = None

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=model_name_or_path,
      pipeline_class=ZImagePipeline,
      transformer=transformer,  # maybe use Nunchaku zimage transformer
      bnb_4bit_components=["text_encoder"],
      extra_optimize_kwargs={
        "steps_computation_mask": steps_computation_mask,
      },
    ),
    input_data=ExampleInputData(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=height,
      width=width,
      guidance_scale=guidance_scale,  # Guidance should be 0 for the Turbo models
      num_inference_steps=num_inference_steps,
      extra_input_kwargs={
        "cfg_normalization": False,
      },
    ),
  )


@ExampleRegister.register("zimage_turbo_controlnet_2.1",
                          default="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1")
@ExampleRegister.register("zimage_turbo_controlnet_2.0",
                          default="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0")
def zimage_controlnet_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import ZImageControlNetPipeline, ZImageControlNetModel

  if args.cache:
    # Only warmup 4 steps (total 9 steps) for distilled models
    args.max_warmup_steps = min(4, args.max_warmup_steps)

  if "2.0" in args.example.lower():
    controlnet_dir = _path(
      "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0",
      args=args,
      controlnet=True,
    )
    controlnet_path = os.path.join(controlnet_dir,
                                   "Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors")
    controlnet = ZImageControlNetModel.from_single_file(
      controlnet_path,
      torch_dtype=torch.bfloat16,
      config="hlky/Z-Image-Turbo-Fun-Controlnet-Union-2.0",
    )
  else:
    controlnet_dir = _path(
      "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
      args=args,
      controlnet=True,
    )
    controlnet_path = os.path.join(controlnet_dir,
                                   "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors")
    controlnet = ZImageControlNetModel.from_single_file(
      controlnet_path,
      torch_dtype=torch.bfloat16,
    )

  if _force_use_local_data():
    control_image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/pose.jpg"))
  else:
    control_image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/pose.jpg")
  steps_computation_mask = _zimage_turbo_steps_mask(args)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("Tongyi-MAI/Z-Image-Turbo"),
      pipeline_class=ZImageControlNetPipeline,
      controlnet=controlnet,
      bnb_4bit_components=["text_encoder"],
      extra_optimize_kwargs={
        "steps_computation_mask": steps_computation_mask,
      },
    ),
    input_data=ExampleInputData(
      prompt=("一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，"
              "发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；"
              "神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，"
              "是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"),
      control_image=control_image,
      controlnet_conditioning_scale=0.75,
      height=1728,
      width=992,
      num_inference_steps=9,
      guidance_scale=0.0,
    ),
  )


@ExampleRegister.register("longcat_image", default="meituan-longcat/LongCat-Image")
def longcat_image_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import LongCatImagePipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("meituan-longcat/LongCat-Image"),
      pipeline_class=LongCatImagePipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=
      ("A young Asian woman wearing a yellow knit sweater paired with a white necklace. "
       "Her hands rest on her knees, with a serene expression. The background features a "
       "rough brick wall, with warm afternoon sunlight casting upon her, creating a tranquil "
       "and cozy atmosphere. The shot uses a medium-distance perspective, highlighting her "
       "demeanor and the details of her attire. Soft lighting illuminates her face, emphasizing "
       "her facial features and the texture of her accessories, adding depth and warmth to the image. "
       "The overall composition is simple and elegant, with the brick wall's texture complementing "
       "the interplay of sunlight and shadows, showcasing the character's grace and composure."),
      height=1024,
      width=1024,
      num_inference_steps=50,
      guidance_scale=4.5,
    ),
  )


@ExampleRegister.register("longcat_image_edit", default="meituan-longcat/LongCat-Image-Edit")
def longcat_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import LongCatImageEditPipeline

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  else:
    image_url = (
      "https://huggingface.co/meituan-longcat/LongCat-Image-Edit/resolve/main/assets/test.png")
    image = load_image(image_url).convert("RGB")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=_path("meituan-longcat/LongCat-Image-Edit"),
      pipeline_class=LongCatImageEditPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("Turn the cat into a dog"),
      negative_prompt="",
      num_inference_steps=50,
      guidance_scale=4.5,
      image=image,
    ),
  )


@ExampleRegister.register("glm_image", default="zai-org/GLM-Image")
def glm_image_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import GlmImagePipeline

  force_refresh_step_hint = None  # Set to 1 if the 'image' parameter is used in input_data

  model_name_or_path = _path("zai-org/GLM-Image", args=args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=model_name_or_path,
      pipeline_class=GlmImagePipeline,
      bnb_4bit_components=["vision_language_encoder", "transformer"],
      extra_optimize_kwargs={
        "force_refresh_step_hint": force_refresh_step_hint,
      },
    ),
    input_data=ExampleInputData(
      prompt="A photo of an astronaut riding a horse on mars",
      num_inference_steps=50,
      height=1024,
      width=1024,
    ),
  )


@ExampleRegister.register("glm_image_edit", default="zai-org/GLM-Image")
def glm_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import GlmImagePipeline

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  elif _force_use_local_data():
    image_url = os.path.join(CACHE_DIT_DIR, "examples/data/snow_cat.png")
    image = load_image(image_url).convert("RGB")
  else:
    image_url = "https://github.com/vipshop/cache-dit/raw/main/examples/data/snow_cat.png"
    image = load_image(image_url).convert("RGB")

  # Since 'image' parameter is used in input_data, we have set the value of
  # force_refresh_step_hint to the number of prompts x number of images
  # which is 1 x 1 = 1 here. GLM-Image will do processing for the prompt
  # and image at each pipeline inference by calling the transformer, so,
  # we need to force refresh the cached hidden states at after the
  # preprocessing done.
  force_refresh_step_hint = 1

  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  image = image.resize((width, height))

  model_name_or_path = _path("zai-org/GLM-Image", args=args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_name_or_path,
      pipeline_class=GlmImagePipeline,
      bnb_4bit_components=["vision_language_encoder", "transformer"],
      extra_optimize_kwargs={
        "force_refresh_step_hint": force_refresh_step_hint,
      },
    ),
    input_data=ExampleInputData(
      prompt=
      "Replace the background of the snow forest with an underground station featuring an automatic escalator.",
      image=[image],  # pass as a list for GLM-Image Edit
      height=height,
      width=width,
      num_inference_steps=50,
    ),
  )


@ExampleRegister.register("firered_image_edit_1.0", default="FireRedTeam/FireRed-Image-Edit-1.0")
@ExampleRegister.register("firered_image_edit_1.1", default="FireRedTeam/FireRed-Image-Edit-1.1")
def firered_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import QwenImageEditPlusPipeline

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  elif _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR,
                                    "examples/data/firered_edit_example.png")).convert("RGB")
  else:
    image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/firered_edit_example.png"
    ).convert("RGB")

  if args.input_height is not None and args.input_width is not None:
    image = image.resize((args.input_width, args.input_height))

  if args.example.lower() == "firered_image_edit_1.1":
    model_name_or_path = _path("FireRedTeam/FireRed-Image-Edit-1.1", args=args)
  else:
    model_name_or_path = _path("FireRedTeam/FireRed-Image-Edit-1.0", args=args)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,  # Image Editing to Image
      model_name_or_path=model_name_or_path,
      pipeline_class=QwenImageEditPlusPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("在书本封面Python的下方，添加一行英文文字2nd Edition"),
      negative_prompt=" ",
      height=1024,
      width=1024,
      seed=43,
      gen_device="cuda",  # align with the official example.
      num_inference_steps=40,
      true_cfg_scale=4.0,
      image=[image],
    ),
  )


@ExampleRegister.register("helios_t2v", default="BestWishYsh/Helios-Base")
def helios_t2v_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import HeliosPipeline, AutoencoderKLWan
  from ..platforms import current_platform

  model_name_or_path = _path("BestWishYsh/Helios-Base", args=args)
  vae = AutoencoderKLWan.from_pretrained(model_name_or_path,
                                         subfolder="vae",
                                         torch_dtype=torch.float32)

  num_frames = 49  # < Hopper (Ada, Ampere)
  if current_platform.device_type == "cuda":
    if current_platform.get_device_capability() >= (9, 0):
      num_frames = 132  # >= Hopper

  num_inference_steps = 50 if args.num_inference_steps is None else args.num_inference_steps
  # update cache context per num_inference_steps (e.g, 50) since Helios will split the num_frames
  # into multiple chunks and do multiple passes of transformer denoise loop, and the cache context
  # should be refreshed at the end of each loop to ensure the previous cache will never be used
  # in the next loop.
  force_refresh_step_hint = num_inference_steps
  # repeat the same hint for multiple passes of transformer denoise loop
  force_refresh_step_policy = "repeat"  # 'once' or 'repeat'

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,  # Text to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=HeliosPipeline,
      vae=vae,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "force_refresh_step_hint": force_refresh_step_hint,
        "force_refresh_step_policy": force_refresh_step_policy,
      },
    ),
    input_data=ExampleInputData(
      prompt=(
        "A cat and a dog baking a cake together in a kitchen. The cat is "
        "carefully measuring flour, while the dog is stirring the batter with a wooden spoon. "
        "The kitchen is cozy, with sunlight streaming through the window."),
      negative_prompt=(
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
        "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"),
      height=384,
      width=640,
      num_frames=num_frames,
      num_inference_steps=num_inference_steps,
      guidance_scale=5.0,
    ),
  )


@ExampleRegister.register("helios_t2v_distill", default="BestWishYsh/Helios-Distilled")
def helios_t2v_distill_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import HeliosPyramidPipeline, AutoencoderKLWan
  from ..platforms import current_platform

  assert args.num_inference_steps is None, (
    "Helios Distilled example should use the default pyramid_num_inference_steps_list "
    "which is [2,2,2] for better performance. Please do not set num_inference_steps for "
    "this example since it will break the step hint logic and lead to worse performance.")

  model_name_or_path = _path("BestWishYsh/Helios-Distilled", args=args)
  vae = AutoencoderKLWan.from_pretrained(model_name_or_path,
                                         subfolder="vae",
                                         torch_dtype=torch.float32)

  num_frames = 49  # < Hopper (Ada, Ampere)
  if current_platform.device_type == "cuda":
    if current_platform.get_device_capability() >= (9, 0):
      num_frames = 132  # >= Hopper

  force_refresh_step_hint = 2
  # repeat the same hint for multiple passes of transformer denoise loop
  force_refresh_step_policy = "repeat"  # 'once' or 'repeat'

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,  # Text to Video
      model_name_or_path=model_name_or_path,
      pipeline_class=HeliosPyramidPipeline,
      vae=vae,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "force_refresh_step_hint": force_refresh_step_hint,
        "force_refresh_step_policy": force_refresh_step_policy,
        "enable_seperate_cfg": False,  # For distilled models (guidance_scale=1.0).
      },
    ),
    input_data=ExampleInputData(
      prompt=(
        "A cat and a dog baking a cake together in a kitchen. The cat is "
        "carefully measuring flour, while the dog is stirring the batter with a wooden spoon. "
        "The kitchen is cozy, with sunlight streaming through the window."),
      negative_prompt=(
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
        "images, static, overall gray, worst quality, low quality, JPEG compression residue, "
        "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
        "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
        "many people in the background, walking backwards"),
      height=384,
      width=640,
      num_frames=num_frames,
      guidance_scale=1.0,
      gen_device="cuda",  # align with the official example.
      extra_input_kwargs={
        "pyramid_num_inference_steps_list": [2, 2, 2],
        "is_amplify_first_chunk": True,
      },
    ),
  )


def _boogu_post_init_hook(pipe):
  """Wrap pipe.__call__ to translate standard cache-dit param names to BooguImagePipeline names.

  BooguImagePipeline uses ``instruction`` instead of ``prompt`` and ``input_images``
  instead of ``image``. This hook makes the pipeline compatible with cache-dit's
  standard ExampleInputData interface.
  """
  _original_call = pipe.__class__.__call__

  def _wrapped_call(self, **kwargs):
    if "prompt" in kwargs:
      kwargs["instruction"] = kwargs.pop("prompt")
    if "negative_prompt" in kwargs:
      kwargs["negative_instruction"] = kwargs.pop("negative_prompt")
    if "image" in kwargs:
      kwargs["input_images"] = kwargs.pop("image")
    return _original_call(self, **kwargs)

  pipe.__class__.__call__ = _wrapped_call


@ExampleRegister.register("cosmos3_omni_t2v", default="nvidia/Cosmos3-Nano")
@ExampleRegister.register("cosmos3_omni_t2i", default="nvidia/Cosmos3-Nano")
def cosmos3_omni_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Cosmos3OmniPipeline

  model_name_or_path = _path("nvidia/Cosmos3-Nano", args=args)

  class _Cosmos3OmniPipelineLoader:
    """Load Cosmos3OmniPipeline with the optional guardrail disabled.

    The Cosmos guardrail requires the optional `cosmos_guardrail` package
    (plus extra model downloads). Disable it so the benchmark example runs
    out of the box (the returned object is a genuine Cosmos3OmniPipeline,
    so cache-dit adapter matching works). Install `cosmos_guardrail` and
    pass `enable_safety_checker=True` to generate real content under the
    NVIDIA Open Model License terms.
    """

    @staticmethod
    def from_pretrained(*fp_args, **fp_kwargs):
      fp_kwargs.setdefault("enable_safety_checker", False)
      return Cosmos3OmniPipeline.from_pretrained(*fp_args, **fp_kwargs)

  # Cosmos3 Omni selects the task from `num_frames`: 1 -> T2I, >1 -> T2V.
  is_t2i = "t2i" in args.example.lower()
  num_frames = 1 if is_t2i else 93

  # The pipeline output field is `video` (list of PIL frames), not `frames`;
  # rename so the generic T2V/T2I output handling can pick it up.
  def _patch_output_alias(pipe, **kwargs):
    _original_call = pipe.__class__.__call__

    def _wrapped_call(self, **call_kwargs):
      output = _original_call(self, **call_kwargs)
      if hasattr(output, "video") and not hasattr(output, "frames"):
        if is_t2i:
          output.images = output.video  # list of PIL frames; T2I -> 1 frame
        else:
          output.frames = [output.video]
      return output

    pipe.__class__.__call__ = _wrapped_call

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I if is_t2i else ExampleType.T2V,
      model_name_or_path=model_name_or_path,
      pipeline_class=_Cosmos3OmniPipelineLoader,
      bnb_4bit_components=["transformer"],
      post_init_hook=_patch_output_alias,
    ),
    input_data=ExampleInputData(
      prompt=("A high-tech warehouse robot arm carefully picks up a glass vial "
              "from a metal tray and places it into a padded container, smooth "
              "and precise motion, photorealistic, cinematic lighting"),
      height=480,
      width=832,
      num_frames=num_frames,
      num_inference_steps=35,
      guidance_scale=6.0,
    ),
  )


@ExampleRegister.register("boogu_image_edit", default="BOOGU-IMAGE/Boogu-Image-0.1-Edit")
def boogu_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  elif _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/boogu_01.png")).convert("RGB")
  else:
    image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/boogu_01.png").convert("RGB")

  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,
      model_name_or_path=_path("BOOGU-IMAGE/Boogu-Image-0.1-Edit"),
      pipeline_class=BooguImagePipeline,
      trust_remote_code=True,
      post_init_hook=_boogu_post_init_hook,
    ),
    input_data=ExampleInputData(
      prompt="Turn the cat into a golden statue",
      height=height,
      width=width,
      num_inference_steps=50,
      image=image,
      extra_input_kwargs={
        "text_guidance_scale": 4.0,
      },
    ),
  )


@ExampleRegister.register("ernie_image", default="baidu/ERNIE-Image")
def ernie_image_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import ErnieImagePipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("baidu/ERNIE-Image"),
      pipeline_class=ErnieImagePipeline,
    ),
    input_data=ExampleInputData(
      prompt="一只黑白相间的中华田园犬",
      height=1024,
      width=1024,
      num_inference_steps=50,
      guidance_scale=4.0,
      extra_input_kwargs={
        "use_pe": False,
      },
    ),
  )


@ExampleRegister.register("ernie_image_turbo", default="baidu/ERNIE-Image-Turbo")
def ernie_image_turbo_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import ErnieImagePipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("baidu/ERNIE-Image-Turbo"),
      pipeline_class=ErnieImagePipeline,
    ),
    input_data=ExampleInputData(
      prompt="一只黑白相间的中华田园犬",
      height=1024,
      width=1024,
      num_inference_steps=8,
      guidance_scale=1.0,
      extra_input_kwargs={
        "use_pe": False,
      },
    ),
  )


@ExampleRegister.register("krea2", default="krea/Krea-2-Turbo")
def krea2_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import Krea2Pipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("krea/Krea-2-Turbo"),
      pipeline_class=Krea2Pipeline,
      extra_optimize_kwargs={
        "enable_separate_cfg": False,  # guidance_scale=0.0 → no CFG at runtime
      },
    ),
    input_data=ExampleInputData(
      prompt="A fox in the snow, detailed, vibrant colors",
      height=1024,
      width=1024,
      num_inference_steps=8,
      guidance_scale=0.0,
    ),
  )


@ExampleRegister.register("joy_image_edit", default="jdopensource/JoyAI-Image-Edit-Diffusers")
def joy_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import JoyImageEditPipeline

  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  elif _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR, "examples/data/joy_test.png")).convert("RGB")
  else:
    image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/joy_test.png").convert("RGB")

  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,
      model_name_or_path=_path("jdopensource/JoyAI-Image-Edit-Diffusers"),
      pipeline_class=JoyImageEditPipeline,
    ),
    input_data=ExampleInputData(
      prompt="Remove the construction structure from the top of the crane.",
      height=height,
      width=width,
      num_inference_steps=40,
      guidance_scale=4.0,
      image=image,
    ),
  )


@ExampleRegister.register("bria_fibo", default="briaai/FIBO")
def bria_fibo_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import BriaFiboPipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("briaai/FIBO"),
      pipeline_class=BriaFiboPipeline,
    ),
    input_data=ExampleInputData(
      prompt="A cat holding a sign that says hello world",
      height=1024,
      width=1024,
      num_inference_steps=28,
      guidance_scale=5,
    ),
  )


def _bria_fibo2_cache_config(
  args: argparse.Namespace,
  steps: int,
  enable_separate_cfg: bool,
) -> Optional[DBCacheConfig]:
  """DBCache config for Bria Fibo 2.0.

  `Fibo-2.0-dev` runs classifier-free guidance (guidance 5, two transformer forwards per step) and
  needs the separate-CFG bookkeeping; the distilled `Fibo-2.0-edit-turbo` samples without guidance
  in 4 steps and must turn it off, otherwise every step only counts as half a step.

  :param args: CLI arguments.
  :param steps: Denoising steps the pipeline will actually run.
  :param enable_separate_cfg: Whether the checkpoint runs cond/uncond in separate forwards.
  :returns: The DBCache config, or None when caching is disabled.
  """

  if not args.cache:
    return None
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=min(args.max_warmup_steps, max(1, steps // 4)),
    num_inference_steps=steps,
    enable_separate_cfg=enable_separate_cfg,
    residual_diff_threshold=args.residual_diff_threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=args.max_continuous_cached_steps,
  )


def _bria_fibo2_input_image(args: argparse.Namespace) -> PIL.Image.Image:
  """Load (and resize) the image both Fibo 2.0 edit examples run on.

  :param args: CLI arguments, `--image-path` overrides the bundled sample.
  :returns: The edit source image at the requested output resolution.
  """
  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  elif _force_use_local_data():
    image = load_image(os.path.join(CACHE_DIT_DIR,
                                    "examples/data/yarn-art-pikachu.png")).convert("RGB")
  else:
    image = load_image(
      "https://github.com/vipshop/cache-dit/raw/main/examples/data/yarn-art-pikachu.png").convert(
        "RGB")
  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  return image.resize((width, height))


@ExampleRegister.register("bria_fibo2", default="briaai/Fibo-2.0-dev")
def bria_fibo2_example(args: argparse.Namespace, **kwargs) -> Example:
  """Bria Fibo 2.0 image edit example (full 30-step, guidance 5 checkpoint).

  :param args: CLI arguments.
  :returns: The Fibo 2.0 edit example.
  """
  from diffusers import BriaFibo2Pipeline

  steps = 30 if args.num_inference_steps is None else args.num_inference_steps
  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,
      model_name_or_path=_path("briaai/Fibo-2.0-dev", args),
      pipeline_class=BriaFibo2Pipeline,
      extra_optimize_kwargs={
        "cache_config": _bria_fibo2_cache_config(args, steps, enable_separate_cfg=True),
      },
    ),
    input_data=ExampleInputData(
      prompt="Make the background a snowy forest",
      height=height,
      width=width,
      num_inference_steps=steps,
      guidance_scale=5.0,
      image=_bria_fibo2_input_image(args),
    ),
  )


@ExampleRegister.register("bria_fibo2_turbo", default="briaai/Fibo-2.0-edit-turbo")
def bria_fibo2_turbo_example(args: argparse.Namespace, **kwargs) -> Example:
  """Bria Fibo 2.0 distilled edit example (4 steps, guidance 1).

  :param args: CLI arguments.
  :returns: The Fibo 2.0 turbo edit example.
  """
  from diffusers import BriaFibo2Pipeline

  steps = 4 if args.num_inference_steps is None else args.num_inference_steps
  height = 1024 if args.height is None else args.height
  width = 1024 if args.width is None else args.width
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I,
      model_name_or_path=_path("briaai/Fibo-2.0-edit-turbo", args),
      pipeline_class=BriaFibo2Pipeline,
      extra_optimize_kwargs={
        "cache_config": _bria_fibo2_cache_config(args, steps, enable_separate_cfg=False),
      },
    ),
    input_data=ExampleInputData(
      prompt="Make the background a snowy forest",
      height=height,
      width=width,
      num_inference_steps=steps,
      guidance_scale=1.0,
      image=_bria_fibo2_input_image(args),
    ),
  )


def _echo_dmd_steps() -> int:
  """Return the number of denoising steps the released Echo DMD schedule runs.

  :returns: `len(sigmas) - 1`, or 8 when the schedule cannot be imported.
  """
  try:
    from diffusers.modular_pipelines.echo.denoise import DEFAULT_ECHO_SIGMAS

    return len(DEFAULT_ECHO_SIGMAS) - 1
  except ImportError:
    return 8


def _echo_cache_config(
  args: argparse.Namespace,
  steps: int,
) -> Optional[DBCacheConfig]:
  """DBCache config for Echo (JoyAI-Echo).

  Echo is guidance-free and runs one transformer forward per stochastic DMD step, so
  `enable_separate_cfg` stays False. The released schedule is only 8 steps, so the default warmup
  would leave nothing to reuse.

  :param args: CLI arguments.
  :param steps: Denoising steps the DMD schedule runs.
  :returns: The DBCache config, or None when caching is disabled.
  """

  if not args.cache:
    return None
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=min(args.max_warmup_steps, 2),
    num_inference_steps=steps,
    enable_separate_cfg=False,
    residual_diff_threshold=args.residual_diff_threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=args.max_continuous_cached_steps,
  )


_ECHO_PROMPT = (
  "A cinematic dialogue scene in a quiet cafe: warm afternoon light falls through the "
  "window, steam rises from a cup on the table, and the camera slowly pushes in.")


def _echo_post_init_hook(pipe: Any) -> None:
  """Report Echo's native container rate to the video writer.

  :param pipe: The loaded Echo modular pipeline.
  """
  # A modular pipeline exposes no `fps`, and `encode_video` would fall back to 8 fps, which makes the
  # saved clip play at a third of the rate it was generated for.
  pipe.fps = 25.0


@ExampleRegister.register("echo_t2va", default="jdopensource/JoyAI-Echo")
def echo_t2va_example(args: argparse.Namespace, **kwargs) -> Example:
  """Echo (JoyAI-Echo) text-to-video+audio example.

  Echo is a modular pipeline whose DMD schedule is fixed at 8 sigma steps, so
  `num_inference_steps` is not forwarded. Its text encoder lives in the gated
  `google/gemma-3-12b-it` repo instead of the checkpoint, hence the load override.

  :param args: CLI arguments.
  :returns: The Echo t2va example.
  """
  from diffusers import EchoModularPipeline

  gemma_dir = _path("google/gemma-3-12b-it")
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,
      model_name_or_path=_path("jdopensource/JoyAI-Echo", args),
      pipeline_class=EchoModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      # The reference parity recipe keeps `audio_vae` in float32.
      modular_component_dtypes={
        "default": torch.bfloat16,
        "audio_vae": torch.float32,
      },
      modular_load_overrides={
        # The checkpoint points its text encoder at the gated `google/gemma-3-12b-it`; both entries
        # have to be redirected together, otherwise the tokenizer and the embeddings disagree.
        "pretrained_model_name_or_path": {
          "text_encoder": gemma_dir,
          "tokenizer": gemma_dir,
        },
        # The spec leaves `subfolder=None`, which transformers cannot join onto a local path.
        "subfolder": {
          "text_encoder": "",
          "tokenizer": "",
        },
      },
      post_init_hook=_echo_post_init_hook,
      extra_optimize_kwargs={
        "cache_config": _echo_cache_config(args, _echo_dmd_steps()),
      },
    ),
    input_data=ExampleInputData(
      prompt=_ECHO_PROMPT,
      height=384,
      width=640,
      num_frames=121,
      use_num_inference_steps=False,
      extra_input_kwargs={
        "frame_rate": 25.0,
        "model_frame_rate": 24.0,
      },
    ),
  )


_MOTIF_VIDEO_HF_ID = "Motif-Technologies/Motif-Video-2B"


def _motif_video_i2v_image(model_name_or_path: str, args: argparse.Namespace):
  """Pick the i2v conditioning image: the caller's path, the local snapshot's
  `assets/i2v_sample.jpg`, or the Hub URL as the last resort (the local network
  cannot reach huggingface.co directly).
  """

  if args.image_path is not None:
    return load_image(args.image_path).convert("RGB")
  local_asset = os.path.join(model_name_or_path, "assets", "i2v_sample.jpg")
  if os.path.isfile(local_asset):
    return load_image(local_asset).convert("RGB")
  return load_image(
    f"https://huggingface.co/{_MOTIF_VIDEO_HF_ID}/resolve/main/assets/i2v_sample.jpg").convert(
      "RGB")


@ExampleRegister.register("motif_video_t2v", default=_MOTIF_VIDEO_HF_ID)
def motif_video_t2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """Motif-Video-2B text-to-video example.

  The pipeline guides via `AdaptiveProjectedGuidance` (two sequential transformer passes per
  step), so `enable_separate_cfg` is derived from `guider.num_conditions` by the adapter and
  the guidance scale is configured on the guider, not on `__call__`.
  """

  from diffusers import MotifVideoPipeline

  model_name_or_path = _path(_MOTIF_VIDEO_HF_ID, args=args)
  num_inference_steps = 50 if args.num_inference_steps is None else args.num_inference_steps

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,
      model_name_or_path=model_name_or_path,
      pipeline_class=MotifVideoPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("A cinematic drone shot gliding over a misty mountain lake at sunrise. "
              "Golden light pierces the fog, water ripples gently, distant peaks emerge "
              "slowly. Smooth camera motion, natural parallax, ultra-realistic, cinematic."),
      negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
      height=736,
      width=1280,
      num_frames=121,
      num_inference_steps=num_inference_steps,
    ),
  )


@ExampleRegister.register("motif_video_i2v", default=_MOTIF_VIDEO_HF_ID)
def motif_video_i2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """Motif-Video-2B image-to-video example.

  The i2v pipeline requires the `feature_extractor` (SigLIP) component and animates the
  provided first frame; the conditioning image is resized to the target resolution.
  """

  from diffusers import MotifVideoImage2VideoPipeline

  model_name_or_path = _path(_MOTIF_VIDEO_HF_ID, args=args)
  num_inference_steps = 50 if args.num_inference_steps is None else args.num_inference_steps

  height = 736 if args.height is None else args.height
  width = 1280 if args.width is None else args.width
  image = _motif_video_i2v_image(model_name_or_path, args)
  image = image.resize((width, height))

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.I2V,
      model_name_or_path=model_name_or_path,
      pipeline_class=MotifVideoImage2VideoPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=("The scene comes to life with gentle natural motion: soft wind, subtle light "
              "changes, smooth cinematic camera drift."),
      negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
      image=image,
      height=height,
      width=width,
      num_frames=121,
      num_inference_steps=num_inference_steps,
    ),
  )


@ExampleRegister.register("anyflow_t2v", default="nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers")
def anyflow_t2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """AnyFlow T2V example.

  AnyFlow is a flow-map distilled model based on the Wan2.1 3D DiT backbone. The released
  checkpoints fold CFG into the weights, so guidance_scale=1.0 is the recommended setting.
  num_inference_steps should be small (1–8) as the model supports any-step sampling.
  """
  from diffusers import AnyFlowPipeline

  num_frames = 33 if args.num_frames is None else args.num_frames
  num_inference_steps = 4 if args.num_inference_steps is None else args.num_inference_steps

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,
      model_name_or_path=_path(
        "nvidia/AnyFlow-Wan2.1-T2V-14B-Diffusers",
        args=args,
      ),
      pipeline_class=AnyFlowPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "enable_separate_cfg": False,  # CFG is batch-concatenated, not separate passes
      },
    ),
    input_data=ExampleInputData(
      prompt="A red panda eating bamboo in a forest, cinematic lighting",
      height=480,
      width=832,
      num_frames=num_frames,
      num_inference_steps=num_inference_steps,
      guidance_scale=1.0,
    ),
  )


# Demo assets for the conditioning workflows (`fl2va` needs a first and a last frame, `ref2va`
# needs reference media). They live in the checkout under `examples/data/minimax_h3`; the
# environment variable lets a caller point at a different copy, and a checkout that does not have
# them (an installed package) falls back to the copy on the repository's main branch.
_MINIMAX_H3_TEST_DATA_ENV = "MINIMAX_H3_TEST_DATA"
_MINIMAX_H3_ASSET_SUBDIR = os.path.join("examples", "data", "minimax_h3")
_MINIMAX_H3_ASSET_URL = "https://github.com/vipshop/cache-dit/raw/main/examples/data/minimax_h3"


def _minimax_h3_demo_asset(name: str) -> str:
  """Resolve a MiniMax-H3 demo asset to a path or URL.

  :param name: File name inside the demo-asset directory, e.g. ``first_frame.png``.
  :returns: Local path when the checkout (or `MINIMAX_H3_TEST_DATA`) has it, else its URL.
  """
  candidates = []
  env_dir = os.environ.get(_MINIMAX_H3_TEST_DATA_ENV)
  if env_dir:
    candidates.append(env_dir)
  # `src/cache_dit/_utils/examples.py` -> repository root, for an editable / in-tree install.
  candidates.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 _MINIMAX_H3_ASSET_SUBDIR))
  if CACHE_DIT_DIR is not None:
    candidates.append(os.path.join(CACHE_DIT_DIR, _MINIMAX_H3_ASSET_SUBDIR))
  for directory in candidates:
    candidate = os.path.normpath(os.path.join(directory, name))
    if os.path.exists(candidate):
      return candidate
  return f"{_MINIMAX_H3_ASSET_URL}/{name}"


def _minimax_h3_quantize_exclusions(args: argparse.Namespace) -> None:
  """Install the MiniMax-H3 default quantization exclusions on ``args``.

  :param args: CLI arguments, updated in place when the user did not pass the flag.
  """
  if not args.quantize or getattr(args, "quantize_exclude_layers", None):
    return
  args.quantize_exclude_layers = [
    # The checkpoint ships these in float32 (`_keep_in_fp32_modules`), which torchao's per-row
    # float8 path rejects, and they are a rounding-error guard rather than a memory cost.
    "proj_in",
    "audio_proj_in",
    "proj_out",
    "audio_proj_out",
    "time_embedder",
    "rope",
    # Every token of every step reads the same per-block AdaLN table, so a quantization error
    # there is applied coherently across the whole sampling trajectory instead of averaging out.
    # The output norm modulates the sequence the same way.
    "adaln_proj",
    "norm_out",
  ]


def _minimax_h3_drop_lm_head(pipe) -> None:
  """Drop the LM head of the Qwen3-VL conditioner.

  MiniMax-H3 conditions on the residual stream read after the 50th decoder layer, so the
  `lm_head` projection is never executed; dropping it frees about 1.5 GiB of the 63 GiB
  conditioner.

  :param pipe: MiniMax-H3 modular pipeline.
  """
  text_encoder = getattr(pipe, "text_encoder", None)
  if text_encoder is not None and getattr(text_encoder, "lm_head", None) is not None:
    text_encoder.lm_head = None


# The CLI default of `--residual-diff-threshold`; a value that differs from it is an explicit
# request and wins over the per-workflow preset below.
_MINIMAX_H3_DEFAULT_THRESHOLD = DEFAULT_RESIDUAL_DIFF_THRESHOLD

# Per-workflow starting point for the residual threshold. The conditioning workflows pin their
# conditioning rows every step, and those rows contribute ~nothing to the residual mean DBCache
# compares against, so the same threshold accepts more steps than it does for `t2va`. The accepted
# skips show up as a motion-phase lag: with the first and last frame anchored, mid-motion frames are
# the ones that drift, which SSIM penalises heavily. The cliff sits between 0.022 and 0.030 (see the
# PR description for the sweep), so `fl2va` starts where the accuracy bar still holds rather than
# where the speedup peaks.
_MINIMAX_H3_CACHE_THRESHOLDS = {
  "t2va": _MINIMAX_H3_DEFAULT_THRESHOLD,
  "fl2va": 0.022,
  # `ref2va` packs the most pinned rows (a whole reference prefix, up to twelve blocks), so it is
  # expected to need an equally strict start; it is measured with its own example.
  "ref2va": 0.02,
}


def _minimax_h3_cache_config(
  args: argparse.Namespace,
  workflow: str = "t2va",
) -> Optional[DBCacheConfig]:
  """DBCache config for the 50-step guidance-distilled MiniMax-H3 transformer.

  Only the block stack is cached, so `Fn_compute_blocks` stays small and one block at the tail is
  recomputed as a scaler. Every block rebuilds its AdaLN modulation from the current timestep, so
  consecutive steps move further apart than in a fixed-modulation image model: only one
  consecutive step is ever reused. The threshold comes from `--residual-diff-threshold` when the
  user set it, and from `_MINIMAX_H3_CACHE_THRESHOLDS[workflow]` otherwise.

  :param args: CLI arguments.
  :param workflow: One of `t2va`, `fl2va`, `ref2va`; selects the default threshold.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  steps = 50 if args.num_inference_steps is None else args.num_inference_steps
  if workflow not in _MINIMAX_H3_CACHE_THRESHOLDS:
    raise ValueError(f"No MiniMax-H3 cache preset for workflow '{workflow}'; expected one of "
                     f"{sorted(_MINIMAX_H3_CACHE_THRESHOLDS)}.")
  threshold = args.residual_diff_threshold
  if threshold == _MINIMAX_H3_DEFAULT_THRESHOLD:
    threshold = _MINIMAX_H3_CACHE_THRESHOLDS[workflow]
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=1,
    max_warmup_steps=min(args.max_warmup_steps, max(1, steps // 2)),
    num_inference_steps=steps,
    # guidance is distilled into the weights: one forward pass per step, no uncond branch
    enable_separate_cfg=False,
    residual_diff_threshold=threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=1,
  )


@ExampleRegister.register("minimax_h3", default="MiniMaxAI/MiniMax-H3")
def minimax_h3_example(args: argparse.Namespace, **kwargs) -> Example:
  """T2VA (joint text to video + audio) with MiniMax-H3.

  The repository holds two checkpoint partitions: `transformer/` for `t2va` / `fl2va` and the
  larger `transformer_ref/` for `ref2va`. `modular_workflow` prunes the blocks to `t2va`, so
  `load_components` pulls only the 62 GiB transformer and leaves the `ref2va` partition on disk.

  Neither the 62 GiB transformer nor the 63 GiB Qwen3-VL conditioner fits on a 72 GiB device
  together with the model's activations. `--adaln-host-residency` is the tightest fit: it keeps the
  transformer resident minus its 24 GiB of AdaLN tables and streams the conditioner, which peaks at
  about 49 GiB. `--module-layerwise-cpu-offload` streams both components instead, and
  `--parallel tp`/`ulysses` spreads the weights over several devices.

  :param args: CLI arguments.
  :returns: The MiniMax-H3 t2va example.
  """
  from diffusers import MiniMaxH3ModularPipeline

  _minimax_h3_quantize_exclusions(args)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,
      model_name_or_path=_path("MiniMaxAI/MiniMax-H3", args=args),
      pipeline_class=MiniMaxH3ModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      modular_workflow="t2va",
      # The two VAEs ship float32; only the conditioner and the transformer are bfloat16.
      modular_component_dtypes={
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "default": torch.float32,
      },
      post_init_hook=_minimax_h3_drop_lm_head,
      extra_optimize_kwargs={"cache_config": _minimax_h3_cache_config(args, "t2va")},
    ),
    input_data=ExampleInputData(
      prompt=(
        "A red fox trotting through a snowy pine forest at dawn, soft rim light, shallow depth "
        "of field, cinematic."),
      height=448,
      width=768,
      num_frames=124,
      num_inference_steps=50,
    ),
  )


@ExampleRegister.register("minimax_h3_fl2va", default="MiniMaxAI/MiniMax-H3")
def minimax_h3_fl2va_example(args: argparse.Namespace, **kwargs) -> Example:
  """FL2VA (first/last frame + text to video + audio) with MiniMax-H3.

  `modular_workflow="fl2va"` prunes the blocks so `load_components` pulls the `transformer/`
  partition only; the `transformer_ref/` partition that `ref2va` needs is never touched. Both
  conditioning frames become extra video rows of the packed sequence, so the cache, context and
  tensor parallel plans of the `t2va` example apply unchanged.

  The pipeline derives its canvas from the source aspect ratio and stretches the first frame /
  cover-crops the last one itself, which is why `skip_image_resize` is set: resizing here as well
  would resample the conditioning frames twice.

  :param args: CLI arguments.
  :returns: The MiniMax-H3 fl2va example.
  """
  from diffusers import MiniMaxH3ModularPipeline

  _minimax_h3_quantize_exclusions(args)

  def _load(name: str, override):
    # Only resolve a default when the caller did not supply that frame: an override replaces it
    # entirely, and a checkout-less install would otherwise download artwork it never uses.
    return None if override is not None else load_image(_minimax_h3_demo_asset(name)).convert("RGB")

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.FL2VA,
      model_name_or_path=_path("MiniMaxAI/MiniMax-H3", args=args),
      pipeline_class=MiniMaxH3ModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      modular_workflow="fl2va",
      # The two VAEs ship float32; only the conditioner and the transformer are bfloat16.
      modular_component_dtypes={
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "default": torch.float32,
      },
      post_init_hook=_minimax_h3_drop_lm_head,
      extra_optimize_kwargs={"cache_config": _minimax_h3_cache_config(args, "fl2va")},
    ),
    input_data=ExampleInputData(
      prompt=(
        "The astronaut crosses the lunar surface as the camera pushes in towards the crater rim, "
        "dust drifting in the low gravity, earthrise on the horizon, cinematic."),
      image=_load("first_frame.png", args.image_path),
      last_image=_load("last_frame.png", getattr(args, "last_image_path", None)),
      skip_image_resize=True,
      height=448,
      width=768,
      num_frames=124,
      num_inference_steps=50,
    ),
  )


@ExampleRegister.register("minimax_h3_ref2va", default="MiniMaxAI/MiniMax-H3")
def minimax_h3_ref2va_example(args: argparse.Namespace, **kwargs) -> Example:
  """Ref2Va (reference image / video / audio to video + audio) with MiniMax-H3.

  `ref2va` is the third checkpoint partition: this workflow loads `transformer_ref/` instead of
  `transformer/`, so it is never in memory at the same time as the `t2va` / `fl2va` weights.
  References are consumed in the order they are given and packed in front of the target sequence as
  read-only rows; they do not bind the generated geometry, which comes from `height` / `width`, and
  a reference video is cut to the requested duration rather than the other way round.

  :param args: CLI arguments.
  :returns: The MiniMax-H3 ref2va example.
  """
  from diffusers import MiniMaxH3ModularPipeline
  # The reference data classes are exported by the pipeline package, not by `diffusers` itself.
  from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
  )

  _minimax_h3_quantize_exclusions(args)

  input_data = ExampleInputData(
    prompt=(
      "The astronaut from the reference image steps out of the lander and follows the hiker's "
      "trail across the lunar surface, dust drifting in the low gravity, earthrise on the "
      "horizon, cinematic."),
    height=448,
    width=768,
    num_frames=124,
    num_inference_steps=50,
  )
  # `--reference-*-path` replaces the whole list, and constructing a reference decodes its media, so
  # the bundled pair is only built when the caller did not bring their own.
  if not any(
      getattr(args, attr, None)
      for attr in ("reference_image_path", "reference_video_path", "reference_audio_path")):
    input_data.set_references([
      MiniMaxH3ImageReference.from_file(_minimax_h3_demo_asset("first_frame.png")),
      MiniMaxH3VideoReference.from_file(_minimax_h3_demo_asset("reference_video.mp4")),
    ])

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.REF2VA,
      model_name_or_path=_path("MiniMaxAI/MiniMax-H3", args=args),
      pipeline_class=MiniMaxH3ModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      modular_workflow="ref2va",
      # The two VAEs ship float32; only the conditioner and the reference transformer are bfloat16.
      modular_component_dtypes={
        "transformer_ref": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "default": torch.float32,
      },
      post_init_hook=_minimax_h3_drop_lm_head,
      extra_optimize_kwargs={"cache_config": _minimax_h3_cache_config(args, "ref2va")},
    ),
    input_data=input_data,
  )


# The FastH3 students run 4 / 8 transformer forwards, so the cache warmup and the residual
# threshold are keyed on the forward count rather than on the sigma grid points.
_FASTH3_FORWARDS = {"fasth3": 4, "fasth3_v2": 8}
_FASTH3_CACHE_THRESHOLDS = {
  # A 4-step student crosses a much wider noise interval per step than the 50-step base model, so
  # its residuals start an order of magnitude higher; 0.10 is the sweep origin.
  "fasth3": 0.10,
  "fasth3_v2": 0.05,
}


def _fasth3_quantize_exclusions(args: argparse.Namespace) -> None:
  """Add the VSA compression gate to the MiniMax-H3 quantization exclusions.

  :param args: CLI arguments, updated in place.
  """
  _minimax_h3_quantize_exclusions(args)
  layers = getattr(args, "quantize_exclude_layers", None)
  if layers and "to_gate_compress" not in layers:
    args.quantize_exclude_layers = [*layers, "to_gate_compress"]


def _fasth3_cache_config(
  args: argparse.Namespace,
  name: str,
) -> Optional[DBCacheConfig]:
  """DBCache config for the few-step FastH3 students.

  `num_inference_steps` here is the number of **transformer forwards** (4 or 8), not the sigma grid
  points the CLI passes to the pipeline, so `max_warmup_steps` cannot reuse the MiniMax-H3 preset:
  with 4 forwards a warmup of 2 would leave a single skippable step.

  :param args: CLI arguments.
  :param name: `fasth3` or `fasth3_v2`, selects the forward count and the threshold.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  forwards = _FASTH3_FORWARDS[name]
  threshold = args.residual_diff_threshold
  if threshold == _MINIMAX_H3_DEFAULT_THRESHOLD:
    threshold = _FASTH3_CACHE_THRESHOLDS[name]
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=1,
    max_warmup_steps=min(args.max_warmup_steps, 1),
    num_inference_steps=forwards,
    # guidance is distilled into the weights: one forward pass per step, no uncond branch
    enable_separate_cfg=False,
    residual_diff_threshold=threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=1,
  )


def _fasth3_example(args: argparse.Namespace, name: str, default: str, forwards: int) -> Example:
  from ..models.pipelines.fast_h3 import FastH3ModularPipeline

  _fasth3_quantize_exclusions(args)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,
      model_name_or_path=_path(default, args=args),
      # The FastH3 pipeline owns the transformer (compression gates + VSA) and injects the DMD
      # sigma ladder; every other component stays a diffusers class.
      pipeline_class=FastH3ModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      modular_workflow="t2va",
      modular_component_dtypes={
        "transformer": torch.bfloat16,
        "text_encoder": torch.bfloat16,
        "default": torch.float32,
      },
      post_init_hook=_minimax_h3_drop_lm_head,
      extra_optimize_kwargs={"cache_config": _fasth3_cache_config(args, name)},
    ),
    input_data=ExampleInputData(
      prompt=("A glassblower shapes a molten vase in a workshop at dusk, sparks drifting, warm rim "
              "light, shallow depth of field, cinematic."),
      height=448,
      width=768,
      num_frames=124,
      # Sigma grid points: one more than the number of transformer forwards.
      num_inference_steps=forwards + 1,
    ),
  )


@ExampleRegister.register("fasth3",
                          default="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree")
def fasth3_example(args: argparse.Namespace, **kwargs) -> Example:
  """T2VA with the 4-step FastH3 student (VSA 0.9, 4 transformer forwards).

  The checkpoint ships a `fastvideo_inference.json` contract and 50 `to_gate_compress` tensors; the
  pipeline reads the first and the transformer subclass loads the second. `--num-inference-steps`
  keeps the diffusers semantics (sigma grid points), so the 4-step student is run with 5 and the
  cache config is built from the real forward count.

  :param args: CLI arguments.
  :returns: The FastH3 4-step example.
  """
  return _fasth3_example(args, "fasth3",
                         "FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree", 4)


@ExampleRegister.register("fasth3_v2", default="FastVideo/FastVideo-FastH3-8-Step-V2")
def fasth3_v2_example(args: argparse.Namespace, **kwargs) -> Example:
  """T2VA with the 8-step FastH3 V2 student (VSA 0.8, 8 transformer forwards).

  :param args: CLI arguments.
  :returns: The FastH3 V2 example.
  """
  return _fasth3_example(args, "fasth3_v2", "FastVideo/FastVideo-FastH3-8-Step-V2", 8)


@ExampleRegister.register("anyflow_far", default="nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers")
def anyflow_far_example(args: argparse.Namespace, **kwargs) -> Example:
  """AnyFlow-FAR T2V example.

  AnyFlow-FAR is the causal (FAR-based) sibling of AnyFlow: a chunk-wise
  autoregressive flow-map distilled model on the Wan2.1 1.3B backbone. Each
  chunk is denoised with `num_inference_steps` flow-map steps while attending
  only to past chunks via block-sparse causal attention; a cross-chunk KV cache
  is reused.

  Cache and CP are NOT supported (the chunk-wise AR loop is incompatible with
  cache-dit's single-trajectory step tracking and with sequence-split CP). TP
  is supported via the AnyFlow planner with FAR-specific KV-cache head-count
  adaptation. The released checkpoint fuses CFG into the weights, so
  guidance_scale=1.0 is the recommended setting.
  """
  from diffusers import AnyFlowFARPipeline

  num_frames = 81 if args.num_frames is None else args.num_frames
  num_inference_steps = 4 if args.num_inference_steps is None else args.num_inference_steps

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,
      model_name_or_path=_path(
        "nvidia/AnyFlow-FAR-Wan2.1-1.3B-Diffusers",
        args=args,
      ),
      pipeline_class=AnyFlowFARPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
      extra_optimize_kwargs={
        "enable_separate_cfg": False,  # CFG is batch-concatenated, not separate passes
      },
    ),
    input_data=ExampleInputData(
      prompt="A red panda eating bamboo in a forest, cinematic lighting",
      height=480,
      width=832,
      num_frames=num_frames,
      num_inference_steps=num_inference_steps,
      guidance_scale=1.0,
    ),
  )


@ExampleRegister.register("prx_pixel", default="Photoroom/prxpixel-t2i")
def prx_pixel_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import PRXPixelPipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("Photoroom/prxpixel-t2i", args=args),
      pipeline_class=PRXPixelPipeline,
      extra_optimize_kwargs={
        "enable_separate_cfg": False,  # CFG is batch-concatenated, single forward
      },
    ),
    input_data=ExampleInputData(
      prompt="A front-facing portrait of a lion in the golden savanna at sunset.",
      height=1024,
      width=1024,
      num_inference_steps=28,
      guidance_scale=5.0,
    ),
  )


def _anima_cache_config(
  args: argparse.Namespace,
  steps: int,
) -> Optional[DBCacheConfig]:
  """DBCache config for the Anima (Cosmos Predict2 backbone) t2i pipeline.

  Anima samples with classifier-free guidance (the modular guider component holds
  `guidance_scale=4.0`, i.e. a conditional and an unconditional forward per step), so
  `enable_separate_cfg` stays True. The uniform 28-block stack recomputes its first block
  every step (`Fn_compute_blocks=1`) to keep the cached residuals anchored. The residual
  threshold is the generic CLI default (0.12), tuned against this example's 30-step
  schedule; the model card's 50-step default moves less per step and only caches more.

  :param args: CLI arguments.
  :param steps: Denoising steps the pipeline will actually run.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  # A model-provided cache config normally overrides the CLI knobs, so `--ws` would be
  # silently ignored. Honour an explicit value and fall back to the tuned default otherwise
  # (8 is the `--ws` CLI default).
  warmup = args.max_warmup_steps if args.max_warmup_steps != 8 else min(4, max(1, steps // 4))
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=warmup,
    num_inference_steps=steps,
    enable_separate_cfg=True,
    residual_diff_threshold=args.residual_diff_threshold,
  )


_ANIMA_PROMPT = ("A front-facing portrait of a lion in the golden savanna at sunset, detailed fur, "
                 "cinematic lighting, photorealistic")
_ANIMA_NEGATIVE_PROMPT = "blurry, low quality, watermark, text, deformed"


@ExampleRegister.register("anima", default="circlestone-labs/Anima-Base-v1.0-Diffusers")
def anima_example(args: argparse.Namespace, **kwargs) -> Example:
  from diffusers import AnimaModularPipeline

  # 30 steps instead of the 50-step default keeps the benchmark matrix comparable across
  # configs; height/width must be multiples of 16 (vae_scale_factor * patch).
  steps = 30 if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,  # Text to Image
      model_name_or_path=_path("circlestone-labs/Anima-Base-v1.0-Diffusers", args),
      pipeline_class=AnimaModularPipeline,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      # The repo ships `modular_model_index.json`, so `from_pretrained` resolves the blocks
      # and no `modular_blocks_factory` is needed.
      # Guidance lives in the modular guider component (guidance_scale=4.0 by default):
      # `ModularPipeline.__call__` does not route a `guidance_scale` input into it.
      extra_optimize_kwargs={
        "cache_config": _anima_cache_config(args, steps),
      },
    ),
    input_data=ExampleInputData(
      prompt=_ANIMA_PROMPT,
      negative_prompt=_ANIMA_NEGATIVE_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
    ),
  )


# Ideogram 4 is trained exclusively on structured JSON captions. Plain text is not a valid fallback:
# it frequently produces the model's safety-filter placeholder instead of an image.
_IDEOGRAM4_PROMPT = json.dumps(
  {
    "high_level_description":
    "A studio photograph of an orange cat holding a sign that says hello world.",
    "style_description": {
      "aesthetics": "warm, playful, polished",
      "lighting": "soft diffused studio lighting",
      "photo": "eye-level, shallow depth of field, sharp focus",
      "medium": "photograph",
    },
    "compositional_deconstruction": {
      "background":
      "A seamless warm gray studio backdrop with a subtle floor shadow.",
      "elements": [
        {
          "type":
          "obj",
          "bbox": [120, 100, 920, 900],
          "desc":
          "An alert orange tabby cat sitting upright and holding a white rectangular sign "
          "with both front paws.",
        },
        {
          "type": "text",
          "bbox": [500, 300, 680, 700],
          "text": "hello world",
          "desc": "Bold black sans-serif lowercase text centered on the white sign.",
        },
      ],
    },
  },
  separators=(",", ":"),
  ensure_ascii=False,
)
# The model's recommended setting, and the step count the pipeline's default guidance schedule is
# written for.
_IDEOGRAM4_STEPS = 48


def _ideogram4_guidance_schedule(steps: int) -> List[float]:
  """Rebuild Ideogram4's recommended per-step guidance schedule for `steps` steps.

  `Ideogram4Pipeline.__call__` defaults to `(7.0,) * 45 + (3.0,) * 3`, i.e. 48 entries, and
  `check_inputs` rejects a `guidance_schedule` whose length differs from `num_inference_steps`. A
  schedule baked into the input data would therefore break every step-count override, including the
  warmup pass, so it is rebuilt per call from the steps actually forwarded.

  :param steps: Denoising steps the pipeline will actually run.
  :returns: 7.0 for the main steps and a 3.0 "polish" tail of at most 3 steps.
  """
  return [7.0] * max(steps - 3, 0) + [3.0] * min(3, steps)


class _Ideogram4Example(Example):
  """Ideogram4 example whose guidance schedule is derived from the step count.

  The schedule has to match `num_inference_steps` exactly, and the run overrides that count in two
  places a static input dict cannot see: `--num-inference-steps` and `--warmup-num-inference-steps`.
  Deriving it in `call_pipe` keeps both consistent without touching the generic None filtering of
  `ExampleInputData.data()`.
  """

  def call_pipe(self, pipe, kwargs: Dict[str, Any]):
    kwargs = dict(kwargs)
    steps = kwargs.get("num_inference_steps", self.input_data.num_inference_steps)
    if "guidance_schedule" not in kwargs:
      if kwargs.get("guidance_scale") is None:
        kwargs["guidance_schedule"] = _ideogram4_guidance_schedule(steps)
      else:
        # Bypass input-data None filtering only here: the pipeline requires exactly one mode.
        kwargs["guidance_schedule"] = None
    return super().call_pipe(pipe, kwargs)


@ExampleRegister.register("ideogram4", default="ideogram-ai/ideogram-4-fp8")
def ideogram4_example(args: argparse.Namespace, **kwargs) -> Example:
  """Ideogram 4 text-to-image example.

  `Ideogram4Pipeline` applies asymmetric classifier-free guidance: the conditional `transformer` and
  the separate `unconditional_transformer` take one forward pass each per step, and the two
  velocities are blended outside of both, so `enable_separate_cfg` is off.

  The optional `prompt_enhancer_head` is not loaded and `prompt_upsampling` stays off: the upsampler
  rewrites the prompt into a structured JSON caption, which changes the input distribution between
  runs and dominates the measured latency.

  The default checkpoint is the official FP8 release, which only loads through the `Ideogram4`
  loader: it patches `Ideogram4Pipeline.from_pretrained` to rebuild the three FP8 components as
  bfloat16 on the way in, and a bfloat16 checkpoint passes through the same patch unchanged.

  :param args: CLI arguments.
  :returns: The Ideogram 4 t2i example.
  """
  from diffusers import Ideogram4Pipeline
  from ..loaders import Ideogram4Loader

  # Applied inside the factory, not at import time: this module is imported through `_safe_import`,
  # and a failing module-level import would degrade every other example to the import-error stub.
  Ideogram4Loader().apply(Ideogram4Pipeline)

  steps = _IDEOGRAM4_STEPS if args.num_inference_steps is None else args.num_inference_steps
  return _Ideogram4Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("ideogram-ai/ideogram-4-fp8", args=args),
      pipeline_class=Ideogram4Pipeline,
      skip_components=["prompt_enhancer_head"],
      extra_optimize_kwargs={
        "enable_separate_cfg": False,
      },
    ),
    input_data=ExampleInputData(
      prompt=_IDEOGRAM4_PROMPT,
      # Both must be multiples of `vae_scale_factor * patch_size` (16).
      height=1024,
      width=1024,
      num_inference_steps=steps,
      extra_input_kwargs={
        "prompt_upsampling": False,
      },
    ),
  )


_MOSS_IMAGE_PROMPT = "A wide view of a refrigerator beside a kitchen counter with a fruit bowl."


def _moss_image_example(args: argparse.Namespace, default: str) -> Example:
  """Build a MOSS-Image example; SFT and Base differ only by checkpoint.

  Caching assumes one `transformer.forward()` per CFG branch, so `guidance_scale` must stay above 1
  (`do_cfg=True`) and `num_images_per_prompt` at 1; SLG is not part of the cached path.

  :param args: Parsed CLI arguments.
  :param default: Default checkpoint id or path of the variant.
  :returns: The configured example.
  """
  from ..models.pipelines.moss_image import MOSSImagePipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path(default, args=args),
      pipeline_class=MOSSImagePipeline,
    ),
    input_data=ExampleInputData(
      prompt=_MOSS_IMAGE_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=50,
      guidance_scale=3.0,
      extra_input_kwargs={
        # Default CFG recipe of the release: skip block 22 in the negative branch.
        "uncond_skip_layers": [22],
      },
    ),
  )


@ExampleRegister.register("moss_image_sft", default="OpenMOSS-Team/MOSS-Image-SFT")
def moss_image_sft_example(args: argparse.Namespace, **kwargs) -> Example:
  return _moss_image_example(args, "OpenMOSS-Team/MOSS-Image-SFT")


@ExampleRegister.register("moss_image_base", default="OpenMOSS-Team/MOSS-Image-Base")
def moss_image_base_example(args: argparse.Namespace, **kwargs) -> Example:
  return _moss_image_example(args, "OpenMOSS-Team/MOSS-Image-Base")


def _ming_image_example(args: argparse.Namespace, layered: bool) -> Example:
  from ..models.pipelines.ming_image import MingImagePipeline

  default = ("inclusionAI/Ming-Image-0.1-Design-Layer"
             if layered else "inclusionAI/Ming-Image-0.1-Design")
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.IE2I if layered else ExampleType.T2I,
      model_name_or_path=_path(default, args=args),
      pipeline_class=MingImagePipeline,
      preserve_image_list=layered,
      extra_optimize_kwargs={"enable_separate_cfg": False},
    ),
    input_data=ExampleInputData(
      prompt=(
        "Decompose this design into 4 editable RGBA layers." if layered else
        "A minimal transparent sticker of a red paper airplane, centered on a blank background."),
      height=1024,
      width=1024,
      num_inference_steps=12,
      guidance_scale=2.0 if layered else 1.0,
      keep_alpha=True,
      skip_image_resize=True,
      extra_input_kwargs={"num_layers": 4} if layered else {},
    ),
  )


@ExampleRegister.register("ming_image_design", default="inclusionAI/Ming-Image-0.1-Design")
def ming_image_design_example(args: argparse.Namespace, **kwargs) -> Example:
  return _ming_image_example(args, layered=False)


@ExampleRegister.register("ming_image_design_layer",
                          default="inclusionAI/Ming-Image-0.1-Design-Layer")
def ming_image_design_layer_example(args: argparse.Namespace, **kwargs) -> Example:
  return _ming_image_example(args, layered=True)


_HIDREAM_O1_IMAGE_PROMPT = ("A vintage travel poster of a mountain lake at sunrise, layered "
                            "silhouettes of pine trees in the foreground, and the words "
                            "\"SUNRISE LAKE\" printed in bold serif letters at the bottom")


@ExampleRegister.register("hidream_o1_image", default="HiDream-ai/HiDream-O1-Image")
def hidream_o1_image_example(args: argparse.Namespace, **kwargs) -> Example:
  """HiDream-O1-Image text-to-image example.

  Defaults follow the release recipe: 2048x2048, 50 steps, CFG 5.0 with the empty-ish `" "`
  unconditional prompt. The pipeline snaps a requested size to the closest trained aspect ratio;
  pass `--extra-input-kwargs "{'snap_resolution': False}"` to render an arbitrary multiple of 32.

  The model is pixel-space and embeds its own token ids, so there is no VAE and no text encoder to
  shard: `--parallel-vae` / `--parallel-text` cannot work and are rejected here instead of being
  silently ignored.

  :param args: Parsed CLI arguments.
  :param kwargs: Unused extra keyword arguments.
  :returns: The configured example.
  """
  from ..models.pipelines.hidream_o1_image import HiDreamO1ImagePipeline

  if getattr(args, "parallel_text_encoder", False):
    raise NotImplementedError("HiDream-O1-Image embeds its own token ids: it has no text encoder, "
                              "so `--parallel-text` is not applicable.")
  if getattr(args, "parallel_vae", False):
    raise NotImplementedError("HiDream-O1-Image is pixel-space: it has no VAE, so `--parallel-vae` "
                              "is not applicable.")
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("HiDream-ai/HiDream-O1-Image", args=args),
      pipeline_class=HiDreamO1ImagePipeline,
    ),
    input_data=ExampleInputData(
      prompt=_HIDREAM_O1_IMAGE_PROMPT,
      height=2048,
      width=2048,
      num_inference_steps=50,
      guidance_scale=5.0,
    ),
  )


_NUCLEUS_IMAGE_PROMPT = ("A studio photograph of an orange cat holding a wooden sign that says "
                         "hello world, soft window light, shallow depth of field")


def _nucleus_image_cache_config(
  args: argparse.Namespace,
  steps: int,
) -> Optional[DBCacheConfig]:
  """DBCache config for the NucleusMoE-Image t2i pipeline.

  The pipeline samples with classifier-free guidance (`guidance_scale=4.0`), i.e. a conditional and
  an unconditional forward per step, so `enable_separate_cfg` stays True. The stack mixes three
  dense blocks with 29 MoE blocks but every block shares one I/O contract, so a single
  `Fn_compute_blocks=1` prologue anchors the cached residuals well enough.

  :param args: CLI arguments.
  :param steps: Denoising steps the pipeline will actually run.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  warmup = args.max_warmup_steps if args.max_warmup_steps != 8 else min(4, max(1, steps // 4))
  return DBCacheConfig(
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
    max_warmup_steps=warmup,
    num_inference_steps=steps,
    enable_separate_cfg=True,
    residual_diff_threshold=args.residual_diff_threshold,
  )


@ExampleRegister.register("nucleus_image", default="NucleusAI/Nucleus-Image")
def nucleus_image_example(args: argparse.Namespace, **kwargs) -> Example:
  """NucleusMoE-Image text-to-image example.

  28 steps instead of the model card's 50 keeps the benchmark matrix affordable; height/width must
  be multiples of `vae_scale_factor * patch_size` (16) and 1024² gives 4096 image tokens, which
  divides by the CP sizes used in the tests.

  :param args: CLI arguments.
  :returns: The NucleusMoE-Image t2i example.
  """
  from diffusers import NucleusMoEImagePipeline

  steps = 28 if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("NucleusAI/Nucleus-Image", args),
      pipeline_class=NucleusMoEImagePipeline,
      extra_optimize_kwargs={
        "cache_config": _nucleus_image_cache_config(args, steps),
      },
    ),
    input_data=ExampleInputData(
      prompt=_NUCLEUS_IMAGE_PROMPT,
      height=1024,
      width=1024,
      num_inference_steps=steps,
      guidance_scale=4.0,
    ),
  )


_KANDINSKY6_PROMPT = ("A cinematic close-up of a glass teapot on a wooden table, steam curling "
                      "upward in warm afternoon light, the quiet hiss of the kettle in the "
                      "background")
# The distilled checkpoint runs `PiflowScheduler`, which requires guidance_scale=1.0 and therefore
# exactly one transformer forward per step.
_KANDINSKY6_DEFAULT_STEPS = 16


def _kandinsky6_cache_config(
  args: argparse.Namespace,
  steps: int,
) -> Optional[DBCacheConfig]:
  """DBCache config for the distilled Kandinsky 6.0 TI2VA pipeline.

  The distilled checkpoint is guidance-free (`guidance_scale=1.0`), so `enable_separate_cfg` must be
  False: with the pipeline default of 5.0 left in place cache-dit pairs two consecutive forwards
  into one step and almost nothing is cached. 16 steps leaves little room, hence the short warmup.

  :param args: CLI arguments.
  :param steps: Denoising steps the pipeline will actually run.
  :returns: The DBCache config, or None when caching is disabled.
  """
  if not args.cache:
    return None
  return DBCacheConfig(
    Fn_compute_blocks=1,
    # The CP planner gathers the video stream on the last block's output, so a CP run needs at
    # least one tail block; keep it here as well so the two paths share one config.
    Bn_compute_blocks=1,
    max_warmup_steps=min(args.max_warmup_steps, 4),
    num_inference_steps=steps,
    enable_separate_cfg=False,
    residual_diff_threshold=args.residual_diff_threshold,
    max_cached_steps=args.max_cached_steps,
    max_continuous_cached_steps=args.max_continuous_cached_steps,
  )


def _kandinsky6_post_init_hook(pipe: Any) -> None:
  """Report Kandinsky 6.0's native frame rate to the video writer.

  :param pipe: The loaded pipeline.
  """
  # `__call__` defaults to `frame_rate=24.0`, and the writer falls back to 8 fps without this.
  pipe.fps = 24.0


@ExampleRegister.register(
  "kandinsky6_ti2va",
  default="kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers",
)
def kandinsky6_ti2va_example(args: argparse.Namespace, **kwargs) -> Example:
  """Kandinsky 6.0 text-to-video+audio example (distilled 5s checkpoint).

  The released distilled recipe is 16 steps at `guidance_scale=1.0`. 512×768×121 frames is the
  pipeline default canvas; the verification runs in this repo shrink it further, so pass
  `--height/--width/--num-frames` when the whole matrix has to stay under the time budget.

  :param args: CLI arguments.
  :returns: The Kandinsky 6.0 t2va example.
  """
  from diffusers import Kandinsky6TI2VAPipeline

  steps = _KANDINSKY6_CACHE_STEPS if args.num_inference_steps is None else args.num_inference_steps
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2VA,
      model_name_or_path=_path("kandinskylab/Kandinsky-6.0-Pro-distill-5s-Diffusers", args),
      pipeline_class=Kandinsky6TI2VAPipeline,
      post_init_hook=_kandinsky6_post_init_hook,
      extra_optimize_kwargs={
        "cache_config": _kandinsky6_cache_config(args, steps),
      },
    ),
    input_data=ExampleInputData(
      prompt=_KANDINSKY6_PROMPT,
      height=512,
      width=768,
      num_frames=121,
      num_inference_steps=steps,
      guidance_scale=1.0,
      extra_input_kwargs={
        "frame_rate": 24.0,
      },
    ),
  )


@ExampleRegister.register("longcat_audio_dit", default="ruixiangma/LongCat-AudioDiT-1B-Diffusers")
def longcat_audio_dit_example(args: argparse.Namespace, **kwargs) -> Example:
  """LongCat-AudioDiT text-to-speech example (1B diffusers conversion).

  The latent length follows the requested duration (24000 / 2048 = 11.72 frames per second), so
  `audio_duration_s` goes through `extra_input_kwargs`: there is no dedicated input field for it.
  10 s is the reference load used for the acceptance numbers (117 latent frames, well under the
  30 s cap the pipeline hardcodes). Note that 117 is odd: context parallelism over two ranks needs
  `--ulysses-anything`, or a duration whose frame count is even (5 s = 58, 6 s = 70).

  :param args: CLI arguments.
  :returns: The LongCat-AudioDiT text-to-speech example.
  """
  from diffusers import LongCatAudioDiTPipeline

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2A,  # Text to Audio
      model_name_or_path=_path("ruixiangma/LongCat-AudioDiT-1B-Diffusers", args),
      pipeline_class=LongCatAudioDiTPipeline,
    ),
    input_data=ExampleInputData(
      prompt=("The sun is setting over the calm ocean, with gentle waves breaking on the shore."),
      num_inference_steps=16,
      guidance_scale=4.0,
      extra_input_kwargs={
        "audio_duration_s": 10.0,
      },
    ),
  )


_STABLE_AUDIO_3_CONVERT_HINT = (
  "The official `stabilityai/stable-audio-3-*` repositories ship the stable-audio-tools format "
  "(`model_config.json` + a single `model.safetensors`), not a diffusers layout. Convert it once "
  "with the script shipped by diffusers and point the CLI at the output directory:\n"
  "  python scripts/convert_stable_audio_3_to_diffusers.py \\\n"
  "    --checkpoint_path <sa3-native>/model.safetensors \\\n"
  "    --model_config_path <sa3-native>/model_config.json \\\n"
  "    --output_dir <sa3-diffusers> \\\n"
  "    --text_encoder_repo <sa3-native>/t5gemma-b-b-ul2 --dtype bfloat16")


def _assert_stable_audio_3_layout(model_name_or_path: str) -> None:
  """Fail early when the resolved model path is the native SA3 checkpoint instead of a conversion.

  :param model_name_or_path: Resolved model id or local directory.
  :raises ValueError: When the directory is a native SA3 checkpoint (no `model_index.json`).
  """

  if not os.path.isdir(model_name_or_path):
    return  # a Hub id: `from_pretrained` reports its own error
  if os.path.exists(os.path.join(model_name_or_path, "model_index.json")):
    return
  if os.path.exists(os.path.join(model_name_or_path, "model_config.json")):
    raise ValueError(f"{model_name_or_path} is a native Stable Audio 3 checkpoint. "
                     f"{_STABLE_AUDIO_3_CONVERT_HINT}")


@ExampleRegister.register("stable_audio_3", default="stabilityai/stable-audio-3-medium")
def stable_audio_3_example(args: argparse.Namespace, **kwargs) -> Example:
  """Stable Audio 3 text-to-audio example (distilled `medium` checkpoint).

  The latent length follows the requested duration (`ceil(duration * 44100 / 4096)` frames), so
  `duration` goes through `extra_input_kwargs`: there is no dedicated input field for it. 10 s is
  the reference load used for the acceptance numbers (108 latent frames, i.e. `64 + 108 = 172`
  transformer tokens after the 64 memory tokens, divisible by 2 and 4 — context parallelism needs that
  divisibility, or `--ulysses-anything`). Unlike Stable Audio 1/2 the distilled checkpoint default is
  `guidance_scale=1.0` — CFG is baked into the weights, so every step runs a single velocity forward
  and 8 steps is the shipped schedule. The non-distilled `-base` checkpoint needs ~100 steps and
  `guidance_scale > 1`; point `STABLE_AUDIO_3_DIR` at it and override `--steps` / `--guidance-scale`.

  :param args: CLI arguments.
  :returns: The Stable Audio 3 text-to-audio example.
  """

  from diffusers import StableAudio3Pipeline

  if getattr(args, "parallel_vae", False):
    raise NotImplementedError(
      "Stable Audio 3 has an audio VAE (`AutoencoderSAME`, a 1-D convolutional decoder) and no "
      "registered VAE parallel planner; its decode is only ~8% of the run, so `--parallel-vae` "
      "is not applicable.")

  model_name_or_path = _path("stabilityai/stable-audio-3-medium", args)
  _assert_stable_audio_3_layout(model_name_or_path)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2A,  # Text to Audio
      model_name_or_path=model_name_or_path,
      pipeline_class=StableAudio3Pipeline,
    ),
    input_data=ExampleInputData(
      prompt=("A gentle piano melody with soft strings in a concert hall, "
              "warm and intimate, with a slow build."),
      num_inference_steps=8,
      guidance_scale=1.0,
      extra_input_kwargs={
        "duration": 10.0,
      },
    ),
  )


_SANA_WM_PROMPT = ("A young girl holding a white sign stands on a busy city street at dusk, "
                   "pedestrians walking behind her.")


def _sana_wm_intrinsics(image: PIL.Image.Image) -> List[float]:
  """Estimate `[fx, fy, cx, cy]` for a first frame when the caller has no calibration.

  The pipeline treats them as pixels of the *source* image and rescales them onto its own resize +
  center-crop, so a plain pinhole estimate with a ~45° vertical FOV keeps the camera conditioning
  consistent with the frame it actually reads.

  :param image: First-frame image.
  :returns:`[fx, fy, cx, cy]` in source-image pixels.
  """
  width, height = image.size
  focal = 1.2 * max(width, height)
  return [float(focal), float(focal), width / 2.0, height / 2.0]


class _SanaWMExample(Example):
  """SANA-WM example that normalizes the pipeline's un-batched video output.

  `SanaWMPipeline` returns `frames` already un-batched: a `list[PIL.Image]` of length `T` for
  `output_type="pil"` (an `(T, H, W, 3)` array for `"np"`), while `Example.run` reads the batched
  `output.frames[0]` layout that every other video pipeline uses. Wrapping the frame list here
  keeps `run`/`save` unchanged.
  """

  def call_pipe(self, pipe, kwargs: Dict[str, Any]):
    from types import SimpleNamespace

    output = super().call_pipe(pipe, kwargs)
    return SimpleNamespace(frames=[output.frames], c2w=output.c2w)


@ExampleRegister.register(
  "sana_wm",
  default="Efficient-Large-Model/SANA-WM_bidirectional-diffusers",
)
def sana_wm_example(args: argparse.Namespace, **kwargs) -> Example:
  """SANA-WM camera-controlled image-to-video example.

  Stage 1 only: the optional LTX-2 refiner is a separate pipeline
  (`Efficient-Large-Model/SANA-WM_bidirectional-diffusers-refiner`) and is not part of this
  example.

  The released recipe is 704×1280 at 161 frames / 60 steps, which is far too heavy for a
  functional check, so the defaults keep the official canvas but shorten the clip to 25 frames
  (`8k + 1` is a hard LTX-2 VAE constraint; the pipeline snaps silently, so a wrong count would
  only show up as a different length) and 8 steps. `--height/--width/--num-frames` override both.

  `action` and `intrinsics` are passed through `extra_input_kwargs` because the CLI has no flags
  for camera trajectories: `action` is the WASD/IJKL DSL (`w-25` = walk forward for 25 frames) and
  `intrinsics` is `[fx, fy, cx, cy]` of the input image. `skip_image_resize` keeps the first frame
  at its native geometry — the pipeline resizes and center-crops it itself and rescales the
  intrinsics to match, so resizing here would resample twice and desync the camera conditioning.

  :param args: CLI arguments.
  :returns: The SANA-WM image-to-video example.
  """
  from diffusers import SanaWMPipeline

  height = 704 if args.height is None else args.height
  width = 1280 if args.width is None else args.width
  if args.image_path is not None:
    image = load_image(args.image_path).convert("RGB")
  else:
    image = load_image(
      "https://huggingface.co/datasets/a-r-r-o-w/tiny-meme-dataset-captioned/resolve/main/"
      "images/8.png").convert("RGB")
  # The camera trajectory, not `num_frames`, decides the clip length: `prepare_camera_trajectory`
  # trims `num_frames` to the number of poses an action string produces. Deriving the action from
  # the requested length keeps the two in sync instead of silently shortening the clip.
  num_frames = 25 if args.num_frames is None else args.num_frames

  return _SanaWMExample(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.I2V,
      model_name_or_path=_path("Efficient-Large-Model/SANA-WM_bidirectional-diffusers", args),
      pipeline_class=SanaWMPipeline,
    ),
    input_data=ExampleInputData(
      prompt=_SANA_WM_PROMPT,
      image=image,
      height=height,
      width=width,
      num_frames=num_frames,
      num_inference_steps=8,
      guidance_scale=5.0,
      skip_image_resize=True,
      extra_input_kwargs={
        "action": f"w-{num_frames}",
        "intrinsics": _sana_wm_intrinsics(image),
        "output_type": "pil",
      },
    ),
  )


_WAN_ANIMATE_2_PROMPT = ("人物外观描述：一只银灰色虎斑纹的小猫，拥有圆润的脸庞、竖立的耳朵和巨大的圆形眼睛。"
                         "它身穿一套深蓝色的制服套装，包括一件带有金色纽扣的西装外套和一条百褶裙。"
                         "外套里面搭配着白色衬衫，领口处系着一个红色的蝴蝶结，袖口露出白色的衬衫边缘。"
                         "背景描述：背景为纯白色，光线均匀明亮，无其他杂物或装饰。")

# 4k+1 keeps the per-segment token count small enough for the verification matrix; the model default
# is 81 (a full Wan VAE period). The driving video is resampled to the model fps and zigzag-padded up
# to a whole segment by the pipeline, so any length is accepted.
_WAN_ANIMATE_2_SEGMENT_FRAMES = 25
_WAN_ANIMATE_2_DRIVING_FRAMES = 25


def _wan_animate_2_asset_dir() -> Optional[str]:
  """Return the directory holding the official demo assets, if registered.

  :returns: `$WAN_ANIMATE_2_DEMO_DIR` when set and present, else `None`.
  """

  demo_dir = os.environ.get("WAN_ANIMATE_2_DEMO_DIR", None)
  if demo_dir is not None and os.path.isdir(demo_dir):
    return demo_dir
  return None


def _wan_animate_2_reference_image(args: argparse.Namespace,
                                   model_name_or_path: str) -> PIL.Image.Image:
  """Resolve the reference character image.

  :param args: CLI arguments, honoring `--image-path`.
  :param model_name_or_path: Model directory, used as a fallback asset location.
  :returns: The reference image in RGB.
  """

  if args.image_path is not None:
    return load_image(args.image_path).convert("RGB")
  asset_dir = _wan_animate_2_asset_dir()
  if asset_dir is not None and os.path.isfile(os.path.join(asset_dir, "reference.png")):
    return load_image(os.path.join(asset_dir, "reference.png")).convert("RGB")
  local_asset = os.path.join(model_name_or_path, "assets", "reference.png")
  if os.path.isfile(local_asset):
    return load_image(local_asset).convert("RGB")
  logger.warning("No Wan-Animate-2 reference image found: falling back to the bundled cat image. "
                 "Pass `--image-path` or set `WAN_ANIMATE_2_DEMO_DIR` for a meaningful sample.")
  if CACHE_DIT_DIR is not None and os.path.isfile(
      os.path.join(CACHE_DIT_DIR, "examples/data/cat.png")):
    return load_image(os.path.join(CACHE_DIT_DIR, "examples/data/cat.png")).convert("RGB")
  return load_image("https://github.com/vipshop/cache-dit/raw/main/examples/data/cat.png").convert(
    "RGB")


def _wan_animate_2_driving_video(
  args: argparse.Namespace,
  reference_image: PIL.Image.Image,
) -> Tuple[List[PIL.Image.Image], Optional[float]]:
  """Resolve the driving video as a list of frames.

  Order of preference: `--driving-video-path`, the official demo clip in
  `$WAN_ANIMATE_2_DEMO_DIR/template.mp4`, a clip next to the model, then frames synthesized from the
  reference image so that the pipeline still runs offline.

  :param args: CLI arguments.
  :param reference_image: Reference image used for the synthesized fallback.
  :returns: `(frames, fps)` where `fps` is `None` when it cannot be determined.
  """

  from diffusers.utils import load_video

  # `--num-frames` also selects how many driving frames to keep, which is how a request asks for more
  # than one denoising segment: a clip longer than `segment_frame_length` makes the pipeline loop.
  num_frames = args.num_frames if args.num_frames is not None else _WAN_ANIMATE_2_DRIVING_FRAMES
  candidates = []
  if args.driving_video_path is not None:
    candidates.append(args.driving_video_path)
  asset_dir = _wan_animate_2_asset_dir()
  if asset_dir is not None:
    candidates.append(os.path.join(asset_dir, "template.mp4"))
  if len(candidates) > 0:
    frames, fps = load_video(candidates[0], return_fps=True)
    # Only the requested frame count is used, so a long demo clip does not blow up the runtime.
    frames = frames[:num_frames]
    fps = args.driving_video_fps if args.driving_video_fps is not None else fps
    return frames, fps
  logger.warning("No Wan-Animate-2 driving video found: replaying the reference image. Pass "
                 "`--driving-video-path` or set `WAN_ANIMATE_2_DEMO_DIR` for a real motion source.")
  fps = args.driving_video_fps if args.driving_video_fps is not None else 24.0
  return [reference_image] * num_frames, fps


def _wan_animate_2_post_init_hook(pipe) -> None:
  """Patch the Wan-Animate-2 transformer before any optimization touches it.

  The patch functor resolves the per-layer reference K/V inside the block and derives the RoPE
  device from the latents instead of `patch_embedding.weight`, which layerwise offload parks on CPU.
  Cache-dit applies it through the block adapter when caching is enabled; offload-only and parallel
  runs need it too, and `apply()` is idempotent.

  :param pipe: The freshly constructed modular pipeline.
  """

  from ..caching.patch_functors import WanAnimate2PatchFunctor

  transformer = getattr(pipe, "transformer", None)
  if transformer is not None and not getattr(transformer, "_is_patched", False):
    WanAnimate2PatchFunctor().apply(transformer)


def _wan_animate_2_example(
  args: argparse.Namespace,
  hf_model_id: str,
  pipeline_class: type,
  steps: int,
) -> Example:
  """Build the Wan-Animate-2 example shared by the base and distilled checkpoints.

  :param args: CLI arguments.
  :param hf_model_id: Hub id (or env-mapped local directory) of the checkpoint.
  :param pipeline_class: Modular pipeline class to instantiate.
  :param steps: Default number of inference steps.
  :returns: The configured example.
  """

  model_name_or_path = _path(hf_model_id, args=args)
  num_inference_steps = steps if args.num_inference_steps is None else args.num_inference_steps
  height = 320 if args.height is None else args.height
  width = 256 if args.width is None else args.width
  reference_image = _wan_animate_2_reference_image(args, model_name_or_path)
  driving_video, driving_video_fps = _wan_animate_2_driving_video(args, reference_image)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.ANIMATE,
      model_name_or_path=model_name_or_path,
      pipeline_class=pipeline_class,
      bnb_4bit_components=[],  # ignored: modular pipelines take no quantization_config
      post_init_hook=_wan_animate_2_post_init_hook,
      # The released checkpoint has no `guider/` component: the denoise blocks construct the
      # `ClassifierFreeGuidance` guider from their own spec (3.0 for base, 1.0 for distilled).
      modular_component_dtypes={
        "default": torch.bfloat16,
        "vae": torch.float32,
      },
    ),
    input_data=ExampleInputData(
      prompt=_WAN_ANIMATE_2_PROMPT,
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      # The canvas is derived from the reference aspect ratio inside the pipeline, so resizing here
      # would resample twice and shift the crop the model computes.
      skip_image_resize=True,
      image=reference_image,
      extra_input_kwargs={
        "driving_video": driving_video,
        "driving_video_fps": driving_video_fps,
        "segment_frame_length": _WAN_ANIMATE_2_SEGMENT_FRAMES,
        # The writer and the metrics CLI both take PIL frames; the pipeline default returns one
        # `[T, H, W, C]` array per sample instead.
        "output_type": "pil",
      },
    ),
  )


@ExampleRegister.register("wan_animate_2", default="Wan-AI/Wan2.2-Animate-2-14B-Diffusers")
def wan_animate_2_example(args: argparse.Namespace, **kwargs) -> Example:
  """Wan-Animate-2 character animation example (base checkpoint, 40 steps + CFG 3.0).

  A reference character image plus a driving video produce a video of that character following the
  motion. The pipeline is a diffusers `ModularPipeline` that keeps a reference K/V cache per
  transformer layer and denoises in segments of `segment_frame_length` frames, so the verification
  runs shrink the segment length, the canvas and the step count.

  :param args: CLI arguments.
  :returns: The Wan-Animate-2 base example.
  """

  from diffusers import WanAnimate2ModularPipeline

  return _wan_animate_2_example(
    args=args,
    hf_model_id="Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
    pipeline_class=WanAnimate2ModularPipeline,
    steps=40,
  )


@ExampleRegister.register("wan_animate_2_distilled",
                          default="Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers")
def wan_animate_2_distilled_example(args: argparse.Namespace, **kwargs) -> Example:
  """Wan-Animate-2 character animation example (distilled checkpoint, 10 steps, no CFG).

  The distilled guider runs at `guidance_scale=1.0`, i.e. one transformer forward per step, so the
  adapter disables separate-CFG cache bookkeeping for it.

  :param args: CLI arguments.
  :returns: The Wan-Animate-2 distilled example.
  """

  from diffusers import WanAnimate2DistilledModularPipeline

  return _wan_animate_2_example(
    args=args,
    hf_model_id="Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers",
    pipeline_class=WanAnimate2DistilledModularPipeline,
    steps=10,
  )


# LingBot-Video is trained on the structured JSON caption form produced by its prompt rewriter, so
# the bundled prompt keeps that shape: a plain sentence leaves most of the conditioning empty.
_LINGBOT_VIDEO_PROMPT = (
  '{"comprehensive_description": {"scene_content_description": "A red vintage motorcycle is '
  'parked on a quiet cobblestone street at sunset. The camera slowly pushes in toward the fuel '
  'tank while warm light glints off the chrome exhaust pipes and the background buildings blur '
  'into a soft golden bokeh.", "camera_movement_description": "The camera moves forward slowly at '
  'eye level, keeping the motorcycle centered."}, "camera_info": {"color": "Warm", "frame_size": '
  '"Medium", "shot_type_angle": "Eye level", "lens_size": "Medium", "composition": "Center", '
  '"lighting": "Soft light", "lighting_type": "Sunset"}, "world_knowledge": []}')


class _LingBotVideoExample(Example):
  """LingBot-Video example that exposes `frames` through a plain namespace.

  `LingBotVideoPipelineOutput` subclasses diffusers' `BaseOutput`, i.e. it *is* an `OrderedDict`, so
  `LingBotVideoPipeline`'s `frames` field would be dropped by the video-output parse (it reads the
  `videos` key of a dict-shaped output). Wrapping the frame list in a `SimpleNamespace` keeps the
  `frames[0]` layout that the other video examples use.
  """

  def call_pipe(self, pipe, kwargs: Dict[str, Any]):
    from types import SimpleNamespace

    output = super().call_pipe(pipe, kwargs)
    return SimpleNamespace(frames=output.frames)


@ExampleRegister.register(
  "lingbot_video",
  default="robbyant/lingbot-video-dense-1.3b",
)
def lingbot_video_example(args: argparse.Namespace, **kwargs) -> Example:
  """LingBot-Video Dense 1.3B text-to-video example.

  The checkpoint ships its own pipeline (`LingBotVideoPipeline`), so cache-dit loads the built-in
  copy from `cache_dit.models.pipelines.lingbot_video`. Defaults follow the official recipe:
  480x832, 81 frames (24 fps), 40 steps, CFG 3.0 and flow-matching shift 3.0. `--num-frames 1`
  renders a single image instead of a clip; `--num-frames/--height/--width/--steps` override the
  rest.

  :param args: CLI arguments.
  :returns: The LingBot-Video Dense example.
  """
  from ..models.pipelines.lingbot_video import LingBotVideoPipeline

  height = 480 if args.height is None else args.height
  width = 832 if args.width is None else args.width
  num_frames = 81 if args.num_frames is None else args.num_frames
  num_inference_steps = 40 if args.num_inference_steps is None else args.num_inference_steps

  return _LingBotVideoExample(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2V,
      model_name_or_path=_path("robbyant/lingbot-video-dense-1.3b", args),
      pipeline_class=LingBotVideoPipeline,
    ),
    input_data=ExampleInputData(
      prompt=_LINGBOT_VIDEO_PROMPT,
      height=height,
      width=width,
      num_frames=num_frames,
      num_inference_steps=num_inference_steps,
      guidance_scale=3.0,
      # The writer takes PIL frames; the pipeline default returns one `(T, H, W, C)` array instead.
      extra_input_kwargs={"output_type": "pil"},
    ),
  )


_WAN_S2V_PROMPT = ("A person is talking to the camera in a natural indoor setting, medium shot, "
                   "soft light, realistic, high quality.")
_WAN_S2V_NEGATIVE_PROMPT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                            "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                            "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
# Must be a multiple of the VAE temporal factor. 84 - 3 warm-up frames = 81 visible frames
# (5.06 s at 16 fps), which is the 4k+1 length the model was trained on.
_WAN_S2V_FRAMES_PER_CHUNK = 84
_WAN_S2V_SAMPLE_RATE = 16000


def _wan_s2v_asset_dir() -> Optional[str]:
  """Resolve the directory holding the demo image and waveform.

  :returns: `$WAN_2_2_S2V_DEMO_DIR` when it exists, else `None`.
  """

  demo_dir = os.environ.get("WAN_2_2_S2V_DEMO_DIR", None)
  if demo_dir is not None and os.path.isdir(demo_dir):
    return demo_dir
  return None


def _wan_s2v_reference_image(args: argparse.Namespace, model_name_or_path: str) -> PIL.Image.Image:
  """Resolve the reference (identity) image.

  :param args: CLI arguments, honoring `--image-path`.
  :param model_name_or_path: Model directory, used as a fallback asset location.
  :returns: The reference image in RGB.
  """

  if args.image_path is not None:
    return load_image(args.image_path).convert("RGB")
  asset_dir = _wan_s2v_asset_dir()
  if asset_dir is not None and os.path.isfile(os.path.join(asset_dir, "reference.png")):
    return load_image(os.path.join(asset_dir, "reference.png")).convert("RGB")
  local_asset = os.path.join(model_name_or_path, "assets", "reference.png")
  if os.path.isfile(local_asset):
    return load_image(local_asset).convert("RGB")
  logger.warning("No Wan2.2-S2V reference image found: falling back to a bundled portrait. "
                 "Pass `--image-path` or set `WAN_2_2_S2V_DEMO_DIR` for a meaningful sample.")
  # A single-subject portrait, which is what a speech-driven model expects; the repo already ships
  # one for the identity-preserving video models.
  candidates = []
  if CACHE_DIT_DIR is not None:
    candidates.append(os.path.join(CACHE_DIT_DIR, "examples/data/consisid_input.png"))
  # Editable install: this file lives at `<repo>/src/cache_dit/_utils/examples.py`.
  repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
  candidates.append(os.path.join(repo_root, "examples/data/consisid_input.png"))
  for candidate in candidates:
    if os.path.isfile(candidate):
      return load_image(candidate).convert("RGB")
  raise ValueError("No Wan2.2-S2V reference image found. Pass `--image-path` or set "
                   "`WAN_2_2_S2V_DEMO_DIR` (expects `reference.png` and `speech.wav`).")


def _wan_s2v_resample(waveform: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
  """Resample a mono waveform with linear interpolation.

  The audio encoder expects 16 kHz, so a caller that passes 44.1/48 kHz audio would otherwise feed
  the model a waveform at the wrong speed. This is a deliberate approximation: the shipped demo
  asset is prepared at 16 kHz, and this path only exists so an arbitrary wav still runs.

  :param waveform: mono waveform.
  :param source_rate: sample rate of `waveform`.
  :param target_rate: sample rate the encoder expects.
  :returns: the resampled waveform.
  """

  if source_rate == target_rate:
    return waveform
  duration = waveform.shape[0] / float(source_rate)
  target_length = int(round(duration * target_rate))
  source_positions = np.linspace(0.0, duration, num=waveform.shape[0], endpoint=False)
  target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
  return np.interp(target_positions, source_positions, waveform).astype(np.float32)


def _wan_s2v_read_audio(path: str) -> Tuple[np.ndarray, int]:
  """Read a wav file as a 16 kHz mono float32 waveform.

  :param path: wav file path.
  :returns:`(waveform, sampling_rate)`.
  """

  try:
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
  except ImportError:
    # `soundfile` lives in the optional `[metrics]` extra; the builtin `wave` module reads the
    # 16-bit PCM most export tools write, so passing `--audio-path` keeps working without it.
    import wave

    with wave.open(path, "rb") as handle:
      sample_rate = handle.getframerate()
      channels = handle.getnchannels()
      width = handle.getsampwidth()
      raw = handle.readframes(handle.getnframes())
    if width != 2:
      raise ValueError(f"{path}: only 16-bit PCM wav is readable without `soundfile` (got "
                       f"{8 * width}-bit); `pip install soundfile` to read it.") from None
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    audio = data.reshape(-1, channels).mean(axis=1)
  return _wan_s2v_resample(audio, sample_rate, _WAN_S2V_SAMPLE_RATE), _WAN_S2V_SAMPLE_RATE


def _wan_s2v_synthetic_audio(num_frames: int, fps: int) -> np.ndarray:
  """Build a placeholder waveform so the example still runs without any asset.

  The content carries no speech: it only satisfies the audio-conditioning shape. Accuracy gates and
  lip-sync checks need a real recording, passed through `--audio-path`.

  :param num_frames: frames the waveform has to cover.
  :param fps: frame rate, which fixes the duration.
  :returns: a mono waveform at 16 kHz.
  """

  duration = num_frames / float(fps)
  time = np.arange(int(duration * _WAN_S2V_SAMPLE_RATE), dtype=np.float32) / _WAN_S2V_SAMPLE_RATE
  envelope = 0.5 * (1.0 + np.sin(2.0 * np.pi * 3.0 * time))
  return (0.1 * envelope * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)


def _wan_s2v_audio(args: argparse.Namespace, model_name_or_path: str,
                   num_frames: int) -> Tuple[np.ndarray, int]:
  """Resolve the driving waveform.

  Order of preference: `--audio-path`, `$WAN_2_2_S2V_DEMO_DIR/speech.wav`, the wav next to the
  model, then a synthetic placeholder so the pipeline still runs.

  :param args: CLI arguments.
  :param model_name_or_path: Model directory, used as a fallback asset location.
  :param num_frames: frames to cover, used by the synthetic fallback.
  :returns: `(waveform, sampling_rate)`.
  """

  candidates = []
  if args.audio_path is not None:
    candidates.append(args.audio_path)
  asset_dir = _wan_s2v_asset_dir()
  if asset_dir is not None:
    candidates.append(os.path.join(asset_dir, "speech.wav"))
  candidates.append(os.path.join(model_name_or_path, "assets", "speech.wav"))
  for candidate in candidates:
    if os.path.isfile(candidate):
      return _wan_s2v_read_audio(candidate)
  logger.warning("No Wan2.2-S2V driving waveform found: using a synthetic placeholder. Pass "
                 "`--audio-path` or set `WAN_2_2_S2V_DEMO_DIR` for a real sample.")
  return _wan_s2v_synthetic_audio(num_frames, 16), _WAN_S2V_SAMPLE_RATE


@ExampleRegister.register("wan2.2_s2v", default="Wan-AI/Wan2.2-S2V-14B-Diffusers")
def wan_s2v_example(args: argparse.Namespace, **kwargs) -> Example:
  """Wan2.2-S2V speech-to-video example (reference image + waveform -> talking video).

  One chunk renders `num_frames_per_chunk - 3` visible frames; a longer waveform makes the pipeline
  loop over chunks and carry the motion latents forward. Pose conditioning is not implemented, so
  the pipeline always runs the zero-pose path.

  :param args: CLI arguments.
  :returns: The configured example.
  """

  from ..models.pipelines.wan_s2v import WanSpeechToVideoPipeline

  model_name_or_path = _path("Wan-AI/Wan2.2-S2V-14B-Diffusers", args=args)
  num_inference_steps = 40 if args.num_inference_steps is None else args.num_inference_steps
  height = 480 if args.height is None else args.height
  width = 832 if args.width is None else args.width
  num_frames_per_chunk = (args.num_frames
                          if args.num_frames is not None else _WAN_S2V_FRAMES_PER_CHUNK)
  reference_image = _wan_s2v_reference_image(args, model_name_or_path)
  # The pipeline centre-crops the reference to the canvas, which pads the short side with black; a
  # landscape or square portrait set against a 9:16 canvas wastes a large part of the frame.
  aspect_gap = abs(reference_image.width * height - reference_image.height * width)
  if aspect_gap > 0.15 * reference_image.width * height:
    logger.warning(f"Reference image {reference_image.size} does not match the {width}x{height} "
                   f"canvas: the pipeline will letterbox it. Pass `--height/--width` with the "
                   f"reference aspect for a tighter crop.")
  audio, sampling_rate = _wan_s2v_audio(args, model_name_or_path, num_frames_per_chunk)

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.S2V,
      model_name_or_path=model_name_or_path,
      pipeline_class=WanSpeechToVideoPipeline,
      bnb_4bit_components=["text_encoder", "transformer"],
    ),
    input_data=ExampleInputData(
      prompt=_WAN_S2V_PROMPT,
      negative_prompt=_WAN_S2V_NEGATIVE_PROMPT,
      height=height,
      width=width,
      guidance_scale=4.5,
      num_inference_steps=num_inference_steps,
      image=reference_image,
      # The canvas is fixed by the flags, and the pipeline center-crops the reference image itself.
      skip_image_resize=True,
      # The frame count is `num_frames_per_chunk` here, so `--num-frames` must not be forwarded
      # under its usual name.
      use_num_frames=False,
      extra_input_kwargs={
        "num_frames_per_chunk": num_frames_per_chunk,
        "sampling_rate": sampling_rate,
        "sampling_fps": 16,
        "audio": audio,
        # The video writer feeds imageio, which needs ndarray frames (`PIL` output raises
        # "append_data requires ndarray"); the pipeline then returns `(T, H, W, C)` in [0, 1].
        "output_type": "np",
      },
    ),
  )
