import os
import math
import torch
import argparse
import PIL.Image
import numpy as np
from typing import Tuple, List, Optional
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
    "skyreels_v2_example",
    "ltx2_t2v_example",
    "ltx2_i2v_example",
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
    "OVIS_IMAGE_DIR": "AIDC-AI/Ovis-Image-7B",
    "LTX2_DIR": "Lightricks/LTX-2",
    "QWEN_IMAGE_DIR": "Qwen/Qwen-Image",
    "QWEN_IMAGE_2512_DIR": "Qwen/Qwen-Image-2512",
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
    "WAN_VACE_DIR": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "WAN_2_2_VACE_DIR": "linoyts/Wan2.2-VACE-Fun-14B-diffusers",
    "ZIMAGE_DIR": "Tongyi-MAI/Z-Image",
    "ZIMAGE_TURBO_DIR": "Tongyi-MAI/Z-Image-Turbo",
    "NUNCHAKU_ZIMAGE_TURBO_DIR": "nunchaku-tech/nunchaku-z-image-turbo",
    "Z_IMAGE_TURBO_CONTROLNET_2_1_DIR": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
    "Z_IMAGE_TURBO_CONTROLNET_2_0_DIR": "alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0",
    "LONGCAT_IMAGE_DIR": "meituan-longcat/LongCat-Image",
    "LONGCAT_IMAGE_EDIT_DIR": "meituan-longcat/LongCat-Image-Edit",
    "GLM_IMAGE_DIR": "zai-org/GLM-Image",
    "FIRERED_IMAGE_EDIT_1_DIR": "FireRedTeam/FireRed-Image-Edit-1.0",
    "HELIOS_BASE_DIR": "BestWishYsh/Helios-Base",
    "HELIOS_DISTILLED_DIR": "BestWishYsh/Helios-Distilled",
}
_path_env_mapping = {v: k for k, v in _env_path_mapping.items()}


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


@ExampleRegister.register("flux", default="black-forest-labs/FLUX.1-dev")
@ExampleRegister.register("flux_nunchaku", default="nunchaku-tech/nunchaku-flux.1-dev")
def flux_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import FluxPipeline

    if "nunchaku" in args.example.lower():
        from nunchaku.models.transformers.transformer_flux_v2 import (
            NunchakuFluxTransformer2DModelV2,
        )

        nunchaku_flux_dir = _path(
            "nunchaku-tech/nunchaku-flux.1-dev",
            args=args,
            transformer=True,
        )
        transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
            f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors",
        )
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
            image=load_image("https://github.com/vipshop/cache-dit/raw/main/examples/data/cup.png"),
            mask_image=load_image(
                "https://github.com/vipshop/cache-dit/raw/main/examples/data/cup_mask.png"
            ),
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
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=args.residual_diff_threshold,
            ),
        ),
        ParamsModifier(
            # Modified config only for single_transformer_blocks
            # NOTE: FLUX.2, single_transformer_blocks should have `higher`
            # residual_diff_threshold because of the precision error
            # accumulation from previous transformer_blocks
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=args.residual_diff_threshold * 3,
            ),
        ),
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
                "gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
            ),
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
@ExampleRegister.register(
    "flux2_klein_base_4b_edit", default="black-forest-labs/FLUX.2-klein-base-4B"
)
@ExampleRegister.register(
    "flux2_klein_base_9b_edit", default="black-forest-labs/FLUX.2-klein-base-9B"
)
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
    image1 = load_image(
        "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg"
    )  # bear
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
            image=[image1, image2],
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
    prompt = (
        "A coffee shop entrance features a chalkboard sign reading "
        '"Qwen Coffee 😊 $2 per cup," with a neon light beside it '
        'displaying "通义千问". Next to it hangs a poster showing a '
        "beautiful Chinese woman, and beneath the poster is written "
        '"π≈3.1415926-53589793-23846264-33832795-02384197". '
        "Ultra HD, 4K, cinematic composition"
    )
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
@ExampleRegister.register(
    "qwen_image_edit_2511_lightning", default="lightx2v/Qwen-Image-Edit-2511-Lightning"
)
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
            assert steps == 4, "Qwen-Image-Edit-2511-Lightning only supports 4 steps."
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
            prompt=(
                "The magician bear is on the left, the alchemist bear is on the right, "
                "facing each other in the central park square."
            ),
            negative_prompt=" ",
            height=1024,
            width=1024,
            num_inference_steps=steps,
            true_cfg_scale=true_cfg_scale,  # 1.0 means no separate cfg for lightning models
            # image1, image2
            image=[
                load_image(
                    "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_1.jpg"
                ),
                load_image(
                    "https://github.com/vipshop/cache-dit/raw/main/examples/data/edit2509_2.jpg"
                ),
            ],
        ),
    )


@ExampleRegister.register(
    "qwen_image_controlnet", default="InstantX/Qwen-Image-ControlNet-Inpainting"
)
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
        "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting/resolve/main/assets"
    )
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
            image=load_image(
                "https://github.com/vipshop/cache-dit/raw/main/examples/data/yarn-art-pikachu.png"
            ).convert("RGBA"),
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
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        logger.info(
            f"Set UniPCMultistepScheduler with flow_shift={flow_shift} "
            f"for {pipe.__class__.__name__}."
        )

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
            prompt=(
                "A cat and a dog baking a cake together in a kitchen. The cat is "
                "carefully measuring flour, while the dog is stirring the batter "
                "with a wooden spoon. The kitchen is cozy, with sunlight streaming "
                "through the window."
            ),
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
                "Smooth camera motion, natural parallax, ultra-realistic detail, cinematic lighting."
            ),
            negative_prompt=(
                "shaky, glitchy, low quality, worst quality, deformed, distorted, disfigured, motion artifacts, "
                "bad anatomy, ugly, transition, static, text, watermark"
            ),
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
            prompt=(
                "A young girl stands calmly in the foreground, looking directly at the camera, "
                "as a house fire rages in the background."
            ),
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


def _wan_2_2_params_modifiers(args: argparse.Namespace) -> List[ParamsModifier]:
    if not args.cache:
        return None
    return [
        ParamsModifier(
            # high-noise transformer only have 30% steps
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
            bnb_4bit_components=(
                ["text_encoder", "transformer", "transformer_2"]
                if "wan2.2" in args.example.lower()
                else ["text_encoder", "transformer"]
            ),
            extra_optimize_kwargs={
                "params_modifiers": params_modifiers,
            },
        ),
        input_data=ExampleInputData(
            prompt="A cat walks on the grass, realistic",
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst quality, "
                "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
                "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen "
                "limbs, fused fingers, still picture, messy background, three legs, many "
                "people in the background, walking backwards"
            ),
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
            "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/wan_i2v_input.JPG"
        )

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
            bnb_4bit_components=(
                ["text_encoder", "transformer", "transformer_2"]
                if "wan2.2" in args.example.lower()
                else ["text_encoder", "transformer"]
            ),
            extra_optimize_kwargs={
                "params_modifiers": params_modifiers,
            },
        ),
        input_data=ExampleInputData(
            prompt=(
                "Summer beach vacation style, a white cat wearing sunglasses sits on a "
                "surfboard. The fluffy-furred feline gazes directly at the camera with "
                "a relaxed expression. Blurred beach scenery forms the background featuring "
                "crystal-clear waters, distant green hills, and a blue sky dotted with white "
                "clouds. The cat assumes a naturally relaxed posture, as if savoring the sea "
                "breeze and warm sunlight. A close-up shot highlights the feline's intricate "
                "details and the refreshing atmosphere of the seaside."
            ),
            negative_prompt=(
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
                "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
                "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
            ),
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
        logger.info(
            f"Set UniPCMultistepScheduler with flow_shift={flow_shift} "
            f"for {pipe.__class__.__name__}."
        )

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

    first_frame = load_image(
        "https://github.com/vipshop/cache-dit/raw/main/examples/data/flf2v_input_first_frame.png"
    )
    last_frame = load_image(
        "https://github.com/vipshop/cache-dit/raw/main/examples/data/flf2v_input_last_frame.png"
    )

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
            bnb_4bit_components=(
                ["text_encoder", "transformer", "transformer_2"]
                if "wan2.2" in args.example.lower()
                else ["text_encoder", "transformer"]
            ),
            extra_optimize_kwargs={
                "params_modifiers": params_modifiers,
            },
        ),
        input_data=ExampleInputData(
            prompt=(
                "CG animation style, a small blue bird takes off from the ground, "
                "flapping its wings. The bird's feathers are delicate, with a unique "
                "pattern on its chest. The background shows a blue sky with white "
                "clouds under bright sunshine. The camera follows the bird upward, "
                "capturing its flight and the vastness of the sky from a close-up, "
                "low-angle perspective."
            ),
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, "
                "style, works, paintings, images, static, overall gray, worst "
                "quality, low quality, JPEG compression residue, ugly, incomplete, "
                "extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                "disfigured, misshapen limbs, fused fingers, still picture, messy "
                "background, three legs, many people in the background, walking "
                "backwards"
            ),
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
                "texture, 4k detail."
            ),
            height=1024,
            width=1024,
            num_inference_steps=25,
            guidance_scale=5.0,  # has separate cfg for ovis image
        ),
    )


def _zimage_turbo_steps_mask(
    args: argparse.Namespace,
) -> Optional[List[int]]:
    if not args.cache:
        return None
    return (
        steps_mask(
            # slow, medium, fast, ultra.
            mask_policy=args.mask_policy,
            total_steps=9 if args.num_inference_steps is None else args.num_inference_steps,
        )
        if args.mask_policy is not None
        else (
            steps_mask(
                compute_bins=[5, 1, 1],  # = 7 (compute steps)
                cache_bins=[1, 1],  # = 2 (dynamic cache steps)
            )
            if args.steps_mask
            else None
        )
    )


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
            f"{nunchaku_zimage_dir}/svdq-int4_r128-z-image-turbo.safetensors"
        )
    else:
        transformer = None

    if "turbo" in args.example.lower():
        model_name_or_path = _path("Tongyi-MAI/Z-Image-Turbo", args=args)
        prompt = (
            "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
            "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
            "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
            "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
            "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
            "blurred colorful distant lights."
        )
        negative_prompt = None
        height = 1024
        width = 1024
        guidance_scale = 0.0  # Guidance should be 0 for the Turbo models
        num_inference_steps = 9
        steps_computation_mask = _zimage_turbo_steps_mask(args)
    else:
        model_name_or_path = _path("Tongyi-MAI/Z-Image", args=args)
        prompt = (
            "两名年轻亚裔女性紧密站在一起，背景为朴素的灰色纹理墙面，可能是室内地毯地面。"
            "左侧女性留着长卷发，身穿藏青色毛衣，左袖有奶油色褶皱装饰，内搭白色立领衬衫，"
            "下身白色裤子；佩戴小巧金色耳钉，双臂交叉于背后。右侧女性留直肩长发，身穿奶油色卫衣，"
            "胸前印有“Tun the tables”字样，下方为“New ideas”，搭配白色裤子；佩戴银色小环耳环，"
            "双臂交叉于胸前。两人均面带微笑直视镜头。照片，自然光照明，柔和阴影，以藏青、"
            "奶油白为主的中性色调，休闲时尚摄影，中等景深，面部和上半身对焦清晰，姿态放松，"
            "表情友好，室内环境，地毯地面，纯色背景。"
        )
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


@ExampleRegister.register(
    "zimage_turbo_controlnet_2.1", default="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1"
)
@ExampleRegister.register(
    "zimage_turbo_controlnet_2.0", default="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.0"
)
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
        controlnet_path = os.path.join(
            controlnet_dir, "Z-Image-Turbo-Fun-Controlnet-Union-2.0.safetensors"
        )
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
        controlnet_path = os.path.join(
            controlnet_dir, "Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"
        )
        controlnet = ZImageControlNetModel.from_single_file(
            controlnet_path,
            torch_dtype=torch.bfloat16,
        )

    control_image = load_image(
        "https://github.com/vipshop/cache-dit/raw/main/examples/data/pose.jpg"
    )
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
            prompt=(
                "一位年轻女子站在阳光明媚的海岸线上，白裙在轻拂的海风中微微飘动。她拥有一头鲜艳的紫色长发，在风中轻盈舞动，"
                "发间系着一个精致的黑色蝴蝶结，与身后柔和的蔚蓝天空形成鲜明对比。她面容清秀，眉目精致，透着一股甜美的青春气息；"
                "神情柔和，略带羞涩，目光静静地凝望着远方的地平线，双手自然交叠于身前，仿佛沉浸在思绪之中。在她身后，"
                "是辽阔无垠、波光粼粼的大海，阳光洒在海面上，映出温暖的金色光晕。"
            ),
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
            prompt=(
                "A young Asian woman wearing a yellow knit sweater paired with a white necklace. "
                "Her hands rest on her knees, with a serene expression. The background features a "
                "rough brick wall, with warm afternoon sunlight casting upon her, creating a tranquil "
                "and cozy atmosphere. The shot uses a medium-distance perspective, highlighting her "
                "demeanor and the details of her attire. Soft lighting illuminates her face, emphasizing "
                "her facial features and the texture of her accessories, adding depth and warmth to the image. "
                "The overall composition is simple and elegant, with the brick wall's texture complementing "
                "the interplay of sunlight and shadows, showcasing the character's grace and composure."
            ),
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
            "https://huggingface.co/meituan-longcat/LongCat-Image-Edit/resolve/main/assets/test.png"
        )
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
            prompt="Replace the background of the snow forest with an underground station featuring an automatic escalator.",
            image=[image],  # pass as a list for GLM-Image Edit
            height=height,
            width=width,
            num_inference_steps=50,
        ),
    )


@ExampleRegister.register("firered_image_edit_1.0", default="FireRedTeam/FireRed-Image-Edit-1.0")
def firered_image_edit_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import QwenImageEditPlusPipeline

    if args.image_path is not None:
        image = load_image(args.image_path).convert("RGB")
    else:
        image = load_image(
            "https://github.com/vipshop/cache-dit/raw/main/examples/data/firered_edit_example.png"
        ).convert("RGB")

    if args.input_height is not None and args.input_width is not None:
        image = image.resize((args.input_width, args.input_height))

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.IE2I,  # Image Editing to Image
            model_name_or_path=_path("FireRedTeam/FireRed-Image-Edit-1.0", args=args),
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
    vae = AutoencoderKLWan.from_pretrained(
        model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

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
                "The kitchen is cozy, with sunlight streaming through the window."
            ),
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
                "images, static, overall gray, worst quality, low quality, JPEG compression residue, "
                "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
                "many people in the background, walking backwards"
            ),
            height=384,
            width=640,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=5.0,
        ),
    )


@ExampleRegister.register("heliost_t2v_distill", default="BestWishYsh/Helios-Distilled")
def helios_t2v_distill_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import HeliosPyramidPipeline, AutoencoderKLWan
    from ..platforms import current_platform

    assert args.num_inference_steps is None, (
        "Helios Distilled example should use the default pyramid_num_inference_steps_list "
        "which is [2,2,2] for better performance. Please do not set num_inference_steps for "
        "this example since it will break the step hint logic and lead to worse performance."
    )

    model_name_or_path = _path("BestWishYsh/Helios-Distilled", args=args)
    vae = AutoencoderKLWan.from_pretrained(
        model_name_or_path, subfolder="vae", torch_dtype=torch.float32
    )

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
            },
        ),
        input_data=ExampleInputData(
            prompt=(
                "A cat and a dog baking a cake together in a kitchen. The cat is "
                "carefully measuring flour, while the dog is stirring the batter with a wooden spoon. "
                "The kitchen is cozy, with sunlight streaming through the window."
            ),
            negative_prompt=(
                "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
                "images, static, overall gray, worst quality, low quality, JPEG compression residue, "
                "ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, "
                "disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, "
                "many people in the background, walking backwards"
            ),
            height=384,
            width=640,
            num_frames=num_frames,
            guidance_scale=1.0,
            extra_input_kwargs={
                "pyramid_num_inference_steps_list": [2, 2, 2],
                "is_amplify_first_chunk": True,
            },
        ),
    )
