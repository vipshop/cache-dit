import os
import math
import torch
import requests
import argparse
from PIL import Image
from io import BytesIO
import cache_dit
from cache_dit import DBCacheConfig, ParamsModifier

from base import (
    CacheDiTExampleRegister,
    CacheDiTExample,
    ExampleInputData,
    ExampleInitConfig,
    ExampleType,
)
from utils import GiB


def default_path(ENV: str, default: str) -> str:
    return os.environ.get(ENV, default)


def load_image(url: str) -> Image.Image:
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


__all__ = [
    "flux_example",
    "flux_nunchaku_example",
    "flux2_example",
    "ovis_image_example",
    "qwen_image_edit_lightning_example",
    "qwen_image_example",
    "skyreels_v2_example",
    "wan2_2_example",
    "zimage_example",
]


@CacheDiTExampleRegister.register("flux")
def flux_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import FluxPipeline

    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
            pipeline_class=FluxPipeline,
            bnb_4bit_components=["text_encoder_2"],
        ),
        input_data=ExampleInputData(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=28,
        ),
    )


@CacheDiTExampleRegister.register("flux_nunchaku")
def flux_nunchaku_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import FluxPipeline
    from nunchaku.models.transformers.transformer_flux_v2 import (
        NunchakuFluxTransformer2DModelV2,
    )

    nunchaku_flux_dir = default_path(
        "NUNCHAKA_FLUX_DIR",
        "nunchaku-tech/nunchaku-flux.1-dev",
    )
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors",
    )
    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path("FLUX_DIR", "black-forest-labs/FLUX.1-dev"),
            pipeline_class=FluxPipeline,
            transformer=transformer,
            bnb_4bit_components=["text_encoder_2"],
        ),
        input_data=ExampleInputData(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=28,
        ),
    )


@CacheDiTExampleRegister.register("flux2")
def flux2_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import Flux2Pipeline

    if GiB() < 128:
        assert args.quantize, "Quantization is required to fit FLUX.2 in <128GB memory."
        assert args.quantize_type in ["bitsandbytes_4bit", "float8_weight_only"], (
            f"Unsupported quantization type: {args.quantize_type}, only supported"
            "'bitsandbytes_4bit (bnb_4bit)' and 'float8_weight_only'."
        )
    params_modifiers = [
        ParamsModifier(
            # Modified config only for transformer_blocks
            # Must call the `reset` method of DBCacheConfig.
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=args.rdt,
            ),
        ),
        ParamsModifier(
            # Modified config only for single_transformer_blocks
            # NOTE: FLUX.2, single_transformer_blocks should have `higher`
            # residual_diff_threshold because of the precision error
            # accumulation from previous transformer_blocks
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=args.rdt * 3,
            ),
        ),
    ]
    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path("FLUX2_DIR", "black-forest-labs/FLUX.2-dev"),
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


@CacheDiTExampleRegister.register("ovis_image")
def ovis_image_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import OvisImagePipeline

    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path(
                "OVIS_IMAGE_DIR",
                "ovis-models/ovis-image-v1-5b",
            ),
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


@CacheDiTExampleRegister.register("qwen_image_edit_lightning")
def qwen_image_edit_lightning_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import QwenImageEditPlusPipeline, FlowMatchEulerDiscreteScheduler

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
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

    steps = 8 if args.steps is None else args.steps
    assert steps in [8, 4]
    lora_weights_path = os.path.join(
        os.environ.get("QWEN_IMAGE_LIGHT_DIR", "lightx2v/Qwen-Image-Lightning"),
        "Qwen-Image-Edit-2509",
    )
    lora_weight_name = (
        "Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors"
        if steps > 4
        else "Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
    )
    base_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image"

    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.IE2I,  # Image Editing to Image
            model_name_or_path=default_path(
                "QWEN_IMAGE_EDIT_PLUS_DIR",
                "Qwen/Qwen-Image-Edit-2509",
            ),
            pipeline_class=QwenImageEditPlusPipeline,
            scheduler=scheduler,
            bnb_4bit_components=["text_encoder", "transformer"],
            lora_weights_path=lora_weights_path,
            lora_weights_name=lora_weight_name,
            extra_optimize_kwargs={
                "cache_config": (
                    DBCacheConfig(
                        Fn_compute_blocks=16,
                        Bn_compute_blocks=16,
                        max_warmup_steps=4 if steps > 4 else 2,
                        max_cached_steps=2 if steps > 4 else 1,
                        max_continuous_cached_steps=1,
                        enable_separate_cfg=False,  # true_cfg_scale=1.0
                        residual_diff_threshold=0.50 if steps > 4 else 0.8,
                    )
                    if args.cache
                    else None
                ),
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
            true_cfg_scale=1.0,  # means no separate cfg for lightning models
            extra_input_kwargs={
                "image": [
                    # image1, image2
                    load_image(f"{base_image_url}/edit2509/edit2509_1.jpg"),
                    load_image(f"{base_image_url}/edit2509/edit2509_2.jpg"),
                ],
            },
        ),
    )


@CacheDiTExampleRegister.register("qwen_image")
def qwen_image_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import QwenImagePipeline

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
    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path(
                "QWEN_IMAGE_DIR",
                "Qwen/Qwen-Image",
            ),
            pipeline_class=QwenImagePipeline,
            bnb_4bit_components=["text_encoder", "transformer"],
        ),
        input_data=ExampleInputData(
            prompt=prompt + positive_magic["en"],
            negative_prompt=" ",
            height=1024,
            width=1024,
            num_inference_steps=50,
            true_cfg_scale=4.0,
        ),
    )


@CacheDiTExampleRegister.register("skyreels_v2")
def skyreels_v2_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import AutoModel, SkyReelsV2Pipeline, UniPCMultistepScheduler

    model_name_or_path = default_path(
        "SKYREELS_V2_DIR",
        "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
    )
    vae = AutoModel.from_pretrained(
        model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to("cuda")

    def post_init_hook(pipe: SkyReelsV2Pipeline, **kwargs):
        flow_shift = 8.0  # 8.0 for T2V, 5.0 for I2V
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )

    return CacheDiTExample(
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


@CacheDiTExampleRegister.register("wan2.2")
def wan2_2_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import WanPipeline

    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2V,  # Text to Video
            model_name_or_path=default_path("WAN_2_2_DIR", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
            pipeline_class=WanPipeline,
            bnb_4bit_components=["text_encoder", "transformer"],
            extra_optimize_kwargs={
                "params_modifiers": [
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


@CacheDiTExampleRegister.register("zimage")
def zimage_example(args: argparse.Namespace, **kwargs) -> CacheDiTExample:
    from diffusers import ZImagePipeline

    if args.cache:
        # Only warmup 4 steps (total 9 steps) for distilled models
        args.max_warmup_steps = min(4, args.max_warmup_steps)

    steps_computation_mask = (
        cache_dit.steps_mask(
            # slow, medium, fast, ultra.
            mask_policy=args.mask_policy,
            total_steps=9 if args.steps is None else args.steps,
        )
        if args.mask_policy is not None
        else (
            cache_dit.steps_mask(
                compute_bins=[5, 1, 1],  # = 7 (compute steps)
                cache_bins=[1, 1],  # = 2 (dynamic cache steps)
            )
            if args.steps_mask
            else None
        )
    )
    return CacheDiTExample(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=default_path("ZIMAGE_DIR", "Tongyi-MAI/Z-Image-Turbo"),
            pipeline_class=ZImagePipeline,
            bnb_4bit_components=["text_encoder"],
            extra_optimize_kwargs={
                "steps_computation_mask": steps_computation_mask,
            },
        ),
        input_data=ExampleInputData(
            prompt=(
                "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, "
                "red floral forehead pattern. Elaborate high bun, golden phoenix headdress, "
                "red flowers, beads. Holds round folding fan with lady, trees, bird. Neon "
                "lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. "
                "Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), "
                "blurred colorful distant lights."
            ),
            height=1024,
            width=1024,
            guidance_scale=0.0,  # Guidance should be 0 for the Turbo models
            num_inference_steps=9,
        ),
    )
