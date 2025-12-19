import os
import math
import torch
import argparse
import PIL.Image
import cache_dit
from typing import Tuple, List, Optional
from diffusers.utils import load_image
from cache_dit import DBCacheConfig, ParamsModifier
from cache_dit.logger import init_logger

from base import (
    Example,
    ExampleType,
    ExampleInputData,
    ExampleInitConfig,
    ExampleRegister,
)

logger = init_logger(__name__)


__all__ = [
    "flux_example",
    "flux_nunchaku_example",
    "flux2_example",
    "ovis_image_example",
    "qwen_image_edit_lightning_example",
    "qwen_image_example",
    "skyreels_v2_example",
    "wan_example",
    "wan_vace_example",
    "zimage_example",
]


# Please note that the following environment variables is only for debugging
# and development purpose. In practice, users should directly provide the model
# names or paths. The default values are the official model names on
# HuggingFace Hub.
_env_path_mapping = {
    "FLUX_DIR": "black-forest-labs/FLUX.1-dev",
    "NUNCHAKA_ FLUX_DIR": "nunchaku-tech/nunchaku-flux.1-dev",
    "FLUX_2_DIR": "black-forest-labs/FLUX.2-dev",
    "OVIS_IMAGE_DIR": "AIDC-AI/Ovis-Image-7B",
    "QWEN_IMAGE_DIR": "Qwen/Qwen-Image",
    "QWEN_IMAGE_LIGHT_DIR": "lightx2v/Qwen-Image-Lightning",
    "QWEN_IMAGE_EDIT_2509_DIR": "Qwen/Qwen-Image-Edit-2509",
    "SKYREELS_V2_DIR": "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers",
    "WAN_DIR": "Wan2.1-T2V-1.3B-Diffusers",
    "WAN_2_2_DIR": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    "WAN_VACE_DIR": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
    "WAN_2_2_VACE_DIR": "linoyts/Wan2.2-VACE-Fun-14B-diffusers",
    "ZIMAGE_DIR": "Tongyi-MAI/Z-Image-Turbo",
}
_path_env_mapping = {v: k for k, v in _env_path_mapping.items()}


def _path(
    default: str,
    args: Optional[argparse.Namespace] = None,
    ENV: Optional[str] = None,
) -> str:
    # Prefer command line argument if provided
    if args is not None:
        model_path_arg = args.model_path
        if model_path_arg is not None:
            return model_path_arg
    # Next, check environment variable
    if ENV is None:
        ENV = _path_env_mapping.get(default, None)
        if ENV is None:
            return default
    return os.environ.get(ENV, default)


@ExampleRegister.register("flux")
def flux_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import FluxPipeline

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=_path("black-forest-labs/FLUX.1-dev"),
            pipeline_class=FluxPipeline,
            # `text_encoder_2` will be quantized when `--quantize-type`
            # is set to `bnb_4bit`.
            bnb_4bit_components=["text_encoder_2"],
        ),
        input_data=ExampleInputData(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=28,
        ),
    )


@ExampleRegister.register("flux_nunchaku")
def flux_nunchaku_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import FluxPipeline
    from nunchaku.models.transformers.transformer_flux_v2 import (
        NunchakuFluxTransformer2DModelV2,
    )

    nunchaku_flux_dir = _path("nunchaku-tech/nunchaku-flux.1-dev")
    transformer = NunchakuFluxTransformer2DModelV2.from_pretrained(
        f"{nunchaku_flux_dir}/svdq-int4_r32-flux.1-dev.safetensors",
    )
    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=_path("black-forest-labs/FLUX.1-dev"),
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


@ExampleRegister.register("flux2")
def flux2_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import Flux2Pipeline

    params_modifiers = [
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


@ExampleRegister.register("ovis_image")
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


@ExampleRegister.register("qwen_image_edit_lightning")
def qwen_image_edit_lightning_example(args: argparse.Namespace, **kwargs) -> Example:
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

    steps = 8 if args.num_inference_steps is None else args.num_inference_steps
    assert steps in [8, 4]
    lora_weights_path = os.path.join(
        _path("lightx2v/Qwen-Image-Lightning"),
        "Qwen-Image-Edit-2509",
    )
    lora_weight_name = f"Qwen-Image-Edit-2509-Lightning-{steps}steps-V1.0-bf16.safetensors"
    base_image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image"

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.IE2I,  # Image Editing to Image
            model_name_or_path=_path("Qwen/Qwen-Image-Edit-2509"),
            pipeline_class=QwenImageEditPlusPipeline,
            scheduler=scheduler,
            bnb_4bit_components=["text_encoder", "transformer"],
            lora_weights_path=lora_weights_path,
            lora_weights_name=lora_weight_name,
            force_fuse_lora=True,  # For parallelism compatibility
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
            # image1, image2
            image=[
                load_image(f"{base_image_url}/edit2509/edit2509_1.jpg"),
                load_image(f"{base_image_url}/edit2509/edit2509_2.jpg"),
            ],
        ),
    )


@ExampleRegister.register("qwen_image")
def qwen_image_example(args: argparse.Namespace, **kwargs) -> Example:
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
    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=_path("Qwen/Qwen-Image"),
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


@ExampleRegister.register("skyreels_v2")
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


@ExampleRegister.register("wan2.1_t2v")
@ExampleRegister.register("wan2.2_t2v")
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

    params_modifiers = [
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


@ExampleRegister.register("wan2.1_vace")
@ExampleRegister.register("wan2.2_vace")
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

    first_frame = load_image("./data/flf2v_input_first_frame.png")
    last_frame = load_image("./data/flf2v_input_last_frame.png")

    height = 512 if args.height is None else args.height
    width = 512 if args.width is None else args.width
    num_frames = 81 if args.num_frames is None else args.num_frames
    video, mask = _video_and_mask(first_frame, last_frame, height, width, num_frames)

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.FLF2V,  # First and Last Frames to Video
            model_name_or_path=model_name_or_path,
            pipeline_class=WanVACEPipeline,
            vae=vae,
            post_init_hook=post_init_hook,
            bnb_4bit_components=(
                ["text_encoder", "transformer", "transformer_2"]
                if "wan2.2" in args.example.lower()
                else ["text_encoder", "transformer"]
            ),
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


@ExampleRegister.register("zimage")
def zimage_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import ZImagePipeline

    if args.cache:
        # Only warmup 4 steps (total 9 steps) for distilled models
        args.max_warmup_steps = min(4, args.max_warmup_steps)

    steps_computation_mask = (
        cache_dit.steps_mask(
            # slow, medium, fast, ultra.
            mask_policy=args.mask_policy,
            total_steps=9 if args.num_inference_steps is None else args.num_inference_steps,
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
    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,  # Text to Image
            model_name_or_path=_path("Tongyi-MAI/Z-Image-Turbo"),
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
