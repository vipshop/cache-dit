import os
import time
import types
import torch
import argparse
import dataclasses
from PIL import Image
from enum import Enum
import numpy as np
from typing import Dict, Any, Union, Optional, List, Callable
from diffusers.utils import export_to_video, load_image
from diffusers.schedulers import SchedulerMixin
from diffusers import DiffusionPipeline, ModelMixin
from transformers import GenerationMixin
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.quantizers import PipelineQuantizationConfig

from ..summary import summary
from ..logger import init_logger

from .utils import (
    strify,
    maybe_destroy_distributed,
    maybe_init_distributed,
    maybe_apply_optimization,
    pipe_quant_bnb_4bit_config,
    create_profiler_from_args,
    MemoryTracker,
)

logger = init_logger(__name__)


class ExampleType(Enum):
    T2V = "T2V - Text to Video"
    I2V = "I2V - Image to Video"
    T2I = "T2I - Text to Image"
    IE2I = "IE2I - Image Editing to Image"
    FLF2V = "FLF2V - First Last Frames to Video"
    VACE = "VACE - Video All-in-one Creation and Editing"


@dataclasses.dataclass
class ExampleInputData:
    # This class provides default input data for examples.
    # The default values may be overridden by command line
    # args or other means.
    # General inputs for both image and video generation
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None  # for dual guidance scale
    true_cfg_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    num_images_per_prompt: Optional[int] = None
    num_frames: Optional[int] = None
    # Specific inputs for image editing
    image: Optional[Union[List[Image.Image], Image.Image]] = None
    mask_image: Optional[Union[List[Image.Image], Image.Image]] = None
    # Specific inputs for video generation, e.g, Wan VACE
    video: Optional[List[Image.Image]] = None
    mask: Optional[List[Image.Image]] = None
    # Specific inputs for controlnet, e.g, Qwen-Image-ControlNet-Inpainting
    control_image: Optional[Union[List[Image.Image], Image.Image]] = None
    control_mask: Optional[Union[List[Image.Image], Image.Image]] = None
    controlnet_conditioning_scale: Optional[float] = None
    # Specific inputs for Qwen Image Layered
    layers: Optional[int] = None
    resolution: Optional[int] = None
    cfg_normalize: Optional[bool] = None
    use_en_prompt: Optional[bool] = None
    # Other inputs
    seed: int = 0
    # Use 'cpu' by default for better reproducibility across different hardware
    gen_device: str = "cpu"
    generator: torch.Generator = dataclasses.field(
        default_factory=lambda: torch.Generator("cpu").manual_seed(0)
    )
    # Some extra args, e.g, editing model specific inputs
    extra_input_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def data(self, args: argparse.Namespace) -> Dict[str, Any]:
        self._preprocess()
        data = dataclasses.asdict(self)
        # Flatten extra_args and merge into main dict
        extra_args = data.pop("extra_input_kwargs")  # {key: value, ...}
        extra_args = extra_args if extra_args is not None else {}
        # Remove None values from extra_args
        extra_data = {k: v for k, v in extra_args.items() if v is not None}
        input_data = {k: v for k, v in data.items() if v is not None}
        input_data.update(extra_data)
        # Override with args if provided
        if args.prompt is not None:
            input_data["prompt"] = args.prompt
        if args.negative_prompt is not None:
            input_data["negative_prompt"] = args.negative_prompt
        if args.height is not None:
            input_data["height"] = args.height
        if args.width is not None:
            input_data["width"] = args.width
        if args.num_inference_steps is not None:
            input_data["num_inference_steps"] = args.num_inference_steps
        if args.num_frames is not None:
            input_data["num_frames"] = args.num_frames
        if args.image_path is not None:
            if "image" in input_data:
                if isinstance(input_data["image"], list):
                    if len(input_data["image"]) > 1:
                        logger.warning(
                            "Overriding multiple input images with a single image "
                            "from args.image_path."
                        )
            input_data["image"] = load_image(args.image_path).convert("RGB")
        if args.mask_image_path is not None:
            if "mask_image" in input_data:
                if isinstance(input_data["mask_image"], list):
                    if len(input_data["mask_image"]) > 1:
                        logger.warning(
                            "Overriding multiple input mask images with a single mask "
                            "image from args.mask_image_path."
                        )
            input_data["mask_image"] = load_image(args.mask_image_path).convert("RGB")
        # Set generator with seed from input data or args
        if args.generator_device is not None:
            self.gen_device = args.generator_device
        if args.seed is not None:
            self.seed = args.seed
        input_data["generator"] = torch.Generator(self.gen_device).manual_seed(self.seed)
        # Remove redundant keys from input data
        input_data.pop("seed", None)
        input_data.pop("gen_device", None)
        return input_data

    def new_generator(self, args: argparse.Namespace = None) -> torch.Generator:
        # NOTE: We should always create a new generator before each inference to
        # ensure reproducibility when using the same seed. Alawys use cpu generator
        # for better cross-device consistency.
        if args is not None and args.generator_device is not None:
            self.gen_device = args.generator_device
        if args is not None and args.seed is not None:
            return torch.Generator(self.gen_device).manual_seed(args.seed)
        elif self.seed is not None:
            return torch.Generator(self.gen_device).manual_seed(self.seed)
        else:
            return torch.Generator(self.gen_device).manual_seed(0)

    def _preprocess(self):
        if self.image is not None:
            if isinstance(self.image, list) and len(self.image) == 1:
                # unwrap single image from list for general use cases
                self.image = self.image[0]
        if self.mask_image is not None:
            if isinstance(self.mask_image, list) and len(self.mask_image) == 1:
                # unwrap single mask image from list for general use cases
                self.mask_image = self.mask_image[0]

    def summary(self, args: argparse.Namespace) -> str:
        summary_str = "🤖 Example Input Summary:\n"
        data = self.data(args)
        for k, v in data.items():
            if k in ["prompt", "negative_prompt"]:
                summary_str += f"- {k}: {v}\n"
            elif k in ["height", "width", "num_inference_steps", "num_frames"]:
                summary_str += f"- {k}: {v}\n"
            elif k in ["image", "mask_image", "control_image", "control_mask"]:
                if isinstance(v, Image.Image):
                    W, H = v.size
                    summary_str += f"- {k}: Single Image ({H}x{W})\n"
                elif isinstance(v, list):
                    if len(v) > 0:
                        summary_str += f"- {k}: List Images ({len(v)} images)\n"
                        for i in range(min(len(v), 3)):  # show up to 3 images
                            if isinstance(v[i], Image.Image):
                                W, H = v[i].size
                                summary_str += f"    - Image {i}: ({H}x{W})\n"
                            else:
                                summary_str += f"    - Image {i}: Not a valid PIL Image\n"
                    elif len(v) == 1:
                        if isinstance(v[0], Image.Image):
                            W, H = v[0].size
                            summary_str += f"- {k}: Single Image ({H}x{W})\n"
                        else:
                            summary_str += f"- {k}: Not a valid PIL Image\n"
                    else:
                        summary_str += f"- {k}: Empty List\n"
            elif k in ["video", "mask"]:
                if isinstance(v, list):
                    if len(v) > 0:
                        summary_str += f"- {k}: List of Frames ({len(v)} frames)\n"
                        for i in range(min(len(v), 1)):  # show up to 1 frames
                            if isinstance(v[i], Image.Image):
                                W, H = v[i].size
                                summary_str += f"    - Frame {i}: ({H}x{W})\n"
                            else:
                                summary_str += f"    - Frame {i}: Not a valid PIL Image\n"
                    else:
                        summary_str += f"- {k}: Empty List\n"
                else:
                    summary_str += f"- {k}: Not a valid list of frames\n"
            elif k == "generator":
                # Show seed and device info
                if isinstance(v, torch.Generator):
                    gen_device = v.device if hasattr(v, "device") else "cpu"
                    gen_seed = v.initial_seed() if hasattr(v, "initial_seed") else "N/A"
                    summary_str += f"- {k}: device {gen_device}, seed {gen_seed}\n"
                else:
                    summary_str += f"- {k}: Not a valid torch.Generator\n"
            else:
                summary_str += f"- {k}: {v}\n"
        summary_str = summary_str.rstrip("\n")
        logger.info(summary_str)
        return summary_str


@dataclasses.dataclass
class ExampleOutputData:
    # Tag
    model_tag: Optional[str] = None
    strify_tag: Optional[str] = None
    # Generated image or video
    image: Optional[Image.Image | List[Image.Image]] = (
        None  # Single PIL Images or list of PIL Images
    )
    video: Optional[List[Image.Image]] = None  # List of PIL Images or video frames
    # Performance metrics
    load_time: Optional[float] = None
    warmup_time: Optional[float] = None
    inference_time: Optional[float] = None
    memory_usage: Optional[float] = None
    # Other outputs
    extra_output_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def save(self, args: argparse.Namespace) -> None:
        # TODO: Handle other extra outputs as needed
        save_path = args.save_path
        if save_path is None:
            save_path = self._default_save_path()
            if save_path is None:
                logger.warning("No valid save path found for output data.")
                return

        if self.image is not None:
            if isinstance(self.image, Image.Image):
                self.image.save(save_path)
                logger.info(f"Image saved to {save_path}")
            elif isinstance(self.image, list):
                save_pre = ".".join(save_path.split(".")[:-1])
                save_ext = save_path.split(".")[-1]
                for i, img in enumerate(self.image):
                    img_save_path = f"{save_pre}_{i}.{save_ext}"
                    img.save(img_save_path)
                    logger.info(f"Image {i} saved to {img_save_path}")

        if self.video is not None:
            export_to_video(self.video, save_path, fps=8)
            logger.info(f"Video saved to {save_path}")

    def _default_save_path(self) -> Optional[str]:
        if self.image is not None:
            try:
                W, H = self.image.size
                HxW_str = f"{H}x{W}"
            except Exception:
                HxW_str = None
            if HxW_str is not None:
                if HxW_str not in self.strify_tag:
                    return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.png"
                else:
                    return f"{self.model_tag}.{self.strify_tag}.png"
            else:
                return f"{self.model_tag}.{self.strify_tag}.png"
        elif self.video is not None:
            try:
                if isinstance(self.video, (list, np.ndarray)) and len(self.video) > 0:
                    if isinstance(self.video[0], Image.Image):
                        W, H = self.video[0].size
                    elif isinstance(self.video[0], np.ndarray):
                        frame = self.video[0]  # type: np.ndarray
                        H, W = frame.shape[:2]
                    else:
                        raise ValueError("Invalid video frame type.")
                    if isinstance(self.video, list):
                        num_frames = len(self.video)
                    elif isinstance(self.video, np.ndarray):
                        num_frames = self.video.shape[0]
                    else:
                        raise ValueError("Invalid video type.")
                    HxW_str = f"{H}x{W}x{num_frames}"
                else:
                    HxW_str = None
            except Exception:
                HxW_str = None
            if HxW_str is not None:
                if HxW_str not in self.strify_tag:
                    return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.mp4"
                else:
                    return f"{self.model_tag}.{self.strify_tag}.mp4"
            else:
                return f"{self.model_tag}.{self.strify_tag}.mp4"
        else:
            return None

    def summary(self, args: argparse.Namespace) -> str:
        from cache_dit.platforms import current_platform

        logger.info("🤖 Example Output Summary:")
        summary_str = f"- Model: {args.example}\n- Optimization: {self.strify_tag}\n"
        device_name = current_platform.get_device_name()
        world_size = (
            1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
        )
        summary_str += f"- Device: {device_name} x {world_size}\n"
        if self.load_time is not None:
            summary_str += f"- Load Time: {self.load_time:.2f}s\n"
        if self.warmup_time is not None:
            summary_str += f"- Warmup Time: {self.warmup_time:.2f}s\n"
        if self.inference_time is not None:
            summary_str += f"- Inference Time: {self.inference_time:.2f}s\n"
        if self.memory_usage is not None:
            summary_str += f"- Memory Usage: {self.memory_usage:.2f}GiB\n"
        summary_str = summary_str.rstrip("\n")
        logger.info(summary_str)
        return summary_str


@dataclasses.dataclass
class ExampleInitConfig:
    # This class provides default initialization config for examples.
    # The default values may be overridden by command line args or other means.
    task_type: ExampleType
    model_name_or_path: str
    pipeline_class: Optional[type[DiffusionPipeline]] = DiffusionPipeline
    torch_dtype: Optional[torch.dtype] = torch.bfloat16
    bnb_4bit_components: Optional[List[str]] = dataclasses.field(default_factory=list)
    scheduler: Optional[Union[SchedulerMixin, Callable]] = None  # lora case
    transformer: Optional[Union[ModelMixin, Callable]] = None  # lora or nunchaku case
    vae: Optional[Union[ModelMixin, Callable]] = None
    text_encoder: Optional[Union[GenerationMixin, Callable]] = None
    controlnet: Optional[Union[ModelMixin, Callable]] = None
    lora_weights_path: Optional[str] = None
    lora_weights_name: Optional[str] = None
    # For parallelism compatibility, tensor parallelism requires fused LoRA
    force_fuse_lora: bool = True
    pre_init_hook: Optional[Callable[[Any], None]] = None  # For future use
    post_init_hook: Optional[Callable[[DiffusionPipeline], None]] = None
    # For DBCache, Parallelism optimization.
    extra_optimize_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.bnb_4bit_components:
            self.bnb_4bit_components = ["text_encoder"]
        if self.extra_optimize_kwargs:
            # remove None values
            self.extra_optimize_kwargs = {
                k: v for k, v in self.extra_optimize_kwargs.items() if v is not None
            }

    def get_pipe(self, args: argparse.Namespace, **kwargs) -> DiffusionPipeline:
        if self.pipeline_class is None:
            raise ValueError("pipeline_class must be provided to get the pipeline instance.")
        pipeline_quantization_config = self._pipeline_quantization_config(args)
        pipe = self.pipeline_class.from_pretrained(
            self.model_name_or_path if args.model_path is None else args.model_path,
            torch_dtype=self.torch_dtype,
            quantization_config=pipeline_quantization_config,
            device_map="balanced" if args.device_map_balance else None,
            **self._custom_components_kwargs(),
        )  # type: LoraBaseMixin
        if self.post_init_hook is not None:
            self.post_init_hook(pipe, **kwargs)

        # Load lora and fuse if needed
        if self.has_lora:
            assert issubclass(
                type(pipe), LoraBaseMixin
            ), "Pipeline class must inherit from LoraBaseMixin to load LoRA weights."
            assert hasattr(
                pipe, "load_lora_weights"
            ), "Pipeline instance must have load_lora_weights method to load LoRA weights."
            if self.lora_weights_name is None:
                # TODO: Support adapter name in the future
                pipe.load_lora_weights(self.lora_weights_path)
            else:
                pipe.load_lora_weights(self.lora_weights_path, weight_name=self.lora_weights_name)
            if not args.disable_fuse_lora and (
                pipeline_quantization_config is None
                or "transformer" not in pipeline_quantization_config.components_to_quantize
                or self.force_fuse_lora
            ):
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                logger.info("Fused and unloaded LoRA weights into the transformer.")
            else:
                logger.warning("Keep LoRA weights in memory since transformer is quantized.")

        return pipe

    def summary(self, args: argparse.Namespace, **kwargs) -> str:
        logger.info("🤖 Example Init Config Summary:")
        extra_model_path = kwargs.get("extra_model_path", "")
        model_name_or_path = self.model_name_or_path if args.model_path is None else args.model_path
        summary_str = "- Model: "
        if (
            os.path.basename(extra_model_path).lower()
            != os.path.basename(model_name_or_path).lower()
        ):
            summary_str += f"\n    - {model_name_or_path}\n"
            summary_str += f"    - {extra_model_path}\n"
        else:
            summary_str += f"{model_name_or_path}\n"
        summary_str += f"- Task Type: {self.task_type.value}\n"
        summary_str += f"- Torch Dtype: {self.torch_dtype}\n"
        if self.lora_weights_path is not None and self.lora_weights_name is not None:
            summary_str += (
                f"- LoRA Weights: {os.path.join(self.lora_weights_path, self.lora_weights_name)}\n"
            )
        elif self.lora_weights_path is not None:
            summary_str += f"- LoRA Path: {self.lora_weights_path}\n"
        else:
            summary_str += "- LoRA Weights: None\n"
        summary_str = summary_str.rstrip("\n")
        logger.info(summary_str)
        return summary_str

    def _custom_components_kwargs(self) -> Dict[str, Any]:
        custom_components_kwargs = {}

        custom_components_kwargs["scheduler"] = (
            self.scheduler
            if not _is_function_or_method(
                self.scheduler,
            )
            else self.scheduler()  # get scheduler instance
        )
        custom_components_kwargs["transformer"] = (
            self.transformer
            if not _is_function_or_method(
                self.transformer,
            )
            else self.transformer()  # get transformer instance
        )
        custom_components_kwargs["vae"] = (
            self.vae
            if not _is_function_or_method(
                self.vae,
            )
            else self.vae()  # get vae instance
        )
        custom_components_kwargs["text_encoder"] = (
            self.text_encoder
            if not _is_function_or_method(
                self.text_encoder,
            )
            else self.text_encoder()  # get text_encoder instance
        )
        custom_components_kwargs["controlnet"] = (
            self.controlnet
            if not _is_function_or_method(
                self.controlnet,
            )
            else self.controlnet()  # get controlnet instance
        )
        # Remove None components
        custom_components_kwargs = {
            k: v for k, v in custom_components_kwargs.items() if v is not None
        }
        return custom_components_kwargs

    @property
    def has_lora(self) -> bool:
        return (
            self.lora_weights_path is not None
            and os.path.exists(self.lora_weights_path)
            and self.lora_weights_name is not None
            and os.path.exists(os.path.join(self.lora_weights_path, self.lora_weights_name))
        )

    def _pipeline_quantization_config(
        self, args: argparse.Namespace
    ) -> Optional[PipelineQuantizationConfig]:
        if self.bnb_4bit_components is None or len(self.bnb_4bit_components) == 0:
            return None
        return pipe_quant_bnb_4bit_config(
            args=args,
            components_to_quantize=self.bnb_4bit_components,
        )


def _is_function_or_method(component: Any) -> bool:
    func_types = (
        types.FunctionType,
        types.BuiltinFunctionType,
        types.LambdaType,
    )
    excluded_module_classes = (
        SchedulerMixin,
        ModelMixin,
        GenerationMixin,
        torch.nn.Module,
    )

    is_basic_func = isinstance(component, func_types)
    is_excluded_instance = isinstance(component, excluded_module_classes)
    is_method = isinstance(
        component,
        (
            types.MethodType,
            types.ClassMethodDescriptorType,
        ),
    )
    return is_basic_func and not is_excluded_instance and not is_method


class Example:
    def __init__(
        self,
        args: argparse.Namespace,
        init_config: Optional[ExampleInitConfig] = None,
        input_data: Optional[ExampleInputData] = None,
    ):
        self.args = args
        self.init_config: Optional[ExampleInitConfig] = init_config
        self.input_data: Optional[ExampleInputData] = input_data
        self.output_data: Optional[ExampleOutputData] = None
        self.rank, self.device = maybe_init_distributed(self.args)

    def check_valid(self) -> bool:
        if self.args is None:
            raise ValueError("args must be provided.")
        if self.input_data is None:
            raise ValueError("input_data must be provided.")
        if self.init_config is None:
            raise ValueError("init_config must be provided.")
        return True

    def prepare_input_data(self):
        input_kwargs = self.input_data.data(self.args)
        default_num_inference_steps = input_kwargs.get("num_inference_steps", None)
        extra_optimize_kwargs = self.init_config.extra_optimize_kwargs
        extra_optimize_kwargs["default_num_inference_steps"] = default_num_inference_steps
        return input_kwargs, extra_optimize_kwargs

    def run(self) -> None:
        self.check_valid()
        start_time = time.time()
        pipe = self.init_config.get_pipe(self.args)
        load_time = time.time() - start_time

        input_kwargs, extra_optimize_kwargs = self.prepare_input_data()
        default_num_inference_steps = input_kwargs.get("num_inference_steps", None)

        maybe_apply_optimization(self.args, pipe, **extra_optimize_kwargs)

        pipe.set_progress_bar_config(disable=self.rank != 0)

        # track memory if needed
        memory_tracker = MemoryTracker() if self.args.track_memory else None
        if memory_tracker:
            memory_tracker.__enter__()

        # warm up
        start_time = time.time()
        for _ in range(self.args.warmup):
            input_kwargs = self.new_generator(input_kwargs, self.args)
            if self.args.warmup_num_inference_steps is not None:
                input_kwargs["num_inference_steps"] = self.args.warmup_num_inference_steps
            _ = pipe(**input_kwargs)
        if self.args.warmup > 0:
            warmup_time = (time.time() - start_time) / self.args.warmup
        else:
            warmup_time = None
        # restore num_inference_steps
        input_kwargs["num_inference_steps"] = default_num_inference_steps

        start_time = time.time()
        # actual inference
        model_tag = self.args.example if self.args.example is not None else "None"
        if self.args.profile:
            requested_profile_name = getattr(self.args, "profile_name", None)
            profile_name = requested_profile_name or f"{model_tag}_profile"
            profiler = create_profiler_from_args(self.args, profile_name=profile_name)
            with profiler:
                for _ in range(self.args.repeat):
                    input_kwargs = self.new_generator(input_kwargs, self.args)
                    output = pipe(**input_kwargs)
            if self.rank == 0:
                logger.info(
                    f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}"
                )
        else:
            for _ in range(self.args.repeat):
                input_kwargs = self.new_generator(input_kwargs, self.args)
                output = pipe(**input_kwargs)
        if self.args.repeat > 0:
            inference_time = (time.time() - start_time) / self.args.repeat
        else:
            inference_time = None

        if self.args.cache_summary:
            if self.rank == 0:
                summary(pipe)

        if memory_tracker:
            memory_tracker.__exit__(None, None, None)
            peak_gb = memory_tracker.report()
        else:
            peak_gb = None

        # Prepare output data
        output_data = ExampleOutputData(
            model_tag=model_tag,
            strify_tag=f"{strify(self.args, pipe)}",
            load_time=load_time,
            warmup_time=warmup_time,
            inference_time=inference_time,
            memory_usage=peak_gb,
        )

        if self.init_config.task_type in [ExampleType.T2I, ExampleType.IE2I]:
            output_data.image = (
                output.images[0] if isinstance(output.images, list) else output.images
            )
        elif self.init_config.task_type in [
            ExampleType.T2V,
            ExampleType.I2V,
            ExampleType.FLF2V,
            ExampleType.VACE,
        ]:
            output_data.video = output.frames[0] if hasattr(output, "frames") else output

        self.output_data = output_data

        if self.rank == 0:
            logger.info("-" * 100)
            self.init_config.summary(
                self.args,
                # path for extra model, e.g., lora weights, svdq int4 weights, etc.
                extra_model_path=ExampleRegister.get_default(
                    self.args.example,
                ),
            )
            self.input_data.summary(self.args)
            self.output_data.summary(self.args)
            self.output_data.save(self.args)
            logger.info("-" * 100)

        maybe_destroy_distributed()

    def new_generator(
        self, input_kwargs: Dict[str, Any], args: argparse.Namespace
    ) -> torch.Generator:
        # NOTE: We should always create a new generator before each inference to
        # ensure reproducibility when using the same seed.
        input_kwargs["generator"] = self.input_data.new_generator(args=args)
        return input_kwargs


class ExampleRegister:
    _example_registry: Dict[str, Callable[..., Example]] = {}
    _example_registry_defaults: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, default: str = ""):
        def decorator(example_func: Callable[..., Example]):
            if name in cls._example_registry:
                raise ValueError(f"Example '{name}' is already registered.")
            cls._example_registry[name] = example_func
            cls._example_registry_defaults[name] = default
            return example_func

        return decorator

    @classmethod
    def get_example(cls, args: argparse.Namespace, name: str, **kwargs) -> Example:
        if name not in cls._example_registry:
            raise ValueError(f"Example '{name}' is not registered.")
        example_func = cls._example_registry[name]
        return example_func(args, **kwargs)

    @classmethod
    def list_examples(cls) -> List[str]:
        return list(cls._example_registry.keys())

    @classmethod
    def get_default(cls, name: str) -> str:
        return cls._example_registry_defaults.get(name, "")
