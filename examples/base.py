import os
import time

import torch
import argparse
import dataclasses
from PIL import Image
from enum import Enum
from typing import Dict, Any, Union, Optional, List, Callable
from diffusers.utils import export_to_video
from diffusers.schedulers import SchedulerMixin
from diffusers import DiffusionPipeline, ModelMixin
from transformers import GenerationMixin
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.quantizers import PipelineQuantizationConfig
from cache_dit.logger import init_logger
import cache_dit

from utils import (
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
    T2V = "text_to_video"
    I2V = "image_to_video"
    T2I = "text_to_image"
    IE2I = "image_editing_to_image"


@dataclasses.dataclass
class ExampleInputData:
    # General inputs for both image and video generation
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    guidance_scale: Optional[float] = None
    true_cfg_scale: Optional[float] = None
    num_inference_steps: Optional[int] = None
    num_images_per_prompt: Optional[int] = None
    # Specific inputs for image editing
    image: Optional[Union[List[Image.Image], Image.Image]] = None
    mask_image: Optional[Union[List[Image.Image], Image.Image]] = None
    # Specific inputs for video generation
    num_frames: Optional[int] = None
    # Other inputs
    seed: Optional[int] = None
    generator: torch.Generator = torch.Generator("cpu").manual_seed(0)
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
            input_data["image"] = Image.open(args.image_path).convert("RGB")
        if args.mask_image_path is not None:
            if "mask_image" in input_data:
                if isinstance(input_data["mask_image"], list):
                    if len(input_data["mask_image"]) > 1:
                        logger.warning(
                            "Overriding multiple input mask images with a single mask "
                            "image from args.mask_image_path."
                        )
            input_data["mask_image"] = Image.open(args.mask_image_path).convert("RGB")
        if self.seed is not None:
            input_data["generator"] = torch.Generator("cpu").manual_seed(self.seed)
        # Maybe override generator with args.seed
        if args.seed is not None:
            input_data["generator"] = torch.Generator("cpu").manual_seed(args.seed)
        return input_data

    def new_generator(self, seed: int = None) -> torch.Generator:
        # NOTE: We should always create a new generator before each inference to
        # ensure reproducibility when using the same seed. Alawys use cpu generator
        # for better cross-device consistency.
        if seed is not None:
            return torch.Generator("cpu").manual_seed(seed)
        elif self.seed is not None:
            return torch.Generator("cpu").manual_seed(self.seed)
        else:
            return torch.Generator("cpu").manual_seed(0)

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
        summary_str = "Example Input Summary:\n"
        data = self.data(args)
        for k, v in data.items():
            if k in ["prompt", "negative_prompt"]:
                summary_str += f"- {k}: {v}\n"
            elif k in ["height", "width", "num_inference_steps", "num_frames"]:
                summary_str += f"- {k}: {v}\n"
            elif k in ["image", "mask_image"]:
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
            elif k == "generator":
                # Show seed and device info
                gen_device = v.device if hasattr(v, "device") else "cpu"
                gen_seed = v.initial_seed() if hasattr(v, "initial_seed") else "N/A"
                summary_str += f"- {k}: device {gen_device}, seed {gen_seed}\n"
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
    image: Optional[Image.Image] = None  # Single PIL Images
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
            self.image.save(save_path)
            logger.info(f"Image saved to {save_path}")

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
                return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.png"
            else:
                return f"{self.model_tag}.{self.strify_tag}.png"
        elif self.video is not None:
            try:
                W, H = self.video[0].size
                num_frames = len(self.video)
                HxW_str = f"{H}x{W}x{num_frames}"
            except Exception:
                HxW_str = None
            if HxW_str is not None:
                return f"{self.model_tag}.{HxW_str}.{self.strify_tag}.mp4"
            else:
                return f"{self.model_tag}.{self.strify_tag}.mp4"
        else:
            return None

    def summary(self, args: argparse.Namespace) -> str:
        logger.info("Example Output Summary:")
        summary_str = f"- Model: {args.example}\n- Optimization: {self.strify_tag}\n"
        if self.load_time is not None:
            summary_str += f"- Load Time: {self.load_time:.2f}s\n"
        if self.warmup_time is not None:
            summary_str += f"- Warmup Mean Time: {self.warmup_time:.2f}s\n"
        if self.inference_time is not None:
            summary_str += f"- Inference Mean Time: {self.inference_time:.2f}s\n"
        if self.memory_usage is not None:
            summary_str += f"- Memory Usage: {self.memory_usage:.2f}GiB\n"
        summary_str = summary_str.rstrip("\n")
        logger.info(summary_str)
        return summary_str


@dataclasses.dataclass
class ExampleInitConfig:
    task_type: ExampleType
    model_name_or_path: str
    pipeline_class: Optional[type[DiffusionPipeline]] = DiffusionPipeline
    torch_dtype: Optional[torch.dtype] = torch.bfloat16
    bnb_4bit_components: Optional[List[str]] = dataclasses.field(default_factory=list)
    scheduler: Optional[Union[SchedulerMixin, Callable]] = None  # lora case
    transformer: Optional[Union[ModelMixin, Callable]] = None  # lora or nunchaku case
    vae: Optional[Union[ModelMixin, Callable]] = None
    text_encoder: Optional[Union[GenerationMixin, Callable]] = None
    lora_weights_path: Optional[str] = None
    lora_weights_name: Optional[str] = None
    pre_init_hook: Optional[Callable[[Any], None]] = None  # For future use
    post_init_hook: Optional[Callable[[DiffusionPipeline], None]] = None
    # For DBCache, Parallelism optimization.
    extra_optimize_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        if not self.bnb_4bit_components:
            self.bnb_4bit_components = ["text_encoder"]

    def get_pipe(self, args: argparse.Namespace, **kwargs) -> DiffusionPipeline:
        if self.pipeline_class is None:
            raise ValueError("pipeline_class must be provided to get the pipeline instance.")
        pipeline_quantization_config = self._pipeline_quantization_config(args)
        pipe: DiffusionPipeline = self.pipeline_class.from_pretrained(
            self.model_name_or_path if args.model_path is None else args.model_path,
            torch_dtype=self.torch_dtype,
            quantization_config=pipeline_quantization_config,
            **self._custom_components_kwargs(),
        )
        if self.post_init_hook is not None:
            self.post_init_hook(pipe, **kwargs)
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
            if (
                pipeline_quantization_config is None
                or "transformer" not in pipeline_quantization_config.components_to_quantize
            ):
                pipe.fuse_lora()
                pipe.unload_lora_weights()
            else:
                logger.warning("Keep LoRA weights in memory since transformer is quantized.")

        return pipe

    def summary(self, args: argparse.Namespace) -> str:
        logger.info("Example Init Config Summary:")
        model_name_or_path = self.model_name_or_path if args.model_path is None else args.model_path
        summary_str = f"- Model: {model_name_or_path}\n"
        summary_str += f"- Task Type: {self.task_type.value}\n"
        summary_str += f"- Torch Dtype: {self.torch_dtype}\n"
        if self.lora_weights_path is not None and self.lora_weights_name is not None:
            summary_str += (
                f"- LoRA Weights: {os.path.join(self.lora_weights_path, self.lora_weights_name)}\n"
            )
        else:
            summary_str += "- LoRA Weights: None\n"
        summary_str = summary_str.rstrip("\n")
        logger.info(summary_str)
        return summary_str

    def _custom_components_kwargs(self) -> Dict[str, Any]:
        custom_components_kwargs = {}
        custom_components_kwargs["scheduler"] = (
            self.scheduler if not callable(self.scheduler) else self.scheduler()
        )
        custom_components_kwargs["transformer"] = (
            self.transformer if not callable(self.transformer) else self.transformer()
        )
        custom_components_kwargs["vae"] = self.vae if not callable(self.vae) else self.vae()
        custom_components_kwargs["text_encoder"] = (
            self.text_encoder if not callable(self.text_encoder) else self.text_encoder()
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


class CacheDiTExample:
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
        if self.input_data is None:
            raise ValueError("input_data must be provided.")
        if self.init_config is None:
            raise ValueError("init_config must be provided.")
        return True

    def run(self) -> None:
        self.check_valid()
        if self.rank == 0:
            self.init_config.summary(self.args)
        start_time = time.time()
        pipe = self.init_config.get_pipe(self.args)
        load_time = time.time() - start_time

        maybe_apply_optimization(self.args, pipe, **self.init_config.extra_optimize_kwargs)

        input_kwargs = self.input_data.data(self.args)
        pipe.set_progress_bar_config(disable=self.rank != 0)

        # track memory if needed
        memory_tracker = MemoryTracker() if self.args.track_memory else None
        if memory_tracker:
            memory_tracker.__enter__()

        # warm up
        start_time
        for _ in range(self.args.warmup):
            input_kwargs = self.new_generator(input_kwargs, self.args)
            _ = pipe(**input_kwargs)
        warmup_time = (time.time() - start_time) / self.args.warmup

        start_time = time.time()
        # actual inference
        model_tag = self.args.example if self.args.example is not None else "None"
        if self.args.profile:
            profiler = create_profiler_from_args(self.args, profile_name=f"{model_tag}_profile")
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
        inference_time = (time.time() - start_time) / self.args.repeat

        if self.args.cache_summary:
            if self.rank == 0:
                cache_dit.summary(pipe)

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
        elif self.init_config.task_type in [ExampleType.T2V, ExampleType.I2V]:
            output_data.video = output.frames[0] if hasattr(output, "frames") else output

        self.output_data = output_data

        if self.rank == 0:
            self.init_config.summary(self.args)
            self.input_data.summary(self.args)
            self.output_data.summary(self.args)
            self.output_data.save(self.args)

        maybe_destroy_distributed()

    def new_generator(
        self, input_kwargs: Dict[str, Any], args: argparse.Namespace
    ) -> torch.Generator:
        # NOTE: We should always create a new generator before each inference to
        # ensure reproducibility when using the same seed.
        input_kwargs["generator"] = self.input_data.new_generator(seed=args.seed)
        return input_kwargs


class CacheDiTExampleRegister:
    _example_registry: Dict[str, Callable[..., CacheDiTExample]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(example_func: Callable[..., CacheDiTExample]):
            if name in cls._example_registry:
                raise ValueError(f"Example '{name}' is already registered.")
            cls._example_registry[name] = example_func
            return example_func

        return decorator

    @classmethod
    def get_example(cls, args: argparse.Namespace, name: str, **kwargs) -> CacheDiTExample:
        if name not in cls._example_registry:
            raise ValueError(f"Example '{name}' is not registered.")
        example_func = cls._example_registry[name]
        return example_func(args, **kwargs)

    @classmethod
    def list_examples(cls) -> List[str]:
        return list(cls._example_registry.keys())
