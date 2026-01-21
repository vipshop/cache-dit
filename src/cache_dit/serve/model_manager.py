"""Model Manager for cache-dit serving.

Adapted from SGLang's model management:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py
"""

import os
import base64
import inspect
import tempfile
import math
import uuid
from datetime import datetime, timezone
import torch
import torch.distributed as dist
import requests
from io import BytesIO
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video
from diffusers.loaders.lora_base import LoraBaseMixin
from PIL import Image
import cache_dit
from cache_dit.logger import init_logger
from diffusers import WanImageToVideoPipeline
from ..platforms import current_platform
from .utils import prepare_extra_parallel_modules
from .cache_alignment import get_default_params_modifiers

logger = init_logger(__name__)


def load_pipeline_quant_config(pipeline_quant_config_path: str):
    """Load pipeline quantization config from a custom module."""

    from diffusers.quantizers import PipelineQuantizationConfig

    logger.info(f"Loading pipeline quantization config from: {pipeline_quant_config_path}")

    try:
        import importlib.util
        import sys

        # Load the custom module
        spec = importlib.util.spec_from_file_location(
            "pipeline_quant_config", pipeline_quant_config_path
        )
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load module from {pipeline_quant_config_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        # Get the pipeline quantization config from the module
        if not hasattr(module, "get_pipeline_quant_config"):
            raise ValueError(
                f"Module {pipeline_quant_config_path} must have a 'get_pipeline_quant_config()' function"
            )

        quantization_config = module.get_pipeline_quant_config()

        if not isinstance(quantization_config, PipelineQuantizationConfig):
            raise ValueError(
                f"get_pipeline_quant_config() must return a PipelineQuantizationConfig object, "
                f"got {type(quantization_config)}"
            )

        logger.info("Successfully loaded quantization config from custom module")
        return quantization_config

    except Exception as e:
        logger.error(f"Failed to load quantization config from {pipeline_quant_config_path}: {e}")
        raise


@dataclass
class GenerateRequest:
    """Image/Video generation request."""

    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    sigmas: Optional[List[float]] = None
    seed: Optional[int] = None
    num_images: int = 1
    image_urls: Optional[List[str]] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = 16
    include_stats: bool = False
    output_format: str = "base64"
    output_dir: Optional[str] = None

    def __repr__(self):
        image_urls_repr = None
        if self.image_urls:
            image_urls_repr = [
                f"<data:{len(url)} chars>" if len(url) > 100 else url for url in self.image_urls
            ]
        return (
            f"GenerateRequest(prompt={self.prompt[:50]!r}..., "
            f"width={self.width}, height={self.height}, "
            f"num_inference_steps={self.num_inference_steps}, "
            f"guidance_scale={self.guidance_scale}, seed={self.seed}, "
            f"num_images={self.num_images}, image_urls={image_urls_repr})"
        )


@dataclass
class GenerateResponse:
    """Image/Video generation response."""

    images: Optional[List[str]] = None  # Base64 encoded images or file paths
    video: Optional[str] = None  # Base64 encoded video (mp4) or file path
    stats: Optional[Dict[str, Any]] = None
    time_cost: Optional[float] = None
    inference_start_time: Optional[str] = None
    inference_end_time: Optional[str] = None


class ModelManager:
    """Manages diffusion model loading and inference."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        generator_device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        enable_cache: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: bool = False,
        device_map: Optional[str] = None,
        enable_compile: bool = False,
        parallel_type: Optional[str] = None,
        parallel_args: Optional[Dict[str, Any]] = None,
        attn_backend: Optional[str] = None,
        quantize: bool = False,
        quantize_type: Optional[str] = None,
        pipeline_quant_config_path: Optional[str] = None,
        lora_path: Optional[str] = None,
        lora_name: Optional[str] = None,
        fuse_lora: bool = True,
    ):
        self.model_path = model_path
        self.device = device or (
            current_platform.device_type if current_platform.is_accelerator_available() else "cpu"
        )
        self.generator_device = generator_device
        self.torch_dtype = torch_dtype
        self.enable_cache = enable_cache
        self.cache_config = cache_config or {}
        self.enable_cpu_offload = enable_cpu_offload
        self.device_map = device_map
        self.enable_compile = enable_compile
        self.parallel_type = parallel_type
        self.parallel_args = parallel_args or {}
        self.attn_backend = attn_backend
        self.quantize = quantize
        self.quantize_type = quantize_type
        self.pipeline_quant_config_path = pipeline_quant_config_path
        self.lora_path = lora_path
        self.lora_name = lora_name
        self.fuse_lora = fuse_lora
        self.pipe = None
        self.warmed_up_shapes = set()

        logger.info(
            f"Initializing ModelManager: model_path={model_path}, device={self.device}, "
            f"parallel_type={parallel_type}, attn_backend={attn_backend}"
        )

    def startup_warmup(self, resolutions: List[tuple[int, int]], prompt: str):
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        for width, height in resolutions:
            shape_key = (width, height)
            if shape_key in self.warmed_up_shapes:
                continue

            if self.parallel_type in ["tp", "ulysses", "ring"]:
                dist.barrier()

            logger.info(f"Startup warming up for shape {width}x{height}...")
            _ = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=1,
            )
            self.warmed_up_shapes.add(shape_key)
            logger.info(f"Startup warmup completed for {width}x{height}")

            if self.parallel_type in ["tp", "ulysses", "ring"]:
                dist.barrier()

    def load_model(self):
        """Load the diffusion model."""
        logger.info(f"Loading model: {self.model_path}")

        # Load pipeline quantization config
        quantization_config = None
        components_quantized_by_diffusers: set[str] = set()
        if self.quantize and self.pipeline_quant_config_path:
            quantization_config = load_pipeline_quant_config(self.pipeline_quant_config_path)
            components_quantized_by_diffusers = set(
                getattr(quantization_config, "components_to_quantize", []) or []
            )
        elif self.quantize:
            logger.warning("Quantization enabled but no pipeline_quant_config_path provided")

        # Will we quantize transformer via cache-dit(torchao) after parallelism is applied?
        # NOTE: This is different from diffusers' PipelineQuantizationConfig (e.g., bitsandbytes_4bit).
        will_torchao_quantize_transformer = (
            self.quantize
            and (self.quantize_type is not None)
            and (self.quantize_type not in ("bitsandbytes_4bit",))
            and ("transformer" not in components_quantized_by_diffusers)
            and ("transformer_2" not in components_quantized_by_diffusers)
        )

        if "Wan2.2-I2V-A14B-Diffusers" in self.model_path:
            logger.info("Detected Wan2.2-I2V model, using WanImageToVideoPipeline")
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map,
                quantization_config=quantization_config,
            )
        else:
            if "LTX-2" in self.model_path:
                ltx2_pipeline = os.environ.get("CACHE_DIT_LTX2_PIPELINE", "t2v").strip().lower()
                if ltx2_pipeline in ("t2v", "text2video", "text", "default"):
                    from diffusers import LTX2Pipeline

                    logger.info("Detected LTX-2 model, using LTX2Pipeline (text-to-video)")
                    self.pipe = LTX2Pipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device_map,
                        quantization_config=quantization_config,
                    )
                elif ltx2_pipeline in ("i2v", "image2video", "image"):
                    from diffusers import LTX2ImageToVideoPipeline

                    logger.info(
                        "Detected LTX-2 model, using LTX2ImageToVideoPipeline (image-to-video)"
                    )
                    self.pipe = LTX2ImageToVideoPipeline.from_pretrained(
                        self.model_path,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device_map,
                        quantization_config=quantization_config,
                    )
                else:
                    raise ValueError(
                        "Invalid CACHE_DIT_LTX2_PIPELINE. Please set it to 't2v' or 'i2v'."
                    )
            else:
                self.pipe = DiffusionPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device_map,
                    quantization_config=quantization_config,
                )

        if self.lora_path is not None and self.lora_name is not None:
            if not isinstance(self.pipe, LoraBaseMixin):
                logger.error("Pipeline does not support LoRA. Skipping LoRA loading.")
            else:
                logger.info(f"Loading LoRA weights from: {self.lora_path}/{self.lora_name}")
                self.pipe.load_lora_weights(self.lora_path, weight_name=self.lora_name)
                logger.info("LoRA weights loaded successfully")

                if "qwen" in self.lora_name.lower() and "light" in self.lora_name.lower():
                    logger.info("Detected Qwen-Image-Lightning LoRA, updating scheduler...")
                    scheduler_config = {
                        "base_image_seq_len": 256,
                        "base_shift": math.log(3),
                        "invert_sigmas": False,
                        "max_image_seq_len": 8192,
                        "max_shift": math.log(3),
                        "num_train_timesteps": 1000,
                        "shift": 1.0,
                        "shift_terminal": None,
                        "stochastic_sampling": False,
                        "time_shift_type": "exponential",
                        "use_beta_sigmas": False,
                        "use_dynamic_shifting": True,
                        "use_exponential_sigmas": False,
                        "use_karras_sigmas": False,
                    }
                    self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                        scheduler_config
                    )
                    logger.info("Scheduler updated for Lightning model")

                # If transformer will be quantized (either by diffusers quantization_config
                # or by cache-dit/torchao quantization), do NOT fuse LoRA into transformer.
                transformer_quantized_or_will_be = (
                    ("transformer" in components_quantized_by_diffusers)
                    or ("transformer_2" in components_quantized_by_diffusers)
                    or will_torchao_quantize_transformer
                )
                should_fuse = self.fuse_lora and (not transformer_quantized_or_will_be)

                if should_fuse:
                    logger.info("Fusing LoRA weights into transformer...")
                    self.pipe.fuse_lora()
                    self.pipe.unload_lora_weights()
                    logger.info("LoRA weights fused and unloaded successfully")
                else:
                    logger.info(
                        "Keeping LoRA weights separate (fusion disabled or transformer quantized)"
                    )
        elif self.lora_path is not None or self.lora_name is not None:
            logger.warning("Both --lora-path and --lora-name must be provided to load LoRA weights")

        cache_config_obj = None
        if self.enable_cache:
            logger.info("Enabling DBCache acceleration")
            from cache_dit import DBCacheConfig

            cache_config_obj = DBCacheConfig(
                residual_diff_threshold=0.24,
            )
            if self.cache_config:
                for key, value in self.cache_config.items():
                    setattr(cache_config_obj, key, value)

        params_modifiers = None
        if self.enable_cache and cache_config_obj is not None:
            params_modifiers = get_default_params_modifiers(
                pipe=self.pipe,
                model_path=self.model_path,
                cache_config_obj=cache_config_obj,
            )

        parallelism_config = None
        if self.parallel_type is not None:
            logger.info(
                f"Enabling parallelism: type={self.parallel_type}, args={self.parallel_args}"
            )
            from cache_dit import ParallelismConfig
            from cache_dit.parallelism import ParallelismBackend
            import torch.distributed as dist

            world_size = dist.get_world_size() if dist.is_initialized() else 1

            backend = (
                ParallelismBackend.NATIVE_PYTORCH
                if self.parallel_type == "tp"
                else ParallelismBackend.NATIVE_DIFFUSER
            )

            # Build extra_parallel_modules for text encoder and vae
            parallel_text_encoder = self.parallel_args.pop("parallel_text_encoder", False)
            parallel_vae = self.parallel_args.pop("parallel_vae", False)
            extra_parallel_modules = prepare_extra_parallel_modules(
                self.pipe,
                parallel_text_encoder=parallel_text_encoder,
                parallel_vae=parallel_vae,
            )
            self.parallel_args["extra_parallel_modules"] = extra_parallel_modules

            parallelism_config = ParallelismConfig(
                backend=backend,
                ulysses_size=world_size if self.parallel_type == "ulysses" else None,
                ring_size=world_size if self.parallel_type == "ring" else None,
                tp_size=world_size if self.parallel_type == "tp" else None,
                parallel_kwargs=self.parallel_args,
            )

        if cache_config_obj is not None or parallelism_config is not None:
            cache_dit.enable_cache(
                self.pipe,
                cache_config=cache_config_obj,
                params_modifiers=params_modifiers,
                parallelism_config=parallelism_config,
            )

        # Quantize transformer by quantize_type (torchao backend).
        # WARN: Must apply torchao quantization after tensor/context parallelism is applied.
        if self.quantize and self.quantize_type is not None:
            if self.quantize_type in ("bitsandbytes_4bit",):
                if quantization_config is None:
                    logger.warning(
                        "Requested bitsandbytes_4bit quantization but no "
                        "--pipeline-quant-config-path provided. "
                        "Please provide a PipelineQuantizationConfig that sets "
                        "quant_backend='bitsandbytes_4bit'."
                    )
            else:
                if ("transformer" in components_quantized_by_diffusers) or (
                    "transformer_2" in components_quantized_by_diffusers
                ):
                    logger.warning(
                        "Transformer is already quantized by diffusers PipelineQuantizationConfig; "
                        f"skipping cache-dit(torchao) quantize_type={self.quantize_type}."
                    )
                else:
                    # Mirror logic from examples: some models do not support per-row quantization.
                    class_not_supported_per_row = {
                        "QwenImageTransformer2DModel",
                    }

                    def is_per_row_supported(m: torch.nn.Module) -> bool:
                        return m.__class__.__name__ not in class_not_supported_per_row

                    if hasattr(self.pipe, "transformer"):
                        transformer = getattr(self.pipe, "transformer", None)
                        if isinstance(transformer, torch.nn.Module):
                            logger.info(
                                f"Quantizing transformer module: {transformer.__class__.__name__} "
                                f"to {self.quantize_type} (torchao) ..."
                            )
                            setattr(
                                self.pipe,
                                "transformer",
                                cache_dit.quantize(
                                    transformer,
                                    quant_type=self.quantize_type,
                                    per_row=is_per_row_supported(transformer),
                                ),
                            )
                        elif transformer is not None:
                            logger.warning(
                                "Cannot quantize transformer: it is not a torch.nn.Module "
                                f"(got {type(transformer)})."
                            )

                    if hasattr(self.pipe, "transformer_2"):
                        transformer_2 = getattr(self.pipe, "transformer_2", None)
                        if isinstance(transformer_2, torch.nn.Module):
                            logger.info(
                                f"Quantizing transformer_2 module: {transformer_2.__class__.__name__} "
                                f"to {self.quantize_type} (torchao) ..."
                            )
                            setattr(
                                self.pipe,
                                "transformer_2",
                                cache_dit.quantize(
                                    transformer_2,
                                    quant_type=self.quantize_type,
                                    per_row=is_per_row_supported(transformer_2),
                                ),
                            )
                        elif transformer_2 is not None:
                            logger.warning(
                                "Cannot quantize transformer_2: it is not a torch.nn.Module "
                                f"(got {type(transformer_2)})."
                            )

        # Move pipeline to device
        if self.device_map is None and self.device == current_platform.device_type:
            logger.info(f"Moving pipeline to {current_platform.device_type}")
            self.pipe.to(self.device)

        if self.enable_cpu_offload and current_platform.device_count() <= 1:
            logger.info("Enabling CPU offload")
            self.pipe.enable_model_cpu_offload()

        if self.attn_backend is not None:
            if hasattr(self.pipe.transformer, "set_attention_backend"):
                logger.info(f"Setting attention backend to {self.attn_backend}")
                self.pipe.transformer.set_attention_backend(self.attn_backend)
            else:
                logger.warning(
                    f"Transformer does not support set_attention_backend, ignoring --attn {self.attn_backend}"
                )

        if self.enable_compile:
            logger.info("Enabling torch.compile")
            cache_dit.set_compile_configs()
            self.pipe.transformer = torch.compile(self.pipe.transformer)

        logger.info("Model loaded successfully")

    def _warmup_if_needed(self, width: int, height: int, prompt: str):
        shape_key = (width, height)
        if self.enable_compile and shape_key not in self.warmed_up_shapes:
            if self.parallel_type in ["tp", "ulysses", "ring"]:
                dist.barrier()

            logger.info(f"Warming up for shape {width}x{height}...")
            try:
                _ = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=1,
                )
                self.warmed_up_shapes.add(shape_key)
                logger.info(f"Warmup completed for {width}x{height}")
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

            if self.parallel_type in ["tp", "ulysses", "ring"]:
                dist.barrier()

    def _load_images_from_urls(self, image_urls: List[str]) -> Optional[List[Image.Image]]:
        """Load images from URLs, local paths, or base64 strings."""
        if not image_urls:
            return None

        images = []
        for idx, url in enumerate(image_urls):
            try:
                if url.startswith("data:image/"):
                    log_desc = f"data URI (length: {len(url)})"
                    logger.info(f"Loading image {idx + 1} from {log_desc}")
                    header, base64_data = url.split(",", 1)
                    img_data = base64.b64decode(base64_data)
                    image = Image.open(BytesIO(img_data)).convert("RGB")
                elif url.startswith(("http://", "https://")):
                    log_desc = f"URL: {url[:80]}{'...' if len(url) > 80 else ''}"
                    logger.info(f"Downloading image {idx + 1} from {log_desc}")
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                elif len(url) > 100:
                    log_desc = f"raw base64 string (length: {len(url)})"
                    logger.info(f"Loading image {idx + 1} from {log_desc}")
                    try:
                        img_data = base64.b64decode(url, validate=True)
                        image = Image.open(BytesIO(img_data)).convert("RGB")
                    except Exception:
                        raise
                else:
                    log_desc = f"local path: {url}"
                    logger.info(f"Loading image {idx + 1} from {log_desc}")
                    image = Image.open(url).convert("RGB")
                images.append(image)
                logger.info(f"Image {idx + 1} loaded successfully: {image.size}")
            except Exception as e:
                if len(url) > 100:
                    error_url = f"<data of length {len(url)}>"
                else:
                    error_url = url
                logger.error(f"Failed to load image {idx + 1} from {error_url}: {e}")
                raise RuntimeError(f"Failed to load image {idx + 1}: {e}")

        return images

    def _resolve_output_dir(self, output_dir: Optional[str]) -> str:
        if output_dir is not None:
            return os.path.abspath(output_dir)
        return os.path.join(os.getcwd(), "outputs")

    def _save_image_to_dir(self, image: Image.Image, output_dir: str, name: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        image.save(path, format="PNG")
        return os.path.abspath(path)

    def _save_video_to_dir(self, video_frames, output_dir: str, name: str, fps: int) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, name)
        export_to_video(video_frames, path, fps=fps)
        return os.path.abspath(path)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if request.output_format not in ("base64", "path"):
            raise ValueError(
                f"Invalid output_format: {request.output_format}. Must be 'base64' or 'path'."
            )

        is_edit_mode = request.image_urls is not None and len(request.image_urls) > 0
        is_video_mode = request.num_frames is not None and request.num_frames > 1
        is_image2video_mode = is_edit_mode and is_video_mode
        input_images = None
        if is_edit_mode:
            input_images = self._load_images_from_urls(request.image_urls)
            if input_images:
                logger.info(
                    f"Loaded {len(input_images)} input image(s) for {'image2video' if is_image2video_mode else 'editing'}"
                )

        if not is_edit_mode and not is_video_mode:
            self._warmup_if_needed(request.width, request.height, request.prompt)

        seed = request.seed
        if seed is None and self.parallel_type in ["tp", "ulysses", "ring"]:
            seed = 0
            logger.info(f"{self.parallel_type} mode: using fixed seed {seed}")

        if is_image2video_mode:
            mode_str = "image2video"
        elif is_video_mode:
            mode_str = "video generation"
        elif is_edit_mode:
            mode_str = "edit"
        else:
            mode_str = "generation"
        logger.info(f"{mode_str}: prompt='{request.prompt[:50]}...', seed={seed}")

        generator = None
        if seed is not None:
            gen_device = self.generator_device
            if gen_device is None:
                gen_device = (
                    current_platform.device_type
                    if current_platform.is_accelerator_available()
                    else "cpu"
                )
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            logger.debug(f"Created generator with seed {seed} on {gen_device}")

        if self.parallel_type in ["tp", "ulysses", "ring"]:
            import torch.distributed as dist

            dist.barrier()

        start_dt_raw = datetime.now(timezone.utc)

        pipe_to_use = self.pipe

        if is_image2video_mode:
            try:
                sig = inspect.signature(pipe_to_use.__call__)
                accepts_image = "image" in sig.parameters
            except Exception:
                accepts_image = True
            if not accepts_image:
                raise ValueError(
                    "Current LTX-2 pipeline does not support image2video. "
                    "Please restart server with CACHE_DIT_LTX2_PIPELINE=i2v."
                )

        # Build kwargs for pipe call
        pipe_kwargs = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
        }

        if request.sigmas is not None:
            try:
                sig = inspect.signature(self.pipe.__call__)
                if "sigmas" in sig.parameters:
                    pipe_kwargs["sigmas"] = request.sigmas
                else:
                    logger.warning("Pipeline does not support sigmas, ignoring request.sigmas")
            except Exception:
                pipe_kwargs["sigmas"] = request.sigmas

        # Add num_frames for video generation
        if is_video_mode:
            pipe_kwargs["num_frames"] = request.num_frames
            # For some video pipelines (e.g. LTX2), `frame_rate` is an input condition.
            # We unify it with request.fps to avoid redundant parameters.
            try:
                sig = inspect.signature(pipe_to_use.__call__)
                if "frame_rate" in sig.parameters:
                    pipe_kwargs["frame_rate"] = (
                        float(request.fps) if request.fps is not None else 24.0
                    )
                # For LTX2 i2v, exporting + audio handling is easier with numpy output
                if "output_type" in sig.parameters:
                    pipe_kwargs["output_type"] = "np"
                if "return_dict" in sig.parameters:
                    pipe_kwargs["return_dict"] = True
            except Exception:
                pipe_kwargs["frame_rate"] = float(request.fps) if request.fps is not None else 24.0
        else:
            pipe_kwargs["num_images_per_prompt"] = request.num_images

        # Add input images to pipe_kwargs if in edit mode or image2video mode
        if is_edit_mode and input_images:
            # For image2video, always use single image (first one if multiple provided)
            if is_image2video_mode:
                pipe_kwargs["image"] = input_images[0]
                logger.info(f"Using first image for image2video: {input_images[0].size}")
            elif len(input_images) == 1:
                pipe_kwargs["image"] = input_images[0]
            else:
                pipe_kwargs["image"] = input_images

        # Some pipelines (like Flux2Pipeline) don't support negative_prompt
        if request.negative_prompt:
            try:
                sig = inspect.signature(self.pipe.__call__)
                if "negative_prompt" in sig.parameters:
                    pipe_kwargs["negative_prompt"] = request.negative_prompt
            except Exception:
                # If we can't inspect, try to add it anyway
                pipe_kwargs["negative_prompt"] = request.negative_prompt

        output = pipe_to_use(**pipe_kwargs)

        if self.parallel_type in ["tp", "ulysses", "ring"]:
            import torch.distributed as dist

            dist.barrier()

        end_dt_raw = datetime.now(timezone.utc)

        start_dt = start_dt_raw.replace(microsecond=(start_dt_raw.microsecond // 1000) * 1000)
        end_dt = end_dt_raw.replace(microsecond=(end_dt_raw.microsecond // 1000) * 1000)

        time_cost = (end_dt - start_dt).total_seconds()

        inference_start_time = start_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
        inference_end_time = end_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        is_primary_rank = True
        if (
            self.parallel_type in ["tp", "ulysses", "ring"]
            and dist.is_available()
            and dist.is_initialized()
        ):
            try:
                is_primary_rank = dist.get_rank() == 0
            except Exception:
                is_primary_rank = True

        # Debug: Check output shape in distributed mode
        if self.parallel_type is not None:
            import torch.distributed as dist

            rank = dist.get_rank()
            if is_video_mode:
                logger.info(f"Rank {rank}: Generated video with {len(output.frames[0])} frames")
            else:
                logger.info(f"Rank {rank}: Generated {len(output.images)} images")

        stats = None
        if is_primary_rank and request.include_stats and self.enable_cache:
            stats_list = cache_dit.summary(self.pipe)
            # Convert List[CacheStats] to dict for JSON serialization
            if stats_list:
                stats = {
                    "cache_stats": [
                        {
                            "cache_options": str(s.cache_options) if s.cache_options else None,
                            "cached_steps": list(s.cached_steps) if s.cached_steps else [],
                            "parallelism_config": (
                                str(s.parallelism_config) if s.parallelism_config else None
                            ),
                        }
                        for s in stats_list
                    ]
                }

        images_payload = None
        video_payload = None

        if not is_primary_rank:
            return GenerateResponse(
                images=None,
                video=None,
                stats=None,
                time_cost=time_cost,
                inference_start_time=inference_start_time,
                inference_end_time=inference_end_time,
            )

        if is_video_mode:
            video_frames = output.frames[0]
            logger.info(
                f"Video generation completed with {len(video_frames)} frames in {time_cost:.2f}s"
            )

            if request.output_format == "path":
                out_dir = self._resolve_output_dir(request.output_dir)
                # If pipeline returns audio (LTX2), export mp4 with audio track.
                audio = getattr(output, "audio", None)
                if audio is not None:
                    try:
                        from diffusers.pipelines.ltx2.export_utils import encode_video

                        video_np = video_frames
                        # video_np: (T, H, W, C) float in [0,1]
                        video_uint8 = (video_np * 255).round().astype("uint8")
                        video_t = torch.from_numpy(video_uint8)
                        audio_t = audio[0]
                        if not isinstance(audio_t, torch.Tensor):
                            audio_t = torch.from_numpy(audio_t)
                        audio_t = audio_t.float().cpu()
                        sample_rate = getattr(getattr(pipe_to_use, "vocoder", None), "config", None)
                        sample_rate = getattr(sample_rate, "output_sampling_rate", 24000)

                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.abspath(
                            os.path.join(out_dir, f"video_{uuid.uuid4().hex}.mp4")
                        )
                        encode_video(
                            video_t,
                            fps=float(request.fps),
                            audio=audio_t,
                            audio_sample_rate=sample_rate,
                            output_path=out_path,
                        )
                        video_payload = out_path
                    except Exception as e:
                        logger.warning(
                            f"encode_video(with audio) failed ({type(e).__name__}: {e}), "
                            "falling back to export_to_video(video-only)."
                        )
                        video_payload = self._save_video_to_dir(
                            video_frames,
                            out_dir,
                            name=f"video_{uuid.uuid4().hex}.mp4",
                            fps=request.fps,
                        )
                else:
                    video_payload = self._save_video_to_dir(
                        video_frames,
                        out_dir,
                        name=f"video_{uuid.uuid4().hex}.mp4",
                        fps=request.fps,
                    )
            else:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
                    tmp_path = tmp_file.name

                try:
                    audio = getattr(output, "audio", None)
                    if audio is not None:
                        from diffusers.pipelines.ltx2.export_utils import encode_video

                        video_np = video_frames
                        video_uint8 = (video_np * 255).round().astype("uint8")
                        video_t = torch.from_numpy(video_uint8)
                        audio_t = audio[0]
                        if not isinstance(audio_t, torch.Tensor):
                            audio_t = torch.from_numpy(audio_t)
                        audio_t = audio_t.float().cpu()
                        sample_rate = getattr(getattr(pipe_to_use, "vocoder", None), "config", None)
                        sample_rate = getattr(sample_rate, "output_sampling_rate", 24000)

                        encode_video(
                            video_t,
                            fps=float(request.fps),
                            audio=audio_t,
                            audio_sample_rate=sample_rate,
                            output_path=tmp_path,
                        )
                    else:
                        export_to_video(video_frames, tmp_path, fps=request.fps)

                    with open(tmp_path, "rb") as f:
                        video_bytes = f.read()
                        video_payload = base64.b64encode(video_bytes).decode()
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
        else:
            images_payload = []
            if request.output_format == "path":
                out_dir = self._resolve_output_dir(request.output_dir)
                for idx, image in enumerate(output.images):
                    images_payload.append(
                        self._save_image_to_dir(
                            image,
                            out_dir,
                            name=f"image_{uuid.uuid4().hex}_{idx}.png",
                        )
                    )
            else:
                for image in output.images:
                    buffered = BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    images_payload.append(img_str)

            logger.info(f"Image generation completed in {time_cost:.2f}s")

        return GenerateResponse(
            images=images_payload,
            video=video_payload,
            stats=stats,
            time_cost=time_cost,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "enable_cache": self.enable_cache,
            "is_loaded": self.pipe is not None,
        }
