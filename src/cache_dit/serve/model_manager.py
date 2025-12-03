"""Model Manager for cache-dit serving.

Adapted from SGLang's model management:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py
"""

import time
import base64
import torch
from io import BytesIO
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from diffusers import DiffusionPipeline
import cache_dit
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclass
class GenerateRequest:
    """Image generation request."""

    prompt: str
    negative_prompt: Optional[str] = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1


@dataclass
class GenerateResponse:
    """Image generation response."""

    images: List[str]  # Base64 encoded images
    stats: Optional[Dict[str, Any]] = None
    time_cost: Optional[float] = None


class ModelManager:
    """Manages diffusion model loading and inference."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        enable_cache: bool = True,
        cache_config: Optional[Dict[str, Any]] = None,
        enable_cpu_offload: bool = False,
        device_map: Optional[str] = None,
        enable_compile: bool = False,
        parallel_type: Optional[str] = None,
        parallel_args: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.enable_cache = enable_cache
        self.cache_config = cache_config or {}
        self.enable_cpu_offload = enable_cpu_offload
        self.device_map = device_map
        self.enable_compile = enable_compile
        self.parallel_type = parallel_type
        self.parallel_args = parallel_args or {}
        self.pipe = None
        self.warmed_up_shapes = set()

        logger.info(
            f"Initializing ModelManager: model_path={model_path}, device={self.device}, parallel_type={parallel_type}"
        )

    def load_model(self):
        """Load the diffusion model."""
        logger.info(f"Loading model: {self.model_path}")

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        cache_config_obj = None
        if self.enable_cache:
            logger.info("Enabling DBCache acceleration")
            from cache_dit import DBCacheConfig
            import os

            # Skip forward pattern check in serving to speed up initialization
            # This is safe for officially supported models
            os.environ["CACHE_DIT_SKIP_PATTERN_CHECK"] = "1"

            cache_config_obj = DBCacheConfig(
                residual_diff_threshold=0.08,
            )
            if self.cache_config:
                for key, value in self.cache_config.items():
                    setattr(cache_config_obj, key, value)

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
                parallelism_config=parallelism_config,
            )

        # Move pipeline to CUDA
        if self.device_map is None and self.device == "cuda":
            logger.info("Moving pipeline to CUDA")
            self.pipe.to("cuda")

        if self.enable_cpu_offload and torch.cuda.device_count() <= 1:
            logger.info("Enabling CPU offload")
            self.pipe.enable_model_cpu_offload()

        if self.enable_compile:
            logger.info("Enabling torch.compile")
            cache_dit.set_compile_configs()
            self.pipe.transformer = torch.compile(self.pipe.transformer)

        logger.info("Model loaded successfully")

    def _warmup_if_needed(self, width: int, height: int, prompt: str):
        shape_key = (width, height)
        if self.enable_compile and shape_key not in self.warmed_up_shapes:
            logger.info(f"Warming up for shape {width}x{height}...")
            try:
                _ = self.pipe(
                    prompt=prompt,
                    height=height,
                    width=width,
                    num_inference_steps=4,
                    guidance_scale=1.0,
                )
                self.warmed_up_shapes.add(shape_key)
                logger.info(f"Warmup completed for {width}x{height}")
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        self._warmup_if_needed(request.width, request.height, request.prompt)

        # If no seed is provided, generate one deterministically
        seed = request.seed
        if seed is None and self.parallel_type == "tp":
            # Use a deterministic seed based on prompt hash to ensure reproducibility
            # All ranks will compute the same hash
            import hashlib

            seed = int(hashlib.md5(request.prompt.encode()).hexdigest()[:8], 16)
            logger.info(f"TP mode: auto-generated seed {seed} from prompt hash")

        logger.info(f"Generating image: prompt='{request.prompt[:50]}...', seed={seed}")

        generator = None
        if seed is not None:
            # IMPORTANT: Always use CPU generator for TP mode
            # GPU generators on different devices produce different random sequences,
            # causing inconsistent results across ranks and blurry images.
            # CPU generator ensures all ranks use the same random sequence.
            generator = torch.Generator(device="cpu").manual_seed(seed)
            logger.debug(f"Created generator with seed {seed} on CPU")

        start_time = time.time()

        output = self.pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            num_images_per_prompt=request.num_images,
        )

        time_cost = time.time() - start_time

        stats = None
        if self.enable_cache:
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

        images_base64 = []
        for image in output.images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images_base64.append(img_str)

        logger.info(f"Image generation completed in {time_cost:.2f}s")

        return GenerateResponse(
            images=images_base64,
            stats=stats,
            time_cost=time_cost,
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
