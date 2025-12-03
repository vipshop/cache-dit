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
    ):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.enable_cache = enable_cache
        self.cache_config = cache_config or {}
        self.enable_cpu_offload = enable_cpu_offload
        self.device_map = device_map
        self.pipe = None

        logger.info(f"Initializing ModelManager: model_path={model_path}, device={self.device}")

    def load_model(self):
        """Load the diffusion model."""
        logger.info(f"Loading model: {self.model_path}")

        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
        )

        if self.enable_cache:
            logger.info("Enabling DBCache acceleration")
            from cache_dit import DBCacheConfig

            default_cache_config = DBCacheConfig(
                residual_diff_threshold=0.12,
                enable_separate_cfg=True,
            )

            if self.cache_config:
                for key, value in self.cache_config.items():
                    setattr(default_cache_config, key, value)

            cache_dit.enable_cache(
                self.pipe,
                cache_config=default_cache_config,
            )

        if self.enable_cpu_offload and torch.cuda.device_count() <= 1:
            logger.info("Enabling CPU offload")
            self.pipe.enable_model_cpu_offload()

        logger.info("Model loaded successfully")

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Generate images from text prompt."""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Generating image: prompt='{request.prompt[:50]}...'")

        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(request.seed)

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
