"""FastAPI HTTP Server for cache-dit.

Adapted from SGLang's HTTP server:
https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py
"""

import asyncio
from typing import Optional, Dict, Any, List, Literal
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .model_manager import ModelManager, GenerateRequest
from cache_dit.logger import init_logger

logger = init_logger(__name__)

_global_model_manager: Optional[ModelManager] = None
_request_semaphore: Optional[asyncio.Semaphore] = None


class GenerateRequestAPI(BaseModel):
    """API request model for image/video generation."""

    prompt: str = Field(..., description="Text prompt")
    negative_prompt: Optional[str] = Field("", description="Negative prompt")
    width: int = Field(1024, description="Image/Video width", ge=64, le=4096)
    height: int = Field(1024, description="Image/Video height", ge=64, le=4096)
    num_inference_steps: int = Field(50, description="Number of inference steps", ge=1, le=200)
    guidance_scale: float = Field(7.5, description="Guidance scale", ge=0.0, le=20.0)
    sigmas: Optional[List[float]] = Field(
        None,
        description="Custom sigma schedule (e.g. for turbo inference). Length should typically match num_inference_steps.",
    )
    seed: Optional[int] = Field(None, description="Random seed")
    num_images: int = Field(1, description="Number of images to generate", ge=1, le=4)
    image_urls: Optional[List[str]] = Field(
        None,
        description="Input images for image editing. Supports: URLs (http/https), local file paths, base64 strings (with or without data URI prefix)",
    )
    num_frames: Optional[int] = Field(
        None, description="Number of frames for video generation", ge=1, le=200
    )
    fps: Optional[int] = Field(16, description="Frames per second for video output", ge=1, le=60)
    include_stats: bool = Field(False, description="Include stats field in response")
    output_format: Literal["base64", "path"] = Field(
        "base64",
        description="Output format: base64 or path",
    )
    output_dir: Optional[str] = Field(
        None,
        description="Output directory when output_format=path (server-side path)",
    )


class GenerateResponseAPI(BaseModel):
    """API response model for image/video generation."""

    images: Optional[list[str]] = Field(None, description="Base64 encoded images or file paths")
    video: Optional[str] = Field(None, description="Base64 encoded video (mp4) or file path")
    stats: Optional[Dict[str, Any]] = Field(None, description="Cache statistics")
    time_cost: Optional[float] = Field(None, description="Generation time in seconds")
    inference_start_time: Optional[str] = Field(
        None, description="Inference start time (local time with timezone offset)"
    )
    inference_end_time: Optional[str] = Field(
        None, description="Inference end time (local time with timezone offset)"
    )


def create_app(model_manager: ModelManager) -> FastAPI:
    """Create FastAPI application."""
    global _global_model_manager, _request_semaphore
    _global_model_manager = model_manager
    _request_semaphore = asyncio.Semaphore(1)

    app = FastAPI(
        title="Cache-DiT Serving API",
        description="Text-to-image model serving API with cache-dit acceleration",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        if _global_model_manager is None or _global_model_manager.pipe is None:
            return Response(status_code=503, content="Model not loaded")
        return Response(status_code=200, content="OK")

    @app.get("/get_model_info")
    async def get_model_info():
        """Get model information."""
        if _global_model_manager is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")

        return JSONResponse(content=_global_model_manager.get_model_info())

    @app.post("/generate", response_model=GenerateResponseAPI, response_model_exclude_none=True)
    async def generate(request: GenerateRequestAPI):
        """Generate images from text prompt."""
        if _global_model_manager is None:
            raise HTTPException(status_code=503, detail="Model manager not initialized")

        if _global_model_manager.pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        async with _request_semaphore:
            try:
                gen_request = GenerateRequest(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.num_inference_steps,
                    guidance_scale=request.guidance_scale,
                    sigmas=request.sigmas,
                    seed=request.seed,
                    num_images=request.num_images,
                    image_urls=request.image_urls,
                    num_frames=request.num_frames,
                    fps=request.fps,
                    include_stats=request.include_stats,
                    output_format=request.output_format,
                    output_dir=request.output_dir,
                )

                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None, _global_model_manager.generate, gen_request
                )

                return GenerateResponseAPI(
                    images=response.images,
                    video=response.video,
                    stats=response.stats,
                    time_cost=response.time_cost,
                    inference_start_time=response.inference_start_time,
                    inference_end_time=response.inference_end_time,
                )

            except Exception as e:
                logger.error(f"Error generating image: {type(e).__name__}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    @app.post("/flush_cache")
    async def flush_cache():
        """Flush cache."""
        if _global_model_manager is None or _global_model_manager.pipe is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            return JSONResponse(content={"message": "Cache flushed successfully"})
        except Exception as e:
            logger.error(f"Error flushing cache: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to flush cache: {str(e)}")

    return app
