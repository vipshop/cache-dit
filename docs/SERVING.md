# Cache-DiT Serving

HTTP serving for diffusion models with cache-dit acceleration. Supports **text-to-image**, **image editing**, **multi-image editing**, **text-to-video**, and **image-to-video** generation.

Adapted from [SGLang](https://github.com/sgl-project/sglang).

## Quick Start

```bash
pip install -e ".[serving]"

cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache

curl http://localhost:8000/health
```

## Supported Tasks

### Text-to-Image
Generate images from text prompts using models like FLUX, Qwen-Image, CogView3+/4, HunyuanDiT, etc.

### Image Editing
Edit single images with text instructions (e.g., "Put a birthday hat on the dog").

### Multi-Image Editing
Combine multiple images with text prompts (e.g., "Dog chases frisbee" with dog and frisbee images).

### Text-to-Video
Generate videos from text prompts using models like Wan, HunyuanVideo, Mochi, LTX-Video, etc.

### Image-to-Video
Generate videos from input images with text prompts using models like Wan2.2-I2V.

See [tests/serving/](https://github.com/vipshop/cache-dit/tree/main/tests/serving) for detailed examples.

## API Endpoints

- `GET /health` - Health check
- `GET /get_model_info` - Model information
- `POST /generate` - Generate images/videos
- `POST /flush_cache` - Flush cache
- `GET /docs` - API documentation

## Usage Examples

### Text-to-Image

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A beautiful sunset over the ocean",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50
    }
)

img_data = base64.b64decode(response.json()["images"][0])
Image.open(BytesIO(img_data)).save("output.png")
```

### Image Editing

```python
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Put a birthday hat on the dog",
        "image_urls": ["https://example.com/dog.png"],
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 50
    }
)
```

### Text-to-Video

```python
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cat walks on the grass, realistic",
        "width": 832,
        "height": 480,
        "num_frames": 49,
        "fps": 16,
        "num_inference_steps": 30
    }
)

video_data = base64.b64decode(response.json()["video"])
with open("output.mp4", "wb") as f:
    f.write(video_data)
```

### Image-to-Video

```python
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cat on a surfboard at the beach, summer vacation style",
        "image_urls": ["https://example.com/cat.jpg"],
        "width": 832,
        "height": 480,
        "num_frames": 49,
        "fps": 16,
        "guidance_scale": 3.5,
        "num_inference_steps": 50
    }
)

video_data = base64.b64decode(response.json()["video"])
with open("output.mp4", "wb") as f:
    f.write(video_data)
```

## Key Arguments

### Server
- `--model-path` - Model path (required)
- `--host` - Server host (default: 0.0.0.0)
- `--port` - Server port (default: 8000)
- `--device` - Device (default: cuda)
- `--dtype` - Model dtype (default: bfloat16)

### Cache
- `--cache` - Enable DBCache
- `--rdt` - Residual diff threshold (default: 0.08)
- `--Fn` - First N compute blocks (default: 8)
- `--Bn` - Last N compute blocks (default: 0)

### Parallelism
- `--parallel-type` - Parallelism type (tp/ulysses/ring)
  - **Tensor Parallelism (tp)**: Supported via broadcast-based synchronization
  - **Context Parallelism (ulysses/ring)**: Supported
- `--compile` - Enable torch.compile (enables auto warmup per shape)

### Memory
- `--enable-cpu-offload` - Enable CPU offload
- `--device-map` - Device map strategy

## Server Examples

### Basic
```bash
cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache
```

### With Compile
```bash
cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache --compile
```

### Context Parallelism
```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache --parallel-type ulysses
```

### Tensor Parallelism
```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache --parallel-type tp
```


## Attribution

Serving code mostly adapted from [SGLang](https://github.com/sgl-project/sglang):
- [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py)
- [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
- [launch_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py)
