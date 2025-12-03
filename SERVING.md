# Cache-DiT Serving

HTTP serving for text-to-image diffusion models with cache-dit acceleration.

Adapted from [SGLang](https://github.com/sgl-project/sglang)'s serving architecture.

## Quick Start

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Start server with cache enabled
python -m cache_dit.serve.serve --model-path Qwen/Qwen-Image --cache

# Test
curl http://localhost:8000/health
```

## Usage

### Start Server

```bash
python -m cache_dit.serve.serve \
    --model-path Qwen/Qwen-Image \
    --host 0.0.0.0 \
    --port 8000 \
    --cache
```

### Generate Images

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }'
```

### Python Client

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cute cat",
        "width": 1024,
        "height": 1024,
    }
)

result = response.json()
img_data = base64.b64decode(result["images"][0])
img = Image.open(BytesIO(img_data))
img.save("output.png")
```

## API Endpoints

- `GET /health` - Health check
- `GET /get_model_info` - Model information
- `POST /generate` - Generate images
- `POST /flush_cache` - Flush cache
- `GET /docs` - API documentation (Swagger UI)

## Command Line Arguments

The server supports all arguments from `examples/utils.py` plus server-specific arguments.

### Model Arguments

- `--model-path` - Model path or HuggingFace model ID (required)
- `--device` - Device (cuda/cpu), auto-detect by default
- `--dtype` - Model dtype (float32/float16/bfloat16), default: bfloat16

### Cache Arguments (from utils.py)

- `--cache` - Enable DBCache acceleration
- `--Fn` - First N compute blocks (default: 8)
- `--Bn` - Last N compute blocks (default: 0)
- `--rdt` - Residual difference threshold (default: 0.08)
- `--max-warmup-steps`, `--w` - Maximum warmup steps (default: 8)
- `--warmup-interval`, `--wi` - Warmup interval (default: 1)
- `--max-cached-steps`, `--mc` - Maximum cached steps (default: -1)
- `--max-continuous-cached-steps`, `--mcc` - Maximum continuous cached steps (default: -1)
- `--taylorseer` - Enable TaylorSeer calibrator
- `--taylorseer-order`, `-order` - TaylorSeer order (default: 1)

### Parallelism Arguments (from utils.py)

- `--parallel-type`, `--parallel` - Parallelism type (tp/ulysses/ring)
- `--attn` - Attention backend (flash/native/_native_cudnn/_sdpa_cudnn/sage)
- `--ulysses-anything`, `--uaa` - Enable Ulysses Anything Attention
- `--ulysses-async-qkv-proj`, `--ulysses-async`, `--uaqkv` - Enable async QKV projection

### Quantization Arguments (from utils.py)

- `--quantize`, `-q` - Enable quantization
- `--quantize-type` - Quantization type (float8/float8_weight_only/int8/int8_weight_only/int4/int4_weight_only/bitsandbytes_4bit)

### Server Arguments

- `--host` - Server host (default: 0.0.0.0)
- `--port` - Server port (default: 8000)
- `--workers` - Number of worker processes (default: 1)

### Memory Arguments

- `--enable-cpu-offload` - Enable CPU offload (saves GPU memory)
- `--device-map` - Device map strategy (e.g., balanced)

### Other Arguments (from utils.py)

- `--compile` - Enable torch.compile
- `--compile-repeated-blocks` - Compile repeated blocks
- `--max-autotune` - Enable max autotune
- `--profile` - Enable profiling
- `--profile-name` - Profile name
- `--profile-dir` - Profile output directory
- `--track-memory` - Track and report peak GPU memory usage

## Supported Models

All models supported by cache-dit:

- Qwen-Image
- Flux / Flux.1
- Wan / Wan 2.2
- CogView3+ / CogView4
- HunyuanDiT / HunyuanVideo
- Mochi
- LTX-Video
- And more...

## Examples

### Basic Generation

```bash
python -m cache_dit.serve.serve --model-path Qwen/Qwen-Image --cache
```

### With Custom Cache Settings

```bash
python -m cache_dit.serve.serve \
    --model-path Qwen/Qwen-Image \
    --cache \
    --rdt 0.15 \
    --Fn 8
```

### Memory Optimization

```bash
python -m cache_dit.serve.serve \
    --model-path Qwen/Qwen-Image \
    --enable-cpu-offload \
    --device-map balanced
```

### Without Cache

```bash
python -m cache_dit.serve.serve \
    --model-path Qwen/Qwen-Image
    # Don't add --cache flag to disable cache
```

## Attribution

This serving implementation is adapted from [SGLang](https://github.com/sgl-project/sglang):

- Model management: [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py)
- HTTP server: [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
- Server launcher: [launch_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py)
