# Cache-DiT Serving

HTTP serving for text-to-image diffusion models with cache-dit acceleration.

Adapted from [SGLang](https://github.com/sgl-project/sglang).

## Quick Start

```bash
pip install -e ".[serving]"

cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache

curl http://localhost:8000/health
```

## API Endpoints

- `GET /health` - Health check
- `GET /get_model_info` - Model information
- `POST /generate` - Generate images
- `POST /flush_cache` - Flush cache
- `GET /docs` - API documentation

## Generate Images

### Using Client Script

```bash
python -m cache_dit.serve.client \
    --prompt "A beautiful sunset over the ocean" \
    --width 1024 \
    --height 1024 \
    --steps 50 \
    --output output.png
```

### Using Python

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

result = response.json()
img_data = base64.b64decode(result["images"][0])
img = Image.open(BytesIO(img_data))
img.save("output.png")
```

### Using curl + jq

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "width": 1024,
    "height": 1024,
    "num_inference_steps": 50
  }' | jq -r '.images[0]' | base64 -d > output.png
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

## Examples

### Basic
```bash
cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache
```

### With Compile (Auto Warmup)
```bash
cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache --compile
```

### Context Parallelism
```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type ulysses
```

### Tensor Parallelism
```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --parallel-type tp
```

## Supported Models

Flux, Qwen-Image, Wan, CogView3+/4, HunyuanDiT/Video, Mochi, LTX-Video, etc.

## Attribution

Adapted from [SGLang](https://github.com/sgl-project/sglang):
- [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py)
- [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
- [launch_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py)
