# Cache-DiT Serving

HTTP serving for diffusion models with cache-dit acceleration. Supports **text-to-image**, **image editing**, **multi-image editing**, **text-to-video**, and **image-to-video** generation.

Adapted from [SGLang](https://github.com/sgl-project/sglang).

## Quick Start

```bash
pip install -e ".[serving]"

cache-dit-serve --model-path black-forest-labs/FLUX.1-dev --cache

curl http://localhost:8000/health

curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A beautiful landscape with mountains and a lake"}' \
  http://localhost:8000/generate
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
- `--parallel-text-encoder` - Enable text encoder parallelism if applicable
- `--parallel-vae` - Enable VAE parallelism if applicable

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

### Context & Tensor Parallelism
```bash
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache --parallel-type ulysses --parallel-text-encoder
```

This enables hybrid parallelism where the Transformer uses Context Parallelism (CP) while the TextEncoder uses Tensor Parallelism (TP). Both CP and TP groups operate on the same set of GPUs, allowing efficient resource utilization across different model components.

### Quantization

Enable model quantization to reduce memory usage:

```bash
cache-dit-serve \
    --model-path black-forest-labs/FLUX.1-dev \
    --cache \
    --quantize-type float8_wo
```

**Notes:**
- **`float8_wo` / `float8_weight_only` uses torchao online quantization** (via `cache_dit.quantize(...)`) and does **NOT** require a `PipelineQuantizationConfig` file.
- Install deps: `pip install "cache-dit[quantization]"`.
- FP8 requires modern GPUs (see `cache_dit/quantize/torchao/quantize_ao.py`, currently checks compute capability \(\ge 8.9\)).
- If you load LoRA weights, LoRA fusion will be automatically disabled when transformer is (or will be) quantized.

For **4-bit W4A16** quantization (recommended: **bitsandbytes nf4**), use `--pipeline-quant-config-path` and `PipelineQuantizationConfig`:

```python
import torch
from diffusers.quantizers import PipelineQuantizationConfig

def get_pipeline_quant_config():
    return PipelineQuantizationConfig(
        quant_backend="bitsandbytes_4bit",
        quant_kwargs={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        components_to_quantize=["text_encoder", "transformer"],
    )
```

### LoRA Models

Load and serve models with LoRA weights for fine-tuned generation:

```bash
cache-dit-serve \
    --model-path Qwen/Qwen-Image \
    --lora-path /path/to/lora/weights \
    --lora-name model.safetensors \
    --cache
```

**Key Arguments:**
- `--lora-path`: Directory containing LoRA weights (supports both local paths and HuggingFace model IDs)
- `--lora-name`: LoRA weight filename (e.g., `model.safetensors`)
- `--disable-fuse-lora`: Disable LoRA fusion and keep weights separate (default: fusion enabled for better performance)

**Example: Qwen-Image-Lightning (8-step distilled model)**

```bash
# Download LoRA weights from https://huggingface.co/lightx2v/Qwen-Image-Lightning
cache-dit-serve \
    --model-path Qwen/Qwen-Image \
    --lora-path lightx2v/Qwen-Image-Lightning \
    --lora-name Qwen-Image-Lightning-8steps-V1.0-bf16.safetensors \
    --cache --rdt 0.2
```

Then generate images with 8 inference steps:

```python
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A beautiful landscape with mountains and a lake",
        "width": 1024,
        "height": 1024,
        "num_inference_steps": 8,  # Lightning model uses 8 steps
        "guidance_scale": 1.0,
        "seed": 42
    }
)

img_data = base64.b64decode(response.json()["images"][0])
Image.open(BytesIO(img_data)).save("output.png")
```

**LoRA with Other Optimizations:**

```bash
# LoRA + Cache + Compile
cache-dit-serve \
    --model-path Qwen/Qwen-Image \
    --lora-path lightx2v/Qwen-Image-Lightning \
    --lora-name Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors \
    --cache --compile

# LoRA + Context Parallelism
torchrun --nproc_per_node=2 -m cache_dit.serve.serve \
    --model-path Qwen/Qwen-Image \
    --lora-path lightx2v/Qwen-Image-Lightning \
    --lora-name Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors \
    --cache --parallel-type ulysses
```

**Notes:**
- Both `--lora-path` and `--lora-name` must be provided together
- LoRA fusion is automatically disabled when transformer is quantized
- Supports `.safetensors` and `.bin` format LoRA weights


## Attribution

Serving code mostly adapted from [SGLang](https://github.com/sgl-project/sglang):  

- [tokenizer_manager.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/tokenizer_manager.py)
- [http_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py)
- [launch_server.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/launch_server.py)
