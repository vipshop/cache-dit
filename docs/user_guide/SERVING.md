# Cache-DiT Serving

HTTP serving for diffusion models with cache-dit acceleration. Supports **text-to-image**, **image editing**, **multi-image editing**, **text-to-video**, and **image-to-video** generation.

Adapted from [SGLang](https://github.com/sgl-project/sglang).

## Supported Tasks

- **Text-to-Image (t2i)**
- **Image Editing (edit)**
- **Text-to-Video (t2v)**
- **Image-to-Video (i2v)**

Serving setups for LoRA, multi-image editing, distributed parallelism, etc. are available as runnable recipes.

## Start Server

```bash
pip install -e ".[serving]"

torchrun --nproc_per_node=1 -m cache_dit.serve.serve \
  --model-path black-forest-labs/FLUX.1-dev \
  --cache

curl http://localhost:8000/health
open http://localhost:8000/docs
```

## Example 1: Text-to-Image

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A beautiful sunset over the ocean","width":1024,"height":1024,"num_inference_steps":50}' \
  http://localhost:8000/generate
```

## Example 2: Text-to-Video

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A cat walks on the grass, realistic","width":832,"height":480,"num_frames":49,"fps":16,"num_inference_steps":30}' \
  http://localhost:8000/generate
```

## More Recipes

For t2i / edit / t2v / i2v, LoRA, and multi-GPU launch examples, see:

https://github.com/vipshop/cache-dit/tree/main/tests/serving
