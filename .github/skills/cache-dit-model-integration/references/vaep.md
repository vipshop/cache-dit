# VAE Parallelism Reference

When to read this: read this file when a model uses a VAE not already supported by cache-dit, or when adding a new VAE data-parallel planner. Return to `../SKILL.md` for the integration order.

## 5. VAE Parallelism (VAE-P)

### 5.1 Concept

VAE-P applies **data parallelism** (not tensor parallelism) to the VAE decoder. Multiple GPUs each decode a portion of the latent grid, then results are stitched together. This is useful when the VAE decoder is a memory bottleneck. Activate via `--parallel-vae`.

### 5.2 When to implement

Implement VAE-P only if your model uses a VAE that is **not yet supported**. Currently supported: AutoencoderKL (generic), plus specialized variants for LTX2, QwenImage, Wan, HunyuanVideo, and Flux2.

### 5.3 Implementation

In `src/cache_dit/distributed/autoencoders/<vae_name>.py`:

```python
from ..config import ParallelismConfig
from .register import (
    AutoEncoderDataParallelismPlanner,
    AutoEncoderDataParallelismPlannerRegister,
)

@AutoEncoderDataParallelismPlannerRegister.register("MyAutoencoderKL")
class MyAutoencoderKLDataParallelismPlanner(AutoEncoderDataParallelismPlanner):

    def _apply(
        self,
        auto_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        # Obtain the data-parallel device mesh
        dp_mesh = self.mesh(parallelism_config=parallelism_config)
        # Apply your model-specific tiling / P2P communication logic here.
        # See src/cache_dit/distributed/autoencoders/autoencoder_kl.py for the
        # reference implementation (TileBatchedP2PComm + parallelize_tiling).
        return auto_encoder
```

### 5.4 Registration

Add to `src/cache_dit/distributed/autoencoders/planners.py` inside `_activate_auto_encoder_dp_planners()`:

```python
MyAutoencoderKLDataParallelismPlanner = _safe_import(
    ".my_vae", "MyAutoencoderKLDataParallelismPlanner")
```

---

## More references 

We recommend reading the following files for additional context:

- VAE data parallelism source code: `src/cache_dit/distributed/autoencoders`
