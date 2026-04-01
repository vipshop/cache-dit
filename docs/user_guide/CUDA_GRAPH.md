# How to use CUDA Graph

## What It Is

CUDA Graph captures a stable GPU execution path and replays it, which can reduce CPU launch overhead and improve execution stability in some workloads. In Cache-DiT example CLI (`cache_dit.generate`), CUDA Graph is enabled through `torch.compile` options:   
- `_compile_options(args)`: returns <span style="color:green">{"triton.cudagraphs": True}</span> when `--cuda-graph` is enabled (and dynamic/max-autotune constraints are satisfied); 
- `_compile_mode(args)`: selects compile mode (<span style="color:green">max-autotune</span> / `max-autotune-no-cudagraphs` / default behavior).

## FLUX.1-dev Example

Here is an End-to-End Python Example (same style as Cache-DiT usage)

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
	"black-forest-labs/FLUX.1-dev",
	torch_dtype=torch.bfloat16,
).to("cuda")

# Enable compile + CUDA Graph through torch.compile options
pipe.transformer = torch.compile(
	pipe.transformer,
	options={"triton.cudagraphs": True},
)

# Enable compile + CUDA Graph through torch.compile max-autotune mode 
# (which will automatically enable cudagraphs if constraints are satisfied)
pipe.transformer = torch.compile(
  pipe.transformer,
  mode="max-autotune",
)
```

## Performance Comparison

Environment: NVIDIA L20 x 1, FLUX.1-dev, 28 steps, 1024x1024.

| FLUX.1-dev, compile (no CUDA Graph)| FLUX.1-dev, compile + CUDA Graph |
|:--:|:--:|
| 20.73s | <span style="color:green">20.69s</span> |

## Cache-DiT CLI Example

```bash
python3 -m cache_dit.generate flux --compile --cuda-graph
```

## Important Constraints

- Do not use regional compile with CUDA Graph: When CUDA Graph is enabled, repeated-block regional compilation (`compile_repeated_blocks`) can cause replay-overwrite issues in some transformer loops (for example FLUX blocks). Use full-module compile for transformer.

- Dynamic shape is currently not recommended: CUDA Graph generally expects stable shapes and stable execution paths.(1) Do not enable `torch.compile(..., dynamic=True)` when using CUDA Graph; (2) In Cache-DiT example CLI, avoid `--force-compile-dynamic` together with `--cuda-graph`.


## Troubleshooting

### 1) RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten

Typical message:

```text
RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run.
```

Why it happens:

- A captured graph output is referenced after a later replay has already overwritten the same output buffer.
- This often appears when CUDA Graph is combined with regional compile (`compile_repeated_blocks`) in transformer loops.

How to fix:

1. Disable regional compile and compile the full transformer when using CUDA Graph.
2. Keep `dynamic=False` and stable input shapes.
3. If you invoke compiled modules manually in a loop, call `torch.compiler.cudagraph_mark_step_begin()` before each model invocation.

Quick check for Cache-DiT CLI, if you see this error in the logs, make sure to disable regional compile and use full-module compile for transformer by adding the flag:

- Use `--compile --cuda-graph --no-regional-compile`.

### 2) Graph breaks or repeated recompilation

Typical signals:

- Frequent recompilation logs.
- Throughput drops after enabling CUDA Graph.

Why it happens:

- Dynamic shapes, changing control flow, or changing optional inputs between runs can invalidate capture assumptions.

How to fix:

1. Keep inference settings fixed across runs (height/width/steps/batch size).
2. Avoid `dynamic=True` and avoid `--force-compile-dynamic` with CUDA Graph.
3. Keep optional branches stable (for example, consistently enable/disable ControlNet or IP-Adapter for a run).

### 3) CUDA Graph enabled but little/no speedup

Possible reasons:

- Workload is already kernel-bound with low CPU launch overhead.
- First-run compile and warmup dominate short benchmark windows.
- Extra fallback/recompile events offset replay gains.

How to validate:

1. Compare steady-state runs after warmup (not first-run latency).
2. Keep benchmark setup identical (same prompt length, steps, resolution, and seed policy).
3. Profile CPU launch overhead to confirm CUDA Graph is the right optimization target.
