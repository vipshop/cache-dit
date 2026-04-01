# How to use CUDA Graph

## CUDA Graph + torch.compile

CUDA Graph captures a stable GPU execution path and replays it, which can reduce CPU launch overhead and improve execution stability in some workloads. In Cache-DiT example CLI (<span style="color:#c77dff;">cache_dit.generate</span>), CUDA Graph is enabled through <span style="color:#c77dff;">torch.compile</span> options <span style="color:green">{"triton.cudagraphs": True}</span> or <span style="color:green">max-autotune</span> mode, which automatically enables CUDA Graph when capture conditions are met. Here is an End-to-End Python Example (same style as Cache-DiT usage):

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

Quick start for Cache-DiT example CLI: NVIDIA L20 x 1, FLUX.1-dev, 28 steps, 1024x1024.

```bash
python3 -m cache_dit.generate flux --compile --no-regional-compile # w/o CUDA Graph
python3 -m cache_dit.generate flux --compile --cuda-graph --no-regional-compile # w/ CUDA Graph
```

First-run includes compile and warmup; steady-state is after warmup. For FLUX.1-dev, we see a modest speedup in steady-state runs after enabling CUDA Graph, which suggests that GPU execution is already efficient and CUDA Graph is effectively reducing CPU launch overhead.

| FLUX.1-dev, compile (no CUDA Graph)| FLUX.1-dev, compile + CUDA Graph |
|:--:|:--:|
| 20.73s | <span style="color:green">20.70s</span> |



## Constraints & Troubleshooting

- <span style="color:#c77dff;">Do not use regional compile with CUDA Graph</span>: When CUDA Graph is enabled, repeated-block regional compilation (`compile_repeated_blocks`) can cause replay-overwrite issues in some transformer loops (for example FLUX blocks). Use full-module compile for transformer.

- <span style="color:#c77dff;">Dynamic shape is currently not recommended</span>: CUDA Graph generally expects stable shapes and stable execution paths.(1) Do not enable `torch.compile(..., dynamic=True)` when using CUDA Graph; (2) In Cache-DiT example CLI, avoid `--force-compile-dynamic` together with `--cuda-graph`.


- <span style="color:#c77dff;">RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten</span>: Why it happens? (1) A captured graph output is referenced after a later replay has already overwritten the same output buffer. (2) This often appears when CUDA Graph is combined with regional compile (`compile_repeated_blocks`) in transformer loops. The typical message is <span style="color:red">"RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run"</span>. How to fix: (Quick check for Cache-DiT CLI, if you see this error in the logs, make sure to disable regional compile and use full-module compile for transformer by adding the flag: Use `--compile --cuda-graph --no-regional-compile`.)
    1. Disable regional compile and compile the full transformer when using CUDA Graph.
    2. DON'T use `dynamic=True` and ensure stable input shapes.
    3. If you invoke compiled modules manually in a loop, call <span style="color:#c77dff;">torch.compiler.cudagraph_mark_step_begin()</span> before each model invocation.

- <span style="color:#c77dff;">Graph breaks or repeated recompilation</span>. Typical signals: (1) Frequent recompilation logs. (2) Throughput drops after enabling CUDA Graph. Why it happens: (1) Dynamic shapes, changing control flow, or changing optional inputs between runs can invalidate capture assumptions. How to fix:
    1. Keep inference settings fixed across runs (height/width/steps/batch size).
    2. Avoid `dynamic=True` and avoid `--force-compile-dynamic` with CUDA Graph.
    3. Keep optional branches stable (for example, consistently enable/disable ControlNet or IP-Adapter for a run).

- <span style="color:#c77dff;">CUDA Graph enabled but little/no speedup</span>. Possible reasons: (1) Workload is already kernel-bound with low CPU launch overhead. (2) First-run compile and warmup dominate short benchmark windows. (3) Extra fallback/recompile events offset replay gains. How to validate:
    1. Compare steady-state runs after warmup (not first-run latency).
    2. Keep benchmark setup identical (same prompt length, steps, resolution, and seed policy).
    3. Profile CPU launch overhead to confirm CUDA Graph is the right optimization target.
