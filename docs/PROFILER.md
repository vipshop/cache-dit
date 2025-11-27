# Torch Profiler Usage

Reference: Adapted from `sglang/python/sglang/bench_one_batch.py` and `sglang/python/sglang/srt/managers/scheduler_profiler_mixin.py`

## Quick Start

### Basic Usage

Add profiler to your example script with minimal changes:

```python
from utils import create_profiler_from_args

def run_pipe():
    # Reduce steps when profiling to keep trace file small
    steps = args.steps if args.steps is not None else 28
    if args.profile and args.steps is None:
        steps = 3
    image = pipe(
        prompt,
        num_inference_steps=steps,
        ...
    ).images[0]
    return image

# Only create and use profiler when --profile is enabled
if args.profile:
    profiler = create_profiler_from_args(args, profile_name="flux_inference")
    with profiler:
        image = run_pipe()
    print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe()
```

### Example: run_flux.py Integration

```python
import time
import torch
from diffusers import FluxPipeline, FluxTransformer2DModel
from utils import get_args, strify, cachify, MemoryTracker, create_profiler_from_args
import cache_dit

args = get_args()
pipe = FluxPipeline.from_pretrained(...)

# ... model setup code ...

def run_pipe():
    steps = args.steps if args.steps is not None else 28
    if args.profile and args.steps is None:
        steps = 3
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=steps,
        generator=torch.Generator("cpu").manual_seed(0),
    ).images[0]
    return image

_ = run_pipe()  # warmup

memory_tracker = MemoryTracker() if args.track_memory else None

if memory_tracker:
    memory_tracker.__enter__()

start = time.time()
if args.profile:
    profiler = create_profiler_from_args(args, profile_name="flux_inference")
    with profiler:
        image = run_pipe()
    print(f"Profiler traces saved to: {profiler.output_dir}/{profiler.trace_path.name}")
else:
    image = run_pipe()
end = time.time()

if memory_tracker:
    memory_tracker.__exit__(None, None, None)
    memory_tracker.report()
```

## Command-Line Arguments

```bash
# Basic profiling
python examples/pipeline/run_flux.py --profile

# With custom profile name and output directory
python examples/pipeline/run_flux.py --profile --profile-name flux_test --profile-dir /tmp/profiles

# Profile with memory tracking
python examples/pipeline/run_flux.py --profile --profile-activities CPU GPU MEM
```

## Parameters

### `create_profiler_from_args(args, profile_name=None)`

Creates a ProfilerContext from command-line arguments.

**Arguments:**
- `args`: Parsed command-line arguments containing profiler settings
- `profile_name` (str, optional): Override the profile name

**Command-Line Arguments:**
- `--profile`: Enable profiler (default: False)
- `--profile-name` (str): Profile name prefix (default: auto-generated timestamp)
- `--profile-dir` (str): Output directory (default: $CACHE_DIT_TORCH_PROFILER_DIR or `/tmp/cache_dit_profiles`)
- `--profile-activities` (list): Activities to profile - CPU, GPU, MEM (default: ["CPU", "GPU"])
- `--profile-with-stack`: Record stack traces (default: False)
- `--profile-record-shapes`: Record tensor shapes (default: True)
- `--profile-wait` (int): Steps to wait before profiling (default: 0)
- `--profile-warmup` (int): Warmup steps (default: 1)
- `--profile-active` (int): Active profiling steps (default: 3)
- `--profile-repeat` (int): Repeat profiling cycle (default: 1, recommended to keep at 1 for inference)

**Returns:**
- `ProfilerContext`: Context manager for profiling

**Environment Variables:**
- `CACHE_DIT_TORCH_PROFILER_DIR`: Default output directory

### Understanding Profiler Schedule Parameters

PyTorch profiler uses a scheduling mechanism to control when profiling happens:

**`--profile-wait`**: Number of steps to skip before starting profiling
- Used to skip initial overhead or compilation steps
- Default: 0

**`--profile-warmup`**: Number of warmup steps to record
- These steps are profiled but marked as warmup in the trace
- Default: 1

**`--profile-active`**: Number of steps to actively profile
- Main profiling steps that capture steady-state performance
- Default: 3

**Example Timeline** with default settings `wait=0, warmup=1, active=3`:
```
[warmup] [profile] [profile] [profile]
```

**Recommended Settings:**

For diffusion inference (image/video generation):
```bash
python examples/pipeline/run_flux.py --profile
```

**Important:** When `--profile` is enabled without specifying `--steps`, the inference steps are automatically reduced to 3 to keep trace files small. This profiles 1 warmup step + 3 active steps, providing sufficient performance data while minimizing file size. If you need to profile with more steps, explicitly set `--steps`.

## View Results

### Perfetto UI (Recommended)
Visit https://ui.perfetto.dev/ and drag-drop the generated `.trace.json.gz` file. Perfetto provides a more powerful and feature-rich interface compared to Chrome Tracing.

### Chrome Tracing
Open `chrome://tracing` in Chrome browser and load the generated `.trace.json.gz` file.

### TensorBoard
```bash
pip install tensorboard
tensorboard --logdir=/path/to/profiles
```

### Memory Analysis
```bash
pip install memory_viz
python -c "import torch; torch.cuda.memory._load_snapshot('profile-rank0-memory-*.pickle')"
```

## Multi-GPU Usage

The profiler automatically handles distributed environments. Each rank will generate its own trace file.

### Example: Tensor Parallelism

```bash
# 2 GPUs with tensor parallelism
torchrun --nproc_per_node=2 examples/parallelism/run_flux_tp.py \
    --profile --profile-name flux_tp

# Output files:
# - flux_tp-rank0.trace.json.gz
# - flux_tp-rank1.trace.json.gz
```

### Example: Context Parallelism

```bash
# 4 GPUs with context parallelism
torchrun --nproc_per_node=4 examples/parallelism/run_flux_cp.py \
    --profile --profile-name flux_cp --profile-activities CPU GPU MEM

# Output files:
# - flux_cp-rank0.trace.json.gz
# - flux_cp-rank1.trace.json.gz
# - flux_cp-rank2.trace.json.gz
# - flux_cp-rank3.trace.json.gz
# - flux_cp-rank0-memory-*.pickle (if MEM profiling enabled)
# - flux_cp-rank1-memory-*.pickle
# - ...
```

You can view each rank's trace separately in Perfetto UI or Chrome Tracing to analyze per-GPU performance.

## Advanced Usage

### Direct API Usage

```python
from cache_dit import ProfilerContext

with ProfilerContext(
    enabled=True,
    activities=["CPU", "GPU"],
    output_dir="/tmp/profiles",
    profile_name="my_inference",
    with_stack=False,
    record_shapes=True,
):
    output = model(input)
```

### Decorator

```python
from cache_dit import profile_function

@profile_function(enabled=True, profile_name="forward_pass")
def my_function():
    return model(input)
```
