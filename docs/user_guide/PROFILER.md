# Torch Profiler Usage

## Quick Start

### Basic Usage

`cache-dit` examples have Torch Profiler built in: pass `--profile` to `examples/generate.py` to generate a trace file.

Before running examples, make sure `cache_dit` is importable by Python.

Recommended: run from the `examples/` directory (consistent with `examples/README.md`):

```bash
cd examples

# List all available examples
python3 generate.py list

# Basic profiling (recommended: reduce steps to keep the trace small)
python3 generate.py flux --profile --steps 3
```

If you want to write traces to a specific directory (or customize the filename prefix):

```bash
cd examples
python3 generate.py flux --profile --steps 3 --profile-dir /tmp/cache_dit_profiles --profile-name flux_test
```

> Note: for multi-GPU runs (`torchrun`), each rank produces its own trace file, e.g. `flux_test-rank0.trace.json.gz`.

---

If you want minimal-intrusion integration in your own script, reuse `create_profiler_from_args`:

```python
from utils import create_profiler_from_args

def run_pipe():
    # Recommended: reduce steps during profiling to keep the trace file small
    steps = args.num_inference_steps if args.num_inference_steps is not None else 28
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

### Example: `examples/base.py` Integration (Already Done)

`generate.py` eventually calls `ExampleBase.run()`, which already integrates `--profile/--profile-dir/--profile-activities`; you only need to pass these flags on the command line.

## Command-Line Arguments

```bash
# Basic profiling
cd examples
python3 generate.py flux --profile --steps 3

# With custom profile name and output directory
cd examples
python3 generate.py flux --profile --steps 3 --profile-name flux_test --profile-dir /tmp/profiles

# Profile with memory tracking
cd examples
python3 generate.py flux --profile --steps 3 --profile-activities CPU GPU MEM --track-memory
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
- `--profile-with-stack`: Record stack traces (default: True, enable for detailed debugging)
- `--profile-record-shapes`: Record tensor shapes (default: True)

**Returns:**
- `ProfilerContext`: Context manager for profiling

**Environment Variables:**
- `CACHE_DIT_TORCH_PROFILER_DIR`: Default output directory

### Controlling Trace File Size

Torch Profiler trace files can be large. Recommendations:
- Reduce `--steps` (e.g., 3–5)
- Reduce `--repeat`
- Optionally disable `--profile-with-stack` / `--profile-record-shapes` (if you add a way to disable them in your workflow)

```bash
# Profile with 3 steps (small trace file, recommended)
cd examples
python3 generate.py flux --profile --steps 3 --warmup 0 --repeat 1

# Profile with full 28 steps (larger trace file)
cd examples
python3 generate.py flux --profile --steps 28 --warmup 0 --repeat 1
```

## View Results

### Perfetto UI (Recommended)
Visit https://ui.perfetto.dev/ and drag-drop the generated `.trace.json.gz` file. Perfetto provides a more powerful and feature-rich interface compared to Chrome Tracing.

The screenshots below show an example profiling result from `generate.py flux` (model: FLUX.1-dev).


<img width="1240" height="711" alt="图片" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/profile_0.png" />

<img width="1225" height="545" alt="图片" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/profile_1.png" />

<img width="1111" height="281" alt="图片" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/profile_2.png" />


### Chrome Tracing
Open `chrome://tracing` in Chrome browser and load the generated `.trace.json.gz` file.

### TensorBoard
```bash
pip install tensorboard
tensorboard --logdir=/path/to/profiles
```

## Multi-GPU Usage

The profiler automatically handles distributed environments. Each rank will generate its own trace file.

### Example: Tensor Parallelism

```bash
# 2 GPUs with tensor parallelism
cd examples
torchrun --nproc_per_node=2 generate.py flux \
    --parallel tp \
    --profile --profile-name flux_tp --steps 3 --warmup 0 --repeat 1

# Output files:
# - flux_tp-rank0.trace.json.gz
# - flux_tp-rank1.trace.json.gz
```

### Example: Context Parallelism

```bash
# 4 GPUs with context parallelism
cd examples
torchrun --nproc_per_node=4 generate.py flux \
    --parallel ulysses \
    --profile --profile-name flux_cp --profile-activities CPU GPU MEM \
    --steps 3 --warmup 0 --repeat 1

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
    with_stack=True,
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

---

# Nsight Systems (nsys) Usage

If you need a lower-level CUDA view (kernel timeline, CUDA API, CPU/GPU concurrency, etc.), use Nsight Systems.

## Installation

Follow NVIDIA Nsight Systems installation instructions (the CLI is usually `nsys`), or your internal environment setup.

## Basic Profiling

The example below profiles a single inference (recommended: set `--warmup 0` so warmup is not included):

```bash
cd examples
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  -o cache_dit_flux \
  python3 generate.py flux --steps 28 --warmup 0 --repeat 1
```

## Targeted Capture (reduce file size)

Use `--delay/--duration` to skip model loading/initialization and capture only the main inference window:

```bash
cd examples
nsys profile \
  --trace=cuda,nvtx,osrt \
  --force-overwrite=true \
  --delay 10 \
  --duration 30 \
  -o cache_dit_flux_infer \
  python3 generate.py flux --steps 28 --warmup 0 --repeat 1
```

**Parameter notes:**
- `--delay N`: wait N seconds before capture (commonly used to skip initialization)
- `--duration N`: stop capture after N seconds (commonly used to limit file size)
- `-o <NAME>`: output file prefix
