# Profiler Usage

Cache-DiT uses Torch Profiler to collect requests or selected complete denoising steps.

## Python Requests and Loop Boundaries

`ProfilerContext` wraps one inference. It defaults to `enabled=True`,
`ranges="all"`, `with_stack=True` and `record_shapes=True`:

```python
from cache_dit import ProfilerContext

# After the original initialization, optimization and warmup.
with ProfilerContext(ranges="all", devices=["cuda:0"],
                     output_dir="/tmp/cache_dit_profiles") as request:
    output = pipe(**inputs)

for result in request.results:
    print(result.loop_id, result.requested_range, result.actual_range,
          result.status, result.trace_path)
```

`create_profiler_context()` forwards context options and defaults to `enabled=False`.
`profile_function()` accepts keyword-only `ranges` and `devices` and defaults to
`with_stack=False`.

Disabled contexts perform no collection and work on CPU. Enabled Torch requests require CUDA or
Ascend NPU. Only one Cache-DiT profiler request can be active in a process,
including the gaps between numeric windows.

### Add Complete-Step Boundaries

Integrate `profile_steps()` in the pipeline's actual loop before selecting numeric ranges:

```python
from cache_dit import profile_steps

with profile_steps(timesteps) as steps:
    for i, t in steps:
        ...  # Original input preparation, prediction, CFG and scheduler updates.
        ...  # Original callbacks, progress and all remaining work for this step.
```

Keep the context around the loop so `break`, `return` and exceptions close its numeric window
before later work. Without an active numeric request, it behaves as ordinary `enumerate()`;
`all` stays open until the request exits. Wrap the inference with
`ProfilerContext(ranges="0-2,8-10", devices=...)` to select those complete steps.

### Results and Early Exit

`request.results` contains zero or more window results. Each records request/loop/window IDs,
requested and actual ranges, `last_step_complete`, status and successful artifact paths.
Whole-request `all` results have no step range or last-step flag.

| Status | Meaning |
|---|---|
| `completed` | The selected endpoint finished, or an `all` request returned normally |
| `exhausted` | The loop ended before the selected endpoint |
| `interrupted` | The loop exited early, for example by `break` or `return` |
| `error` | An error occurred while opening, running or closing this window |

An early-exit step whose iterator has not resumed is not marked complete. A start index never
reached creates no window. A range extending past the loop records the steps actually reached.

`request.trace_path` retains the most recent successful Torch trace, or `None`; use `results` for
all windows. CUDA memory snapshots and summaries are recorded independently, so an error window
can still contain usable memory artifacts. Cleanup attempts each device synchronization, stop,
export and memory-history shutdown independently, preserving an existing model exception.

## Torch Artifacts and Configuration

Use activities `CPU`, `GPU` and optional `MEM`; `GPU` maps to CUDA or Ascend NPU.
`MEM` alone enables the current accelerator activity. CUDA memory recording exports a snapshot
and a text summary independently from the trace. Each window has its own collector and unique
host/PID/rank/request/loop/window filename. CUDA exports `.trace.json.gz`; Ascend exports the full
CANN directory, including `ASCEND_PROFILER_OUTPUT`.

The default output directory is `CACHE_DIT_TORCH_PROFILER_DIR`, or `/tmp/cache_dit_profiles`.
`get_profiler_output_dir()` reads it and `set_profiler_output_dir(path)` updates future contexts;
an explicit `output_dir` takes priority. `CACHE_DIT_NPU_PROFILER_LEVEL` accepts Level0, Level1,
Level2 or none. All declared devices are synchronized at window boundaries.

Disable Python stacks with `with_stack=False` when an operator/kernel timeline is sufficient.
Recording, synchronization and export add overhead; benchmark separately without profiling.
