# Pipeline Profiling

Run from a source checkout with `cache_dit` installed, or set `PYTHONPATH=src` from the repository
root. This runner owns profiling, NVTX annotations and model instrumentation. The generate CLI
keeps its original Torch profiling behavior.

```bash
PYTHONPATH=src python bench/profile/run.py --help
PYTHONPATH=src python bench/profile/run.py flux --profile --steps 3 --warmup 1 --repeat 2 \
    --no-profile-with-stack --profile-dir /tmp/cache_dit_profiles
```

The runner reuses the registered example's inputs, loading, optimization, generators, media parsers
and output writer. It completes the original warmup, then collects each formal repeat separately.
`--repeat 2 --profile-ranges all` produces two CUDA traces per rank; `generate --profile` combines
its formal repeats into one trace per rank. Initialization, optimization and warmup are outside
collection in both paths.

## Model-Specific Instrumentation

The default factory is the existing `ExampleRegister`. An optional
`--example-factory module:function` selects a callable with the same `(args) -> Example` form.
It configures the example before pipeline construction. The runner does not select instrumentation
or reject numeric windows based on a model name.

[model/minimax_h3.py](model/minimax_h3.py) provides `create_example`, which selects profiling blocks through the
existing `ExampleInitConfig.modular_blocks_factory` field. Use the same factory for all three
registered workflows:

```bash
PYTHONPATH=src python bench/profile/run.py minimax_h3 \
    --example-factory model.minimax_h3:create_example --model-path /path/to/MiniMax-H3 \
    --profile --profile-ranges 0-2 --no-profile-with-stack
```

Use `minimax_h3_fl2va` or `minimax_h3_ref2va` as the example for those workflows, retaining the same
factory and the ordinary image/reference arguments. Their original components, state handling,
computation blocks and workflow pruning remain unchanged. A normal generate invocation uses the
upstream blocks.

Other models can supply their own factory without changing `run.py`. Whole-request collection
works with ordinary examples. Numeric windows require an actual loop integrated with
[`profile_steps()`](../../docs/user_guide/PROFILER.md#add-complete-step-boundaries); selecting a
factory does not itself provide that integration. A normally completed numeric request without a
step context raises `RuntimeError` after the inference call. It does not fall back to `all`.

## Request and Step Windows

Both Torch and Nsight use `--profile-ranges`:

| Value | Collection boundary |
|---|---|
| `all` (default) | Each complete formal request, including encoding and decoding |
| `0-2` | Three complete logical steps, from the start of step 0 through the end of step 2 |
| `0,2` | Two separate one-step windows |
| `0-2,8-10` | Two separate three-step windows; all intervening and remaining work still runs |

Indices start at zero in each integrated loop, and endpoints are inclusive. Overlapping or adjacent
ranges merge. Forward calls and scheduler call counts do not define logical step indices. A full
step includes prediction, all scheduler updates, callbacks and progress in the original loop body.
Numeric windows exclude conditioning and decoding outside that loop.

MiniMax-H3 uses the requested number of sigma grid points and performs one fewer logical step:
`--steps 50` produces steps 0–48. Empty loops or unmatched ranges produce no trace; the runner logs
`empty_reason`. Early exits and errors retain the core profiler's result and cleanup semantics.

## Options

Model, input, optimization and output arguments come from the existing CLI parser.

| Option | Default | Meaning |
|---|---|---|
| `--example-factory` | Existing registry | Explicit `module:function` that returns an `Example` |
| `--profile` | Off | Enable collection for each formal repeat |
| `--profile-collector` | `torch` | Torch export or CUDA gates for external Nsight collection |
| `--profile-ranges` | `all` | Whole request or inclusive complete-step windows |
| `--profile-name` | `<example>_profile` | Torch filename prefix |
| `--profile-dir` | Environment value or `/tmp/cache_dit_profiles` | Torch output directory |
| `--profile-activities` | `CPU GPU` | Torch activities; add `MEM` for memory recording |
| `--no-profile-with-stack` | Stacks enabled | Omit Python stacks from Torch traces |
| `--profile-record-shapes` | On | Record tensor shapes with Torch |
| `--nvtx` | Off | Enable explicit stage annotations after warmup |
| `--layerwise-nvtx` | Off | Enable stages and transformer module hooks after warmup |

Collector, range and annotation options do not enable `--profile`. NVTX can be used independently
with an external collector. Use Torch and Nsight in separate runs with the same inputs and
optimization settings. Nsight ignores Torch's directory, filename, activity, stack and shape options;
its external session controls report files and capture activities.

## Nsight Systems

Launch under an installed `nsys` CLI with CUDA profiler API capture enabled:

```bash
PYTHONPATH=src nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --kill=none \
    -o /tmp/cache_dit_flux \
    python bench/profile/run.py flux --steps 3 --warmup 1 --repeat 1 \
        --profile --profile-collector nsys --layerwise-nvtx
```

`stop` captures the first reached window and lets inference finish. For multiple windows or
repeats, use `--capture-range-end=repeat:N:defer`, where `N` is the number of reached windows across
requests and loops. Each window produces a report; `defer` postpones report generation.

For single-node multi-GPU collection, put one `nsys profile` outside `torchrun` so each report
contains the workers together. Every rank must enter the same requests and windows. The profiler
uses the existing default CPU/Gloo group for healthy start/stop coordination; the examples' hybrid
group satisfies this contract. It does not initialize or reconfigure groups itself. Error cleanup
avoids new collectives and preserves the original model exception.

## MiniMax-H3 with Cache and Compile

Keep the complete workload and warm up the same computation and cache paths before collecting.
This t2va example uses TP2, text encoder parallelism, default DBCache, regional compile and native
attention. Replace the checkpoint path:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    bench/profile/run.py minimax_h3 --example-factory model.minimax_h3:create_example \
    --model-path /path/to/MiniMax-H3 \
    --parallel tp --parallel-text --cache --compile --attn native --vae-tiling \
    --steps 50 --height 448 --width 768 --num-frames 124 --seed 42 \
    --warmup 1 --warmup-num-inference-steps 50 --repeat 1 \
    --profile --profile-ranges 0-2,8-10 --no-profile-with-stack \
    --profile-name h3 --profile-dir /tmp/cache_dit_profiles
```

This completes all 49 steps and decoding, exporting two Torch traces per rank. Collect the same
windows with stage and module annotations in a separate Nsight run:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=src \
nsys profile --trace=cuda,nvtx --cuda-flush-interval=100 --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=repeat:2:defer --kill=none \
    -o /tmp/minimax_h3_tp2 \
    python -m torch.distributed.run --standalone --nproc_per_node=2 \
        bench/profile/run.py minimax_h3 --example-factory model.minimax_h3:create_example \
        --model-path /path/to/MiniMax-H3 \
        --parallel tp --parallel-text --cache --compile --attn native --vae-tiling \
        --steps 50 --height 448 --width 768 --num-frames 124 --seed 42 \
        --warmup 1 --warmup-num-inference-steps 50 --repeat 1 \
        --profile --profile-collector nsys --profile-ranges 0-2,8-10 --layerwise-nvtx
```

Expect two reports, each containing both ranks. Use `all` and `repeat:1:defer` for a separate complete
request including encoding and decoding. Increasing repeats also increases the report count.

The `100` ms flush interval was used after a buffering warning with Nsight 2026.6.1; it is an
external buffering choice, not a window requirement. Inspect diagnostics for missing or incomplete
data. Process exit alone does not prove capture integrity.

## NVTX Annotations

The runner installs module hooks once after optimization and its original warmup, keeps them across
formal repeats, then removes them on normal or exceptional exit. `--nvtx` enables stages without
resolving module roots. `--layerwise-nvtx` also annotates final transformer roots, including
`transformer_ref` for ref2va and applicable secondary transformers. Text encoders, VAEs and
schedulers are not automatically traversed by transformer hooks.

The H3 factory provides these explicit stages:

| Stage | Scope |
|---|---|
| `encode_text` | Text encoding, including tokenization and state updates |
| `denoise_loop` | State preparation, the complete loop and state writeback |
| `denoise_step i` | A complete logical step, including progress |
| `predict` | The original denoiser call |
| `scheduler_step` | Both video and audio scheduler updates and latent writes |
| `decode_video`, `decode_audio` | Decoding and output conversion |

FL2VA and Ref2VA share denoising and decoding stages; their conditioning encoders are not yet
annotated. Other models supply their own stage integration. NVTX labels do not automatically appear
in Torch traces. Use CPU `record_function()` annotations where Torch labels are needed.

Hooks run outside Dynamo. Modules compiled with `nn.Module.compile()` retain their outer labels,
while compiled descendants are left unmodified to preserve their existing guards. Cache skips,
direct `forward()` calls and graph replay can reduce visible labels. See the
[library visibility limits](../../docs/user_guide/PROFILER.md#visibility-with-compile-and-cuda-graph)
before interpreting other compile or graph combinations. Warmup with different shapes or too few
steps may still leave compilation in formal requests.

## Validation Scope

The existing regression suites cover all three H3 workflows with real CPU/CUDA tiny loops, output
and RNG comparisons, both schedulers, progress, request windows and TP2 warmed compilation. The
runner tests cover default/explicit factories, non-H3 step integration, independent NVTX switches
and cleanup. Full-checkpoint evidence is scoped to H3 t2va with TP2/TE-P2, cache and regional compile;
FL2VA/Ref2VA full checkpoints, CUDA Graph, other compile modes, NPU and Ray require separate checks.

Keep source versions, workload, raw outputs, collector diagnostics and report checks with each
measurement. Use a separate unprofiled run for speed comparisons: collection, synchronization,
annotation and export add overhead to the reported inference time.
