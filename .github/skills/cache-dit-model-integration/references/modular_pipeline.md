# Diffusers Modular Pipeline Reference

When to read this file: the target model is only available as a diffusers **modular pipeline**
(`ModularPipeline` + `ModularPipelineBlocks`, e.g. FLUX.3), i.e. `pipe` is neither a
`DiffusionPipeline` nor a `torch.nn.Module`. Read this before the Cache chapter; the rest of the
workflow (Cache → CP → TP → TE-P → VAE-P → CLI → Testing) is unchanged.

Reference integration: FLUX.3 (`Flux3Transformer2DModel` + `Flux3DistilledModularPipeline`).

## 1. What is different from a DiffusionPipeline

| Aspect | `DiffusionPipeline` | `ModularPipeline` |
|---|---|---|
| Type | `torch.nn.Module` subclass | plain Python object (`ConfigMixin`, `PushToHubMixin`) |
| Loading | `Cls.from_pretrained(id)` | `ModularPipeline.from_pretrained(id)` → then `pipe.load_components(dtype=...)` |
| Device move | `pipe.to("cuda")` | `pipe.to("cuda")` (same) |
| Call | `pipe(...)` returns `BaseOutput` | `pipe(..., output="images")` returns the selected state value; **without `output=` it returns the whole `PipelineState`** |
| Denoise loop | hard-coded in `__call__` | `LoopSequentialPipelineBlocks`; the step calls `components.transformer(...)` |
| Extra inputs | `**kwargs` | fields flowing through `PipelineState` (`denoiser_input_fields`) |
| Components | attributes | attributes (`pipe.transformer`, `pipe.text_encoder`, `pipe.vae`) |

Consequences for cache-dit:

1. Every pipeline-level `isinstance(x, DiffusionPipeline)` check must also accept modular
   pipelines, otherwise `enable_cache(pipe)` raises `assert isinstance(pipe_or_adapter, BlockAdapter)`.
2. The per-step transformer kwargs are built by **introspecting `transformer.forward`**
   (`inspect.signature(...)`, then only the fields that match are forwarded). Cache-dit replaces
   `transformer.forward` to install the block hooks, so the replacement **must keep the original
   signature** — see §3.1.
3. There is no `pipe.__class__.__call__` special case: patching it (as cache-dit already does for
   `DiffusionPipeline`) works, which is what resets the cache context once per inference.

## 2. Core support (cache-dit side)

Added once for all modular pipelines; a new modular model does not repeat this work.

| File | Change |
|---|---|
| `src/cache_dit/utils.py` | `is_modular_pipeline()` / `is_pipeline()` / `_resolve_pipe()`; all `check_*` helpers and `parse_extra_modules` resolve the pipe through `_resolve_pipe()` |
| `src/cache_dit/caching/block_adapters/block_registers.py` | `get_adapter()` only wraps a bare transformer in `FakeDiffusionPipeline` (a modular pipeline is a real pipe); `has_separate_cfg()` accepts it |
| `src/cache_dit/caching/cache_adapters/cache_adapter.py` | `CachedAdapter.apply()` accepts a modular pipeline; **`mock_transformer()` preserves the original `forward` signature**; `release_hooks()` treats it as a pipe |
| `src/cache_dit/caching/cache_interface.py` | transformer collection and `pipe` resolution during `enable_cache()` |
| `src/cache_dit/summary.py` | `summary()` / `strify()` treat a modular pipeline as a pipe, so `cache_dit.summary(pipe)` and `--cache-summary` work |
| `src/cache_dit/_utils/registers.py` | `ExampleInitConfig.is_modular_pipeline` + modular `get_pipe()` + `Example.call_pipe()` (`output=` selector) |

Check the guards still exist before integrating a new modular model:

```bash
grep -rn "is_pipeline\|is_modular_pipeline" src/cache_dit/utils.py \
  src/cache_dit/caching/block_adapters/block_registers.py \
  src/cache_dit/caching/cache_adapters/cache_adapter.py \
  src/cache_dit/caching/cache_interface.py src/cache_dit/summary.py
grep -n "__signature__" src/cache_dit/caching/cache_adapters/cache_adapter.py
```

## 3. Cache: which block list to cache

A modular transformer can own several block stacks. FLUX.3 has four:

| Stack | Blocks | Tokens per block call | `ForwardPattern` |
|---|---|---|---|
| `txt_mode_blocks` | 5 | text only (right-padded to a multiple of 80) | `Pattern_3` |
| `mode_blocks["image"]` | 5 | image tokens | `Pattern_3` |
| `mode_blocks["image_cond"]` | 5 | reference-image tokens (I2I only) | `Pattern_3` |
| `single_transformer_blocks` | 28 | joint `[text, image(, reference)]` | `Pattern_1` |

Two constraints decide the adapter:

1. **`blocks_name` must be an attribute of the transformer.** `mock_transformer()` installs the
   wrapper with `unittest.mock.patch.object(self, name, ...)`, so a nested list such as
   `mode_blocks["image"]` cannot be named. Only top-level `nn.ModuleList` attributes can be cached
   without a `PatchFunctor` that first aliases them (`self.image_mode_blocks = ...`) **and** rewrites
   the loop in `forward()` to iterate the alias.
2. **Cache the stack that carries the whole sequence.** The joint stack is where text and image
   tokens attend to each other, and it holds most of the FLOPs (28/43 blocks, `text + image`
   tokens). The per-mode stacks are cheap (text is short; the reference stream only exists for I2I)
   and recomputing them every step keeps the cached joint residuals anchored to the current
   per-mode features.

Adapter (shipped as `flux3_adapter` in `adapters.py`):

```python
@BlockAdapterRegister.register("Flux3")
def flux3_adapter(pipe, **kwargs) -> BlockAdapter:
  from diffusers import Flux3Transformer2DModel

  _relaxed_assert(pipe.transformer, Flux3Transformer2DModel)

  # CFG (only the non-distilled checkpoint) runs cond and uncond as two separate forwards
  guider = getattr(pipe, "guider", None)
  has_separate_cfg = bool(guider is not None and getattr(guider, "num_conditions", 1) > 1)

  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.single_transformer_blocks,
    forward_pattern=ForwardPattern.Pattern_1,
    check_forward_pattern=True,
    has_separate_cfg=has_separate_cfg,
    **kwargs,
  )
```

**Registry ordering**: `BlockAdapterRegister` returns the *first* class-name prefix that matches,
so `"Flux3"` must be registered **before** `"Flux"` (a `Flux3*` class name also starts with
`Flux`, and the Flux adapter's `_relaxed_assert` would reject it). The Flux3 adapter therefore sits
above `flux_adapter` in `adapters.py`.

### 3.1 The signature pitfall (silent, corrupts every step)

`mock_transformer()` replaces `transformer.forward` with a `functools.partial` of
`new_forward_with_hf_hook(self, *args, **kwargs)`. A modular denoiser step builds its transformer
kwargs from `inspect.signature(components.transformer.forward)`, so after the patch the signature
collapses to `(*args, **kwargs)` and every required input (`img_ids`, `txt_ids`,
`image_cond_*`) is silently dropped:

```
TypeError: Flux3Transformer2DModel.forward() missing 2 required positional arguments: 'img_ids' and 'txt_ids'
```

The fix (already in `cache_adapter.py`) copies the original signature onto the patched callable:

```python
try:
  new_forward_with_hf_hook.__signature__ = inspect.signature(original_forward)
except (TypeError, ValueError):
  pass
```

Symptom to remember: **the run crashes on the first denoise step, right after
`Match Blocks: CachedBlocks_...` is logged.** If a future modular model reports missing kwargs
after `enable_cache`, check this first.

## 4. CLI template

Three pieces, all in `_utils/`:

```python
# examples.py — one example per workflow (T2I / edit), plus a model-specific cache config
def _model_drop_unused_step(pipe) -> None:
  # `pipe.blocks` returns a deepcopy; pop from the live tree
  pipe._blocks.sub_blocks.pop("prompt_upsampler", None)
  # components are loaded before the hook runs: free the ones the pruned step owned
  pipe.unload_components(["prompt_upsampler", "prompt_upsampler_processor"])


@ExampleRegister.register("flux3", default="black-forest-labs/flux-3-image-klein")
def flux3_example(args, **kwargs) -> Example:
  from diffusers import Flux3DistilledModularPipeline
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path("black-forest-labs/flux-3-image-klein"),
      pipeline_class=Flux3DistilledModularPipeline,
      bnb_4bit_components=[],                 # modular pipelines do not take a quantization_config
      post_init_hook=_model_drop_unused_step, # optional: prune blocks the example does not use
      extra_optimize_kwargs={"cache_config": _flux3_cache_config(args)},
    ),
    input_data=ExampleInputData(prompt=..., height=1024, width=1024, num_inference_steps=16),
  )
```

* `ExampleInitConfig.get_pipe()` branches on `is_modular_pipeline`:
  `pipeline_class.from_pretrained(path, trust_remote_code=...)` → `load_components(dtype=...)` →
  `post_init_hook(pipe)`.
* `Example.call_pipe()` adds `output="images"` for `T2I` / `IE2I`, and `Example.run()` accepts
  either a `BaseOutput` (`.images`) or a plain list.
* `_env_path_mapping` needs a `<MODEL>_DIR` entry; `__all__` and `_utils/__init__.py` need the new
  factory.
* Cache defaults: pass a model-specific `cache_config` like the Qwen-Lightning examples do. For a
  16-step distilled flow-map model the generic `max_warmup_steps=8`, `residual_diff_threshold=0.12`
  caches nothing (measured diffs are `0.20–0.80`); `Fn=4, warmup=4, threshold=0.30` caches
  `[5, 7, 9, 11, 13]` of 16 steps.

## 5. Parallelism on a modular pipeline

| Feature | Status | Notes |
|---|---|---|
| TP | works | `pipe.transformer` is collected through the adapter; add a `TensorParallelismPlanner` under the exact transformer class name (`"Flux3Transformer2DModel"`) |
| CP | works | one **root split** covers FLUX.3's three token streams; see below |
| TE-P | works | the planner is selected by the text-encoder class (`"Qwen3VLForConditionalGeneration"`); `parse_extra_modules(pipe, ...)` resolves the modular pipe through `_resolve_pipe()` |
| VAE-P | works | selected by the VAE class (`"AutoencoderKLFlux2"`) |

### 5.1 CP: why one root split is enough although every block rebuilds `torch.cat`

FLUX.3 keeps three token streams with different lengths (text padded to a multiple of 80, image
`H/16 x W/16`, each reference its own grid), and every block re-assembles the sequence locally:

* joint stack: `content_states = cat([image, reference])`, then the block does
  `combined = cat([text, content])` and `torch.split(output, [text_len, content_len])`;
* per-mode stacks: one stream each.

A root split therefore hands each rank a **contiguous chunk of each stream**, so after the local
`cat` a rank holds `[text_c_i, image_c_i, reference_c_i]` — a *permuted* slice of the global order
`[text, image, reference]`. That is harmless because

* attention has **no mask** (softmax over the key set is order-invariant), and
* RoPE is applied **per token from the ids** before the all-to-all, so every token carries its own
  position.

The union of the ranks' local chunks is exactly the global token set, so each block's attention
equals full-sequence attention; the inverse all-to-all restores the local layout, which is what the
in-block `torch.split`, the residual adds and the final
`content_states[:, :image_seq_len]` slice need. The per-mode modulations are per-token broadcasts of
one timestep embedding, so they are simply recomputed from the local lengths.

```python
_cp_plan = {
  "": {
    "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    "image_cond_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    "img_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
    "txt_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
    "image_cond_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
  },
  "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

Two **engine** limitations had to be fixed first — both hit any modular model, not FLUX.3:

1. `_ModuleForwardMetadata.get_parameter_from_args_kwargs` raised
   `ValueError: Expected N arguments but got 0` when a planned input was absent. A modular step
   passes every input as a keyword argument and legitimately omits workflow-specific ones
   (`image_cond_*` for text-to-image), so an absent planned input must resolve to `None` (skip the
   split) instead of raising.
2. `HookRegistry.register_hook` replaced `forward` with a `(*args, **kwargs)` wrapper, which
   collapsed the signature the modular step introspects → `missing 2 required positional arguments:
   'img_ids' and 'txt_ids'`. It now forwards the previous `__signature__`, same rule as
   `CachedAdapter.mock_transformer`.

Non-divisible stream lengths (e.g. `H/16 x W/16` odd, so `1089 % cp_size != 0`) need
`--ulysses-anything`; without it the split fails with "Tensor size along dimension to be sharded
must be divisible by mesh size".

TP planner notes for a FLUX.3-like FFN (`Flux3FeedForward`):

* `linear_in` fuses the SwiGLU gate and up projections into `[2 * inner_dim, dim]`, and
  `Flux3SwiGLU` splits the last dim in half. A plain `ColwiseParallel` would hand the gate rows to
  one rank and the up rows to the other, so each local SwiGLU would mix gate with gate.
* Fix: reorder the weight to `[gate_0, up_0, gate_1, up_1, ...]` (one `(gate, up)` pair per TP
  chunk) before `parallelize_module`; `linear_out` is `RowwiseParallel` and shards its input columns
  contiguously in the same order. See `rearrange_ffn_weight()` in
  `src/cache_dit/distributed/transformers/flux3.py`.
* `shard_div_attr(block.attn, "heads", tp_size)` for every block, because the processor reshapes
  with `attn.heads` (24 → 12 for `tp_size=2`).
* Leave `early_stream_modulations` / `single_stream_modulations` replicated: they produce per-token
  `shift/scale/gate` that are applied elementwise to the full hidden states, and the block does
  `torch.chunk(temb_mod, 3, dim=-1)` — a sharded last dim would break that split.

## 6. Verification checklist

```
- [ ] baseline: pipe(prompt, output="images") → image loads, looks right
- [ ] enable_cache(pipe) succeeds, log shows "<PipeClass> is officially supported"
- [ ] first denoise step does not raise missing-kwargs (signature pitfall)
- [ ] cache run: same seed, cached steps list non-empty, latency lower
- [ ] PSNR > 30 dB and SSIM > 0.90 vs baseline (both!)
- [ ] run pipe(...) twice: second call must reproduce the first bit-exactly
      (proves the per-inference context reset works for the modular pipe)
- [ ] I2I path (reference image) with the same adapter
- [ ] cache_dit.summary(pipe) / --cache-summary prints the stats
- [ ] CLI: python3 -m cache_dit.generate <name> list entry, baseline + --cache
- [ ] CP: torchrun --nproc_per_node=2 -m cache_dit.generate <name> --parallel ulysses
      (and a non-divisible resolution with --ulysses-anything)
- [ ] python3 -m pytest tests/api/test_modular_pipeline.py -q
      (guards the forward-signature preservation and the TP FFN rearrangement)
```

## 7. FLUX.3 case study (numbers)

Model: `black-forest-labs/flux-3-image-klein` (distilled, `flow_map=True`, 16 steps, no CFG),
bf16, 1024×1024, RTX PRO 5000 72GB. `prompt_upsampler` popped (it is a separate checkpoint).

Single GPU, T2I, CLI:

| Config | Latency | PSNR / SSIM vs baseline |
|---|---|---|
| baseline | 4.90 s | ∞ / 1.0000 |
| cache `F4 B0 W4 R0.30` (5/16 steps cached) | 3.86 s (1.27×) | 34.21 / 0.9569 |

Threshold/tuning sweep (T2I, same seed, denoise phase 4.72 s):

| Fn | Bn | warmup | threshold | cached steps | denoise | PSNR / SSIM |
|---|---|---|---|---|---|---|
| 8 | 0 | 4 | 0.30 | 5 | 3.84 s | 34.92 / 0.9764 |
| 4 | 0 | 4 | 0.30 | 5 | 3.67 s | 35.26 / 0.9723 |
| 2 | 0 | 4 | 0.30 | 5 | 3.58 s | 35.18 / 0.9714 |
| 4 | 0 | 4 | 0.40 | 7 | 3.25 s | 32.39 / 0.9370 |
| 8 | 0 | 4 | 0.20 | 0 | 4.72 s | ∞ / 1.0000 |

I2I, CLI `flux3_edit` (reference = `examples/data/edit2509_2.jpg`, `max_warmup_steps=8`):
10.42 s → 8.12 s (1.28×), 5/16 steps cached, PSNR 34.87 / SSIM 0.9670.
Image conditioning changes less per step, so it caches earlier than T2I — and reusing an **early**
step is what breaks it: the same run with `max_warmup_steps=4` caches steps `[4, 6, 7, 9]` and drops
to PSNR 31.62 / SSIM 0.8876 (fails the 0.90 bar) even though only 4 steps are cached. Keep the
warmup at half of the steps for edit workflows.

I2I with a stronger conditioning change (reference = the knitted-diorama baseline, "same scene at
night"): 10.47 s → 6.77 s (1.55×), 8/16 steps cached, PSNR 39.03 / SSIM 0.9963.

2 GPUs:

| Config | Latency | PSNR / SSIM |
|---|---|---|
| TP (2 GPU) | 3.02 s (1.62× vs 1 GPU) | 42.02 / 0.9940 (vs 1 GPU baseline) |
| TP + TE-P | 3.07 s | 36.22 / 0.9600 |
| TP + VAE-P | 3.03 s | 42.02 / 0.9940 (bit-identical to TP) |
| TP + cache | 2.43 s (2.02× vs 1 GPU) | 34.09 / 0.9552 (vs TP baseline) |
| CP Ulysses (2 GPU) | 3.20 s (1.53×) | 40.13 / 0.9899 (vs 1 GPU baseline) |
| CP Ring (2 GPU) | 2.95 s (1.66×) | 38.48 / 0.9816 |
| CP Ulysses + cache | 2.54 s (1.93×) | 33.95 / 0.9527 |
| CP Ulysses, I2I (`flux3_edit`) | 6.58 s (1.58× vs 10.42 s) | 40.28 / 0.9849 |
| CP Ulysses + UAA, 528×528 (1089 image tokens) | 1.08 s | 46.28 / 0.9987 (vs 1 GPU 528×528) |

On 2 GPUs CP and TP land at a similar throughput (1.53–1.66× vs 1.62×). Pick CP when the token
streams dominate memory (it shards the sequence of all three streams, including the long text
stream); pick TP when the weights dominate. `parallelism_config` also allows combining them
(`--parallel ulysses_tp` / `ring_tp`), which is untested for FLUX.3.

### 7.1 Quantization: fp8 per-row + compile

`python3 -m cache_dit.generate flux3 --quantize --compile [--cache]` (`--quantize` defaults to
torchao `float8_per_row`, i.e. `Float8DynamicActivationFloat8WeightConfig`). Regional quantization
covers the 4 block stacks only: 258 linear layers (`43 blocks x 6`), 0 skipped.

| Config | Latency | Peak mem | PSNR / SSIM vs bf16 baseline |
|---|---|---|---|
| bf16 baseline | 4.91 s | 21.21 GiB | — |
| bf16 + cache | 3.86 s (1.27×) | — | 34.21 / 0.9569 |
| bf16 + compile | 3.96 s (1.24×) | — | 38.00 / 0.9781 |
| bf16 + compile + cache | 3.12 s (1.57×) | — | 34.00 / 0.9519 |
| fp8 per-row | 4.40 s (1.11×) | — | 32.32 / 0.9257 |
| fp8 per-row + compile | 2.36 s (2.08×) | — | 31.85–32.81 / 0.906–0.943 |
| fp8 per-row + compile + cache | **1.88 s (2.60×)** | **16.39 GiB** | 31.81–32.41 / 0.895–0.935 |

**fp8 needs compile to pay off.** fp8 alone only reaches 1.11× (4.90 s → 4.40 s) because torchao's
rowwise `addmm_float8_unwrapped_inference` (`_scaled_mm`) runs unfused in eager; under
`torch.compile` it is lowered into a compiled subgraph (cache-dit ships that wiring in
`quantization/torchao/_scaled_mm.py`), and the combination reaches 2.08× — more than fp8 (1.11×)
and compile (1.24×) add up to.

**`--compile` is not bit-reproducible.** Two identical runs of the same config differ by
PSNR ≈ 34 dB / SSIM ≈ 0.93 (fp8) and ≈ 38 dB / 0.97 (bf16), while the non-compiled runs are
bit-identical (PSNR inf). The cached-step set and latency stay stable (5/16 steps, ±0.01 s), so the
drift is Inductor kernel/fusion numerics amplified by the 16-step sampler — budget for it when
judging fp8 on high-frequency content: SSIM lands in a 0.89–0.94 band, i.e. right at the 0.90
acceptance bar, whereas PSNR stays comfortably above 30.
