# Custom (Non-diffusers) Model Reference

When to read this file: **two distinct tasks share this workflow**, and both must start here.

1. **Adding a model that has no diffusers implementation.** The checkpoint ships its own
   `pipeline_<name>.py` / transformer code, `DiffusionPipeline.from_pretrained("owner/repo")` cannot
   import it, or the code only lives in a third-party package.
2. **Absorbing a diffusers model into cache-dit** — copying the model source into this repo so
   cache-dit owns it, instead of monkey-patching diffusers from the outside. This is the path for
   deep optimization (custom/fused kernels, changed attention math, a different block loop): once the
   change has no hook point, patches stop being maintainable.

Read this **before** the Cache chapter in `../SKILL.md`. The phase order (Cache → CP → TP → TE-P →
VAE-P → CLI → Testing) does not change, but three premises do: where the code lives, how it is loaded,
and which dispatch contracts the owned code must keep for cache-dit to see it at all.

Reference integration: **MOSS-Image** (`OpenMOSS-Team/MOSS-Image-SFT` / `-Base`), PR #1149 — entry
point 1. Entry point 2 has no precedent in cache-dit yet: §3 is the runbook, not a measured case.
File paths are relative to the cache-dit repo root.

## 1. Two entry points

### 1.1 Entry point 1 — the model is absent from diffusers

Three flavours; detect first:

```bash
# What does the checkpoint think its pipeline / transformer is?
cat <ckpt>/model_index.json            # "_class_name", and the module of each component
cat <ckpt>/transformer/config.json     # transformer "_class_name"

# Can diffusers import those names?
python -c "import diffusers; print(hasattr(diffusers, '<ClassName>'))"
```

| Flavour | Signal | What it needs |
|---|---|---|
| **A. Checkpoint ships its own pipeline** | `model_index.json` names a `pipeline_<model>` module, or the repo holds `pipeline_*.py` / `transformer_*.py` next to the weights (MOSS-Image; SANA-Video 2.0 ships a native non-diffusers pipeline) | **Vendor** the release code into cache-dit, then write the adapter (§4–§6) |
| **B. Code lives in an installed third-party package** | the pipeline/transformer class imports from a package, e.g. `from boogu.pipelines.boogu.pipeline_boogu import BooguImagePipeline`, or `sglang.multimodal_gen...` | **Adapter only**; no vendoring, the CLI example imports that package |
| **C. User supplies an importable class** | a local checkout or `trust_remote_code` repo the user can point at | **Adapter only**; possibly a transformer-only adapter (no pipeline) |

Checklist item 0 in your plan: state which flavour the model is, and why.

### 1.2 Entry point 2 — the model is in diffusers, but cache-dit takes ownership

The model already runs through cache-dit, usually with a pile of patches. Migrate when any of these
is true:

* the optimization changes **math or tensor layout inside the block** (fused/quantized kernel, a
  different attention layout, changed RoPE), i.e. there is no hook point left to patch;
* several patches target the same class — `PatchFunctor` rewriting `forward`, a class-level attention
  processor patch, a loader patch for checkpoint conversion, `declare_repeated_blocks()` shims — and
  they now interact;
* you need to change the **block loop itself** (extra per-step state, different traversal order,
  conditional block lists), which a hook-based patch cannot express;
* you want the model to compile as one unit with your kernels instead of holding a patched copy of
  upstream code.

The benefit is not speed by itself — it is that the optimization becomes first-class, reviewable, and
independent of upstream attribute names. The cost is that you now own the code and its drift (§3.5).

**A migration must not be a rewrite.** Copy the diffusers source verbatim (same class names, same
state-dict keys, same math), then change only what the optimization needs. Keep the file layout close
to upstream so a re-sync stays a small diff.

### 1.3 Entry point 3 — second-pass optimization by subclassing (`_NAME`)

The model is in diffusers, runs through cache-dit, and the optimization touches **only seams**: the
block forward, the FFN forward, the attention QKV preparation, the transformer forward. No attention
math, no block-loop change, no checkpoint conversion. Then do **not** absorb it (§3) and do not pile
`PatchFunctor`s on it — subclass in place:

```python
# src/cache_dit/models/pipelines/_<model>/pipeline_<model>.py
class _<Model>Pipeline(<Model>Pipeline):          # diffusers class, one underscore in front
  ...
# src/cache_dit/models/pipelines/_<model>/transformer_<model>.py
class _<Model>Transformer2DModel(<Model>Transformer2DModel):
  ...
```

`_` means *"diffusers owns this model; cache-dit ships an optimized pass over it"*. The user opts in by
naming the subclass; `from_pretrained` keeps working and the state dict is untouched (§15.3).

| | Subclass in place (`_NAME`) | Absorb / vendor (§3) |
|---|---|---|
| Optimization lives in seams (block/FFN/QKV/transformer forward) | **yes** | overkill |
| Changes attention math, block loop, tensor layout, or checkpoint keys | no | **yes** |
| Drift cost | ~zero (upstream stays the implementation) | you own the code |
| Registry work | `_NAME` prefixes everywhere (§15.4) | nothing extra |

Full runbook, including the worked `_QwenImage21` case: **§15**.

## 2. Ownership decision: vendor or depend

| Option | Choose when | Cost |
|---|---|---|
| Vendor into `src/cache_dit/models/pipelines/<model>.py` | Entry point 1, flavour A. The release's own code is the only runnable path and the licence allows redistribution (MOSS-Image is Apache-2.0; keep the copyright line in the module docstring) | You own the code: every contract in §4 must be preserved *in the vendored copy*, and the copy drifts from upstream |
| Absorb (fork) a diffusers model | Entry point 2 (§1.2) | You own upstream's code too: re-sync work, Apache-2.0 notice, and the whole feature matrix must be re-verified (§3.6) |
| Depend on a third-party package | Entry point 1, flavour B. The package is already the product's stack | Extra install requirement; keep the import lazy (inside the adapter / example factory) so the rest of cache-dit works without it |
| Subclass in place (`_<Model>Pipeline`) | Entry point 3 (§1.3): the model stays diffusers', the optimization only rebinds seams | Every prefix-keyed registry needs the `_NAME` entry (§15.4); the seam set decides whether CP / flex attention keep the fusion (§15.5) |
| Transformer-only integration | Flavour C, or the pipeline is trivially replaceable | `BlockAdapter` wraps the bare transformer in `FakeDiffusionPipeline`; CP/TP work, pipeline-level features (offload, `summary(pipe)`) degrade |

`models/__init__.py` and `models/pipelines/__init__.py` keep `__all__ = []` on purpose: the vendored
module is imported lazily (CLI example / adapter), so importing `cache_dit` never pulls model code.

## 3. Absorbing a diffusers model (migration runbook)

### 3.1 What changes when `__module__` stops starting with `diffusers`

Every diffusers-detection branch in cache-dit flips when the class moves into `cache_dit.*`. Check all
of them before declaring the migration done:

| Check site | Before (diffusers) | After (owned copy) | Required action |
|---|---|---|---|
| `BlockAdapter.maybe_skip_checks()` (`caching/block_adapters/block_adapters.py`) | `check_forward_pattern` auto-True | auto-**False** (logs "NOT from diffusers") | pass `check_forward_pattern=True` explicitly in the adapter |
| `_relaxed_assert()` (`caching/block_adapters/adapters.py`) | strict `isinstance` check | skipped with a warning | keep the call for documentation, or assert the stack/class directly |
| `PatchFunctor.apply()` / `is_from_diffusers()` (`caching/patch_functors/functor_base.py`) | functor runs | **functor is skipped** (one warning line) | fold the patch into the owned `forward()` and delete the functor — this is the point of migrating |
| `check_text_encoder()` / `check_auto_encoder()` (`utils.py`, used by `distributed/dispatch.py`) | matches `transformers` / `diffusers.models.autoencoder` module paths | a locally copied TE/VAE no longer matches, so `--parallel-text` / `--parallel-vae` are skipped | do **not** move the text encoder or VAE (§3.2) |
| CP/TP planner registry (`get_planner`) | keyed by class name | unchanged | keep the class name identical (§3.3) |
| Adapter registry (`BlockAdapterRegister`) | keyed by class-name prefix | unchanged | keep the class name identical (§3.3) |
| Ray worker importability (`ray/engine.py`) | `diffusers` is a well-known prefix | `cache_dit` is in `_well_known` too | none |

Two escape hatches exist during an incremental migration, and both are traps if left on:
`CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK=1` lets a leftover functor still run (useful to A/B
the old path, wrong as a permanent state), and `check_forward_pattern=None` silently disables the
pattern check — always set the flag explicitly.

### 3.2 Scope: what to move

| Component | Move it? | Why |
|---|---|---|
| Transformer + its block classes | **Yes** — this is the point | the optimized kernels live here |
| Pipeline `__call__` | Only when the loop itself is the target (per-step state, conditioned traversal, changed VAE/CFG bookkeeping) | otherwise keep the diffusers pipeline and let the adapter point at your transformer |
| Scheduler | Keep the diffusers class and subclass it if only `set_timesteps` / `step` differ (MOSS: `MOSSImageScheduler(FlowMatchEulerDiscreteScheduler)`) | a subclass keeps `scheduler/` loading and config keys working |
| VAE / text encoder | **No**, unless you also add their planners | `_is_auto_encoder()` / `_is_text_encoder()` gate the VAE-P / TE-P dispatch on module path (§3.1). A copied VAE silently disables `--parallel-vae` |
| ControlNet | No | `_is_controlnet()` matches `diffusers.models.controlnet` |

### 3.3 Naming and checkpoint compatibility

Keep these **identical to upstream**; they are load-bearing strings, not style:

* `TransformerCls` class name — adapter prefix matching, CP/TP planner lookup, and the checkpoint's
  `transformer/config.json::_class_name`.
* Block class name and the block stack attribute name (`transformer_blocks`, `layers`, ...) —
  `blocks_name` in the adapter and `_repeated_blocks` on the class.
* Every `state_dict` key. The acceptance test is `load_state_dict(diffusers_state_dict, strict=True)`
  passing with zero missing/unexpected keys; `strict=False` hides a renamed projection and changes
  numerics without an error.
* The prompt/`__call__` keyword names, so the existing CLI example and users keep working.

Renaming is only justified when the name collides with another adapter prefix (§12.8).

### 3.4 Fold the patches in, then delete them

In one migration commit:

```
- [ ] inline the PatchFunctor's forward rewrite into the owned forward() (and delete the functor)
- [ ] inline the class-level attention-processor patch; call `_dispatch_attention_fn` directly if the
      model has no diffusers `Attention` module ([`./attention.md`](./attention.md))
- [ ] inline the loader conversion, or keep `BaseLoader` if it is genuinely checkpoint-specific
      ([`./loader.md`](./loader.md))
- [ ] replace `declare_repeated_blocks()` shims with a `_repeated_blocks` class attribute
- [ ] move `_cp_plan` construction into the owned class / planner, keep `_cp_config` reads
- [ ] keep reading `_cp_config` and honouring `transformer._is_cached` (the pipeline guards)
- [ ] delete the functor/loader registration lines and the tests that only covered the patch
- [ ] drop the model from the "patched at runtime" list in docs
```

### 3.5 Drift control

* Keep the Apache-2.0 header of the diffusers files and add a module docstring line with the upstream
  **tag/commit** the copy came from ("copied from diffusers vX.Y.Z, `models/transformers/<file>.py`").
* Maintain a short list of intentional diffs (the optimizations) in that docstring. That list is the
  spec: a reviewer must be able to see what changed and why without diffing two repos.
* Re-sync on upstream releases by re-copying the file and re-applying the listed diffs — not by
  merging patch code back in. Format the copy to the repository style (2-space indent, Sphinx
  docstrings) once, so later diffs stay small.

### 3.6 Verification (stricter than a normal integration)

```
- [ ] state_dict: strict=True from the diffusers checkpoint, zero missing/unexpected keys
- [ ] bit-exact A/B against the diffusers model before any optimization: same seed, same prompt,
      compare latent + decoded image with torch.equal (not PSNR)
- [ ] re-run the full matrix on the owned copy: cache / CP / TP / TE-P / VAE-P / quant / compile
      (each was verified against a diffusers class, and the code path is now different)
- [ ] CPU unit tests for the owned block/pipeline ([`./testing.md`](./testing.md))
- [ ] record a fresh baseline latency/PSNR before starting the optimization work
```

Once the first optimization lands, bit-exactness will legitimately break; from that point the
acceptance bar is PSNR/SSIM against the pre-optimization baseline, plus the FLOPs story in
[`./perf_analysis.md`](./perf_analysis.md).

## 4. Contracts the vendored code must keep

These are not style preferences. Each row maps to a cache-dit call that silently degrades or fails
when broken. They apply to both entry points.

| Contract | Why it matters | Where cache-dit depends on it |
|---|---|---|
| `PipelineCls(DiffusionPipeline)` | `is_pipeline()` must return True, and `CachedAdapter.apply` patches `pipe.__class__.__call__` to reset the cache context once per inference | `utils.is_pipeline`, `CachedAdapter.apply` |
| `TransformerCls(ModelMixin, ConfigMixin)` | `from_pretrained` / `config` / `register_to_config` are used by TP, quantization and `summary` | `ModelMixin` assertions in the adapters |
| `vae` is a **diffusers** autoencoder (`AutoencoderKL*`) | `_is_auto_encoder()` matches on `module.__class__.__module__.startswith("diffusers.models.autoencoder")`; a local copy makes `--parallel-vae` silently do nothing | `utils._is_auto_encoder` |
| `text_encoder` is a **transformers** module | same rule for TE-P (`startswith("transformers")`) | `utils._is_text_encoder` |
| Denoising stack is a **top-level `nn.ModuleList` attribute** | `mock_transformer` replaces it with `setattr(transformer, blocks_name, ModuleList([...]))`; a nested list/dict cannot be named | `CachedAdapter.mock_transformer` |
| The block loop calls **one block per iteration and nothing else** | the wrapper replays the block call; anything else in the loop body is skipped | §6.1 |
| Block's first parameter is named `hidden_states`, returns one tensor | `Pattern_3`; `check_forward_pattern=True` then validates it | `ForwardPattern` check |
| `_repeated_blocks = ["<BlockClass>"]` on the transformer | regional quantization and `compile_repeated_blocks()` discover the stack from it; without it both widen to the whole module (worse accuracy, slower compile) | `utils.declare_repeated_blocks` |
| Config holds the real depth (`num_layers`) | under cache the module list has **one** element, so `len(...)` is not the depth | §6.1 |

If the checkpoint ships a consolidated training state dict rather than a `save_pretrained` tree, add a
strict loader helper (MOSS: `MOSSImageTransformer.load_training_state_dict(state_dict, prefix=...)`)
instead of `strict=False`. For dtype/fused/renamed checkpoints see [`./loader.md`](./loader.md).

## 5. Loading: the `from_pretrained` contract

The CLI never "just calls" your loader. `ExampleInitConfig.get_pipe()` always passes this exact set:

```python
pipeline_class.from_pretrained(
  model_path,
  torch_dtype=self.torch_dtype,
  quantization_config=pipeline_quantization_config,
  device_map="balanced" if args.device_map_balance else None,
  trust_remote_code=self.trust_remote_code,
  **custom_components_kwargs,   # transformer / vae / text_encoder / scheduler / controlnet
)
```

For an **absorbed diffusers model you normally change nothing here**: the class still accepts the
diffusers signature, and `ExampleInitConfig` keeps loading `from_pretrained(model_path)`. The rows
below matter when you vendor a release whose loader is its own (entry point 1, flavour A):

| Argument | What the vendored loader should do | MOSS-Image |
|---|---|---|
| `torch_dtype` / `dtype` | accept both spellings; support a per-component mapping (`{"default": bf16, "vae": fp32}`) | `_resolve_dtypes()` |
| `quantization_config` | ignore with a warning: cache-dit quantizes the DiT itself, diffusers' quantizer cannot find the blocks | warn + drop |
| `device_map` | **raise** — the CLI skips `pipe.to(device)` whenever `--device-map-balance` is set, so accepting it would leave the pipeline on CPU | `NotImplementedError` |
| `trust_remote_code` | ignore; the code is local | debug log |
| `transformer` / `vae` / `text_encoder` / `scheduler` | honour them; the CLI uses them to inject pre-built components, and silently ignoring them breaks LoRA/scheduler overrides | used as-is |
| `controlnet` | reject if the model has no ControlNet path | `ValueError` |

Component assembly: read the subdirectories directly (`transformer/`, `text_encoder/`, `tokenizer/`,
`vae/`, `scheduler/`) instead of routing through `model_index.json`. Two traps:

* the tokenizer does not necessarily sit next to the text encoder — take an explicit
  `tokenizer_path` (MOSS's checkpoint keeps a separate `tokenizer/`), else `AutoTokenizer` loads the
  wrong files;
* some checkpoints ship `chat_template.jinja` as a sibling file with no inline `chat_template` in
  `tokenizer_config.json`; older transformers releases do not pick it up, so attach it explicitly
  (MOSS: `_ensure_chat_template`, a no-op on transformers 5.x).

## 6. Cache

Adapter skeleton (MOSS: `moss_image_adapter`, registered under `"MOSSImage"`):

```python
@BlockAdapterRegister.register("MOSSImage")
def moss_image_adapter(pipe, **kwargs) -> BlockAdapter:
  return BlockAdapter(
    pipe=pipe,
    transformer=pipe.transformer,
    blocks=pipe.transformer.layers,
    blocks_name="layers",
    forward_pattern=ForwardPattern.Pattern_3,
    # Non-diffusers transformers default this check OFF; for vendored code you control the
    # signature, so turn it on instead of relying on the lenient default.
    check_forward_pattern=True,
    # cond and uncond are two separate transformer.forward() calls per step.
    has_separate_cfg=True,
    **kwargs,
  )
```

* **Registry order**: `BlockAdapterRegister` returns the *first* class-name prefix that matches, so a
  new `"MOSSImage"`-style prefix must sit above any shorter prefix that also matches. A generic class
  name (`Transformer2DModel`) will be captured by an unrelated adapter — give the vendored class a
  distinctive name.
* **`_relaxed_assert`**: for a transformer whose `__module__` is not `diffusers`, it logs and skips the
  type check, so no allow-list is needed for vendored code — but nothing validates the wiring either.
  If the transformer owns more than one block stack, assert the chosen attribute in the adapter
  (`assert isinstance(pipe.transformer.layers[0], _Block)`), because a wrong stack only surfaces as a
  wrong image.
* **`has_separate_cfg`**: decide by counting `transformer.forward()` calls per denoising step, not by
  whether guidance is on. MOSS runs cond then uncond (two calls, independent cache contexts, one
  shared step counter) → `True`.

### 6.1 Branch/skip state must live *inside* the block

`mock_transformer` replaces the block list with a **one-element** `ModuleList`:

```python
torch.nn.ModuleList([UnifiedBlocks(...)])
```

Inside the patched `forward`, `for layer in self.layers:` therefore runs once and
`len(self.layers) == 1` — any `enumerate` index is 0. Consequences for vendored code:

1. **Per-block skipping cannot be done in the loop.** `if index in skip_layers: continue` in
   `transformer.forward` either disappears (the loop body is not replayed) or degenerates to index 0.
   Move it into the block and pass the set as an argument:

   ```python
   class _Block(nn.Module):
     def __init__(self, dim, ffn_dim, heads, index=0):
       super().__init__()
       self.index = index          # zero-based position; the wrapper keeps the original blocks
       ...

     def forward(self, hidden_states, rope, modulation, skip_layers=(), cp_config=None):
       if self.index in skip_layers:
         return hidden_states      # identity, so the cached graph stays identical
       ...
   ```

2. **Depth validation must use `config.num_layers`**, never `len(self.layers)`:

   ```python
   skip_layers = _validate_skip_layers(skip_layers, self.config.num_layers)
   ```

3. Any other tensor that varies per step (modulation, per-block cache, extra ids) has to arrive as a
   block **argument**. Loop-body operations are lost after the wrapper takes over; if the model has
   several of them, write a `PatchFunctor` (see [`./cache.md`](./cache.md)) instead of contorting the
   block signature.

### 6.2 Guard the pipeline entry point

A pipeline that can run a different number of forwards per step desynchronises cache-dit's separate-CFG
bookkeeping. Assert it in `__call__`, at the point where the mode is known:

```python
if getattr(self.transformer, "_is_cached", False):
  if not do_cfg:                    # guidance_scale == 1 -> one forward per step
    raise ValueError("cache-dit caching assumes one conditional and one unconditional forward per "
                     "step; use a guidance scale above 1 (or disable separate CFG).")
  if slg_layers and slg_scale > 0:  # skip-layer guidance -> a third forward per step
    raise ValueError("cache-dit caching assumes two forward passes per step; SLG adds a third. "
                     "Leave `slg_layers`/`slg_scale` unset when caching is enabled.")
```

The CLI example must then default to the guarded mode (`guidance_scale > 1`,
`num_images_per_prompt=1`, no SLG), otherwise the first `--cache` run fails on the user.

## 7. Context parallelism

CP needs **module boundaries**. Vendored transformers rarely have diffusers' `Attention` modules, so
add two zero-parameter `nn.Identity()` markers — they change nothing numerically and do not appear in
the state dict:

```python
self.rope_split = nn.Identity()        # output-split of the shared RoPE table
self.sequence_gather = nn.Identity()   # gather the joint sequence before the image slice
```

```python
_cp_plan = {
  # split_output: the split applies to output index 0 of the marker call.
  "rope_split": {0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
  # input split: keyed by parameter name; cache-dit also replays this hook for the Fn residual.
  "layers.0": {"hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3)},
  "sequence_gather": _ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

Rules that are easy to get wrong:

* **Split the RoPE table through the framework sharder** (`split_output=True`, no manual
  `torch.chunk`), or `--ulysses-anything` and plain Ulysses will disagree on the per-rank offsets.
  Split it *after* the stage that consumes the full-length table (MOSS: text refinement).
* **Gather before the local slice.** If `forward` slices image tokens out of the joint sequence,
  gather first, otherwise the velocity silently loses tokens on every rank but one.
* **No diffusers `Attention` module → dispatch directly**, and mind the layout:

  ```python
  out = _dispatch_attention_fn(q, k, v, dropout_p=0.0, is_causal=False, cp_config=cp_config)
  ```

  `_dispatch_attention_fn` (and the templated Ulysses backend) takes `[B, S, H, D]`, while
  `F.scaled_dot_product_attention` wants `[B, H, S, D]`. Feeding SDPA's layout into the CP branch
  does **not** raise: MOSS measured PSNR 27.9 / SSIM 0.09 (garbled) while the tiny fp32 parity test
  showed rel ≈ 6e-2. Keep `[B, S, H, D]` through RoPE, transpose only on the non-CP path.
* **Reject Ring when the joint length is not divisible** by the CP size (Ring needs equal chunks;
  UAA only covers Ulysses). Raise in the planner with the reason, as MOSS does.
* **Text/`encoder_attention_mask`**: the cheapest fix is to remove the mask before the split, as MOSS
  does — it compacts each sample with `encoder_hidden_states[i, mask[i].bool()]` before building the
  RoPE table, so no mask survives into the block loop. If the mask has to stay, split it with the same
  boundary as the sequence, otherwise a rank can end up with an empty text slice.

## 8. Tensor parallelism

* Planners are keyed by transformer **class name** (`"MOSSImageTransformer"`); register the exact
  name, and keep it distinct from other families.
* `nn.ModuleDict` projections keep dotted plan keys: `proj.q` / `proj.k` / `proj.v` / `proj.o`.
* **Fused FFN weights must be rearranged before `parallelize_module`.** MOSS packs
  `(hidden, gate)` into `Linear(dim, 2 * ffn_dim)` and consumes it with `chunk(2, -1)`; a plain
  `ColwiseParallel` gives one rank only half of each operand. Reorder rows to
  `[hidden_0, gate_0, hidden_1, gate_1, ...]` with the shared helper
  (`rearrange_ffn_0_swiglu_proj_weight`) and **assert** `2 * ffn_dim % (2 * tp_size) == 0` first,
  because the helper pads with zero rows instead of failing.
* `shard_div_attr(block, "heads", tp_size)` whenever the block reshapes with `self.heads`
  (MOSS: `unflatten(-1, (self.heads, -1))`). Per-head norms stay replicated.
* Everything outside the stack (embeddings, timestep MLP, modulation/head projections) stays
  replicated: the residual stream is full-length after each all-reduce.

## 9. TE-P and VAE-P

Both dispatch by **module path / class name**, so an owned copy silently disables them:

* VAE must be `diffusers.models.autoencoder.*` (§4). If the release ships its own VAE, either keep
  using an equivalent diffusers autoencoder (MOSS uses `AutoencoderKLFlux2`) or accept that VAE-P is
  unavailable and say so in the plan. The same applies to a migration that copies the VAE.
* Text encoders must be `transformers.*`. A family new to cache-dit needs its own planner under
  `distributed/text_encoders/<family>.py`; prefix matching means `Qwen3_5Model` does **not** match a
  `Qwen3` registration — register it explicitly.
* `--parallel-vae` needs a real distributed group: it must be combined with `--parallel tp/ulysses`,
  otherwise both ranks land on GPU 0. See [`./vaep.md`](./vaep.md) for the tile-size trap.

## 10. CLI registration

```python
def _moss_image_example(args, default: str) -> Example:
  from ..models.pipelines.moss_image import MOSSImagePipeline   # lazy: keeps cache_dit import cheap

  return Example(
    args=args,
    init_config=ExampleInitConfig(
      task_type=ExampleType.T2I,
      model_name_or_path=_path(default, args=args),
      pipeline_class=MOSSImagePipeline,
    ),
    input_data=ExampleInputData(
      prompt=_MOSS_IMAGE_PROMPT,
      height=1024, width=1024,
      num_inference_steps=50,
      guidance_scale=3.0,                     # keep > 1: cache guard in §6.2
      extra_input_kwargs={"uncond_skip_layers": [22]},
    ),
  )


@ExampleRegister.register("moss_image_sft", default="OpenMOSS-Team/MOSS-Image-SFT")
def moss_image_sft_example(args, **kwargs) -> Example:
  return _moss_image_example(args, "OpenMOSS-Team/MOSS-Image-SFT")
```

Plus the usual wiring: `_env_path_mapping["<MODEL>_DIR"]`, the name in `examples.py::__all__`, and
the factory in `_utils/__init__.py`. One factory per checkpoint variant that differs only by weights
(`moss_image_sft` / `moss_image_base`). Full CLI mechanics: [`./cli.md`](./cli.md).

For entry point 2 the example usually **stays unchanged**: the class name and constructor are kept
(§3.3), so only the import path moves from `diffusers` to `cache_dit.models...`.

## 11. Tests

Once vendored, the release's own CI no longer covers this copy, so pin the contracts with **CPU-only**
tests (no weights, no GPU) — they are the only fast guard against a silent regression:

| File | Locks |
|---|---|
| `tests/api/test_<model>_adapter.py` | adapter wiring (`layers` + `Pattern_3` + `has_separate_cfg`), `skip_layers` semantics under the cached wrapper, per-branch cache bookkeeping |
| `tests/api/test_<model>_pipeline.py` | `__call__` against a hand-written denoise loop built from the same components (forward count, fp32 combination, cache guards) |
| `tests/parallelism/test_<model>_cp_plan.py` | CP plan strings (`rope_split` / `layers.0` / `sequence_gather`), TP layer plan, fused-weight rearrangement |
| `tests/parallelism/test_<model>_cp_tp_tiny.py` | 2-rank tiny parity against a cp-free reference, plus a case with indivisible length for `--ulysses-anything` |
| `tests/utils/test_<model>_example.py` | CLI example registration, env mapping, default steps/guidance |

Prove the vendored copy equals the release **bit-exactly** before benchmarking: import the upstream
classes, load the same state dict into both, and compare the transformer output and the per-step fp32
latents with `torch.equal`. MOSS did this and reported bit-identical results; without it, a
"PSNR looks fine" run hides a wrong RoPE table or a dropped mask. For entry point 2, the same test
doubles as the §3.6 "did the copy change anything?" gate.

## 12. Silent traps

1. **`len(self.layers)` under cache is 1.** Any depth/index logic in `forward` must use
   `config.num_layers` and `block.index`.
2. **Skip/branch state in the loop body is lost.** Move it into the block (§6.1); a `PatchFunctor` is
   the fallback, not the default.
3. **Attention layout under CP.** `_dispatch_attention_fn` wants `[B, S, H, D]`; a wrong layout gives
   a coherent-looking-but-wrong image, not a crash.
4. **A locally copied VAE / text encoder disables VAE-P / TE-P** without any error (§9).
5. **`device_map` accepted by a vendored `from_pretrained`.** The CLI skips `pipe.to(device)` for
   `--device-map-balance`, so the run ends up entirely on CPU; fail fast instead.
6. **`quantization_config` forwarded into the release loader.** It cannot find the blocks; warn and
   quantize through cache-dit's own flags.
7. **Tokenizer next to the text encoder is an assumption, not a rule.** Pass the tokenizer directory
   explicitly.
8. **Generic transformer class names collide with existing adapter prefixes**; a dedicated, long
   prefix is cheaper than debugging a wrong stack.
9. **Two edits to the same planner file applied in parallel can overwrite each other** (MOSS lost its
   CP planner registration this way, surfacing as `No planner registered under name: ...`). Apply
   multi-site edits serially and re-read the registry after editing.
10. **`--compile` compiles one graph per skip-set value** when the block branches on `skip_layers`
    (cond vs uncond), which explains a long first-run warmup (MOSS: 162 s) and is not a bug.
11. **(Migration) A leftover `PatchFunctor` stops running** once the class moves into `cache_dit.*`:
    `is_from_diffusers()` returns False, the functor returns the module untouched, and the run
    continues with the unpatched forward — correct shapes, wrong output. Delete the functor in the
    same commit and grep for its registration line.
12. **(Migration) `check_forward_pattern` silently turns off** because the adapter default depends on
    `__module__`; pass it explicitly, otherwise a wrong block signature is never validated.
13. **(Migration) A renamed projection keeps loading with `strict=False`.** Always start with
    `load_state_dict(..., strict=True)`; a renamed key is a silent numerics change.

## 13. Verification checklist

```
- [ ] Entry point recorded: (1) model absent from diffusers — flavour A/B/C, or
      (2) absorbing a diffusers model into cache-dit (§3)
- [ ] Owned classes keep the §4 contracts (DiffusionPipeline, ModelMixin, diffusers VAE,
      transformers text encoder, top-level ModuleList, one block call per iteration,
      `hidden_states` first arg, `_repeated_blocks` declared)
- [ ] `from_pretrained` handles torch_dtype / quantization_config / device_map / trust_remote_code
      / pre-built components explicitly (§5)
- [ ] bit-exact vs the upstream source (transformer output and per-step latents)
- [ ] baseline CLI run renders, then `--cache` run: PSNR > 30 dB **and** SSIM > 0.90
- [ ] cache guard fires for the unsupported modes (`guidance_scale == 1`, SLG enabled)
- [ ] CP: `--parallel ulysses` on 2 GPUs, plus a non-divisible length with `--ulysses-anything`
      (or a documented Ring rejection)
- [ ] TP: 2 GPUs, fused FFN rearranged and heads divided
- [ ] TE-P / VAE-P: either verified or explicitly declared unsupported
- [ ] CPU tests for adapter/pipeline/plans/examples pass
- [ ] (Migration) diffusers-detection flips audited: pattern check set explicitly, functor deleted,
      class/attribute/state-dict names unchanged, upstream commit recorded, patch list emptied (§3)
```

## 14. MOSS-Image case study (numbers)

`MOSS-Image-SFT`, RTX PRO 5000 72GB, 512×512, 25 steps, `rdt 0.3` unless noted. Cache on the SFT
checkpoint: 4.97 s → 2.56 s (1.94×), 33.09 dB / SSIM 0.9263; Base: 32.16 dB / 0.9577.

| Config | Latency | PSNR / SSIM |
|---|---|---|
| cache (1 GPU) | 2.56 s (1.94×) | 33.09 / 0.9263 |
| CP Ulysses 2 GPU | — | inf / 1.0000 (pixel-identical to 1 GPU) |
| CP 2 GPU + cache | — | identical to 1-GPU cache |
| TP 2 GPU | — | 33.17 / 0.9546 |
| TP 2 GPU + cache | — | 31.72 / 0.9184 |
| TP + TE-P (2+2) | — | 32.07 / 0.9618 |
| hybrid `ulysses_tp` (2+2) | — | 34.65 / 0.9570 |
| VAE-P 2 GPU, 1536² | — | 46.76 / 0.9973 |

The parallel rows come from a separate run set (different step budget), so compare TP against TP, not
against the single-GPU row. Their ~32–33 dB floor is bf16 all-reduce noise, not an integration bug:
the tiny fp32 parity test (rel ≈ 5e-8) is what proves the sharding is correct.

Notes for the next custom model:

* 512² + 25 text tokens gives a joint length of **1049** — odd, so plain Ulysses cannot split it and
  every CP run needs `--ulysses-anything`. Check divisibility before promising Ring.
* VAE-P needs a latent larger than the VAE tile threshold (1536² here), otherwise the patch is a
  no-op that still reports success.
* `--compile` goes block-level ("Compiling repeated blocks"), warmup 162 s, 12 steps in 1.69 s.

## 15. Second-pass optimization of an existing diffusers pipeline (`_NAME`)

Worked example: `_QwenImage21` (`src/cache_dit/models/pipelines/_qwen_image21/`, PR #1274). Use this
chapter when the model already runs through cache-dit and the optimization is confined to seams.

### 15.1 Decide the shape first

| Question | If yes |
|---|---|
| Does the optimization change attention math, the block loop, tensor layout or checkpoint keys? | Absorb (§3). A subclass cannot express it without duplicating `__call__` |
| Is it confined to elementwise chains / projection epilogues around existing modules? | Subclass (§1.3) + fused ops (§15.6) |
| Is inductor already fusing the chain you want to replace? | Read §15.9 **before** writing a kernel: under `compile` a replacement for an already-fused pointwise chain usually loses |

### 15.2 Package layout

```
src/cache_dit/models/pipelines/_<model>/
  __init__.py                  # lazy exports, __all__ = [] (same rule as §2)
  transformer_<model>.py       # _<Model>Transformer2DModel + apply_fused_ops()
  pipeline_<model>.py          # _<Model>Pipeline.from_pretrained -> upgrade in place
  block.py                     # fused block / FFN forwards + install_fused_modules()
  fused_ops.py                 # apply_* helpers: preconditions + literal eager fallback + switches
  attention_processor.py       # fusion installed on the QKV seam (name it after what it patches)
  functor.py                   # cache-dit PatchFunctor subclass used when enable_cache runs
csrc/kernels/models/<model>_*.{h,cpp,cu}          # kernel family, linked into the single _C_ops extension
src/cache_dit/kernels/_models/_<model>.py         # six-section bridge (see cache-dit-custom-kernel)
```

### 15.3 The `from_pretrained` upgrade

`diffusers`' `pipeline_loading_utils` returns `cls` itself when `cls.__name__ != "DiffusionPipeline"`,
so a plain `super().from_pretrained(...)` already yields the subclass. Then swap the transformer class
and install the fusions:

```python
class _QwenImage21Pipeline(QwenImage21Pipeline):
  @classmethod
  def from_pretrained(cls, *args, **kwargs):
    pipe = super().from_pretrained(*args, **kwargs)
    if isinstance(pipe, cls) and not isinstance(pipe.transformer, _QwenImage21Transformer2DModel):
      pipe.transformer.__class__ = _QwenImage21Transformer2DModel   # in place, no weight copy
      pipe.transformer.apply_fused_ops()
    return pipe
```

* **Why a class swap is safe here:** the subclass adds no parameters, no submodules and no `__init__`
  arguments, so module names and state-dict keys are identical and nothing has to be transferred.
* Do not override `register_modules` / `_optional_components` / `__init__` — there is nothing new.
* Install **before** `enable_cache` / `maybe_compile_transformer` / offload, and keep `_old_forward`
  in sync (accelerate hooks read it): every rewrite of `forward` must be followed by
  `module._old_forward = module.forward`.

### 15.4 Registry checklist for the `_` prefix

Prefix registries match with `cls_name.startswith(name)` and take the **first** hit, so an unregistered
`_NAME` silently falls through to the un-optimized name (or to nothing).

| Site | Entry to add | Symptom when missing |
|---|---|---|
| `caching/block_adapters/adapters.py` | `@BlockAdapterRegister.register("_<Model>")` **above** the plain name | `enable_cache` finds no adapter → cache silently off |
| same file | pick the fused `PatchFunctor` when `pipe.__class__.__name__.startswith("_")` | cache path installs the stock forward and drops the fusion |
| CP planner | `ContextParallelismPlannerRegister.register("_<Model>")`, plus a `patch_functor()` hook so CP installs the *fused* functor | CP splits/gathers never installed, or installed over the fused forward |
| TP planner | `TensorParallelismPlannerRegister.register("_<Model>Transformer2DModel")` | TP falls back to a generic plan |
| `_utils/utils.py` `_class_maybe_force_compile_dynamic` | `"_<Model>"` | `dynamic` differs from stock, so the TP+compile arm is not comparable |
| `_utils/examples.py` | `<model>_fused` (+ turbo variant) examples | no CLI path to the optimized pipeline |
| `kernels/_models/__init__.py`, `kernels/__init__.py` | new `__all__` entries | `test_public_api_matches_all` fails |
| `setup.py` `_required_models_sources()` | every new `.cu` / `.cpp` | the file is **never compiled** (the list is explicit, not a glob) |

### 15.5 Choosing seams (the part that decides CP)

| Seam | Rebind | Notes |
|---|---|---|
| block forward | `block.forward = fused_block_forward.__get__(block)` | must keep the exact parameter names/count of the pattern (`check_forward_pattern`); a closure per block creates a code object per block and blows up compile |
| FFN forward | `block.mlp.forward = fused_ffn_forward.__get__(block.mlp)` | cheap, self-contained |
| transformer forward | leave it to the `PatchFunctor` | `enable_cache` replaces it; own it only if you need the block loop |
| attention q/k norm + RoPE | **module-level function**, e.g. `_<model>_prepare_qkv` | see below |

**Prefer patching the module-level helper over subclassing the processor.** cache-dit's CP support
replaces `AttnProcessor.__call__` wholesale (Ulysses sharding + block-causal segments). A processor
*subclass* either bypasses that `__call__` (CP silently loses its sharding) or delegates to it (no
fusion at all). In `_QwenImage21` three call sites share one helper — the stock processor, the flex
processor and cache-dit's `cp_call` — so installing the fused version on that symbol (in both module
namespaces, since cache-dit imported it by value) keeps fusion in every mode. Verified: CP2 output is
**bit-identical** to single-GPU, and fused-vs-stock deviation is identical with and without CP.

Keep a rollback (`install_qkv_patch()` / `uninstall_qkv_patch()` / `qkv_patch_is_installed()`): tests
and the operator switch both need it, and it makes the "was it installed?" question answerable.

### 15.6 Fused-op integration

Standard shape (details in `cache-dit-custom-kernel`):

* kernels live in the model family (`csrc/kernels/models/`) and are linked into the single
  `_C_ops` extension: `models.cpp` is the family registrar, each model adds a
  `register_<model>_ops(py::module_&)` call to it;
* Python side is the bridge: `from .._cuda_ext import _get_required_ops_module` → `_call_*`
  → `torch.library` `define` / `impl("CUDA")` / `register_fake` → public wrapper;
* one switch for the family (`CACHE_DIT_<MODEL>_FUSED_OPS`, default on) and, if you measured a
  per-operator difference, one allow-list (`CACHE_DIT_<MODEL>_OPS=all|a,b`) — the default must be the
  measured winner, not "everything";
* every `apply_*` re-checks dtype / contiguity / alignment / shape and otherwise evaluates the
  **literal upstream expression**, so a precondition failure degrades to the same values rather than an
  error. Put the reference transcription in the same module (`_eager_*`) so the two can be diffed and
  tested against each other.

### 15.7 Numerical contract: what can stay exact

| Operator kind | Contract |
|---|---|
| pure `mul` / `add` / `select` (gated residual, `1+scale`, SwiGLU product) | **bit-exact** is achievable and cheap: `__fmul_rn`/`__fadd_rn`, or `__hmul2`/`__hadd2` (a bf16 product is exact in fp32, so packed bf16 == fp32-then-round). Never `__hfma2` — it drops a rounding |
| fuses `rsqrtf` / `tanhf` / `expf` / `/` | exact **only if** the extension is built without `--use_fast_math`. cache-dit's `setup.py` enables it globally, so these operators are accepted at rel-L2 ≤ 1e-2 plus an end-to-end PSNR gate; `setup.py` documents that dropping the flag restores exactness |
| reduction (LayerNorm/RMSNorm mean) | reproduce ATen's leaf tree if you can; if the device's layout is not reproducible, pick **one** implementation and validate it (measured `_QwenImage21` value: rel-L2 7.9e-06, one bf16 ulp). Do **not** ship a runtime probe that silently picks between implementations |

Corollaries learned the hard way:

* Welford merges are neither commutative nor associative: a butterfly leaves every lane with a
  different mean. Use a `shfl_down` tree and broadcast lane 0; merge with `coef = 1/count`, and
  precompute the reciprocals per block with `__frcp_rn` (bit-identical to `1.0f/x` under
  `-prec-div=true`) instead of keeping a division in the serial chain.
* A `--use_fast_math` build makes `expf`/`tanhf`/`rsqrtf` approximate, so tests must compare at a
  tolerance and the PR must say so. `torch.equal` assertions become `rel-L2`, and only the pure
  mul/add operators keep exact checks.

### 15.8 Fold constants before writing any kernel

`_QwenImage21`'s block modulation recomputed `1 + scale` and `tanh(gate)` **per token** although they
only depend on the sample (and the target/t0 split). The kernel not only burned 4096× the arithmetic,
it also forced a `(B, S, D)` gate tensor to exist, which the gated residual then read back.

Folding it into `(R, D)` rows — and letting the residual select a row by mask — removed one full
`(B, S, D)` write plus one full `(B, S, D)` read per block, and dropped the per-token `tanhf`. That is
a traffic reduction, so it is the only change in the whole PR that moves the total byte count.

Rule: before optimizing a kernel, ask **which inputs are constant along the token axis**. Layer
modulation, gate rows, positional tables and masks all are.

### 15.9 compile interaction

Two things must be checked before and after wiring kernels:

1. **Is a per-block constant inside the compiled region?** `_QwenImage21`'s functor resolved its layer
   cache with `kv_caches.get_layer(self._idx)`. `_idx` is a per-block int, so dynamo specialised one
   graph per block; the 9th tripped `recompile_limit` (8) and blocks 9-32 silently fell back to eager:
   420.6 → 372.8 ms/step (+11% recovered). Wrap the resolution in `@torch.compiler.disable` and verify
   with `torch._dynamo.utils.counters` (`unique_graphs`, `recompile limit exceeded`).
2. **Does inductor already fuse the chain?** Measured on `_QwenImage21` (1024², compile + DBCache,
   ms/step): off 372.9, `qk_norm_rope` only 366.5, `ln_modulate` 377.2, `gated_residual`+`silu_mul`
   378.4, all 369.2. Three of four replacements **lose** under compile because inductor already emitted
   one triton kernel for each pointwise chain, and an opaque op moves the fusion boundary. The one that
   wins replaces something inductor refuses to codegen (`complex64`).
   → Ship the measured winner as the default allow-list; keep the rest available but off.

### 15.10 Validation protocol

| Level | Check |
|---|---|
| op | kernel vs the eager transcription in the same module: `torch.equal` for mul/add operators, rel-L2 ≤ 1e-2 for fast-math ones; include mask-full / mask-partial / `mask=None`; assert fallback for CPU, non-contiguous and unsupported head_dim |
| model | fused vs stock with identical inputs (same seed, same latent); cover `use_kv_cache` on/off and prefill/decode |
| e2e | `{stock, fused} × {eager, compile}` **without** DBCache to isolate the operator effect, then the same with cache for the product number; report absolute ms/step, not just deltas |
| accuracy | PSNR/SSIM between arms (peak 255 on 8-bit PNG — do not feed float [0,1] into `calculate_psnr`), and compare the operator delta against compile's own drift as the yardstick |
| CP | CP2 output must be bit-identical to single-GPU (PSNR `inf`) with fusion enabled; if it is not, the seam bypassed the sharding |
| guards | a test that the model source list covers every kernel file (`tests/kernels/test_models_source_manifest.py`), and one that the QKV seam installs/restores |

### 15.11 Worked numbers (`_QwenImage21`, 1024², 40 steps, no DBCache)

| arm | ms/step | vs stock |
|---|---|---|
| stock, eager (1 GPU) | 403.0 | – |
| fused, eager (1 GPU) | **355.3** | **-11.8%** |
| stock + compile (1 GPU) | 361.3 | – |
| fused + compile (1 GPU) | **352.5** | **-2.4%** |
| CP2, eager | 256.8 | – |
| CP2 + fused, eager | **238.0** | **-7.3%** |
| CP2 + compile | 223.8 | – |
| CP2 + fused + compile | 222.5 | -0.6% (noise) |

Accuracy: fused+compile vs stock+compile 43.74 dB / 0.9920, against compile's own drift of 44.87 dB /
0.9935 — no accumulation. CP2 stock is bit-identical to single-GPU.

### 15.12 Traps

| Trap | Symptom | Fix |
|---|---|---|
| `PatchFunctor.is_from_diffusers()` not overridden | functor no-ops **silently**, KV cache collapses to one layer, output is noise | override to `isinstance(t, <DiffusersTransformer>)`, assert `_is_patched` and the bound function identity |
| per-block constant inside the graph | recompile limit hit, half the blocks run eager, +11% | §15.9, point 1 |
| processor subclass instead of the module-level seam | CP loses sharding, or CP runs un-fused | §15.5 |
| kernels installed after `enable_cache` / `compile` | cache semantics lost, or the compiled graph keeps the old forward | install in `from_pretrained`, before any of them |
| `storage_offset() == 0` as a precondition | the decode step passes `joint[:, prefix_len:]`; a compiled graph cannot inspect the offset, so it either falls back or raises | require **pointer alignment** in C++, keep `is_contiguous()` in Python |
| per-token recomputation of per-sample constants | extra arithmetic **and** an extra `(B, S, D)` materialisation | §15.8 |
| expecting a win from every fused operator under compile | several operators measure slower and still ship "because they are fused" | §15.9, point 2; default to the measured winner |
| runtime probe to pick an implementation | host sync per step, two code paths to validate, flaky tests | pick one implementation, prove it with rel-L2 |
