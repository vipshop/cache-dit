# Cache Integration Reference

When to read this: read this file when implementing Cache support, selecting a `BlockAdapter`, handling third-party models, or deciding whether a `PatchFunctor` is required. Return to `../SKILL.md` for the high-level workflow.

## 1. Cache Integration: BlockAdapter + ForwardPattern

### 1.1 Concept

cache-dit's caching engine works by intercepting the forward pass of DiT transformer blocks. To do this, it needs to know:

1. **Where the blocks are** — which `ModuleList` attribute holds the repeated transformer blocks.
2. **What goes in and out** — the block's `forward()` input/output signature ("forward pattern").
3. **Any model quirks** — separate CFG passes, special patching needs, etc.

All of this is described by a single `BlockAdapter` dataclass instance.

### 1.2 ForwardPattern — The 6 Block I/O Contracts

`ForwardPattern` is an enum in `src/cache_dit/caching/forward_pattern.py`. It captures the hidden-state ordering and forward-signature shape of a family of transformer blocks. Choose the pattern that matches your block's `forward()` signature:

| Pattern             | `forward()` inputs                       | `forward()` returns                      | Return_H_First | Return_H_Only | Forward_H_only | Typical Models                                                         |
| ------------------- | ------------------------------------------ | ------------------------------------------ | -------------- | ------------- | -------------- | ---------------------------------------------------------------------- |
| **Pattern_0** | `(hidden_states, encoder_hidden_states)` | `(hidden_states, encoder_hidden_states)` | `True`       | `False`     | `False`      | Mochi, CogVideoX, CogView4, HunyuanVideo, EasyAnimate                  |
| **Pattern_1** | `(hidden_states, encoder_hidden_states)` | `(encoder_hidden_states, hidden_states)` | `False`      | `False`     | `False`      | Flux transformer_blocks, QwenImage, SD3, VisualCloze                   |
| **Pattern_2** | `(hidden_states, encoder_hidden_states)` | `(hidden_states,)`                       | `False`      | `True`      | `False`      | Wan, Allegro, Cosmos, LTX-1                                            |
| **Pattern_3** | `(hidden_states,)`                       | `(hidden_states,)`                       | `False`      | `True`      | `True`       | Flux single_transformer_blocks, DiT, PixArt, Sana, Lumina2, SkyReelsV2 |
| **Pattern_4** | `(hidden_states,)`                       | `(hidden_states, encoder_hidden_states)` | `True`       | `False`     | `True`       | (rare)                                                                 |
| **Pattern_5** | `(hidden_states,)`                       | `(encoder_hidden_states, hidden_states)` | `False`      | `False`     | `True`       | (rare)                                                                 |

**How to determine the correct pattern for your model:**

1. Open the block's `forward()` method in diffusers source.
2. Check the parameter list: does it take only `hidden_states`, or also `encoder_hidden_states`? This determines `Forward_H_only`.
3. Check the return statement: does it return one tensor or two? In what order? This determines `Return_H_Only` / `Return_H_First`.
4. Match against the table above. If none fits exactly, open an issue.

### 1.3 BlockAdapter Parameters

Defined in `src/cache_dit/caching/block_adapters/block_adapters.py`. Key parameters:

| Parameter                 | Type                                               | Description                                                                                                                                                                 |
| ------------------------- | -------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipe`                  | `DiffusionPipeline` or `FakeDiffusionPipeline` | The pipeline instance (or a placeholder if no pipeline is available).                                                                                                       |
| `transformer`           | `nn.Module` or `List[nn.Module]`               | The transformer module(s). Single module for most models; list of 2 for dual-transformer models (e.g., Wan 2.2 MoE).                                                        |
| `blocks`                | `nn.ModuleList` or `List[nn.ModuleList]`       | The block collection(s). Single ModuleList for most models; list of 2 for models with dual block types (e.g., Flux:`transformer_blocks` + `single_transformer_blocks`). |
| `forward_pattern`       | `ForwardPattern` or `List[ForwardPattern]`     | Must match`blocks` count. Single pattern for single block list; list of patterns for multiple block lists.                                                                |
| `check_forward_pattern` | `Optional[bool]`                                 | Validate that each block's I/O matches the declared pattern. If left `None` (default), cache-dit **auto-detects**: `True` for `diffusers` transformers, `False` for third-party ones (`maybe_skip_checks()`); it is also forced `False` when the transformer already has an `_hf_hook` / `_diffusers_hook`. Set explicitly for new models.                                                             |
| `check_num_outputs`     | `bool`                                           | If`True`, cache-dit additionally validates that each block returns the exact number of outputs the pattern declares. Needed for models whose blocks can return a variable tuple (e.g., HiDream, HunyuanVideo 1.0). Default `False`.                                             |
| `has_separate_cfg`      | `bool`                                           | Set`True` if the model performs separate conditional/unconditional forward passes for Classifier-Free Guidance.                                                           |
| `patch_functor`         | `PatchFunctor` or `None`                       | Optional pre-patch logic. Used when the model needs structural modification before caching hooks are installed (e.g., Flux dummy block merging, DiT re-patching).           |
| `blocks_name`           | `str` or `List[str]`                           | Override block attribute names (advanced).                                                                                                                                  |
| `dummy_blocks_names`    | `List[str]`                                      | Names of blocks that should be treated as dummy/merged (advanced, e.g., Flux single_transformer_blocks when merged into transformer_blocks).                                |

### 1.4 Implementation Templates

#### Template A: Single block list (most common)

```python
# In src/cache_dit/caching/block_adapters/adapters.py

@BlockAdapterRegister.register("MyModel")
def mymodel_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import MyModelTransformer2DModel

    _relaxed_assert(pipe.transformer, MyModelTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,    # adjust to your model
        check_forward_pattern=True,
        **kwargs,
    )
```

#### Template B: Dual block lists (like Flux)

```python
# Standard Flux: both block types use Pattern_1.
# For Flux2 / Nunchaku variants: single_transformer_blocks use Pattern_3 instead.

@BlockAdapterRegister.register("Flux")
def flux_adapter(pipe, **kwargs) -> BlockAdapter:
    from diffusers import FluxTransformer2DModel

    _relaxed_assert(pipe.transformer, FluxTransformer2DModel)
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_1,
        ],
        check_forward_pattern=True,
        **kwargs,
    )
```

#### Template C: Dual transformers (like Wan 2.2 MoE)

```python
@BlockAdapterRegister.register("Wan")
def wan_adapter(pipe, **kwargs) -> BlockAdapter:
    return BlockAdapter(
        pipe=pipe,
        transformer=[
            pipe.transformer,
            pipe.transformer_2,          # second transformer (MoE)
        ],
        blocks=[
            pipe.transformer.blocks,
            pipe.transformer_2.blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_2,
            ForwardPattern.Pattern_2,
        ],
        check_forward_pattern=True,
        has_separate_cfg=True,
        **kwargs,
    )
```

### 1.5 Registration

After implementing the adapter function, register it in `src/cache_dit/caching/block_adapters/__init__.py`:

```python
# Add this line:
mymodel_adapter = _safe_import(".adapters", "mymodel_adapter")
```

The `BlockAdapterRegister.register("MyModel")` decorator on your function already links the model name. The `__init__.py` import ensures the adapter is discovered at runtime.

### 1.6 Third-Party (Non-Diffusers) Models

If your model does **not** come from the official `diffusers` library (e.g., it is defined in `sglang` or another third-party package), follow these rules:

**Do NOT hardcode `from diffusers import ...`.** Instead, use `_safe_import` with name-based matching, or simply skip the diffusers-specific import entirely.

**`_relaxed_assert` is NOT mandatory.** The function (`src/cache_dit/caching/block_adapters/adapters.py`) checks `transformer.__module__` — if it does not start with `"diffusers"`, the function logs a warning and skips the strict type check automatically. For third-party models, you can:

- Omit `_relaxed_assert` entirely, or
- Call it with `allow_classes=None` to rely on the automatic skip behavior.

**Example — third-party BlockAdapter without `_relaxed_assert`:**

```python
@BlockAdapterRegister.register("MyThirdPartyModel")
def mythirdparty_adapter(pipe, **kwargs) -> BlockAdapter:
    # No `from diffusers import ...` — the transformer type is resolved at runtime.
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        check_forward_pattern=True,
        **kwargs,
    )
```

**The same principle applies to distributed modules** (CP, TP, TE-P, VAE-P planners): do not hardcode diffusers class names in `@...PlannerRegister.register(...)` if the model is from a third-party library. Instead, register under a descriptive name and ensure the dispatch logic matches on that name.

**Dependency management for third-party models:** The goal is to run the model through cache-dit inside the existing `cdit` conda environment. Follow these rules strictly:

- **Do NOT change** the versions of `torch`, `torchvision`, `transformers`, `diffusers`, `cache-dit`, or `triton` — these are cache-dit's core stack and altering them may break cache-dit itself.
- **Other dependencies** (e.g., `einops`, `opencv-python`, model-specific packages) may be installed ONLY if they do not conflict with the core stack above. Check `pip check` after installing.
- **Prefer importing optional dependencies lazily** (inside adapter/planner functions) rather than at module level, so the model integration does not force all users to install extra packages.
- If the third-party model requires a newer version of a core dependency, work with the cache-dit maintainers to upgrade it centrally rather than changing it unilaterally.

### 1.7 PatchFunctor — When the Default BlockAdapter Is Not Enough

> ⚠️ **Always check for these pitfalls before declaring the cache integration "done."** A BlockAdapter that looks correct on paper can silently produce wrong results if the `transformer.forward()` has any of the structural issues below. When in doubt, run a full inference with caching enabled and compare PSNR/SSIM against the uncached baseline.

The `BlockAdapter` works by intercepting the block-loop inside `transformer.forward()`. It replaces the original `ModuleList` (e.g., `self.transformer_blocks`) with `UnifiedBlocks` — a wrapper that injects cache look-up/save logic around each block call. This interception is mechanical: it relies on `inspect.signature` to bind arguments and on the assumption that the for-loop body contains **nothing but a single block call**. When the model's `forward()` violates these assumptions, the cache produces wrong results silently (no crash, just corrupted output).

A **`PatchFunctor`** is a monkey-patch that rewrites `transformer.forward()` *before* the `BlockAdapter` is applied. It exists solely to fix structural incompatibilities that would otherwise break the caching interceptor.

Two categories of pitfalls require a `PatchFunctor`:

#### Pitfall A: Block call argument mismatch (keyword vs positional)

**Problem**: `transformer.forward()` calls blocks with **keyword arguments** (e.g., `block(hidden_states=x, encoder_hidden_states=e, temb=t)`), but the block's `forward()` signature defines those parameters as **positional**. When cache-dit's `UnifiedBlocks` wrapper intercepts the call, it uses `inspect.signature.bind()` to match arguments — keyword-to-positional mismatches cause `bind()` to fail or bind to the wrong parameters.

**Symptom**: `TypeError` from `inspect.signature.bind()`, or the cache silently feeds wrong tensors to the block.

**Solution**: Write a `PatchFunctor` that rewrites the call site to pass positional arguments as positional (matching the block's actual signature), keeping only truly keyword-only parameters as keyword args.

**Canonical example — `LTX2PatchFunctor`** (`src/cache_dit/caching/patch_functors/functor_ltx2.py`):

The original diffusers code for LTX-2.0 passes all block arguments as keywords:

```python
# Original (diffusers) — ALL keyword args:
hidden_states, audio_hidden_states = block(
    hidden_states=hidden_states,
    audio_hidden_states=audio_hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    audio_encoder_hidden_states=audio_encoder_hidden_states,
    temb=temb,
    temb_audio=temb_audio,
    ...
)
```

The patched version converts the first four positional parameters to positional form, keeping the rest as keyword:

```python
# Patched — positional args match the block's forward(hidden_states, audio_hidden_states, ...):
hidden_states, audio_hidden_states = block(
    hidden_states,
    audio_hidden_states,
    encoder_hidden_states,
    audio_encoder_hidden_states,
    temb=temb,
    temb_audio=temb_audio,
    ...
)
```

**How to detect this pitfall**: Read the block's `forward()` signature in the diffusers source. Count how many parameters are positional (before any `*` or `*args`). Then check how `transformer.forward()` invokes the block — if it passes any of those positional params as keyword args, you need a PatchFunctor.

#### Pitfall B: For-loop body has extra operations

**Problem**: The `for block in self.blocks:` loop in `transformer.forward()` contains operations *other than* the block call itself — such as `temb` reassignment, conditional checks, or tensor reshaping. After `CacheAdapter.apply()` replaces `self.blocks` with `UnifiedBlocks`, the caching wrapper **takes over the iteration** and only executes the block call; all extra operations inside the original loop body are **silently skipped**.

**Symptom**: Cache-enabled output is corrupted (low PSNR/SSIM, visual artifacts) because modulation parameters or intermediate tensors are stale or missing.

**Solution**: Write a `PatchFunctor` that moves the extra operations **outside** (before or after) the for-loop, so the loop body contains only the block call.

**Canonical example — `ErnieImagePatchFunctor`** (`src/cache_dit/caching/patch_functors/functor_ernie_image.py`):

The original diffusers code reconstructs `temb` inside the loop body:

```python
# Original (diffusers) — temb reassigned INSIDE the for-loop:
for layer in self.layers:
    temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
    x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)
```

After `CacheAdapter.apply()` replaces `self.layers` with `UnifiedBlocks`, the `temb = [...]` line is never executed — each block receives a stale or undefined `temb`. The patched version moves `temb` construction **before** the loop:

```python
# Patched — temb constructed ONCE before the loop:
temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
for layer in self.layers:
    x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)
```

**How to detect this pitfall**: Inspect the for-loop body in `transformer.forward()`. If *any* line between `for ... in self.XXX:` and the actual block call does something other than a trivial `if torch.is_grad_enabled()` guard, you likely need a PatchFunctor.

#### Beyond the two canonical pitfalls

Pitfalls A and B are the two simplest cases (fix at the call site, or hoist one line out of the loop). Real models often need heavier PatchFunctors. Browse `src/cache_dit/caching/patch_functors/` (13+ functors) before writing your own — recurring patterns include:

- **Per-block `forward()` replacement + block-id injection** — when the loop body has *per-block* extra operations that cannot simply be hoisted (they depend on the block index). The functor patches `transformer.forward()` **and** each block's `forward()`, and injects a `_block_id` / `_layer_id` onto every block so the patched block can look up per-block data (skip-connection lists, per-block encoder states, control hints). Examples: `HiDreamPatchFunctor`, `HunyuanDiTPatchFunctor`, `WanVACEPatchFunctor`, `ChromaPatchFunctor`, `GlmImagePatchFunctor`, `BriaFiboPatchFunctor`.
- **Block signature modification** — rewriting a block's `forward()` signature so the caching wrapper can bind it (e.g. `FluxPatchFunctor` adds an `encoder_hidden_states` parameter to `FluxSingleTransformerBlock` in older diffusers).
- **Block-list merge / dummy blocks** — structurally merging two `ModuleList`s into one for unified caching (e.g. `FluxPatchFunctor` merging `transformer_blocks` + `single_transformer_blocks` when `dummy_blocks_names` is set).

The rules below (identical signature, identical output when cache is disabled, minimal changes) apply to all of these.

#### Implementing a PatchFunctor (template)

```python
# In src/cache_dit/caching/patch_functors/functor_my_model.py

from .functor_base import PatchFunctor

class MyModelPatchFunctor(PatchFunctor):

    def _apply(self, transformer, **kwargs):
        # Always check the transformer type if importing from diffusers:
        # from diffusers.models.transformers.transformer_xxx import MyTransformer
        # assert isinstance(transformer, MyTransformer)

        # Replace transformer.forward with the patched version:
        transformer.forward = _patched_forward.__get__(transformer)
        transformer._is_patched = True
        return transformer


def _patched_forward(self, hidden_states, ...):
    """Patched forward — same logic as original, with structural fixes applied."""
    ...  # Copy the original forward body and apply fixes at the affected sites.
```

Then reference it in your `BlockAdapter`:

```python
@BlockAdapterRegister.register("MyModel")
def mymodel_adapter(pipe, **kwargs) -> BlockAdapter:
    return BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_0,
        check_forward_pattern=True,
        patch_functor=MyModelPatchFunctor(),    # ← wire up the patch
        **kwargs,
    )
```

**Key rules for PatchFunctor:**

- The patched `forward()` **must** keep the exact same signature (parameter names, types, defaults) as the original — the pipeline calls it with the same arguments.
- The patched `forward()` **must** produce identical output to the original when caching is disabled — verify with PSNR/SSIM before enabling the cache.
- Only fix the structural issue; do not refactor, optimize, or "improve" unrelated code in the patched forward.
- Import the transformer class **lazily** (inside `_apply()`) if the model's diffusers version may not be installed everywhere.


## More references 

We recommend reading the following files for additional context:

- cache related source code: `src/cache_dit/caching/`
