# Context Parallelism Reference

When to read this: read this file when implementing CP planners, choosing hook-based vs hybrid CP, handling UAA, or fixing `attention_mask` sequence reordering. Return to `../SKILL.md` for the decision chart.

## 2. Context Parallelism (CP)

### 2.1 Concept

Context Parallelism splits the **sequence dimension** of hidden states across multiple GPUs. Each GPU computes attention over a local chunk, then gathers results. cache-dit supports two CP strategies:

- **Ulysses** (`--parallel ulysses`): All-to-all communication. Better for shorter sequences or when combined with TP. **Always prefer Ulysses over Ring** — it is more mature, better tested, and supports `--ulysses-anything` (UAA) for non-divisible head counts and sequence lengths.
- **Ring** (`--parallel ring`): Peer-to-peer communication in a ring topology. Better for very long sequences. Only consider Ring if Ulysses is proven inadequate for your use case.

### 2.2 Two Implementation Approaches

> **⚠️ MANDATORY: Hook-based first, hybrid second, pure patch-based NEVER.**
>
> The implementation priority is:
> 1. **Pure hook-based** (§2.3): Every sequence tensor is a function parameter of an `nn.Module` → use declarative hooks only.
> 2. **Hybrid: hook-based + minimal method patch** (§2.4): ONE or TWO tensors are produced by a plain method (e.g. `get_rotary_pos_embed`) or a nested tuple → write a ~10-line patch for that method, keep everything else hook-based.
> 3. **Pure patch-based: REJECTED.** Patching the entire `transformer.forward()` copies ~100+ lines of diffusers code, breaks on every upstream change, and bypasses all framework validation. **If hybrid cannot work, re-analyze the model structure — pure patch-based is NOT an acceptable answer.**

cache-dit offers **two** CP implementation patterns:

| Priority | Approach | Mechanism | When to use |
|---|---|---|---|
| **1st** | **Hook-based** (§2.3) | Declarative `_ContextParallelInput` / `_ContextParallelOutput` dict; framework inserts split/gather hooks at specified module-call boundaries. | All sequence tensors are **plain tensors or 1-D flat lists** passed as parameters to `nn.Module.forward()`. |
| **2nd** | **Hybrid: hook plan + minimal patch** (§2.4) | Hook-based CP plan handles most tensors; a **single method patch** (~10 lines) fixes the one or two tensors hooks cannot reach (e.g. a RoPE method, a nested tuple return). Return the hook plan (not `{}`) from `_apply()`. | One or two sequence-dependent tensors are produced by a **plain method** (not a sub-module), have a **nested tuple type** hooks cannot iterate, or are **local variables** that a tiny wrapper can expose. |

> **❌ Pure patch-based (patching the entire `forward()`) is explicitly REJECTED.** The two shipped "patch-based" planners — ErnieImage (legacy, pre-dates hook list support) and BooguImage (genuinely complex double→single stream fusion + per-stream internal concat) — should NOT be cited as justification. Any new model must use hook-based or hybrid. If a model appears to need a full forward patch, escalate for architecture review.

### 2.3 Hook-Based CP

The hook mechanism inserts split/gather operations at **function-call boundaries** of the transformer module tree. Hooks are specified as a dict `{module_path: {param_name: _ContextParallelInput(...)}, ...}`. The framework uses `inspect.signature` on each intercepted function to match parameter names; non-tensor parameters (str, int, list, dict) are silently ignored.

There are two **granularity levels** within hook-based CP:

#### 2.3.1 Transformer-Level CP Plan (Recommended Default)

The hook key `""` (empty string) means **"apply to `transformer.forward()`"** — the hooks intercept the top-level forward's arguments and return values. All sequence-dependent tensors are split **before** the block loop and gathered **after** it.

**When to use**: The model's `forward()` is a simple "preprocess → block loop → postprocess" pipeline where all block inputs are plain tensors. No inter-block operations need the full sequence.

**Examples**: `FluxContextParallelismPlanner` (`flux.py`), `OvisImageContextParallelismPlanner` (`ovis_image.py`), `ChromaContextParallelismPlanner` (`chroma.py`), `LongCatImageContextParallelismPlanner` (`longcat_image.py`), and the vanilla `Flux2ContextParallelismPlanner` (`flux2.py`). (Note: many models do NOT split at the root `""` but at a single block boundary `transformer_blocks.0` and rely on the attention processor's all-to-all to keep later blocks consistent — see §2.3.5.)

**Template** (Flux / OvisImage pattern):

```python
from .register import (
    ContextParallelismPlanner,
    ContextParallelismPlannerRegister,
)
from ...distributed.core import _ContextParallelInput, _ContextParallelOutput

@ContextParallelismPlannerRegister.register("MyModelTransformer2DModel")
class MyModelContextParallelismPlanner(ContextParallelismPlanner):

    def _apply(self, transformer=None, parallelism_config=None, **kwargs):
        _cp_plan = {
            # "" = transformer.forward() — intercept top-level args
            "": {
                "hidden_states":
                    _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "encoder_hidden_states":
                    _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "img_ids":
                    _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
                "txt_ids":
                    _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            },
            # Gather the output of the final projection layer
            "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
```

The `""` key exposes every named parameter of `transformer.forward()` to hooks. Split happens once, before any block executes. Gather happens once, after all blocks finish. This is the **simplest, most efficient** pattern — use it unless your model's architecture prevents it.

**When transformer-level is NOT suitable** — the root-level plan can break in two distinct ways, and each has a lighter-weight fix than "convert everything to per-block split/gather":

1. **A per-block sub-module produces a sequence-dependent output that must match the LOCAL (split) sequence** — e.g. BriaFIBO's per-block `caption_projection`, QwenImage's `pos_embed`, or Wan/SkyReels/HunyuanImage's `rope`. You do NOT need to abandon the root plan: keep splitting the main tensors at the root, and add an **output-split hook** (`split_output=True`) on that one sub-module so its output is re-split to the local sequence. See §2.3.4.

2. **An inter-block operation genuinely needs the FULL (gathered) sequence** — e.g. history/current fusion, ControlNet injection, or multiple `ModuleList`s with different structures. Only then drop down to **sub-module-level hooks** (§2.3.2), which gather → run the op on the full sequence → re-split at each boundary (at the cost of extra communication).

> ⚠️ **Do NOT reflexively rewrite a root plan into a per-block split-and-gather of `transformer_blocks.*` + `single_transformer_blocks.*`.** That structure double-splits, is almost never correct, and **no shipped planner uses it** — including BriaFIBO, whose real CP plan is a root split plus a `caption_projection` output-split plus a mask permute (see §2.3.4 and §2.7), NOT a per-block plan.

#### 2.3.2 Sub-Module-Level CP Plan (Per-Block / Per-Layer)

Hooks are inserted on **specific sub-module paths** (e.g., `"blocks.0.attn1"`, `"blocks.*.ffn"`). Split/gather happens at each sub-module boundary — the sequence is split, a single sub-module runs on the local chunk, then the sequence is gathered again before the next sub-module runs.

**When to use**: The model has inter-block operations that **require the full sequence** between blocks. Common triggers:

1. **Cross-attention between different sequence groups** where two sub-sequences are split/merged between blocks (e.g., Helios's history-current state fusion).
2. **ControlNet block sample injection** where an external tensor is added to the hidden states between layers (e.g., ZImage's `unified + controlnet_block_samples[layer_idx]`).
3. **Multiple `ModuleList`s with different structures** (e.g., ZImage's `noise_refiner` → `context_refiner` → `layers`).
4. **Per-block outputs that diverge into separate paths** before being re-merged later.

**Examples**: `HeliosContextParallelismPlanner` (`helios.py`), `ZImageContextParallelismPlanner` (`zimage.py`).

**Template** (Helios pattern — per-attn/ffn hooks):

```python
@ContextParallelismPlannerRegister.register("HeliosTransformer3DModel")
class HeliosContextParallelismPlanner(ContextParallelismPlanner):

    def _apply(self, transformer=None, parallelism_config=None, **kwargs):
        num_blocks = len(transformer.blocks)
        _cp_plan = {
            # Split at each attn/ffn sub-module boundary.
            # Wildcard "blocks.*" matches all blocks.
            "blocks.*.attn1": {
                "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "blocks.*.attn2": {
                "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "blocks.*.ffn": {
                "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            # Gather after every sub-module (needed because of history-current fusion between them).
            **{f"blocks.{i}.attn1": _ContextParallelOutput(gather_dim=1, expected_dims=3)
               for i in range(num_blocks)},
            **{f"blocks.{i}.attn2": _ContextParallelOutput(gather_dim=1, expected_dims=3)
               for i in range(num_blocks)},
            **{f"blocks.{i}.ffn": _ContextParallelOutput(gather_dim=1, expected_dims=3)
               for i in range(num_blocks)},
        }
        return _cp_plan
```

**Template** (ZImage pattern — multi-ModuleList + ControlNet injection):

```python
@ContextParallelismPlannerRegister.register("ZImageTransformer2DModel")
class ZImageContextParallelismPlanner(ContextParallelismPlanner):

    def _apply(self, transformer=None, parallelism_config=None, **kwargs):
        n_noise_refiner = len(transformer.noise_refiner)
        n_context_refiner = len(transformer.context_refiner)
        _cp_plan = {
            # ModuleList 1: noise_refiner (split at first block, gather after last)
            "noise_refiner.0": {
                "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "noise_refiner.*": {
                "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"noise_refiner.{n_noise_refiner - 1}":
                _ContextParallelOutput(gather_dim=1, expected_dims=3),
            # ModuleList 2: context_refiner (same pattern)
            "context_refiner.0": {
                "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"context_refiner.{n_context_refiner - 1}":
                _ContextParallelOutput(gather_dim=1, expected_dims=3),
            # ModuleList 3: main layers (split/gather at each block due to ControlNet injection)
            "layers.*": {
                "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            **{f"layers.{i}": _ContextParallelOutput(gather_dim=1, expected_dims=3)
               for i in range(len(transformer.layers))},
        }
        return _cp_plan
```

**⚠️ Trade-off**: Sub-module-level plans introduce **extra all-gather + re-split overhead** at every hook boundary. For a 30-layer model with per-block hooks, this adds 30× more communication than a transformer-level plan. Only use this when the model's inter-block logic genuinely requires the full sequence. For most models, the transformer-level plan (§2.3.1) is sufficient and more performant.

#### 2.3.3 Hook Key Reference

| Key pattern | Intercepts | Example |
|---|---|---|
| `""` | `transformer.forward()` args & return | `"hidden_states": _ContextParallelInput(...)` inside `""` |
| `"blocks.0"` | `transformer.blocks[0].forward()` args & return | First block only |
| `"blocks.*.attn1"` | `attn1.forward()` of every block | Wildcard — applies to all matching sub-modules |
| `"blocks.{N-1}"` | `transformer.blocks[N-1].forward()` | Last block only (gather output) |
| `"proj_out"` | `transformer.proj_out()` return value | Output-only hook (no input split) |
| `"pos_embed"` | `transformer.pos_embed()` args & return | Pre-embedding split |

**Hook semantics** (`_ContextParallelInput` / `_ContextParallelOutput`):

- **`split_dim`**: Dimension to split. For `[B, S, D]` tensors, `split_dim=1` splits sequence. For `[B, S]` position IDs, `split_dim=0` splits batch.
- **`expected_dims`**: Number of tensor dimensions expected (2, 3, or 4). Used for validation.
- **`split_output`**: If `True`, also splits the return value after the function runs (rarely needed; default `False`).
- **`gather_dim`**: Dimension along which to all-gather the output (usually matches the corresponding `split_dim`).
- **Parameters not listed** in the hook dict are passed through unchanged (not split, not gathered).
- **Scalar non-tensor parameters** (str, int, float, bool) are silently ignored by hooks — they cannot be split.
- **1-D flat lists of tensors** (e.g., `text_encoder_layers: list[Tensor]`, `temb: list[Tensor]`) are supported — the hook iterates and splits each tensor element independently. Nested structures (list of lists, dict of tensors) are NOT supported.

#### 2.3.4 Output-Split Hook (`split_output=True`) — Re-Splitting a Sub-Module Output

This is the **correct, lightweight fix** for the most common "root plan is almost right, but one sub-module output has the wrong sequence length" situation. Instead of dropping to a per-block plan, you keep the root `""` split and add **one** hook on the offending sub-module with `split_output=True`.

**The problem it solves.** When you split `hidden_states` / `encoder_hidden_states` at the root, some models have a sub-module (usually a *per-block projection* or a *RoPE / position-embedding builder*) that either (a) is called on a **full-sequence** input and returns a full-sequence output that must be re-split to the local chunk, or (b) recomputes a sequence-length-dependent tensor internally. The main tensors are already local, but this one sub-module output is not — so the block sees mismatched sequence lengths and produces garbage.

**The mechanism.** A `_ContextParallelInput(..., split_output=True)` hook on a sub-module path does two things: it splits the named **input** before the sub-module runs *and* splits the sub-module's **return value** after it runs. Keyed on a `ModuleList` wildcard (e.g. `"caption_projection.*"`), it applies to every element of that list. The dict key is the **positional argument index** (an `int`) or the **parameter name** (a `str`).

**Canonical example — BriaFIBO** (`bria_fibo.py`, `BriaFiboContextParallelismPlanner`). BriaFIBO has one `caption_projection` per block, each projecting the text embedding for that block. The real plan is a **root split** plus a **`caption_projection.*` output-split** plus a **mask permute** (see §2.7) — NOT a per-block plan:

```python
@ContextParallelismPlannerRegister.register("BriaFiboTransformer2DModel")
class BriaFiboContextParallelismPlanner(ContextParallelismPlanner):
    def _apply(self, transformer=None, parallelism_config=None, **kwargs):
        _patch_bria_fibo_mask_permute(transformer)   # §2.7: permute the 2D mask under CP
        _cp_plan = {
            # Root split: main sequence tensors + position ids.
            "": {
                "hidden_states":
                    _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "encoder_hidden_states":
                    _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "img_ids":
                    _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
                "txt_ids":
                    _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            },
            # Output-split: each per-block caption_projection output is re-split
            # so it stays aligned with the LOCAL encoder_hidden_states.
            # Key 0 = the first positional arg of caption_projection[i].forward().
            "caption_projection.*": {
                0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            },
            # Single gather restores the full sequence for the output head.
            "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
```

**Other real users of output-split:**

| Model | Sub-module | Why `split_output=True` |
|---|---|---|
| **BriaFIBO** (`bria_fibo.py`) | `caption_projection.*` (positional arg `0`) | Per-block text projection must match the local sequence. |
| **QwenImage** (`qwen_image.py`, newer diffusers) | `pos_embed` (args `0`, `1`) | The position embedder returns full-sequence RoPE tables that must be re-split. |
| **Wan / ChronoEdit / SkyReels** (`wan.py`, `chrono_edit.py`, `skyreels.py`) | `rope` (args `0`, `1`) | RoPE freqs are recomputed on the full sequence; split the two returned tensors. |
| **HunyuanImage / HunyuanVideo** (`hunyuan.py`) | `rope` (args `0`, `1`) | Same RoPE re-split, combined with the mask reorder of §2.7. |
| **LTX2** (`ltx2.py`) | `rope`, `audio_rope`, `cross_attn_rope`, `cross_attn_audio_rope` | Four separate RoPE tables, each output-split. |

**Rule of thumb**: if a root plan gives wrong results and the culprit is a *single* projection / RoPE / pos-embed sub-module whose output length is full instead of local, reach for an output-split hook **before** considering §2.3.2 (sub-module-level) or §2.4 (hybrid).

#### 2.3.5 Single-Point Split at `transformer_blocks.0` (Common "Processor-Driven" Pattern)

Many models do NOT split at the root `""`. Instead they split the sequence **once**, on the input of the **first block** (`transformer_blocks.0`), and rely on the **attention processor's Ulysses all-to-all** to keep every subsequent block consistent (each block's attention internally all-to-all's the sequence back and forth). A matching output gather is placed on the last block or on the final projection.

**When to use**: The block loop is uniform (every block has the same signature) and the attention processor is patched to call `_dispatch_attention_fn` with the CP config, so the sequence "just flows" through the local chunks. This is the single **most common** CP shape in the codebase.

**Real users**: `CogVideoX`, `CogView3Plus`, `CogView4`, `ConsisID`, `DiT`, `PixArt`, `HunyuanImage`, `HunyuanVideo`, and newer `QwenImage`. Most of these also patch the attention processor and add a `rope` output-split (§2.3.4).

**Sketch**:

```python
_cp_plan = {
    # Split once, on the first block's input.
    "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    },
    # Often combined with a rope output-split (see §2.3.4):
    "rope": {0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
             1: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)},
    # Gather after the last block (or on the final projection).
    f"transformer_blocks.{n_blocks - 1}": _ContextParallelOutput(gather_dim=1, expected_dims=3),
}
```

> The gather key is model-specific — Flux/OvisImage gather on `proj_out`, DiT on `proj_out_2`, CogVideoX/ConsisID on the last block, ZImage on its final layer. Always check where the full sequence must be restored for the output head.

### 2.4 Hybrid CP (Hook-Based + Minimal Method Patch)

> **This is the 2nd-priority approach — use when pure hook-based hits ONE or TWO obstacles.**

When most sequence tensors are reachable by hooks but one or two specific tensors are not — because they are produced by a **plain method** (not an `nn.Module`), have a **nested tuple type** the hook framework cannot iterate, or are **local variables** that a tiny wrapper can expose — the correct answer is **hybrid**: a hook-based CP plan PLUS a minimal patch on the offending method. **Do NOT patch the entire `transformer.forward()`.**

#### 2.4.1 Recognizing a Hybrid-Suitable Model

The tell-tale sign of a hybrid-suitable model: you can write a complete hook-based CP plan (root split or single-point split) that covers ALL tensors EXCEPT one. That one exception falls into exactly one of these categories:

| Obstacle | Why hooks fail | Hybrid fix |
|---|---|---|
| **RoPE / position embedding method** (e.g. `get_rotary_pos_embed()`) | The tensor is returned by a plain `def`, not an `nn.Module.forward()`. Hooks only intercept module call boundaries. | **Patch the method** to return local-chunk tensors when CP is active. |
| **Nested tuple return type** (e.g. `((vis_cos, vis_sin), (txt_cos, txt_sin))`) | The hook framework supports flat tensors and 1-D lists, but NOT `tuple[tuple[Tensor, Tensor], ...]`. This prevents a hook from splitting the output. | **Same as above**: patch the method that produces the tuple so it returns already-split local chunks. |
| **Local variable not exposed as a module parameter** (e.g. attention mask built from position IDs) | The tensor is created inside `forward()` and never passes through an `nn.Module` boundary that hooks can intercept. | **Wrap the tensor creation in a tiny `nn.Module`** subclass and register it, so hooks can intercept. Or, if the mask is simple, **patch the parent method** to split/reorder it. |

> **Key principle**: in a hybrid plan, the `_apply()` method returns a **real hook-based CP plan** (not `{}`). The method patch runs BEFORE the hook framework, making the problematic tensor hook-compatible.

#### 2.4.2 Template: Patching a RoPE Method

The most common hybrid scenario: a model computes RoPE via a plain method `get_rotary_pos_embed(vis_rope_size, txt_rope_size)` that returns a nested tuple `((vis_cos, vis_sin), (txt_cos, txt_sin))`.

**Hook-based plan** (handles all tensors EXCEPT RoPE):

```python
@ContextParallelismPlannerRegister.register("MyModelTransformer")
class MyModelCPPlanner(ContextParallelismPlanner):

    def _apply(self, transformer=None, parallelism_config=None, **kwargs):
        _patch_rope_for_cp(transformer)          # §2.4.3: patch the RoPE method
        _patch_attn_processor_for_cp(transformer) # set _parallel_config

        n_blocks = len(transformer.double_blocks)
        _cp_plan = {
            # Single-point split on the first block.
            # image_rotary_emb is NOT listed — the rope patch handles it.
            "double_blocks.0": {
                "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
                "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
            },
            # Gather after the last block.
            f"double_blocks.{n_blocks - 1}": _ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan   # ← REAL hook plan, NOT {}
```

#### 2.4.3 Template: The RoPE Method Patch (~10 lines)

```python
def _patch_rope_for_cp(transformer):
    """Patch get_rotary_pos_embed to return local-chunk RoPE under CP.

    This is the ONLY patch needed — everything else is hook-based.
    """
    orig_rope = transformer.get_rotary_pos_embed

    def patched_rope(self, vis_rope_size, txt_rope_size=None):
        vis_freqs, txt_freqs = orig_rope(self, vis_rope_size, txt_rope_size)
        cp = getattr(self, "_cp_config", None)
        if cp is not None and cp.world_size > 1:
            rank, ws = cp.rank, cp.world_size
            # vis_freqs = (cos, sin), each [S_vis, head_dim]
            vis_freqs = (
                vis_freqs[0].chunk(ws, dim=0)[rank],
                vis_freqs[1].chunk(ws, dim=0)[rank],
            )
            if txt_freqs is not None:
                txt_freqs = (
                    txt_freqs[0].chunk(ws, dim=0)[rank],
                    txt_freqs[1].chunk(ws, dim=0)[rank],
                )
        return vis_freqs, txt_freqs

    transformer.get_rotary_pos_embed = patched_rope.__get__(transformer)
```

#### 2.4.4 Cross-Cutting Rules (Apply to BOTH Hook-Based and Hybrid)

These rules apply regardless of whether you use pure hook-based or hybrid CP:

1. **Split AFTER preprocessing, BEFORE the main block loop.** Preprocessing (embedding, patching, RoPE, modulation) runs on the full sequence on every GPU — it is cheap (no attention compute) and avoids complex split/gather logic for non-standard tensors.

2. **DO NOT split `attention_mask` — REORDER it.** Ulysses all-to-all internally recovers the full sequence, so each GPU keeps the complete mask. However, all-to-all *reorders* the sequence into rank-concatenated order. Any mask indexed by sequence position must be permuted. A 1D key-only mask `[B, 1, 1, S_full]` needs reordering along the key dim only. A 2D full mask `[B, 1, S, S]` must be permuted along both query and key dims. See §2.7.

3. **All-gather the output** before the final projection (`norm_out`, `proj_out`) so the output head sees the full sequence.

4. **Patch the attention processor** to call `_dispatch_attention_fn` with `cp_config`, ensuring Ulysses all-to-all is used inside each attention layer.

5. **GQA handling**: If the model has GQA with indivisible `num_kv_heads`, do the KV→Q repeat BEFORE calling `_dispatch_attention_fn` and pass `enable_gqa=False`.

#### 2.4.5 Legacy Note: Existing Patch-Based Planners

The codebase contains two planners that patch the entire `transformer.forward()`:

- **`ErnieImageContextParallelismPlanner`** (`ernie_image.py`) — historical, written before the hook framework supported 1-D lists of tensors.
- **`BooguImageContextParallelismPlanner`** (`boogu_image.py`) — genuinely complex structure: internal `flat_and_pad_to_seq`, double→single stream fusion, per-stream internal concatenation.

**These are NOT templates for new models.** New models MUST use hook-based (§2.3) or hybrid (§2.4). If you believe a new model genuinely needs a full forward patch like BooguImage, escalate for architecture review first.

### 2.5 Registration

Add to `src/cache_dit/distributed/transformers/planners.py` inside `_activate_cp_planners()`:

```python
MyModelContextParallelismPlanner = _safe_import(
    ".my_model", "MyModelContextParallelismPlanner",
    ImportErrorContextParallelismPlanner)
```

### 2.6 Ulysses Anything Attention (UAA)

**UAA** (Ulysses Anything Attention) extends Ulysses CP to handle two common scenarios that vanilla Ulysses cannot:

1. **Head count not divisible by `cp_size`** — e.g., `num_kv_heads=7` with `cp_size=2`.
2. **Sequence length not divisible by `cp_size`** — uneven sequence splits.

UAA is enabled via the `--ulysses-anything` CLI flag:

```bash
# task: task name for logging and output organization
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <model_name> \
    --parallel ulysses --ulysses-anything \
    --save-path .tmp/{task}/<model_name>_ulysses_uaa.png
```

**How it works:** UAA internally pads K/V heads and/or sequence lengths to the nearest divisible size before the all-to-all communication, then strips the padding after. The padding is transparent — no model code changes are needed. The all-to-all primitives (`_all_to_all_single_any_qkv_async` / `_all_to_all_single_any_o_async` in `src/cache_dit/distributed/core/_distributed_primitives.py`) handle the padding logic automatically.

**When to use UAA:**

- Model has GQA with `num_kv_heads` not divisible by `cp_size` (e.g., Boogu-Image with 7 KV heads).
- Sequence lengths vary across samples in a batch or are not divisible by `cp_size`.
- Any model where vanilla Ulysses crashes with divisibility errors.

### 2.7 ⚠️ Attention Mask vs CP Sequence Reordering (Critical Pitfall)

> **If the model uses an `attention_mask`, CP can silently corrupt the output even when nothing crashes.** This is the single most common "CP runs but the image is wrong" bug. ALWAYS check for a mask before assuming a model is CP-safe.

**Why it happens.** Ulysses all-to-all does NOT preserve the original sequence order. Each rank holds a local sequence `[text_local, image_local]`, and the all-to-all concatenates them **in rank order**, so the global sequence the attention kernel actually sees is:

```
[text_0, image_0, text_1, image_1, ...]   # rank-concatenated
```

not the original `[text_all, image_all]`. For a **mask-free** model this is harmless — attention is permutation-equivariant, so every token still gets the correct output regardless of order (RoPE is applied per-token *before* the all-to-all). But if the model applies an `attention_mask` that is indexed by absolute sequence position, the mask no longer lines up with the reordered sequence → wrong entries get masked → **localized corruption**.

**Symptoms (how to recognize it):**

- The image is *mostly correct* but a **small localized region is garbled** — very often the **top-left corner** (the first image patch, which sits right at the text/image concatenation boundary).
- Metrics plateau: PSNR stuck around ~28–30, SSIM ~0.80–0.85, and **no amount of RoPE / split-position tweaking helps**.
- The corruption is **consistent across seeds** and independent of whether you use hybrid or hook-based CP.

**Diagnosis (do this FIRST when metrics are stuck ~30):**

Dump the transformer's real first-step inputs on a single GPU and inspect for a hidden mask:

```python
saved = {}
def pre_hook(m, args, kwargs):
    saved.setdefault("kwargs", {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                for k, v in kwargs.items()})
pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
pipe(prompt=..., num_inference_steps=1)
# Inspect saved["kwargs"] — look inside joint_attention_kwargs / attention_kwargs too!
# A tensor of -inf/0 with shape [B,1,S,S] or [B,1,1,S] is a padding mask.
```

The mask is frequently **buried inside `joint_attention_kwargs["attention_mask"]`** (BriaFIBO) rather than a top-level forward argument, so it is easy to miss. It is typically a **text-padding mask**: real text tokens attend freely, padding text tokens are `-inf` (masked as both key and query), image attends to all image + real text.

**Fix.** Do NOT split the mask. Instead, patch `transformer.forward()` to **permute** the mask into all-to-all order under CP, then delegate the rest to the original forward (local RoPE from split ids is already correct). Build the permutation from each rank's local (text, image) lengths via `all_gather_object`:

```python
import torch, torch.distributed as dist

def _build_ulysses_mask_perm(local_txt, local_img, cp_config, device):
    """Map original [text, image] order -> all-to-all [text_0, img_0, text_1, img_1, ...]."""
    group = cp_config._ulysses_mesh.get_group()
    ws = dist.get_world_size(group)
    gathered = [None] * ws
    dist.all_gather_object(gathered, (int(local_txt), int(local_img)), group=group)
    txt_sizes = [g[0] for g in gathered]
    img_sizes = [g[1] for g in gathered]
    global_txt = sum(txt_sizes)
    perm, t_off, i_off = [], 0, 0
    for r in range(ws):
        perm += range(t_off, t_off + txt_sizes[r])                 # text chunk r
        base = global_txt + i_off
        perm += range(base, base + img_sizes[r])                   # image chunk r
        t_off += txt_sizes[r]; i_off += img_sizes[r]
    return torch.tensor(perm, device=device, dtype=torch.long)

def _patch_forward_for_mask(transformer):
    orig = transformer.forward
    def patched(hidden_states, encoder_hidden_states=None, *, joint_attention_kwargs=None, **kw):
        cp = getattr(transformer, "_cp_config", None)
        if cp is not None and getattr(cp, "_world_size", 1) > 1 and joint_attention_kwargs:
            mask = joint_attention_kwargs.get("attention_mask")
            if mask is not None and mask.dim() == 4:
                perm = _build_ulysses_mask_perm(
                    encoder_hidden_states.shape[1], hidden_states.shape[1], cp, mask.device)
                # 2D full mask: permute BOTH query (-2) and key (-1) dims.
                # 1D key mask [B,1,1,S]: permute only the key dim (-1).
                mask = mask.index_select(-2, perm).index_select(-1, perm)
                joint_attention_kwargs = {**joint_attention_kwargs, "attention_mask": mask}
        return orig(hidden_states, encoder_hidden_states=encoder_hidden_states,
                    joint_attention_kwargs=joint_attention_kwargs, **kw)
    transformer.forward = patched
```

Call `_patch_forward_for_mask(transformer)` inside the planner's `_apply()` before returning the (otherwise standard root-split) CP plan.

**Reference implementations:**

- **BriaFIBO** (`bria_fibo.py`) — 2D `[B,1,S,S]` text-padding mask, permuted on both dims. Fix lifted PSNR 30 → 34.6–37.8, SSIM 0.83 → 0.92–0.97, and eliminated the top-left corruption.
- **HunyuanImage** (`hunyuan.py`, `__patch__HunyuanImageTransformer2DModel_forward__`) — 1D key mask, reordered along the key dim only via interleaved `chunk`+`cat`.
- **HunyuanVideo** (`hunyuan.py`, `__patch__HunyuanVideoTransformer3DModel_forward__`) — same 1D key-mask `chunk`+`cat` reorder, plus an attention-processor patch; a second reference for the 1D pattern.

**Checklist when integrating any new model's CP:**

1. Does the model (or its pipeline) build an `attention_mask`? Search the pipeline `__call__` and the transformer `forward` / attention processor. **Check inside `joint_attention_kwargs` / `attention_kwargs` dicts.**
2. If yes, is it 1D key-mask `[B,1,1,S]` or 2D full `[B,1,S,S]`?
3. Add a forward patch to permute it into all-to-all order (key-only for 1D, both dims for 2D).
4. Verify: top-left corner clean, PSNR > 34, SSIM > 0.90.

---

## More references 

We recommend reading the following files for additional context:

- context parallelism source code: `src/cache_dit/distributed/transformers`
