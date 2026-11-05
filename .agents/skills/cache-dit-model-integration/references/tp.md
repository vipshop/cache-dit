# Tensor Parallelism Reference

When to read this: read this file when implementing TP planners, using `shard_div_attr`, handling GQA/non-divisible KV heads, fused QKV, `Replicate()`, or DTensor-unsafe fused ops. Return to `../SKILL.md` for the decision chart.

## 3. Tensor Parallelism (TP)

### 3.1 Concept

Tensor Parallelism splits **model parameters** (weight matrices) across GPUs. Unlike CP which splits the input, TP splits the linear layers themselves. TP reduces per-GPU memory and can be combined with CP for hybrid parallelism (`--parallel ulysses_tp`, `--parallel ring_tp`).

cache-dit's TP is built on top of the **PyTorch tensor parallel API** (`torch.distributed.tensor.parallel`). The core primitives are:

- **`ColwiseParallel()`**: Shards a linear layer along its **output (column) dimension**. For a weight matrix `[out_features, in_features]`, each GPU holds `[out_features / tp_size, in_features]`. The input is replicated (each GPU has a full copy), and each GPU computes a partial output. Use for: Q/K/V projections, FFN first layer — anywhere the output is naturally partitioned (e.g., per-head).
- **`RowwiseParallel()`**: Shards a linear layer along its **input (row) dimension**. For a weight matrix `[out_features, in_features]`, each GPU holds `[out_features, in_features / tp_size]`. Each GPU receives a partial input and computes a partial output, then an **all-reduce** automatically sums the partial results into the full output. Use for: output projections, FFN second layer — anywhere the output needs to be reassembled.

#### `output_layouts` / `input_layouts`: Controlling DTensor Placement

Every `ColwiseParallel` and `RowwiseParallel` layer has two hidden parameters that control how its result is placed across GPUs:

| Parameter | Default | Meaning |
|---|---|---|
| `output_layouts` | `Shard(-1)` | How the layer's **output** DTensor is sharded across GPUs. `Shard(-1)` = each GPU holds a slice along the last dimension. `Replicate()` = each GPU holds a full copy. |
| `input_layouts` | `Shard(-1)` for Rowwise, `Replicate()` for Colwise | How the layer expects its **input** to be placed. `Shard(-1)` = input is already sharded along the last dim. `Replicate()` = input is a full copy on each GPU. |

**Default behavior (efficient, works for most models):**

- `ColwiseParallel()`: each GPU computes a partial output from a replicated input. By default `use_local_output=True`, so the caller receives a **plain local tensor** `[..., out_features/tp]` (not a DTensor). The `output_layouts` default `Shard(-1)` only matters if the output flows into another parallelized layer.
- `RowwiseParallel()`: expects `Shard(-1)` input (from a preceding `ColwiseParallel`), computes a partial output, then **all-reduces** to produce a full result.

**When to override to `Replicate()`:**

The defaults work when tensor flow is a clean chain: `Colwise → Rowwise → Colwise → Rowwise ...`.  You need to override the layouts when there is a **non-TP-aware consumer or producer** between two parallelized layers — typically the **attention processor**.  The processor's `.view()` / `.unflatten()` / RoPE / GQA repeat operations expect a plain tensor of a specific shape and do **not** understand DTensor shard placements.

| Scenario | Override | Why |
|---|---|---|
| **Attention processor between `Colwise(to_q)` and `Rowwise(to_out)`** | `to_q`: `ColwiseParallel(output_layouts=Replicate())` | The processor reshapes the Q projection — it needs the full `[B, S, heads*head_dim]` tensor to correctly `unflatten(-1, (heads, head_dim))`. A `Shard(-1)` output would give the processor a DTensor shard, which corrupts the reshape. `Replicate()` all-gathers the partial Q outputs so the processor sees a complete tensor. |
| **GQA: `num_kv_heads` indivisible by `tp_size`** | `to_q`: `ColwiseParallel(output_layouts=Replicate())`, `to_k`/`to_v`: leave unparallelized | K/V heads cannot be divided evenly — you cannot apply standard `ColwiseParallel` to `to_k`/`to_v`. But you can still shard `to_q` with `Replicate()` and shard `to_out` with `RowwiseParallel(input_layouts=Replicate())` to get partial TP savings for Q + output projection. See §3.4 for full details. |
| **Downstream `RowwiseParallel` receives a Replicate input** | `to_out`: `RowwiseParallel(input_layouts=Replicate())` | If the preceding `ColwiseParallel` used `output_layouts=Replicate()`, the attention processor's output is a full tensor. The downstream `RowwiseParallel` must be told its input is `Replicate()`, not `Shard(-1)`, or it will misinterpret the full tensor as a shard of a larger logical tensor (doubling the effective feature dim → shape crash). |

**Import statement:**

```python
from torch.distributed._tensor import Replicate
```

> **Key takeaway**: `Replicate()` is a **correctness tool, not an optimization**. It adds an all-gather (for output) or a redistribute (for input) at each use — only apply it when the attention processor genuinely cannot handle sharded tensors. For models without GQA and with DTensor-aware attention (or pure FFN sharding), the defaults are both correct and more efficient.  See the decision table in §3.2.1 for when to keep `shard_div_attr` vs. skip it (they are coupled: Replicate + no `shard_div_attr` vs. Shard + `shard_div_attr`).

For a detailed walkthrough, see the official PyTorch tutorial: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

### 3.2 Implementation

Create a TP Planner in the same file as the CP planner:

```python
from torch.distributed.tensor.parallel import (
    ColwiseParallel, RowwiseParallel, parallelize_module,
)
from ..utils import shard_div_attr
from .register import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)

@TensorParallelismPlannerRegister.register("MyModel")
class MyModelTensorParallelismPlanner(TensorParallelismPlanner):

    def _apply(self, transformer, parallelism_config, **kwargs):
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        transformer, layer_plans = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )
        return transformer, layer_plans

    def parallelize_transformer(self, transformer, tp_mesh):
        layer_plans = []
        for _, block in transformer.transformer_blocks.named_children():
            # CRITICAL: divide the head count by tp_size
            shard_div_attr(block.attn, "heads", tp_mesh.size())

            layer_plan = {
                # Column-wise: split along output dimension
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                # Row-wise: split along input dimension (gathers partial results)
                "attn.to_out.0": RowwiseParallel(),
                # FFN layers
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
            layer_plans.append(layer_plan)

        return transformer, layer_plans
```

**Key rules for TP planners:**

1. **`ColwiseParallel()`**: Use for layers whose output dimension is split across GPUs (Q/K/V projections, FFN first layer).
2. **`RowwiseParallel()`**: Use for layers whose input dimension is split (output projections, FFN second layer). Rowwise layers automatically all-reduce partial results.
3. **`shard_div_attr(module, "heads", tp_size)`**: Always call this on the attention module to update the head count metadata. Without it, attention computation will use the wrong number of heads.
4. **Weight rearrangement before sharding** (⚠️ key difficulty): Some models pack multiple logical projections into a single weight matrix. Before applying `RowwiseParallel` or `ColwiseParallel`, you must rearrange the weights so that the TP-shardable dimension is contiguous and correctly aligned. The canonical example is Flux's `rearrange_proj_out_weight` — a **nested helper defined inside `FluxTensorParallelismPlanner.parallelize_transformer()`** in `src/cache_dit/distributed/transformers/flux.py` (it is a local function, not a module-level import): Flux packs both the `out` and `down` projection weights into `proj_out`, so the weight must be split, rearranged with `einops.rearrange`, and re-concatenated before `RowwiseParallel` is applied. Always inspect your model's weight layout — if a single `nn.Linear` serves multiple logical roles (e.g., fused QKV, combined out+down), you must un-fuse it for the TP dimension before sharding.

### 3.2.1 `shard_div_attr`: Updating Attention Metadata (⚠️ #1 TP Gotcha)

> **This is the single most common cause of silent garbled TP output.** When you shard `to_q`/`to_k`/`to_v` with `ColwiseParallel()`, each GPU's projection now produces only `inner_dim / tp_size` output features. But the attention processor still reads the module's **integer metadata** (e.g. `attn.heads`) to reshape that output — `query.unflatten(-1, (attn.heads, -1))`, `query.view(B, S, attn.heads, head_dim)`, etc. If the metadata is not updated to reflect the reduced per-GPU count, the reshape uses the wrong number of heads and either crashes or silently produces a corrupted image. **Always update the metadata with `shard_div_attr` for every TP-sharded attention block.**

**Signature** (`src/cache_dit/distributed/utils.py`):

```python
shard_div_attr(obj, attr, tp_size, *, what=None, context=None) -> int
```

It divides the integer attribute `obj.attr` by `tp_size` **in place**, after a fail-fast divisibility check (raises `ValueError`, listing valid factors, if `obj.attr % tp_size != 0`), and returns the new value. Import it with `from ..utils import shard_div_attr`.

**Which attributes to divide:** divide **every integer attribute the attention processor uses to reshape a TP-sharded tensor**.

- **`heads`** — always. Every processor uses the head count to unflatten/view the sharded Q/K/V.
- **`inner_dim` / `mlp_hidden_dim` / etc.** — only if the processor uses them to `split`/`view` a **fused** projection whose output you sharded. Separate `to_q`/`to_k`/`to_v` projections do not need these.

**Example A — MHA with separate Q/K/V (e.g. ErnieImage, most models): only `heads`.**

```python
for _, block in transformer.layers.named_children():
    shard_div_attr(block.self_attention, "heads", tp_mesh.size())  # e.g. 24 → 12
    layer_plan = {
        "self_attention.to_q": ColwiseParallel(),
        "self_attention.to_k": ColwiseParallel(),
        "self_attention.to_v": ColwiseParallel(),
        "self_attention.to_out.0": RowwiseParallel(),
        # ... FFN ...
    }
    parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
```

The processor's `query.unflatten(-1, (attn.heads, -1))` now sees `heads=12` and correctly infers `head_dim = (inner_dim / tp) / 12`, keeping `head_dim` intact — which also makes RoPE slices like `x_in[..., :rot_dim]` exact.

**Example B — fused QKV+MLP single block (e.g. Flux2): divide `heads` AND the fused split sizes.**

```python
for _, block in transformer.single_transformer_blocks.named_children():
    self.rearrange_singleblock_weight(block, tp_size)   # un-fuse packed weights first
    shard_div_attr(block.attn, "heads", tp_size)
    shard_div_attr(block.attn, "inner_dim", tp_size)      # processor uses this to split QKV
    shard_div_attr(block.attn, "mlp_hidden_dim", tp_size) # processor uses this to split MLP
    layer_plan = {
        "attn.to_qkv_mlp_proj": ColwiseParallel(),
        "attn.to_out": RowwiseParallel(),
    }
    parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
```

Here the processor slices the single `to_qkv_mlp_proj` output using `inner_dim` and `mlp_hidden_dim`; because those slices are now per-GPU sharded, **both** metadata values must be divided in addition to `heads`. See `Flux2TensorParallelismPlanner` in `src/cache_dit/distributed/transformers/flux2.py`.

**How to know which attributes to divide:** open the attention **processor** (`__call__`) and list every place a `Q/K/V/MLP` tensor is reshaped (`.view`, `.unflatten`, `torch.split`, `.reshape`). Any integer attribute (`attn.xxx`) feeding those reshapes on a sharded tensor must be passed to `shard_div_attr`.

**When to SKIP `shard_div_attr`** (see §3.4 for details):

- **Replicate strategy** — if you shard Q/K/V with `ColwiseParallel(output_layouts=Replicate())`, the processor receives a **full (all-gathered) tensor**, so `heads` must stay at its original value. Calling `shard_div_attr` here would corrupt the reshape.
- **GQA with indivisible `num_kv_heads`** — when `num_kv_heads` is not divisible by `tp_size`, you cannot shard K/V by heads; use the Replicate strategy from §3.4 instead (and do not divide `heads`).

### 3.3 Registration

Add to `src/cache_dit/distributed/transformers/planners.py` inside `_activate_tp_planners()`:

```python
MyModelTensorParallelismPlanner = _safe_import(
    ".my_model", "MyModelTensorParallelismPlanner",
    ImportErrorTensorParallelismPlanner)
```

### 3.4 GQA Models with Non-Divisible KV Heads (Boogu-Image Pattern)

**⚠️ Common pitfall**: When a model uses Grouped Query Attention (GQA) and `num_kv_heads` is **not evenly divisible** by `tp_size`, the standard approach of applying `ColwiseParallel` to all of `to_q`, `to_k`, `to_v` will **fail** — `shard_div_attr` raises a `ValueError` because `num_kv_heads` cannot be split.

This section describes a **numerically correct** TP strategy for this case, using the **Boogu-Image** model (`num_attention_heads=28`, `num_kv_heads=7`, `tp_size=2`) as a concrete reference. The full implementation is at `src/cache_dit/distributed/transformers/boogu_image.py`.

> 🔴 **PERFORMANCE CAVEAT — read this before sharding attention on a GQA model.**
> The `Replicate`-based attention TP below is **correct** (it produces the right image), but it is **not necessarily faster** — and for long sequences it is usually **slower than a single GPU**. Because K/V cannot be sharded, the only way to shard `to_q` is `ColwiseParallel(output_layouts=Replicate())`, which inserts an **all-gather on Q at every attention layer**. On a long joint sequence that all-gather dominates the communication budget and overwhelms the compute savings.
>
> **This is exactly what happened to Boogu-Image in production.** The team measured FFN-only TP as both faster and more accurate than attention TP (`108.9s` vs `115.7s` inference; PSNR `50.2` vs `49.9`), so **the shipped `boogu_image.py` now comments out all attention Q/K/V/out sharding and parallelizes only FFN + modulation** (see `_single_stream_layer_plan` / `_double_stream_layer_plan`). The `Replicate` attention plan below is kept here as a *correctness reference* and for models where attention compute genuinely dominates — **benchmark it against FFN-only TP before shipping it, and default to FFN-only for long-sequence GQA models.**

#### 3.4.1 Problem: DTensor Shard Placement in Attention Processors

A naïve attempt is to shard only `to_q` with `ColwiseParallel()` (keeping `to_k`/`to_v` replicated), plus `to_out.0` with `RowwiseParallel()`, and call `shard_div_attr` to update `attn.heads`:

```python
# ❌ WRONG — produces garbled output (SSIM ≈ 0.02)
shard_div_attr(block.attn, "heads", tp_mesh.size())  # 28 → 14
layer_plan = {
    "attn.to_q": ColwiseParallel(),
    "attn.to_out.0": RowwiseParallel(),
}
```

**Why this fails:** `ColwiseParallel()` (without explicit `output_layouts`) produces a `DTensor` with `Shard(-1)` placement along the **last dimension**. When this `Shard`-placed DTensor enters the attention processor's `.view()` / `.transpose()` operations, PyTorch DTensor's sharding propagation can place the shard along `head_dim` (each GPU gets 60 out of 120 dims) rather than along the **heads** dimension (each GPU gets a subset of complete heads). This corrupts the GQA repeat and SDPA computation, producing severely garbled output.

#### 3.4.2 Correct Strategy: `output_layouts=Replicate()` + `input_layouts=Replicate()`

The fix is to ensure the attention processor **always sees a complete (non-DTensor) tensor** for Q, while still parallelizing the linear projections:

```python
# ✅ CORRECT — PSNR > 46 dB, SSIM > 0.99
layer_plan = {
    "attn.to_q": ColwiseParallel(output_layouts=Replicate()),
    "attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
}
# NOTE: Do NOT call shard_div_attr — attn.heads stays at 28
```

**How it works:**

| Component | ParallelStyle | What happens |
|---|---|---|
| `attn.to_q` | `ColwiseParallel(output_layouts=Replicate())` | Each GPU computes half the Q features, then **all-gather** merges them. The attention processor sees a full (replicated) Q tensor. |
| `attn.to_k` / `attn.to_v` | **Not parallelized** (omitted from plan) | Full K/V weights on each GPU. `num_kv_heads=7` is indivisible by `tp_size=2`. |
| `attn.to_out.0` | `RowwiseParallel(input_layouts=Replicate())` | Each GPU computes partial output from its weight shard, then **all-reduce** combines them. |

**Why `input_layouts=Replicate()` on `RowwiseParallel` is critical:**  
`RowwiseParallel`'s default `input_layouts` is `Shard(-1)`. If the actual input is a replicated (full) tensor from the attention processor, PyTorch TP will **misinterpret** the local tensor as a shard of a larger logical tensor — effectively doubling the feature dimension. This causes shape mismatches like `[B, S, 2×D] × [D, D]` → crash. Setting `input_layouts=Replicate()` explicitly tells TP that each GPU holds a full copy, so it can correctly `redistribute` to `Shard(-1)` before computing partial contributions.

**When to keep `shard_div_attr` vs. when to skip it:**

| Scenario | Call `shard_div_attr`? | Reason |
|---|---|---|
| Standard `ColwiseParallel()` (output is `Shard(-1)`) | **Yes** | `attn.heads` must be divided so `head_dim` stays correct for the sharded Q dimension. |
| `ColwiseParallel(output_layouts=Replicate())` | **No** | The attention processor sees the full Q dimension (`3360`). `attn.heads` must stay at the original value (`28`) so `head_dim = 3360/28 = 120` is correct. |

#### 3.4.3 Double-Stream / Processor-Owned QKV Projections

Some models (e.g., Boogu-Image's double-stream layers) have Q/K/V projections that live **inside the attention processor** rather than on the `attn` module. In Boogu-Image, `img_instruct_attn.to_q` / `to_k` / `to_v` are **deleted** at init — the processor owns `img_to_q`, `instruct_to_q`, etc. The same `Replicate` strategy applies, but the module paths in the layer plan must target the processor's attributes:

```python
# Joint cross-attention: QKV weights are in the processor, not on attn
layer_plan = {
    "img_instruct_attn.processor.img_to_q": ColwiseParallel(output_layouts=Replicate()),
    "img_instruct_attn.processor.instruct_to_q": ColwiseParallel(output_layouts=Replicate()),
    "img_instruct_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
    # img_self_attn uses standard Diffusers Attention (same pattern as single-stream)
    "img_self_attn.to_q": ColwiseParallel(output_layouts=Replicate()),
    "img_self_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
}
```

**Key insight for double-stream blocks:** Intermediate output projections (`img_out`, `instruct_out` in the processor) should **not** be parallelized. They sit between the flash-attention output and the final `to_out.0`, and parallelizing them would cause **chained all-reduces** (`img_out` all-reduce → `instruct_out` all-reduce → `to_out.0` all-reduce). Only parallelize the **final output projection** (`attn.to_out.0`) in each attention block.

#### 3.4.4 Verification Results (Boogu-Image, tp_size=2)

> **These numbers were measured on the earlier `Replicate`-based attention-TP prototype**, to prove it is *numerically* correct. They are **not** the shipped configuration — see the performance caveat at the top of §3.4. Note that FFN-only TP already reaches PSNR 47.61 and was ultimately chosen for production because it is faster **and** slightly more accurate (PSNR 50.2 vs 49.9) than full attention TP.

| Configuration | Steps | PSNR (dB) | SSIM |
|---|---|---|---|
| **FFN-only TP (shipped)** | 50 | 47.61 | — |
| Full TP (Q+out+FFN, single-stream only) | 4 | 48.45 | 0.9945 |
| Full TP (single + double-stream attention) | 4 | 46.77 | 0.9944 |
| **Full TP (all layers)** | **50** | **49.99** | **0.9974** |

Acceptance criteria: PSNR > 35 dB, SSIM > 0.90. All configurations pass comfortably.

#### 3.4.5 Case Study: Fused QKV Projection (JoyImage)

JoyImageEditTransformer uses **fused QKV projections** — a single `nn.Linear(dim, 3 * inner_dim)` whose weight layout is `[Q_all_heads, K_all_heads, V_all_heads]` (all Q heads packed first, then all K heads, then all V heads). The attention processor splits the output with `qkv.chunk(3, dim=-1)`.

**Why standard `ColwiseParallel()` breaks:** Linear row-sharding splits the weight into two contiguous halves, but the Q/K/V boundary is at `1/3` and `2/3` of the rows, not at `1/2`. With `tp_size=2`, GPU 0 receives `[Q_all, K_half]` (no V) and GPU 1 receives `[K_half, V_all]` (no Q). The processor's `chunk(3)` then slices garbage, producing PSNR ~28 dB / SSIM ~0.10.

**Two correct strategies** (both verified):

**Strategy A — Replicate (recommended, matches §3.4.2):** Shard the fused QKV weights but all-gather the output so the processor sees the complete Q/K/V tensor. The output projection must use `RowwiseParallel(input_layouts=Replicate())` because its input is a full (all-gathered) tensor, not a shard.

```python
# ✅ CORRECT — PSNR 47.37 dB, SSIM 0.995 (tp_size=2, 4-step)
layer_plan = {
    "attn.img_attn_qkv": ColwiseParallel(output_layouts=Replicate()),
    "attn.txt_attn_qkv": ColwiseParallel(output_layouts=Replicate()),
    "attn.img_attn_proj": RowwiseParallel(input_layouts=Replicate()),  # ⚠️ required
    "attn.txt_attn_proj": RowwiseParallel(input_layouts=Replicate()),  # ⚠️ required
    "img_mlp.net.0.proj": ColwiseParallel(),
    "img_mlp.net.2": RowwiseParallel(),
    "txt_mlp.net.0.proj": ColwiseParallel(),
    "txt_mlp.net.2": RowwiseParallel(),
}
# NOTE: Do NOT call shard_div_attr — see below.
```

**⚠️ Why `input_layouts=Replicate()` on `RowwiseParallel` is mandatory here:** `RowwiseParallel`'s default `input_layouts` is `Shard(-1)`. When the upstream `ColwiseParallel(output_layouts=Replicate())` produces a full (replicated) tensor, the default `Shard(-1)` misinterprets the full local tensor as a shard of a `2×D` logical tensor, causing a shape mismatch crash like `[8192, 8192] X [4096, 4096]`. Explicitly setting `input_layouts=Replicate()` tells the TP runtime that each GPU already holds a full copy, so it redistributes to `Shard(-1)` internally before computing partial contributions.

**Strategy B — Weight rearrangement (more complex, enables true sharding):** Reorder the fused weight from `[Q_all, K_all, V_all]` to head-interleaved `[Q_h0, K_h0, V_h0, Q_h1, K_h1, V_h1, ...]` *before* `parallelize_module`. Then standard `ColwiseParallel()` gives each GPU complete Q/K/V for its head subset. This requires patching the attention processor to split by head instead of `chunk(3)`. Use only when the Replicate strategy's all-gather communication is a measured bottleneck.

##### When to call `shard_div_attr` — the decisive rule

> **`shard_div_attr` divides integer metadata (e.g. `attn.heads`) so that the attention processor's `unflatten(-1, (heads, -1))` produces the correct per-GPU `head_dim`. Whether to call it depends entirely on what the processor actually sees — not on the model architecture, not on whether Q/K/V are fused, not on whether you use ColwiseParallel.**

**Call `shard_div_attr` when the processor sees a sharded tensor.** This happens when:
- `ColwiseParallel()` with default `output_layouts=Shard(-1)` — each GPU's local output is `[B, S, inner_dim/tp]`, and `unflatten(-1, (heads, -1))` would infer `head_dim = (inner_dim/tp) / heads`, which is wrong. Dividing `heads` by `tp` keeps `head_dim` correct.
- This is the **common case** for separate `to_q`/`to_k`/`to_v` projections (Flux, Wan, QwenImage, most models).

**Do NOT call `shard_div_attr` when the processor sees a full (replicated) tensor.** This happens when:
- `ColwiseParallel(output_layouts=Replicate())` — the output is all-gathered, so each GPU sees the full `inner_dim`. `unflatten(-1, (heads, -1))` already produces the correct `head_dim = inner_dim / heads`. Dividing `heads` here would corrupt the reshape.
- This applies to the **Replicate strategy** (§3.4.2, §3.4.5 Strategy A) and to GQA models where K/V are left unparallelized.

**How to decide in 3 seconds:** Open the attention processor's `__call__`. Find the line that reshapes the Q/K/V output of the sharded projection. Is the tensor the processor receives **full-dimension** (all-gathered) or **shard-dimension** (local slice)?
- Full-dimension → **skip** `shard_div_attr`.
- Shard-dimension → **call** `shard_div_attr`.

| Projection style | `output_layouts` | Processor sees | `shard_div_attr`? |
|---|---|---|---|
| Separate `to_q`/`to_k`/`to_v` | `Shard(-1)` (default) | `[B, S, inner_dim/tp]` | **Yes** |
| Fused QKV, Replicate strategy | `Replicate()` | `[B, S, 3*inner_dim]` (full) | **No** |
| Fused QKV, rearranged + standard Colwise | `Shard(-1)` (default) | `[B, S, 3*inner_dim/tp]` | **Yes** (and also divide any split-size attrs the processor uses) |

**Verification (JoyImage, tp_size=2):**

| Configuration | Steps | PSNR (dB) | SSIM | Time |
|---|---|---|---|---|
| Baseline (1 GPU) | 4 | ∞ | 1.000 | 15.77s |
| FFN-only TP | 4 | 29.12 | 0.613 | 13.00s |
| **Replicate TP (Strategy A)** | **4** | **47.37** | **0.995** | **13.20s** |

FFN-only TP alone is insufficient for JoyImage (SSIM 0.613 fails the >0.90 bar) because the attention path is unparallelized and the 4-step diffusion amplifies numerical drift. The Replicate strategy fixes this with negligible overhead (13.20s vs 13.00s).

#### 3.4.6 Head Padding (Alternative, Not Yet Needed)

Another approach for GQA TP is to **pad** `num_kv_heads` to the nearest multiple of `tp_size` (e.g., 7 → 8) by adding zero-initialized rows to `to_k.weight` / `to_v.weight`, and correspondingly pad `to_q.weight` and `to_out.0.weight` to maintain consistent shapes. This would allow standard `ColwiseParallel` for all of Q/K/V. However, the `Replicate` strategy described above is simpler and already achieves near-identical numerical results, so padding is reserved for future optimization when attention compute becomes the bottleneck.

#### 3.4.7 DTensor-Unsafe Fused Ops: Patch Before Parallelizing

**⚠️ Common pitfall**: Some models use environment-variable-gated fused CUDA kernels (e.g., flash_attn SwiGLU, triton RMSNorm) that operate on raw GPU memory and do **not** understand DTensor shard placements. When TP wraps model parameters as DTensors, these fused kernels will either crash (`cudaErrorIllegalAddress`) or silently produce wrong results.

**Symptoms:**
- `CUDA error: an illegal memory access was encountered` during NCCL collective (the kernel corrupted memory that NCCL later tries to read).
- PSNR collapses to ~28 dB, SSIM drops to ~0.28 — the model runs without crashing but produces garbled output.

**Root cause:** The fused kernel reads/writes memory assuming a contiguous full tensor, but DTensor has split the tensor across GPUs. The kernel accesses addresses that belong to another GPU's shard → illegal memory access.

**Fix: Monkey-patch the unsafe ops back to DTensor-compatible PyTorch equivalents BEFORE calling `parallelize_module`.**

Reference implementation: `_patch_dtensor_unsafe_modules()` in `src/cache_dit/distributed/transformers/boogu_image.py`.

**Template:**

```python
def _patch_dtensor_unsafe_for_tp(block: nn.Module) -> None:
    """Replace fused CUDA kernels with DTensor-safe PyTorch ops before TP."""

    # 1. SwiGLU: flash_attn fused kernel → PyTorch F.silu + multiply
    for ffn_name in ("feed_forward", "img_feed_forward", "instruct_feed_forward"):
        ffn = getattr(block, ffn_name, None)
        if ffn is None:
            continue
        swiglu_fn = getattr(ffn, "swiglu", None)
        if swiglu_fn is None:
            continue
        fn_cls = getattr(swiglu_fn, "__self__", None)
        if fn_cls is not None and getattr(fn_cls, "__module__", "").startswith("flash_attn"):
            ffn.swiglu = _dtensor_safe_swiglu  # F.silu(x).to(dtype) * y

    # 2. RMSNorm: triton fused kernel → torch.nn.RMSNorm
    #    CRITICAL: copy the learned weight from the old module
    for name, module in list(block.named_modules()):
        if type(module).__module__ == "boogu.ops.triton.layer_norm":
            weight = module.weight
            new_norm = torch.nn.RMSNorm(
                weight.shape[0], eps=module.eps,
                elementwise_affine=True, device=weight.device, dtype=weight.dtype)
            new_norm.weight.data.copy_(weight.data)  # ← MUST copy weight!
            parent_path, leaf = name.rsplit(".", 1)
            setattr(block.get_submodule(parent_path), leaf, new_norm)
```

**Common fused ops that need patching:**

| Op | Unsafe Variant | DTensor-Safe Replacement |
|---|---|---|
| SwiGLU | `flash_attn.ops.activations.swiglu` | `F.silu(x.float()).to(x.dtype) * y` |
| RMSNorm / LayerNorm | `triton.layer_norm.RMSNorm` | `torch.nn.RMSNorm` |
| Fused MLP | Any single-kernel fused projection+activation | Decompose into individual `nn.Linear` + activation |

**How to detect:** Identify these ops by searching the model source for `import flash_attn` / `import triton` and checking for env-var-gated feature flags like `os.getenv("device")`. Any op behind such a gate that is a custom CUDA kernel is suspect.

---

## More references 

We recommend reading the following files for additional context:

- Tensor parallelism source code: `src/cache_dit/distributed/transformers`
