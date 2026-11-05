# Attention Backend Dispatch

Read this whenever adding or patching a model-specific attention processor, implementing CP attention, or selecting an attention backend.

## 1. Mandatory dispatcher

New cache-dit attention code must call cache-dit's `_dispatch_attention_fn`, not diffusers' `dispatch_attention_fn` and not a backend function directly.

```python
from ...attention import _dispatch_attention_fn

hidden_states = _dispatch_attention_fn(
  query,
  key,
  value,
  attn_mask=attention_mask,
  dropout_p=0.0,
  backend=self._attention_backend,
  parallel_config=None,
)
```

Use the appropriate relative-import depth for the module. Inside `distributed/transformers/`, the import is `from ...attention import _dispatch_attention_fn`.

Why this is required:

- `_dispatch_attention_fn` resolves backends through `_AttnBackendRegistry`, including cache-dit backends such as FFPA, SageAttention, FlashAttention, and native SDPA.
- It applies backend-specific argument filtering and constraints. Calling a backend directly bypasses these checks.
- It accepts either the active backend (`backend=None`) or an explicit backend selected by cache-dit.
- Calling diffusers' dispatcher in a patched cache-dit processor can silently make `--attn` and programmatic backend selection ineffective.

Keeping an unmodified upstream processor as the fallback for an unarmed, non-cache-dit path is fine. Do not patch diffusers itself. The requirement applies to new cache-dit dispatch sites.

## 2. Signature and tensor contract

The main call parameters are:

| Parameter | Meaning |
|---|---|
| `query`, `key`, `value` | Attention tensors in the layout expected by the selected cache-dit backend. Preserve the model processor's established layout. |
| `attn_mask` | Full mask aligned with the actual Q/K/V sequence order. |
| `dropout_p`, `is_causal`, `scale` | Standard attention semantics; inference normally uses `dropout_p=0.0`. |
| `enable_gqa` | Whether the backend may use native GQA. Benchmark against explicit K/V head repetition when relevant. |
| `attention_kwargs` | Optional backend-specific arguments. |
| `backend` | `None` for the active registry backend, or an explicit backend name/enum. |
| `cp_config` / `parallel_config` | CP runtime when the dispatcher owns communication. They are aliases resolved internally; do not pass both with different values. |

Do not assume that a backend accepts every argument. The registry filters unsupported arguments and runs enabled constraints. A model-specific patch must still ensure that its mask, head count, dtype, layout, and head dimension satisfy the chosen backend.

## 3. CP: exactly one owner of communication

This is the most important CP rule.

### Dispatcher owns CP

Pass the active CP config when using the standard templated Ulysses/Ring path. The dispatcher then performs the required communication and backend call.

### Model-specific processor owns CP

When the processor already performs Q/K/V all-to-all, K/V gather, cache management, or output all-to-all, dispatch with `parallel_config=None`:

```python
hidden_states = _dispatch_attention_fn(
  query,
  key,
  value,
  attn_mask=attention_mask,
  dropout_p=0.0,
  backend=self._attention_backend,
  parallel_config=None,
)
```

Passing the active CP config in this case communicates twice: once in the model-specific processor and again inside the templated dispatcher. It can corrupt sequence order even when tensor shapes remain valid.

For a two-phase model such as prefix-KV-cache prefill/decode:

- Prefill may use model-specific block-causal segments and an explicit backend choice required by that path.
- Cached/decode attention should normally forward `self._attention_backend`, so `--attn` remains effective.
- Global masks and segment indices must describe the sequence after any all-to-all or all-gather, not the local input shard.
- Remove sequence padding before attention, or explicitly mask every padded key. Trim padded outputs only after restoring the communication layout required by the inverse collective.

## 4. Backend selection rules

1. Use `backend=None` when the active cache-dit backend should apply.
2. Forward `self._attention_backend` when the upstream processor exposes it and cache-dit sets it for the model.
3. Force a specific backend only when the algorithm requires it, such as a flex block-causal prefill. Document why other backends are invalid.
4. Do not call `torch.nn.functional.scaled_dot_product_attention`, FlashAttention, FFPA, or another backend directly from a model-specific planner unless the task is explicitly implementing a new registry backend.
5. Do not infer performance from FLOPs alone. Benchmark the selected backend at the real prefill and decode shapes, with warmup and synchronized timing.

For GQA, compare native `enable_gqa=True` against repeating K/V to MHA layout and using `enable_gqa=False`. Some SDPA backends otherwise fall back to a slow kernel.

## 5. Verification checklist

- Native single-rank patched processor matches the upstream processor bit-exactly when communication is an identity operation.
- Select a non-default cache-dit backend and verify that the model-specific path actually dispatches to it; a successful run with only the default backend does not prove routing works.
- For explicit CP communication, assert that the dispatcher receives `parallel_config=None`.
- Test prefill and every cached/decode mode separately, including text padding masks and non-divisible sequence lengths.
- Test the actual local head count after TP, not only the transformer's global `attn.heads` attribute.
- Compare outputs with PSNR and SSIM, inspect images for localized corruption, and measure backend performance at both phase shapes.
