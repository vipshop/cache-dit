# Checkpoint Loader Reference

When to read this file: when a checkpoint's stored tensor names, dtypes, or packing cannot be mapped by `diffusers`' plain `from_pretrained`, and the conversion has to happen inside loading. The reference implementation is the official FP8 Ideogram 4 checkpoint: `ideogram-ai/ideogram-4-fp8`.

Return to [`../SKILL.md`](../SKILL.md) for the overall integration workflow.

---

## 1. When a loader is the right layer

Use a loader only when the **checkpoint itself** needs conversion before any model code can run. Typical signals:

- Quantized weights (`float8_e4m3fn`, `float4_e2m1fn_x2`, ...) stored with separate scale tensors.
- Fused matrices (`qkv`, `gate_up`, `out_down`) that do not match the module's parameter names.
- Renamed parameters (`attention.o` in the checkpoint, `attention.to_out.0` in the module).
- A shard/index layout or sibling-file convention that `from_pretrained` does not understand.
- Parameters that remain on `meta` after loading and then fail at `pipe.to(device)`.

Do **not** use a loader for:

| Need | Correct layer |
|---|---|
| Adjusting runtime inputs or outputs | Pipeline wrapper / example |
| Caching or skipping transformer blocks | `BlockAdapter` / `PatchFunctor` |
| CP, TP, TE-P, VAE-P | Distributed planners |
| Torch-native or bitsandbytes quantization after a normal load | Existing quantize path |
| Rewriting a published bfloat16 checkpoint that already loads cleanly | No loader |

Ideogram 4's FP8 release hits every loader signal. Its three components cannot be recovered by `from_pretrained`, even with `device_map`, because the unmatched fused/scaled tensors remain on the meta device.

---

## 2. Architecture

```text
src/cache_dit/loaders/
├── __init__.py       # public exports: BaseLoader, Ideogram4Loader
├── base.py           # class-level from_pretrained patch, restore, dispatch
└── ideogram4.py      # FP8 detection, shard reads, dequantization, component rebuild
```

The boundary is deliberate:

- **`BaseLoader`** owns the `from_pretrained` patch lifecycle. It knows nothing about checkpoints.
- **`Ideogram4Loader`** owns the checkpoint format. It knows nothing about pipelines beyond the component names it returns.
- **Examples/CLI** only activate the loader. They must not contain checkpoint conversion code.

### 2.1 Public API

```python
import torch
from diffusers import Ideogram4Pipeline
from cache_dit import Ideogram4Loader

Ideogram4Loader().apply(Ideogram4Pipeline)

pipe = Ideogram4Pipeline.from_pretrained(
  "ideogram-ai/ideogram-4-fp8",
  torch_dtype=torch.bfloat16,
)
pipe.to("cuda")
```

`apply` patches the class in place and returns the same class. AutoPipeline delegation also works because it resolves `Ideogram4Pipeline` and calls its patched classmethod.

A bare `DiffusionPipeline.from_pretrained` is not covered: it instantiates the resolved class through its own loader path and never calls the patched subclass method. Document this boundary instead of adding a global `DiffusionPipeline` patch.

### 2.2 `BaseLoader` contract

A subclass implements exactly one method:

```python
def _load(self, path: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
  ...
```

Return ready-made components to inject, or `None` to let the original `from_pretrained` run untouched.

`BaseLoader.apply` guarantees:

- Patch is **class-level**, so later calls on the class and its subclasses use it.
- Patch is **idempotent by class**, not by loader instance.
- A subclass call stays a subclass call; the wrapper rebinds `original.__func__` to the receiving class.
- A class carries at most one loader; applying a different loader raises instead of silently replacing it.
- `remove()` restores the exact prior state. A class that only inherited `from_pretrained` gets a `delattr`, not a frozen bound method.
- `functools.wraps` preserves the callable name and signature metadata.
- Caller-provided components win over injected ones, including an explicit `None`.

---

## 3. Implementation workflow

### 3.1 Identify the conversion boundary

Write down, per component:

1. Checkpoint directory and weight basename.
2. Stored tensor names and shapes.
3. Runtime module parameter names and shapes.
4. Dtype and scale placement.
5. Whether the module can be built from config, or must be loaded another way.

For Ideogram 4 FP8:

| Component | Stored | Runtime |
|---|---|---|
| `transformer` | FP8 `layers.N.attention.qkv.weight` + row scales, `attention.o` | BF16 separate `to_q`/`to_k`/`to_v`, `to_out.0` |
| `unconditional_transformer` | Same layout | Same target layout |
| `text_encoder` | FP8 `q_proj`/`k_proj`/`v_proj`, row scales | BF16 `Qwen3VLModel` |

### 3.2 Detect the format without guessing

Use an explicit marker written by the exporter. Ideogram 4 writes `ideogram_fp8_weight_only: true` in `text_encoder/config.json`.

Rules:

- Missing marker/file means a normal bfloat16 checkpoint: return `None`, do not partially inspect weights.
- A network failure must propagate; never treat an unreachable repository as non-FP8.
- For checkpoints without a marker, add a format version or an explicit loader argument. Do not infer a format from a filename alone.

### 3.3 Preserve all loading selectors

Forward `revision`, `token`, `local_files_only`, `cache_dir`, `proxies`, and `force_download` to **every** marker/shard/config lookup. Otherwise the marker and weights can come from different snapshots.

### 3.4 Convert tensors, not whole shards

Ideogram 4's conversion rules:

```text
fused qkv [3 * out, in]  -> split dim 0 into to_q, to_k, to_v
attention.o              -> attention.to_out.0
weight + scale[rows]     -> (weight.float() * scale.float()[:, None]).to(dtype)
scale tensors            -> consumed, never passed to load_state_dict
```

Per-row dequantization is done in FP32 before the target cast. Keep biases and norms in their stored dtype path unless the checkpoint says otherwise.

Read one tensor at a time and release the FP8 source immediately. A 9 GiB FP8 denoiser expands to roughly 18 GiB in BF16; converting all three components at once needlessly doubles peak memory.

Scales may live in a different shard from the weight. Build a `scale_name -> shard` map first, then convert in a second pass.

### 3.5 Rebuild modules strictly

For each component:

1. Read `config.json` and construct the module from config.
2. Initialize temporary parameters directly in the target dtype when supported, restoring any global default afterwards.
3. `load_state_dict(converted, assign=True, strict=True)`.
4. Run in eval mode.

`strict=True` is the proof that every rename and fused split was handled. A silent missing key is exactly how meta parameters survive.

### 3.6 Inject through `from_pretrained`

Return a dictionary keyed by pipeline component names:

```python
{
  "transformer": transformer,
  "unconditional_transformer": unconditional_transformer,
  "text_encoder": text_encoder,
}
```

`from_pretrained` then loads every other component normally. It is a merge, not a replacement: if the caller already passed a component (even `None`), the loader must leave that decision intact.

### 3.7 Activate in the example factory

Call `apply` inside the example factory, not at module import time:

```python
@ExampleRegister.register("ideogram4", default="ideogram-ai/ideogram-4-fp8")
def ideogram4_example(args, **kwargs):
  from diffusers import Ideogram4Pipeline
  from ..loaders import Ideogram4Loader

  Ideogram4Loader().apply(Ideogram4Pipeline)
  return Example(
    args=args,
    init_config=ExampleInitConfig(
      pipeline_class=Ideogram4Pipeline,
      model_name_or_path=_path("ideogram-ai/ideogram-4-fp8"),
      ...
    ),
    ...
  )
```

`_utils` modules are imported through `_safe_import`. A failing top-level import can degrade every example to the import-error stub, so keep the loader import and application inside the factory.

---

## 4. Dtype and device semantics

### 4.1 Dtype

`diffusers` accepts `dtype` and the deprecated `torch_dtype`. Each may be a scalar or a per-component mapping with a `default` entry.

Loader rules:

- `dtype` takes precedence over `torch_dtype`.
- Reject passing both, before reading any weights.
- Apply the same resolved dtype to each rebuilt component.
- When neither is given, set `kwargs["dtype"] = torch.bfloat16` so the components loaded by diffusers stay in one precision. Do not leave a BF16 denoiser beside an FP32 VAE.
- An explicit component bypasses conversion; do not inspect or overwrite it.

### 4.2 `device_map`

A loader-built component is already materialized on the host. `from_pretrained` will not place it, so combining a conversion with `device_map` leaves CPU/GPU placement inconsistent.

Reject `device_map` **before** reading or converting any shard, but only when at least one component still needs conversion. If the caller supplied every affected component, the loader is a pass-through and `device_map` is not its business.

The supported sequence is:

```python
pipe = Ideogram4Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
pipe.to("cuda")
```

CPU offload and cache-dit's offload helpers can run after this placement.

---

## 5. Testing contract

A loader has more failure modes than a normal model adapter because it changes class-level behavior. Test at three levels.

### 5.1 Unit tests with tiny checkpoints

Build miniature safetensors fixtures that reproduce the real layout:

- Fused `qkv` with per-row scale and split verification.
- `attention.o` to `to_out.0` rename.
- Scale located in a different shard.
- Sharded index and single-file layouts.
- Non-FP8 passthrough: original call arguments are byte-for-byte unchanged.
- Marker missing vs marker present.
- `dtype`, `torch_dtype`, and per-component mapping.
- Explicit component and explicit `None` are not replaced.
- `device_map` rejected before any weight read.
- Strict rebuild has no missing or unexpected keys.

### 5.2 Patch lifecycle tests

Do not only test the conversion. Test the patch itself:

- `apply` is idempotent.
- `Sub.from_pretrained` still reports `Sub`, not the patched base.
- `remove` restores an inherited classmethod without freezing it on the subclass.
- A second, different loader is rejected.
- A replaced wrapper is not removed by mistake.
- `inspect.signature` still exposes the original argument names.

Use a recording subclass rather than monkeypatching the real pipeline globally. Remove every applied loader in a fixture teardown; a leaked class patch will shadow later tests.

### 5.3 Real checkpoint smoke

Load the actual checkpoint through the public `from_pretrained` route:

- assert the three affected components are the requested dtype;
- assert `sum(p.numel() for p in module.parameters() if p.is_meta) == 0` across the pipeline;
- assert `pipe.hf_device_map` is `None`;
- run one small generation and inspect the image.

For Ideogram 4, this smoke uses the official structured JSON prompt. Plain text is not a valid visual smoke because it can trigger the safety placeholder.

### 5.4 Acceptance

A loader is accepted only when all of the following hold:

- Non-FP8 checkpoints take the untouched original path.
- A strict converted load leaves zero meta parameters.
- Numeric output against the validated checkpoint is identical or reaches the normal PSNR/SSIM gate.
- The patch lifecycle tests pass without leaking state.
- CLI loading works through the normal `ExampleInitConfig` path.
- No conversion code remains in `examples.py`.

---

## 6. Anti-patterns

| Anti-pattern | Why it fails |
|---|---|
| Conversion in `examples.py` | Couples checkpoint format to CLI; external `from_pretrained` still fails |
| Patching all of `DiffusionPipeline.from_pretrained` | Changes global loading for every pipeline and still misses class-specific dispatch in cases |
| Auto-applying loaders at package import | Slows imports, hides side effects, and can break unrelated examples |
| Catching every hub exception as "not FP8" | Turns an auth/network failure into a misleading bfloat16 load |
| Inferring FP8 from filenames only | A re-uploaded BF16 file keeps the name and would be misread |
| Converting all components into one dict | Peak memory is the sum of every dequantized model |
| `strict=False` | Meta/unexpected-key bugs silently survive |
| Ignoring caller components | Overwrites explicit component selection and `None` |
| Allowing `device_map` with rebuilt components | Components stay on CPU while the rest move to GPU |
| Freezing `original` into the class on `remove` | Subclasses become bound to the wrong class |

---

## 7. Code map

| Purpose | File |
|---|---|
| Base patch lifecycle | `src/cache_dit/loaders/base.py` |
| Ideogram 4 FP8 loader | `src/cache_dit/loaders/ideogram4.py` |
| Public exports | `src/cache_dit/loaders/__init__.py` |
| Top-level API | `src/cache_dit/__init__.py` |
| CLI activation | `src/cache_dit/_utils/examples.py` |
| Loader tests | `tests/loaders/test_ideogram4_loader.py` |
| CLI regression tests | `tests/utils/test_ideogram4_example.py` |

---

## 8. Integration checklist

```text
- [ ] Proved the checkpoint cannot load through plain from_pretrained
- [ ] Format marker chosen and documented
- [ ] Shard/index discovery supports local dir and Hub repo
- [ ] Hub selectors forwarded to every file lookup
- [ ] Every rename and fused split enumerated
- [ ] Dequantization dtype and scale axis documented
- [ ] One-component-at-a-time memory strategy implemented
- [ ] Strict state_dict load with zero meta parameters
- [ ] dtype/torch_dtype/per-component mapping honored
- [ ] Explicit components (including None) preserved
- [ ] device_map rejected before conversion when needed
- [ ] Non-FP8 path proven unchanged
- [ ] apply/remove/subclass/idempotence tests pass
- [ ] Real checkpoint smoke through public from_pretrained
- [ ] CLI applies the loader inside the factory
- [ ] No checkpoint conversion left in examples.py
```
