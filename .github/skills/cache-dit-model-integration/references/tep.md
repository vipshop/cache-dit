# Text Encoder Parallelism Reference

When to read this: read this file when a model uses a text encoder not already supported by cache-dit, or when adding a new text encoder TP planner. Return to `../SKILL.md` for the integration order.

## 4. Text Encoder Parallelism (TE-P)

### 4.1 Concept

TE-P shards the text encoder (e.g., T5, CLIP) across GPUs using the same TP mechanism. It is independent of transformer TP — you can use TE-P with or without transformer TP. Activate via `--parallel-text` (or `--parallel-text-encoder`).

### 4.2 When to implement

Implement TE-P only if your model uses a text encoder that is **not yet supported** by cache-dit. Currently supported encoders include: T5, UMT5, Mistral, Qwen2.5-VL, Qwen3, Llama, Gemma, Glm, GlmImage, SmolLM3. If your model uses one of these, TE-P works out of the box.

### 4.3 Finding the TP Plan from HuggingFace Transformers

**⚠️ Key technique**: Most HuggingFace transformer models already define a canonical TP plan in their config class. This is the best reference for which layers should be `colwise` vs `rowwise`. For example, `cache-dit`'s `Qwen3TensorParallelismPlanner` (in `src/cache_dit/distributed/text_encoders/qwen3.py`) hardcodes a layer plan that **mirrors the structure of** `Qwen3Config.base_model_tp_plan` in the transformers library (it does not read the config attribute at runtime — it just uses the same colwise/rowwise mapping):

```python
# From transformers: Qwen3Config.base_model_tp_plan
base_model_tp_plan = {
    "layers.*.self_attn.q_proj": "colwise",
    "layers.*.self_attn.k_proj": "colwise",
    "layers.*.self_attn.v_proj": "colwise",
    "layers.*.self_attn.o_proj": "rowwise",
    "layers.*.mlp.gate_proj": "colwise",
    "layers.*.mlp.up_proj": "colwise",
    "layers.*.mlp.down_proj": "rowwise",
}
```

When implementing a new TE-P planner, always check the encoder's `Config` class (e.g., `T5Config`, `LlamaConfig`, `GemmaConfig`) for a similar `base_model_tp_plan` — it tells you exactly which layer names map to `colwise`/`rowwise`, saving you from reverse-engineering the architecture.

### 4.4 Implementation

In `src/cache_dit/distributed/text_encoders/<encoder_name>.py`:

```python
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from .register import (
    TextEncoderTensorParallelismPlanner,
    TextEncoderTensorParallelismPlannerRegister,
)

@TextEncoderTensorParallelismPlannerRegister.register("MyTextEncoderModel")
class MyTextEncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

    def _apply(self, text_encoder, parallelism_config, **kwargs):
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        layer_plans = []
        for _, block in text_encoder.encoder.block.named_children():
            layer_plan = {
                "attention.self.query": ColwiseParallel(),
                "attention.self.key": ColwiseParallel(),
                "attention.self.value": ColwiseParallel(),
                "attention.output.dense": RowwiseParallel(),
                "intermediate.dense": ColwiseParallel(),
                "output.dense": RowwiseParallel(),
            }
            parallelize_module(block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
            layer_plans.append(layer_plan)
        return text_encoder, layer_plans
```

### 4.5 Registration

Add to `src/cache_dit/distributed/text_encoders/planners.py` inside `_activate_text_encoder_tp_planners()`:

```python
MyTextEncoderTensorParallelismPlanner = _safe_import(
    ".my_encoder", "MyTextEncoderTensorParallelismPlanner")
```

---

## More references 

We recommend reading the following files for additional context:

- Text encoder Tensor parallelism source code: `src/cache_dit/distributed/text_encoders`
