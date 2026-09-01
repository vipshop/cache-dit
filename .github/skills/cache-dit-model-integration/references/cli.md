# Generate CLI Integration Reference

When to read this: read this file when registering a model for `python3 -m cache_dit.generate`, adding `ExampleRegister` entries, or wiring local model path environment variables. Return to `../SKILL.md` for the workflow.

## 6. generate CLI Integration

### 6.1 Concept

Cache-dit's `python3 -m cache_dit.generate` CLI discovers models through `ExampleRegister`. Each model registers a factory function that returns an `Example` dataclass specifying the pipeline class, default parameters, and model path.

### 6.2 Implementation

In `src/cache_dit/_utils/examples.py`:

```python
@ExampleRegister.register("my_model", default="org/my-model-on-huggingface")
def my_model_example(args: argparse.Namespace, **kwargs) -> Example:
    from diffusers import MyModelPipeline

    return Example(
        args=args,
        init_config=ExampleInitConfig(
            task_type=ExampleType.T2I,            # T2I, IE2I, T2V, I2V, etc.
            model_name_or_path=_path("org/my-model-on-huggingface"),
            pipeline_class=MyModelPipeline,
            # Optional: hints for BitsAndBytes quantization
            bnb_4bit_components=["text_encoder_2"],
        ),
        input_data=ExampleInputData(
            prompt="A cat holding a sign that says hello world",
            height=1024,
            width=1024,
            num_inference_steps=28,
        ),
    )
```

**`ExampleType` options:**

- `ExampleType.T2I` — Text to Image
- `ExampleType.IE2I` — Image Editing to Image
- `ExampleType.T2V` — Text to Video
- `ExampleType.I2V` — Image to Video
- `ExampleType.FLF2V` — First/Last Frames to Video
- `ExampleType.VACE` — Video All-in-one Creation and Editing

**`ExampleInputData` key fields:**

- `prompt`, `negative_prompt`: text prompts
- `height`, `width`: output resolution
- `num_inference_steps`: diffusion steps (default 28)
- `num_frames`: for video models
- `guidance_scale`: CFG scale
- `seed`: for reproducibility
- `image`, `mask_image`: for image-editing models
- `extra_input_kwargs`: dict of model-specific kwargs

### 6.3 Registration

Add to `src/cache_dit/_utils/__init__.py`:

```python
my_model_example = _safe_import(".examples", "my_model_example")
```

### 6.4 Environment Variable Mapping

In `src/cache_dit/_utils/examples.py`, the `_env_path_mapping` dict maps environment variables to default HuggingFace model IDs. Add your model:

```python
_env_path_mapping = {
    # ... existing entries ...
    "MY_MODEL_DIR": "org/my-model-on-huggingface",
}
```

Users then set `export MY_MODEL_DIR=/path/to/local/model` to avoid downloading from HuggingFace Hub.

---

## More references 

We recommend reading the following files for additional context:

- example CLI source code: `src/cache_dit/_utils`
