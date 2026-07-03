# Installation Reference

When to read this: read this file before running any cache-dit model integration test after Python code changes, or when deciding whether a full CUDA/SVDQuant rebuild is needed. Return to `../SKILL.md` for the integration workflow.

## I.1 Environment

Use the `cdit` conda environment when it is available:

```bash
conda activate cdit
```

If the environment does not exist on the current machine, use the default `python3` environment only when the user or workspace setup explicitly allows it.

## I.2 Python-Only Model Integration

For model integration changes that only touch Python files, install cache-dit in editable mode:

```bash
pip install -e "." --no-build-isolation
```

SVDQuant C++ compilation is not required for Python-only model integration, Cache adapters, CP/TP planners, CLI example registration, TE-P planners, VAE-P planners, or test command updates.

## I.3 When a Rebuild Is Needed

Only use the CUDA/SVDQuant build path when the integration changes C++/CUDA extension code or explicitly needs the optional quantization extension. Do not rebuild it for normal new-model support.

```bash
CACHE_DIT_BUILD_SVDQUANT=1 pip install -e ".[quantization]" --no-build-isolation
```

## I.4 Pre-Test Checklist

- Activate the environment before running tests.
- Install editable cache-dit after Python code changes.
- Set local model path environment variables before generation tests; see [`./testing.md`](./testing.md).
- Do not change core dependency versions (`torch`, `torchvision`, `transformers`, `diffusers`, `cache-dit`, `triton`) while preparing the environment.
