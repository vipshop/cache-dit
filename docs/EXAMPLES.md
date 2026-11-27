# Examples for Cache-DiT

Minimal `cache_dit.generate` commands for the bundled examples. Single-GPU first, then multi-GPU.

## Installation

```bash
pip install -U cache-dit
```

For a source install, optional kernels and platform packages, see
[Installation](./user_guide/INSTALL.md).

## Single GPU

```bash
python3 -m cache_dit.generate \
  /path/to/FLUX.1-dev \
  --cache-config 'DBCache,rdt=0.12' \
  --calibrator-config 'taylorseer,order=1' \
  --quantize-config 'float8_per_row' \
  --attn-config 'native' \
  --compile
```

This is the full single-card stack: DBCache + TaylorSeer + FP8 + native attention + compile. Drop the
optimization flags for a plain baseline; use the `flux` example name with `FLUX_DIR` instead of the
model path if preferred; add `--save-path` to control the output name.

Common overrides: `--model-path`, `--prompt`, `--steps`, `--height`, `--width`, `--seed`,
`--save-path`. A model directory or repo id can stand in for the example name; add `--task T2I` when
the checkpoint serves several tasks.

## Multi GPU

The CLI starts the workers itself (`--launcher auto`), so no external `torchrun` is required. The
product of the parallel degrees must equal the number of processes.

```bash
CUDA_VISIBLE_DEVICES=4,5 python3 -m cache_dit.generate \
  /path/to/FLUX.1-dev \
  --parallel-config 'ulysses=2' \
  --quant-communicate-config 'bits=8' \
  --cache-config 'DBCache,rdt=0.12' \
  --calibrator-config 'taylorseer,order=1' \
  --quantize-config 'float8_per_row' \
  --attn-config 'native' \
  --compile
```

This is the full two-card stack: Ulysses 2 + FP8 communication quantization + DBCache + TaylorSeer +
FP8 weights + native attention + compile. Replace `'ulysses=2'` with `'tp=2'` for tensor parallelism,
or `'ulysses=2,tp=2'` for the 4-card CP+TP hybrid, as long as the degrees multiply to the process
count.

`vae=true`, `text=true` and `controlnet=true` are modifiers: keep them in the same
`--parallel-params` value as an active `ulysses` / `ring` / `tp` degree. A value with only a
modifier is a no-op.

## Discover more

```bash
python3 -m cache_dit.generate list      # registered examples and their default models
python3 -m cache_dit.generate --help    # every option, default and alias
```

Feature guides: [cache](./user_guide/DBCACHE_DESIGN.md), [quantization](./user_guide/QUANTIZATION.md),
[attention](./user_guide/ATTENTION.md), [parallelism](./user_guide/CONTEXT_PARALLEL.md),
[CPU offload](./user_guide/OFFLOAD.md) and [compile](./user_guide/COMPILE.md).
