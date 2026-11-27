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
# baseline (FLUX.1-dev; set FLUX_DIR or pass --model-path)
python3 -m cache_dit.generate flux --save-path flux.base.png

# DBCache + compile
python3 -m cache_dit.generate flux \
  --cache-params 'type=DBCache' \
  --compile \
  --save-path flux.cache.png

# FP8 + compile
python3 -m cache_dit.generate flux \
  --quantize-params 'quant_type=float8_per_row' \
  --compile \
  --save-path flux.fp8.png
```

Common overrides: `--model-path`, `--prompt`, `--steps`, `--height`, `--width`, `--seed`,
`--save-path`. A model directory or repo id can stand in for the example name, e.g.
`python3 -m cache_dit.generate /path/to/FLUX.1-dev`; add `--task T2I` when the checkpoint serves
several tasks.

## Multi GPU

The CLI starts the workers itself (`--launcher auto`), so no external `torchrun` is required. The
product of the parallel degrees must equal the number of processes.

```bash
# Ulysses 2 (2 cards)
CUDA_VISIBLE_DEVICES=4,5 python3 -m cache_dit.generate flux \
  --parallel-params 'ulysses=2' \
  --cache-params 'type=DBCache' \
  --save-path flux.ulysses2.png

# TP2 + text-encoder parallelism
CUDA_VISIBLE_DEVICES=4,5 python3 -m cache_dit.generate flux \
  --parallel-params 'tp=2,text=true' \
  --save-path flux.tp2.tep.png

# Ulysses 2 x TP 2 (4 cards)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m cache_dit.generate flux \
  --parallel-params 'ulysses=2,tp=2' \
  --compile \
  --save-path flux.ulysses2.tp2.png
```

`vae=true`, `text=true` and `controlnet=true` are modifiers: keep them in the same
`--parallel-params` value as an active `ulysses` / `ring` / `tp` degree. A value with only a
modifier is a no-op. `--quant-communicate-params 'bits=8'` additionally requires `ulysses > 1`.

## Discover more

```bash
python3 -m cache_dit.generate list      # registered examples and their default models
python3 -m cache_dit.generate --help    # every option, default and alias
```

Feature guides: [cache](./user_guide/DBCACHE_DESIGN.md), [quantization](./user_guide/QUANTIZATION.md),
[attention](./user_guide/ATTENTION.md), [parallelism](./user_guide/CONTEXT_PARALLEL.md),
[CPU offload](./user_guide/OFFLOAD.md) and [compile](./user_guide/COMPILE.md).
