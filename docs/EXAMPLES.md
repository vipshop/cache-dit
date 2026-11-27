# Examples for Cache-DiT

Minimal `cache_dit.generate` commands for the bundled examples. Single-GPU first, then multi-GPU.

## Installation

```bash
# base runtime
pip install -U cache-dit
# quantization extras: torchao (>=0.17), bitsandbytes and safetensors
pip install -U "cache-dit[quantization]"
```

TorchAO-backed quantization (`--quantize-params quant_type=float8_*`, `int8_*`, `nvfp4`, ...) needs
`torchao`, which the `quantization` extra installs. SVDQuant (`quant_type=svdq_*`) and the FP8
Ulysses communication path (`--quant-communicate-params`) additionally require the CUDA extension
compiled into cache-dit. The PyPI wheel is pure Python, so use the `+cu13` wheel from the project
index or build from source:

```bash
# CUDA 13.0+, PyTorch 2.11+ (a CUDA 13 build)
pip install -U "cache-dit[quantization]" --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/

# or build from source
git clone https://github.com/vipshop/cache-dit.git && cd cache-dit
git submodule update --init --recursive --force  # SVDQuant spdlog; SM120 CuTe CUTLASS
CACHE_DIT_BUILD_CUDA=1 MAX_JOBS=32 pip install -e ".[quantization]" --no-build-isolation
```

Without the extension, `svdq_*` and FP8 communication cannot run. For the full platform and
architecture matrix, see [Installation](./user_guide/INSTALL.md).

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
