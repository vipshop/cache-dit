# Installation  

## Prequisites

- Python >= 3.11 (<span style="color:#c77dff;">3.12</span>, recommended)
- PyTorch >= 2.7.1 (<span style="color:#c77dff;">2.11.0</span>, recommended)
- CUDA >= 12.6 (>= <span style="color:#c77dff;">13.0</span>, recommended) for Nvidia GPU
- Diffusers >= 0.36.0 (>= <span style="color:#c77dff;">0.37.0</span>, recommended)
- TorchAo >= 0.15.0 (>= <span style="color:#c77dff;">0.17.0</span>, recommended)

## Installation with Nvidia GPU

<div id="installation"></div>

Firstly, install the required dependencies, including PyTorch, Diffusers, and TorchAo. We recommend installing the <span style="color:#c77dff;">latest</span> versions for better compatibility and performance.

```bash
pip install torch==2.11.0 torchvision torchaudio triton \
  transformers diffusers accelerate torchao opencv-python-headless \
  einops imageio-ffmpeg ftfy numpy
```

Then, you can install <span style="color:#c77dff;">Cache-DiT</span> from PyPI:

```bash
pip install -U cache-dit # PyPI, stable release.
pip install git+https://github.com/vipshop/cache-dit.git # latest
```

Or, install Cache-DiT with <span style="color:#c77dff;">SVDQuant</span> support (Experimental) -- the
same distribution with the CUDA extension compiled in, served from our own wheel index:

```bash
# Required: CUDA 13.0+, PyTorch 2.11+ (a CUDA 13 build), and the GLIBC floor the
# current release was built against (2.38) -- the tag of every wheel is derived from
# the symbols it needs, and the index page lists what it serves.
pip install -U cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/
# Optional: build Cache-DiT from source with the CUDA extensions.
git clone https://github.com/vipshop/cache-dit.git && cd cache-dit
git submodule update --init --recursive --force # SVDQuant needs spdlog; SM120 CuTe needs CUTLASS 4.8.1
CACHE_DIT_BUILD_CUDA=1 MAX_JOBS=32 pip install -e ".[quantization]" --no-build-isolation
```

`CACHE_DIT_BUILD_CUDA=1` is the only CUDA build switch, and it builds the single
`cache_dit._C_ops` extension, which links every operator family into one module: the
SVDQuant W4A4 kernels, the cache-dit FP8 quantize/GEMM kernels, the SM120 NVFP4 quantize
and CuTe GEMM kernels, the communication operators used by FP8 Ulysses, the attention-state
merge operator used by Ring Attention, and the model-specific fused operators. There is no
per-family switch. Because SVDQuant is part of the same module, the spdlog and CUTLASS
submodules are now prerequisites of the whole extension rather than of one family.

A build whose resolved SM targets cannot host the extension **fails** instead of producing a
wheel that silently lacks every CUDA operator (`_warn_skipped` still reports which individual
targets the architecture predicate dropped). The floor is the intersection of the per-family
requirements: **sm89+ and sm90 (Hopper) excluded** -- the communication and attention kernels
need sm89+, and the SVDQuant INT4 kernels exclude Hopper. A Hopper-only architecture list
therefore builds nothing, and every fused/FP8/FP4/SVDQuant call site falls back to its eager
path or raises the guided "extension is not available" error.

Importing the extension successfully only means it was compiled in: the wheel ships cubins for
the architectures it was built for, so running it on a GPU outside that set fails at the first
kernel launch. Build with `CACHE_DIT_CUDA_ARCH_LIST` (or `CACHE_DIT_CUDA_BUILD_MODE=FAST` on
the target machine) to cover the card you will actually use.

The architecture set defaults to `89` (Ada) and `120a` (Blackwell consumer) only -- see
`DEFAULT_CUDA_ARCH_TARGETS` in `setup.py`; every extra target multiplies the compile time of
the single extension. `CACHE_DIT_CUDA_ARCH_LIST` can add more (e.g. `100` for Blackwell
datacenter), while Hopper (`sm90`) is always dropped by the SM predicate.
`CACHE_DIT_CUDA_ARCH_LIST` overrides it -- note that an inherited `TORCH_CUDA_ARCH_LIST` is
what mostly decides whether the extension survives, because it is only consulted when
`CACHE_DIT_CUDA_ARCH_LIST` is unset. `CACHE_DIT_CUDA_BUILD_MODE=FAST` compiles for the
local GPU only, which is the quickest way to build on a single machine.

Run this build alone. The wheel is assembled *after* the extensions finish compiling, and
that step imports `setuptools-scm` / `vcs-versioning`; installing packages in the same
environment while the build runs can replace them under the running process, which then
fails at the very end with an `ImportError` raised from `vcs_versioning`. Compile time is
also why `--no-build-isolation` matters here: the isolated build environment would compile
everything from scratch.

> [!WARNING]
> `cache-dit-cu13` (the former distribution name on PyPI, last release `1.5.2`) is
> retired: the CUDA build is now `cache-dit` itself, tagged with the local version
> `+cu13`. Both names ship the same `cache_dit/` package, so uninstall the old one
> before switching:
>
> ```bash
> pip uninstall cache-dit-cu13
> pip install -U cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/
> ```

> [!NOTE]
> Those wheels carry the local version `+cu13` (`cache-dit==1.5.5+cu13`), which is what
> makes them a different artefact from the PyPI release of the same version: **on the
> same version** a local version sorts above the plain one, so it wins, and a
> pure-Python wheel of that version which is already installed no longer satisfies the
> requirement as-is. Pin it with `cache-dit==<version>+cu13` whenever the channel is
> what you want: if the CUDA channel has not caught up with a newer release on PyPI,
> an unpinned `-U` resolves to the newer plain release, without an error.
>
> Our index carries `cache-dit` alone, with no dependencies, so it is added as an
> **extra** index and PyPI stays the index the requirements come from: `pip` merges the
> two and takes the highest version, so this one command resolves the CUDA build. Install
> torch first, from the PyTorch index you normally use, or the resolver may pick a torch
> that is not the CUDA 13 build.

## Optional Sol-Attn backend

The `sol_attn` attention backend uses NVIDIA's upstream Sol-Attn package and requires BF16,
`torch>=2.10`, CUDA>=12.8 and Triton>=3.6. Its optimized CuTe path additionally requires
`nvidia-cutlass-dsl>=4.5` and `cuda-python`. These requirements are declared by the upstream README,
not enforced by its package metadata, so check them in your environment before running inference.

Install the pinned upstream source used for the current validation:

```bash
pip install "git+https://github.com/NVlabs/Sana.git@cf67a3acad7511acb85053e56a05e5dd24c82ba0#subdirectory=techniques/sparse_backends"
```

No Cache-DiT native extension is required for this backend. The default attention backend remains
`native`; installing Sol-Attn does not change existing inference behavior.

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.
