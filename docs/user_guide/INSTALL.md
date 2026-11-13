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
git submodule update --init --recursive --force # SVDQuant needs spdlog; SM120 CuTe needs CUTLASS 4.6.1
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
> Those wheels carry the local version `+cu13` (`cache-dit==1.5.3+cu13`), which is what
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

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.

## Building and publishing the wheels (maintainers)

`tools/build_releases.sh` builds both wheels of a release -- the pure-Python one PyPI
serves, and the CUDA ones the index above serves -- in a workspace prepared by
`tools/release_workspace.py`, so nothing lands in the development tree. The platform
tag of every CUDA wheel is derived from the GLIBC symbols the extension actually needs
(`tools/repair_wheel_platform_tag.py`), and the index is generated by
`tools/make_index.py`.

| Profile | Compiles | Version | Use |
|---|---|---|---|
| `mini` (default) | one conda env, one arch (`120a`) | derived from git, always carries `.dev` | exercises the whole build -> wheel -> index -> install path in minutes; its wheels are never publishable |
| `full` | `py310`..`py314`, the setup.py default arch set | `CACHE_DIT_RELEASE_VERSION=<x.y.z>`, and the sources have to be committed | the published set |

```bash
# both wheels, in .tmp/release_build/{base,cu13}, landing in dist/.  Run this from a
# committed tree: profile full refuses uncommitted sources, and the release tag is
# created at the commit the publish runs in -- push it first.
CACHE_DIT_RELEASE_PROFILE=full CACHE_DIT_RELEASE_VERSION=1.5.3 bash tools/build_releases.sh
# the CUDA channel: uploads the wheels as release assets of v1.5.3+cu13, regenerates
# whl/cu13/ in the site work tree and pushes it.  The work tree is created once; every
# later publish starts by fast-forwarding it.
git fetch origin gh-pages && git worktree add .tmp/gh-pages gh-pages   # once
git -C .tmp/gh-pages pull --rebase --ff-only                           # every publish
bash tools/publish_index.sh --tag v1.5.3+cu13 \
  --wheels "dist/cache_dit-1.5.3+cu13-"*.whl --site-repo .tmp/gh-pages
# the PyPI release: the same version without the local segment
python -m twine upload dist/cache_dit-1.5.3-py3-none-any.whl
```

The channel is what separates one CUDA toolkit's build from another, and it appears
twice: in the index directory (`whl/cu13/`) and as the local version of the wheel
(`1.5.3+cu13`, from `CACHE_DIT_RELEASE_CHANNEL`). `publish_index.sh` refuses a
pre-release version, a wheel without the channel's local version, a wheel without a
compiled extension and a wheel whose platform tag is not manylinux; it writes each
asset's sha256 into the index, which pip enforces, and it merges into what is already
published so older versions keep resolving. Wheels are not byte-reproducible, so
replacing an asset that is already published needs `CACHE_DIT_FORCE_UPLOAD=1` --
bumping the version is usually the better answer.

The platform tag of each CUDA wheel follows from the GLIBC symbols that wheel needs, so
the floor moves with the toolchain: the wheels of the current release are
`manylinux_2_38_x86_64`, and the index page lists the tags it actually serves. Read the
floor from there rather than from this page.

Verify an install from the tree, and then from the published index:

```bash
python3 -m http.server -d .tmp/gh-pages 8765 &
pip download "cache-dit==1.5.3+cu13" --no-deps -d /tmp/dl --index-url http://127.0.0.1:8765/whl/cu13/
pip download "cache-dit==1.5.3+cu13" --no-deps -d /tmp/dl \
  --index-url https://vipshop.github.io/cache-dit/whl/cu13/
```

A mirror under a directory of another Pages site is generated from the same wheels,
with the assets still pointing at this repository's releases. `--index-only` skips the
upload but still commits and pushes the site it writes into:

```bash
CACHE_DIT_SITE_URL=https://deftruth.github.io CACHE_DIT_INDEX_PREFIX=cache-dit \
  bash tools/publish_index.sh --index-only --tag v1.5.3+cu13 \
  --wheels "dist/cache_dit-1.5.3+cu13-"*.whl \
  --release-url https://github.com/vipshop/cache-dit/releases/download/v1.5.3+cu13 \
  --site-repo <that checkout>
```
