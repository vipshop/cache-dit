# Installation  

## Prequisites

- Python >= 3.10 (<span style="color:#c77dff;">3.12</span>, recommended)
- PyTorch >= 2.7.1 (<span style="color:#c77dff;">2.11.0</span>, recommended)
- CUDA >= 12.6 (>= <span style="color:#c77dff;">13.0</span>, recommended) for Nvidia GPU
- Diffusers >= 0.36.0 (>= <span style="color:#c77dff;">0.37.0</span>, recommended)
- TorchAo >= 0.15.0 (>= <span style="color:#c77dff;">0.17.0</span>, recommended)

## Installation with Nvidia GPU

<div id="installation"></div>

Firstly, install the required dependencies, including PyTorch, Diffusers, and TorchAo. We recommend installing the <span style="color:#c77dff;">latest</span> versions for better compatibility and performance.

```bash
pip install -U uv # use uv for faster installation
uv pip install torch==2.11.0 torchvision torchaudio triton \
  transformers diffusers accelerate torchao opencv-python-headless \
  einops imageio-ffmpeg ftfy numpy
```

Then, you can install <span style="color:#c77dff;">Cache-DiT</span> from PyPI:

```bash
uv pip install -U cache-dit # PyPI, stable release.
uv pip install git+https://github.com/vipshop/cache-dit.git # latest
```

Or, install Cache-DiT with <span style="color:#c77dff;">SVDQuant</span> support (Experimental):

```bash
# Required: CUDA 13.0+, PyTorch 2.11+, Ubuntu 22.04+ (GLIBC 2.32+).
uv pip install -U cache-dit-cu13 # PyPI, stable release with SVDQ.
# Optional: just build Cache-DiT with SVDQuant support from source.
git clone https://github.com/vipshop/cache-dit.git && cd cache-dit
git submodule update --init --recursive --force # init submodules 
CACHE_DIT_BUILD_SVDQUANT=1 MAX_JOBS=32 uv pip install -e ".[quantization]"
```

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.
