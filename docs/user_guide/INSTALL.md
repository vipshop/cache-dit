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
# recommend: install latest pytorch for better compile compatiblity.
uv pip install torch==2.11.0 torchvision torchaudio triton --upgrade
# recommend: install latest torchao nightly due to issue: https://github.com/pytorch/ao/issues/3670
uv pip install --pre torchao --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers accelerate opencv-python-headless einops imageio-ffmpeg ftfy 
```

Then, you can install the stable release of <span style="color:#c77dff;">cache-dit</span> from PyPI:

```bash
uv pip install -U cache-dit # or, pip3 install -U "cache-dit[all]" for all features
```
Or you can install the latest develop version from GitHub:

```bash
uv pip install git+https://github.com/vipshop/cache-dit.git
```
Please also install the <span style="color:#c77dff;">latest</span> main branch of <span style="color:#c77dff;">diffusers</span> for context parallelism:  
```bash
uv pip install git+https://github.com/huggingface/diffusers.git # or >= 0.37.0
```

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.
