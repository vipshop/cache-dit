# Installation  

## Prequisites

- Python >= 3.10 (<span style="color:hotpink;">3.12</span>, recommended)
- PyTorch >= 2.7.1 (<span style="color:hotpink;">2.11.0</span>, recommended)
- CUDA >= 12.6 (>= <span style="color:hotpink;">13.0</span>, recommended) for Nvidia GPU
- Diffusers >= 0.36.0 (>= <span style="color:hotpink;">0.37.0</span>, recommended)
- TorchAo >= 0.15.0 (>= <span style="color:hotpink;">0.16.0</span>, recommended)

## Installation with Nvidia GPU

<div id="installation"></div>

Firstly, install the required dependencies, including PyTorch, Diffusers, and TorchAo. We recommend installing the <span style="color:hotpink;">latest</span> versions for better compatibility and performance.

```bash
# recommend: latest pytorch for better compile compatiblity.
pip3 install torch==2.11.0 torchvision torchaudio triton --upgrade 
# recommend: install torchao nightly due to: https://github.com/pytorch/ao/issues/3670
pip3 install --pre torchao --index-url https://download.pytorch.org/whl/cu130
pip3 install transformers accelerate opencv-python-headless einops imageio-ffmpeg ftfy # optional
```

Then, you can install the stable release of <span style="color:hotpink;">cache-dit</span> from PyPI:

```bash
pip3 install -U cache-dit # or, pip3 install -U "cache-dit[all]" for all features
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```
Please also install the <span style="color:hotpink;">latest</span> main branch of <span style="color:hotpink;">diffusers</span> for context parallelism:  
```bash
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.37.0
```

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.
