# Installation  

## Prequisites

- Python >= 3.10 (<span style="color:hotpink;">3.12</span>, recommended)
- PyTorch >= 2.7.1 (<span style="color:hotpink;">2.10.0</span>, recommended)
- CUDA >= 12.6 (>= <span style="color:hotpink;">12.9</span>, recommended) for Nvidia GPU
- Diffusers >= 0.36.0 (>= <span style="color:hotpink;">0.37.0</span>, recommended)
- TorchAo >= 0.15.0 (>= <span style="color:hotpink;">0.16.0</span>, recommended)

## Installation with Nvidia GPU

<div id="installation"></div>

You can install the stable release of <span style="color:hotpink;">cache-dit</span> from PyPI:

```bash
pip3 install -U cache-dit # or, pip3 install -U "cache-dit[all]" for all features
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```
Please also install the <span style="color:hotpink;">latest</span> main branch of <span style="color:hotpink;">diffusers</span> for context parallelism:  
```bash
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.36.0
```

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.

## Installation with AMD GPU

Please refer to [AMD GPU Support](./AMD_GPU.md) documentation for more details.
