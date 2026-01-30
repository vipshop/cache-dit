# Installation  

## Prequisites

- Python >= 3.10 (3.12, recommended)
- PyTorch >= 2.7.1 (2.10.0, recommended)
- CUDA >= 12.6 (>= 12.9, recommended) for Nvidia GPU
- Diffusers >= 0.36.0 or latest main branch from GitHub

## Installation with Nvidia GPU

<div id="installation"></div>

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit # or, pip3 install -U "cache-dit[all]" for all features
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```
Please also install the latest main branch of diffusers for context parallelism:  
```bash
pip3 install git+https://github.com/huggingface/diffusers.git # or >= 0.36.0
```

## Installation with Ascend NPU

Please refer to [Ascend NPU Support](./ASCEND_NPU.md) documentation for more details.
