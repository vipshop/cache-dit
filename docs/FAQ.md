# Frequently Asked Questions (FAQ)

## Installation & Dependencies

### How to install Flash Attention 3 (FA3)?

<span style="color:hotpink;">Flash Attention 3</span> provides optimized attention kernels for better performance. To install:

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

After installation, you need to modify the attention dispatch file:

```bash
vi /usr/local/lib/python3.12/dist-packages/diffusers/models/attention_dispatch.py
```

Find `_diffusers_flash_attn_3::_flash_attn_forward` and add <span style="color:hotpink;">return_attn_probs=True</span>:

```python
return_attn_probs=True
```

### How to install Sage Attention?

<span style="color:hotpink;">Sage Attention</span> is an efficient attention implementation. To install:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32  # Optional
export TORCH_CUDA_ARCH_LIST=9.0 # 9.0 for Hopper, 8.9 for Ada
python setup.py install
```

## Common Issues

### torch.compile errors when running examples

If you encounter errors with `torch.compile` when running cache-dit examples, try the following solutions:

- <span style="color:hotpink;">Clear</span> the torch inductor <span style="color:hotpink;">cache:</span>

```bash
rm -rf /tmp/torchinductor_root/
```
   Then retry running your example.

- <span style="color:hotpink;">Upgrade</span> PyTorch to the latest version:

```bash
pip install --upgrade torch torchvision
```

If the issue persists, please [open an issue](https://github.com/vipshop/cache-dit/issues) with:    
   - Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - The complete error traceback
   - Your system configuration (GPU model, CUDA version, etc.)


### NCCL errors during distributed inference

Please consider to upgrade to the latest PyTorch and NCCL versions, as they may contain important bug fixes and performance improvements. You can upgrade PyTorch and NCCL using the following commands:

```bash
pip install --upgrade torch torchvision torchaudio triton
pip install --upgrade nvidia-nccl-cu12 # or, for CUDA 13: pip install --upgrade nvidia-nccl-cu13
```

## Performance Optimization

### Which attention backend should I use?

Cache-DiT supports multiple attention backends for different use cases. For a complete overview of attention backends in diffusers, see the [Attention Backends](./user_guide/ATTENTION.md). The main attention backends supported by cache-dit are:  

- **<span style="color:hotpink;">flash</span>**: Flash Attention 2 - Good performance on Ampere/Ada GPUs
- **<span style="color:hotpink;">_flash_3</span>**: Flash Attention 3 - Best for Hopper architecture GPUs (H100, H200)
- **<span style="color:hotpink;">native</span>**: Native PyTorch SDPA - Default, works on all devices
- **<span style="color:hotpink;">_native_cudnn</span>**: cuDNN-based native attention
- **<span style="color:hotpink;">_sdpa_cudnn</span>**: SDPA with cuDNN (cache-dit specific, supports context parallelism with attention masks)
- **<span style="color:hotpink;">sage</span>**: Sage Attention - Good balance between performance and compatibility

**Recommendation:**  

- **H100/H200**: Use <span style="color:hotpink;">_flash_3</span> for best performance
- **A100/A6000**: Use <span style="color:hotpink;">flash</span> or <span style="color:hotpink;">sage</span>
- **Other GPUs**: Use <span style="color:hotpink;">native</span> or <span style="color:hotpink;">sage</span>


## Other Questions

For other questions or issues not covered here, please:  

1. Check the [documentation](https://cache-dit.readthedocs.io/en/latest/)
2. Search [existing issues](https://github.com/vipshop/cache-dit/issues)
3. [Open a new issue](https://github.com/vipshop/cache-dit/issues/new) if needed
