# Frequently Asked Questions (FAQ)

## Installation & Dependencies

### How to install Flash Attention 3 (FA3)?

Flash Attention 3 provides optimized attention kernels for better performance. To install:

```bash
git clone git@github.com:Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

After installation, you need to modify the attention dispatch file:

```bash
vi /usr/local/lib/python3.12/dist-packages/diffusers/models/attention_dispatch.py
```

Find `_diffusers_flash_attn_3::_flash_attn_forward` and add `return_attn_probs=True`:

```python
return_attn_probs=True
```

### How to install Sage Attention?

Sage Attention is an efficient attention implementation. To install:

```bash
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32  # Optional
export TORCH_CUDA_ARCH_LIST=9.0
python setup.py install
```

## Common Issues

### torch.compile errors when running examples

If you encounter errors with `torch.compile` when running cache-dit examples, try the following solutions:

1. **Clear the torch inductor cache:**
   ```bash
   rm -rf /tmp/torchinductor_root/
   ```
   Then retry running your example.

2. **Upgrade PyTorch to the latest version:**
   ```bash
   pip install --upgrade torch torchvision
   ```

3. **If the issue persists:**  
   Please [open an issue](https://github.com/vipshop/cache-dit/issues) with:  
   - Your PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - The complete error traceback
   - Your system configuration (GPU model, CUDA version, etc.)

## Performance Optimization

### Which attention backend should I use?

Cache-DiT supports multiple attention backends for different use cases. For a complete overview of attention backends in diffusers, see the [Attention Backends](./user_guide/ATTENTION.md). The main attention backends supported by cache-dit are:  

- **`flash`**: Flash Attention 2 - Good performance on Ampere/Ada GPUs
- **`_flash_3`**: Flash Attention 3 - Best for Hopper architecture GPUs (H100, H200)
- **`native`**: Native PyTorch SDPA - Default, works on all devices
- **`_native_cudnn`**: cuDNN-based native attention
- **`_sdpa_cudnn`**: SDPA with cuDNN (cache-dit specific, supports context parallelism with attention masks)
- **`sage`**: Sage Attention - Good balance between performance and compatibility

**Recommendation:**  

- **H100/H200**: Use `_flash_3` for best performance
- **A100/A6000**: Use `flash` or `sage`
- **Other GPUs**: Use `native` or `sage`


## Other Questions

For other questions or issues not covered here, please:  

1. Check the [documentation](https://cache-dit.readthedocs.io/en/latest/)
2. Search [existing issues](https://github.com/vipshop/cache-dit/issues)
3. [Open a new issue](https://github.com/vipshop/cache-dit/issues/new) if needed
