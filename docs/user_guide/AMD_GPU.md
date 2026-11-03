# AMD GPU Support 

## Install AMD GPU version Torch 
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 
pip install --pre torchao --index-url https://download.pytorch.org/whl/rocm7.1
```

Then, you can install cache-dit and other libraries as usual, please refer to the [Installation section](./INSTALL.md) and [Examples](../EXAMPLES.md) in the user guide for more details.
