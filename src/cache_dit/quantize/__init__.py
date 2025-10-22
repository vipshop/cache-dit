try:
    import torchao
except ImportError:
    raise ImportError(
        "Quantization functionality requires the 'quantization' extra dependencies. "
        "Install with: pip install cache-dit[quantization]"
    )
from cache_dit.quantize.quantize_interface import quantize
