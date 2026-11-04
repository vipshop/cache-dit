"""Vendored MAGI-2 Preview (t2va / i2va) audio-video pipeline.

The released ``sand-ai/MAGI-2-preview`` checkpoint ships its own pipeline, transformer and decoders,
none of which diffusers can import, so cache-dit owns the code. ``__all__`` stays empty: the modules
are imported lazily by the CLI example and the block adapter, so importing ``cache_dit`` never pulls
the 114B model code.
"""

__all__ = []
