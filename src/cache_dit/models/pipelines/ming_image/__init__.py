"""Built-in Ming-Image 0.1 Design and Design-Layer implementation."""

__all__ = ["MingImagePipeline", "MingImageTransformer2DModel"]


def __getattr__(name):
  if name == "MingImagePipeline":
    from .pipeline import MingImagePipeline

    return MingImagePipeline
  if name == "MingImageTransformer2DModel":
    from .transformer import MingImageTransformer2DModel

    return MingImageTransformer2DModel
  raise AttributeError(name)
