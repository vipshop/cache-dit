"""Loaders that convert a checkpoint while `from_pretrained` loads it."""

from .base import BaseLoader
from .ideogram4 import Ideogram4Loader

__all__ = [
  "BaseLoader",
  "Ideogram4Loader",
]
