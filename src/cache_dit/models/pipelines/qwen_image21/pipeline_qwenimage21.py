"""Qwen-Image 2.1 pipeline that opts into cache-dit's fused transformer.

The pipeline class exists so the choice is explicit: `QwenImage21Pipeline` keeps the
stock behaviour, `_QwenImage21Pipeline` installs the fused operators. Everything else —
`__call__`, components, quantization, offload, CPU placement — is inherited unchanged,
because no new component or `__init__` argument is introduced.
"""

from diffusers import QwenImage21Pipeline

from ....logger import init_logger
from .transformer_qwen_image21 import upgrade_transformer

logger = init_logger(__name__)


class _QwenImage21Pipeline(QwenImage21Pipeline):
  """`QwenImage21Pipeline` whose transformer runs cache-dit's fused operators."""

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load the pipeline, then upgrade the transformer in place.

    The upgrade happens after `from_pretrained`, so `torch_dtype`, `device_map`,
    `quantization_config`, `variant`, accelerate hooks and the `model_index.json`
    resolution all run on diffusers' own code path; only the transformer's Python class
    changes afterwards.

    :param pretrained_model_name_or_path: Model id or local directory.
    :param kwargs: Forwarded unchanged to `QwenImage21Pipeline.from_pretrained`.
    :returns: The pipeline with a fused transformer.
    """

    pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    upgrade_transformer(getattr(pipe, "transformer", None))
    return pipe


__all__ = ["_QwenImage21Pipeline"]
