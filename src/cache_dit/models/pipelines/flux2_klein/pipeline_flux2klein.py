"""FLUX.2-Klein pipeline that opts into cache-dit's fused transformer.

The pipeline class exists so the choice is explicit: `Flux2KleinPipeline` keeps the stock
behaviour, `_Flux2KleinPipeline` installs the fused operators. Everything else - `__call__`,
components, quantization, offload, CPU placement - is inherited unchanged, because no new
component and no new `__init__` argument is introduced.
"""

from diffusers import Flux2KleinPipeline

from ....logger import init_logger
from .transformer_flux2_klein import upgrade_transformer

logger = init_logger(__name__)


class _Flux2KleinPipeline(Flux2KleinPipeline):
  """`Flux2KleinPipeline` whose transformer runs cache-dit's fused operators."""

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load the pipeline, then upgrade the transformer in place.

    The upgrade happens after `from_pretrained`, so `torch_dtype`, `device_map`,
    `quantization_config`, `variant`, accelerate hooks and the `model_index.json` resolution all
    run on diffusers' own code path; only the transformer's Python class changes afterwards. It
    also happens before `enable_cache`, `enable_parallelism` and `maybe_compile_transformer`,
    which is the only ordering that survives them: those steps rebind or compile the forwards,
    and a later install would either be overwritten or force a recompilation.

    :param pretrained_model_name_or_path: Model id or local directory.
    :param kwargs: Forwarded unchanged to `Flux2KleinPipeline.from_pretrained`.
    :returns: The pipeline with a fused transformer.
    """

    pipe = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    upgrade_transformer(getattr(pipe, "transformer", None))
    return pipe


__all__ = ["_Flux2KleinPipeline"]
