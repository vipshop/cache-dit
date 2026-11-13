import torch
from typing import Any, Optional, Set, Tuple, List, Dict, Callable, Union

from diffusers import DiffusionPipeline
from .block_adapters import (
  BlockAdapter,
  FakeDiffusionPipeline,
)
from ...utils import is_pipeline
from ...utils import is_modular_pipeline

from ...logger import init_logger

logger = init_logger(__name__)


class BlockAdapterRegister:
  """Registry for predefined `BlockAdapter` builders.

  The registry maps pipeline or transformer class-name prefixes to functions that construct
  compatible `BlockAdapter` instances, enabling `enable_cache` to auto-detect supported models.

  Two kinds of registrations share the prefix table:

  * **family** adapters (`owned=False`) only encode the block I/O contract, so any transformer
    exposing the documented stack can be bound to them: diffusers' own class, a downstream fork,
    or cache-dit's vendored copy.
  * **owned** adapters (`owned=True`) additionally rely on cache-dit's own implementation detail:
    a class identity assertion, a layer-KV contract such as `force_full_compute`, or a marker only
    the vendored transformer sets. Binding one to a same-named class from another package would
    silently cache the wrong architecture, so `get_adapter`/`is_supported` skip them for foreign
    objects.
  """

  _adapters: Dict[str, Callable[..., BlockAdapter]] = {}
  # Prefixes whose adapter is only valid for cache-dit's own pipeline/transformer implementation.
  _owned_adapters: Set[str] = set()
  # De-duplicates the foreign-object warning; a single enable can probe the registry several times.
  _warned_owned_mismatch: Set[str] = set()
  # Modular pipeline families whose adapter has been verified end to end (see the
  # cache-dit-model-integration skill, `references/modular_pipeline.md`).
  _verified_modular_pipelines: Tuple[str,
                                     ...] = ("Flux3", "QwenImage21", "MiniMaxH3", "MiniMaxH3Image",
                                             "FastH3", "PikaH3Flex", "Anima", "Echo", "WanAnimate2")
  # QwenImage / QwenImage21 are deliberately absent: their adapters decide per checkpoint
  # (Qwen-Image 2.1 samples without guidance by default, so a fixed True would halve the step count).
  # Wan is absent for the same reason: `WanAnimate2` shares the "Wan" prefix but its distilled
  # checkpoint runs a single guidance-free forward per step, and this list is a prefix fallback that
  # would overrule the adapter's own (and correct) `False`.
  _predefined_adapters_has_separate_cfg: List[str] = [
    "CogView4",
    "Cosmos",
    "SkyReelsV2",
    "Chroma",
    "Lumina2",
    "Kandinsky5",
    "ChronoEdit",
    "HunyuanVideo15",
    "OvisImage",
    "BooguImage",
    "JoyImage",
  ]

  @classmethod
  def register(cls, name: str, *, supported: bool = True, owned: bool = False):
    """Register an adapter factory under a class-name prefix.

    :param name: Class-name prefix associated with the adapter factory.
    :param supported: Whether the registered prefix should appear in the supported-model registry.
    :param owned: Whether the adapter is only valid for cache-dit's own pipeline/transformer. Owned
        adapters are skipped for foreign objects whose class name happens to match the prefix.
    """

    if owned and not supported:
      raise ValueError("`owned=True` requires `supported=True`.")

    def decorator(func: Callable[..., BlockAdapter]) -> Callable[..., BlockAdapter]:
      if supported:
        cls._adapters[name] = func
        if owned:
          cls._owned_adapters.add(name)
      return func

    return decorator

  @classmethod
  def _is_cache_dit_owned(cls, pipe_or_module: Any) -> bool:
    """Return whether the object is (a subclass of) cache-dit's own implementation.

    :param pipe_or_module: Pipeline instance or transformer module to inspect.
    :returns: `True` when any class in the MRO lives inside the `cache_dit` package.
    """

    return any(base.__module__ in ("cache_dit", ) or base.__module__.startswith("cache_dit.")
               for base in type(pipe_or_module).__mro__)

  @classmethod
  def _owned_prefix_for(cls, pipe_or_module: Any) -> Optional[str]:
    """Return the owned prefix that matched a foreign object, or `None`.

    :param pipe_or_module: Pipeline instance or transformer module to inspect.
    :returns: The owned prefix whose adapter is skipped for this object, or `None`.
    """

    if isinstance(pipe_or_module, str):
      return None

    cls_name = pipe_or_module.__class__.__name__
    for name in cls._adapters:
      if cls_name.startswith(name):
        if name in cls._owned_adapters and not cls._is_cache_dit_owned(pipe_or_module):
          return name
        return None
    return None

  @classmethod
  def _log_owned_mismatch_once(cls, cls_name: str, prefix: str) -> None:
    if cls_name in cls._warned_owned_mismatch:
      return
    cls._warned_owned_mismatch.add(cls_name)
    logger.warning(
      f"{cls_name} reuses cache-dit's '{prefix}' adapter name but is not cache-dit's own "
      "implementation; skipping that adapter because it relies on the vendored transformer's "
      "internals (class identity, layer-KV handling such as `force_full_compute`, or a marker "
      "only the vendored model sets). Pass an explicit `BlockAdapter(pipe=..., transformer=..., "
      "blocks=...)` to cache this model, or disable the cache.")

  @classmethod
  def _match_adapter(
    cls,
    pipe_or_module: Any,
    *,
    log_owned_mismatch: bool = False,
  ) -> Tuple[Optional[str], Optional[Callable[..., BlockAdapter]]]:
    """Return the matching `(prefix, factory)` pair, or `(None, None)`.

    :param pipe_or_module: Pipeline instance, transformer module, or class name to resolve.
    :param log_owned_mismatch: Whether to log when an owned prefix is skipped for a foreign object.
    :returns: The matched prefix together with its factory, or `(None, None)`.
    """

    if isinstance(pipe_or_module, str):
      cls_name: str = pipe_or_module
    else:
      cls_name = pipe_or_module.__class__.__name__

    for name, factory in cls._adapters.items():
      if not cls_name.startswith(name):
        continue
      if name in cls._owned_adapters and not cls._is_cache_dit_owned(pipe_or_module):
        if log_owned_mismatch:
          cls._log_owned_mismatch_once(cls_name, name)
        # Stop instead of falling through to a shorter family prefix: a foreign `WanS2V*` class
        # would otherwise be bound to the generic `Wan` adapter and cached as a plain Wan model.
        return None, None
      return name, factory

    return None, None

  @classmethod
  def get_adapter(
    cls,
    pipe_or_module: DiffusionPipeline | torch.nn.Module | str | Any,
    **kwargs,
  ) -> BlockAdapter | None:
    """Return a predefined adapter for a pipeline, transformer, or class name.

    :param pipe_or_module: Pipeline instance, transformer module, or class name to resolve.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: The matched adapter, or `None` when no predefined adapter applies.
    """

    name, factory = cls._match_adapter(pipe_or_module, log_owned_mismatch=True)
    if factory is None:
      return None

    if not is_pipeline(pipe_or_module):
      assert isinstance(pipe_or_module, torch.nn.Module)
      # NOTE: Make pre-registered adapters support Transformer-only case.
      # WARN: This branch is not officially supported and only for testing
      # purpose. We construct a fake diffusion pipeline that contains the
      # given transformer module. Currently, only works for DiT models which
      # only have one transformer module. Case like multiple transformers
      # is not supported, e.g, Wan2.2. Please use BlockAdapter directly for
      # such cases.
      return factory(FakeDiffusionPipeline(pipe_or_module), **kwargs)

    cls_name = pipe_or_module.__class__.__name__
    if is_modular_pipeline(pipe_or_module) and not cls_name.startswith(
        cls._verified_modular_pipelines):
      # Modular pipelines reuse the adapter of the matching transformer family. Only a few
      # families are verified so far, so flag the others instead of silently claiming
      # support.
      logger.warning(f"Modular pipeline {cls_name} is dispatched to the '{name}' "
                     f"BlockAdapter. Only {cls._verified_modular_pipelines} modular pipelines "
                     "are verified, please check the output (PSNR and SSIM) before relying "
                     "on caching.")
    return factory(pipe_or_module, **kwargs)

  @classmethod
  def has_separate_cfg(
    cls,
    pipe_or_adapter_or_module: Union[
      DiffusionPipeline,
      FakeDiffusionPipeline,
      BlockAdapter,
      torch.nn.Module,  # e.g., transformer-only case
    ],
  ) -> bool:
    """Infer whether a model executes CFG and non-CFG in separate forwards.

    :param pipe_or_adapter_or_module: Pipeline, adapter, or transformer to inspect.
    :returns: `True` when the model uses separate CFG and non-CFG forward passes.
    """

    # 0. Prefer custom setting from block adapter.
    if isinstance(pipe_or_adapter_or_module, BlockAdapter):
      return pipe_or_adapter_or_module.has_separate_cfg

    has_separate_cfg = False
    if isinstance(pipe_or_adapter_or_module, FakeDiffusionPipeline):
      return False

    if isinstance(pipe_or_adapter_or_module,
                  torch.nn.Module) or is_pipeline(pipe_or_adapter_or_module):
      adapter = cls.get_adapter(
        pipe_or_adapter_or_module,
        skip_post_init=True,  # check cfg setting only
      )
      if adapter is not None:
        has_separate_cfg = adapter.has_separate_cfg

    if has_separate_cfg:
      return True

    # A class that only reuses an owned family name was refused above; the CFG prefix fallback must
    # not answer for it, otherwise a single-forward vendored model would be counted as two forwards.
    if cls._owned_prefix_for(pipe_or_adapter_or_module) is not None:
      return False

    pipe_cls_name = pipe_or_adapter_or_module.__class__.__name__
    for name in cls._predefined_adapters_has_separate_cfg:
      if pipe_cls_name.startswith(name):
        return True

    return False

  @classmethod
  def is_supported(cls, pipe_or_module) -> bool:
    """Return whether a pipeline or transformer has a registered adapter.

    An owned registration only counts for cache-dit's own implementation, so a foreign class that
    reuses the same family name is reported as unsupported (and `get_adapter` returns `None`).

    :param pipe_or_module: Pipeline or transformer to inspect.
    :returns: `True` when a predefined adapter is usable for the object type.
    """

    _, factory = cls._match_adapter(pipe_or_module, log_owned_mismatch=True)
    return factory is not None

  @classmethod
  def supported_pipelines(cls, **kwargs) -> Tuple[int, List[str]]:
    """Return the number and names of registered predefined pipelines.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A tuple `(count, names)` listing the registered predefined adapter prefixes.
    """

    val_pipelines = cls._adapters.keys()
    return len(val_pipelines), [p for p in val_pipelines]
