import gc
import time
import torch
import builtins as __builtin__
import contextlib
import functools
from typing import Tuple, List, Union, Optional, Any
from diffusers import DiffusionPipeline

from .platforms import current_platform
from .logger import init_logger

logger = init_logger(__name__)


# Utils for checking module types and states, e.g., whether a module
# is text encoder, controlnet, auto encoder, or whether it's already
# parallelized, quantized or cached. These utils are used in the
# parallelism and quantization dispatching to determine which modules
# to apply parallelism or quantization to, and to avoid applying parallelism
# or quantization to modules that are already parallelized or quantized.
@functools.lru_cache(maxsize=1)
def _modular_pipeline_cls() -> Optional[type]:
  """Resolve the diffusers `ModularPipeline` class.

  :returns: The class, or None when the installed diffusers has no modular pipelines.
  """
  try:
    from diffusers import ModularPipeline
  except ImportError:
    return None
  return ModularPipeline


def is_modular_pipeline(pipe_or_module: Any) -> bool:
  """Check whether the given object is a diffusers `ModularPipeline`.

  Modular pipelines (e.g. FLUX.3) are plain Python objects, not `torch.nn.Module` nor
  `DiffusionPipeline`, but they own the same components (`transformer`, `text_encoder`, `vae`)
  and are callable like a pipeline. All pipeline-level dispatch in cache-dit must treat them
  the same way as `DiffusionPipeline`.

  :param pipe_or_module: Object to inspect.
  :returns: Whether the object is a modular pipeline.
  """
  modular_pipeline_cls = _modular_pipeline_cls()
  return modular_pipeline_cls is not None and isinstance(pipe_or_module, modular_pipeline_cls)


def is_pipeline(pipe_or_module: Any) -> bool:
  """Check whether the given object is a pipeline, standard or modular.

  :param pipe_or_module: Object to inspect.
  :returns: Whether the object is a `DiffusionPipeline` or a `ModularPipeline`.
  """
  return isinstance(pipe_or_module, DiffusionPipeline) or is_modular_pipeline(pipe_or_module)


def resolve_pipe(pipe_or_adapter: Any) -> Any:
  """Resolve the pipeline object from a pipeline, a modular pipeline or a `BlockAdapter`.

  :param pipe_or_adapter: Pipeline, modular pipeline or `BlockAdapter`.
  :returns: The pipeline object, or None if it can not be resolved.
  """
  if is_pipeline(pipe_or_adapter):
    return pipe_or_adapter
  return getattr(pipe_or_adapter, "pipe", None)


# Attribute names a pipeline can use for its denoising transformer. MiniMax-H3 keeps two
# checkpoint partitions and loads exactly one of them (`transformer/` for `t2va` / `fl2va`,
# `transformer_ref/` for `ref2va`), so the second name is a substitute, never an extra component.
TRANSFORMER_ATTR_NAMES = ("transformer", "transformer_ref")
SECONDARY_TRANSFORMER_ATTR_NAMES = ("transformer_2", "unconditional_transformer")


def iter_transformer_modules(pipe_or_adapter: Any) -> List[Tuple[Optional[str], Any]]:
  """Iterate the primary and secondary denoising transformers owned by a pipeline.

  :param pipe_or_adapter: Pipeline, adapter or bare transformer module.
  :returns: ``(attribute_name, module)`` pairs in denoiser execution order.
  """
  pipe = resolve_pipe(pipe_or_adapter)
  if pipe is None:
    pipe = pipe_or_adapter
  name, transformer = resolve_transformer(pipe)
  modules = [(name, transformer)] if transformer is not None else []
  seen = {id(transformer)} if transformer is not None else set()
  for secondary_name in SECONDARY_TRANSFORMER_ATTR_NAMES:
    module = getattr(pipe, secondary_name, None)
    if isinstance(module, torch.nn.Module) and id(module) not in seen:
      modules.append((secondary_name, module))
      seen.add(id(module))
  return modules


# Denoising transformers that own a repeated block stack but never declare it. Regional
# quantization and `compile_repeated_blocks()` discover the stack from `_repeated_blocks`, so
# without the declaration they fall back to whole-module work: less accurate (the modulation and
# pixel-output projections get quantized too) and slower (a full-graph compile of a 24-block
# model with data-dependent reshapes loses to eager).
_MISSING_REPEATED_BLOCKS = {
  "PRXTransformer2DModel": ["PRXBlock"],
  # Cosmos Predict2 (incl. the Anima t2i modular pipeline) ships the same uniform
  # `transformer_blocks` stack without declaring it.
  "CosmosTransformer3DModel": ["CosmosTransformerBlock"],
}


def declare_repeated_blocks(transformer: Optional[torch.nn.Module]) -> None:
  """Declare the repeated block class for models diffusers never annotated.

  :param transformer: Denoising transformer; `None` is a no-op.
  """
  if transformer is None or getattr(transformer, "_repeated_blocks", None):
    return
  repeated_blocks = _MISSING_REPEATED_BLOCKS.get(transformer.__class__.__name__)
  if repeated_blocks:
    transformer._repeated_blocks = list(repeated_blocks)


def resolve_transformer(pipe_or_adapter: Any) -> Tuple[Optional[str], Any]:
  """Resolve the denoising transformer of a pipeline or adapter, with the attribute it came from.

  Callers that only read the module can use `pipe.transformer` directly; this helper exists for the
  pipelines that only have a differently named partition, and for the callers that have to write the
  module back — assigning `pipe.transformer` on a `ModularPipeline` that owns `transformer_ref`
  would silently create a second, unused attribute.

  :param pipe_or_adapter: Pipeline, modular pipeline, `BlockAdapter` or bare `nn.Module`.
  :returns: ``(attribute_name, module)``; the name is None for a bare module, and both are
    ``(None, None)`` when nothing matches.
  """
  # `resolve_pipe` returns None for anything that is not a (modular) pipeline, and the callers also
  # pass bare objects that merely carry the attribute, so fall back to the object itself.
  pipe = resolve_pipe(pipe_or_adapter)
  if pipe is None:
    pipe = pipe_or_adapter
  for name in TRANSFORMER_ATTR_NAMES:
    module = getattr(pipe, name, None)
    if module is not None:
      return name, module
  # A caller holding the transformer itself (the transformer-only adapter path) gets it back with
  # no attribute name; this is a last resort so a wrapper module that *owns* a transformer is still
  # resolved through its attribute above.
  if isinstance(pipe_or_adapter, torch.nn.Module):
    return None, pipe_or_adapter
  return None, None


def assign_transformer(pipe: Any, module: torch.nn.Module, name: Optional[str] = None) -> str:
  """Store a (possibly replaced) transformer back on ``pipe`` under the right attribute name.

  :param pipe: Pipeline that owns the transformer.
  :param module: Transformer to store.
  :param name: Attribute to use; resolved from ``pipe`` when omitted.
  :returns: The attribute name that was written.
  :raises AttributeError: When the pipeline has no such attribute, which would otherwise create a
    second, unused partition next to the one the denoise step reads.
  """
  resolved_name, _ = resolve_transformer(pipe)
  target = name or resolved_name or TRANSFORMER_ATTR_NAMES[0]
  if not hasattr(pipe, target):
    raise AttributeError(f"{type(pipe).__name__} has no '{target}' attribute to replace; pass the "
                         f"name of the partition it actually loaded "
                         f"({', '.join(TRANSFORMER_ATTR_NAMES)}).")
  setattr(pipe, target, module)
  return target


def _is_text_encoder(module: torch.nn.Module) -> bool:
  _import_module = module.__class__.__module__
  # Including the cases for normal text encoder and vision-language
  # model (e.g, GLM Image) in transformers
  return (_import_module.startswith("transformers")
          or bool(getattr(module, "_cache_dit_text_encoder", False)))


def _is_controlnet(module: torch.nn.Module) -> bool:
  _import_module = module.__class__.__module__
  return _import_module.startswith("diffusers.models.controlnet")


def _is_auto_encoder(module: torch.nn.Module) -> bool:
  _import_module = module.__class__.__module__
  return _import_module.startswith("diffusers.models.autoencoder")


def _is_parallelized(module: torch.nn.Module) -> bool:
  """Check if the given module is already parallelized.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  return getattr(module, "_is_parallelized", False)


def _is_quantized(module: torch.nn.Module) -> bool:
  """Check if the given module is already quantized.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  return getattr(module, "_is_quantized", False)


def _is_cached(module: torch.nn.Module) -> bool:
  """Check if the given module is already cached.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  return getattr(module, "_is_cached", False)


def _is_patched(module: torch.nn.Module) -> bool:
  """Check if the given module is already patched (patch functor applied).

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  return getattr(module, "_is_patched", False)


def check_text_encoder(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline has text encoder.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_text_encoder(module)

  pipe = resolve_pipe(module)
  if hasattr(pipe, "text_encoder") and getattr(pipe, "text_encoder") is not None:
    return True
  # For some pipelines (e.g., FLUX), the text encoder may have different names.
  for attr_name in dir(pipe):
    attr = getattr(pipe, attr_name)
    if isinstance(attr, torch.nn.Module) and _is_text_encoder(attr):
      return True
  return False


def check_controlnet(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline has ControlNet.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_controlnet(module)

  pipe = resolve_pipe(module)
  if hasattr(pipe, "controlnet") and getattr(pipe, "controlnet") is not None:
    return True
  return False


def check_auto_encoder(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline has auto encoder.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_auto_encoder(module)

  pipe = resolve_pipe(module)
  if hasattr(pipe, "vae") and getattr(pipe, "vae") is not None:
    return True
  return False


def check_parallelized(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline is already parallelized.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_parallelized(module)

  pipe = resolve_pipe(module)
  for attr_name in dir(pipe):
    attr = getattr(pipe, attr_name)
    if isinstance(attr, torch.nn.Module) and _is_parallelized(attr):
      return True
  return False


def check_quantized(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline is already quantized.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_quantized(module)

  pipe = resolve_pipe(module)
  for attr_name in dir(pipe):
    attr = getattr(pipe, attr_name)
    if isinstance(attr, torch.nn.Module) and _is_quantized(attr):
      return True
  return False


def check_cached(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline is already cached.

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_cached(module)

  pipe = resolve_pipe(module)
  for attr_name in dir(pipe):
    attr = getattr(pipe, attr_name)
    if isinstance(attr, torch.nn.Module) and _is_cached(attr):
      return True
  return False


def check_patched(module: torch.nn.Module | DiffusionPipeline | Any) -> bool:
  """Check if the given pipeline is already patched (patch functor applied).

  :param module: Module to process.
  :returns: Whether the requested condition is satisfied.
  """
  if isinstance(module, torch.nn.Module) and not is_pipeline(module):
    return _is_patched(module)

  pipe = resolve_pipe(module)
  for attr_name in dir(pipe):
    attr = getattr(pipe, attr_name)
    if isinstance(attr, torch.nn.Module) and _is_patched(attr):
      return True
  return False


def parse_text_encoder(
  pipe: DiffusionPipeline, ) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
  pipe_cls_name = pipe.__class__.__name__
  if (hasattr(pipe, "text_encoder_2") and not pipe_cls_name.startswith("Hunyuan")
      and not pipe_cls_name.startswith("Kandinsky")):
    # Specific for FluxPipeline, FLUX.1-dev
    return getattr(pipe, "text_encoder_2"), "text_encoder_2"
  elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
    return getattr(pipe, "text_encoder_3"), "text_encoder_3"
  elif hasattr(
      pipe,
      "vision_language_encoder") and pipe_cls_name.startswith("GlmImage"):  # GLM Image pipeline
    return getattr(pipe, "vision_language_encoder"), "vision_language_encoder"
  elif hasattr(pipe, "text_encoder"):  # General case
    return getattr(pipe, "text_encoder"), "text_encoder"
  else:
    return None, None


def parse_extra_modules(
  pipe_or_adapter: DiffusionPipeline | Any,
  extra_modules: List[str | torch.nn.Module],
  skip_text_encoder_override: bool = False,
) -> Union[List[torch.nn.Module], List]:
  """Parse extra modules according to the given names in extra_modules to actual modules in the.

  pipeline.

  Useful for extra parallelism and extra quantization outside of the transformer module, e.g.,
  applying parallelism or quantization to text encoder and vae at the same time. For example, when
  extra_modules is set to ['text_encoder', 'vae'], we will try to find the text encoder and vae
  modules in the pipeline and apply parallelism or quantization to these modules as well. Note that
  the supported extra module names may vary for different pipelines, but generally include common
  components such as 'text_encoder', 'vae', 'unet', etc. User can also directly pass the actual
  module objects in extra_modules for more precise control.

  :param pipe_or_adapter: The DiffusionPipeline or BlockAdapter to parse the extra modules from.
  :param extra_modules: A list of module names or actual module objects to be parsed.
  :param skip_text_encoder_override: Whether to skip the text encoder override for some special
    pipelines (e.g., FLUX). By default, it's set to False, which means we will try to find the
    text_encoder with largest parameters as the text encoder to apply parallelism or quantization,
    instead of using the default text_encoder or text_encoder_2 attribute in the pipeline. For
    example, in FLUX pipeline, there are two text encoders, text_encoder and text_encoder_2, and
    text_encoder_2 has much more parameters than text_encoder. So by default, we will apply
    parallelism or quantization to text_encoder_2. But if skip_text_encoder_override is set to True,
    we will apply parallelism or quantization to the default text_encoder attribute in the pipeline.
  :returns: A list of parsed extra modules as actual module objects. If a module name is not found
    in the pipeline, it will be skipped with a warning.
  """
  pipe = resolve_pipe(pipe_or_adapter)

  if not extra_modules or pipe is None:  # empty list
    return []

  parsed_extra_modules: List[torch.nn.Module] = []
  for module_or_name in extra_modules:
    if isinstance(module_or_name, torch.nn.Module):
      parsed_extra_modules.append(module_or_name)
      continue

    if hasattr(pipe, module_or_name):
      if module_or_name.lower() == "text_encoder" and not skip_text_encoder_override:
        # Special handling for text encoder
        text_encoder, text_encoder_name = parse_text_encoder(pipe)
        if text_encoder is not None:
          text_encoder._actual_module_name = text_encoder_name
          parsed_extra_modules.append(text_encoder)
        else:
          logger.warning("Text encoder not found in the pipeline for extra modules.")
      else:
        parsed_extra_modules.append(getattr(pipe, module_or_name))
    else:
      logger.warning(f"Extra module name {module_or_name} not found in the pipeline.")
  return parsed_extra_modules


# Some others trival utils for logging and cache management, e.g.,
# disabling print for some noisy functions, maybe_empty_cache to
# clear cache and collect garbage to free up memory.
def dummy_print(*args, **kwargs):
  ...  # noqa: E704


@contextlib.contextmanager
def disable_print():
  origin_print = __builtin__.print
  __builtin__.print = dummy_print
  yield
  __builtin__.print = origin_print


def maybe_empty_cache():
  try:
    time.sleep(1)
    gc.collect()
    current_platform.empty_cache()
    current_platform.ipc_collect()
    time.sleep(1)
    gc.collect()
    current_platform.empty_cache()
    current_platform.ipc_collect()
  except Exception:
    pass
