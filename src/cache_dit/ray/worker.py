from __future__ import annotations

import copy
import importlib
import os
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin

from ..distributed import ParallelismConfig
from ..distributed import enable_parallelism
from ..logger import init_logger
from ..quantization import QuantizeConfig
from ..quantization import quantize
from ..utils import parse_extra_modules
from ._tree import cpu_tensor_tree
from ._tree import device_tensor_tree
from .dist import destroy_worker_process_group
from .dist import init_worker_process_group

logger = init_logger(__name__)


def _resolve_class(class_ref: str) -> type:
  """Resolve a ``module:qualname`` string returned by :func:`_qualified_class_name`.

  The format mirrors what ``copyreg._reduce_ex`` stores for a class, so it is
  round-trip-safe for standard and nested classes alike.

  :param class_ref: Encoded class reference produced by ``_qualified_class_name``.
  :returns: The resolved Python class object.
  :raises ModuleNotFoundError: The module portion of *class_ref* cannot be imported.
  :raises AttributeError: The qualname portion of *class_ref* is missing from the module.
  :raises TypeError: *class_ref* is not a string (e.g. a raw class object from an
    older cache-dit that did not use ``_qualified_class_name``).
  """

  if not isinstance(class_ref, str):
    raise TypeError(f"_resolve_class expected a 'module:qualname' string, got "
                    f"{type(class_ref).__name__} ({class_ref!r}).  This usually means "
                    f"the calling code passed a raw class object instead of using "
                    f"_qualified_class_name().")

  module_name, _, qualname = class_ref.rpartition(":")
  module = importlib.import_module(module_name)
  # qualname may contain dots for nested classes, e.g. "Outer.Inner"
  obj = module
  for part in qualname.split("."):
    obj = getattr(obj, part)
  return obj


def _maybe_compile_transformer(
  transformer: torch.nn.Module | ModelMixin,
  parallelism_config: ParallelismConfig,
) -> torch.nn.Module | ModelMixin:
  """Compile a Ray-owned transformer when requested by the parallelism config.

  :param transformer: Transformer copy already moved to the actor device and parallelized.
  :param parallelism_config: Ray-enabled parallelism configuration.
  :returns: The same transformer object after any in-place compile step.
  """

  if not parallelism_config.ray_use_compile:
    return transformer

  compile_repeated_blocks = getattr(transformer, "compile_repeated_blocks", None)
  if callable(compile_repeated_blocks):
    logger.info("Compiling Ray-owned transformer with compile_repeated_blocks().")
    transformer.compile_repeated_blocks()
    return transformer

  compile_module = getattr(transformer, "compile", None)
  if callable(compile_module):
    logger.info("Compiling Ray-owned transformer with nn.Module.compile().")
    transformer.compile()
    return transformer

  logger.warning("ray_use_compile=True, but transformer does not support "
                 "compile_repeated_blocks() or nn.Module.compile(); skipping compile.")
  return transformer


def _maybe_apply_cache(
  module_or_pipe: torch.nn.Module | ModelMixin | DiffusionPipeline,
  cache_context_kwargs: dict[str, Any] | None,
) -> torch.nn.Module | ModelMixin | DiffusionPipeline:
  """Apply cache hooks inside a Ray worker when cache config is provided.

  :param module_or_pipe: Worker-local transformer or pipeline copy.
  :param cache_context_kwargs: Cache context keyword arguments from ``cache_dit.enable_cache``.
  :returns: The same object with cache hooks applied, when requested.
  """

  if cache_context_kwargs is None:
    return module_or_pipe

  from ..caching.cache_adapters import CachedAdapter

  logger.info(f"Applying cache hooks inside Ray worker for {module_or_pipe.__class__.__name__}.")
  return CachedAdapter.apply(module_or_pipe, **copy.deepcopy(cache_context_kwargs))


def _maybe_quantize_transformer(
  transformer: torch.nn.Module | ModelMixin,
  quantize_config: QuantizeConfig | None,
) -> torch.nn.Module | ModelMixin:
  """Quantize a worker-local transformer when requested.

  :param transformer: Worker-local transformer after cache and parallelism have been applied.
  :param quantize_config: Optional quantization configuration.
  :returns: Quantized transformer when requested, otherwise the original transformer.
  """

  if quantize_config is None:
    return transformer
  logger.info(f"Applying quantization inside Ray worker for {transformer.__class__.__name__}.")
  return quantize(transformer, quantize_config=copy.deepcopy(quantize_config))


def _maybe_quantize_pipeline(
  pipe: DiffusionPipeline,
  quantize_config: QuantizeConfig | None,
) -> DiffusionPipeline:
  """Quantize worker-local pipeline components when requested.

  :param pipe: Worker-local pipeline after cache and transformer parallelism have been applied.
  :param quantize_config: Optional quantization configuration.
  :returns: Pipeline with requested components quantized.
  """

  if quantize_config is None:
    return pipe
  if quantize_config.components_to_quantize is None:
    pipe.transformer = _maybe_quantize_transformer(pipe.transformer, quantize_config)
    return pipe

  expanded_quantize_configs = QuantizeConfig.expand_configs(quantize_config)
  for config in expanded_quantize_configs:
    components_to_quantize = config.components_to_quantize
    components = parse_extra_modules(pipe, components_to_quantize)
    assert len(components) == len(components_to_quantize), (
      f"Some components in quantize_config.components_to_quantize: {components_to_quantize} "
      "are not found in the pipeline, please check the component names or directly pass the "
      "actual modules in components_to_quantize.")
    for component, name in zip(components, components_to_quantize):
      name = getattr(component, "_actual_module_name", name)
      quantized_component = quantize(component, quantize_config=copy.deepcopy(config))
      setattr(pipe, name, quantized_component)
  return pipe


class RayTransformerWorker:
  """Ray actor body that owns one rank of a cache-dit parallel transformer.

  :param rank: Global rank assigned by the Ray engine.
  :param world_size: Number of Ray actors in the model-parallel group.
  :param parallelism_config: Parallelism configuration to apply inside the actor.
  :param master_port: TCPStore port shared by all actors.
  """

  def __init__(
    self,
    rank: int,
    world_size: int,
    parallelism_config: ParallelismConfig,
    cache_context_kwargs: dict[str, Any] | None,
    quantize_config: QuantizeConfig | None,
    master_port: int,
  ):
    self.rank = rank
    self.world_size = world_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.device.type == "cuda":
      torch.cuda.set_device(torch.cuda.current_device())

    init_worker_process_group(
      rank=rank,
      world_size=world_size,
      master_port=master_port,
    )

    self.parallelism_config = copy.deepcopy(parallelism_config)
    self.cache_context_kwargs = copy.deepcopy(cache_context_kwargs)
    self.quantize_config = copy.deepcopy(quantize_config)
    self.parallelism_config.ray_num_workers = world_size
    if self.parallelism_config.hybrid_enabled():
      self.parallelism_config._maybe_init_hybrid_meshes()

    self.transformer: torch.nn.Module | ModelMixin | None = None

  def load_transformer(self, transformer: torch.nn.Module | ModelMixin) -> dict[str, Any]:
    """Load and parallelize the transformer copy owned by this actor.

    :param transformer: CPU transformer copy from the Ray object store.
    :returns: Device placement and memory information after loading.
    """

    self.transformer = transformer.to(self.device)
    self.transformer.eval()
    self.transformer = _maybe_apply_cache(self.transformer, self.cache_context_kwargs)
    if not self.parallelism_config._ray_skip_native_parallelism:
      self.transformer = enable_parallelism(self.transformer, self.parallelism_config)
    self.transformer = _maybe_quantize_transformer(self.transformer, self.quantize_config)
    self.transformer = _maybe_compile_transformer(self.transformer, self.parallelism_config)
    return self.device_info()

  def load_transformer_from_file(self, path: str) -> dict[str, Any]:
    """Load and parallelize a transformer serialized on the local filesystem.

    :param path: Path to a CPU transformer checkpoint written by the Ray engine.
    :returns: Device placement and memory information after loading.
    """

    transformer = torch.load(path, map_location="cpu", weights_only=False)
    return self.load_transformer(transformer)

  def load_transformer_from_safetensors(
    self,
    transformer_cls_ref: str,
    transformer_config: dict[str, Any],
    path: str,
  ) -> dict[str, Any]:
    """Load a diffusers transformer from a safetensors state dict.

    :param transformer_cls_ref: ``module:qualname`` string encoding the transformer class.
    :param transformer_config: Serialized transformer config passed to ``from_config``.
    :param path: Path to a safetensors state dict written by the Ray engine.
    :returns: Device placement and memory information after loading.
    """

    try:
      from safetensors.torch import load_file
    except ImportError as exc:
      raise ImportError(
        "Ray safetensors transfer requires `safetensors`. Install with "
        "`pip install cache-dit[ray,parallelism]` or `pip install safetensors`.") from exc

    transformer_cls = _resolve_class(transformer_cls_ref)
    with torch.device("meta"):
      transformer = transformer_cls.from_config(transformer_config)
    state_dict = load_file(path, device=str(self.device))
    transformer.load_state_dict(state_dict, assign=True)
    self.transformer = transformer.eval()
    self.transformer = _maybe_apply_cache(self.transformer, self.cache_context_kwargs)
    if not self.parallelism_config._ray_skip_native_parallelism:
      self.transformer = enable_parallelism(self.transformer, self.parallelism_config)
    self.transformer = _maybe_quantize_transformer(self.transformer, self.quantize_config)
    self.transformer = _maybe_compile_transformer(self.transformer, self.parallelism_config)
    return self.device_info()

  def load_transformer_from_pretrained(
    self,
    transformer_cls_ref: str,
    model_path: str,
    torch_dtype: torch.dtype | None,
    use_flashpack: bool,
  ) -> dict[str, Any]:
    """Load a diffusers transformer snapshot inside this actor.

    :param transformer_cls_ref: ``module:qualname`` string encoding the transformer class.
    :param model_path: Local snapshot directory written by the Ray engine.
    :param torch_dtype: Optional dtype for model loading.
    :param use_flashpack: Whether to prefer FlashPack weights during loading.
    :returns: Device placement and memory information after loading.
    """

    transformer_cls = _resolve_class(transformer_cls_ref)
    load_kwargs = {
      "use_safetensors": True,
      "use_flashpack": use_flashpack,
    }
    if torch_dtype is not None:
      load_kwargs["torch_dtype"] = torch_dtype
    transformer = transformer_cls.from_pretrained(model_path, **load_kwargs)
    return self.load_transformer(transformer)

  def ready(self) -> int:
    """Return the rank after actor initialization has completed.

    :returns: The actor rank.
    """

    return self.rank

  def device_info(self) -> dict[str, Any]:
    """Return actor device placement details for diagnostics.

    :returns: Rank, torch device, and visible accelerator ids for this actor.
    """

    info = {
      "rank": self.rank,
      "device": str(self.device),
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if self.device.type == "cuda":
      info["memory_allocated_mib"] = torch.cuda.memory_allocated() // 1024 // 1024
      info["memory_reserved_mib"] = torch.cuda.memory_reserved() // 1024 // 1024
    return info

  def forward(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    """Run a transformer forward on this Ray rank.

    :param args: CPU-staged positional arguments from the main process.
    :param kwargs: CPU-staged keyword arguments from the main process.
    :returns: CPU-staged output on rank 0 and ``None`` on other ranks.
    """

    device_args = device_tensor_tree(args, self.device)
    device_kwargs = device_tensor_tree(kwargs, self.device)
    if self.transformer is None:
      raise RuntimeError("RayTransformerWorker.forward called before load_transformer.")
    with torch.no_grad():
      output = self.transformer(*device_args, **device_kwargs)
    if self.rank != 0:
      return None
    return cpu_tensor_tree(output)

  def shutdown(self) -> None:
    """Release actor-local distributed state."""

    destroy_worker_process_group()


class RayPipelineWorker:
  """Ray actor body that owns one full pipeline and one distributed rank.

  :param rank: Global rank assigned by the Ray engine.
  :param world_size: Number of Ray actors in the model-parallel group.
  :param parallelism_config: Parallelism configuration to apply inside the actor.
  :param master_port: TCPStore port shared by all actors.
  """

  def __init__(
    self,
    rank: int,
    world_size: int,
    parallelism_config: ParallelismConfig,
    cache_context_kwargs: dict[str, Any] | None,
    quantize_config: QuantizeConfig | None,
    master_port: int,
  ):
    self.rank = rank
    self.world_size = world_size
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if self.device.type == "cuda":
      torch.cuda.set_device(torch.cuda.current_device())

    init_worker_process_group(
      rank=rank,
      world_size=world_size,
      master_port=master_port,
    )

    self.parallelism_config = copy.deepcopy(parallelism_config)
    self.cache_context_kwargs = copy.deepcopy(cache_context_kwargs)
    self.quantize_config = copy.deepcopy(quantize_config)
    self.parallelism_config.ray_num_workers = world_size
    if self.parallelism_config.hybrid_enabled():
      self.parallelism_config._maybe_init_hybrid_meshes()

    self.pipe: DiffusionPipeline | None = None

  def load_pipeline(self, pipe: DiffusionPipeline) -> dict[str, Any]:
    """Load a pipeline copy and parallelize its transformer inside this actor.

    :param pipe: CPU pipeline copy from the Ray object store.
    :returns: Device placement and memory information after loading.
    """

    for component in pipe.components.values():
      if isinstance(component, torch.nn.Module):
        component.to(self.device)
    self.pipe = pipe
    self.pipe.set_progress_bar_config(disable=True)
    self.pipe = _maybe_apply_cache(self.pipe, self.cache_context_kwargs)
    self.pipe.transformer.eval()
    if not self.parallelism_config._ray_skip_native_parallelism:
      self.pipe.transformer = enable_parallelism(self.pipe.transformer, self.parallelism_config)
    self.pipe = _maybe_quantize_pipeline(self.pipe, self.quantize_config)
    self.pipe.transformer = _maybe_compile_transformer(
      self.pipe.transformer,
      self.parallelism_config,
    )
    return self.device_info()

  def load_pipeline_from_pretrained(
    self,
    pipe_cls_ref: str,
    model_path: str,
    torch_dtype: torch.dtype | None,
    use_flashpack: bool,
  ) -> dict[str, Any]:
    """Load a pipeline from its pretrained directory inside this actor.

    :param pipe_cls_ref: ``module:qualname`` string encoding the pipeline class.
    :param model_path: Local path or model id passed to ``from_pretrained``.
    :param torch_dtype: Optional dtype for model loading.
    :param use_flashpack: Whether to prefer FlashPack weights during loading.
    :returns: Device placement and memory information after loading.
    """

    pipe_cls = _resolve_class(pipe_cls_ref)
    load_kwargs = {}
    if torch_dtype is not None:
      load_kwargs["torch_dtype"] = torch_dtype
    load_kwargs["use_safetensors"] = True
    load_kwargs["use_flashpack"] = use_flashpack
    pipe = pipe_cls.from_pretrained(model_path, **load_kwargs)
    return self.load_pipeline(pipe)

  def ready(self) -> int:
    """Return the rank after actor initialization has completed.

    :returns: The actor rank.
    """

    return self.rank

  def device_info(self) -> dict[str, Any]:
    """Return actor device placement details for diagnostics.

    :returns: Rank, torch device, and visible accelerator ids for this actor.
    """

    info = {
      "rank": self.rank,
      "device": str(self.device),
      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if self.device.type == "cuda":
      info["memory_allocated_mib"] = torch.cuda.memory_allocated() // 1024 // 1024
      info["memory_reserved_mib"] = torch.cuda.memory_reserved() // 1024 // 1024
    return info

  def call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any | None:
    """Run a full pipeline call on this Ray rank.

    :param args: Positional arguments from the main process.
    :param kwargs: Keyword arguments from the main process.
    :returns: Pipeline output on rank 0 and ``None`` on other ranks.
    """

    if self.pipe is None:
      raise RuntimeError("RayPipelineWorker.call called before load_pipeline.")
    device_args = device_tensor_tree(args, self.device)
    device_kwargs = device_tensor_tree(kwargs, self.device)
    with torch.no_grad():
      output = self.pipe(*device_args, **device_kwargs)
    if self.rank != 0:
      return None
    return cpu_tensor_tree(output)

  def shutdown(self) -> None:
    """Release actor-local distributed state."""

    destroy_worker_process_group()
