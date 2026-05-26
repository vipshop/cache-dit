from __future__ import annotations

import dataclasses
import json
import socket
import shutil
import uuid
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers import pipelines as diffusers_pipelines
from diffusers.models.modeling_utils import ModelMixin

from ..distributed import ParallelismConfig
from ..logger import init_logger
from ..quantization import QuantizeConfig
from ..utils import maybe_empty_cache
from ._tree import cpu_tensor_tree
from ._tree import device_tensor_tree
from ._tree import first_tensor_device
from .worker import RayTransformerWorker
from .worker import RayPipelineWorker

logger = init_logger(__name__)


def _qualified_class_name(cls: type) -> str:
  """Encode a class as ``module:qualname`` to avoid pickle serialization of class objects.

  Passing a raw class object as a Ray task argument forces pickle to serialize
  the entire inheritance chain, which fails on workers when the class (or any
  base class) is defined in a user-application module that the workers cannot
  import.  A string reference defers the import to the worker side, where it can
  be resolved via ``importlib`` after the application's ``runtime_env`` has been
  applied.
  """

  return f"{cls.__module__}:{cls.__qualname__}"


def _maybe_user_module(cls: type) -> bool:
  """Return ``True`` when *cls* is likely from user-application code.

  Classes from ``diffusers``, ``transformers``, ``torch``, or ``cache_dit`` are
  expected to be importable on every Ray worker, so no warning is needed.
  """

  _well_known = ("diffusers", "transformers", "torch", "cache_dit", "builtins")
  return not cls.__module__.startswith(_well_known)


def _warn_if_ray_pre_initialized(
  ray: Any,
  cls: type,
  config: ParallelismConfig,
  ray_initialized_by_engine: bool = False,
) -> None:
  """Warn when Ray was initialized *before* cache-dit and the transferred class is from a user-
  application module.

  When Ray is already running, cache-dit cannot inject ``runtime_env`` — the
  user's ``ray.init()`` is responsible for making the application code available
  to every worker.  If that was not configured, worker-side pickle deserialization
  (or ``from_pretrained`` class imports) will fail with ``ModuleNotFoundError``.

  :param ray: Imported Ray module.
  :param cls: The pipeline or transformer class being transferred.
  :param config: The user-provided parallelism configuration.
  :param ray_initialized_by_engine: ``True`` when cache-dit itself just called
    ``ray.init()``.  In that case the ``runtime_env`` from *config* was already
    applied and no warning is needed.
  """

  if not ray.is_initialized():
    return
  if ray_initialized_by_engine:
    return
  if not _maybe_user_module(cls):
    return

  extra = ""
  if config.ray_runtime_env is not None:
    extra = (" (ParallelismConfig.ray_runtime_env is set but will NOT be applied "
             "because Ray was already initialized)")
  logger.warning(f"Ray is already initialized and {cls.__name__} is defined in "
                 f"'{cls.__module__}', which may not be importable on Ray workers.{extra} "
                 f"Ensure your ray.init() call includes a runtime_env that exposes this "
                 f"module, e.g. ray.init(runtime_env={{'working_dir': '...'}}). "
                 f"Alternatively, configure ParallelismConfig.ray_runtime_env and let "
                 f"cache_dit initialize Ray for you.")


def _require_ray():
  try:
    import ray
  except ImportError as exc:
    raise ImportError("Ray wrapper requires ray. Install it with `pip install ray` or "
                      "`pip install cache-dit[ray]`.") from exc
  return ray


def _init_ray(ray: Any, **init_kwargs: Any) -> None:
  """Initialize Ray while suppressing its accelerator override transition warning.

  :param ray: Imported Ray module.
  :param init_kwargs: Keyword arguments forwarded to ``ray.init``.
  """

  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message=("Tip: In future versions of Ray, Ray will no longer override accelerator "
               "visible devices env var if num_gpus=0 or num_gpus=None.*"),
      category=FutureWarning,
      module="ray\\._private\\.worker",
    )
    ray.init(**init_kwargs)


def _get_free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return int(sock.getsockname()[1])


def _import_safetensors_save_file():
  try:
    from safetensors.torch import save_file
  except ImportError as exc:
    raise ImportError("Ray safetensors transfer requires `safetensors`. Install with "
                      "`pip install cache-dit[ray]` or `pip install safetensors`.") from exc
  return save_file


def _save_state_dict_safetensors(module: torch.nn.Module, path: Path) -> None:
  save_file = _import_safetensors_save_file()
  tensors = {
    name: tensor.detach().cpu().contiguous()
    for name, tensor in module.state_dict().items()
  }
  save_file(tensors, str(path))


def _first_parameter_device(module: torch.nn.Module) -> torch.device | None:
  for parameter in module.parameters(recurse=True):
    return parameter.device
  return None


def _first_parameter_dtype(module: torch.nn.Module) -> torch.dtype | None:
  for parameter in module.parameters(recurse=True):
    return parameter.dtype
  return None


def _first_pipeline_module_device(pipe: DiffusionPipeline) -> torch.device | None:
  for component in pipe.components.values():
    if isinstance(component, torch.nn.Module):
      device = _first_parameter_device(component)
      if device is not None:
        return device
  return None


def _first_pipeline_module_dtype(pipe: DiffusionPipeline) -> torch.dtype | None:
  for component in pipe.components.values():
    if isinstance(component, torch.nn.Module):
      dtype = _first_parameter_dtype(component)
      if dtype is not None:
        return dtype
  return None


def _move_pipeline_modules(pipe: DiffusionPipeline, device: str | torch.device) -> None:
  for component in pipe.components.values():
    if isinstance(component, torch.nn.Module):
      component.to(device)


def _pipeline_supports_save_pretrained(pipe: DiffusionPipeline) -> bool:
  for component in pipe.components.values():
    if component is None:
      continue
    if not callable(getattr(component, "save_pretrained", None)):
      return False
  return callable(getattr(pipe, "save_pretrained", None))


def _model_supports_save_pretrained(model: ModelMixin) -> bool:
  return (callable(getattr(model, "save_pretrained", None))
          and callable(getattr(model.__class__, "from_pretrained", None)))


def _fix_model_index_for_custom_components(pipe: DiffusionPipeline, pipeline_path: Path) -> None:
  """Post-process ``model_index.json`` to correct misclassified custom component library names.

  Diffusers' ``_fetch_class_library_tuple`` can misclassify a custom component whose
  Python module path happens to contain a ``diffusers.pipelines`` submodule name.
  For example, a custom scheduler at ``myapp.flux.scheduler`` is assigned
  ``library_name="flux"``, causing ``from_pretrained`` to look for the class in
  ``diffusers.pipelines.flux`` where it doesn't exist.

  This function detects such entries and replaces the pipeline-submodule
  ``library_name`` with the component's full ``__module__`` path.  After the fix,
  ``from_pretrained`` resolves the class via ``importlib.import_module``, which
  succeeds as long as the module is importable on the worker.

  :param pipe: The pipeline whose in-memory components provide ground-truth class modules.
  :param pipeline_path: Directory where ``pipe.save_pretrained`` wrote ``model_index.json``.
  """

  model_index_path = pipeline_path / "model_index.json"
  with open(model_index_path, "r") as f:
    model_index = json.load(f)

  modified = False
  for component_name, entry in list(model_index.items()):
    # Skip metadata keys (they start with "_")
    if component_name.startswith("_"):
      continue
    # Each component entry is [library_name, class_name]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
      continue
    library_name, class_name = entry
    # Only inspect entries whose library_name matches a pipelines submodule
    if not hasattr(diffusers_pipelines, library_name):
      continue
    pipeline_module = getattr(diffusers_pipelines, library_name)
    if hasattr(pipeline_module, class_name):
      # Class legitimately lives in that pipelines submodule — no misclassification
      continue

    # Misclassification: the pipelines submodule exists but does NOT export this class.
    # Use the component's actual module path instead.
    component = pipe.components.get(component_name)
    if component is None:
      continue
    actual_module = component.__class__.__module__
    logger.info(f"Fixing model_index.json: component '{component_name}' "
                f"({class_name}) was classified as pipelines submodule "
                f"'{library_name}', correcting library_name to '{actual_module}'.")
    model_index[component_name] = [actual_module, class_name]
    modified = True

  if modified:
    with open(model_index_path, "w") as f:
      json.dump(model_index, f, indent=2)
    logger.info("Updated model_index.json with corrected library_name entries.")


def _build_custom_class_map(pipe: DiffusionPipeline) -> dict[str, str]:
  """Build a mapping from every component's class name to its ``module:qualname`` string.

  This map is passed to Ray workers as a safety net: if diffusers' normal class
  resolution fails during ``from_pretrained``, the worker can look up the actual
  class via this registry and monkey-patch it into the expected module namespace.

  :param pipe: The pipeline whose components supply the class references.
  :returns: A dict mapping ``class_name`` → ``"module:qualname"`` for every
    non-None component.
  """

  class_map: dict[str, str] = {}
  for component_name, component in pipe.components.items():
    if component is None:
      continue
    cls = component.__class__
    class_map[cls.__name__] = _qualified_class_name(cls)
  return class_map


class RayParallelEngine:
  """Main-process engine that dispatches transformer forwards to Ray actors.

  :param transformer: User-visible transformer whose forward will be proxied.
  :param parallelism_config: Ray-enabled parallelism configuration.
  """

  def __init__(
    self,
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
    cache_context_kwargs: dict[str, Any] | None = None,
    quantize_config: QuantizeConfig | None = None,
  ):
    self.ray = _require_ray()
    self.parallelism_config = parallelism_config
    self.cache_context_kwargs = cache_context_kwargs
    self.quantize_config = quantize_config
    self._source_transformer = transformer
    self._source_device = _first_parameter_device(transformer)
    parallel_world_size = parallelism_config._get_world_size()
    self.world_size = parallelism_config.ray_num_workers or parallel_world_size
    if self.world_size <= 1:
      raise ValueError("Ray parallelism requires a world size greater than 1.")
    if (parallelism_config.ray_num_workers is not None
        and parallelism_config.ray_num_workers != parallel_world_size):
      raise ValueError("ray_num_workers must match the parallelism world size for the minimal "
                       f"Ray wrapper. Got ray_num_workers={parallelism_config.ray_num_workers}, "
                       f"world_size={parallel_world_size}.")

    self._ray_initialized_by_engine = False
    self._transfer_dir: Path | None = None
    if not self.ray.is_initialized():
      init_kwargs = dict(parallelism_config.ray_init_kwargs)
      if parallelism_config.ray_address is not None:
        init_kwargs["address"] = parallelism_config.ray_address
      if parallelism_config.ray_runtime_env is not None:
        init_kwargs["runtime_env"] = parallelism_config.ray_runtime_env
      _init_ray(self.ray, **init_kwargs)
      self._ray_initialized_by_engine = True

    _warn_if_ray_pre_initialized(
      self.ray,
      transformer.__class__,
      self.parallelism_config,
      ray_initialized_by_engine=self._ray_initialized_by_engine,
    )

    self.master_port = parallelism_config.ray_master_port or _get_free_port()
    if self.parallelism_config.ray_transfer_fn is not None:
      raise NotImplementedError(
        "ray_transfer_fn is only supported for pipeline-level Ray wrapper "
        "(enable_cache(pipe, ...)). Transformer-level init-fn is not yet supported.")
    self._actors = self._create_workers(transformer)

  def _create_workers(self, transformer: torch.nn.Module | ModelMixin) -> list[Any]:
    remote_worker = self.ray.remote(RayTransformerWorker)
    worker_options = {"num_gpus": 1}
    worker_options.update(self.parallelism_config.ray_worker_options)
    actors = [
      remote_worker.options(**worker_options).remote(
        rank,
        self.world_size,
        self.parallelism_config,
        self.cache_context_kwargs,
        self.quantize_config,
        self.master_port,
      ) for rank in range(self.world_size)
    ]
    self.ray.get([actor.ready.remote() for actor in actors])
    device_infos = self.ray.get([actor.device_info.remote() for actor in actors])
    logger.info(f"Ray transformer worker placement before load: {device_infos}")

    if self._source_device is not None and self._source_device.type != "cpu":
      logger.info("Moving the main-process transformer to CPU before Ray worker loading.")
      offload_start = time.perf_counter()
      transformer.to("cpu")
      maybe_empty_cache()
      logger.info(f"Moved the main-process transformer to CPU in "
                  f"{time.perf_counter() - offload_start:.2f}s.")
    else:
      logger.info("The main-process transformer is already on CPU before Ray worker loading.")

    transfer_backend = self.parallelism_config.ray_transfer_backend
    if transfer_backend == "auto":
      transfer_backend = "file" if isinstance(transformer, ModelMixin) else "object_store"

    if transfer_backend == "file":
      if not isinstance(transformer, ModelMixin):
        raise ValueError("ray_transfer_backend='file' currently requires a diffusers ModelMixin "
                         "transformer. Use ray_transfer_backend='object_store' for generic "
                         "torch.nn.Module instances.")
      self._transfer_dir = Path.cwd() / ".tmp" / "cache_dit_ray" / uuid.uuid4().hex
      self._transfer_dir.mkdir(parents=True, exist_ok=True)
      save_start = time.perf_counter()
      load_start = time.perf_counter()
      if _model_supports_save_pretrained(transformer):
        transformer_path = self._transfer_dir / "transformer"
        logger.info(f"Saving the current transformer snapshot for Ray workers to "
                    f"{transformer_path}.")
        transformer.save_pretrained(
          transformer_path,
          safe_serialization=True,
          use_flashpack=self.parallelism_config.ray_use_flashpack,
        )
        logger.info(f"Saved the transformer snapshot in "
                    f"{time.perf_counter() - save_start:.2f}s.")
        tfmr_cls_ref = _qualified_class_name(transformer.__class__)
        load_infos = self.ray.get([
          actor.load_transformer_from_pretrained.remote(
            tfmr_cls_ref,
            str(transformer_path),
            _first_parameter_dtype(transformer),
            self.parallelism_config.ray_use_flashpack,
          ) for actor in actors
        ])
        logger.info(f"Loaded pretrained transformer snapshots on Ray workers in "
                    f"{time.perf_counter() - load_start:.2f}s.")
      else:
        transformer_path = self._transfer_dir / "transformer.safetensors"
        logger.info(f"Saving the CPU transformer state_dict for Ray workers to "
                    f"{transformer_path}.")
        _save_state_dict_safetensors(transformer, transformer_path)
        logger.info(f"Saved the transformer safetensors file in "
                    f"{time.perf_counter() - save_start:.2f}s.")
        tfmr_cls_ref = _qualified_class_name(transformer.__class__)
        load_infos = self.ray.get([
          actor.load_transformer_from_safetensors.remote(
            tfmr_cls_ref,
            dict(transformer.config),
            str(transformer_path),
          ) for actor in actors
        ])
        logger.info(f"Loaded the safetensors transformer on Ray workers in "
                    f"{time.perf_counter() - load_start:.2f}s.")
    elif transfer_backend == "object_store":
      logger.info("Putting the CPU transformer into the Ray object store.")
      put_start = time.perf_counter()
      transformer_ref = self.ray.put(transformer)
      logger.info(f"Put the CPU transformer into the Ray object store in "
                  f"{time.perf_counter() - put_start:.2f}s.")
      load_start = time.perf_counter()
      load_infos = self.ray.get(
        [actor.load_transformer.remote(transformer_ref) for actor in actors])
      logger.info(f"Loaded the object-store transformer on Ray workers in "
                  f"{time.perf_counter() - load_start:.2f}s.")
    else:
      raise ValueError(f"Unsupported ray_transfer_backend: {transfer_backend!r}.")
    logger.info(f"Ray transformer worker placement after load: {load_infos}")
    return actors

  def forward(self, *args: Any, **kwargs: Any) -> Any:
    """Proxy one transformer forward through all Ray ranks.

    :param args: Positional arguments passed to the original transformer forward.
    :param kwargs: Keyword arguments passed to the original transformer forward.
    :returns: Rank-0 output moved back to the caller's tensor device when possible.
    """

    output_device = first_tensor_device((args, kwargs))
    cpu_args = cpu_tensor_tree(args)
    cpu_kwargs = cpu_tensor_tree(kwargs)
    results = self.ray.get([actor.forward.remote(cpu_args, cpu_kwargs) for actor in self._actors])
    rank0_output = next((result for result in results if result is not None), None)
    if rank0_output is None:
      raise RuntimeError("Ray transformer workers did not return a rank-0 output.")
    if output_device is not None:
      rank0_output = device_tensor_tree(rank0_output, output_device)
    return rank0_output

  def shutdown(self) -> None:
    """Shutdown worker actors and optionally the Ray runtime initialized by this engine."""

    if self._actors:
      self.ray.get([actor.shutdown.remote() for actor in self._actors])
      for actor in self._actors:
        self.ray.kill(actor)
      self._actors = []
    if self._ray_initialized_by_engine and self.parallelism_config.ray_auto_shutdown:
      self.ray.shutdown()
    if self._source_device is not None and self._source_device.type != "cpu":
      restore_start = time.perf_counter()
      self._source_transformer.to(self._source_device)
      maybe_empty_cache()
      logger.info(f"Restored the main-process transformer to {self._source_device} in "
                  f"{time.perf_counter() - restore_start:.2f}s.")
    if self._transfer_dir is not None and self._transfer_dir.exists():
      shutil.rmtree(self._transfer_dir)
      self._transfer_dir = None


class RayPipelineEngine:
  """Main-process engine that dispatches full pipeline calls to Ray actors.

  :param pipe: User-visible pipeline whose ``__call__`` will be proxied.
  :param parallelism_config: Ray-enabled parallelism configuration.
  """

  def __init__(
    self,
    pipe: DiffusionPipeline | None,
    parallelism_config: ParallelismConfig,
    cache_context_kwargs: dict[str, Any] | None = None,
    quantize_config: QuantizeConfig | None = None,
  ):
    if pipe is not None:
      if not hasattr(pipe, "transformer"):
        raise ValueError("Ray pipeline parallelism requires a pipeline with a transformer "
                         "attribute.")
    self.ray = _require_ray()
    self.parallelism_config = parallelism_config
    self.cache_context_kwargs = cache_context_kwargs
    self.quantize_config = quantize_config
    self._source_pipe = pipe
    self._source_device = _first_pipeline_module_device(pipe) if pipe is not None else None
    self._source_dtype = _first_pipeline_module_dtype(pipe) if pipe is not None else None
    parallel_world_size = parallelism_config._get_world_size()
    self.world_size = parallelism_config.ray_num_workers or parallel_world_size
    if self.world_size <= 1:
      raise ValueError("Ray parallelism requires a world size greater than 1.")
    if (parallelism_config.ray_num_workers is not None
        and parallelism_config.ray_num_workers != parallel_world_size):
      raise ValueError("ray_num_workers must match the parallelism world size for the minimal "
                       f"Ray wrapper. Got ray_num_workers={parallelism_config.ray_num_workers}, "
                       f"world_size={parallel_world_size}.")

    self._ray_initialized_by_engine = False
    self._transfer_dir: Path | None = None
    if not self.ray.is_initialized():
      init_kwargs = dict(parallelism_config.ray_init_kwargs)
      if parallelism_config.ray_address is not None:
        init_kwargs["address"] = parallelism_config.ray_address
      if parallelism_config.ray_runtime_env is not None:
        init_kwargs["runtime_env"] = parallelism_config.ray_runtime_env
      _init_ray(self.ray, **init_kwargs)
      self._ray_initialized_by_engine = True

    if pipe is not None:
      _warn_if_ray_pre_initialized(
        self.ray,
        pipe.__class__,
        self.parallelism_config,
        ray_initialized_by_engine=self._ray_initialized_by_engine,
      )

    self.master_port = parallelism_config.ray_master_port or _get_free_port()
    self._actors = self._create_workers(pipe)

  def _create_workers(self, pipe: DiffusionPipeline) -> list[Any]:
    # Init-fn fast path: user-provided function controls pipeline creation on workers.
    transfer_fn = self.parallelism_config.ray_transfer_fn
    if transfer_fn is not None:
      return self._create_workers_with_init_fn(pipe, transfer_fn)

    remote_worker = self.ray.remote(RayPipelineWorker)
    worker_options = {"num_gpus": 1}
    worker_options.update(self.parallelism_config.ray_worker_options)
    actors = [
      remote_worker.options(**worker_options).remote(
        rank,
        self.world_size,
        self.parallelism_config,
        self.cache_context_kwargs,
        self.quantize_config,
        self.master_port,
      ) for rank in range(self.world_size)
    ]
    self.ray.get([actor.ready.remote() for actor in actors])
    device_infos = self.ray.get([actor.device_info.remote() for actor in actors])
    logger.info(f"Ray pipeline worker placement before load: {device_infos}")

    if self._source_device is not None and self._source_device.type != "cpu":
      logger.info("Moving the main-process pipeline to CPU before Ray worker loading.")
      offload_start = time.perf_counter()
      _move_pipeline_modules(pipe, "cpu")
      maybe_empty_cache()
      logger.info(f"Moved the main-process pipeline to CPU in "
                  f"{time.perf_counter() - offload_start:.2f}s.")
    else:
      logger.info("The main-process pipeline is already on CPU before Ray worker loading.")

    transfer_backend = self.parallelism_config.ray_transfer_backend
    model_path = getattr(pipe, "name_or_path", None)
    if transfer_backend == "auto":
      transfer_backend = "save_pretrained" if _pipeline_supports_save_pretrained(
        pipe) else "object_store"

    load_start = time.perf_counter()
    if transfer_backend == "save_pretrained":
      self._transfer_dir = Path.cwd() / ".tmp" / "cache_dit_ray" / uuid.uuid4().hex
      pipeline_path = self._transfer_dir / "pipeline"
      pipeline_path.mkdir(parents=True, exist_ok=True)
      logger.info(f"Saving the current pipeline snapshot for Ray workers to {pipeline_path}.")
      save_start = time.perf_counter()
      pipe.save_pretrained(
        pipeline_path,
        safe_serialization=True,
        use_flashpack=self.parallelism_config.ray_use_flashpack,
      )
      logger.info(f"Saved the pipeline snapshot in {time.perf_counter() - save_start:.2f}s.")

      # Build a custom-class registry for components that diffusers may not be
      # able to resolve through its normal pipeline-module lookup.  The registry
      # is passed to workers so they can monkey-patch missing classes into the
      # correct ``diffusers.pipelines`` submodule before ``from_pretrained``.
      # We intentionally do NOT modify model_index.json here — changing the
      # library_name to a full module path breaks diffusers' ``_get_load_method``
      # detection (it cannot determine which load method to use for non-standard
      # libraries).
      custom_class_map: dict[str, str] | None = None
      if self.parallelism_config.ray_transfer_custom_obj:
        custom_class_map = _build_custom_class_map(pipe)
        custom_map_path = pipeline_path / "_cache_dit_custom_classes.json"
        with open(custom_map_path, "w") as f:
          json.dump(custom_class_map, f, indent=2)

      pipe_cls_ref = _qualified_class_name(pipe.__class__)
      load_infos = self.ray.get([
        actor.load_pipeline_from_pretrained.remote(
          pipe_cls_ref,
          str(pipeline_path),
          self._source_dtype,
          self.parallelism_config.ray_use_flashpack,
          custom_class_map,
        ) for actor in actors
      ])
      logger.info(f"Loaded saved pipeline snapshots on Ray workers in "
                  f"{time.perf_counter() - load_start:.2f}s.")
    elif transfer_backend == "from_pretrained":
      if model_path is None:
        raise ValueError("ray_transfer_backend='from_pretrained' requires pipeline.name_or_path.")
      logger.info(f"Loading Ray worker pipelines from pretrained source: {model_path}.")
      pipe_cls_ref = _qualified_class_name(pipe.__class__)
      custom_class_map = _build_custom_class_map(
        pipe) if self.parallelism_config.ray_transfer_custom_obj else None
      load_infos = self.ray.get([
        actor.load_pipeline_from_pretrained.remote(
          pipe_cls_ref,
          model_path,
          self._source_dtype,
          self.parallelism_config.ray_use_flashpack,
          custom_class_map,
        ) for actor in actors
      ])
      logger.info(f"Loaded pretrained pipelines on Ray workers in "
                  f"{time.perf_counter() - load_start:.2f}s.")
    elif transfer_backend == "object_store":
      logger.info("Putting the CPU pipeline into the Ray object store.")
      put_start = time.perf_counter()
      pipe_ref = self.ray.put(pipe)
      logger.info(f"Put the CPU pipeline into the Ray object store in "
                  f"{time.perf_counter() - put_start:.2f}s.")
      load_infos = self.ray.get([actor.load_pipeline.remote(pipe_ref) for actor in actors])
      logger.info(f"Loaded the object-store pipeline on Ray workers in "
                  f"{time.perf_counter() - load_start:.2f}s.")
    else:
      raise ValueError(f"Unsupported Ray pipeline transfer backend: {transfer_backend!r}.")
    logger.info(f"Ray pipeline worker placement after load: {load_infos}")
    return actors

  def _create_workers_with_init_fn(self, pipe: DiffusionPipeline, transfer_fn) -> list[Any]:
    """Create workers that each call *transfer_fn* to obtain their own pipeline.

    This bypasses all serialization/deserialization logic.  The function is passed
    via the Ray object store separately from the parallelism config so that it is
    never serialized as part of the actor constructor arguments.

    :param pipe: Main-process pipeline (used for metadata and device tracking only).
    :param transfer_fn: User-provided zero-argument function that returns a
      ``DiffusionPipeline`` on each worker.
    :returns: List of Ray actor handles after pipeline loading.
    """

    logger.info(
      "ray_transfer_fn is set; ray_transfer_backend=%r is ignored. "
      "Workers will call the user-provided initialize function.",
      self.parallelism_config.ray_transfer_backend,
    )
    # Strip ray_transfer_fn from the config copy sent to workers so Ray never
    # attempts to cloudpickle the function as part of the actor constructor args.
    worker_config = dataclasses.replace(self.parallelism_config, ray_transfer_fn=None)

    remote_worker = self.ray.remote(RayPipelineWorker)
    worker_options = {"num_gpus": 1}
    worker_options.update(worker_config.ray_worker_options)
    actors = [
      remote_worker.options(**worker_options).remote(
        rank,
        self.world_size,
        worker_config,
        self.cache_context_kwargs,
        self.quantize_config,
        self.master_port,
      ) for rank in range(self.world_size)
    ]
    self.ray.get([actor.ready.remote() for actor in actors])
    device_infos = self.ray.get([actor.device_info.remote() for actor in actors])
    logger.info(f"Ray pipeline worker placement before load: {device_infos}")

    # Offload the main-process pipeline to CPU to free GPU memory for workers.
    if self._source_device is not None and self._source_device.type != "cpu":
      logger.info("Moving the main-process pipeline to CPU before Ray worker loading.")
      offload_start = time.perf_counter()
      _move_pipeline_modules(pipe, "cpu")
      maybe_empty_cache()
      logger.info(f"Moved the main-process pipeline to CPU in "
                  f"{time.perf_counter() - offload_start:.2f}s.")
    else:
      logger.info("The main-process pipeline is already on CPU before Ray worker loading.")

    load_start = time.perf_counter()
    fn_ref = self.ray.put(transfer_fn)
    load_infos = self.ray.get([
      actor.load_pipeline_with_init_fn.remote(
        fn_ref,
        self.cache_context_kwargs,
        self.quantize_config,
      ) for actor in actors
    ])
    logger.info(f"Loaded pipelines via ray_transfer_fn on Ray workers in "
                f"{time.perf_counter() - load_start:.2f}s.")
    logger.info(f"Ray pipeline worker placement after load: {load_infos}")
    return actors

  def __call__(self, *args: Any, **kwargs: Any) -> Any:
    """Make the engine itself callable, proxying to :meth:`call`."""

    return self.call(*args, **kwargs)

  def call(self, *args: Any, **kwargs: Any) -> Any:
    """Proxy one full pipeline call through all Ray ranks.

    :param args: Positional arguments passed to the original pipeline call.
    :param kwargs: Keyword arguments passed to the original pipeline call.
    :returns: Rank-0 pipeline output.
    """

    cpu_args = cpu_tensor_tree(args)
    cpu_kwargs = cpu_tensor_tree(kwargs)
    results = self.ray.get([actor.call.remote(cpu_args, cpu_kwargs) for actor in self._actors])
    rank0_output = next((result for result in results if result is not None), None)
    if rank0_output is None:
      raise RuntimeError("Ray pipeline workers did not return a rank-0 output.")
    return rank0_output

  def shutdown(self) -> None:
    """Shutdown worker actors and optionally the Ray runtime initialized by this engine."""

    if self._actors:
      self.ray.get([actor.shutdown.remote() for actor in self._actors])
      for actor in self._actors:
        self.ray.kill(actor)
      self._actors = []
    if self._ray_initialized_by_engine and self.parallelism_config.ray_auto_shutdown:
      self.ray.shutdown()
    if self._source_pipe is not None and self._source_device is not None and self._source_device.type != "cpu":
      restore_start = time.perf_counter()
      _move_pipeline_modules(self._source_pipe, self._source_device)
      maybe_empty_cache()
      logger.info(f"Restored the main-process pipeline to {self._source_device} in "
                  f"{time.perf_counter() - restore_start:.2f}s.")
    if self._transfer_dir is not None and self._transfer_dir.exists():
      shutil.rmtree(self._transfer_dir)
      self._transfer_dir = None
