from __future__ import annotations

import socket
import shutil
import uuid
import time
from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from diffusers.models.modeling_utils import ModelMixin

from ..distributed import ParallelismConfig
from ..logger import init_logger
from ..utils import maybe_empty_cache
from ._tree import cpu_tensor_tree
from ._tree import device_tensor_tree
from ._tree import first_tensor_device
from .worker import RayTransformerWorker
from .worker import RayPipelineWorker

logger = init_logger(__name__)


def _require_ray():
  try:
    import ray
  except ImportError as exc:
    raise ImportError("Ray wrapper requires ray. Install it with `pip install ray` or "
                      "`pip install cache-dit[ray]`.") from exc
  return ray


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


class RayParallelEngine:
  """Main-process engine that dispatches transformer forwards to Ray actors.

  :param transformer: User-visible transformer whose forward will be proxied.
  :param parallelism_config: Ray-enabled parallelism configuration.
  """

  def __init__(
    self,
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
  ):
    self.ray = _require_ray()
    self.parallelism_config = parallelism_config
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
      self.ray.init(**init_kwargs)
      self._ray_initialized_by_engine = True

    self.master_port = parallelism_config.ray_master_port or _get_free_port()
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
        load_infos = self.ray.get([
          actor.load_transformer_from_pretrained.remote(
            transformer.__class__,
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
        load_infos = self.ray.get([
          actor.load_transformer_from_safetensors.remote(
            transformer.__class__,
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
    pipe: DiffusionPipeline,
    parallelism_config: ParallelismConfig,
  ):
    if not hasattr(pipe, "transformer"):
      raise ValueError("Ray pipeline parallelism requires a pipeline with a transformer "
                       "attribute.")
    self.ray = _require_ray()
    self.parallelism_config = parallelism_config
    self._source_pipe = pipe
    self._source_device = _first_pipeline_module_device(pipe)
    self._source_dtype = _first_pipeline_module_dtype(pipe)
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
      self.ray.init(**init_kwargs)
      self._ray_initialized_by_engine = True

    self.master_port = parallelism_config.ray_master_port or _get_free_port()
    self._actors = self._create_workers(pipe)

  def _create_workers(self, pipe: DiffusionPipeline) -> list[Any]:
    remote_worker = self.ray.remote(RayPipelineWorker)
    worker_options = {"num_gpus": 1}
    worker_options.update(self.parallelism_config.ray_worker_options)
    actors = [
      remote_worker.options(**worker_options).remote(
        rank,
        self.world_size,
        self.parallelism_config,
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
      load_infos = self.ray.get([
        actor.load_pipeline_from_pretrained.remote(
          pipe.__class__,
          str(pipeline_path),
          self._source_dtype,
          self.parallelism_config.ray_use_flashpack,
        ) for actor in actors
      ])
      logger.info(f"Loaded saved pipeline snapshots on Ray workers in "
                  f"{time.perf_counter() - load_start:.2f}s.")
    elif transfer_backend == "from_pretrained":
      if model_path is None:
        raise ValueError("ray_transfer_backend='from_pretrained' requires pipeline.name_or_path.")
      logger.info(f"Loading Ray worker pipelines from pretrained source: {model_path}.")
      load_infos = self.ray.get([
        actor.load_pipeline_from_pretrained.remote(
          pipe.__class__,
          model_path,
          self._source_dtype,
          self.parallelism_config.ray_use_flashpack,
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
    if self._source_device is not None and self._source_device.type != "cpu":
      restore_start = time.perf_counter()
      _move_pipeline_modules(self._source_pipe, self._source_device)
      maybe_empty_cache()
      logger.info(f"Restored the main-process pipeline to {self._source_device} in "
                  f"{time.perf_counter() - restore_start:.2f}s.")
    if self._transfer_dir is not None and self._transfer_dir.exists():
      shutil.rmtree(self._transfer_dir)
      self._transfer_dir = None
