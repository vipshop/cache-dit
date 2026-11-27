"""CPU / layerwise / AdaLN offload application."""

import argparse
from typing import Any, Optional

import torch
from diffusers import DiffusionPipeline

from ..caching import BlockAdapter
from ..logger import init_logger
from ..offload import _default_module_filter
from ..offload import _find_offload_related_hf_hook
from ..offload import layerwise_cpu_offload
from ..offload import remove_layerwise_offload
from ..utils import is_modular_pipeline
from ..utils import resolve_pipe
from .parallel import get_rank_device

logger = init_logger(__name__)


def maybe_vae_tiling_or_slicing(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> DiffusionPipeline | BlockAdapter:
  if args.vae_tiling or args.vae_slicing:
    assert not args.parallel_vae, "VAE tiling/slicing is not compatible with VAE parallelism."

    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please enable VAE tiling/slicing manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    if hasattr(pipe, "vae"):
      vae = getattr(pipe, "vae", None)
      if vae is not None:
        vae_cls_name = vae.__class__.__name__
        if args.vae_tiling:
          if hasattr(vae, "enable_tiling"):
            logger.info(f"Enabling VAE tiling for module: {vae_cls_name} ...")
            vae.enable_tiling()
          else:
            logger.warning(f"Cannot enable VAE tiling for module: {vae_cls_name} No enable_tiling"
                           " method.")
        if args.vae_slicing:
          if hasattr(vae, "enable_slicing"):
            logger.info(f"Enabling VAE slicing for module: {vae_cls_name} ...")
            vae.enable_slicing()
          else:
            logger.warning(f"Cannot enable VAE slicing for module: {vae_cls_name} No enable_slicing"
                           " method.")
        setattr(pipe, "vae", vae)
      else:
        logger.warning("vae-tiling is set but no VAE found in the pipeline.")
  return pipe_or_adapter


def maybe_cpu_offload(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter,
) -> bool:
  _, device = get_rank_device()
  if args.cpu_offload or args.sequential_cpu_offload:
    if isinstance(pipe_or_adapter, BlockAdapter):
      pipe = pipe_or_adapter.pipe
      assert pipe is not None, "Please enable cpu offload manually if pipe is None."
    else:
      pipe = pipe_or_adapter

    pipe_cls_name = pipe.__class__.__name__
    if is_modular_pipeline(pipe):
      # diffusers implements the offload switches on `DiffusionPipeline` only, while a modular
      # pipeline owns its components directly. Stream the components layer by layer with
      # cache-dit's own offload instead of failing on the missing pipeline method.
      logger.info(f"{pipe_cls_name} is a modular pipeline and has no diffusers CPU offload API, "
                  "using cache-dit layerwise CPU offload for its components instead ...")
      # Pass the resolved pipe, not the adapter: an adapter would only expose the transformer and
      # the offloaded component set would silently depend on whether `--cache-params` is enabled.
      return _apply_layerwise_cpu_offload(args, pipe, strict=False)

    if args.sequential_cpu_offload:
      logger.info(f"Enabling Sequential CPU offload for the model {pipe_cls_name} ...")
      pipe.enable_sequential_cpu_offload(device=device)
    else:
      logger.info(f"Enabling CPU offload for the model {pipe_cls_name} ...")
      pipe.enable_model_cpu_offload(device=device)

    return True

  return False


def _append_generic_offload_target(
  targets: list[tuple[str, torch.nn.Module]],
  seen_module_ids: set[int],
  name: str,
  module: Any,
) -> None:
  if not isinstance(module, torch.nn.Module):
    return
  if id(module) in seen_module_ids:
    return
  seen_module_ids.add(id(module))
  targets.append((name, module))


def _resolve_generic_offload_targets(
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
) -> list[tuple[str, torch.nn.Module]]:
  targets: list[tuple[str, torch.nn.Module]] = []
  seen_module_ids: set[int] = set()

  if isinstance(pipe_or_adapter, torch.nn.Module):
    return [("module", pipe_or_adapter)]

  if isinstance(pipe_or_adapter, BlockAdapter):
    transformer = getattr(pipe_or_adapter, "transformer", None)
    if isinstance(transformer, list):
      for index, module in enumerate(transformer):
        _append_generic_offload_target(targets, seen_module_ids, f"transformer[{index}]", module)
    else:
      _append_generic_offload_target(targets, seen_module_ids, "transformer", transformer)
    if targets:
      return targets
    pipe_or_adapter = pipe_or_adapter.pipe

  components = getattr(pipe_or_adapter, "components", None)
  if isinstance(components, dict):
    # A pipeline may name the components that must stay resident: a conditioner whose forward calls a
    # hand-written kernel on a raw weight would otherwise be streamed and fail on a device mismatch.
    resident = tuple(
      getattr(pipe_or_adapter, "layerwise_offload_skip_components", None)
      or getattr(getattr(pipe_or_adapter, "pipe", None), "layerwise_offload_skip_components",
                 ()) or ())
    for name, module in components.items():
      if str(name) in resident:
        continue
      _append_generic_offload_target(targets, seen_module_ids, str(name), module)
    if targets:
      return targets

  if hasattr(pipe_or_adapter, "__dict__"):
    resident = tuple(
      getattr(pipe_or_adapter, "layerwise_offload_skip_components", None)
      or getattr(getattr(pipe_or_adapter, "pipe", None), "layerwise_offload_skip_components",
                 ()) or ())
    for name, module in vars(pipe_or_adapter).items():
      if str(name) in resident:
        continue
      _append_generic_offload_target(targets, seen_module_ids, str(name), module)

  return targets


def _is_text_encoder_offload_target(name: str) -> bool:
  return (name == "text_encoder" or name.startswith("text_encoder_")
          or name == "vision_language_encoder")


# Attribute names a vision-language conditioner can use for its vision tower.
_VISION_TOWER_ATTRS = ("visual", "vision_tower", "vision_model")


def _keep_vision_tower_resident(
  module: torch.nn.Module,
  device: torch.device,
) -> Optional[str]:
  """Move a conditioner's vision tower onto ``device`` and return its qualified name.

  A vision tower is small next to the language stack but cannot be streamed: its forward builds the
  interpolation constants from `self.pos_embed.weight.device` inside a plain method
  (`Qwen3VLVisionModel.fast_pos_embed_interpolate`), so a streamed tower puts them on the offload
  device and the first `hidden_states + pos_embeds` fails with a device mismatch. Keeping the tower
  resident and streaming only the decoder avoids that without touching the model.

  :param module: Conditioner being streamed.
  :param device: Device its forward runs on.
  :returns: Qualified name prefix of the tower, or None when the module has no vision tower.
  """
  named_modules = dict(module.named_modules())
  for holder_name, holder in (("", module), ("model", getattr(module, "model", None))):
    if not isinstance(holder, torch.nn.Module):
      continue
    for attr in _VISION_TOWER_ATTRS:
      tower = getattr(holder, attr, None)
      if not isinstance(tower, torch.nn.Module):
        continue
      prefix = f"{holder_name}.{attr}" if holder_name else attr
      if named_modules.get(prefix) is not tower:
        # An aliased attribute: the name we would exclude is not the path the offload discovery
        # walks, so the exclusion would silently do nothing.
        logger.warning(
          "Vision tower of %s is not reachable at '%s'; leaving it on the layerwise "
          "path.", module.__class__.__name__, prefix)
        continue
      tower.to(device)
      logger.info(
        "Keeping the vision tower %s of %s resident on %s (%.2f GiB): it derives its "
        "position-embedding constants from the parameter device, so a streamed tower would build "
        "them on the offload device.",
        prefix,
        module.__class__.__name__,
        device,
        sum(p.numel() * p.element_size() for p in tower.parameters()) / 2 ** 30,
      )
      return prefix
  return None


def _stream_filter_excluding(prefix: str):
  """Build a layerwise-offload predicate that keeps ``prefix`` and its subtree resident.

  :param prefix: Qualified name of the subtree to keep on the device.
  :returns: Predicate accepting the default leaf targets outside ``prefix``.
  """
  excluded = prefix

  def _filter(name: str, child: torch.nn.Module) -> bool:
    if name == excluded or name.startswith(excluded + "."):
      return False
    return _default_module_filter(name, child)

  return _filter


def _needs_vision_inputs(pipe_or_adapter: Any) -> bool:
  """Whether the conditioner of this pipeline is going to receive images or videos.

  The workflows that condition on frames (`fl2va`, `ref2va`) load an `image_processor`; the text-only
  workflow does not, because its preprocessing blocks are pruned away with the workflow. When the
  answer cannot be determined the tower is kept resident, which is the safe side.

  :param pipe_or_adapter: Pipeline, modular pipeline or adapter.
  :returns: Whether a vision tower has to stay resident.
  """
  pipe = resolve_pipe(pipe_or_adapter)
  if pipe is None:
    return True
  for name in ("image_processor", "video_processor"):
    if getattr(pipe, name, None) is not None:
      return True
  return False


def _layerwise_kwargs_for(
  args: argparse.Namespace,
  *,
  target_name: str,
  module: torch.nn.Module,
  onload_device: torch.device,
  keep_vision_resident: bool,
) -> dict[str, Any]:
  """Layerwise-offload kwargs for one component, including the vision-tower carve-out.

  Both streaming paths (`--module-layerwise-cpu-offload` and the conditioner of
  `--adaln-host-residency`) go through here, so a vision-language conditioner behaves the same way
  whichever switch the user picks.

  :param args: CLI arguments with the `layerwise_*` options.
  :param target_name: Component name the target was resolved under.
  :param module: Component about to be streamed.
  :param onload_device: Device its forward runs on.
  :param keep_vision_resident: Whether a vision tower has to stay on the device.
  :returns: Keyword arguments for `layerwise_cpu_offload`.
  """
  kwargs = _resolve_layerwise_target_kwargs(args,
                                            target_name=target_name,
                                            onload_device=onload_device)
  if keep_vision_resident and _is_text_encoder_offload_target(target_name):
    vision_prefix = _keep_vision_tower_resident(module, onload_device)
    if vision_prefix is not None:
      kwargs["module_filter"] = _stream_filter_excluding(vision_prefix)
  declared_owner = _declared_offload_owner(module)
  declared_filter = (getattr(type(declared_owner), "layerwise_offload_filter", None)
                     if declared_owner is not None else None)
  if declared_filter is not None and "module_filter" not in kwargs:
    # A model that streams subtrees instead of leaves has to say so: the default leaf discovery would
    # pin the nested parameter blocks on the compute device.
    kwargs["module_filter"] = declared_filter
  return kwargs


def _resolve_layerwise_target_arg(
  args: argparse.Namespace,
  *,
  target_name: str,
  base_attr: str,
  text_attr: str,
):
  base_value = getattr(args, base_attr)
  if not _is_text_encoder_offload_target(target_name):
    return base_value
  text_value = getattr(args, text_attr)
  return base_value if text_value is None else text_value


def _resolve_layerwise_target_kwargs(
  args: argparse.Namespace,
  *,
  target_name: str,
  onload_device: torch.device,
) -> dict[str, Any]:
  return {
    "onload_device":
    onload_device,
    "async_transfer":
    bool(
      _resolve_layerwise_target_arg(
        args,
        target_name=target_name,
        base_attr="layerwise_async_transfer",
        text_attr="layerwise_text_async_transfer",
      )),
    "transfer_buckets":
    int(
      _resolve_layerwise_target_arg(
        args,
        target_name=target_name,
        base_attr="layerwise_transfer_buckets",
        text_attr="layerwise_text_transfer_buckets",
      )),
    "prefetch_limit":
    bool(
      _resolve_layerwise_target_arg(
        args,
        target_name=target_name,
        base_attr="layerwise_prefetch_limit",
        text_attr="layerwise_text_prefetch_limit",
      )),
    "max_copy_streams":
    _resolve_layerwise_target_arg(
      args,
      target_name=target_name,
      base_attr="layerwise_max_copy_streams",
      text_attr="layerwise_text_max_copy_streams",
    ),
    "max_inflight_prefetch_bytes":
    _resolve_layerwise_target_arg(
      args,
      target_name=target_name,
      base_attr="layerwise_max_inflight_prefetch_bytes",
      text_attr="layerwise_text_max_inflight_prefetch_bytes",
    ),
    "persistent_buckets":
    int(
      _resolve_layerwise_target_arg(
        args,
        target_name=target_name,
        base_attr="layerwise_persistent_buckets",
        text_attr="layerwise_text_persistent_buckets",
      )),
    "persistent_bins":
    int(
      _resolve_layerwise_target_arg(
        args,
        target_name=target_name,
        base_attr="layerwise_persistent_bins",
        text_attr="layerwise_text_persistent_bins",
      )),
  }


def _apply_layerwise_cpu_offload(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
  *,
  strict: bool = True,
) -> bool:
  """Stream every discovered component through cache-dit's layerwise CPU offload.

  Shared by the `--module-layerwise-cpu-offload` switch and by the modular-pipeline fallback of
  `--cpu-offload` / `--sequential-cpu-offload` (diffusers has no pipeline-level offload API there).

  :param args: CLI arguments with the `layerwise_*` options.
  :param pipe_or_adapter: Pipeline, modular pipeline, `BlockAdapter`, or plain module.
  :param strict: Whether an empty target set is an error (`True`) or only a warning (`False`, used
    by the `--cpu-offload` fallback so the caller can still move the pipeline to the device).
  :returns: Whether at least one module is now covered by layerwise offload.
  """

  _, device = get_rank_device()
  targets = _resolve_generic_offload_targets(pipe_or_adapter)
  if not targets:
    if not strict:
      logger.warning("No nn.Module components were found for cache-dit layerwise CPU offload, "
                     "falling back to a plain pipeline move.")
      return False
    raise ValueError("Layerwise CPU offload was requested, but no nn.Module targets were found.")

  handles = []
  skipped_due_to_existing_offload = False
  for name, module in targets:
    remove_layerwise_offload(module)

    existing_offload_hook = _find_offload_related_hf_hook(module)
    if existing_offload_hook is not None:
      hook_module_name, hook = existing_offload_hook
      logger.warning(
        "Skipping cache-dit layerwise CPU offload for module %s because an existing accelerate "
        "offload hook %s is already registered on %s.",
        name,
        type(hook).__name__,
        hook_module_name or "<root>",
      )
      skipped_due_to_existing_offload = True
      continue

    logger.info(
      "Enabling cache-dit layerwise CPU offload for module %s: %s ...",
      name,
      module.__class__.__name__,
    )
    handles.append(
      layerwise_cpu_offload(
        module,
        **_layerwise_kwargs_for(
          args,
          target_name=name,
          module=module,
          onload_device=device,
          keep_vision_resident=_needs_vision_inputs(pipe_or_adapter),
        ),
      ))
    _apply_declared_residency(module, device)
  return bool(handles) or skipped_due_to_existing_offload


def _declared_offload_owner(module: torch.nn.Module) -> torch.nn.Module | None:
  """Return the module that may declare MAGI-style offload hooks, if any.

  A model whose streaming unit is a *subtree* (a whole transformer layer) rather than a leaf has to
  declare that itself, and it also has to move the parts that are never streamed onto the compute
  device. Both are optional model-declared hooks: `layerwise_offload_filter` and `resident_to`.

  :param module: Component about to be streamed.
  :returns: The module carrying the declarations, or ``None``.
  """

  for candidate in (module, getattr(module, "transformer", None)):
    if candidate is not None and (hasattr(type(candidate), "layerwise_offload_filter")
                                  or hasattr(candidate, "resident_to")):
      return candidate
  return None


def _apply_declared_residency(module: torch.nn.Module, device: torch.device) -> None:
  """Move a model's non-streamed parts onto the compute device after offload is installed.

  :param module: Component that was just put under layerwise offload.
  :param device: Device the streamed module runs on.
  """

  for candidate in (module, getattr(module, "transformer", None)):
    resident_to = getattr(candidate, "resident_to", None)
    if resident_to is not None:
      resident_to(device)
      return


def _adaln_offload_target_names(transformer: torch.nn.Module) -> list[str]:
  """Names of the AdaLN modulation projections of a transformer.

  :param transformer: Transformer module to scan.
  :returns: Qualified names of every `Linear` that sits under an `adaln_proj` submodule.
  """
  return [
    name for name, module in transformer.named_modules()
    if isinstance(module, torch.nn.Linear) and "adaln_proj" in name
  ]


def _move_to_device_without_adaln(module: torch.nn.Module, device: torch.device) -> None:
  """Move every leaf of `module` to `device`, leaving the `adaln_proj` tables on the host.

  `module.to(device)` would take the whole 62 GiB of a MiniMax-H3 checkpoint up even when the
  memoized AdaLN tables make 24 GiB of it dead weight; moving the leaves one by one keeps the device
  peak at the resident size. Only used without tensor parallelism and quantization, where every leaf
  is a plain tensor.

  :param module: Transformer whose leaves are moved.
  :param device: Execution device.
  """
  for name, parameter in module.named_parameters():
    if "adaln_proj" not in name:
      parameter.data = parameter.data.to(device)
  for name, buffer in module.named_buffers():
    if "adaln_proj" not in name:
      buffer.data = buffer.data.to(device)


def maybe_adaln_host_residency(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
) -> bool:
  """Stream only the AdaLN tables of the transformer and keep the rest of it resident.

  Every block of a joint video + audio DiT rebuilds its modulation from the current timestep, and
  those tables can be a large share of the transformer (24.2 GiB of MiniMax-H3's 62 GiB). Moving the
  whole transformer in and out on every step, as `--module-layerwise-cpu-offload` does, copies every
  weight of the model per step; streaming only the AdaLN tables pays for their share and leaves the
  attention / FFN weights resident. The text encoder, which does not fit either way, stays on the
  full layerwise path, and the remaining components simply move to the device.

  With `--adaln-online` the same placement applies, but the AdaLN tables are memoized for the
  checkpoint's fixed sampling grid instead of streamed: the tables are computed once with the
  weights on the device, kept (a few MiB), and the 24 GiB of projections are parked on the host, so
  no host-to-device traffic happens per step at all.

  :param args: CLI arguments with the `layerwise_*` options.
  :param pipe_or_adapter: Pipeline or adapter to modify in place.
  :returns: Whether an AdaLN residency scheme is now installed.
  """

  online = getattr(args, "adaln_online", False)
  if not getattr(args, "adaln_host_residency", False) and not online:
    return False
  if online and getattr(args, "adaln_host_residency", False):
    raise ValueError("--adaln-online and --adaln-host-residency both own the `adaln_proj` weights; "
                     "pass one of the two.")
  if online and (getattr(args, "cpu_offload", False)
                 or getattr(args, "sequential_cpu_offload", False)
                 or getattr(args, "module_layerwise_cpu_offload", False)):
    raise ValueError("--adaln-online keeps the transformer resident by design, and the generic "
                     "offload switches stream every block again; pass one of them, not both.")
  if getattr(args, "device_map_balance", False):
    raise ValueError("--adaln-host-residency/--adaln-online cannot be combined with "
                     "--device-map-balance.")

  _, device = get_rank_device()
  targets = _resolve_generic_offload_targets(pipe_or_adapter)
  if not targets:
    raise ValueError("--adaln-host-residency/--adaln-online found no nn.Module components to "
                     "manage.")

  # Tensor parallelism already divides the AdaLN tables across the ranks, and a sharded table is a
  # DTensor, which the offload mirror cannot stream per tensor: it would fall back to moving the
  # whole module on every block. A quantized transformer is small enough to stay resident, so
  # streaming a bf16 AdaLN table out of it would cost more than it saves. In both cases only the
  # text encoder is streamed.
  tp_enabled = "tp" in (getattr(args, "parallel_type", "") or "")
  keep_transformer_resident = tp_enabled or getattr(args, "quantize", False)
  if online and getattr(args, "cuda_graph", False):
    # The plan lookup keys on the timestep embedding's bytes, i.e. a device-to-host copy on every
    # step, which a capture cannot replay.
    raise ValueError("--adaln-online cannot be combined with --cuda-graph: keying the plan copies "
                     "the timestep embedding to the host on every step.")
  if online and getattr(args, "compile_full_graph", False):
    raise ValueError("--adaln-online cannot be combined with --compile-full-graph: keying the plan "
                     "is a graph break, which `fullgraph=True` refuses by construction.")
  if getattr(args, "cuda_graph", False) and not keep_transformer_resident:
    # CUDA Graph capture bakes the parameter pointers into the graph, and the host mirror rebinds
    # `parameter.data` on every block, so the capture would replay stale device buffers.
    raise ValueError("--adaln-host-residency cannot be combined with --cuda-graph unless the "
                     "transformer stays resident (transformer quantization or `--parallel tp`).")
  if getattr(args, "compile_full_graph", False) and not keep_transformer_resident:
    # `fullgraph=True` refuses the graph break the streaming hook causes by construction.
    raise ValueError(
      "--adaln-host-residency cannot be combined with --compile-full-graph unless "
      "the transformer stays resident (transformer quantization or `--parallel tp`).")
  if getattr(args, "compile", False) and online:
    logger.info(
      "Compiling the repeated blocks with the AdaLN plans on the device: keying a plan is "
      "a per-block graph break, so each block compiles around it.")
  elif getattr(args, "compile", False) and not keep_transformer_resident:
    # The device move happens inside the layerwise hooks, which are `torch.compiler.disable`d, so
    # each streamed block shows up as a graph break instead of an error: the compiled region is
    # everything around it. That still pays off — the attention / FFN bodies are the bulk of the
    # block — but the win is smaller than for a resident transformer.
    logger.info("Streaming the AdaLN tables of the transformer while compiling: the per-block "
                "host-to-device copy is a graph break, so each block compiles around it.")

  def _is_transformer(name: str, module: torch.nn.Module) -> bool:
    if name.startswith("transformer"):
      return True
    # A bare transformer handed to the CLI resolves to a single ("module", module) target
    return len(targets) == 1 and name == "module" and bool(_adaln_offload_target_names(module))

  # The transformer has to be on the device before any handle is installed: the pinned host mirror
  # rebinds `parameter.data`, so a `module.to()` afterwards would copy those tables back and leave
  # the mirror stale. `--parallel tp` only places the sharded projections, so the replicated parts
  # (`proj_in`, `token_refiner`, the rope buffer) still need this move.
  for name, module in targets:
    if _is_transformer(name, module):
      if online and not keep_transformer_resident:
        _move_to_device_without_adaln(module, device)
      else:
        module.to(device)

  installed = False
  for name, module in targets:
    if not _is_transformer(name, module):
      continue
    if online:
      from ..offload.adaln_plan import (
        adaln_plan_tembeddings,
        install_adaln_plan_cache,
        timestep_sets_from_pipeline,
      )

      pipe = pipe_or_adapter.pipe if isinstance(pipe_or_adapter, BlockAdapter) else pipe_or_adapter
      timestep_sets = timestep_sets_from_pipeline(pipe)
      embeddings = adaln_plan_tembeddings(module, timestep_sets)
      plans, tables = install_adaln_plan_cache(module, embeddings, compute_device=device)
      logger.info(
        "AdaLN online for module %s: %d plans over %d modulation tables for %d "
        "timestep vectors; projection weights that are not tensor-parallel shards are "
        "parked on the host.", name, plans, tables, len(timestep_sets))
      installed = True
      continue
    if keep_transformer_resident:
      logger.info(
        "Keeping the transformer %s resident (%s): streaming the text encoder only.", name,
        "tensor parallelism already shards its AdaLN tables"
        if tp_enabled else "quantized weights are small enough to stay on the device")
      installed = True
      continue
    target_names = _adaln_offload_target_names(module)
    if not target_names:
      continue
    logger.info(
      "Enabling cache-dit AdaLN host residency for module %s: streaming %d modulation tables ...",
      name,
      len(target_names),
    )
    layerwise_cpu_offload(
      module,
      module_names=target_names,
      **_resolve_layerwise_target_kwargs(args, target_name=name, onload_device=device),
    )
    installed = True

  if not installed:
    raise ValueError("--adaln-host-residency needs a transformer with `adaln_proj` modulation "
                     "tables (or, under TP, any transformer); none of the resolved components has "
                     "one.")

  for name, module in targets:
    if _is_transformer(name, module):
      continue
    if _is_text_encoder_offload_target(name):
      logger.info(
        "Enabling cache-dit layerwise CPU offload for module %s: %s ...",
        name,
        module.__class__.__name__,
      )
      layerwise_cpu_offload(
        module,
        **_layerwise_kwargs_for(
          args,
          target_name=name,
          module=module,
          onload_device=device,
          keep_vision_resident=_needs_vision_inputs(pipe_or_adapter),
        ),
      )
    else:
      module.to(device)
  return True


def maybe_generic_module_offload(
  args,
  pipe_or_adapter: DiffusionPipeline | BlockAdapter | torch.nn.Module | Any,
) -> bool:
  if not getattr(args, "module_layerwise_cpu_offload", False):
    return False
  return _apply_layerwise_cpu_offload(args, pipe_or_adapter)
