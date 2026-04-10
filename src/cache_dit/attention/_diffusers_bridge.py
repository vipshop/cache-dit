from ._attention_dispatch import _resolve_cp_config
from ..logger import init_logger
from ._attention_dispatch import _AttnBackend, _AttnBackendRegistry

logger = init_logger(__name__)


def _pop_diffusers_attn_backend(diffusers_registry, attn_backend) -> None:
  diffusers_registry._backends.pop(attn_backend, None)
  diffusers_registry._constraints.pop(attn_backend, None)
  diffusers_registry._supported_arg_names.pop(attn_backend, None)
  if isinstance(diffusers_registry._supports_context_parallel, dict):
    diffusers_registry._supports_context_parallel.pop(attn_backend, None)
  elif attn_backend.value in diffusers_registry._supports_context_parallel:
    diffusers_registry._supports_context_parallel.remove(attn_backend.value)


def _ensure_diffusers_attn_backend(diffusers_backend_enum, backend: _AttnBackend):
  try:
    return diffusers_backend_enum(backend.value)
  except ValueError:
    new_member = str.__new__(diffusers_backend_enum, backend.value)
    new_member._name_ = backend.name
    new_member._value_ = backend.value
    setattr(diffusers_backend_enum, backend.name, new_member)
    diffusers_backend_enum._member_map_[backend.name] = new_member
    diffusers_backend_enum._member_names_.append(backend.name)
    diffusers_backend_enum._value2member_map_[backend.value] = new_member
    return new_member


def _set_diffusers_context_parallel_support(diffusers_registry, attn_backend,
                                            supported: bool) -> None:
  if isinstance(diffusers_registry._supports_context_parallel, dict):
    if supported:
      diffusers_registry._supports_context_parallel[attn_backend] = True
    else:
      diffusers_registry._supports_context_parallel.pop(attn_backend, None)
    return

  if supported:
    diffusers_registry._supports_context_parallel.add(attn_backend.value)
  else:
    diffusers_registry._supports_context_parallel.discard(attn_backend.value)


def _register_cache_dit_attn_backends_to_diffusers() -> list[str]:
  """Register cache-dit backends into diffusers' attention registry."""

  from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

  registered_backends: list[str] = []

  for backend in _AttnBackendRegistry.list_bridge_backends():
    backend_fn = _AttnBackendRegistry.get_backend(backend)
    if backend_fn is None:
      continue

    def _make_diffusers_backend_wrapper(cache_dit_backend_fn):

      def _diffusers_backend_wrapper(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        _parallel_config=None,
        **kwargs,
      ):
        return cache_dit_backend_fn(
          query=query,
          key=key,
          value=value,
          attn_mask=attn_mask,
          dropout_p=dropout_p,
          is_causal=is_causal,
          scale=scale,
          enable_gqa=enable_gqa,
          _cp_config=_resolve_cp_config(_parallel_config=_parallel_config),
          **kwargs,
        )

      _diffusers_backend_wrapper.__name__ = cache_dit_backend_fn.__name__
      return _diffusers_backend_wrapper

    diffusers_backend_fn = _make_diffusers_backend_wrapper(backend_fn)

    diffusers_backend = _ensure_diffusers_attn_backend(AttentionBackendName, backend)
    _pop_diffusers_attn_backend(_AttentionBackendRegistry, diffusers_backend)

    _AttentionBackendRegistry._backends[diffusers_backend] = diffusers_backend_fn
    _AttentionBackendRegistry._constraints[diffusers_backend] = list(
      _AttnBackendRegistry.get_constraints(backend))
    _AttentionBackendRegistry._supported_arg_names[diffusers_backend] = {
      "_parallel_config" if arg_name == "_cp_config" else arg_name
      for arg_name in _AttnBackendRegistry.get_supported_arg_names(backend)
    }
    _set_diffusers_context_parallel_support(
      _AttentionBackendRegistry,
      diffusers_backend,
      _AttnBackendRegistry.is_context_parallel_available(backend),
    )
    registered_backends.append(backend.value)

  logger.debug(f"Bridged cache-dit attention backends to diffusers: {registered_backends}")
  return registered_backends
