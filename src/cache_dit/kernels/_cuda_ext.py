"""Single loader for the optional ``cache_dit._C_ops`` CUDA extension.

Every kernel family (communication, attention, SVDQuant, model-specific) is linked into one
module, so availability is a single fact rather than one per family. Both the module and the
import error are cached: a missing ``.so`` must not fail ``import cache_dit``, it must fail the
first operator call with a message naming the build command and the architecture floor.
"""

import importlib
from types import ModuleType

_EXTENSION_MODULE_NAME = "cache_dit._C_ops"
_cached_extension_module: ModuleType | None = None
_cached_load_error: Exception | None = None


def _load_extension_module() -> ModuleType | None:
  """Import the optional CUDA extension at most once.

  :returns: The loaded extension module, or `None` when the import failed and the failure
    was cached.
  """

  global _cached_extension_module, _cached_load_error

  if _cached_extension_module is not None:
    return _cached_extension_module
  if _cached_load_error is not None:
    return None

  try:
    _cached_extension_module = importlib.import_module(_EXTENSION_MODULE_NAME)
  except Exception as exc:  # pragma: no cover - exercised in environments without the extension
    _cached_load_error = exc
    return None

  return _cached_extension_module


def cuda_ext_is_available() -> bool:
  """Return whether the optional Cache-DiT CUDA extension can be imported.

  This only reports "compiled in", not "runnable on this device": the module ships cubins for
  the architectures the build targeted (sm89+ excluding sm90 by default), so a GPU outside that
  set imports fine and fails at the first kernel launch. Callers that need a hard guarantee
  must compare the device capability with the build's `CACHE_DIT_CUDA_ARCH_LIST` themselves.

  :returns: `True` when the compiled `cache_dit._C_ops` module is available.
  """

  return _load_extension_module() is not None


def cuda_ext_get_load_error() -> Exception | None:
  """Return the cached import error of the optional Cache-DiT CUDA extension.

  :returns: The import error raised while loading the extension, or `None` when the
    extension loaded successfully.
  """

  _load_extension_module()
  return _cached_load_error


def _get_required_module() -> ModuleType:
  """Return the loaded extension module or raise a guided runtime error.

  :returns: The loaded Python extension module backing the Cache-DiT CUDA ops.
  :raises RuntimeError: If the optional extension is unavailable.
  """

  extension_module = _load_extension_module()
  if extension_module is None:
    raise RuntimeError(
      "The optional Cache-DiT CUDA extension is not available. Build it with "
      "`CACHE_DIT_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation` after "
      "`conda activate cdit` and `git submodule update --init --recursive`. Every kernel "
      "family ships in the single `cache_dit._C_ops` module, which needs sm89+ and excludes "
      "sm90 (Hopper), so a build that printed a skipped-SM warning omits all CUDA operators: "
      "rerun it with CACHE_DIT_CUDA_ARCH_LIST set to a supported target."
    ) from cuda_ext_get_load_error()
  return extension_module


def _get_required_ops_module() -> ModuleType:
  """Return the extension's `ops` submodule used by the Python wrappers.

  :returns: The `ops` submodule exported by the loaded extension.
  :raises RuntimeError: If the extension does not expose `ops`.
  """

  ops_module = getattr(_get_required_module(), "ops", None)
  if ops_module is None:
    raise RuntimeError("The loaded Cache-DiT CUDA extension does not expose an `ops` submodule.")
  return ops_module


__all__ = [
  "cuda_ext_get_load_error",
  "cuda_ext_is_available",
]
