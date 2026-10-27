"""Base class of the checkpoint loaders that hook a pipeline's `from_pretrained`.

A loader exists for a checkpoint whose stored weights no `from_pretrained` route can map onto the
module parameters. It converts them, hands the result to `from_pretrained` as ready-made
components, and leaves every other component on `diffusers`' own loading path.

The hook belongs to the pipeline class rather than to one call, so it outlives the code that
installed it: every later `from_pretrained` on that class, and on any subclass, goes through it.

Subclasses implement `_load`, which receives the resolved checkpoint path plus the keyword
arguments of the call and returns the components to inject — or `None` to let the original
`from_pretrained` run untouched.
"""

import abc
import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional

_FROM_PRETRAINED = "from_pretrained"
# Attribute a patch leaves on the class it patched, holding the loader that installed it. Read back
# through `cls.__dict__`, so a subclass that only inherits a wrapper is not mistaken for a class
# that was patched itself.
LOADER_PATCH_ATTR = "__cache_dit_loader__"
# Tag on the installed wrapper itself, holding the loader class that built it. Read through
# `getattr`, so a wrapper inherited from a patched base class still counts as a patch and is never
# wrapped a second time.
LOADER_WRAPPER_ATTR = "__cache_dit_loader_class__"


def _installed_wrapper(cls: type) -> Optional[Callable[..., Any]]:
  """The loader wrapper `cls.from_pretrained` resolves to, or `None`.

  Resolution goes through `getattr`, not `cls.__dict__`, because the wrapper is inherited: a
  subclass calling `Sub.from_pretrained` already runs the wrapper its base class installed.

  :param cls: The pipeline class to inspect.
  :returns: The installed wrapper, or `None` when `from_pretrained` is an unpatched one.
  """
  method = getattr(cls, _FROM_PRETRAINED, None)
  func = getattr(method, "__func__", method)
  if getattr(func, LOADER_WRAPPER_ATTR, None) is None:
    return None
  return func


class BaseLoader(abc.ABC):
  """Base class of the loaders that convert a checkpoint inside `from_pretrained`.

  `apply` patches the pipeline class in place and returns that same class, so a caller reads as
  `SomeLoader().apply(SomePipeline)`. Passing an instance patches its class, which is what makes
  the patch effective for the pipelines built afterwards.

  The patch is idempotent by class, not by loader instance: a class whose `from_pretrained`
  already resolves to a loader wrapper — its own, or one inherited from a patched base class —
  is returned as is. Stacking a second wrapper would convert the checkpoint twice.

  A pipeline class carries at most one loader. `remove` undoes the patch of the loader that
  installed it and refuses to touch a wrapper somebody else put in place, and it restores
  inheritance with a `delattr` when the patched class had no `from_pretrained` of its own: freezing
  the saved bound method into the class would bind every subclass call to the patched class.
  """

  def __init__(self) -> None:
    self._patched_class: Optional[type] = None
    self._original: Optional[Callable[..., Any]] = None
    self._wrapper: Optional[Callable[..., Any]] = None
    self._owns_from_pretrained = False

  @property
  def original(self) -> Optional[Callable[..., Any]]:
    """The `from_pretrained` the patch wraps, or `None` before `apply` (read-only).

    :returns: The classmethod that was inherited when `apply` ran, bound to the patched class. The
      wrapper re-binds its `__func__` to the class each call arrives on, which keeps a subclass a
      subclass instead of degrading it to the patched class.
    """
    return self._original

  @property
  def wrapper(self) -> Optional[Callable[..., Any]]:
    """The function installed in place of `from_pretrained` (read-only).

    :returns: The plain function that `apply` installed as a `classmethod`, or `None` before
      `apply` and after `remove`.
    """
    return self._wrapper

  @property
  def patched_class(self) -> Optional[type]:
    """The class `apply` patched (read-only).

    :returns: The patched class, or `None` before `apply` and after `remove`.
    """
    return self._patched_class

  def apply(self, pipeline_or_cls: Any) -> type:
    """Patch `from_pretrained` on a pipeline class, in place.

    :param pipeline_or_cls: The pipeline class, or an instance of it.
    :returns: The patched class, i.e. `pipeline_or_cls` when that already is a class and its type
      otherwise.
    """
    cls = pipeline_or_cls if isinstance(pipeline_or_cls, type) else type(pipeline_or_cls)
    installed = _installed_wrapper(cls)
    if installed is not None:
      # Idempotent by class: the same loader applied twice, a second loader instance, or a
      # subclass that inherited the wrapper all end up here, and none of them may add a wrapper.
      installed_loader = getattr(installed, LOADER_WRAPPER_ATTR)
      if installed_loader is not type(self):
        raise RuntimeError(
          f"{cls.__name__}.from_pretrained is already patched by {installed_loader.__name__}; "
          f"cannot also apply {type(self).__name__}.")
      return cls
    if self._patched_class is not None and self._patched_class is not cls:
      raise RuntimeError(
        f"{type(self).__name__} already patched {self._patched_class.__name__}; create another "
        f"loader to patch {cls.__name__} as well.")
    original = getattr(cls, _FROM_PRETRAINED, None)
    if not inspect.ismethod(original):
      raise TypeError(
        f"{cls.__name__}.from_pretrained must be a classmethod for {type(self).__name__} to wrap "
        f"it, got {original!r}.")
    wrapper = self._make_wrapper(original)
    self._owns_from_pretrained = _FROM_PRETRAINED in cls.__dict__
    self._patched_class = cls
    self._original = original
    self._wrapper = wrapper
    setattr(cls, _FROM_PRETRAINED, classmethod(wrapper))
    setattr(cls, LOADER_PATCH_ATTR, self)
    return cls

  def remove(self) -> None:
    """Undo this loader's patch.

    Only the class this loader patched is touched, and only while its `__dict__` still holds the
    wrapper this loader installed; a wrapper that somebody replaced in the meantime is left alone.
    """
    cls = self._patched_class
    if cls is None:
      return
    installed = cls.__dict__.get(_FROM_PRETRAINED)
    if isinstance(installed, classmethod) and installed.__func__ is self._wrapper:
      if self._owns_from_pretrained:
        setattr(cls, _FROM_PRETRAINED, classmethod(self._original.__func__))
      else:
        delattr(cls, _FROM_PRETRAINED)
    if cls.__dict__.get(LOADER_PATCH_ATTR) is self:
      delattr(cls, LOADER_PATCH_ATTR)
    self._forget()

  @abc.abstractmethod
  def _load(self, path: str, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert a checkpoint into the components to inject into `from_pretrained`.

    :param path: The checkpoint path or repository id of the call, as a string.
    :param kwargs: Keyword arguments of the call, e.g. the requested dtype and the hub selectors.
    :returns: Components to inject, keyed by component name, or `None` to leave the call
      completely untouched.
    """

  def _make_wrapper(self, original: Callable[..., Any]) -> Callable[..., Any]:
    """Build the `classmethod` body installed in place of `original`.

    `functools.wraps` carries the name and the `__wrapped__` function over, so `inspect.signature`
    of the patched `from_pretrained` still reports the signature of the original.

    :param original: The classmethod being replaced, bound to the patched class.
    :returns: The function to install as a `classmethod`.
    """
    original_func = original.__func__

    @functools.wraps(original_func)
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
      return self._dispatch(original_func, cls, pretrained_model_name_or_path, args, kwargs)

    setattr(from_pretrained, LOADER_WRAPPER_ATTR, type(self))
    return from_pretrained

  def _dispatch(
    self,
    original_func: Callable[..., Any],
    cls: type,
    pretrained_model_name_or_path: Any,
    args: tuple,
    kwargs: Dict[str, Any],
  ) -> Any:
    """Run `_load` and forward the call with whatever it asks to inject.

    :param original_func: The unbound `from_pretrained` this patch replaced.
    :param cls: The class the call arrived on, which may be a subclass of the patched class.
    :param pretrained_model_name_or_path: The checkpoint path or repository id of the call.
    :param args: Positional arguments after the checkpoint path.
    :param kwargs: Keyword arguments of the call.
    :returns: Whatever the original `from_pretrained` returns.
    """
    path = os.fspath(pretrained_model_name_or_path)
    components = self._load(path, kwargs)
    if components:
      # The caller's keyword arguments win: a component it passed itself — an explicit `None`,
      # which means "leave this component unloaded", included — is never replaced by a converted
      # one.
      merged = dict(components)
      merged.update(kwargs)
      kwargs = merged
    return original_func(cls, pretrained_model_name_or_path, *args, **kwargs)

  def _forget(self) -> None:
    """Drop the record of the patch, so this loader can be applied to a class again."""
    self._patched_class = None
    self._original = None
    self._wrapper = None
    self._owns_from_pretrained = False
