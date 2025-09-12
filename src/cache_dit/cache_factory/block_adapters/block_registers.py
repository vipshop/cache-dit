from typing import Any, Tuple, List, Dict, Callable

from diffusers import DiffusionPipeline
from cache_dit.cache_factory.block_adapters.block_adapters import BlockAdapter

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class BlockAdapterRegistry:
    _adapters: Dict[str, Callable[..., BlockAdapter]] = {}
    _predefined_adapters_has_separate_cfg: List[str] = [
        "QwenImage",
        "Wan",
        "CogView4",
        "Cosmos",
        "SkyReelsV2",
        "Chroma",
        "Lumina2",
    ]

    @classmethod
    def register(cls, name: str, supported: bool = True):
        def decorator(
            func: Callable[..., BlockAdapter]
        ) -> Callable[..., BlockAdapter]:
            if supported:
                cls._adapters[name] = func
            return func

        return decorator

    @classmethod
    def get_adapter(
        cls,
        pipe: DiffusionPipeline | str | Any,
        **kwargs,
    ) -> BlockAdapter:
        if not isinstance(pipe, str):
            pipe_cls_name: str = pipe.__class__.__name__
        else:
            pipe_cls_name = pipe

        for name in cls._adapters:
            if pipe_cls_name.startswith(name):
                return cls._adapters[name](pipe, **kwargs)

        return BlockAdapter()

    @classmethod
    def has_separate_cfg(
        cls,
        pipe_or_adapter: DiffusionPipeline | BlockAdapter | Any,
    ) -> bool:

        # Prefer custom setting from block adapter.
        if isinstance(pipe_or_adapter, BlockAdapter):
            return pipe_or_adapter.has_separate_cfg

        has_separate_cfg = False
        if isinstance(pipe_or_adapter, DiffusionPipeline):
            has_separate_cfg = cls.get_adapter(
                pipe_or_adapter,
                skip_post_init=True,  # check cfg setting only
            ).has_separate_cfg

        if has_separate_cfg:
            return True

        pipe_cls_name = pipe_or_adapter.__class__.__name__
        for name in cls._predefined_adapters_has_separate_cfg:
            if pipe_cls_name.startswith(name):
                return True

        return False

    @classmethod
    def is_supported(cls, pipe) -> bool:
        pipe_cls_name: str = pipe.__class__.__name__

        for name in cls._adapters:
            if pipe_cls_name.startswith(name):
                return True
        return False

    @classmethod
    def supported_pipelines(cls, **kwargs) -> Tuple[int, List[str]]:
        val_pipelines = cls._adapters.keys()
        return len(val_pipelines), [p + "*" for p in val_pipelines]
