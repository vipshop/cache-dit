import torch
from typing import Any, Tuple, List, Dict, Callable, Union

from diffusers import DiffusionPipeline
from .block_adapters import (
    BlockAdapter,
    FakeDiffusionPipeline,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class BlockAdapterRegister:
    _adapters: Dict[str, Callable[..., BlockAdapter]] = {}
    _predefined_adapters_has_separate_cfg: List[str] = [
        "QwenImage",
        "Wan",
        "CogView4",
        "Cosmos",
        "SkyReelsV2",
        "Chroma",
        "Lumina2",
        "Kandinsky5",
        "ChronoEdit",
        "HunyuanVideo15",
        "OvisImage",
    ]

    @classmethod
    def register(cls, name: str, supported: bool = True):
        def decorator(func: Callable[..., BlockAdapter]) -> Callable[..., BlockAdapter]:
            if supported:
                cls._adapters[name] = func
            return func

        return decorator

    @classmethod
    def get_adapter(
        cls,
        pipe_or_module: DiffusionPipeline | torch.nn.Module | str | Any,
        **kwargs,
    ) -> BlockAdapter | None:
        if not isinstance(pipe_or_module, str):
            cls_name: str = pipe_or_module.__class__.__name__
        else:
            cls_name = pipe_or_module

        for name in cls._adapters:
            if cls_name.startswith(name):
                if not isinstance(pipe_or_module, DiffusionPipeline):
                    assert isinstance(pipe_or_module, torch.nn.Module)
                    # NOTE: Make pre-registered adapters support Transformer-only case.
                    # WARN: This branch is not officially supported and only for testing
                    # purpose. We construct a fake diffusion pipeline that contains the
                    # given transformer module. Currently, only works for DiT models which
                    # only have one transformer module. Case like multiple transformers
                    # is not supported, e.g, Wan2.2. Please use BlockAdapter directly for
                    # such cases.
                    return cls._adapters[name](FakeDiffusionPipeline(pipe_or_module), **kwargs)
                else:
                    return cls._adapters[name](pipe_or_module, **kwargs)

        return None

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

        # 0. Prefer custom setting from block adapter.
        if isinstance(pipe_or_adapter_or_module, BlockAdapter):
            return pipe_or_adapter_or_module.has_separate_cfg

        has_separate_cfg = False
        if isinstance(pipe_or_adapter_or_module, FakeDiffusionPipeline):
            return False

        if isinstance(pipe_or_adapter_or_module, (DiffusionPipeline, torch.nn.Module)):
            adapter = cls.get_adapter(
                pipe_or_adapter_or_module,
                skip_post_init=True,  # check cfg setting only
            )
            if adapter is not None:
                has_separate_cfg = adapter.has_separate_cfg

        if has_separate_cfg:
            return True

        pipe_cls_name = pipe_or_adapter_or_module.__class__.__name__
        for name in cls._predefined_adapters_has_separate_cfg:
            if pipe_cls_name.startswith(name):
                return True

        return False

    @classmethod
    def is_supported(cls, pipe_or_module) -> bool:
        cls_name: str = pipe_or_module.__class__.__name__

        for name in cls._adapters:
            if cls_name.startswith(name):
                return True
        return False

    @classmethod
    def supported_pipelines(cls, **kwargs) -> Tuple[int, List[str]]:
        val_pipelines = cls._adapters.keys()
        return len(val_pipelines), [p for p in val_pipelines]
