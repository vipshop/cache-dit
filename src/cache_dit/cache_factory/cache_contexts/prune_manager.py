import torch
from typing import Dict, List

from cache_dit.cache_factory.cache_contexts.cache_manager import (
    CachedContextManager,
)
from cache_dit.cache_factory.cache_contexts.prune_context import (
    PrunedContext,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class PrunedContextManager(CachedContextManager):
    # Reuse CachedContextManager for Dynamic Block Prune

    def __init__(self, name: str = None):
        super().__init__(name)
        # Overwrite for Dynamic Block Prune
        self._current_context: PrunedContext = None
        self._cached_context_manager: Dict[str, PrunedContext] = {}

    # Overwrite for Dynamic Block Prune
    def new_context(self, *args, **kwargs) -> PrunedContext:
        _context = PrunedContext(*args, **kwargs)
        self._cached_context_manager[_context.name] = _context
        return _context

    def set_context(self, cached_context) -> PrunedContext:
        return super().set_context(cached_context)

    def get_context(self, name: str = None) -> PrunedContext:
        return super().get_context(name)

    def reset_context(self, name: str = None) -> PrunedContext:
        return super().reset_context(name)

    # Specially for Dynamic Block Prune
    @torch.compiler.disable
    def add_pruned_step(self):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.add_pruned_step()

    @torch.compiler.disable
    def add_pruned_block(self, num_blocks):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.add_pruned_block(num_blocks)

    @torch.compiler.disable
    def add_actual_block(self, num_blocks):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.add_actual_block(num_blocks)

    @torch.compiler.disable
    def get_pruned_steps(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_pruned_steps()

    @torch.compiler.disable
    def get_cfg_pruned_steps(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_pruned_steps()

    @torch.compiler.disable
    def get_pruned_blocks(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_pruned_blocks()

    @torch.compiler.disable
    def get_actual_blocks(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_actual_blocks()

    @torch.compiler.disable
    def get_cfg_pruned_blocks(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_pruned_blocks()

    @torch.compiler.disable
    def get_cfg_actual_blocks(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_actual_blocks()
