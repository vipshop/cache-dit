import torch
import functools
from typing import Dict, List, Tuple, Union

from .cache_manager import CachedContextManager
from .prune_context import PrunedContext
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class PrunedContextManager(CachedContextManager):
    # Reuse CachedContextManager for Dynamic Block Prune

    def __init__(self, name: str = None, **kwargs):
        super().__init__(name, **kwargs)
        # Overwrite for Dynamic Block Prune
        self._current_context: PrunedContext = None
        self._cached_context_manager: Dict[str, PrunedContext] = {}

    # Overwrite for Dynamic Block Prune
    def new_context(self, *args, **kwargs) -> PrunedContext:
        _context = PrunedContext(*args, **kwargs)
        # NOTE: Patch args and kwargs for implicit refresh.
        _context._init_args = args  # maybe empty tuple: ()
        _context._init_kwargs = kwargs  # maybe empty dict: {}
        self._cached_context_manager[_context.name] = _context
        return _context

    def set_context(
        self,
        cached_context: PrunedContext | str,
    ) -> PrunedContext:
        return super().set_context(cached_context)

    def get_context(self, name: str = None) -> PrunedContext:
        return super().get_context(name)

    def reset_context(
        self,
        cached_context: PrunedContext | str,
        *args,
        **kwargs,
    ) -> PrunedContext:
        return super().reset_context(cached_context, *args, **kwargs)

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

    @torch.compiler.disable
    @functools.lru_cache(maxsize=8)
    def get_non_prune_blocks_ids(self, num_blocks: int) -> List[int]:
        assert num_blocks is not None, "num_blocks must be provided"
        assert num_blocks > 0, "num_blocks must be greater than 0"
        # Get the non-prune block ids for current context
        # Never prune the first `Fn` and last `Bn` blocks.
        Fn_compute_blocks_ids = list(
            range(self.Fn_compute_blocks() if self.Fn_compute_blocks() < num_blocks else num_blocks)
        )

        Bn_compute_blocks_ids = list(
            range(
                num_blocks
                - (
                    self.Bn_compute_blocks()
                    if self.Bn_compute_blocks() < num_blocks
                    else num_blocks
                ),
                num_blocks,
            )
        )
        context = self.get_context()
        assert context is not None, "cached_context must be set before"

        non_prune_blocks_ids = list(
            set(
                Fn_compute_blocks_ids
                + Bn_compute_blocks_ids
                + context.cache_config.non_prune_block_ids
            )
        )
        non_prune_blocks_ids = [d for d in non_prune_blocks_ids if d < num_blocks]
        return sorted(non_prune_blocks_ids)

    @torch.compiler.disable
    def can_prune(self, *args, **kwargs) -> bool:
        # Directly reuse can_cache for Dynamic Block Prune
        return self.can_cache(*args, **kwargs)

    @torch.compiler.disable
    def apply_prune(self, *args, **kwargs) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        # Directly reuse apply_cache for Dynamic Block Prune
        return self.apply_cache(*args, **kwargs)
