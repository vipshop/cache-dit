from dataclasses import dataclass
from typing import Any, Dict, Literal
from diffusers import ContextParallelConfig


@dataclass
class _ExtendedContextParallelConfig(ContextParallelConfig):
    rotate_method: Literal["allgather", "alltoall", "p2p"] = "allgather"
    extra_kwargs: Dict[str, Any] = None  # For future extensions

    def __post_init__(self):
        # Override the __post_init__ method to allow the extended features
        # in cache-dit to work properly.
        if self.ring_degree is None:
            self.ring_degree = 1
        if self.ulysses_degree is None:
            self.ulysses_degree = 1

        if self.ring_degree == 1 and self.ulysses_degree == 1:
            raise ValueError(
                "Either ring_degree or ulysses_degree must be greater than 1 in order "
                "to use context parallel inference"
            )
        if self.ring_degree < 1 or self.ulysses_degree < 1:
            raise ValueError(
                "`ring_degree` and `ulysses_degree` must be greater than or equal to 1."
            )
        if self.rotate_method not in ["allgather", "p2p"]:
            raise NotImplementedError(
                "Only the 'allgather' and 'p2p' rotate methods are supported for now, "
                f"but got {self.rotate_method}."
            )
