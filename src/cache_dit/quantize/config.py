import dataclasses
from typing import Optional, Dict, Any
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class QuantizeConfig:
    quant_type: str = "float8_weight_only"
    per_row: bool = True
    exclude_layers: Optional[list] = dataclasses.field(
        default_factory=lambda: ["embedder", "embed"]
    )
    filter_fn: Optional[Any] = None  # type: ignore
    verbose: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def update(self, **kwargs) -> "QuantizeConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
        return self

    def strify(self) -> str:
        return self.quant_type.lower()
