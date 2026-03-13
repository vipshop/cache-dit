import dataclasses
from typing import Optional, Dict, Any, List, Union
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class QuantizeConfig:
    # quantization backend, only "ao" (torchao) is supported
    # for now, more backends will be supported in the future.
    backend: str = "ao"
    quant_type: str = "float8_weight_only"
    per_row: bool = True
    exclude_layers: Optional[list] = dataclasses.field(
        default_factory=lambda: ["embedder", "embed"]
    )
    filter_fn: Optional[Any] = None  # type: ignore
    # components_to_quantize: (list[str] or dict[str, str], optional)
    # specify the components to quantize, if None, only the transformer
    # module will be quantized. e.g:
    # - List[str]: ['transformer', 'text_encoder'] quantize to 'quant_type'
    # - Dict[str, str]: {'transformer': 'float8', 'text_encoder': 'float8_weight_only'}.
    #   The 'quant_type' will be ignored in this case, each module will quantized to
    #   it's specified quantization type.
    components_to_quantize: Optional[Union[List[str], Dict[str, str]]] = None
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
        return f"{self.quant_type.lower()}"

    # Transformer components_to_quantize to mutiple simple configs, each
    # with only one component to quantize, and the same quantization type.
    @classmethod
    def expand_configs(cls, config: "QuantizeConfig") -> List["QuantizeConfig"]:
        if config.components_to_quantize is None:
            return [config]

        if isinstance(config.components_to_quantize, list):
            return [
                dataclasses.replace(config, components_to_quantize=[component])
                for component in config.components_to_quantize
            ]

        if isinstance(config.components_to_quantize, dict):
            return [
                dataclasses.replace(
                    config, components_to_quantize=[component], quant_type=quant_type
                )
                for component, quant_type in config.components_to_quantize.items()
            ]

        raise ValueError("components_to_quantize should be either a list or a dict.")
