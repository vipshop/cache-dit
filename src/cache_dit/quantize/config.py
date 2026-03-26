import dataclasses
from typing import Optional, Dict, Any, List, Union
from ..logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class QuantizeConfig:
    # quantization backend, only "ao" (torchao) is supported
    # for now, more backends will be supported in the future.
    backend: str = "ao"
    # quantization type, currently support "float8_weight_only" and "float8",
    # "float8_blockwise", "int8", "int8_weight_only", "int4_weight_only", etc.
    quant_type: str = "float8_weight_only"
    # Whether to quantize the weights in per-row or per-tensor manner when
    # quant_type is float8, default to per-row quantization, which is more
    # accurate but may not be supported for some layers, setting this flag
    # to False will quantize those layers to float8 per-tensor.
    per_row: bool = True
    # The layers specified in this variable will be excluded from quantization,
    # even if they are in the repeated blocks or not filtered out by filter_fn.
    # The format of the layer name should be the same as the name in the model's
    # state_dict, e.g, "transformer.blocks.0.attn.to_k.weight". This is useful
    # for cases when some specific layers cannot be quantized for some reasons,
    # e.g, they are already very small and quantization may cause significant
    # accuracy drop, or they are not supported to be quantized due to some
    # technical reasons, etc.
    exclude_layers: Optional[list] = dataclasses.field(
        default_factory=lambda: [
            "embedder",
            "embed",
            "modulation",
            "mod",
        ]
    )
    # Quantize the _repeated_ blocks in the transformer (Diffusers).
    regional_quantize: bool = True  # name 'regional', vs regional compile.
    # For models outside of diffusers, users can specify the repeated blocks
    # by setting this variable to a list of block names.
    repeated_blocks: List[str] = dataclasses.field(default_factory=list)
    # A filter function to determine whether to quantize a specific module or not,
    # it will be called in the format of filter_fn(m: nn.Module, name: str) -> bool.
    # It should return True if the module needs to be quantized, otherwise False.
    # If filter_fn is specified, the exclude_layers will be ignored.
    filter_fn: Optional[Any] = None  # Usually not use.
    # components_to_quantize: (list[str] or dict[str, str], optional)
    # specify the components to quantize, if None, only the transformer
    # module will be quantized. e.g:
    # - List[str]: ['transformer', 'text_encoder'] quantize to 'quant_type'
    # - Dict[str, Dict[str, str]]: {
    #     'transformer': {'quant_type': 'float8'},
    #     'text_encoder': {'quant_type': 'float8_weight_only'}
    #   }.
    # The 'quant_type' will be ignored in this case, each module will quantized to
    # it's specified quantization type.
    components_to_quantize: Optional[Union[List[str], Dict[str, Dict[str, str]]]] = None
    # Whether to fallback to float8 quantization when float8 per-row or per-block
    # quantization is not supported for some layers. This is useful for cases when
    # tensor parallelism is applied, and some layers cannot be quantized to float8
    # per-row or per-block, e.g, layers applied RowwiseParallel may not support
    # float8 per-row quantization currently, _scaled_mm will raise memory layout
    # mismatch error when quantized to float8 per-row, setting this flag to True will fallback
    # to float8 per tensor quantization for those layers, instead of raising error.
    float8_per_tensor_fallback: bool = True
    # Whether to print detailed quantization information, such as the quantization
    # type of each layer, the reason for skipping quantization, etc. This is useful
    # for debugging and analysis.
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
        if self.components_to_quantize is None or isinstance(self.components_to_quantize, list):
            return f"{self.quant_type.lower()}"
        else:
            quant_str = ""
            if isinstance(self.components_to_quantize, dict):
                for component, d in self.components_to_quantize.items():
                    quant_str += f"<{component}:{d.get('quant_type', self.quant_type)}>"
            return quant_str

    def component_quant_types(self) -> Dict[str, str]:
        if self.components_to_quantize is None:
            return {"transformer": self.quant_type}
        elif isinstance(self.components_to_quantize, list):
            return {component: self.quant_type for component in self.components_to_quantize}
        elif isinstance(self.components_to_quantize, dict):
            return {
                component: d.get("quant_type", self.quant_type)
                for component, d in self.components_to_quantize.items()
            }
        else:
            raise ValueError("components_to_quantize should be either a list or a dict.")

    @classmethod
    def expand_configs(cls, config: "QuantizeConfig") -> List["QuantizeConfig"]:
        # Transfer components_to_quantize to mutiple simple configs, each
        # with only 1 component to quantize, and the same quantization type.
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
                    config,
                    backend=cfg.get("backend", config.backend),
                    components_to_quantize=[component],
                    quant_type=cfg.get("quant_type", config.quant_type),
                    per_row=cfg.get("per_row", config.per_row),
                    exclude_layers=cfg.get("exclude_layers", config.exclude_layers),
                    regional_quantize=cfg.get("regional_quantize", config.regional_quantize),
                    repeated_blocks=cfg.get("repeated_blocks", config.repeated_blocks),
                    filter_fn=cfg.get("filter_fn", config.filter_fn),
                    float8_per_tensor_fallback=cfg.get(
                        "float8_per_tensor_fallback", config.float8_per_tensor_fallback
                    ),
                    verbose=cfg.get("verbose", config.verbose),
                )
                for component, cfg in config.components_to_quantize.items()
            ]

        raise ValueError("components_to_quantize should be either a list or a dict.")

    @classmethod
    def from_kwargs(cls, **kwargs) -> "QuantizeConfig":
        config = cls()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
