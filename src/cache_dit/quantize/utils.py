from __future__ import annotations


def normalize_quantize_type(quantize_type: str | None) -> str | None:
    if quantize_type is None:
        return None
    mapping = {
        "float8_wo": "float8_weight_only",
        "int8_wo": "int8_weight_only",
        "int4_wo": "int4_weight_only",
        "bnb_4bit": "bitsandbytes_4bit",
    }
    return mapping.get(quantize_type, quantize_type)
