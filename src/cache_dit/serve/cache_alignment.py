from __future__ import annotations

from typing import Optional, List
from cache_dit import DBCacheConfig, ParamsModifier

def get_default_params_modifiers(
    *,
    pipe,
    model_path: str | None,
    cache_config_obj,
) -> Optional[List[object]]:
    if cache_config_obj is None:
        return None

    model_path_lower = (model_path or "").lower()
    is_flux2 = (pipe is not None and pipe.__class__.__name__ == "Flux2Pipeline") or (
        "flux.2" in model_path_lower
    )
    if not is_flux2:
        return None


    rdt = getattr(cache_config_obj, "residual_diff_threshold", 0.08)
    return [
        ParamsModifier(
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=rdt,
            ),
        ),
        ParamsModifier(
            cache_config=DBCacheConfig().reset(
                residual_diff_threshold=rdt * 3,
            ),
        ),
    ]
