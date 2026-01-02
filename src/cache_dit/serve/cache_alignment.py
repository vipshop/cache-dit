from __future__ import annotations

from typing import List, Optional

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
        is_wan_2_2 = "wan2.2" in model_path_lower
        if not is_wan_2_2:
            return None

        return [
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=4,
                    max_cached_steps=8,
                ),
            ),
            ParamsModifier(
                cache_config=DBCacheConfig().reset(
                    max_warmup_steps=2,
                    max_cached_steps=20,
                ),
            ),
        ]

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
