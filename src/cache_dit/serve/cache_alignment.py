from __future__ import annotations

from typing import Any, Dict, List, Optional

import cache_dit
from .. import DBCacheConfig, ParamsModifier


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

    rdt = getattr(cache_config_obj, "residual_diff_threshold", 0.24)
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


def align_cache_config(
    *,
    model_path: str,
    args,
    base_cache_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if base_cache_config is None:
        return None

    model_path_lower = (model_path or "").lower()
    cache_config = dict(base_cache_config)

    is_qwen_lightning = (
        "qwen-image-lightning" in model_path_lower
        or "qwen-image-edit-2511-lightning" in model_path_lower
        or "qwen-image-edit-2509-lightning" in model_path_lower
    )
    if is_qwen_lightning:
        steps = (
            8 if getattr(args, "num_inference_steps", None) is None else args.num_inference_steps
        )
        if steps not in (4, 8):
            raise ValueError("Qwen-Image Lightning only supports 4 or 8 steps.")
        cache_config.update(
            {
                "Fn_compute_blocks": 16,
                "Bn_compute_blocks": 16,
                "max_warmup_steps": 4 if steps > 4 else 2,
                "max_cached_steps": 2 if steps > 4 else 1,
                "max_continuous_cached_steps": 1,
                "enable_separate_cfg": False,
                "residual_diff_threshold": 0.50 if steps > 4 else 0.8,
            }
        )
        return cache_config

    if "qwen-image-layered" in model_path_lower:
        cache_config.setdefault("enable_separate_cfg", False)

    if "z-image-turbo" in model_path_lower and base_cache_config is not None:
        cache_config["max_warmup_steps"] = min(
            int(cache_config.get("max_warmup_steps", 8)),
            4,
        )
        total_steps = (
            9 if getattr(args, "num_inference_steps", None) is None else args.num_inference_steps
        )
        steps_computation_mask = None
        if getattr(args, "mask_policy", None) is not None:
            steps_computation_mask = cache_dit.steps_mask(
                mask_policy=args.mask_policy,
                total_steps=total_steps,
            )
        elif getattr(args, "steps_mask", False):
            steps_computation_mask = cache_dit.steps_mask(
                compute_bins=[5, 1, 1],
                cache_bins=[1, 1],
            )
        cache_config["steps_computation_mask"] = steps_computation_mask

    return cache_config
