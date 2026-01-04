import math
from typing import Optional, Tuple


def _ceil_to_multiple(x: int, base: int) -> int:
    return int(math.ceil(x / base) * base)


def maybe_align_resolution_for_context_parallel(
    *,
    pipe,
    model_path: str,
    parallel_type: Optional[str],
    world_size: int,
    width: int,
    height: int,
) -> Tuple[int, int]:
    if parallel_type not in ("ulysses", "ring"):
        return width, height

    if world_size <= 1 or pipe is None:
        return width, height

    pipe_name = pipe.__class__.__name__
    model_path_lower = (model_path or "").lower()
    is_flux2 = pipe_name == "Flux2Pipeline" or ("flux.2" in model_path_lower)
    if not is_flux2:
        return width, height

    scale_factor = int(getattr(pipe, "vae_scale_factor", 8) or 8)
    if scale_factor <= 0:
        scale_factor = 8

    base_w = _ceil_to_multiple(width, scale_factor)
    base_h = _ceil_to_multiple(height, scale_factor)

    def num_latents(w: int, h: int) -> int:
        return (w // scale_factor) * (h // scale_factor)

    if num_latents(base_w, base_h) % world_size == 0:
        return base_w, base_h

    best: Optional[Tuple[int, int, int, int]] = None
    max_step = scale_factor * world_size
    for dw in range(0, max_step + 1, scale_factor):
        for dh in range(0, max_step + 1, scale_factor):
            w = base_w + dw
            h = base_h + dh
            if num_latents(w, h) % world_size != 0:
                continue
            delta_area = (w * h) - (base_w * base_h)
            cand = (delta_area, dw + dh, w, h)
            if best is None or cand < best:
                best = cand

    if best is None:
        raise ValueError(
            f"Flux2 + {parallel_type}: cannot find a valid resolution near {width}x{height}. "
            f"Require (width/{scale_factor})*(height/{scale_factor}) divisible by world_size={world_size}."
        )

    _, _, new_w, new_h = best
    return new_w, new_h


def center_crop_pil_image(img, *, target_width: int, target_height: int):
    w, h = img.size
    if (w, h) == (target_width, target_height):
        return img
    left = max((w - target_width) // 2, 0)
    top = max((h - target_height) // 2, 0)
    right = min(left + target_width, w)
    bottom = min(top + target_height, h)
    return img.crop((left, top, right, bottom))
