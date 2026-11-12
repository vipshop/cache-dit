"""MiniMax-H3-Image-PE prompt expansion, wired as an optional stage before the DiT.

`MiniMax-H3-Image-PE` rewrites a short request into the long, complete description MiniMax-H3-Image
was distilled on and recommends an aspect ratio. It is a *text* model
(`Qwen3_5ForConditionalGeneration`, NVFP4 weights): nothing in it is a diffusion transformer, so the
only thing cache-dit does here is run it before the image pipeline, feed the expanded caption forward
and hand the recommended ratio to `height` / `width`.

Two checkpoints are involved and they are not the same repository:

* `MiniMaxAI/MiniMax-H3-Image-PE` — the **weights** (config, NVFP4 safetensors, tokenizer,
  processor and the task prompts under `prompt/{t2i,edit}/task_prompt.txt`);
* `diffusers/minimax-h3-image-pe-blocks` — the **Modular Diffusers blocks** that own the request
  shape (task prompt assembled from those files, XGrammar constrained decoding to
  `{"expanded_caption", "target_aspect_ratio"}`, FP4 / bitsandbytes placement).

`ModularPipeline.from_pretrained(..., trust_remote_code=True)` is what runs them, and it needs the
*blocks* repository: pointing it at the weights repository fails with a missing model index. The
weights alone also cannot be loaded by plain `transformers`, because the NVFP4 packing halves every
input dimension (e.g. `[6144, 2560]` on disk against `[6144, 5120]` in the config).

Keep this stage **off** for benchmarks: it changes the input distribution (the DiT then encodes a
different, much longer prompt), and a ~19 GiB VLM generating a few hundred tokens dominates the
measured latency.
"""

import gc
import os
from typing import Optional, Sequence, Tuple

import torch

from ....logger import init_logger

logger = init_logger(__name__)

# The checkpoint generates at 4 megapixels or more; below that the pipeline draws on a larger canvas
# and resizes back, which costs the same and loses the detail the request asked for. Sides are
# multiples of 32 (the VAE's 16x compression times the 2x patch).
_DEFAULT_AREA = 2048 * 2048
_CANVAS_MULTIPLE = 32
# The pipeline rejects a `min_generation_area` above 4096x4096, so a canvas side beyond that cannot
# be rendered at the documented floor.
_MAX_CANVAS_SIDE = 4096


def aspect_ratio_to_size(aspect_ratio: str, area: int = _DEFAULT_AREA) -> Tuple[int, int]:
  """Turn the enhancer's `W:H` recommendation into a `(height, width)` canvas.

  MiniMax's own recommendations are larger (4096x4096 for 1:1, up to 6272x2688 for 21:9); this
  helper keeps the documented 4 MP minimum so that a CLI run stays affordable. Pass
  `--height/--width` for one of the recommended canvases.

  :param aspect_ratio: Ratio as `W:H`, e.g. `"3:2"`, or a bare float, e.g. `"1.5"`.
  :param area: Target pixel area.
  :returns:`(height, width)`, both multiples of 32.
  :raises ValueError: When the ratio cannot be parsed or is not positive.
  """
  text = str(aspect_ratio).strip()
  if ":" in text:
    width_part, height_part = text.split(":", 1)
  else:
    width_part, height_part = text, "1"
  try:
    width = float(width_part)
    height = float(height_part)
  except ValueError as error:
    raise ValueError(f"`aspect_ratio` must be `W:H`, got {aspect_ratio!r}.") from error
  if width <= 0 or height <= 0:
    raise ValueError(f"`aspect_ratio` must be positive, got {aspect_ratio!r}.")

  # Snap the width, derive the height from the same ratio, then grow the width until the canvas meets
  # the area floor: snapping both sides independently can land just below it (16:9 at 4 MP snaps to
  # 2720x1536, 0.3 % short), and the pipeline would then silently generate on a larger canvas.
  aspect = width / height
  if not 1 / 8 <= aspect <= 8:
    raise ValueError(
      f"`aspect_ratio` {aspect_ratio!r} is outside the supported range (1:8 .. 8:1); "
      "pass --height/--width explicitly for such a canvas.")
  snapped_width = max(_CANVAS_MULTIPLE,
                      int(round((area * aspect) ** 0.5 / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)
  snapped_height = max(_CANVAS_MULTIPLE,
                       int(round(snapped_width / aspect / _CANVAS_MULTIPLE)) * _CANVAS_MULTIPLE)
  while snapped_width * snapped_height < area:
    snapped_width += _CANVAS_MULTIPLE
    if snapped_width > _MAX_CANVAS_SIDE:
      raise ValueError(
        f"`aspect_ratio` {aspect_ratio!r} does not fit a {_MAX_CANVAS_SIDE} px side.")
  return snapped_height, snapped_width


def _resolve_blocks_dir(blocks_dir: str) -> str:
  """Check that `blocks_dir` is the PE *blocks* repository, not the weights repository.

  :param blocks_dir: Local path of the PE pipeline.
  :returns: The same path.
  :raises ValueError: When the directory holds weights without Modular Diffusers blocks.
  """
  if not os.path.isdir(blocks_dir):
    raise ValueError(f"--prompt-enhancer expects a directory, got {blocks_dir!r}.")
  if any(
      os.path.exists(os.path.join(blocks_dir, name))
      for name in ("modular_model_index.json", "model_index.json")):
    return blocks_dir
  raise ValueError(
    f"{blocks_dir} holds the MiniMax-H3-Image-PE *weights*, not the Modular Diffusers blocks that "
    "own its request shape. Point --prompt-enhancer at "
    "`diffusers/minimax-h3-image-pe-blocks` (e.g. `huggingface-cli download "
    "diffusers/minimax-h3-image-pe-blocks --local-dir <dir>`). The weights alone cannot be loaded by "
    "plain transformers either: the NVFP4 packing halves every input dimension.")


def _read_state(state, names: Tuple[str, ...]) -> Optional[str]:
  """Read the first non-empty field of `state`, if it exposes the mapping interface.

  :param state: `PipelineState` returned by the blocks, or any mapping-like object.
  :param names: Field names to try, in priority order.
  :returns: The first non-empty value, or None.
  """
  if not hasattr(state, "get"):
    return None
  for name in names:
    value = state.get(name)
    if value:
      return value
  return None


def _explicit_ratio(value: Optional[str]) -> Optional[str]:
  """Keep only an explicit `W:H` recommendation.

  The blocks echo the requested `aspect_ratio` back into the state, so `"adaptive"` (the request
  default) must not be mistaken for advice — it would reach `aspect_ratio_to_size` and raise.

  :param value: Candidate ratio.
  :returns: The ratio when it is `W:H`, else None.
  """
  if not value:
    return None
  text = str(value).strip()
  if ":" not in text:
    return None
  width, _, height = text.partition(":")
  if not (width.strip().isdigit() and height.strip().isdigit()):
    return None
  return text


def expand_prompt(
  blocks_dir: str,
  prompt: str,
  images: Optional[Sequence] = None,
  aspect_ratio: str = "adaptive",
  device: str = "cuda",
) -> Tuple[str, Optional[str]]:
  """Expand one request with MiniMax-H3-Image-PE and release the model again.

  The enhancer is loaded for the single call and dropped afterwards: it is ~19 GiB of FP4 weights and
  the image transformer plus its Qwen3-VL conditioner need the device back.

  :param blocks_dir: Local path of the PE blocks repository.
  :param prompt: The short request to expand.
  :param images: Reference images for the editing workflow, in prompt order.
  :param aspect_ratio: `"adaptive"` to let the model choose, or an explicit `W:H`.
  :param device: Device the enhancer runs on.
  :returns: `(expanded_caption, aspect_ratio)`; the ratio may be None when the model omits it.
  :raises ValueError: When the directory is the weights repository.
  """
  from diffusers import ModularPipeline

  _resolve_blocks_dir(blocks_dir)
  pipe = ModularPipeline.from_pretrained(blocks_dir, trust_remote_code=True)
  try:
    pipe.load_components(dtype=torch.bfloat16, device_map=device)
    kwargs = {"prompt": prompt, "aspect_ratio": aspect_ratio}
    if images:
      kwargs["image"] = list(images)
    state = pipe(**kwargs)
    # The blocks publish `expanded_caption` / `target_aspect_ratio` (their own documented schema), but
    # the state also carries the request fields they were built from, so both spellings are accepted.
    caption = _read_state(state, ("expanded_caption", "prompt"))
    ratio = _explicit_ratio(_read_state(state, ("target_aspect_ratio", "aspect_ratio")))
  finally:
    del pipe
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  if not caption:
    raise ValueError(
      "MiniMax-H3-Image-PE returned neither `expanded_caption` nor `prompt`; the blocks repository "
      "may have changed its output names.")
  if caption.strip() == prompt.strip():
    raise ValueError(
      "MiniMax-H3-Image-PE returned the request unchanged: the expansion did not run, which usually "
      "means the blocks write a different field than the one read here.")
  logger.info("MiniMax-H3-Image-PE expanded the prompt to %d characters (aspect ratio %s).",
              len(caption), ratio)
  return caption, ratio
