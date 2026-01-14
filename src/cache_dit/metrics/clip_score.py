import os
import re
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from transformers import CLIPProcessor, CLIPModel

from typing import Tuple, Union
from .config import _IMAGE_EXTENSIONS
from .config import get_metrics_verbose
from ..platforms import current_platform
from cache_dit.logger import init_logger

logger = init_logger(__name__)


DISABLE_VERBOSE = not get_metrics_verbose()


class CLIPScore:
    def __init__(
        self,
        device=(
            current_platform.device_type if current_platform.is_accelerator_available() else "cpu"
        ),
        clip_model_path: str = None,
    ):
        self.device = device
        if clip_model_path is None:
            clip_model_path = os.environ.get(
                "CLIP_MODEL_DIR", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
            )

        # Load models
        self.clip_model = CLIPModel.from_pretrained(clip_model_path)
        self.clip_model = self.clip_model.to(device)  # type: ignore
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    @torch.no_grad()
    def compute_clip_score(
        self,
        img: Image.Image | np.ndarray,
        prompt: str,
    ) -> float:
        if isinstance(img, Image.Image):
            img_pil = img.convert("RGB")
        elif isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img).convert("RGB")
        else:
            img_pil = Image.open(img).convert("RGB")
        with torch.no_grad():
            inputs = self.clip_processor(
                text=prompt,
                images=img_pil,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            outputs = self.clip_model(**inputs)
        return outputs.logits_per_image.item()


clip_score_instance: CLIPScore = None


def compute_clip_score_img(
    img: Image.Image | np.ndarray | str,
    prompt: str,
    clip_model_path: str = None,
) -> float:
    global clip_score_instance
    if clip_score_instance is None:
        clip_score_instance = CLIPScore(clip_model_path=clip_model_path)
    assert clip_score_instance is not None
    return clip_score_instance.compute_clip_score(img, prompt)


def compute_clip_score(
    img_dir: Image.Image | np.ndarray | str,
    prompts: str | list[str],
    clip_model_path: str = None,
) -> Union[Tuple[float, int], Tuple[None, None]]:
    if not os.path.isdir(img_dir) or (
        not isinstance(prompts, list) and not os.path.isfile(prompts)
    ):
        return (
            compute_clip_score_img(
                img_dir,
                prompts,
                clip_model_path=clip_model_path,
            ),
            1,
        )

    # compute dir metric
    def natural_sort_key(filename):
        match = re.search(r"(\d+)\D*$", filename)
        return int(match.group(1)) if match else filename

    img_dir: pathlib.Path = pathlib.Path(img_dir)
    img_files = [file for ext in _IMAGE_EXTENSIONS for file in img_dir.rglob("*.{}".format(ext))]
    img_files = [file.as_posix() for file in img_files]
    img_files = sorted(img_files, key=natural_sort_key)

    if os.path.isfile(prompts):
        """Load prompts from file"""
        with open(prompts, "r", encoding="utf-8") as f:
            prompts_load = [line.strip() for line in f.readlines()]
        prompts = prompts_load.copy()

    vaild_len = min(len(img_files), len(prompts))
    img_files = img_files[:vaild_len]
    prompts = prompts[:vaild_len]

    clip_scores = []

    for img_file, prompt in tqdm(
        zip(img_files, prompts),
        total=vaild_len,
        disable=not get_metrics_verbose(),
    ):
        clip_scores.append(
            compute_clip_score_img(
                img_file,
                prompt,
                clip_model_path=clip_model_path,
            )
        )

    if vaild_len > 0:
        return np.mean(clip_scores), vaild_len
    return None, None
