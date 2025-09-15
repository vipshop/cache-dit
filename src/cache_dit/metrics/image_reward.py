import os
import pathlib
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import ImageReward as RM
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as T

from cache_dit.metrics.config import _IMAGE_EXTENSIONS
from cache_dit.metrics.config import get_metrics_verbose
from cache_dit.logger import init_logger

logger = init_logger(__name__)


DISABLE_VERBOSE = not get_metrics_verbose()


class ImageRewardScore:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        imagereward_model_path: str = None,
    ):
        self.device = device
        if imagereward_model_path is None:
            imagereward_model_path = os.environ.get(
                "IMAGEREWARD_MODEL_DIR", "zai-org/ImageReward"
            )

        # Load ImageReward model
        self.med_config = os.path.join(
            imagereward_model_path, "med_config.json"
        )
        self.imagereward_path = os.path.join(
            imagereward_model_path, "ImageReward.pt"
        )
        self.imagereward_model = RM.load(
            self.imagereward_path,
            download_root=imagereward_model_path,
            med_config=self.med_config,
        ).to(self.device)

        # ImageReward transform
        self.reward_transform = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    @torch.no_grad()
    def compute_reward_score(
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
            img_tensor = TF.pil_to_tensor(img_pil).unsqueeze(0).to(self.device)
            img_reward = self.reward_transform(img_tensor)
            inputs = self.imagereward_model.blip.tokenizer(
                [prompt],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            score = self.imagereward_model.score_gard(
                inputs.input_ids, inputs.attention_mask, img_reward
            )
        return score.item()


image_reward_score_instance = None


def compute_reward_score_img(
    img: Image.Image | np.ndarray | str,
    prompt: str,
    imagereward_model_path: str = None,
) -> float:
    global image_reward_score_instance
    if image_reward_score_instance is None:
        image_reward_score_instance = ImageRewardScore(
            imagereward_model_path=imagereward_model_path
        )
    assert image_reward_score_instance is not None
    return image_reward_score_instance.compute_reward_score(img, prompt)


def compute_reward_score(
    img_dir: Image.Image | np.ndarray | str,
    prompts: str | list[str],
    imagereward_model_path: str = None,
):
    if not os.path.isdir(img_dir) or not isinstance(prompts, list):
        return compute_reward_score_img(
            img_dir,
            prompts,
            imagereward_model_path=imagereward_model_path,
        )

    # compute dir metric
    img_dir: pathlib.Path = pathlib.Path(img_dir)
    img_files = sorted(
        [
            file
            for ext in _IMAGE_EXTENSIONS
            for file in img_dir.rglob("*.{}".format(ext))
        ]
    )
    img_files = [file.as_posix() for file in img_files]

    vaild_len = min(len(img_files), len(prompts))
    img_files = img_files[:vaild_len]
    prompts = img_files[:vaild_len]

    reward_scores = []

    for img_file, prompt in tqdm(
        zip(img_files, prompts),
        total=vaild_len,
        disable=DISABLE_VERBOSE,
    ):
        reward_scores.append(
            compute_reward_score_img(
                img_file,
                prompt,
                clip_model_path=imagereward_model_path,
            )
        )

    if vaild_len > 0:
        return np.mean(reward_scores), vaild_len
    return None, None
