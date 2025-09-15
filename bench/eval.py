import os
import torch
import cv2
import numpy as np
import re
import argparse
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim
from transformers import CLIPProcessor, CLIPModel
import ImageReward as RM
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as T

from cache_dit import init_logger

logger = init_logger(__name__)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_prompts(prompt_file_path):
    """Load prompts from file"""
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_sorted_image_files(folder_path):
    """Get sorted image files from folder"""
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        ):
            image_files.append(filename)

    # Natural sort by number in filename
    def natural_sort_key(filename):
        match = re.search(r"img_(\d+)\.", filename)
        return int(match.group(1)) if match else 0

    return sorted(image_files, key=natural_sort_key)


def calculate_psnr(img1, img2):
    """Calculate PSNR"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """Calculate SSIM"""
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, data_range=255)
    return ssim(img1, img2, data_range=255)


def preprocess_for_lpips(img):
    """Preprocess image for LPIPS"""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img * 2 - 1


def evaluate_all_metrics(
    test_folder,
    prompt_file_path=None,
    reference_folder=None,
    clip_model_path=None,
    imagereward_model_path=None,
):
    """Evaluate all metrics and return results"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if clip_model_path is None:
        clip_model_path = os.environ.get(
            "CLIP_MODEL_DIR", "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
        )

    if imagereward_model_path is None:
        imagereward_model_path = os.environ.get(
            "IMAGEREWARD_MODEL_DIR", "zai-org/ImageReward"
        )

    # Load models
    clip_model = CLIPModel.from_pretrained(clip_model_path)
    clip_model = clip_model.to(device)  # type: ignore
    clip_processor = CLIPProcessor.from_pretrained(clip_model_path)

    # Load ImageReward model
    med_config = os.path.join(imagereward_model_path, "med_config.json")
    imagereward_path = os.path.join(imagereward_model_path, "ImageReward.pt")
    imagereward_model = RM.load(
        imagereward_path,
        download_root=imagereward_model_path,
        med_config=med_config,
    ).to(device)

    # LPIPS model
    lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device)

    # ImageReward transform
    reward_transform = T.Compose(
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

    # Load data
    image_files = get_sorted_image_files(test_folder)
    prompts = load_prompts(prompt_file_path) if prompt_file_path else []

    # Initialize score lists
    clip_scores = []
    imagereward_scores = []
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Process images
    for i, filename in enumerate(image_files):
        try:
            img_path = os.path.join(test_folder, filename)
            img_pil = Image.open(img_path).convert("RGB")

            # CLIP Score and ImageReward (require prompts)
            if i < len(prompts):
                prompt = prompts[i]

                # CLIP Score
                with torch.no_grad():
                    inputs = clip_processor(
                        text=prompt,
                        images=img_pil,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(device)
                    outputs = clip_model(**inputs)
                    clip_scores.append(outputs.logits_per_image.item())

                # ImageReward
                if imagereward_model:
                    with torch.no_grad():
                        img_tensor = (
                            TF.pil_to_tensor(img_pil).unsqueeze(0).to(device)
                        )
                        img_reward = reward_transform(img_tensor)
                        inputs = imagereward_model.blip.tokenizer(
                            [prompt],
                            padding="max_length",
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(device)
                        score = imagereward_model.score_gard(
                            inputs.input_ids, inputs.attention_mask, img_reward
                        )
                        imagereward_scores.append(score.item())

            # Quality metrics (require reference)
            if reference_folder:
                ref_path = os.path.join(reference_folder, filename)
                if os.path.exists(ref_path):
                    img_cv = cv2.imread(img_path)
                    ref_cv = cv2.imread(ref_path)

                    if img_cv is not None and ref_cv is not None:
                        # Resize if needed
                        if img_cv.shape != ref_cv.shape:
                            ref_cv = cv2.resize(
                                ref_cv, (img_cv.shape[1], img_cv.shape[0])
                            )

                        # Calculate metrics
                        psnr_values.append(calculate_psnr(img_cv, ref_cv))
                        ssim_values.append(calculate_ssim(img_cv, ref_cv))

                        with torch.no_grad():
                            img_lpips = preprocess_for_lpips(img_cv).to(device)
                            ref_lpips = preprocess_for_lpips(ref_cv).to(device)
                            lpips_values.append(
                                lpips_model(img_lpips, ref_lpips).item()
                            )

        except Exception as e:
            logger.error(f"{e}")
            continue

    # Calculate averages
    results = {}
    if clip_scores:
        results["clip_score"] = np.mean(clip_scores)
    if imagereward_scores:
        results["imagereward"] = np.mean(imagereward_scores)
    if psnr_values:
        results["psnr"] = np.mean(psnr_values)
    if ssim_values:
        results["ssim"] = np.mean(ssim_values)
    if lpips_values:
        results["lpips"] = np.mean(lpips_values)

    return results


def main():
    parser = argparse.ArgumentParser(description="Unified metrics evaluation")
    parser.add_argument(
        "--test_folder", type=str, required=True, help="Test images folder"
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default="prompts/DrawBench200.txt",
        help="Prompts file",
    )
    parser.add_argument(
        "--reference_folder",
        type=str,
        default="./tmp/DrawBench200/C0_Q0_NONE",
        help="Reference images folder for quality metrics",
    )
    # "laion/CLIP-ViT-g-14-laion2B-s12B-b42K"
    parser.add_argument("--clip_model_path", type=str, default=None)
    # "zai-org/ImageReward"
    parser.add_argument("--imagereward_model_path", type=str, default=None)

    args = parser.parse_args()

    results = evaluate_all_metrics(
        test_folder=args.test_folder,
        prompt_file_path=args.prompt_file,
        reference_folder=args.reference_folder,
        clip_model_path=args.clip_model_path,
        imagereward_model_path=args.imagereward_model_path,
    )

    print("Result:(ClipScore, ImageReward, PSNR, SSIM, LPIPS)")
    print(f"{results.get('clip_score', 0):.4f}")
    print(f"{results.get('imagereward', 0):.4f}")
    print(f"{results.get('psnr', 0):.3f}")
    print(f"{results.get('ssim', 0):.4f}")
    print(f"{results.get('lpips', 0):.4f}")


if __name__ == "__main__":
    main()
