import os
import cv2
import torch
import argparse
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import (
    calculate_frechet_distance,
    calculate_activation_statistics,
)
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def compute_psnr(
    image_true: np.ndarray | str,
    image_test: np.ndarray | str,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    PSNR = compute_psnr(img_true, img_test)
    """
    if isinstance(image_true, str):
        image_true = cv2.imread(image_true)
    if isinstance(image_test, str):
        image_test = cv2.imread(image_test)
    return peak_signal_noise_ratio(
        image_true,
        image_test,
    )


def compute_video_psnr(
    video_true: str,
    video_test: str,
) -> float:
    """
    video_true = "video_true.mp4"
    video_test = "video_test.mp4"
    PSNR = compute_video_psnr(video_true, video_test)
    """
    cap1 = cv2.VideoCapture(video_true)
    cap2 = cv2.VideoCapture(video_test)

    if not cap1.isOpened() or not cap2.isOpened():
        logger.error("Could not open video files")
        return None

    frame_count = min(
        int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    total_psnr = 0.0
    valid_frames = 0

    logger.debug(f"Total frames: {frame_count}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        psnr = compute_psnr(frame1, frame2)

        if psnr != float("inf"):
            total_psnr += psnr
            valid_frames += 1

        if valid_frames % 10 == 0:
            logger.debug(f"Processed {valid_frames}/{frame_count} frames")

    cap1.release()
    cap2.release()

    if valid_frames > 0:
        average_psnr = total_psnr / valid_frames
        logger.debug(f"Average PSNR: {average_psnr:.2f}")
        return average_psnr
    else:
        logger.debug("No valid frames to compare")
        return None


def compute_mse(
    image_true: np.ndarray | str,
    image_test: np.ndarray | str,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    MSE = compute_mse(img_true, img_test)
    """
    if isinstance(image_true, str):
        image_true = cv2.imread(image_true)
    if isinstance(image_test, str):
        image_test = cv2.imread(image_test)
    return mean_squared_error(
        image_true,
        image_test,
    )


def compute_ssim(
    image_true: np.ndarray | str,
    image_test: np.ndarray | str,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    SSIM = compute_ssim(img_true, img_test)
    """
    if isinstance(image_true, str):
        image_true = cv2.imread(image_true)
    if isinstance(image_test, str):
        image_test = cv2.imread(image_test)
    return structural_similarity(
        image_true,
        image_test,
        multichannel=True,
        channel_axis=2,
    )


class FrechetInceptionDistance:
    IMAGE_EXTENSIONS = {
        "bmp",
        "jpg",
        "jpeg",
        "pgm",
        "png",
        "ppm",
        "tif",
        "tiff",
        "webp",
    }

    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dims: int = 2048,
        num_workers: int = 1,
        batch_size: int = 1,
    ):
        # https://github.com/mseitzer/pytorch-fid/src/pytorch_fid/fid_score.py
        self.dims = dims
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([self.block_idx]).to(self.device)
        self.model = self.model.eval()

    def compute_fid(
        self,
        image_true: str,
        image_test: str,
    ):
        """Calculates the FID of two file paths
        FID = FrechetInceptionDistance()
        fid = FID.compute_fid("img_true.png", "img_test.png")
        """
        assert os.path.exists(image_true)
        assert os.path.exists(image_test)
        assert image_true.split(".")[-1] in self.IMAGE_EXTENSIONS
        assert image_test.split(".")[-1] in self.IMAGE_EXTENSIONS

        m1, s1 = calculate_activation_statistics(
            [image_true],
            self.model,
            1,
            self.dims,
            self.device,
            self.num_workers,
        )
        m2, s2 = calculate_activation_statistics(
            [image_test],
            self.model,
            1,
            self.dims,
            self.device,
            self.num_workers,
        )
        fid_value = calculate_frechet_distance(
            m1,
            s1,
            m2,
            s2,
        )

        return fid_value

    def compute_batch_fid(
        self,
        images_true: list[str],
        images_test: list[str],
    ):
        assert len(images_true) == len(images_test)
        for image_true, image_test in zip(images_true, images_test):
            assert os.path.exists(image_true)
            assert os.path.exists(image_test)
            assert image_true.split(".")[-1] in self.IMAGE_EXTENSIONS
            assert image_test.split(".")[-1] in self.IMAGE_EXTENSIONS

        batch_size = min(self.batch_size, len(images_true))
        m1, s1 = calculate_activation_statistics(
            images_true,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
        )
        m2, s2 = calculate_activation_statistics(
            images_test,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
        )
        fid_value = calculate_frechet_distance(
            m1,
            s1,
            m2,
            s2,
        )

        return fid_value


# Entrypoints
def get_args():
    parser = argparse.ArgumentParser(
        description="CacheDiT's Metrics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--img-true",
        type=str,
        default=None,
        help="Path to ground truth image",
    )
    parser.add_argument(
        "--img-test",
        type=str,
        default=None,
        help="Path to predicted image",
    )
    parser.add_argument(
        "--video-true",
        type=str,
        default=None,
        help="Path to ground truth video",
    )
    parser.add_argument(
        "--video-test",
        type=str,
        default=None,
        help="Path to predicted video",
    )
    parser.add_argument(
        "--compute-fid",
        "--fid",
        action="store_true",
        default=False,
        help="Compute FID for image",
    )

    return parser.parse_args()


def main():
    args = get_args()
    logger.info(args)

    if args.img_true is not None and args.img_test is not None:
        if any(
            (
                not os.path.exists(args.img_true),
                not os.path.exists(args.img_test),
            )
        ):
            return
        img_psnr = compute_psnr(args.img_true, args.img_test)
        logger.info(f"{args.img_true} vs {args.img_test}, PSNR: {img_psnr}")
        if args.compute_fid:
            FID = FrechetInceptionDistance()
            img_fid = FID.compute_fid(args.img_true, args.img_test)
            logger.info(f"{args.img_true} vs {args.img_test}, FID: {img_fid}")
    if args.video_true is not None and args.video_test is not None:
        if any(
            (
                not os.path.exists(args.video_true),
                not os.path.exists(args.video_test),
            )
        ):
            return
        video_psnr = compute_video_psnr(args.video_true, args.video_test)
        logger.info(
            f"{args.video_true} vs {args.video_test}, PSNR: {video_psnr}"
        )


if __name__ == "__main__":
    main()
