import os
import cv2
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from cache_dit.metrics.fid import FrechetInceptionDistance
from cache_dit.metrics.config import set_metrics_verbose
from cache_dit.metrics.config import get_metrics_verbose
from cache_dit.metrics.config import _IMAGE_EXTENSIONS
from cache_dit.metrics.config import _VIDEO_EXTENSIONS
from cache_dit.logger import init_logger

logger = init_logger(__name__)


DISABLE_VERBOSE = not get_metrics_verbose()


def compute_psnr_file(
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


def compute_mse_file(
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


def compute_ssim_file(
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


def compute_dir_metric(
    image_true_dir: np.ndarray | str,
    image_test_dir: np.ndarray | str,
    compute_file_func: callable = compute_psnr_file,
) -> float:
    # Image
    if isinstance(image_true_dir, np.ndarray) or isinstance(
        image_test_dir, np.ndarray
    ):
        return compute_file_func(image_true_dir, image_test_dir), 1
    # File
    if not os.path.isdir(image_true_dir) or not os.path.isdir(image_test_dir):
        return compute_file_func(image_true_dir, image_test_dir), 1
    # Dir
    image_true_dir: pathlib.Path = pathlib.Path(image_true_dir)
    image_true_files = sorted(
        [
            file
            for ext in _IMAGE_EXTENSIONS
            for file in image_true_dir.rglob("*.{}".format(ext))
        ]
    )
    image_test_dir: pathlib.Path = pathlib.Path(image_test_dir)
    image_test_files = sorted(
        [
            file
            for ext in _IMAGE_EXTENSIONS
            for file in image_test_dir.rglob("*.{}".format(ext))
        ]
    )
    image_true_files = [file.as_posix() for file in image_true_files]
    image_test_files = [file.as_posix() for file in image_test_files]

    # select valid files
    image_true_files_selected = []
    image_test_files_selected = []
    for i in range(min(len(image_true_files), len(image_test_files))):
        selected_image_true = image_true_files[i]
        selected_image_test = image_test_files[i]
        # Image pair must have the same basename
        if os.path.basename(selected_image_test) == os.path.basename(
            selected_image_true
        ):
            image_true_files_selected.append(selected_image_true)
            image_test_files_selected.append(selected_image_test)
    image_true_files = image_true_files_selected.copy()
    image_test_files = image_test_files_selected.copy()
    if len(image_true_files) == 0:
        logger.error(
            "No valid Image pairs, please note that Image "
            "pairs must have the same basename."
        )
        return None, None

    logger.debug(f"image_true_files: {image_true_files}")
    logger.debug(f"image_test_files: {image_test_files}")

    total_metric = 0.0
    valid_files = 0
    for image_true, image_test in tqdm(
        zip(image_true_files, image_test_files),
        total=len(image_true_files),
        disable=DISABLE_VERBOSE,
    ):
        metric = compute_file_func(image_true, image_test)
        if metric != float("inf"):
            total_metric += metric
            valid_files += 1

    if valid_files > 0:
        average_metric = total_metric / valid_files
        logger.debug(f"Average: {average_metric:.2f}")
        return average_metric, valid_files
    else:
        logger.debug("No valid files to compare")
        return None, None


def _fetch_video_frames(
    video_true: str,
    video_test: str,
):
    cap1 = cv2.VideoCapture(video_true)
    cap2 = cv2.VideoCapture(video_test)

    if not cap1.isOpened() or not cap2.isOpened():
        logger.error("Could not open video files")
        return [], [], 0

    frame_count = min(
        int(cap1.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap2.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    valid_frames = 0
    video_true_frames = []
    video_test_frames = []

    logger.debug(f"Total frames: {frame_count}")

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            break

        video_true_frames.append(frame1)
        video_test_frames.append(frame2)

        valid_frames += 1

    cap1.release()
    cap2.release()

    if valid_frames <= 0:
        return [], [], 0

    return video_true_frames, video_test_frames, valid_frames


def compute_video_metric(
    video_true: str,
    video_test: str,
    compute_frame_func: callable = compute_psnr_file,
) -> float:
    """
    video_true = "video_true.mp4"
    video_test = "video_test.mp4"
    PSNR = compute_video_psnr(video_true, video_test)
    """
    if os.path.isfile(video_true) and os.path.isfile(video_test):
        video_true_frames, video_test_frames, valid_frames = (
            _fetch_video_frames(
                video_true=video_true,
                video_test=video_test,
            )
        )
    elif os.path.isdir(video_true) and os.path.isdir(video_test):
        # Glob videos
        video_true_dir: pathlib.Path = pathlib.Path(video_true)
        video_true_files = sorted(
            [
                file
                for ext in _VIDEO_EXTENSIONS
                for file in video_true_dir.rglob("*.{}".format(ext))
            ]
        )
        video_test_dir: pathlib.Path = pathlib.Path(video_test)
        video_test_files = sorted(
            [
                file
                for ext in _VIDEO_EXTENSIONS
                for file in video_test_dir.rglob("*.{}".format(ext))
            ]
        )
        video_true_files = [file.as_posix() for file in video_true_files]
        video_test_files = [file.as_posix() for file in video_test_files]

        # select valid video files
        video_true_files_selected = []
        video_test_files_selected = []
        for i in range(min(len(video_true_files), len(video_test_files))):
            selected_video_true = video_true_files[i]
            selected_video_test = video_test_files[i]
            # Video pair must have the same basename
            if os.path.basename(selected_video_test) == os.path.basename(
                selected_video_true
            ):
                video_true_files_selected.append(selected_video_true)
                video_test_files_selected.append(selected_video_test)

        video_true_files = video_true_files_selected.copy()
        video_test_files = video_test_files_selected.copy()
        if len(video_true_files) == 0:
            logger.error(
                "No valid Video pairs, please note that Video "
                "pairs must have the same basename."
            )
            return None, None
        logger.debug(f"video_true_files: {video_true_files}")
        logger.debug(f"video_test_files: {video_test_files}")

        # Fetch all frames
        video_true_frames = []
        video_test_frames = []
        valid_frames = 0

        for video_true_, video_test_ in zip(video_true_files, video_test_files):
            video_true_frames_, video_test_frames_, valid_frames_ = (
                _fetch_video_frames(
                    video_true=video_true_, video_test=video_test_
                )
            )
            video_true_frames.extend(video_true_frames_)
            video_test_frames.extend(video_test_frames_)
            valid_frames += valid_frames_
    else:
        raise ValueError("video_true and video_test must be files or dirs.")

    if valid_frames <= 0:
        logger.debug("No valid frames to compare")
        return None, None

    total_metric = 0.0
    valid_frames = 0  # reset
    for frame1, frame2 in tqdm(
        zip(video_true_frames, video_test_frames),
        total=len(video_true_frames),
        disable=DISABLE_VERBOSE,
    ):
        metric = compute_frame_func(frame1, frame2)
        if metric != float("inf"):
            total_metric += metric
            valid_frames += 1

    if valid_frames > 0:
        average_metric = total_metric / valid_frames
        logger.debug(f"Average: {average_metric:.2f}")
        return average_metric, valid_frames
    else:
        logger.debug("No valid frames to compare")
        return None, None


compute_psnr = partial(
    compute_dir_metric,
    compute_file_func=compute_psnr_file,
)

compute_ssim = partial(
    compute_dir_metric,
    compute_file_func=compute_ssim_file,
)

compute_mse = partial(
    compute_dir_metric,
    compute_file_func=compute_mse_file,
)

compute_video_psnr = partial(
    compute_video_metric,
    compute_frame_func=compute_psnr_file,
)
compute_video_ssim = partial(
    compute_video_metric,
    compute_frame_func=compute_ssim_file,
)
compute_video_mse = partial(
    compute_video_metric,
    compute_frame_func=compute_mse_file,
)


# Entrypoints
def get_args():
    parser = argparse.ArgumentParser(
        description="CacheDiT's Metrics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    METRICS_CHOICES = [
        "psnr",
        "ssim",
        "mse",
        "fid",
        "all",
    ]
    parser.add_argument(
        "metrics",
        type=str,
        nargs="+",
        default="psnr",
        choices=METRICS_CHOICES,
        help=f"Metric choices: {METRICS_CHOICES}",
    )
    parser.add_argument(
        "--img-true",
        "-i1",
        type=str,
        default=None,
        help="Path to ground truth image or Dir to ground truth images",
    )
    parser.add_argument(
        "--img-test",
        "-i2",
        type=str,
        default=None,
        help="Path to predicted image or Dir to predicted images",
    )
    parser.add_argument(
        "--video-true",
        "-v1",
        type=str,
        default=None,
        help="Path to ground truth video or Dir to ground truth videos",
    )
    parser.add_argument(
        "--video-test",
        "-v2",
        type=str,
        default=None,
        help="Path to predicted video or Dir to predicted videos",
    )
    parser.add_argument(
        "--enable-verbose",
        "-verbose",
        action="store_true",
        default=False,
        help="Show metrics progress verbose",
    )
    return parser.parse_args()


def entrypoint():
    args = get_args()
    logger.debug(args)

    if args.enable_verbose:
        global DISABLE_VERBOSE
        set_metrics_verbose(True)
        DISABLE_VERBOSE = not get_metrics_verbose()

    # run one metric
    def _run_metric(mertric_type: str) -> None:
        mertric_type = mertric_type.lower()
        if args.img_true is not None and args.img_test is not None:
            if any(
                (
                    not os.path.exists(args.img_true),
                    not os.path.exists(args.img_test),
                )
            ):
                return
            # img_true and img_test can be files or dirs
            if mertric_type == "psnr" or mertric_type == "all":
                img_psnr, n = compute_psnr(args.img_true, args.img_test)
                logger.info(
                    f"{args.img_true} vs {args.img_test}, Num: {n}, PSNR: {img_psnr}"
                )
            if mertric_type == "ssim" or mertric_type == "all":
                img_ssim, n = compute_ssim(args.img_true, args.img_test)
                logger.info(
                    f"{args.img_true} vs {args.img_test}, Num: {n}, SSIM: {img_ssim}"
                )
            if mertric_type == "mse" or mertric_type == "all":
                img_mse, n = compute_mse(args.img_true, args.img_test)
                logger.info(
                    f"{args.img_true} vs {args.img_test}, Num: {n},  MSE: {img_mse}"
                )
            if mertric_type == "fid" or mertric_type == "all":
                FID = FrechetInceptionDistance(disable_tqdm=DISABLE_VERBOSE)
                img_fid, n = FID.compute_fid(args.img_true, args.img_test)
                logger.info(
                    f"{args.img_true} vs {args.img_test}, Num: {n},  FID: {img_fid}"
                )
        if args.video_true is not None and args.video_test is not None:
            if any(
                (
                    not os.path.exists(args.video_true),
                    not os.path.exists(args.video_test),
                )
            ):
                return
            # video_true and video_test can be files or dirs
            if mertric_type == "psnr" or mertric_type == "all":
                video_psnr, n = compute_video_psnr(
                    args.video_true, args.video_test
                )
                logger.info(
                    f"{args.video_true} vs {args.video_test}, Frames: {n}, PSNR: {video_psnr}"
                )
            if mertric_type == "ssim" or mertric_type == "all":
                video_ssim, n = compute_video_ssim(
                    args.video_true, args.video_test
                )
                logger.info(
                    f"{args.video_true} vs {args.video_test}, Frames: {n}, SSIM: {video_ssim}"
                )
            if mertric_type == "mse" or mertric_type == "all":
                video_mse, n = compute_video_mse(
                    args.video_true, args.video_test
                )
                logger.info(
                    f"{args.video_true} vs {args.video_test}, Frames: {n},  MSE: {video_mse}"
                )
            if mertric_type == "fid" or mertric_type == "all":
                FID = FrechetInceptionDistance(disable_tqdm=DISABLE_VERBOSE)
                video_fid, n = FID.compute_video_fid(
                    args.video_true, args.video_test
                )
                logger.info(
                    f"{args.video_true} vs {args.video_test}, Frames: {n},  FID: {video_fid}"
                )

    # run selected metrics
    logger.info(f"Selected metrics: {args.metrics}")
    for metric in args.metrics:
        _run_metric(mertric_type=metric)


if __name__ == "__main__":
    entrypoint()
