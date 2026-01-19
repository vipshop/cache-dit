import os
import re
import cv2
import pathlib
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial
from typing import Callable, Union, Tuple, List
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from .config import set_metrics_verbose
from .config import get_metrics_verbose
from .config import _IMAGE_EXTENSIONS
from .config import _VIDEO_EXTENSIONS
from .fid import compute_fid
from .fid import compute_video_fid
from .lpips import compute_lpips_img
from .clip_score import compute_clip_score
from .image_reward import compute_reward_score
from cache_dit.logger import init_logger

logger = init_logger(__name__)


DISABLE_VERBOSE = not get_metrics_verbose()
PSNR_TYPE = "custom"


def compute_lpips_file(
    image_true: np.ndarray | str,
    image_test: np.ndarray | str,
) -> float:
    import torch
    from PIL import Image
    from torchvision.transforms.v2.functional import (
        convert_image_dtype,
        normalize,
        pil_to_tensor,
    )

    def load_img_as_tensor(path):
        pil = Image.open(path)
        img = pil_to_tensor(pil)
        img = convert_image_dtype(img, dtype=torch.float32)
        img = normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        return img

    if isinstance(image_true, str):
        image_true = load_img_as_tensor(image_true)
    if isinstance(image_test, str):
        image_test = load_img_as_tensor(image_test)
    return compute_lpips_img(
        image_true,
        image_test,
    )


def set_psnr_type(psnr_type: str):
    global PSNR_TYPE
    PSNR_TYPE = psnr_type
    assert PSNR_TYPE in ["skimage", "custom"]


def get_psnr_type():
    global PSNR_TYPE
    return PSNR_TYPE


def calculate_psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        image_true (ndarray): Images with range [0, 255].
        image_test (ndarray): Images with range [0, 255].
    """
    mse = np.mean((image_true - image_test) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


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
    if get_psnr_type() == "skimage":
        return peak_signal_noise_ratio(
            image_true,
            image_test,
        )
    else:
        return calculate_psnr(image_true, image_test)


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
) -> Union[Tuple[float, int], Tuple[None, None]]:
    # Image
    if isinstance(image_true_dir, np.ndarray) or isinstance(image_test_dir, np.ndarray):
        return compute_file_func(image_true_dir, image_test_dir), 1
    # File
    if not os.path.isdir(image_true_dir) or not os.path.isdir(image_test_dir):
        return compute_file_func(image_true_dir, image_test_dir), 1

    # Dir
    # compute dir metric
    def natural_sort_key(filename):
        match = re.search(r"(\d+)\D*$", filename)
        return int(match.group(1)) if match else filename

    image_true_dir: pathlib.Path = pathlib.Path(image_true_dir)
    image_true_files = [
        file for ext in _IMAGE_EXTENSIONS for file in image_true_dir.rglob("*.{}".format(ext))
    ]
    image_true_files = [file.as_posix() for file in image_true_files]
    image_true_files = sorted(image_true_files, key=natural_sort_key)

    image_test_dir: pathlib.Path = pathlib.Path(image_test_dir)
    image_test_files = [
        file for ext in _IMAGE_EXTENSIONS for file in image_test_dir.rglob("*.{}".format(ext))
    ]
    image_test_files = [file.as_posix() for file in image_test_files]
    image_test_files = sorted(image_test_files, key=natural_sort_key)

    # select valid files
    image_true_files_selected = []
    image_test_files_selected = []
    for i in range(min(len(image_true_files), len(image_test_files))):
        selected_image_true = image_true_files[i]
        selected_image_test = image_test_files[i]
        # Image pair must have the same basename
        if os.path.basename(selected_image_test) == os.path.basename(selected_image_true):
            image_true_files_selected.append(selected_image_true)
            image_test_files_selected.append(selected_image_test)

    image_true_files = image_true_files_selected.copy()
    image_test_files = image_test_files_selected.copy()
    if len(image_true_files) == 0:
        logger.error(
            "No valid Image pairs, please note that Image " "pairs must have the same basename."
        )
        return None, None

    logger.debug(f"image_true_files: {image_true_files}")
    logger.debug(f"image_test_files: {image_test_files}")

    total_metric = 0.0
    valid_files = 0
    total_files = 0
    for image_true, image_test in tqdm(
        zip(image_true_files, image_test_files),
        total=len(image_true_files),
        disable=DISABLE_VERBOSE,
    ):
        metric = compute_file_func(image_true, image_test)
        if metric != float("inf"):  # means no cache apply to image_test
            total_metric += metric
            valid_files += 1
        total_files += 1

    if valid_files > 0:
        average_metric = total_metric / valid_files
        logger.debug(f"Average: {average_metric:.2f}")
        return average_metric, total_files
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
) -> Union[Tuple[float, int], Tuple[None, None]]:
    """
    video_true = "video_true.mp4"
    video_test = "video_test.mp4"
    PSNR = compute_video_psnr(video_true, video_test)
    """
    if os.path.isfile(video_true) and os.path.isfile(video_test):
        video_true_frames, video_test_frames, valid_frames = _fetch_video_frames(
            video_true=video_true,
            video_test=video_test,
        )
    elif os.path.isdir(video_true) and os.path.isdir(video_test):
        # Glob videos
        video_true_dir: pathlib.Path = pathlib.Path(video_true)
        video_true_files = sorted(
            [file for ext in _VIDEO_EXTENSIONS for file in video_true_dir.rglob("*.{}".format(ext))]
        )
        video_test_dir: pathlib.Path = pathlib.Path(video_test)
        video_test_files = sorted(
            [file for ext in _VIDEO_EXTENSIONS for file in video_test_dir.rglob("*.{}".format(ext))]
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
            if os.path.basename(selected_video_test) == os.path.basename(selected_video_true):
                video_true_files_selected.append(selected_video_true)
                video_test_files_selected.append(selected_video_test)

        video_true_files = video_true_files_selected.copy()
        video_test_files = video_test_files_selected.copy()
        if len(video_true_files) == 0:
            logger.error(
                "No valid Video pairs, please note that Video " "pairs must have the same basename."
            )
            return None, None
        logger.debug(f"video_true_files: {video_true_files}")
        logger.debug(f"video_test_files: {video_test_files}")

        # Fetch all frames
        video_true_frames = []
        video_test_frames = []
        valid_frames = 0

        for video_true_, video_test_ in zip(video_true_files, video_test_files):
            video_true_frames_, video_test_frames_, valid_frames_ = _fetch_video_frames(
                video_true=video_true_, video_test=video_test_
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


compute_lpips: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_dir_metric,
    compute_file_func=compute_lpips_file,
)

compute_psnr: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_dir_metric,
    compute_file_func=compute_psnr_file,
)

compute_ssim: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_dir_metric,
    compute_file_func=compute_ssim_file,
)

compute_mse: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_dir_metric,
    compute_file_func=compute_mse_file,
)

compute_video_lpips: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_video_metric,
    compute_frame_func=compute_lpips_file,
)
compute_video_psnr: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_video_metric,
    compute_frame_func=compute_psnr_file,
)
compute_video_ssim: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_video_metric,
    compute_frame_func=compute_ssim_file,
)
compute_video_mse: Callable[..., Union[Tuple[float, int], Tuple[None, None]]] = partial(
    compute_video_metric,
    compute_frame_func=compute_mse_file,
)


METRICS_CHOICES = [
    "lpips",  # img vs img
    "psnr",  # img vs img
    "ssim",  # img vs img
    "mse",  # img vs img
    "fid",  # img vs img
    "all",  # img vs img
    "clip_score",  # img vs prompt
    "image_reward",  # img vs prompt
]


# Entrypoints
def get_args():
    global METRICS_CHOICES
    parser = argparse.ArgumentParser(
        description="CacheDiT's Metrics CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
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
        "--prompt-true",
        "-p",
        type=str,
        default=None,
        help="Path to ground truth prompt file for CLIP Score and Image Reward Score.",
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

    # Image 1 vs N pattern
    parser.add_argument(
        "--img-source-dir",
        "-d",
        type=str,
        default=None,
        help="Path to dir that contains dirs of images",
    )
    parser.add_argument(
        "--ref-img-dir",
        "-r",
        type=str,
        default=None,
        help="Path to ref dir that contains ground truth images",
    )
    parser.add_argument(
        "--ref-prompt-true",
        "-rp",
        type=str,
        default=None,
        help="Path to ground truth prompt file for CLIP Score and Image Reward Score.",
    )

    # Video 1 vs N pattern
    parser.add_argument(
        "--video-source-dir",
        "-vd",
        type=str,
        default=None,
        help="Path to dir that contains many videos",
    )
    parser.add_argument(
        "--ref-video",
        "-rv",
        type=str,
        default=None,
        help="Path to ground truth video",
    )

    # FID batch size
    parser.add_argument(
        "--fid-batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for FID compute",
    )

    # Verbose
    parser.add_argument(
        "--enable-verbose",
        "-verbose",
        action="store_true",
        default=False,
        help="Show metrics progress verbose",
    )

    # Format output
    parser.add_argument(
        "--summary",
        "-s",
        action="store_true",
        default=False,
        help="Summary the outupt metrics results",
    )

    # Addtional perf log
    parser.add_argument(
        "--perf-log",
        "-plog",
        type=str,
        default=None,
        help="Path to addtional perf log",
    )
    parser.add_argument(
        "--perf-tags",
        "-ptags",
        nargs="+",
        type=str,
        default=[],
        help="Tag to parse perf time from perf log",
    )
    parser.add_argument(
        "--extra-perf-tags",
        "-extra-ptags",
        nargs="+",
        default=[],
        help="Extra tags to parse perf time from perf log",
    )
    parser.add_argument(
        "--psnr-type",
        type=str,
        default="custom",
        choices=["custom", "skimage"],
        help="The compute type of PSNR, [custom, skimage]",
    )
    parser.add_argument(
        "--cal-speedup",
        action="store_true",
        default=False,
        help="Calculate performance speedup.",
    )
    parser.add_argument(
        "--gen-markdown-table",
        "-table",
        action="store_true",
        default=False,
        help="Generate performance markdown table",
    )
    return parser.parse_args()


def entrypoint():
    global METRICS_CHOICES
    args = get_args()
    logger.debug(args)

    if args.metrics in ["clip_score", "image_reward"]:
        assert args.prompt_true is not None or args.ref_prompt_true is not None
        assert args.img_test is not None or args.img_source_dir is not None

    if args.enable_verbose:
        global DISABLE_VERBOSE
        set_metrics_verbose(True)
        DISABLE_VERBOSE = not get_metrics_verbose()

    set_psnr_type(args.psnr_type)

    METRICS_META: dict[str, float] = {}

    # run one metric
    def _run_metric(
        metric: str,
        img_true: str = None,
        prompt_true: str = None,
        img_test: str = None,
        video_true: str = None,
        video_test: str = None,
    ) -> None:
        nonlocal METRICS_META
        metric = metric.lower()
        if img_true is not None and img_test is not None:
            if any(
                (
                    not os.path.exists(img_true),
                    not os.path.exists(img_test),
                )
            ):
                logger.error(f"Not exist: {img_true} or {img_test}, skip.")
                return
            # img_true and img_test can be files or dirs
            img_true_info = os.path.basename(img_true)
            img_test_info = os.path.basename(img_test)

            def _logging_msg(value: float, name, n: int):
                if value is None or n is None:
                    return
                msg = (
                    f"{img_true_info} vs {img_test_info}, " f"Num: {n}, {name.upper()}: {value:.5f}"
                )
                METRICS_META[msg] = value
                logger.info(msg)

            if metric == "lpips" or metric == "all":
                img_lpips, n = compute_lpips(img_true, img_test)
                _logging_msg(img_lpips, "lpips", n)
            if metric == "psnr" or metric == "all":
                img_psnr, n = compute_psnr(img_true, img_test)
                _logging_msg(img_psnr, "psnr", n)
            if metric == "ssim" or metric == "all":
                img_ssim, n = compute_ssim(img_true, img_test)
                _logging_msg(img_ssim, "ssim", n)
            if metric == "mse" or metric == "all":
                img_mse, n = compute_mse(img_true, img_test)
                _logging_msg(img_mse, "mse", n)
            if metric == "fid" or metric == "all":
                img_fid, n = compute_fid(img_true, img_test)
                _logging_msg(img_fid, "fid", n)

        if prompt_true is not None and img_test is not None:
            if any(
                (
                    not os.path.exists(prompt_true),  # file
                    not os.path.exists(img_test),  # dir
                )
            ):
                logger.error(f"Not exist: {prompt_true} or {img_test}, skip.")
                return

            # img_true and img_test can be files or dirs
            prompt_true_info = os.path.basename(prompt_true)
            img_test_info = os.path.basename(img_test)

            def _logging_msg(value: float, name, n: int):
                if value is None or n is None:
                    return
                msg = (
                    f"{prompt_true_info} vs {img_test_info}, "
                    f"Num: {n}, {name.upper()}: {value:.5f}"
                )
                METRICS_META[msg] = value
                logger.info(msg)

            if metric == "clip_score":
                clip_score, n = compute_clip_score(img_test, prompt_true)
                _logging_msg(clip_score, "clip_score", n)
            if metric == "image_reward":
                image_reward, n = compute_reward_score(img_test, prompt_true)
                _logging_msg(image_reward, "image_reward", n)

        if video_true is not None and video_test is not None:
            if any(
                (
                    not os.path.exists(video_true),
                    not os.path.exists(video_test),
                )
            ):
                logger.error(f"Not exist: {video_true} or {video_test}, skip.")
                return

            # video_true and video_test can be files or dirs
            video_true_info = os.path.basename(video_true)
            video_test_info = os.path.basename(video_test)

            def _logging_msg(value: float, name, n: int):
                if value is None or n is None:
                    return
                msg = (
                    f"{video_true_info} vs {video_test_info}, "
                    f"Frames: {n}, {name.upper()}: {value:.5f}"
                )
                METRICS_META[msg] = value
                logger.info(msg)

            if metric == "lpips" or metric == "all":
                video_lpips, n = compute_video_lpips(video_true, video_test)
                _logging_msg(video_lpips, "lpips", n)
            if metric == "psnr" or metric == "all":
                video_psnr, n = compute_video_psnr(video_true, video_test)
                _logging_msg(video_psnr, "psnr", n)
            if metric == "ssim" or metric == "all":
                video_ssim, n = compute_video_ssim(video_true, video_test)
                _logging_msg(video_ssim, "ssim", n)
            if metric == "mse" or metric == "all":
                video_mse, n = compute_video_mse(video_true, video_test)
                _logging_msg(video_mse, "mse", n)
            if metric == "fid" or metric == "all":
                video_fid, n = compute_video_fid(video_true, video_test)
                _logging_msg(video_fid, "fid", n)

    # run selected metrics
    if not DISABLE_VERBOSE:
        logger.info(f"Selected metrics: {args.metrics}")

    def _is_image_1vsN_pattern() -> bool:
        return args.img_source_dir is not None and args.ref_img_dir is not None

    def _is_video_1vsN_pattern() -> bool:
        return args.video_source_dir is not None and args.ref_video is not None

    def _is_prompt_1vsN_pattern() -> bool:
        return args.img_source_dir is not None and args.ref_prompt_true is not None

    assert not all(
        (
            _is_image_1vsN_pattern(),
            _is_video_1vsN_pattern(),
            _is_prompt_1vsN_pattern(),
        )
    )

    if _is_image_1vsN_pattern():
        # Glob Image dirs
        if not os.path.exists(args.img_source_dir):
            logger.error(f"{args.img_source_dir} not exist!")
            return
        if not os.path.exists(args.ref_img_dir):
            logger.error(f"{args.ref_img_dir} not exist!")
            return

        directories = []
        for item in os.listdir(args.img_source_dir):
            item_path = os.path.join(args.img_source_dir, item)
            if os.path.isdir(item_path):
                if os.path.basename(item_path) == os.path.basename(args.ref_img_dir):
                    continue
                directories.append(item_path)

        if len(directories) == 0:
            return

        directories = sorted(directories)
        if not DISABLE_VERBOSE:
            logger.info(
                f"Compare {args.ref_img_dir} vs {directories}, " f"Num compares: {len(directories)}"
            )

        for metric in args.metrics:
            for img_test_dir in directories:
                _run_metric(
                    metric=metric,
                    img_true=args.ref_img_dir,
                    img_test=img_test_dir,
                )

    elif _is_video_1vsN_pattern():
        # Glob videos
        if not os.path.exists(args.video_source_dir):
            logger.error(f"{args.video_source_dir} not exist!")
            return
        if not os.path.exists(args.ref_video):
            logger.error(f"{args.ref_video} not exist!")
            return

        video_source_dir: pathlib.Path = pathlib.Path(args.video_source_dir)
        video_source_files = sorted(
            [
                file
                for ext in _VIDEO_EXTENSIONS
                for file in video_source_dir.rglob("*.{}".format(ext))
            ]
        )
        video_source_files = [file.as_posix() for file in video_source_files]

        video_source_selected = []
        for video_source_file in video_source_files:
            if os.path.basename(video_source_file) == os.path.basename(args.ref_video):
                continue
            video_source_selected.append(video_source_file)

        if len(video_source_selected) == 0:
            return

        video_source_selected = sorted(video_source_selected)
        if not DISABLE_VERBOSE:
            logger.info(
                f"Compare {args.ref_video} vs {video_source_selected}, "
                f"Num compares: {len(video_source_selected)}"
            )

        for metric in args.metrics:
            for video_test in video_source_selected:
                _run_metric(
                    metric=metric,
                    video_true=args.ref_video,
                    video_test=video_test,
                )

    elif _is_prompt_1vsN_pattern():
        # Glob Image dirs
        if not os.path.exists(args.img_source_dir):
            logger.error(f"{args.img_source_dir} not exist!")
            return

        directories = []
        for item in os.listdir(args.img_source_dir):
            item_path = os.path.join(args.img_source_dir, item)
            if os.path.isdir(item_path):
                directories.append(item_path)

        if len(directories) == 0:
            return

        directories = sorted(directories)
        if not DISABLE_VERBOSE:
            logger.info(
                f"Compare {args.ref_prompt_true} vs {directories}, "
                f"Num compares: {len(directories)}"
            )

        for metric in args.metrics:
            for img_test_dir in directories:
                _run_metric(
                    metric=metric,
                    prompt_true=args.ref_prompt_true,
                    img_test=img_test_dir,
                )

    else:
        for metric in args.metrics:
            _run_metric(
                metric=metric,
                img_true=args.img_true,
                prompt_true=args.prompt_true,
                img_test=args.img_test,
                video_true=args.video_true,
                video_test=args.video_test,
            )

    if args.summary:

        def _fetch_perf():
            if args.perf_log is None or len(args.perf_tags) == 0:
                return []
            if not os.path.exists(args.perf_log):
                return []
            perf_texts = []
            with open(args.perf_log, "r") as file:
                perf_lines = file.readlines()
                for line in perf_lines:
                    line = line.strip()
                    for perf_tag in args.perf_tags:
                        if perf_tag.lower() in line.lower():
                            if len(args.extra_perf_tags) == 0:
                                perf_texts.append(line)
                                break
                            else:
                                has_all_extra_tag = True
                                for ext_tag in args.extra_perf_tags:
                                    if ext_tag.lower() not in line.lower():
                                        has_all_extra_tag = False
                                        break
                                if has_all_extra_tag:
                                    perf_texts.append(line)
                                    break
            return perf_texts

        PERF_TEXTS: list[str] = _fetch_perf()

        def _parse_value(
            text: str,
            tag: str = "Num",
        ) -> float | None:
            import re

            escaped_tag = re.escape(tag)
            processed_tag = escaped_tag.replace(r"\ ", r"\s+")

            pattern = re.compile(rf"{processed_tag}:\s*(\d+\.?\d*)\D*", re.IGNORECASE)

            match = pattern.search(text)

            if not match:
                return None

            value_str = match.group(1)
            try:
                if tag.lower() in METRICS_CHOICES:
                    return float(value_str)
                if len(args.perf_tags) > 0:
                    perf_tags = [tag.lower() for tag in args.perf_tags]
                    if tag.lower() in perf_tags:
                        return float(value_str)
                return int(value_str)
            except ValueError:
                return None

        def _parse_perf(
            compare_tag: str,
            perf_tag: str,
        ) -> float | None:
            nonlocal PERF_TEXTS
            perf_values = []
            for line in PERF_TEXTS:
                if compare_tag in line:
                    perf_value = _parse_value(line, perf_tag)
                    if perf_value is not None:
                        perf_values.append(perf_value)
            if len(perf_values) == 0:
                return None
            return sum(perf_values) / len(perf_values)

        def _ref_perf(
            key: str,
        ):
            # U1-Q0-C0-NONE vs U4-Q1-C1-NONE
            header = key.split(",")[0].strip()
            reference_tag = None
            if args.prompt_true is None:
                reference_tag = header.split("vs")[0].strip()  # U1-Q0-C0-NONE

            if reference_tag is None:
                return []

            ref_perf_values = []
            for perf_tag in args.perf_tags:
                perf_value = _parse_perf(reference_tag, perf_tag)
                ref_perf_values.append(perf_value)

            return ref_perf_values

        def _format_item(
            key: str,
            metric: str,
            value: float,
            max_key_len: int,
        ):
            nonlocal PERF_TEXTS
            # U1-Q0-C0-NONE vs U4-Q1-C1-NONE
            header = key.split(",")[0].strip()
            compare_tag = header.split("vs")[1].strip()  # U4-Q1-C1-NONE
            has_perf_texts = len(PERF_TEXTS) > 0

            def _perf_msg(perf_tag: str):
                if "time" in perf_tag.lower():
                    perf_msg = "Latency(s)"
                elif "tflops" in perf_tag.lower():
                    perf_msg = "TFLOPs"
                elif "flops" in perf_tag.lower():
                    perf_msg = "FLOPs"
                else:
                    perf_msg = perf_tag.upper()
                return perf_msg

            format_str = ""
            # Num / Frames
            perf_values = []
            perf_msgs = []
            if n := _parse_value(key, "Num"):
                if not has_perf_texts:
                    format_str = (
                        f"{header:<{max_key_len}}, Num: {n}, " f"{metric.upper()}: {value:<7.4f}"
                    )
                else:
                    format_str = (
                        f"{header:<{max_key_len}}, Num: {n}, " f"{metric.upper()}: {value:<7.4f}, "
                    )
                    for perf_tag in args.perf_tags:
                        perf_value = _parse_perf(compare_tag, perf_tag)
                        perf_values.append(perf_value)

                        perf_value = f"{perf_value:<.2f}" if perf_value else None
                        perf_msg = _perf_msg(perf_tag)
                        format_str += f"{perf_msg}: {perf_value}, "

                        perf_msgs.append(perf_msg)

                    if not args.cal_speedup:
                        format_str = format_str.removesuffix(", ")

            elif n := _parse_value(key, "Frames"):
                if not has_perf_texts:
                    format_str = (
                        f"{header:<{max_key_len}}, Frames: {n}, " f"{metric.upper()}: {value:<7.4f}"
                    )
                else:
                    format_str = (
                        f"{header:<{max_key_len}}, Frames: {n}, "
                        f"{metric.upper()}: {value:<7.4f}, "
                    )
                    for perf_tag in args.perf_tags:
                        perf_value = _parse_perf(compare_tag, perf_tag)
                        perf_values.append(perf_value)

                        perf_value = f"{perf_value:<.2f}" if perf_value else None
                        perf_msg = _perf_msg(perf_tag)
                        format_str += f"{perf_msg}: {perf_value}, "
                        perf_msgs.append(perf_msg)

                    if not args.cal_speedup:
                        format_str = format_str.removesuffix(", ")
            else:
                raise ValueError("Num or Frames can not be NoneType.")

            return format_str, perf_values, perf_msgs

        def _format_table(format_strs: List[str], metric: str):
            if not format_strs:
                return ""

            metric_upper = metric.upper()
            all_headers = {"Config", metric_upper}
            row_data = []

            for line in format_strs:
                parts = [p.strip() for p in line.split(",")]

                config_part = parts[0].strip()
                if "vs" in config_part:
                    config = config_part.split("vs", 1)[1].strip()
                    if "_DBCACHE_" in config:
                        config = config.split("_DBCACHE_", 1)[1].strip()
                else:
                    config = config_part

                metric_value = next(
                    p.split(":")[1].strip() for p in parts if p.startswith(metric_upper)
                )

                perf_data = {}
                for part in parts:
                    if part.startswith(("Num:", "Frames:", metric_upper)):
                        continue
                    if ":" in part:
                        key, value = part.split(":", 1)
                        key = key.strip()
                        value = value.strip()
                        perf_data[key] = value
                        all_headers.add(key)

                row_data.append({"Config": config, metric_upper: metric_value, **perf_data})

            sorted_headers = ["Config", metric_upper] + sorted(
                [h for h in all_headers if h not in ["Config", metric_upper]]
            )

            table = "| " + " | ".join(sorted_headers) + " |\n"
            table += "| " + " | ".join(["---"] * len(sorted_headers)) + " |\n"

            for row in row_data:
                row_values = [row.get(header, "") for header in sorted_headers]
                table += "| " + " | ".join(row_values) + " |\n"

            return table.strip()

        selected_metrics = args.metrics
        if "all" in selected_metrics:
            selected_metrics = METRICS_CHOICES.copy()
            selected_metrics.remove("all")

        for metric in selected_metrics:
            selected_items = {}
            for key in METRICS_META.keys():
                if metric.upper() in key or metric.lower() in key:
                    selected_items[key] = METRICS_META[key]

            # skip unselected metric
            if len(selected_items) == 0:
                continue

            reverse = (
                True
                if metric.lower()
                in [
                    "psnr",
                    "ssim",
                    "clip_score",
                    "image_reward",
                ]
                else False
            )
            sorted_items = sorted(selected_items.items(), key=lambda x: x[1], reverse=reverse)
            selected_keys = [key.split(",")[0].strip() for key in selected_items.keys()]
            max_key_len = max(len(key) for key in selected_keys)

            ref_perf_values = _ref_perf(key=selected_keys[0])
            max_perf_values: List[float] = []

            if ref_perf_values and None not in ref_perf_values:
                max_perf_values = ref_perf_values.copy()

            for key, value in sorted_items:
                format_str, perf_values, perf_msgs = _format_item(key, metric, value, max_key_len)
                # skip 'None' msg but not 'NONE', 'NONE' means w/o cache
                if "None" in format_str:
                    continue

                if not perf_values or None in perf_values or not perf_msgs or not args.cal_speedup:
                    continue

                if not max_perf_values:
                    max_perf_values = perf_values
                else:
                    for i in range(len(max_perf_values)):
                        max_perf_values[i] = max(max_perf_values[i], perf_values[i])

            format_strs = []
            for key, value in sorted_items:
                format_str, perf_values, perf_msgs = _format_item(key, metric, value, max_key_len)

                # skip 'None' msg but not 'NONE', 'NONE' means w/o cache
                if "None" in format_str:
                    continue

                if (
                    not perf_values
                    or None in perf_values
                    or not perf_msgs
                    or not max_perf_values
                    or not args.cal_speedup
                ):
                    format_strs.append(format_str)
                    continue

                for perf_value, perf_msg, max_perf_value in zip(
                    perf_values, perf_msgs, max_perf_values
                ):
                    perf_speedup = max_perf_value / perf_value
                    format_str += f"{perf_msg}(↑): {perf_speedup:<.2f}, "

                format_str = format_str.removesuffix(", ")
                format_strs.append(format_str)

            format_len = max(len(format_str) for format_str in format_strs)

            res_len = format_len - len(f"Summary: {metric.upper()}")
            left_len = res_len // 2
            right_len = res_len - left_len
            print("-" * format_len)
            print(" " * left_len + f"Summary: {metric.upper()}" + " " * right_len)
            print("-" * format_len)
            for format_str in format_strs:
                print(format_str)
            print("-" * format_len)

            if args.gen_markdown_table:
                table = _format_table(format_strs, metric)
                table = table.replace("Latency(s)(↑)", "SpeedUp(↑)")
                table = table.replace("TFLOPs(↑)", "SpeedUp(↑)")
                table = table.replace("FLOPs(↑)", "SpeedUp(↑)")
                print("-" * format_len)
                print(f"{table}")
            print("-" * format_len)


if __name__ == "__main__":
    entrypoint()
