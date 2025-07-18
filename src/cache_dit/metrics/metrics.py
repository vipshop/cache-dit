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
        "--sort-output",
        "-sort",
        action="store_true",
        default=False,
        help="Sort the outupt metrics results",
    )
    return parser.parse_args()


def entrypoint():
    args = get_args()
    logger.debug(args)

    if args.enable_verbose:
        global DISABLE_VERBOSE
        set_metrics_verbose(True)
        DISABLE_VERBOSE = not get_metrics_verbose()

    if "all" in args.metrics or "fid" in args.metrics:
        FID = FrechetInceptionDistance(
            disable_tqdm=DISABLE_VERBOSE,
            batch_size=args.fid_batch_size,
        )

    METRICS_META: dict[str, float] = {}

    # run one metric
    def _run_metric(
        metric: str,
        img_true: str = None,
        img_test: str = None,
        video_true: str = None,
        video_test: str = None,
    ) -> None:
        nonlocal FID
        nonlocal METRICS_META
        metric = metric.lower()
        if img_true is not None and img_test is not None:
            if any(
                (
                    not os.path.exists(img_true),
                    not os.path.exists(img_test),
                )
            ):
                return
            # img_true and img_test can be files or dirs
            img_true_info = os.path.basename(img_true)
            img_test_info = os.path.basename(img_test)

            def _logging_msg(value: float, n: int):
                if value is None or n is None:
                    return
                msg = (
                    f"{img_true_info} vs {img_test_info}, "
                    f"Num: {n}, {metric.upper()}: {value:.5f}"
                )
                METRICS_META[msg] = value
                logger.info(msg)

            if metric == "psnr" or metric == "all":
                img_psnr, n = compute_psnr(img_true, img_test)
                _logging_msg(img_psnr, n)
            if metric == "ssim" or metric == "all":
                img_ssim, n = compute_ssim(img_true, img_test)
                _logging_msg(img_ssim, n)
            if metric == "mse" or metric == "all":
                img_mse, n = compute_mse(img_true, img_test)
                _logging_msg(img_mse, n)
            if metric == "fid" or metric == "all":
                img_fid, n = FID.compute_fid(img_true, img_test)
                _logging_msg(img_fid, n)

        if video_true is not None and video_test is not None:
            if any(
                (
                    not os.path.exists(video_true),
                    not os.path.exists(video_test),
                )
            ):
                return

            # video_true and video_test can be files or dirs
            video_true_info = os.path.basename(video_true)
            video_test_info = os.path.basename(video_test)

            def _logging_msg(value: float, n: int):
                if value is None or n is None:
                    return
                msg = (
                    f"{video_true_info} vs {video_test_info}, "
                    f"Frames: {n}, {metric.upper()}: {value:.5f}"
                )
                METRICS_META[msg] = value
                logger.info(msg)

            if metric == "psnr" or metric == "all":
                video_psnr, n = compute_video_psnr(video_true, video_test)
                _logging_msg(video_psnr, n)
            if metric == "ssim" or metric == "all":
                video_ssim, n = compute_video_ssim(video_true, video_test)
                _logging_msg(video_ssim, n)
            if metric == "mse" or metric == "all":
                video_mse, n = compute_video_mse(video_true, video_test)
                _logging_msg(video_mse, n)
            if metric == "fid" or metric == "all":
                video_fid, n = FID.compute_video_fid(video_true, video_test)
                _logging_msg(video_fid, n)

    # run selected metrics
    if not DISABLE_VERBOSE:
        logger.info(f"Selected metrics: {args.metrics}")

    def _is_image_1vsN_pattern() -> bool:
        return args.img_source_dir is not None and args.ref_img_dir is not None

    def _is_video_1vsN_pattern() -> bool:
        return args.video_source_dir is not None and args.ref_video is not None

    assert not all((_is_image_1vsN_pattern(), _is_video_1vsN_pattern()))

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
                if os.path.basename(item_path) == os.path.basename(
                    args.ref_img_dir
                ):
                    continue
                directories.append(item_path)

        if len(directories) == 0:
            return

        directories = sorted(directories)
        if not DISABLE_VERBOSE:
            logger.info(
                f"Compare {args.ref_img_dir} vs {directories}, "
                f"Num compares: {len(directories)}"
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
            if os.path.basename(video_source_file) == os.path.basename(
                args.ref_video
            ):
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

    else:
        for metric in args.metrics:
            _run_metric(
                metric=metric,
                img_true=args.img_true,
                img_test=args.img_test,
                video_true=args.video_true,
                video_test=args.video_test,
            )

    if args.sort_output:
        sorted_items = sorted(
            METRICS_META.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        max_key_len = max(len(key) for key in METRICS_META.keys())

        print("-" * (max_key_len * 1.5))
        for key, value in sorted_items:
            print(f"{key:<{max_key_len}}: {value:<.4f}")
        print("-" * (max_key_len * 1.5))


if __name__ == "__main__":
    entrypoint()
