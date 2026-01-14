import os
import cv2
import pathlib
import warnings

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy import linalg
import torch
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d

from typing import Tuple, Union
from .inception import InceptionV3
from .config import _IMAGE_EXTENSIONS
from .config import _VIDEO_EXTENSIONS
from .config import get_metrics_verbose
from ..platforms import current_platform
from ..utils import disable_print
from cache_dit.logger import init_logger

warnings.filterwarnings("ignore")

logger = init_logger(__name__)

DISABLE_VERBOSE = not get_metrics_verbose()


# Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files_or_imgs, transforms=None):
        self.files_or_imgs = files_or_imgs
        self.transforms = transforms

    def __len__(self):
        return len(self.files_or_imgs)

    def __getitem__(self, i):
        file_or_img = self.files_or_imgs[i]
        if isinstance(file_or_img, (str, pathlib.Path)):
            img = Image.open(file_or_img).convert("RGB")
        elif isinstance(file_or_img, np.ndarray):
            # Assume the img is a standard OpenCV image.
            img = cv2.cvtColor(file_or_img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
        else:
            raise ValueError("file_or_img must be a file path or an OpenCV image.")
        if self.transforms is not None:
            img = self.transforms(img)
        return img


def get_activations(
    files_or_imgs,
    model,
    batch_size=50,
    dims=2048,
    device="cpu",
    num_workers=1,
    disable_tqdm=True,
):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files_or_imgs : List of image files paths or OpenCV image
    -- model         : Instance of inception model
    -- batch_size    : Batch size of images for the model to process at once.
                       Make sure that the number of samples is a multiple of
                       the batch size, otherwise some samples are ignored. This
                       behavior is retained to match the original FID score
                       implementation.
    -- dims          : Dimensionality of features returned by Inception
    -- device        : Device to run calculations
    -- num_workers   : Number of parallel dataloader workers

    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if batch_size > len(files_or_imgs):
        logger.info(
            ("Warning: batch size is bigger than the data size. " "Setting batch size to data size")
        )
        batch_size = len(files_or_imgs)

    dataset = ImagePathDataset(files_or_imgs, transforms=TF.ToTensor())
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    pred_arr = np.empty((len(files_or_imgs), dims))

    start_idx = 0

    for batch in tqdm(dataloader, disable=disable_tqdm):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx : start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_frechet_distance(
    mu1,
    sigma1,
    mu2,
    sigma2,
    eps=1e-6,
):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; " "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(
    files_or_imgs,
    model,
    batch_size=50,
    dims=2048,
    device="cpu",
    num_workers=1,
    disable_tqdm=True,
):
    """Calculation of the statistics used by the FID.
    Params:
    -- files_or_imgs : List of image files paths or OpenCV image
    -- model         : Instance of inception model
    -- batch_size    : Batch size of images for the model to process at once.
                       Make sure that the number of samples is a multiple of
                       the batch size, otherwise some samples are ignored. This
                       behavior is retained to match the original FID score
                       implementation.
    -- dims          : Dimensionality of features returned by Inception
    -- device        : Device to run calculations
    -- num_workers   : Number of parallel dataloader workers

    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(
        files_or_imgs,
        model,
        batch_size,
        dims,
        device,
        num_workers,
        disable_tqdm,
    )
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


class FrechetInceptionDistance:
    def __init__(
        self,
        device=(
            current_platform.device_type if current_platform.is_accelerator_available() else "cpu"
        ),
        dims: int = 2048,
        num_workers: int = 1,
        batch_size: int = 1,
        disable_tqdm: bool = True,
    ):
        # https://github.com/mseitzer/pytorch-fid/src/pytorch_fid/fid_score.py
        self.dims = dims
        self.device = device
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.disable_tqdm = disable_tqdm
        self.block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([self.block_idx]).to(self.device)
        self.model = self.model.eval()

    def compute_fid(
        self,
        image_true: np.ndarray | str,
        image_test: np.ndarray | str,
    ):
        """
        Calculates the FID of two file paths
        FID = FrechetInceptionDistance()
        img_fid = FID.compute_fid("img_true.png", "img_test.png")
        img_dir_fid = FID.compute_fid("img_true_dir", "img_test_dir")
        """
        if isinstance(image_true, str) or isinstance(image_test, str):
            if os.path.isfile(image_true) or os.path.isfile(image_test):
                assert os.path.exists(image_true)
                assert os.path.exists(image_test)
                assert image_true.split(".")[-1] in _IMAGE_EXTENSIONS
                assert image_test.split(".")[-1] in _IMAGE_EXTENSIONS
                image_true_files = [image_true]
                image_test_files = [image_test]
            else:
                # glob image files from dir
                assert os.path.isdir(image_true)
                assert os.path.isdir(image_test)
                image_true_dir = pathlib.Path(image_true)
                image_true_files = sorted(
                    [
                        file
                        for ext in _IMAGE_EXTENSIONS
                        for file in image_true_dir.rglob("*.{}".format(ext))
                    ]
                )
                image_test_dir = pathlib.Path(image_test)
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
        else:
            image_true_files = [image_true]
            image_test_files = [image_test]

        batch_size = min(16, self.batch_size)
        batch_size = min(batch_size, len(image_test_files))
        m1, s1 = calculate_activation_statistics(
            image_true_files,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
            self.disable_tqdm,
        )
        m2, s2 = calculate_activation_statistics(
            image_test_files,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
            self.disable_tqdm,
        )
        fid_value = calculate_frechet_distance(
            m1,
            s1,
            m2,
            s2,
        )

        return fid_value, len(image_true_files)

    def compute_video_fid(
        self,
        # file or dir
        video_true: str,
        video_test: str,
    ):
        if os.path.isfile(video_true) and os.path.isfile(video_test):
            video_true_frames, video_test_frames, valid_frames = self._fetch_video_frames(
                video_true=video_true,
                video_test=video_test,
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
                if os.path.basename(selected_video_test) == os.path.basename(selected_video_true):
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
                video_true_frames_, video_test_frames_, valid_frames_ = self._fetch_video_frames(
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

        batch_size = min(16, self.batch_size)
        m1, s1 = calculate_activation_statistics(
            video_true_frames,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
            self.disable_tqdm,
        )
        m2, s2 = calculate_activation_statistics(
            video_test_frames,
            self.model,
            batch_size,
            self.dims,
            self.device,
            self.num_workers,
            self.disable_tqdm,
        )
        fid_value = calculate_frechet_distance(
            m1,
            s1,
            m2,
            s2,
        )

        return fid_value, valid_frames

    def _fetch_video_frames(
        self,
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


fid_instance: FrechetInceptionDistance = None


def compute_fid(
    image_true: np.ndarray | str,
    image_test: np.ndarray | str,
) -> Union[Tuple[float, int], Tuple[None, None]]:
    global fid_instance
    if fid_instance is None:
        with disable_print():
            fid_instance = FrechetInceptionDistance(
                disable_tqdm=not get_metrics_verbose(),
            )
    assert fid_instance is not None
    return fid_instance.compute_fid(image_true, image_test)


def compute_video_fid(
    # file or dir
    video_true: str,
    video_test: str,
) -> Union[Tuple[float, int], Tuple[None, None]]:
    global fid_instance
    if fid_instance is None:
        with disable_print():
            fid_instance = FrechetInceptionDistance(
                disable_tqdm=not get_metrics_verbose(),
            )
    assert fid_instance is not None
    return fid_instance.compute_fid(video_true, video_test)
