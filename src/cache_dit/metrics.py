import numpy as np
import torch
from torch import Tensor
from typing import Iterable
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3


def compute_psnr(
    image_true: np.ndarray,
    image_test: np.ndarray,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    PSNR = compute_psnr(img_true, img_test)
    """
    return peak_signal_noise_ratio(image_true, image_test)


def compute_mse(
    image_true: np.ndarray,
    image_test: np.ndarray,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    MSE = mean_squared_error(img_true, img_test)
    """
    return mean_squared_error(image_true, image_test)


def compute_ssim(
    image_true: np.ndarray,
    image_test: np.ndarray,
) -> float:
    """
    img_true = cv2.imread(img_true_file)
    img_test = cv2.imread(img_test_file)
    SSIM = compute_ssim(img_true, img_test)
    """
    return structural_similarity(
        image_true,
        image_test,
        multichannel=True,
        channel_axis=2,
    )


class FrechetInceptionDistance:
    def __init__(
        self,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = self.get_inception_model().to(device)
        self.model = self.model.eval()
        self.device = device

    def compute_fid(
        self,
        image_true: np.ndarray,
        image_test: np.ndarray,
        *args,
        **kwargs,
    ) -> float:
        """
        fid_computer = FrechetInceptionDistance()
        img_true = cv2.imread(img_true_file)
        img_test = cv2.imread(img_test_file)
        fid = fid_computer.compute_fid(img_true, img_test)
        """
        image_true_tensors = [torch.from_numpy(image_true).to(self.device)]
        image_test_tensors = [torch.from_numpy(image_test).to(self.device)]
        fid_value = (
            self._compute_batch_fid(
                image_true_tensors,
                image_test_tensors,
                *args,
                **kwargs,
            )
            .detach()
            .cpu()
            .numpy()[0]
        )
        return fid_value

    def compute_batch_fid(
        self,
        images_true: Iterable[np.ndarray],
        images_test: Iterable[np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        images_true_tensors = [
            torch.from_numpy(x).to(self.device) for x in images_true
        ]
        images_test_tensors = [
            torch.from_numpy(x).to(self.device) for x in images_test
        ]
        fid_values = (
            self._compute_batch_fid(
                images_true_tensors,
                images_test_tensors,
                *args,
                **kwargs,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return fid_values

    def _compute_batch_fid(
        self,
        batches_true: Iterable[Tensor],
        batches_pred: Iterable[Tensor],
        *args,
        **kwargs,
    ) -> Tensor:
        act_true = self.get_activations(
            self.model,
            batches_true,
            self.device,
            *args,
            **kwargs,
        )
        mu_true, sigma_true = self.calculate_activation_statistics(act_true)

        act_pred = self.get_activations(
            self.model,
            batches_pred,
            self.device,
            *args,
            **kwargs,
        )
        mu_pred, sigma_pred = self.calculate_activation_statistics(act_pred)

        fid_values = self.frechet_distance(
            mu_true,
            sigma_true,
            mu_pred,
            sigma_pred,
        )
        fid_values[fid_values < 0] = 0

        return fid_values

    def get_inception_model(
        self,
        dims: int = 2048,
    ):
        """
        Inspired by: https://github.com/mseitzer/pytorch-fid/src/pytorch_fid/fid_score.py
        """
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        return InceptionV3([block_idx])

    def get_activations(
        model: torch.nn.Module,
        batches: Iterable[Tensor],
        device: torch.device,
    ):
        """
        Inspired by: https://github.com/mseitzer/pytorch-fid/src/pytorch_fid/fid_score.py
        """
        with torch.no_grad():
            activations: list[Tensor] = []

            for batch in batches:
                batch = batch.to(device)
                pred = model(batch)[0]
                if pred.size(2) != 1 or pred.size(3) != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
                activations.append(pred.cpu().data.reshape(pred.size(0), -1))

        return torch.cat(activations, dim=0)

    def calculate_activation_statistics(
        self,
        activations: Tensor,
    ):
        """
        Inspired by: https://github.com/mseitzer/pytorch-fid/src/pytorch_fid/fid_score.py
        """
        mu = activations.mean(dim=0)
        sigma = activations.t().cov()
        return mu, sigma

    def frechet_distance(
        self,
        mu_x: Tensor,
        sigma_x: Tensor,
        mu_y: Tensor,
        sigma_y: Tensor,
    ) -> Tensor:
        """
        Inspired by: https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
        Issues: https://github.com/mseitzer/pytorch-fid/issues/95
        """
        a = (mu_x - mu_y).square().sum(dim=-1)
        b = sigma_x.trace() + sigma_y.trace()
        c = torch.linalg.eigvals(sigma_x @ sigma_y).sqrt().real.sum(dim=-1)

        return a + b - 2 * c
