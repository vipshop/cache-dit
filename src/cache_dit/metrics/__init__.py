try:
    import ImageReward
    import lpips
    import skimage
    import scipy
except ImportError:
    raise ImportError(
        "Metrics functionality requires the 'metrics' extra dependencies. "
        "Install with:\npip install cache-dit[metrics]"
    )

from .metrics import compute_psnr
from .metrics import compute_ssim
from .metrics import compute_mse
from .metrics import compute_video_psnr
from .metrics import compute_video_ssim
from .metrics import compute_video_mse
from .fid import FrechetInceptionDistance
from .fid import compute_fid
from .fid import compute_video_fid
from .config import set_metrics_verbose
from .config import get_metrics_verbose
from .metrics import entrypoint


def main():
    entrypoint()
