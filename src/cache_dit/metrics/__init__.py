from cache_dit.metrics.metrics import compute_psnr
from cache_dit.metrics.metrics import compute_ssim
from cache_dit.metrics.metrics import compute_mse
from cache_dit.metrics.metrics import compute_video_psnr
from cache_dit.metrics.metrics import compute_video_ssim
from cache_dit.metrics.metrics import compute_video_mse
from cache_dit.metrics.metrics import entrypoint
from cache_dit.metrics.fid import FrechetInceptionDistance
from cache_dit.metrics.config import set_metrics_verbose
from cache_dit.metrics.config import get_metrics_verbose


def main():
    entrypoint()
