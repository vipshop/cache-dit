from .utils import _safe_import

compute_psnr = _safe_import(".metrics", "compute_psnr")
compute_ssim = _safe_import(".metrics", "compute_ssim")
compute_mse = _safe_import(".metrics", "compute_mse")
compute_video_psnr = _safe_import(".metrics", "compute_video_psnr")
compute_video_ssim = _safe_import(".metrics", "compute_video_ssim")
compute_video_mse = _safe_import(".metrics", "compute_video_mse")
FrechetInceptionDistance = _safe_import(".fid", "FrechetInceptionDistance")
compute_fid = _safe_import(".fid", "compute_fid")
compute_video_fid = _safe_import(".fid", "compute_video_fid")
entrypoint = _safe_import(".metrics", "entrypoint")


def main():
    entrypoint()
