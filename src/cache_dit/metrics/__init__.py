import os
import argparse
from cache_dit.metrics.metrics import compute_psnr
from cache_dit.metrics.metrics import compute_ssim
from cache_dit.metrics.metrics import compute_mse
from cache_dit.metrics.metrics import compute_video_psnr
from cache_dit.metrics.metrics import compute_video_ssim
from cache_dit.metrics.metrics import compute_video_mse
from cache_dit.metrics.fid import FrechetInceptionDistance
from cache_dit.logger import init_logger

logger = init_logger(__name__)


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
        "metric",
        type=str,
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
        help="Path to ground truth video",
    )
    parser.add_argument(
        "--video-test",
        "-v2",
        type=str,
        default=None,
        help="Path to predicted video",
    )
    return parser.parse_args()


def main():
    args = get_args()
    logger.debug(args)

    if args.img_true is not None and args.img_test is not None:
        if any(
            (
                not os.path.exists(args.img_true),
                not os.path.exists(args.img_test),
            )
        ):
            return
        # img_true and img_test can be files or dirs
        if args.metric == "psnr" or args.metric == "all":
            img_psnr = compute_psnr(args.img_true, args.img_test)
            logger.info(f"{args.img_true} vs {args.img_test}, PSNR: {img_psnr}")
        if args.metric == "ssim" or args.metric == "all":
            img_ssim = compute_ssim(args.img_true, args.img_test)
            logger.info(f"{args.img_true} vs {args.img_test}, SSIM: {img_ssim}")
        if args.metric == "mse" or args.metric == "all":
            img_mse = compute_mse(args.img_true, args.img_test)
            logger.info(f"{args.img_true} vs {args.img_test},  MSE: {img_mse}")
        if args.metric == "fid" or args.metric == "all":
            FID = FrechetInceptionDistance()
            img_fid = FID.compute_fid(args.img_true, args.img_test)
            logger.info(f"{args.img_true} vs {args.img_test},  FID: {img_fid}")
    if args.video_true is not None and args.video_test is not None:
        if any(
            (
                not os.path.exists(args.video_true),
                not os.path.exists(args.video_test),
            )
        ):
            return
        if args.metric == "psnr" or args.metric == "all":
            assert not os.path.isdir(args.video_true)
            assert not os.path.isdir(args.video_test)
            video_psnr = compute_video_psnr(args.video_true, args.video_test)
            logger.info(
                f"{args.video_true} vs {args.video_test}, PSNR: {video_psnr}"
            )
        if args.metric == "ssim" or args.metric == "all":
            assert not os.path.isdir(args.video_true)
            assert not os.path.isdir(args.video_test)
            video_ssim = compute_video_ssim(args.video_true, args.video_test)
            logger.info(
                f"{args.video_true} vs {args.video_test}, SSIM: {video_ssim}"
            )
        if args.metric == "mse" or args.metric == "all":
            assert not os.path.isdir(args.video_true)
            assert not os.path.isdir(args.video_test)
            video_mse = compute_video_mse(args.video_true, args.video_test)
            logger.info(
                f"{args.video_true} vs {args.video_test},  MSE: {video_mse}"
            )
        if args.metric == "fid" or args.metric == "all":
            assert not os.path.isdir(args.video_true)
            assert not os.path.isdir(args.video_test)
            FID = FrechetInceptionDistance()
            video_fid = FID.compute_video_fid(args.video_true, args.video_test)
            logger.info(
                f"{args.video_true} vs {args.video_test},  FID: {video_fid}"
            )


if __name__ == "__main__":
    main()
