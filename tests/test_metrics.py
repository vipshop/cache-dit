import os
import argparse
from cache_dit.metrics import compute_psnr
from cache_dit.metrics import compute_video_psnr
from cache_dit.metrics import FrechetInceptionDistance  # FID


def get_args():
    parser = argparse.ArgumentParser(
        description="Test TaylorSeer approximation."
    )
    parser.add_argument(
        "--img-true",
        type=str,
        default=None,
        help="Path to groud true image",
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
        help="Path to groud true video",
    )
    parser.add_argument(
        "--video-test",
        type=str,
        default=None,
        help="Path to predicted video",
    )

    return parser.parse_args()


def main():
    args = get_args()
    print(args)

    if args.img_true is not None and args.img_test is not None:
        if any(
            (
                not os.path.exists(args.img_true),
                not os.path.exists(args.img_test),
            )
        ):
            return
        img_psnr = compute_psnr(args.img_true, args.img_test)
        print(f"{args.img_true} vs {args.img_test}, PSNR: {img_psnr}")
        FID = FrechetInceptionDistance()
        img_fid = FID.compute_fid(args.img_true, args.img_test)
        print(f"{args.img_true} vs {args.img_test}, FID: {img_fid}")
    if args.video_true is not None and args.video_test is not None:
        if any(
            (
                not os.path.exists(args.video_true),
                not os.path.exists(args.video_test),
            )
        ):
            return
        video_psnr = compute_video_psnr(args.video_true, args.video_test)
        print(f"{args.video_true} vs {args.video_test}, PSNR: {video_psnr}")


if __name__ == "__main__":
    main()
    # python3 test_metrics.py --img-true true.png --img-test test.png
