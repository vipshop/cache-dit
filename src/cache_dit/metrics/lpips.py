import builtins as __builtin__
import contextlib
import warnings

import lpips
import torch

warnings.filterwarnings("ignore")

lpips_loss_fn_vgg = None
lpips_loss_fn_alex = None


def dummy_print(*args, **kwargs):
    pass


@contextlib.contextmanager
def disable_print():
    origin_print = __builtin__.print
    __builtin__.print = dummy_print
    yield
    __builtin__.print = origin_print


def compute_lpips_img(img0, img1, net: str = "alex"):
    global lpips_loss_fn_vgg
    global lpips_loss_fn_alex
    if net.lower() == "alex":
        if lpips_loss_fn_alex is None:
            with disable_print():
                lpips_loss_fn_alex = lpips.LPIPS(net="alex")
        loss_fn = lpips_loss_fn_alex
    elif net.lower() == "vgg":
        if lpips_loss_fn_vgg is None:
            with disable_print():
                lpips_loss_fn_vgg = lpips.LPIPS(net="vgg")
        loss_fn = lpips_loss_fn_vgg
    else:
        assert False, f"unsupport net {net}"

    with torch.no_grad():
        return loss_fn(img0, img1).item()
