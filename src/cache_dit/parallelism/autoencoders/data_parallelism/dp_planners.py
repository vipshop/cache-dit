# NOTE: must import all planner classes to register them
def _activate_auto_encoder_dp_planners():
    """Function to register all built-in auto encoder data parallelism planners."""
    from .dp_plan_autoencoder_kl import AutoencoderKLDataParallelismPlanner  # noqa: F401
    from .dp_plan_autoencoder_kl_qwen_image import (  # noqa: F401
        AutoencoderKLQwenImageDataParallelismPlanner,
    )
    from .dp_plan_autoencoder_kl_wan import (  # noqa: F401
        AutoencoderKLWanDataParallelismPlanner,
    )
    from .dp_plan_autoencoder_kl_hunyuanvideo import (  # noqa: F401
        AutoencoderKLHunyuanVideoDataParallelismPlanner,
    )
    from .dp_plan_autoencoder_kl_flux2 import (  # noqa: F401
        AutoencoderKLFlux2DataParallelismPlanner,
    )
