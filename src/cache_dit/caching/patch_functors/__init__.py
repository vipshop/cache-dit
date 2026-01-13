from cache_dit.caching.patch_functors.functor_base import PatchFunctor
from cache_dit.caching.patch_functors.functor_dit import DiTPatchFunctor
from cache_dit.caching.patch_functors.functor_flux import FluxPatchFunctor
from cache_dit.caching.patch_functors.functor_chroma import (
    ChromaPatchFunctor,
)
from cache_dit.caching.patch_functors.functor_hidream import (
    HiDreamPatchFunctor,
)
from cache_dit.caching.patch_functors.functor_hunyuan_dit import (
    HunyuanDiTPatchFunctor,
)
from cache_dit.caching.patch_functors.functor_qwen_image_controlnet import (
    QwenImageControlNetPatchFunctor,
)
from cache_dit.caching.patch_functors.functor_wan_vace import (
    WanVACEPatchFunctor,
)


class ImportErrorPatchFunctor(PatchFunctor):
    def _apply(
        self,
        transformer,
        **kwargs,
    ):
        raise ImportError(
            "This PatchFunctor requires latest diffusers to be installed. "
            "Please install diffusers from source."
        )


try:
    from cache_dit.caching.patch_functors.functor_ltx2 import (
        LTX2PatchFunctor,
    )
except ImportError:
    LTX2PatchFunctor = ImportErrorPatchFunctor


try:
    from cache_dit.caching.patch_functors.functor_zimage_controlnet import (
        ZImageControlNetPatchFunctor,
    )
except ImportError:
    ZImageControlNetPatchFunctor = ImportErrorPatchFunctor
