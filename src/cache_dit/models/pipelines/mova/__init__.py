"""Vendored MOVA (OpenMOSS MOSS-Video-Audio, ``MOVA-360p``) support.

The released checkpoint is a diffusers-style tree whose components point at the training package
``mova.diffusion.*``, so cache-dit owns the code instead of depending on that package. See
:mod:`.pipeline` for the loader contract and :mod:`.stack` for the fused stack cache-dit caches.

Nothing here is imported by ``import cache_dit``; the CLI example and the block adapter import this
module lazily.

Why the released code cannot simply be imported (and what the bit-exact gate stubs)
------------------------------------------------------------------------------------
``mova.diffusion.models.wan_video_dit`` fails at **import time** on a machine without ``yunchang``:
its fallback defines ``class AttnType: FA3 = None`` while ``USPAttention.__init__`` defaults to
``AttnType.FA``, so the class body raises ``AttributeError``. It also calls
``kernels.get_kernel("kernels-community/flash-attn3")`` inside a ``ModuleNotFoundError`` handler, but
the Hub client raises ``ValueError`` when the publisher trust status cannot be verified offline, so
that escapes the fallback too. ``dac_vae`` additionally imports ``audiotools`` (``BaseModel`` /
``AudioSignal``), which cache-dit must not depend on.

To reproduce ``.tmp/mova/test_bitexact_small.py`` outside this repo:

* import ``diffusers`` **first**, then install a stub ``kernels`` module whose ``get_kernel`` raises
  ``ModuleNotFoundError`` and which carries a ``ModuleSpec`` (diffusers probes the package during
  import);
* install a stub ``yunchang`` / ``yunchang.kernels`` exposing ``LongContextAttention`` and an
  ``AttnType`` with ``FA`` / ``FA3`` (never instantiated by the parity test);
* after importing the released module, set ``FLASH_ATTN_2_AVAILABLE``, ``FLASH_ATTN_3_AVAILABLE``
  and ``SAGE_ATTN_AVAILABLE`` to ``False`` so both sides run the same scaled-dot-product path.

This vendored copy is what removes those requirements: it calls cache-dit's
``_dispatch_attention_fn`` and ships a decode-only DAC.
"""

from .pipeline import MOVA, MOVAPipeline, MOVAOutput
from .stack import MOVAFusedBlock, MOVAStack

__all__ = ["MOVA", "MOVAPipeline", "MOVAOutput", "MOVAFusedBlock", "MOVAStack"]
