"""Guards for the readable parts of the AdaLN plan of a MiniMax-H3 family pipeline.

`timestep_sets_from_pipeline` reads a pipeline's fixed sampling grid, which needs no weights: the
two MiniMax-H3 image variants differ only in the name they give the pinned conditioning level.
"""

import argparse

from cache_dit.offload.adaln_plan import timestep_sets_from_pipeline


def test_adaln_plan_reads_the_ph3_conditioning_level():
  """`--adaln-online` must see the two-row timestep vectors of an edit request.

  P-H3-Image names the pinned conditioning level `keyframe_noise_aug`; reading only MiniMax-H3's
  `reference_noise_aug` would leave every edit step without a memoized table (slow, not wrong), so
  the fallback is asserted on a stub instead of on a loaded checkpoint.
  """

  class _Stub:
    config = argparse.Namespace(sample_sigmas=[1.0, 0.5, 0.0])
    keyframe_noise_aug = 0.999

  sets = timestep_sets_from_pipeline(_Stub())
  assert any(len(entry) == 2 for entry in sets)
  assert [0.0, 0.999] in sets


def test_adaln_plan_prefers_the_reference_name_when_both_exist():
  """A pipeline exposing both names resolves to `reference_noise_aug`, so the fallback stays unused.

  Neither shipped checkpoint does: MiniMax-H3-Image only declares `reference_noise_aug`, P-H3-Image
  only `keyframe_noise_aug`. The stub covers the ordering rule itself, so a future checkpoint that
  carries both cannot flip the meaning silently.
  """

  class _Stub:
    config = argparse.Namespace(sample_sigmas=[1.0, 0.5, 0.0])
    reference_noise_aug = 0.999
    keyframe_noise_aug = 0.5

  sets = timestep_sets_from_pipeline(_Stub())
  assert [0.0, 0.999] in sets
  assert [0.0, 0.5] not in sets


def test_adaln_plan_requires_the_fixed_grid():
  """A pipeline without `sample_sigmas` has no announced grid, so `--adaln-online` cannot key on
  one."""

  class _Stub:
    config = argparse.Namespace()
    keyframe_noise_aug = 0.999

  try:
    timestep_sets_from_pipeline(_Stub())
  except ValueError as error:
    assert "sample_sigmas" in str(error)
  else:
    raise AssertionError("a pipeline without sample_sigmas must not silently key on a guess")
