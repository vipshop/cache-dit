"""Audio input types and validation for the Wan2.2-S2V pipeline."""

from typing import List, Union

import numpy as np
import torch

PipelineAudioInput = Union[
  np.ndarray,
  torch.Tensor,
  List[np.ndarray],
  List[torch.Tensor],
]


def is_valid_audio(audio) -> bool:
  """Check whether `audio` is one valid waveform.

  A valid waveform is a 2-D or 3-D `numpy.ndarray` / `torch.Tensor`: mono or multi-channel,
  optionally batched.

  :param audio: candidate waveform.
  :returns: `True` when `audio` is a 2-D or 3-D array or tensor.
  """
  return isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim in (2, 3)


def is_valid_audio_audiolist(audios) -> bool:
  """Check whether `audios` is a batch of waveforms.

  :param audios: a 4-D batch, a single valid waveform, or a list of valid waveforms.
  :returns: `True` when the input matches one of those shapes.
  """
  if isinstance(audios, (np.ndarray, torch.Tensor)) and audios.ndim == 4:
    return True
  elif is_valid_audio(audios):
    return True
  elif isinstance(audios, list):
    return all(is_valid_audio(audio) for audio in audios)
  return False
