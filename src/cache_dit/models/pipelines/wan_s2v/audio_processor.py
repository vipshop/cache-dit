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

  A valid waveform is a 1-D to 3-D `numpy.ndarray` / `torch.Tensor`: the pipeline itself takes a
  mono 1-D waveform, while batched and multi-channel inputs are accepted by the feature extractor.

  :param audio: candidate waveform.
  :returns: `True` when `audio` is a 1-D to 3-D array or tensor.
  """
  return isinstance(audio, (np.ndarray, torch.Tensor)) and audio.ndim in (1, 2, 3)


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
