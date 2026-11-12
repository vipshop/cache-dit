"""Guards for the exact AdaLN plan memoization (`--adaln-online`).

The property that matters is that memoization changes nothing: a hit must return the same tensors the
projection would have produced, and a timestep the announced grid did not contain must be recomputed
rather than answered with the wrong table. Both are silent failures in a benchmark (a miss looks like
"same image, no speedup"), so they are checked numerically here.
"""

import torch
import torch.nn as nn

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.transformers.transformer_minimax_h3_image import (
  MINIMAX_H3_MODALITY_NUM,
  MiniMaxH3ImageAdaLayerNormModulation,
)

from cache_dit.offload.adaln_plan import (
  adaln_plan_tembeddings,
  install_adaln_plan_cache,
  minimax_h3_adaln_timestep_sets,
)


class _TinyTransformer(nn.Module):
  """Stand-in for the real transformer: two `adaln_proj` tables plus its timestep embedding."""

  def __init__(self):
    super().__init__()
    self.time_proj = Timesteps(num_channels=8, flip_sin_to_cos=True, downscale_freq_shift=0)
    self.time_embedder = TimestepEmbedding(in_channels=8, time_embed_dim=16, out_dim=12)
    self.blocks = nn.ModuleList([
      nn.ModuleDict(
        {"adaln_proj": MiniMaxH3ImageAdaLayerNormModulation(time_embed_dim=12, hidden_size=4)})
      for _ in range(2)
    ])


def _tables(transformer: _TinyTransformer) -> list[MiniMaxH3ImageAdaLayerNormModulation]:
  return [block["adaln_proj"] for block in transformer.blocks]


def test_plan_sets_cover_a_six_step_grid_and_its_edit_pairs():
  sigmas = [1.0, 0.968, 0.888, 0.75, 0.533, 0.232, 0.0]
  t2i = minimax_h3_adaln_timestep_sets(sigmas)
  edit = minimax_h3_adaln_timestep_sets(sigmas, reference_level=0.999)

  assert t2i == [[1.0 - sigma] for sigma in sigmas[:-1]]
  assert len(edit) == 2 * len(t2i)
  # The reference level sits above every sampling timestep, so the pairs are sorted ascending, which
  # is the order `torch.unique(..., sorted=True)` hands to the transformer.
  for pair, step in zip(edit[len(t2i):], [1.0 - sigma for sigma in sigmas[:-1]]):
    assert pair == sorted([step, 0.999])


def test_hits_are_bit_identical_and_pairs_do_not_collide():
  torch.manual_seed(0)
  transformer = _TinyTransformer()
  tables = _tables(transformer)
  embeddings = adaln_plan_tembeddings(
    transformer,
    minimax_h3_adaln_timestep_sets([1.0, 0.968, 0.888, 0.75, 0.533, 0.232, 0.0],
                                   reference_level=0.999))

  expected = [[table(temb) for temb in embeddings] for table in tables]
  plans, count = install_adaln_plan_cache(transformer,
                                          embeddings,
                                          compute_device="cpu",
                                          keep_weights_on="cpu")

  assert count == len(tables)
  assert plans == len(embeddings) * len(tables)
  for table, expected_per_vector in zip(tables, expected):
    for temb, want in zip(embeddings, expected_per_vector):
      got = table(temb)
      assert len(got) == 6
      for chunk, reference in zip(got, want):
        assert chunk.shape == reference.shape
        assert torch.equal(chunk, reference), "a memoized table must be bit-identical"
  # A single-row embedding and a two-row one sharing the first value must not share a plan.
  single = embeddings[0]
  pair = torch.cat([single, embeddings[0][:1] * 2.0], dim=0)
  assert tables[0](pair)[0].shape[0] == pair.shape[0] * MINIMAX_H3_MODALITY_NUM


def test_unannounced_timestep_is_recomputed_not_answered_stale():
  torch.manual_seed(0)
  transformer = _TinyTransformer()
  table = _tables(transformer)[0]
  embeddings = adaln_plan_tembeddings(transformer, [[0.5]])
  install_adaln_plan_cache(transformer, embeddings, compute_device="cpu", keep_weights_on="cpu")

  unseen = adaln_plan_tembeddings(transformer, [[0.25]])[0]
  # The reference has to come from a table that was never patched, i.e. the plain projection.
  reference = MiniMaxH3ImageAdaLayerNormModulation(time_embed_dim=12, hidden_size=4)
  reference.load_state_dict(table.state_dict())
  with torch.no_grad():
    want = reference(unseen)
  got = table(unseen)
  for chunk, expected in zip(got, want):
    assert torch.equal(chunk, expected), "a miss must recompute the table, not reuse another one"


def test_grid_matches_the_timesteps_the_scheduler_hands_to_the_transformer():
  """The plan keys are only useful if they equal what `forward` actually receives.

  `before_denoise.MiniMaxH3ImageSetTimestepsStep` rounds every sigma through `t = 1 - sigma` in
  float32 and back through `scheduler.set_timesteps`; a bit of drift there would make every lookup a
  miss, which looks like "the switch does nothing" rather than an error.
  """
  from diffusers import MiniMaxH3Scheduler

  sigmas = [
    1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0
  ]
  scheduler = MiniMaxH3Scheduler()
  timesteps = torch.tensor([1.0 - sigma for sigma in sigmas[:-1]], dtype=torch.float32)
  scheduler.set_timesteps(sigmas=torch.cat([1.0 - timesteps, torch.zeros(1)]), device="cpu")

  planned = [vector[0] for vector in minimax_h3_adaln_timestep_sets(sigmas)]
  assert len(planned) == len(scheduler.timesteps)
  for handed_to_the_transformer, planned_timestep in zip(scheduler.timesteps.tolist(), planned):
    assert float(handed_to_the_transformer) == planned_timestep
