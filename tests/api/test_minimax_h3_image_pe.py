"""Guards for the optional MiniMax-H3-Image-PE prompt-expansion stage.

Two things can go wrong silently here: the recommended ratio can be turned into a canvas the image
model refuses (it requires multiples of 32 and an area of at least `min_generation_area`), and the
weights repository can be mistaken for the blocks repository, which fails deep inside
`ModularPipeline.from_pretrained` instead of saying which checkpoint is missing.
"""

import os

import pytest

from cache_dit.models.pipelines.minimax_h3_image import aspect_ratio_to_size, expand_prompt


def test_recommended_ratio_becomes_a_canvas_the_model_accepts():
  for ratio in ("1:1", "4:3", "3:2", "16:9", "21:9", "5:4"):
    height, width = aspect_ratio_to_size(ratio)
    assert height % 32 == 0 and width % 32 == 0, (ratio, height, width)
    assert height * width >= 2048 * 2048, (ratio, height, width)

  height, width = aspect_ratio_to_size("16:9")
  assert width > height
  # A portrait request must come back portrait.
  portrait_height, portrait_width = aspect_ratio_to_size("9:16")
  assert portrait_height > portrait_width


def test_ratio_parser_rejects_garbage():
  for bad in ("", "wide", "3:x", "0:2", "-1:2"):
    with pytest.raises(ValueError):
      aspect_ratio_to_size(bad)


def test_weights_repository_is_rejected_with_an_actionable_message(tmp_path):
  # The weights repository ships `config.json` + shards and no model index; `from_pretrained` would
  # fail on the missing index deep inside diffusers.
  (tmp_path / "config.json").write_text("{}", encoding="utf-8")

  with pytest.raises(ValueError) as error:
    expand_prompt(str(tmp_path), "a cozy cabin in the snow")

  assert "minimax-h3-image-pe-blocks" in str(error.value)

  with pytest.raises(ValueError):
    expand_prompt(os.path.join(str(tmp_path), "missing"), "a cozy cabin in the snow")


class _FakePipeline:
  """Stand-in for the PE blocks: echoes the request and adds the documented output fields."""

  def load_components(self, **kwargs):
    self.load_kwargs = kwargs

  def __call__(self, **kwargs):
    return {
      "prompt": kwargs["prompt"],
      "aspect_ratio": kwargs.get("aspect_ratio"),
      "expanded_caption": "A long, complete description of the requested scene.",
      "target_aspect_ratio": "3:2",
    }


def test_happy_path_reads_the_blocks_schema_not_the_echo(monkeypatch, tmp_path):
  import diffusers

  (tmp_path / "modular_model_index.json").write_text("{}", encoding="utf-8")

  class _FakeModularPipeline:

    @staticmethod
    def from_pretrained(path, **kwargs):
      return _FakePipeline()

  monkeypatch.setattr(diffusers, "ModularPipeline", _FakeModularPipeline)

  caption, ratio = expand_prompt(str(tmp_path), "a cozy cabin in the snow")
  assert caption.startswith("A long, complete description")
  # `aspect_ratio` in the state is the echoed request ("adaptive"), which is not advice.
  assert ratio == "3:2"


def test_echoed_request_without_an_expansion_is_an_error(monkeypatch, tmp_path):
  import diffusers

  (tmp_path / "modular_model_index.json").write_text("{}", encoding="utf-8")

  class _EchoOnly:

    def load_components(self, **kwargs):
      pass

    def __call__(self, **kwargs):
      return {"prompt": kwargs["prompt"], "aspect_ratio": kwargs.get("aspect_ratio")}

  class _FakeModularPipeline:

    @staticmethod
    def from_pretrained(path, **kwargs):
      return _EchoOnly()

  monkeypatch.setattr(diffusers, "ModularPipeline", _FakeModularPipeline)

  with pytest.raises(ValueError) as error:
    expand_prompt(str(tmp_path), "a cozy cabin in the snow")

  assert "unchanged" in str(error.value)


def test_absurd_ratio_is_rejected_instead_of_building_a_huge_canvas():
  for ratio in ("1000:1", "1:1000"):
    with pytest.raises(ValueError):
      aspect_ratio_to_size(ratio)
