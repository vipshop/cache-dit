"""Ming-Image conditioning: query-token connector plus direct-VLM projection.

The Ming-Image checkpoint prepares the DiT condition in two independent paths, both read from the
checkpoint subtree only:

* **Query tokens.** ``mlp/model.safetensors`` ships ``query_tokens_dict.16x16``, a 16x16 grid of
  learnable tokens (``[256, 2048]``) that the MLLM receives at the end of the prompt. The MLLM
  hidden states at those 256 positions go through ``proj_in`` -> bidirectional Qwen2 connector ->
  ``proj_out``, producing the ``cap_feats`` list ZImage-style DiTs consume (one ``[256, 2560]``
  tensor per sample).
* **Direct VLM condition.** The prompt-token hidden states of the selected MLLM layers
  (``selected_hidden_states_layers``, ``[5, 12, 20]`` by default) are concatenated along the
  feature axis and projected by an RMSNorm + Linear (``proj_directvlm``): ``[B, S, 6144]`` ->
  ``[B, S, 3840]``. ``S`` is the number of prompt tokens, not one row per sample: Ming's
  ``get_condition_embeds_for_image_gen``, sglang's encoder and vLLM-Omni's ``patchify_and_embed``
  (``cap_ori_len = len(cap_feat) + len(cap_feat_2)``) all treat the direct rows as a sequence
  appended after ``cap_feats``.

Only ``<root>/connector/`` and ``<root>/mlp/`` are loaded; the MLLM (thinker) and the
reference-image path are not part of this module.

Note:
  Keep the following semantics intact, they are checkpoint facts rather than preferences:

  * The connector is the checkpoint's own Qwen2 stack, used as an *encoder*: a float 4D all-ones
    attention mask makes every query token attend to every other one. ``Qwen2Model`` is built, never
    ``Qwen2ForCausalLM``, so the ``lm_head`` (absent from the checkpoint) plays no role.
  * The connector's final RMSNorm is ``Qwen2Model``'s own ``norm``, loaded from
    ``connector/model.norm.weight`` and applied exactly once at the end of ``Qwen2Model.forward``.
    Ming's reference ``get_condition_embeds_for_image_gen`` and vLLM-Omni's encoder do the same.
    sglang names that same tensor a separate ``connector_norm`` module *and* uses the
    ``connector_norm`` config flag to gate L2 normalization; we follow the checkpoint, so the flag
    only gates ``F.normalize`` and the tensor stays Qwen2Model's final norm.
  * ``text_encoder_norm`` is not a norm either: it scales the final ``cap_feats`` by 1000, which is
    what ``ZImageModel_withMLP`` would otherwise do. ``Ming-Image-0.1-Design`` sets both flags to
    false, so ``cap_feats`` leaves this module unnormalized and unscaled.
  * ``use_identity_mlp`` must be true (as in ``Ming-Image-0.1-Design``): this module hands
    ``cap_feats`` straight to the DiT and does not implement the ``ToClipMLP`` variant.
  * The direct condition's channel blocks follow ``selected_hidden_states_layers`` order, so leave
    that tuple to ``mlp/config.json``: the MLLM derives its capture layers from the same field, and
    overriding it on one side only silently misaligns the 2048-wide blocks.
  * The token embedding table is never used (the connector runs on ``inputs_embeds``) and is
    dropped after construction, which saves the 466MB of a 151936x1536 table in bf16.
  * The connector's RoPE frequency tables stay float32 even for a bfloat16 module, as in
    transformers' own bf16 checkpoints: a bf16 table moves every query-token position. ``_apply``
    enforces this, so a later ``pipe.to(device, dtype)`` cannot undo it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2Model

from ....logger import init_logger

logger = init_logger(__name__)

# Connector tensors that exist in the checkpoint but have no counterpart in this module.
_UNUSED_CONNECTOR_KEYS = ("model.embed_tokens.weight", )


class _RMSNorm(nn.Module):
  """RMSNorm with fp32 statistics, matching the checkpoint's native norm.

  :param dim: Normalized feature size.
  :param eps: Epsilon added to the mean square.
  """

  def __init__(self, dim: int, eps: float = 1e-5):
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
    return y.to(x.dtype) * self.weight


def _read_safetensors(directory: Path, label: str) -> dict[str, torch.Tensor]:
  """Read every safetensors shard of one checkpoint subfolder.

  :param directory: Subfolder to read, e.g. ``<root>/connector``.
  :param label: Name used in the error message.
  :returns: Merged state dict of all shards.
  """
  files = sorted(directory.glob("*.safetensors"))
  if not files:
    raise FileNotFoundError(f"Ming-Image {label} weights not found under {directory}.")
  state: dict[str, torch.Tensor] = {}
  for file in files:
    state.update(load_file(str(file)))
  return state


# Connector buffers that stay float32 whatever dtype the module is cast to.
_FP32_CONNECTOR_BUFFERS = ("inv_freq", "original_inv_freq")


def _read_mllm_hidden_size(root: Path) -> int:
  """Read the MLLM hidden size from a Ming-Image checkpoint root.

  :param root: Checkpoint root holding ``config.json``.
  :returns: ``llm_config.hidden_size`` of the root config.
  """
  config_path = root / "config.json"
  if not config_path.is_file():
    raise FileNotFoundError(
      f"Missing {config_path}, pass hidden_size explicitly for a checkpoint without a root config.")
  return int(json.loads(config_path.read_text())["llm_config"]["hidden_size"])


class MingImageConditioning(nn.Module):
  """Query-token connector and direct-VLM projector of Ming-Image.

  :param connector_config: Qwen2 config of ``<root>/connector``.
  :param hidden_size: MLLM hidden size, i.e. the width of the query-token hidden states.
  :param query_token_grid: Side of the learnable query-token grid; ``grid * grid`` tokens.
  :param diffusion_c_input_dim: DiT condition width, the last dim of ``cap_feats``.
  :param diffusion_inner_dim: DiT inner width, the last dim of the direct condition.
  :param selected_hidden_states_layers: MLLM layers concatenated into the direct condition, in
    output order.
  :param normalize_cap_feats: L2-normalize ``cap_feats`` (``connector_norm`` in the checkpoint).
  :param text_encoder_norm: Scale ``cap_feats`` by 1000 (``text_encoder_norm`` in the checkpoint).
  :param direct_norm_eps: Epsilon of the RMSNorm in the direct projector.
  """

  def __init__(
    self,
    connector_config: Qwen2Config,
    *,
    hidden_size: int,
    query_token_grid: int = 16,
    diffusion_c_input_dim: int = 2560,
    diffusion_inner_dim: int = 3840,
    selected_hidden_states_layers: Sequence[int] = (5, 12, 20),
    normalize_cap_feats: bool = False,
    text_encoder_norm: bool = False,
    direct_norm_eps: float = 1e-5,
  ):
    super().__init__()
    layers = tuple(int(index) for index in selected_hidden_states_layers)
    if not layers:
      raise ValueError("selected_hidden_states_layers must name at least one MLLM layer.")
    self.hidden_size = hidden_size
    self.query_token_grid = int(query_token_grid)
    self.num_query_tokens = self.query_token_grid ** 2
    self.diffusion_c_input_dim = diffusion_c_input_dim
    self.diffusion_inner_dim = diffusion_inner_dim
    self.selected_hidden_states_layers = layers
    self.normalize_cap_feats = normalize_cap_feats
    self.text_encoder_norm = text_encoder_norm

    connector_hidden_size = int(connector_config.hidden_size)
    self.connector_hidden_size = connector_hidden_size
    self.connector = Qwen2Model(connector_config)
    # The connector only ever runs on `inputs_embeds`; the token table would cost 466MB in bf16.
    del self.connector.embed_tokens

    self.query_tokens = nn.Parameter(torch.empty(self.num_query_tokens, hidden_size))
    self.proj_in = nn.Linear(hidden_size, connector_hidden_size, bias=True)
    self.proj_out = nn.Linear(connector_hidden_size, diffusion_c_input_dim, bias=True)
    # Channel blocks of the direct condition, in `selected_hidden_states_layers` order.
    self.direct_input_dim = hidden_size * len(layers)
    self.direct_projector = nn.Sequential(
      _RMSNorm(self.direct_input_dim, direct_norm_eps),
      nn.Linear(self.direct_input_dim, diffusion_inner_dim, bias=True),
    )

  def _apply(self, fn, recurse=True):
    """Apply a tensor transform, restoring the connector's float32 RoPE tables afterwards.

    ``nn.Module._apply`` reaches every parameter *and* buffer, so ``DiffusionPipeline.to(device,
    dtype)`` would cast the connector's non-persistent frequency tables; a bf16 table moves every
    query-token position. transformers keeps them float32 for bf16 checkpoints, and so do we. The
    pristine float32 tables are snapshotted before the transform and copied back onto the resulting
    device, which also survives ``to_empty``.

    :param fn: Tensor transform applied by ``nn.Module._apply``.
    :param recurse: Whether to recurse into submodules.
    :returns: The transformed module.
    """
    rope = self.connector.rotary_emb
    tables = {
      name: getattr(rope, name).float().clone()
      for name in _FP32_CONNECTOR_BUFFERS if hasattr(rope, name)
    }
    module = super()._apply(fn, recurse=recurse)
    for name, table in tables.items():
      buffer = getattr(module.connector.rotary_emb, name)
      setattr(module.connector.rotary_emb, name, table.to(buffer.device))
    return module

  @property
  def dtype(self) -> torch.dtype:
    """Parameter dtype of the module.

    ``DiffusionPipeline.to`` reads this on every component to pick the cast target.

    :returns: The dtype of ``proj_in.weight``.
    """
    return self.proj_in.weight.dtype

  @property
  def device(self) -> torch.device:
    """Device the module lives on.

    :returns: The device of ``proj_in.weight``.
    """
    return self.proj_in.weight.device

  @classmethod
  def from_config(
    cls,
    connector_config: Qwen2Config,
    mlp_config: dict,
    *,
    hidden_size: int,
    direct_norm_eps: float = 1e-5,
  ) -> MingImageConditioning:
    """Build the module from the two checkpoint configs, without weights.

    :param connector_config: Parsed ``<root>/connector/config.json``.
    :param mlp_config: Parsed ``<root>/mlp/config.json``.
    :param hidden_size: MLLM hidden size (``llm_config.hidden_size`` of the root config).
    :param direct_norm_eps: Epsilon of the direct-projector RMSNorm.
    :returns: A randomly initialized module.
    """
    scales = list(mlp_config["img_gen_scales"])
    if len(scales) != 1:
      raise ValueError(f"Ming-Image conditioning expects one query-token grid, got {scales}.")
    return cls(
      connector_config,
      hidden_size=hidden_size,
      query_token_grid=int(scales[0]),
      diffusion_c_input_dim=int(mlp_config["diffusion_c_input_dim"]),
      diffusion_inner_dim=int(mlp_config["diffusion_inner_dim"]),
      selected_hidden_states_layers=tuple(mlp_config["selected_hidden_states_layers"]),
      normalize_cap_feats=bool(mlp_config.get("connector_norm", False)),
      text_encoder_norm=bool(mlp_config.get("text_encoder_norm", False)),
      direct_norm_eps=direct_norm_eps,
    )

  @classmethod
  def from_pretrained(
    cls,
    model_path: str | Path,
    *,
    hidden_size: int | None = None,
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    direct_norm_eps: float = 1e-5,
    local_files_only: bool = True,
  ) -> MingImageConditioning:
    """Build the module from a Ming-Image checkpoint directory and load its weights.

    :param model_path: Checkpoint root holding ``connector/``, ``mlp/`` and ``config.json``.
    :param hidden_size: MLLM hidden size; read from ``config.json`` when omitted.
    :param torch_dtype: Parameter dtype after loading; the module default when omitted.
    :param device: Placement after loading; left on CPU when omitted.
    :param direct_norm_eps: Epsilon of the direct-projector RMSNorm.
    :param local_files_only: Accepted for ``from_pretrained`` compatibility; every read is local.
    :returns: The loaded module.
    """
    root = Path(model_path)
    if hidden_size is None:
      hidden_size = _read_mllm_hidden_size(root)
    connector_path = root / "connector" / "config.json"
    if not connector_path.is_file():
      raise FileNotFoundError(f"Missing Ming-Image connector config: {connector_path}.")
    connector_config = Qwen2Config(**json.loads(connector_path.read_text()))
    mlp_config = json.loads((root / "mlp" / "config.json").read_text())
    module = cls.from_config(
      connector_config,
      mlp_config,
      hidden_size=hidden_size,
      direct_norm_eps=direct_norm_eps,
    )
    if torch_dtype is not None:
      module = module.to(dtype=torch_dtype)
    if device is not None:
      module = module.to(device=device)
    module.load_from_checkpoint(root)
    return module

  def load_from_checkpoint(self, model_path: str | Path) -> MingImageConditioning:
    """Strictly load the connector and MLP weights of one checkpoint root.

    :param model_path: Checkpoint root holding ``connector/`` and ``mlp/``.
    :returns: ``self``, for chaining.
    """
    root = Path(model_path)
    self._load_connector(root / "connector")
    self._load_mlp(root / "mlp")
    logger.info(
      "Ming-Image conditioning ready: %d query tokens %d -> %d -> %d, direct %d -> %d, dtype=%s.",
      self.num_query_tokens,
      self.hidden_size,
      self.connector_hidden_size,
      self.diffusion_c_input_dim,
      self.hidden_size * len(self.selected_hidden_states_layers),
      self.diffusion_inner_dim,
      self.proj_in.weight.dtype,
    )
    return self

  def _load_connector(self, connector_dir: Path) -> None:
    """Load ``Qwen2Model`` strictly; the checkpoint prefixes its keys with ``model.``.

    :param connector_dir: ``<root>/connector``.
    """
    state = _read_safetensors(connector_dir, "connector")
    absent = [key for key in _UNUSED_CONNECTOR_KEYS if key not in state]
    if absent:
      raise ValueError(f"Ming-Image connector is missing the expected layout key(s) {absent}.")
    expected = self.connector.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
      if key in _UNUSED_CONNECTOR_KEYS:
        continue
      if not key.startswith("model."):
        raise ValueError(
          f"Unexpected Ming-Image connector key {key!r}, expected a 'model.' prefix.")
      mapped[key[len("model."):]] = tensor
    missing = sorted(set(expected) - set(mapped))
    unexpected = sorted(set(mapped) - set(expected))
    if missing or unexpected:
      raise ValueError(f"Ming-Image connector weights do not cover the module: missing={missing}, "
                       f"unexpected={unexpected}.")
    self.connector.load_state_dict(mapped)

  def _load_mlp(self, mlp_dir: Path) -> None:
    """Load the eight MLP tensors strictly, key by key.

    :param mlp_dir: ``<root>/mlp``.
    """
    state = _read_safetensors(mlp_dir, "mlp")
    query_key = f"query_tokens_dict.{self.query_token_grid}x{self.query_token_grid}"
    expected = {
      "proj_in.weight",
      "proj_in.bias",
      "proj_out.weight",
      "proj_out.bias",
      query_key,
      "proj_directvlm.0.weight",
      "proj_directvlm.1.weight",
      "proj_directvlm.1.bias",
    }
    if set(state) != expected:
      raise ValueError(
        f"Ming-Image mlp weights must be exactly {sorted(expected)}, got {sorted(state)}.")
    self.proj_in.load_state_dict({
      "weight": state["proj_in.weight"],
      "bias": state["proj_in.bias"],
    })
    self.proj_out.load_state_dict({
      "weight": state["proj_out.weight"],
      "bias": state["proj_out.bias"],
    })
    self.direct_projector.load_state_dict({
      "0.weight": state["proj_directvlm.0.weight"],
      "1.weight": state["proj_directvlm.1.weight"],
      "1.bias": state["proj_directvlm.1.bias"],
    })
    with torch.no_grad():
      self.query_tokens.copy_(state[query_key])

  def forward(
    self,
    query_hidden_states: torch.Tensor,
    direct_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
  ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Project the MLLM hidden states into DiT conditions.

    :param query_hidden_states: MLLM hidden states at the ``grid * grid`` query-token positions,
      shape ``[B, 256, 2048]``.
    :param direct_hidden_states: Concatenated prompt-token hidden states of
      ``selected_hidden_states_layers``, shape ``[B, S, direct_input_dim]`` with ``S`` the
      prompt-token count (a single row per sample only when the prompt has one token) and the
      channel blocks in ``selected_hidden_states_layers`` order.
    :param attention_mask: Optional mask for the connector; defaults to full bidirectional
      attention over the query tokens. A 2D ``[B, N]`` mask is expanded to ``[B, 1, N, N]``.
    :returns: ``cap_feats`` as a list of ``[256, 2560]`` tensors (one per sample) and the direct
      condition of shape ``[B, S, 3840]``.
    """
    tokens = query_hidden_states.shape[1]
    if tokens != self.num_query_tokens:
      raise ValueError(f"Expected {self.num_query_tokens} query-token hidden states "
                       f"({self.query_token_grid}x{self.query_token_grid}), got {tokens}.")
    if direct_hidden_states.dim() != 3:
      raise ValueError(f"Expected direct hidden states of shape [B, S, {self.direct_input_dim}], "
                       f"got {tuple(direct_hidden_states.shape)}.")
    channels = direct_hidden_states.shape[-1]
    if channels != self.direct_input_dim:
      raise ValueError(f"Expected direct hidden states with {self.direct_input_dim} channels "
                       f"({len(self.selected_hidden_states_layers)} layers of {self.hidden_size}), "
                       f"got {channels}.")
    x = self.proj_in(query_hidden_states.to(self.proj_in.weight.dtype))
    batch = x.shape[0]
    if attention_mask is None:
      attention_mask = torch.ones((batch, 1, tokens, tokens), dtype=x.dtype, device=x.device)
    elif attention_mask.dim() == 2:
      attention_mask = attention_mask.to(x.dtype).view(batch, 1, 1, tokens)
      attention_mask = attention_mask.expand(batch, 1, tokens, tokens)
    hidden = self.connector(inputs_embeds=x, attention_mask=attention_mask,
                            use_cache=False).last_hidden_state
    cap_feats = self.proj_out(hidden)
    if self.normalize_cap_feats:
      cap_feats = F.normalize(cap_feats, dim=-1)
    if self.text_encoder_norm:
      cap_feats = cap_feats * 1000.0
    direct_condition = self.direct_projector(
      direct_hidden_states.to(self.direct_projector[1].weight.dtype))
    return list(cap_feats.unbind(0)), direct_condition


__all__ = ["MingImageConditioning"]
