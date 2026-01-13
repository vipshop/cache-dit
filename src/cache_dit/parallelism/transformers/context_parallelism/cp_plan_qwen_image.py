import torch
import functools
import diffusers
from typing import Optional
from torch.distributed import DeviceMesh
from diffusers.models.modeling_utils import ModelMixin
from diffusers import QwenImageTransformer2DModel
from diffusers.models.transformers.transformer_qwenimage import (
    QwenDoubleStreamAttnProcessor2_0,
    dispatch_attention_fn,
    apply_rotary_emb_qwen,
    Attention,
)

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from .cp_plan_registers import (
    ContextParallelismPlanner,
    ContextParallelismPlannerRegister,
)

from cache_dit.parallelism.attention import _unified_all_to_all_o_async_fn
from cache_dit.parallelism.attention import _unified_all_to_all_qkv_async_fn
from cache_dit.parallelism.attention import _prepare_ulysses_comm_metadata

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("QwenImage")
class QwenImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Set it as False to use custom CP plan defined here.
        self._cp_planner_preferred_native_diffusers = False

        experimental_ulysses_async = kwargs.get("experimental_ulysses_async", False)
        if experimental_ulysses_async:
            QwenDoubleStreamAttnProcessor2_0.__call__ = (
                __patch_QwenDoubleStreamAttnProcessor2_0_ulysses_async__call__
            )

            logger.info(
                "Enabled experimental Async QKV Projection with Ulysses style "
                "Context Parallelism for QwenImageTransformer2DModel."
            )

        if transformer is not None and self._cp_planner_preferred_native_diffusers:

            assert isinstance(
                transformer, QwenImageTransformer2DModel
            ), "Transformer must be an instance of QwenImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        if diffusers.__version__ <= "0.36.0":
            _cp_plan = {
                "": {
                    "hidden_states": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                    "encoder_hidden_states": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                    "encoder_hidden_states_mask": ContextParallelInput(
                        split_dim=1, expected_dims=2, split_output=False
                    ),
                },
                "pos_embed": {
                    0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                    1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                },
                "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
            }
        else:
            # Make CP plan compatible with https://github.com/huggingface/diffusers/pull/12702
            _cp_plan = {
                "transformer_blocks.0": {
                    "hidden_states": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                    "encoder_hidden_states": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                "pos_embed": {
                    0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                    1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                },
                "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
            }

        zero_cond_t = getattr(transformer, "zero_cond_t", False)
        if zero_cond_t:
            # modulate_index: [b, l=seq_len], Qwen-Image-Edit-2511
            _cp_plan.update(
                {
                    "transformer_blocks.*": {
                        "modulate_index": ContextParallelInput(
                            split_dim=1, expected_dims=2, split_output=False
                        ),
                    }
                }
            )

        return _cp_plan


# NOTE: Support Async Ulysses QKV projection for Qwen-Image
def _ulysses_attn_with_async_qkv_proj_qwen_image(
    self: QwenDoubleStreamAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.FloatTensor,  # Image stream
    encoder_hidden_states: torch.FloatTensor = None,  # Text stream
    encoder_hidden_states_mask: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    if encoder_hidden_states is None:
        raise ValueError(
            "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)"
        )

    ulysses_mesh: DeviceMesh = self._parallel_config.context_parallel_config._ulysses_mesh
    group = ulysses_mesh.get_group()

    _all_to_all_o_async_func = _unified_all_to_all_o_async_fn()
    _all_to_all_qv_async_func = _unified_all_to_all_qkv_async_fn()
    _all_to_all_k_async_func = _unified_all_to_all_qkv_async_fn(fp8=False)

    seq_txt = encoder_hidden_states.shape[1]

    img_value = attn.to_v(hidden_states)
    txt_value = attn.add_v_proj(encoder_hidden_states)
    img_value = img_value.unflatten(-1, (attn.heads, -1))
    txt_value = txt_value.unflatten(-1, (attn.heads, -1))
    joint_value = torch.cat([txt_value, img_value], dim=1)

    metadata = _prepare_ulysses_comm_metadata(joint_value)

    # Async all to all for value
    joint_value_wait = _all_to_all_qv_async_func(joint_value, group, **metadata)

    # Compute QKV for image stream (sample projections)
    img_query = attn.to_q(hidden_states)
    # Compute QKV for text stream (context projections)
    txt_query = attn.add_q_proj(encoder_hidden_states)
    # Reshape for multi-head attention
    img_query = img_query.unflatten(-1, (attn.heads, -1))
    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    # Apply QK normalization
    if attn.norm_q is not None:
        img_query = attn.norm_q(img_query)
    if attn.norm_added_q is not None:
        txt_query = attn.norm_added_q(txt_query)
    # Apply RoPE
    if image_rotary_emb is not None:
        img_freqs, txt_freqs = image_rotary_emb
        img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
        txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
    # Concatenate for joint attention
    # Order: [text, image]
    joint_query = torch.cat([txt_query, img_query], dim=1)

    # Async all to all for query
    joint_query_wait = _all_to_all_qv_async_func(joint_query, group, **metadata)

    img_key = attn.to_k(hidden_states)
    txt_key = attn.add_k_proj(encoder_hidden_states)
    img_key = img_key.unflatten(-1, (attn.heads, -1))
    txt_key = txt_key.unflatten(-1, (attn.heads, -1))
    if attn.norm_k is not None:
        img_key = attn.norm_k(img_key)
    if attn.norm_added_k is not None:
        txt_key = attn.norm_added_k(txt_key)
    # Apply RoPE
    if image_rotary_emb is not None:
        img_freqs, txt_freqs = image_rotary_emb
        img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
        txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
    joint_key = torch.cat([txt_key, img_key], dim=1)

    # Async all to all for key
    joint_key_wait = _all_to_all_k_async_func(joint_key, group, **metadata)

    # (S_GLOBAL, B, H_LOCAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
    joint_value = joint_value_wait()  # type: torch.Tensor
    joint_query = joint_query_wait()  # type: torch.Tensor
    joint_key = joint_key_wait()  # type: torch.Tensor

    # Compute joint attention
    out = dispatch_attention_fn(
        joint_query,
        joint_key,
        joint_value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=None,  # set to None to avoid double parallelism
    )  # (B, S_GLOBAL, H_LOCAL, D)

    # TODO: Split output before all to all to apply Async all to all,
    # overlap comm and compute of _to_out and to_add_out projections.
    out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)
    joint_hidden_states = out_wait()  # type: torch.Tensor

    # Reshape back
    joint_hidden_states = joint_hidden_states.flatten(2, 3)
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

    # Split attention outputs back
    txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
    img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

    # Apply output projections
    img_attn_output = attn.to_out[0](img_attn_output)
    if len(attn.to_out) > 1:
        img_attn_output = attn.to_out[1](img_attn_output)  # dropout

    txt_attn_output = attn.to_add_out(txt_attn_output)

    return img_attn_output, txt_attn_output


QwenDoubleStreamAttnProcessor2_0_original__call__ = QwenDoubleStreamAttnProcessor2_0.__call__


@functools.wraps(QwenDoubleStreamAttnProcessor2_0_original__call__)
def __patch_QwenDoubleStreamAttnProcessor2_0_ulysses_async__call__(
    self: QwenDoubleStreamAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.FloatTensor,  # Image stream
    encoder_hidden_states: torch.FloatTensor = None,  # Text stream
    encoder_hidden_states_mask: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.FloatTensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _ulysses_attn_with_async_qkv_proj_qwen_image(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            attention_mask,
            image_rotary_emb,
        )
    else:
        return QwenDoubleStreamAttnProcessor2_0_original__call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            encoder_hidden_states_mask,
            attention_mask,
            image_rotary_emb,
        )
