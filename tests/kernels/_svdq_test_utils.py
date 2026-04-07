from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_quantize_w4a4_act_fuse_lora
from cache_dit.quantization.svdquant import SVDQW4A4Linear
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4


RANKS_WITH_BASELINE = (0, 16, 32, 128)
EVALUATED_RANKS = (16, 32, 128)
TOY_ATTENTION_LINEAR_NAMES = (
    "block.to_q",
    "block.to_k",
    "block.to_v",
    "block.to_out",
)


@dataclass(frozen=True)
class SVDQAccuracyMetrics:
    mae: float
    rmse: float
    max_abs: float
    rel_l2: float
    cosine: float
    latency_ms: float = 0.0


class ToyTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.norm = nn.LayerNorm(embed_dim)
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=True)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=True)

    def _shape_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return hidden_states.transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        query = self._shape_heads(self.to_q(hidden_states))
        key = self._shape_heads(self.to_k(hidden_states))
        value = self._shape_heads(self.to_v(hidden_states))
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.embed_dim,
        )
        return self.to_out(attended)


class ToyModel(nn.Module):
    def __init__(self, embed_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.block = ToyTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


def runtime_dtype() -> torch.dtype:
    return (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )


def _cpu_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def make_token_batch(
    *,
    batch_size: int,
    seq_len: int,
    width: int,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = _cpu_generator(seed)
    tokens = torch.randn(batch_size, seq_len, width, generator=generator, dtype=torch.float32)
    positions = torch.linspace(-1.0, 1.0, steps=seq_len, dtype=torch.float32).view(1, seq_len, 1)
    features = torch.linspace(0.75, 1.25, steps=width, dtype=torch.float32).view(1, 1, width)
    tokens = tokens + 0.2 * positions * features
    return tokens.to(device=device, dtype=dtype)


def make_token_samples(
    *,
    num_samples: int,
    batch_size: int,
    seq_len: int,
    width: int,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    return [
        make_token_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            width=width,
            seed=seed + index,
            device=device,
            dtype=dtype,
        )
        for index in range(num_samples)
    ]


def make_spectral_decay_weight(
    out_features: int,
    in_features: int,
    *,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype,
    noise_scale: float = 0.01,
    decay: float = 0.965,
) -> torch.Tensor:
    generator = _cpu_generator(seed)
    rank = min(out_features, in_features)
    left = torch.randn(out_features, rank, generator=generator, dtype=torch.float32).to(
        device=device
    )
    right = torch.randn(in_features, rank, generator=generator, dtype=torch.float32).to(
        device=device
    )
    left = F.normalize(left, dim=0)
    right = F.normalize(right, dim=0)
    singular_values = torch.pow(
        torch.full((rank,), decay, dtype=torch.float32, device=device),
        torch.arange(rank, dtype=torch.float32, device=device),
    )
    weight = (left * singular_values.unsqueeze(0)) @ right.transpose(0, 1)
    weight = 6.0 * weight + noise_scale * torch.randn(
        out_features,
        in_features,
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    return weight.to(device=device, dtype=dtype)


def make_rank_sensitive_linear(
    *,
    in_features: int,
    out_features: int,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype,
    bias: bool = True,
) -> nn.Linear:
    linear = nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    with torch.no_grad():
        linear.weight.copy_(
            make_spectral_decay_weight(
                out_features,
                in_features,
                seed=seed,
                device=device,
                dtype=dtype,
            )
        )
        if linear.bias is not None:
            linear.bias.zero_()
    return linear.eval()


def compute_accuracy_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    latency_ms: float = 0.0,
) -> SVDQAccuracyMetrics:
    reference_fp32 = reference.float().reshape(-1)
    candidate_fp32 = candidate.float().reshape(-1)
    diff = candidate_fp32 - reference_fp32
    ref_norm = reference_fp32.norm().clamp_min(1e-12)
    cand_norm = candidate_fp32.norm().clamp_min(1e-12)
    cosine = torch.dot(reference_fp32, candidate_fp32) / (ref_norm * cand_norm)
    return SVDQAccuracyMetrics(
        mae=diff.abs().mean().item(),
        rmse=diff.square().mean().sqrt().item(),
        max_abs=diff.abs().max().item(),
        rel_l2=(diff.norm() / ref_norm).item(),
        cosine=cosine.item(),
        latency_ms=latency_ms,
    )


def _format_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_markdown_table(
    title: str,
    headers: tuple[str, ...],
    rows: list[tuple[object, ...]],
) -> str:
    if not headers:
        raise ValueError("headers must not be empty.")

    lines = [
        title,
        "",
        f"| {' | '.join(headers)} |",
        f"| {' | '.join('---:' for _ in headers)} |",
    ]
    for row in rows:
        if len(row) != len(headers):
            raise ValueError(f"Row has {len(row)} cells, expected {len(headers)}.")
        lines.append(f"| {' | '.join(_format_markdown_cell(value) for value in row)} |")
    return "\n".join(lines) + "\n"


def format_rank_report(
    title: str,
    metrics_by_rank: dict[int, SVDQAccuracyMetrics],
) -> str:
    rows: list[tuple[object, ...]] = []
    for rank in sorted(metrics_by_rank):
        metrics = metrics_by_rank[rank]
        rows.append(
            (
                rank,
                f"{metrics.mae:.6f}",
                f"{metrics.rmse:.6f}",
                f"{metrics.max_abs:.6f}",
                f"{metrics.rel_l2:.6f}",
                f"{metrics.cosine:.6f}",
                f"{metrics.latency_ms:.6f}",
            )
        )
    return format_markdown_table(
        title,
        ("rank", "mae", "rmse", "max_abs", "rel_l2", "cosine", "latency_ms"),
        rows,
    )


def assert_rank_metric_trend(
    metrics_by_rank: dict[int, SVDQAccuracyMetrics],
    metric_name: str,
    *,
    ranks: tuple[int, ...] = RANKS_WITH_BASELINE,
    atol: float = 5e-4,
    rtol: float = 0.03,
) -> None:
    previous = getattr(metrics_by_rank[ranks[0]], metric_name)
    for rank in ranks[1:]:
        current = getattr(metrics_by_rank[rank], metric_name)
        slack = max(atol, abs(previous) * rtol)
        assert current <= previous + slack, (
            f"Expected {metric_name} to improve with larger rank, but rank {rank} "
            f"has {current:.6f} vs previous {previous:.6f}."
        )
        previous = current


def run_svdq_operator_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    activations: torch.Tensor,
    *,
    output_dtype: torch.dtype,
    act_unsigned: bool = False,
) -> torch.Tensor:
    proj_down = state_dict["proj_down"]
    quantized_activations, ascales, lora_activations = svdq_quantize_w4a4_act_fuse_lora(
        input=activations,
        lora_down=None if proj_down.shape[1] == 0 else proj_down,
        smooth=state_dict["smooth_factor"],
        fp4=False,
        pad_size=256,
    )
    gemm_kwargs = {
        "act": quantized_activations,
        "wgt": state_dict["qweight"],
        "ascales": ascales,
        "wscales": state_dict["wscales"],
        "bias": state_dict.get("bias"),
        "fp4": False,
        "alpha": 1.0,
        "act_unsigned": act_unsigned,
        "output_dtype": output_dtype,
    }
    if proj_down.shape[1] > 0:
        gemm_kwargs["lora_act_in"] = lora_activations
        gemm_kwargs["lora_up"] = state_dict["proj_up"]
    output = svdq_gemm_w4a4(**gemm_kwargs)
    return output[: activations.shape[0]]


def collect_module_inputs(
    model: nn.Module,
    samples: list[torch.Tensor],
    module_names: tuple[str, ...] = TOY_ATTENTION_LINEAR_NAMES,
) -> dict[str, list[torch.Tensor]]:
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in module_names}
    hooks = []
    for module_name in module_names:
        module = model.get_submodule(module_name)

        def hook(
            _module: nn.Module, args: tuple[torch.Tensor, ...], name: str = module_name
        ) -> None:
            # Ensure the data still located on the same device and dtype as the original input,
            # but detach and clone to avoid holding references to the original tensors.
            captured_tensor = args[0].clone()  # clone to ensure it's a separate tensor
            # Keep the captured tensor on 'cpu' to avoid GPU memory issues, but convert
            # it to the same dtype as the input. We should convert it to the same device
            # as the weight tensors in quantizer one by one when computing activation
            # spans to avoid unnecessary GPU memory usage.
            captured_tensor = captured_tensor.to(device="cpu", dtype=args[0].dtype)
            captured[name].append(captured_tensor)

        hooks.append(module.register_forward_pre_hook(hook))

    try:
        with torch.inference_mode():
            for sample in samples:
                model(sample)
    finally:
        for hook in hooks:
            hook.remove()

    return captured


def _module_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    if "." not in module_name:
        return model, module_name
    parent_name, child_name = module_name.rsplit(".", 1)
    return model.get_submodule(parent_name), child_name


def quantize_toy_model(
    model: ToyModel,
    calibration_samples: list[torch.Tensor],
    *,
    rank: int,
    device: str | torch.device,
    dtype: torch.dtype,
    high_precision: bool = False,
    fp32_fallback: bool = False,
    streaming: bool = True,
) -> ToyModel:
    activations_by_module = collect_module_inputs(model, calibration_samples)
    quantized_model = copy.deepcopy(model).eval()
    for module_name in TOY_ATTENTION_LINEAR_NAMES:
        float_module = model.get_submodule(module_name)
        quantized_module = quantize_linear_svdq_w4a4(
            float_module,
            activations_by_module[module_name],
            rank=rank,
            device=device,
            torch_dtype=dtype,
            high_precision=high_precision,
            fp32_fallback=fp32_fallback,
            streaming=streaming,
        )
        parent, child_name = _module_parent(quantized_model, module_name)
        setattr(parent, child_name, quantized_module)
    return quantized_model.eval()


def build_empty_quantized_toy_model(
    *,
    embed_dim: int,
    num_heads: int,
    rank: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> ToyModel:
    model = ToyModel(embed_dim=embed_dim, num_heads=num_heads).to(device=device, dtype=dtype)
    for module_name in TOY_ATTENTION_LINEAR_NAMES:
        parent, child_name = _module_parent(model, module_name)
        float_module = getattr(parent, child_name)
        setattr(
            parent,
            child_name,
            SVDQW4A4Linear.from_linear(
                float_module,
                rank=rank,
                precision="int4",
                torch_dtype=dtype,
                device=device,
            ),
        )
    return model.eval()


def make_toy_model(
    *,
    embed_dim: int,
    num_heads: int,
    seed: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> ToyModel:
    model = ToyModel(embed_dim=embed_dim, num_heads=num_heads).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.block.norm.weight.fill_(1.0)
        model.block.norm.bias.zero_()
        for offset, module_name in enumerate(TOY_ATTENTION_LINEAR_NAMES):
            linear = model.get_submodule(module_name)
            linear.weight.copy_(
                make_spectral_decay_weight(
                    embed_dim,
                    embed_dim,
                    seed=seed + offset * 17,
                    device=device,
                    dtype=dtype,
                    noise_scale=0.015,
                )
            )
            linear.bias.zero_()
    return model.eval()


__all__ = [
    "EVALUATED_RANKS",
    "RANKS_WITH_BASELINE",
    "SVDQAccuracyMetrics",
    "TOY_ATTENTION_LINEAR_NAMES",
    "ToyModel",
    "ToyTransformerBlock",
    "assert_rank_metric_trend",
    "build_empty_quantized_toy_model",
    "collect_module_inputs",
    "compute_accuracy_metrics",
    "format_markdown_table",
    "format_rank_report",
    "make_rank_sensitive_linear",
    "make_token_batch",
    "make_token_samples",
    "make_toy_model",
    "quantize_toy_model",
    "run_svdq_operator_from_state_dict",
    "runtime_dtype",
]
