"""Compiled cudagraph-unsafe wrapper for torchao FP8 inference scaled_mm.

Why this exists:
- Cache-DiT needs `compile + cuda graph + float8_per_row` to stay enabled at the
    same time.
- The problematic path is the rowwise torchao inference call into `torch._scaled_mm`.
- We want CUDA Graph partitioning to skip that callsite, but we do not want to
    fall all the way back to an opaque Python custom op because that regresses
    performance too much.

How this works:
- We patch torchao's `addmm_float8_unwrapped_inference()` helper so that the
    `_scaled_mm` callsite becomes a custom op tagged `cudagraph_unsafe`.
- We also register an Inductor lowering for that custom op which lowers into a
    separately compiled Inductor subgraph.
- That subgraph keeps Triton/template lowering enabled, but disables CUDA Graph
    capture locally so the replay-unsafe rowwise path never gets cudagraphified.
- The result is that `_scaled_mm` still participates in `torch.compile`, but the
    tagged FX node gives Inductor a stable point to exclude from CUDA Graph
    capture.

Expected impact:
- Positive: avoids the rowwise FP8 compile + CUDA Graph replay hang.
- Positive: keeps the `_scaled_mm` call on a compiled path instead of a Python
    fallback or a forced ATen-only extern kernel path.
- Tradeoff: this callsite is still excluded from CUDA Graph capture, so only the
    surrounding safe partitions are cudagraphified.
"""

from typing import Optional

import torch

from ...logger import init_logger

logger = init_logger(__name__)

_DTYPE_TO_ID = {
  torch.float16: 0,
  torch.bfloat16: 1,
  torch.float32: 2,
}
_ID_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ID.items()}
_OPAQUE_SCALED_MM_ENABLED = False
_INDUCTOR_LOWERING_REGISTERED = False


def _encode_output_dtype(output_dtype: torch.dtype) -> int:
  dtype_id = _DTYPE_TO_ID.get(output_dtype)
  if dtype_id is None:
    raise ValueError(f"Unsupported float8 inference output dtype: {output_dtype}")
  return dtype_id


def _decode_output_dtype(dtype_id: int) -> torch.dtype:
  output_dtype = _ID_TO_DTYPE.get(dtype_id)
  if output_dtype is None:
    raise ValueError(f"Unsupported float8 inference output dtype id: {dtype_id}")
  return output_dtype


def _run_float8_inference_scaled_mm(
  a_data: torch.Tensor,
  a_scale: torch.Tensor,
  b_data: torch.Tensor,
  b_scale: torch.Tensor,
  bias: Optional[torch.Tensor],
  output_dtype: torch.dtype,
  use_fast_accum: bool,
) -> torch.Tensor:
  if output_dtype == torch.float32 and bias is not None:
    output = torch._scaled_mm(
      a_data,
      b_data,
      scale_a=a_scale,
      scale_b=b_scale,
      scale_result=None,
      out_dtype=output_dtype,
      use_fast_accum=use_fast_accum,
    )
    return output + bias
  return torch._scaled_mm(
    a_data,
    b_data,
    scale_a=a_scale,
    scale_b=b_scale,
    bias=bias,
    scale_result=None,
    out_dtype=output_dtype,
    use_fast_accum=use_fast_accum,
  )


torch.library.define(
  "cache_dit_torchao_ops::float8_inference_scaled_mm",
  "(Tensor a_data, Tensor a_scale, Tensor b_data, Tensor b_scale, Tensor? bias, int out_dtype_id, bool use_fast_accum) -> Tensor",
  tags=(torch.Tag.cudagraph_unsafe, ),
)


@torch.library.impl("cache_dit_torchao_ops::float8_inference_scaled_mm", "CUDA")
def _float8_inference_scaled_mm(
  a_data: torch.Tensor,
  a_scale: torch.Tensor,
  b_data: torch.Tensor,
  b_scale: torch.Tensor,
  bias: Optional[torch.Tensor],
  out_dtype_id: int,
  use_fast_accum: bool,
) -> torch.Tensor:
  output_dtype = _decode_output_dtype(out_dtype_id)
  return _run_float8_inference_scaled_mm(
    a_data,
    a_scale,
    b_data,
    bias=bias,
    output_dtype=output_dtype,
    use_fast_accum=use_fast_accum,
    b_scale=b_scale,
  )


@torch.library.register_fake("cache_dit_torchao_ops::float8_inference_scaled_mm")
def _fake_float8_inference_scaled_mm(
  a_data: torch.Tensor,
  a_scale: torch.Tensor,
  b_data: torch.Tensor,
  b_scale: torch.Tensor,
  bias: Optional[torch.Tensor],
  out_dtype_id: int,
  use_fast_accum: bool,
) -> torch.Tensor:
  # Fake impl only carries shape/dtype information for tracing. It must match
  # the real op's output contract without performing any computation.
  del a_scale, b_scale, bias, use_fast_accum
  output_dtype = _decode_output_dtype(out_dtype_id)
  return a_data.new_empty((*a_data.shape[:-1], b_data.shape[-1]), dtype=output_dtype)


def _opaque_addmm_float8_unwrapped_inference(
  a_data: torch.Tensor,
  a_scale: torch.Tensor,
  b_data: torch.Tensor,
  b_scale: torch.Tensor,
  output_dtype: torch.dtype,
  output_scale: Optional[torch.Tensor] = None,
  bias: Optional[torch.Tensor] = None,
  use_fast_accum: bool = False,
) -> torch.Tensor:
  # `scale_result` is unused in the torchao rowwise FP8 inference path that we
  # patch today. Refusing it here keeps the wrapper intentionally narrow.
  if output_scale is not None:
    raise NotImplementedError("Opaque float8 scaled_mm wrapper does not support scale_result yet.")
  return torch.ops.cache_dit_torchao_ops.float8_inference_scaled_mm(
    a_data,
    a_scale,
    b_data,
    b_scale,
    bias,
    _encode_output_dtype(output_dtype),
    use_fast_accum,
  )


def _register_inductor_float8_scaled_mm_lowering() -> None:
  global _INDUCTOR_LOWERING_REGISTERED

  if _INDUCTOR_LOWERING_REGISTERED:
    return

  try:
    from torch.fx.experimental.proxy_tensor import make_fx

    import torch._inductor.config as inductor_config
    from torch._inductor import ir as inductor_ir
    from torch._inductor.decomposition import select_decomp_table
    from torch._inductor.ir import (
      FixedLayout,
      TensorBox,
      gm_original_output_strides,
      ir_node_to_tensor,
    )
    from torch._inductor.lowering import lowerings, register_lowering
    from torch._inductor.utils import convert_symint_to_expr
    from torch._inductor.virtualized import V
  except Exception:
    return

  custom_op = torch.ops.cache_dit_torchao_ops.float8_inference_scaled_mm.default
  if custom_op in lowerings:
    _INDUCTOR_LOWERING_REGISTERED = True
    return

  class _CompiledCudagraphUnsafeScaledMMBuffer(inductor_ir.ExternKernel):
    # This wrapper is the key bridge between two competing goals:
    # - the outer compiled graph needs a stable cudagraph_unsafe boundary so
    #   Inductor can keep this rowwise FP8 callsite out of CUDA Graph replay;
    # - the inner `_scaled_mm` work still needs to be compiled so it can keep
    #   using Inductor/Triton kernel selection instead of falling back to a
    #   Python custom op or a permanently forced extern kernel.
    #
    # Modeling the node as an ExternKernel keeps the outer scheduler boundary
    # explicit, while `self.subgraph` lets us compile the unsafe region under
    # a different set of local Inductor options.
    def __init__(
      self,
      layout,
      input_nodes,
      gm,
      example_inputs,
      subgraph_name,
    ):
      # `unwrap_storage()` only accepts realized buffers/views. Small test
      # graphs may reach this lowering with Pointwise/BaseView IR inputs, so
      # realize first and only then unwrap to the storage objects expected by
      # ExternKernel.
      realized_inputs = [self.realize_input(input_node) for input_node in input_nodes]
      unwrapped_inputs = self.unwrap_storage(realized_inputs)
      assert isinstance(unwrapped_inputs, list)
      super().__init__(None, layout, unwrapped_inputs)
      self.gm = gm
      self.example_inputs = example_inputs

      # Register the wrapper as an outer-graph operation/buffer so codegen
      # emits a callable launcher for the nested compiled subgraph instead of
      # trying to inline this region into the surrounding cudagraph-safe code.
      self.name = V.graph.register_buffer(self)
      V.graph.register_operation(self)

      # Build a dedicated nested graph for the tagged scaled_mm region. This
      # is the graph that will be compiled with local config overrides below.
      self.subgraph = V.graph.make_subgraph(self.gm, example_inputs, subgraph_name)

      assert inductor_ir.is_node_sequence(self.inputs)
      sym_inputs = inductor_ir.get_symbolic_inputs(self.inputs)
      # Forward symbolic inputs from the outer graph into the nested graph so
      # the generated subgraph launcher stays valid for symbolic sizes/strides
      # instead of hard-coding only the trace-time values.
      for sym_inp in sym_inputs:
        self.subgraph.graph_inputs[sym_inp.name] = sym_inp
        self.subgraph.graph_input_names.append(sym_inp.name)
      self.sym_inputs = [sym_var.name for sym_var in sym_inputs]

      previous_cudagraphs = inductor_config.triton.cudagraphs
      previous_max_autotune_gemm = inductor_config.max_autotune_gemm
      previous_max_autotune_gemm_backends = inductor_config.max_autotune_gemm_backends
      try:
        # The whole point of the wrapper is to compile this region without
        # letting CUDA Graph capture it. At the same time we explicitly keep
        # GEMM autotuning enabled so `_scaled_mm` can still choose a Triton
        # template when it wins, rather than regressing to an ATen-only path.
        inductor_config.triton.cudagraphs = False
        inductor_config.max_autotune_gemm = True
        inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
        with V.set_graph_handler(self.subgraph):
          self.subgraph.run(*self.example_inputs)
      finally:
        inductor_config.triton.cudagraphs = previous_cudagraphs
        inductor_config.max_autotune_gemm = previous_max_autotune_gemm
        inductor_config.max_autotune_gemm_backends = previous_max_autotune_gemm_backends

    def codegen(self, wrapper) -> None:

      class CodegenGraph:

        def __init__(self, graph):
          self.graph = graph
          self.name = graph.name

      assert inductor_ir.is_node_sequence(self.inputs)
      outer_inputs = [tensor.codegen_reference() for tensor in self.inputs]
      # Emit a flat call into the nested compiled graph. The outer wrapper
      # passes through symbolic arguments first, then the realized tensor
      # inputs for this specific scaled_mm instance.
      wrapper.codegen_subgraph_with_flattened_outputs(
        CodegenGraph(self.subgraph),
        [*self.sym_inputs, *outer_inputs],
        [self.name],
      )

  @register_lowering(custom_op, type_promotion_kind=None)
  def _lower_float8_inference_scaled_mm(
    a_data,
    a_scale,
    b_data,
    b_scale,
    bias,
    out_dtype_id,
    use_fast_accum,
    layout=None,
  ):
    output_dtype = _decode_output_dtype(int(out_dtype_id))
    tensor_inputs = [a_data, a_scale, b_data, b_scale]
    if bias is not None:
      tensor_inputs.append(bias)

    def _subgraph_impl(*tensors):
      local_bias = tensors[4] if len(tensors) == 5 else None
      return _run_float8_inference_scaled_mm(
        tensors[0],
        tensors[1],
        tensors[2],
        tensors[3],
        local_bias,
        output_dtype=output_dtype,
        use_fast_accum=use_fast_accum,
      )

    with V.fake_mode:
      example_inputs = [ir_node_to_tensor(inp) for inp in tensor_inputs]
      gm = make_fx(
        _subgraph_impl,
        decomposition_table=select_decomp_table(),
        tracing_mode="symbolic",
      )(*example_inputs)
      gm_original_output_strides(gm)
      fake_output = _subgraph_impl(*example_inputs)

    inferred_layout = FixedLayout(
      device=fake_output.device,
      dtype=fake_output.dtype,
      size=tuple(convert_symint_to_expr(size) for size in fake_output.shape),
      stride=tuple(convert_symint_to_expr(stride) for stride in fake_output.stride()),
    )
    current_node_name = getattr(V.graph.current_node, "name", "anon")
    return TensorBox.create(
      _CompiledCudagraphUnsafeScaledMMBuffer(
        layout=inferred_layout,
        input_nodes=tensor_inputs,
        gm=gm,
        example_inputs=example_inputs,
        subgraph_name=f"cache_dit_float8_inference_scaled_mm_{current_node_name}",
      ))

  _INDUCTOR_LOWERING_REGISTERED = True


def enable_opaque_torchao_float8_scaled_mm() -> None:
  global _OPAQUE_SCALED_MM_ENABLED

  if _OPAQUE_SCALED_MM_ENABLED:
    return

  import torchao.float8.inference as float8_inference
  from torchao.quantization.quantize_.workflows.float8 import float8_tensor

  _register_inductor_float8_scaled_mm_lowering()

  # Patch both import sites because Float8Tensor keeps a module-local reference
  # to the helper imported from torchao.float8.inference.
  float8_inference.addmm_float8_unwrapped_inference = _opaque_addmm_float8_unwrapped_inference
  float8_tensor.addmm_float8_unwrapped_inference = _opaque_addmm_float8_unwrapped_inference
  _OPAQUE_SCALED_MM_ENABLED = True

  logger.info(
    "Enabled compiled cudagraph-unsafe torchao float8 scaled_mm subgraph wrapper for compile + CUDA Graph."
  )
