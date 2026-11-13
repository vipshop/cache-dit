#pragma once

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/detail/collective/moe_stride_utils.hpp>

namespace cutlass::epilogue::fusion {

template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp4RowColScaleBias : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementScalar = ElementCompute_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentScalar = 1;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerRowScaleSupported = true;
  static constexpr bool IsPerColScaleSupported = true;
  static constexpr bool IsPerColBiasSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp4RowColScaleBias = Sm90EVT<
  Sm90Compute<homogeneous_multiply_add, ElementOutput, ElementCompute,
              FloatRoundStyle::round_to_nearest>,
  Sm90RowBroadcast<0, CtaTileShapeMNK, float, ElementCompute, Stride<_0, bool, int64_t>, 4>,
  Sm90EVT<
    Sm90Compute<multiplies, ElementCompute, ElementCompute, FloatRoundStyle::round_to_nearest>,
    Sm90ColBroadcast<0, CtaTileShapeMNK, float, ElementCompute, Stride<bool, _0, int64_t>, 1>,
    Sm90AccFetch>,
  Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0, _1, int64_t>,
                   128 / cute::sizeof_bits_v<ElementBias>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp4RowColScaleBias<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp4RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp4RowColScaleBias<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp4RowColScaleBias<ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    float const* a_scale_ptr = nullptr;
    float const* b_scale_ptr = nullptr;
    ElementBias const* bias_ptr = nullptr;
    Stride<bool, _0, int64_t> dAScale = {bool(1), _0{}, 0};
    Stride<_0, bool, int64_t> dBScale = {_0{}, bool(1), 0};
    Stride<_0, _1, int64_t> dBias = {};

    operator typename Impl::Arguments() const {
      return {
        {b_scale_ptr, 1.0f, dBScale},
        {
          {a_scale_ptr, 1.0f, dAScale},
          {},
          {},
        },
        {bias_ptr, ElementBias(0), dBias},
        {},
      };
    }
  };

  using Impl::Impl;
};

template <class ElementOutput_, class ElementCompute_, class ElementBias_>
struct Fp4BiasOnly : FusionOperation {
  using ElementOutput = ElementOutput_;
  using ElementCompute = ElementCompute_;
  using ElementBias = ElementBias_;
  static constexpr int AlignmentBias = 128 / cute::sizeof_bits_v<ElementBias>;
  static constexpr bool IsPerColBiasSupported = true;
};

template <class CtaTileShapeMNK, class ElementOutput, class ElementCompute, class ElementBias>
using Sm90Fp4BiasOnly =
  Sm90EVT<Sm90Compute<plus, ElementOutput, ElementCompute, FloatRoundStyle::round_to_nearest>,
          Sm90AccFetch,
          Sm90RowBroadcast<0, CtaTileShapeMNK, ElementBias, ElementCompute, Stride<_0, _1, int64_t>,
                           128 / cute::sizeof_bits_v<ElementBias>>>;

template <int StagesC, int StagesD, int FragmentSize, bool ReuseSmemC, bool DelayTmaStore,
          class ElementOutput, class ElementCompute, class ElementBias, class CtaTileShapeMNK,
          class EpilogueTile>
struct FusionCallbacks<
  epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, FragmentSize, ReuseSmemC, DelayTmaStore>,
  Fp4BiasOnly<ElementOutput, ElementCompute, ElementBias>, CtaTileShapeMNK, EpilogueTile>
    : Sm90Fp4BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias> {
  using Impl = Sm90Fp4BiasOnly<CtaTileShapeMNK, ElementOutput, ElementCompute, ElementBias>;
  using Operation = Fp4BiasOnly<ElementOutput, ElementCompute, ElementBias>;

  struct Arguments {
    ElementBias const* bias_ptr = nullptr;
    Stride<_0, _1, int64_t> dBias = {};

    operator typename Impl::Arguments() const {
      return {{}, {bias_ptr, ElementBias(0), dBias}, {}};
    }
  };

  using Impl::Impl;
};

}  // namespace cutlass::epilogue::fusion

namespace cache_dit::fp4 {

using Fp4CutlassElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Fp4CutlassElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using Fp4CutlassElementD = cutlass::bfloat16_t;
using Fp4CutlassFusion =
  cutlass::epilogue::fusion::Fp4RowColScaleBias<Fp4CutlassElementD, float, Fp4CutlassElementD>;
using Fp4CutlassThreadBlockShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using Fp4CutlassClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using Fp4CutlassEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassThreadBlockShape,
  Fp4CutlassClusterShape, cute::Shape<cute::_64, cute::_64>, float, float, Fp4CutlassElementD,
  cutlass::layout::RowMajor, 8, Fp4CutlassElementD, cutlass::layout::RowMajor, 8,
  cutlass::epilogue::collective::EpilogueScheduleAuto, Fp4CutlassFusion>::CollectiveOp;

using Fp4CutlassMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassElementA,
  cutlass::layout::RowMajor, 32, Fp4CutlassElementB, cutlass::layout::ColumnMajor, 32, float,
  Fp4CutlassThreadBlockShape, Fp4CutlassClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp4CutlassEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using Fp4CutlassGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Fp4CutlassMainloop,
                                       Fp4CutlassEpilogue, void>;
using Fp4CutlassGemm = cutlass::gemm::device::GemmUniversalAdapter<Fp4CutlassGemmKernel>;

using Fp4CutlassSimpleFusion =
  cutlass::epilogue::fusion::Fp4BiasOnly<Fp4CutlassElementD, float, Fp4CutlassElementD>;
using Fp4CutlassSimpleEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassThreadBlockShape,
  Fp4CutlassClusterShape, cute::Shape<cute::_64, cute::_64>, float, float, Fp4CutlassElementD,
  cutlass::layout::RowMajor, 8, Fp4CutlassElementD, cutlass::layout::RowMajor, 8,
  cutlass::epilogue::collective::EpilogueScheduleAuto, Fp4CutlassSimpleFusion>::CollectiveOp;
using Fp4CutlassSimpleMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Fp4CutlassElementA,
  cutlass::layout::RowMajor, 32, Fp4CutlassElementB, cutlass::layout::ColumnMajor, 32, float,
  Fp4CutlassThreadBlockShape, Fp4CutlassClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
    sizeof(typename Fp4CutlassSimpleEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
using Fp4CutlassSimpleGemmKernel =
  cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>, Fp4CutlassSimpleMainloop,
                                       Fp4CutlassSimpleEpilogue, void>;
using Fp4CutlassSimpleGemm =
  cutlass::gemm::device::GemmUniversalAdapter<Fp4CutlassSimpleGemmKernel>;

inline void fp4_gemm_cutlass(const at::Tensor& a, const at::Tensor& b, const at::Tensor& a_scales,
                             const at::Tensor& b_scales, const at::Tensor& a_global_scales,
                             const at::Tensor& b_global_scales,
                             const std::optional<at::Tensor>& bias, at::Tensor& out,
                             cudaStream_t stream) {
  using StrideA = typename Fp4CutlassGemm::GemmKernel::StrideA;
  using StrideB = typename Fp4CutlassGemm::GemmKernel::StrideB;
  using StrideC = typename Fp4CutlassGemm::GemmKernel::StrideC;
  using StrideD = typename Fp4CutlassGemm::GemmKernel::StrideD;
  using ScaleConfig = typename Fp4CutlassGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using FusionArguments =
    typename Fp4CutlassGemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  FusionArguments fusion_args{};
  fusion_args.a_scale_ptr =
    a_global_scales.numel() != 0 ? a_global_scales.data_ptr<float>() : nullptr;
  fusion_args.b_scale_ptr =
    b_global_scales.numel() != 0 ? b_global_scales.data_ptr<float>() : nullptr;
  fusion_args.bias_ptr =
    bias.has_value() ? reinterpret_cast<const Fp4CutlassElementD*>(bias->data_ptr<at::BFloat16>())
                     : nullptr;

  auto* out_ptr = reinterpret_cast<Fp4CutlassElementD*>(out.data_ptr<at::BFloat16>());
  typename Fp4CutlassGemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<const Fp4CutlassElementA::DataType*>(a.data_ptr()),
      stride_a,
      reinterpret_cast<const Fp4CutlassElementB::DataType*>(b.data_ptr()),
      stride_b,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(a_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfa,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(b_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Fp4CutlassGemm gemm;
  cutlass::Status status = cutlass::Status::kSuccess;
  const size_t workspace_size = Fp4CutlassGemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP4 GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 FP4 GEMM launch failed: ", static_cast<int>(status));
}

inline void fp4_gemm_cutlass_simple(const at::Tensor& a, const at::Tensor& b,
                                    const at::Tensor& a_scales, const at::Tensor& b_scales,
                                    const std::optional<at::Tensor>& bias, at::Tensor& out,
                                    cudaStream_t stream) {
  using Gemm = Fp4CutlassSimpleGemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using FusionArguments = typename Gemm::GemmKernel::CollectiveEpilogue::FusionCallbacks::Arguments;

  const int M = static_cast<int>(a.size(0));
  const int N = static_cast<int>(b.size(0));
  const int K = static_cast<int>(a.size(1) * 2);
  auto stride_a = cutlass::make_internal_packed_stride(StrideA{}, {M, K, 1});
  auto stride_b = cutlass::make_internal_packed_stride(StrideB{}, {N, K, 1});
  auto stride_c = cutlass::make_internal_packed_stride(StrideC{}, {M, N, 1});
  auto stride_d = cutlass::make_internal_packed_stride(StrideD{}, {M, N, 1});
  auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
  auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

  FusionArguments fusion_args{};
  fusion_args.bias_ptr =
    bias.has_value() ? reinterpret_cast<const Fp4CutlassElementD*>(bias->data_ptr<at::BFloat16>())
                     : nullptr;
  auto* out_ptr = reinterpret_cast<Fp4CutlassElementD*>(out.data_ptr<at::BFloat16>());
  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, 1},
    {
      reinterpret_cast<const Fp4CutlassElementA::DataType*>(a.data_ptr()),
      stride_a,
      reinterpret_cast<const Fp4CutlassElementB::DataType*>(b.data_ptr()),
      stride_b,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(a_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfa,
      reinterpret_cast<const cutlass::float_ue4m3_t*>(b_scales.data_ptr<at::Float8_e4m3fn>()),
      layout_sfb,
    },
    {fusion_args, out_ptr, stride_c, out_ptr, stride_d},
  };

  Gemm gemm;
  cutlass::Status status = cutlass::Status::kSuccess;
  const size_t workspace_size = Gemm::get_workspace_size(arguments);
  at::Tensor workspace;
  void* workspace_ptr = nullptr;
  if (workspace_size != 0) {
    workspace = at::empty({static_cast<int64_t>(workspace_size)}, a.options().dtype(at::kByte));
    workspace_ptr = workspace.data_ptr();
  }
  status = gemm.initialize(arguments, workspace_ptr, stream);
  TORCH_CHECK(
    status == cutlass::Status::kSuccess,
    "CUTLASS SM120 single-level FP4 GEMM initialization failed: ", static_cast<int>(status));
  status = gemm.run(stream);
  TORCH_CHECK(status == cutlass::Status::kSuccess,
              "CUTLASS SM120 single-level FP4 GEMM launch failed: ", static_cast<int>(status));
}

}  // namespace cache_dit::fp4
