#pragma once

#include <cuda_runtime_api.h> // cudaStream_t
#include <cute/tensor.hpp> // cute::Tensor
#include <cute/util/type_traits.hpp> // cute::remove_poiinter_t
#include <cutlass/cuda_host_adapter.hpp> // CudaHostAdapter
#include <cutlass/detail/layout.hpp> // TagToStride*
#include <cutlass/epilogue/dispatch_policy.hpp> // PtrArrayNoSmemWarpSpecialized
#include <cutlass/epilogue/thread/linear_combination.h> // LinearCombination

namespace cutlass::epilogue::collective {

template <
  typename AccumType,
  typename CDType,
  typename CDLayout
>
class MoECombineEpilogue {
public:
  using CDStride = cutlass::gemm::TagToStrideC_t<CDLayout>*;
  using InternalCDStride = cute::remove_pointer_t<CDStride>;

  using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;
  using DispatchPolicy = EpilogueSchedule;

  using ElementC = CDType;
  using ElementD = CDType;

  using StrideC = CDStride;
  using StrideD = CDStride;

  using InternalStrideC = InternalCDStride;
  using InternalStrideD = InternalCDStride;

  struct SharedStorage {};

  // This is the trivial epilogue op that does nothing,
  // and is not actually used (only needed for backward compatibility).
  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    CDType,
    1, // one element per operation
    AccumType, // accumulator type
    AccumType // linear combination compute type
  >;

  struct Arguments {
    // These fields are only preserved for backward compatibility.
    typename ThreadEpilogueOp::Params thread{};
    const CDType** ptr_C = nullptr;
    CDStride dC{};

    // These fields are actually used.
    CDType** ptr_D = nullptr;
    CDStride dD{};
  };

  using Params = Arguments;

  template <typename ProblemShape>
  static constexpr Params to_underlying_arguments(const ProblemShape&, const Arguments& args, void*) {
    return args;
  }

  template <typename ProblemShape>
  static size_t get_workspace_size(const ProblemShape&, const Arguments&, int) {
    return 0;
  }

  template <typename ProblemShape>
  static cutlass::Status initialize_workspace(const ProblemShape&, const Arguments&, void*, cudaStream_t, CudaHostAdapter*) {
    return cutlass::Status::kSuccess;
  }

  template <typename ProblemShape>
  static bool can_implement(const ProblemShape&, const Arguments&) {
    return true;
  }

  CUTLASS_DEVICE bool is_source_needed() {
    return false;
  }

  CUTLASS_HOST_DEVICE MoECombineEpilogue(const Params& params)
    : params(params)
  {}

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine,
    class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_HOST_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      const cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      TiledMma tiled_mma,
      ResidueMNK,
      int thread_idx,
      char*
  ) {
    auto M = cute::get<0>(problem_shape_mnkl);
    auto N = cute::get<1>(problem_shape_mnkl);
    auto m_coord = cute::get<0>(blk_coord_mnkl);
    auto n_coord = cute::get<1>(blk_coord_mnkl);
    auto l_coord = cute::get<3>(blk_coord_mnkl);

    auto stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD[l_coord]);

    cute::Tensor mD_mnl = cute::make_tensor(
      cute::make_gmem_ptr(params.ptr_D[l_coord]),
      cute::make_shape(M, N, 1),
      stride_d
    );

    cute::Tensor gD_mnl = cute::local_tile(
      mD_mnl,
      blk_shape_MNK,
      cute::make_coord(cute::_, cute::_, cute::_),
      cute::Step<cute::_1, cute::_1, cute::X>{}
    );

    cute::Tensor gD = gD_mnl(cute::_, cute::_, m_coord, n_coord, 0);
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    cute::Tensor tCgD = thr_mma.partition_C(gD);

    auto mn = cute::make_shape(M, N);
    cute::Tensor mD_crd = cute::make_identity_tensor(mn);
    cute::Tensor cD_mn = cute::local_tile(
      mD_crd,
      cute::take<0, 2>(blk_shape_MNK),
      cute::make_coord(m_coord, n_coord)
    );
    cute::Tensor tCcD = thr_mma.partition_C(cD_mn);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(accumulators); ++i) {
      if (cute::elem_less(tCcD(i), mn)) {
        tCgD(i) = static_cast<CDType>(accumulators(i));
      }
    }
  }

private:
  Params params;
};

}
