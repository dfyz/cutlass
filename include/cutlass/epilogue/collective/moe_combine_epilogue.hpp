#pragma once

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

  using DispatchPolicy = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

  using ElementC = CDType;
  using ElementD = CDType;

  using StrideC = CDStride;
  using StrideD = CDStride;

  using InternalStrideC = InternalCDStride;
  using InternalStrideD = InternalCDStride;

  using ThreadEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    CDType,
    1, // one element per operation
    AccumType, // accumulator type
    AccumType // linear combination compute type
  >;

  struct SharedStorage {};

  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    const CDType** ptr_C = nullptr;
    CDStride dC{};
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
    // TODO
  }

private:
  Params params;
};

}
