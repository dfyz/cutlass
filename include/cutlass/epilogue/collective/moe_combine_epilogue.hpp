#pragma once

#include <cuda_runtime_api.h> // cudaStream_t
#include <cute/tensor.hpp> // cute::Tensor
#include <cute/util/type_traits.hpp> // cute::remove_poiinter_t
#include <cutlass/cuda_host_adapter.hpp> // CudaHostAdapter
#include <cutlass/detail/layout.hpp> // TagToStride*
#include <cutlass/epilogue/dispatch_policy.hpp> // PtrArrayNoSmemWarpSpecialized
#include <cutlass/epilogue/thread/linear_combination.h> // LinearCombination

#include <cuda/atomic>

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

  using ThreadEpilogueOp = void;

  using ElementC = CDType;
  using ElementD = CDType;

  using StrideC = CDStride;
  using StrideD = CDStride;

  using InternalStrideC = InternalCDStride;
  using InternalStrideD = InternalCDStride;

  struct SharedStorage {};

  struct Arguments {
    // see the `a2a_kernel()` description of these fields
    uint64_t* out_offs;
    uint8_t* token_owner;
    uint64_t* local_token_to_remote_token_idx;
    float* local_token_scores;

    // token size in elements
    uint64_t dim;

    // `routed_outputs_ptrs[r]` is the final output of GG2 on rank `r`
    __nv_bfloat16** routed_outputs_ptrs;
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
  CUTLASS_DEVICE void operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      const cute::Tensor<FrgEngine, FrgLayout>& accumulators,
      TiledMma tiled_mma,
      ResidueMNK,
      int thread_idx,
      char*
  ) {
    const auto M = cute::get<0>(problem_shape_mnkl);
    const auto N = cute::get<1>(problem_shape_mnkl);
    const auto m_coord = cute::get<0>(blk_coord_mnkl);
    const auto n_coord = cute::get<1>(blk_coord_mnkl);
    const auto l_coord = cute::get<3>(blk_coord_mnkl);

    const auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    const auto mn = cute::make_shape(M, N);
    const cute::Tensor global_coords = cute::make_identity_tensor(mn);
    const cute::Tensor tile_coords = cute::local_tile(
      global_coords,
      cute::take<0, 2>(blk_shape_MNK),
      cute::make_coord(m_coord, n_coord)
    );
    const cute::Tensor thread_coords = thr_mma.partition_C(tile_coords);

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < cute::size(accumulators); ++i) {
      const auto out_coord = thread_coords(i);
      if (cute::elem_less(out_coord, mn)) {
        const auto [row, col] = out_coord;
        const uint64_t local_token_idx = params.out_offs[l_coord] + row;
        const uint8_t peer = params.token_owner[local_token_idx];
        const float score = params.local_token_scores[local_token_idx];

        __nv_bfloat16* peer_output = params.routed_outputs_ptrs[peer];
        // Assume contiguous row-major layout.
        __nv_bfloat16* out_ptr =
          peer_output +
          params.local_token_to_remote_token_idx[local_token_idx] * params.dim +
          col;
        // This is wrong:
        //   * uses the GPU scope instead of the system one
        //   * uses a CAS instead of an actual atomic op
        atomicAdd(
          out_ptr,
          __float2bfloat16(score * accumulators(i))
        );
      }
    }
  }

private:
  Params params;
};

}
