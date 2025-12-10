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

  static constexpr uint64_t kTileRows = 32; // FIXME: need 64 rows
  static constexpr uint64_t kTileCols = 256;

  struct SharedStorage {
    __nv_bfloat16 smem_tile[kTileRows][kTileCols];
  };

  struct Arguments {
    // see the `a2a_kernel()` description of these fields
    uint64_t* out_offs;
    uint8_t* token_owner;
    uint64_t* local_token_to_remote_token_idx;

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
      char* smem_buf
  ) {
    auto& storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    constexpr uint64_t kWarpgroupThreads = 128;

    const int tid = threadIdx.x % kWarpgroupThreads;
    const int warp = tid / 32;
    const int lane = tid % 32;

    // Dump registers arranged as https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-D.png
    // to shared memory in row-major order. This definitely has some unwanted bank conflicts.
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
      #pragma unroll
      for (int col = 0; col < 32; ++col) {
        const int out_row = warp * 16 + row * 8 + (lane / 4);

        if (out_row >= kTileRows) continue; // FIXME

        const int out_col = 2 * (col * 4 + (tid % 4));
        storage.smem_tile[out_row][out_col] = __float2bfloat16(accumulators(
          cute::make_coord(0, row, col),
          0, 0
        ));
        storage.smem_tile[out_row][out_col + 1] = __float2bfloat16(accumulators(
          cute::make_coord(1, row, col),
          0, 0
        ));
      }
    }
    asm volatile("bar.sync 1, 128;");

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

    const uint64_t tile_row_start = cute::get<0>(blk_shape_MNK) * m_coord;
    const uint64_t tile_col_start = cute::get<1>(blk_shape_MNK) * n_coord;
    // The M dimension has 2 64-sizes sub-tiles, so we have to add an offset.
    const uint64_t subtile_row_start = cute::get<0>(
      thread_coords(
        cute::make_coord(0, 0, 0),
        0, 0
      )
    );
    const uint64_t tile_row_offset = (subtile_row_start - tile_row_start) / kTileRows * kTileRows;
    const uint64_t cur_out_off = params.out_offs[l_coord] + tile_row_start + tile_row_offset;

    #pragma unroll
    for (int row = warp; row < kTileRows; row += 4) {
      const auto* vector_row = reinterpret_cast<const uint4*>(storage.smem_tile[row]);

      // FIXME: should make sure the row is in-bounds.
      const uint64_t local_token_idx = cur_out_off + row;
      const uint8_t peer = params.token_owner[local_token_idx];

      __nv_bfloat16* peer_output = params.routed_outputs_ptrs[peer];
      const uint64_t remote_token_idx = params.local_token_to_remote_token_idx[local_token_idx];

      // Assume contiguous row-major layout.
      auto* out_ptr = (uint4*)(peer_output + remote_token_idx * params.dim + tile_col_start);
      *(out_ptr + lane) = vector_row[lane];
    }
    asm volatile("bar.sync 1, 128;");
  }

private:
  Params params;
};

}
