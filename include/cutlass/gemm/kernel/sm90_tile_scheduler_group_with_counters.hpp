#include "cutlass/gemm/kernel/sm90_tile_scheduler_group.hpp"

namespace cutlass::gemm::kernel::detail {

template <class GroupProblemShape, int SchedulerPipelineStageCount>
class PersistentTileSchedulerSm90GroupWithCounters: public PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount> {
  using Base = PersistentTileSchedulerSm90Group<GroupProblemShape, SchedulerPipelineStageCount>;

public:
  PersistentTileSchedulerSm90GroupWithCounters() = default;

  CUTLASS_DEVICE explicit PersistentTileSchedulerSm90GroupWithCounters(
    const typename Base::Params& params, typename Base::SchedulerResponse* response_ptr
  ) : Base(params, response_ptr)
  {}

  CUTLASS_DEVICE
  void wait_on_group_counter(int32_t idx) {
    if (idx == -1) {
      // Invalid group, no need to wait.
      return;
    }
    // this acquire is paired with the release in the token dispatcher
    uint32_t spins = 0;
    auto* counter = Base::scheduler_params.problem_shapes_.group_counters + idx;
    while (__nv_atomic_load_n(counter, __NV_ATOMIC_ACQUIRE) != 0) {
      __nanosleep(16); // sleep for a couple of cycles; constant taken out of thin air
      ++spins;
      if (spins > 100'000'000) {
        printf(
          "Timedout when waiting for the counter for group %d to become zero: its value is still %u\n",
          idx,
          __nv_atomic_load_n(counter, __NV_ATOMIC_RELAXED)
        );
        __trap();
      }
    }
  }

  // Uses the same logic as the base class, but waits for the group tile to be ready first.
  template <typename TileSchedulerPipeline, typename TileSchedulerPipelineState>
  CUTLASS_DEVICE
  auto
  advance_to_next_work(
    TileSchedulerPipeline& scheduler_pipeline,
    TileSchedulerPipelineState scheduler_pipe_producer_state,
    uint32_t advance_count = 1) {

    Base::current_work_linear_idx_ += Base::total_grid_size_ * uint64_t(advance_count);
    const int prev_group_idx = Base::current_group_info_.group_idx;
    auto work_tile = Base::get_current_work_for_linear_idx(Base::current_work_linear_idx_);

    // CUSTOM LOGIC STARTS
    if (work_tile.L_idx != prev_group_idx) {
      if (canonical_lane_idx() == 0) {
        wait_on_group_counter(work_tile.L_idx);
      }
      // Only one warp (scheduler producer) arrives here, so we can use `__syncwarp()`.
      __syncwarp();
    }
    // CUSTOM LOGIC ENDS

    scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
    if (cute::elect_one_sync()) {
      Base::response_ptr_[scheduler_pipe_producer_state.index()] = work_tile;
      cutlass::arch::fence_view_async_shared();
      scheduler_pipeline.producer_commit(scheduler_pipe_producer_state);
    }
    return cute::make_tuple(work_tile, true);
  }

  // Uses the same tile as the base class, but waits for group #0 to be ready first.
  template <class ClusterShape>
  CUTLASS_DEVICE
  auto
  initial_work_tile_info(ClusterShape cluster_shape) {
    if (threadIdx.x == 0) {
      wait_on_group_counter(0);
    }
    __syncthreads();
    return Base::initial_work_tile_info(cluster_shape);
  }
};

}
