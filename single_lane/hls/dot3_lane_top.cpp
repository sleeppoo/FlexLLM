#include "dot3_lane.hpp"

#ifndef PE_PER_LANE
#define PE_PER_LANE 4
#endif

#ifndef ENABLE_ACCUM
#define ENABLE_ACCUM 0
#endif

#ifndef ACC_WIDTH
#define ACC_WIDTH 32
#endif

// Change PE_PER_LANE/ENABLE_ACCUM through hls_config.cfg cflags.  Flattened
// ports keep the generated IP boundary simple while dot3_lane uses [PE][3].
void dot3_lane_top(const ap_int<8> activation[PE_PER_LANE * 3],
                   const ap_int<8> weight[PE_PER_LANE * 3], bool acc_clear,
                   bool acc_valid, bool acc_last, ap_int<ACC_WIDTH>& result,
                   bool& result_valid) {
#pragma HLS interface ap_ctrl_none port=return
#pragma HLS interface ap_none port=activation
#pragma HLS interface ap_none port=weight
#pragma HLS interface ap_none port=acc_clear
#pragma HLS interface ap_none port=acc_valid
#pragma HLS interface ap_none port=acc_last
#pragma HLS interface ap_none port=result
#pragma HLS interface ap_none port=result_valid
#pragma HLS array_partition variable=activation complete dim=1
#pragma HLS array_partition variable=weight complete dim=1

  ap_int<8> pe_activation[PE_PER_LANE][3];
  ap_int<8> pe_weight[PE_PER_LANE][3];
#pragma HLS array_partition variable=pe_activation complete dim=0
#pragma HLS array_partition variable=pe_weight complete dim=0
copy_inputs:
  for (int pe = 0; pe < PE_PER_LANE; ++pe) {
#pragma HLS unroll
    for (int element = 0; element < 3; ++element) {
#pragma HLS unroll
      pe_activation[pe][element] = activation[3 * pe + element];
      pe_weight[pe][element] = weight[3 * pe + element];
    }
  }

  constexpr bool kEnableAccum = ENABLE_ACCUM != 0;
  using Widths = flexllm::single_lane::Dot3LaneWidths<
      PE_PER_LANE, kEnableAccum, ACC_WIDTH>;
  static_assert(ACC_WIDTH >= Widths::kOutputWidth,
                "top-level result port is too narrow");
  ap_int<Widths::kOutputWidth> lane_result;
  flexllm::single_lane::dot3_lane<PE_PER_LANE, kEnableAccum, ACC_WIDTH>(
      pe_activation, pe_weight, acc_clear, acc_valid, acc_last, lane_result,
      result_valid);
  result = ap_int<ACC_WIDTH>(lane_result);
}
