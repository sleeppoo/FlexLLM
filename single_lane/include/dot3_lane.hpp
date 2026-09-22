#ifndef FLEXLLM_SINGLE_LANE_DOT3_LANE_HPP_
#define FLEXLLM_SINGLE_LANE_DOT3_LANE_HPP_

#include <ap_int.h>

#include "dsp58_dot3_pe.hpp"

namespace flexllm {
namespace single_lane {

constexpr int ceil_log2_constexpr(unsigned value) {
  int result = 0;
  unsigned power = 1;
  while (power < value) {
    power <<= 1;
    ++result;
  }
  return result;
}

// Each PE has a signed 24-bit interface (the trusted primitive's width).
// Adding N such results needs 24+ceil(log2(N)) bits.  ACC_WIDTH is explicit
// because only the caller knows the maximum number of accumulated chunks.
template <int NumPes, bool EnableAccum = false, int AccumulatorWidth = 32>
struct Dot3LaneWidths {
  static_assert(NumPes > 0, "PE_PER_LANE must be positive");
  static constexpr int kPeOutputWidth = 24;
  static constexpr int kLaneSumWidth =
      kPeOutputWidth + ceil_log2_constexpr(NumPes);
  static_assert(!EnableAccum || AccumulatorWidth >= kLaneSumWidth,
                "ACC_WIDTH must hold one complete lane sum");
  static constexpr int kOutputWidth =
      EnableAccum ? AccumulatorWidth : kLaneSumWidth;
};

// One lane transaction consumes 3*PE_PER_LANE activation/weight pairs.
// The PE loop is completely unrolled: II=1 therefore requires N independent
// black-box instances.  All adds are explicitly bound to fabric so the target
// resource model is exactly PE_PER_LANE DSP58s.
//
// ENABLE_ACCUM=false:
//   result is the current lane sum; result_valid mirrors acc_valid.
// ENABLE_ACCUM=true:
//   acc_valid qualifies an input chunk. acc_clear starts a new accumulation
//   (the current chunk is included), and acc_last marks the returned final sum.
template <int NumPes, bool EnableAccum = false, int AccumulatorWidth = 32>
void dot3_lane(
    const ap_int<8> activation[NumPes][3],
    const ap_int<8> weight[NumPes][3], bool acc_clear, bool acc_valid,
    bool acc_last,
    ap_int<Dot3LaneWidths<NumPes, EnableAccum, AccumulatorWidth>::kOutputWidth>&
        result,
    bool& result_valid) {
#pragma HLS inline off
#pragma HLS pipeline II=1
#pragma HLS array_partition variable=activation complete dim=0
#pragma HLS array_partition variable=weight complete dim=0
#pragma HLS allocation function instances=dsp58_dot3_int8 limit=NumPes

  using Widths = Dot3LaneWidths<NumPes, EnableAccum, AccumulatorWidth>;
  ap_int<24> partial[NumPes];
#pragma HLS array_partition variable=partial complete dim=1

pe_loop:
  for (int pe = 0; pe < NumPes; ++pe) {
#pragma HLS unroll
    dsp58_dot3_pe(activation[pe][0], activation[pe][1], activation[pe][2],
                  weight[pe][0], weight[pe][1], weight[pe][2], partial[pe]);
  }

  ap_int<Widths::kLaneSumWidth> lane_sum = 0;
#pragma HLS bind_op variable=lane_sum op=add impl=fabric
reduction_loop:
  for (int pe = 0; pe < NumPes; ++pe) {
#pragma HLS unroll
    lane_sum += ap_int<Widths::kLaneSumWidth>(partial[pe]);
  }

  if constexpr (EnableAccum) {
    static ap_int<AccumulatorWidth> accumulator = 0;
#pragma HLS bind_op variable=accumulator op=add impl=fabric
    result_valid = acc_valid && acc_last;
    if (acc_valid) {
      const ap_int<AccumulatorWidth> base =
          acc_clear ? ap_int<AccumulatorWidth>(0) : accumulator;
      const ap_int<AccumulatorWidth> next =
          base + ap_int<AccumulatorWidth>(lane_sum);
#pragma HLS bind_op variable=next op=add impl=fabric
      accumulator = next;
      result = next;
    }
  } else {
    (void)acc_clear;
    (void)acc_last;
    result = lane_sum;
    result_valid = acc_valid;
  }
}

}  // namespace single_lane
}  // namespace flexllm

#endif  // FLEXLLM_SINGLE_LANE_DOT3_LANE_HPP_
