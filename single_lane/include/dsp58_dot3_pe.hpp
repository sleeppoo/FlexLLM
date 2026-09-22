#ifndef FLEXLLM_SINGLE_LANE_DSP58_DOT3_PE_HPP_
#define FLEXLLM_SINGLE_LANE_DSP58_DOT3_PE_HPP_

#include <ap_int.h>

#include "dsp58_dot3.hpp"

namespace flexllm {
namespace single_lane {

// One PE: six signed INT8 inputs, one signed 24-bit result.
//
// The underlying, experimentally verified RTL black box has latency 1, II 1,
// and contains exactly one DSP58 configured with DSP_MODE=INT8.  This wrapper
// deliberately contains no arithmetic so that it cannot alter DSP inference.
inline ap_int<24> dsp58_dot3_pe(ap_int<8> a0, ap_int<8> a1, ap_int<8> a2,
                               ap_int<8> w0, ap_int<8> w1, ap_int<8> w2) {
#pragma HLS inline
  return dsp58_dot3_int8(a0, a1, a2, w0, w1, w2);
}

}  // namespace single_lane
}  // namespace flexllm

#endif  // FLEXLLM_SINGLE_LANE_DSP58_DOT3_PE_HPP_
