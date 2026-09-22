#ifndef FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_
#define FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_

#include <ap_int.h>

// Vitis HLS replaces this non-inlined C model with rtl/dsp58_dot3_int8.v via
// blackbox/dsp58_dot3_int8.json. The result is deliberately a C return value:
// an output-reference temporary creates a false inter-transaction dependence
// in Vitis HLS 2025.1 and limits a latency-1, II-1 black box to lane II=2.
// Do not put an accumulating expression here.
ap_int<24> dsp58_dot3_int8(ap_int<8> a0, ap_int<8> a1, ap_int<8> a2,
                           ap_int<8> w0, ap_int<8> w1, ap_int<8> w2);

#endif  // FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_
