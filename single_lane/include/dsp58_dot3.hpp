#ifndef FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_
#define FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_

#include <ap_int.h>

// Vitis HLS replaces this non-inlined C model with rtl/dsp58_dot3_int8.v via
// blackbox/dsp58_dot3_int8.json. Do not put an accumulating expression here.
void dsp58_dot3_int8(ap_int<8> a0, ap_int<8> a1, ap_int<8> a2,
                     ap_int<8> w0, ap_int<8> w1, ap_int<8> w2,
                     ap_int<24>& dot3);

#endif  // FLEXLLM_SINGLE_LANE_DSP58_DOT3_HPP_
