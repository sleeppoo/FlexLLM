#ifndef FLEXLLM_SINGLE_LANE_LINEAR_HPP_
#define FLEXLLM_SINGLE_LANE_LINEAR_HPP_

#include <ap_int.h>
#include <hls_stream.h>

#include "dsp58_dot3.hpp"

namespace flexllm {
namespace single_lane {

// One token x one contiguous output-channel tile.
//
// Stream contract:
//   activations: exactly in_dim scalar values, x[0..in_dim-1]
//   weights:     channel-major values; for each output channel, W[o][0..in_dim-1]
//   outputs:     exactly out_channels signed accumulators
//
// The activation buffer is analogous to FlexLLM's decode A buffer. The scalar,
// channel-major weight stream is intentionally a narrow reusable-lane boundary;
// an adapter can later transpose FlexLLM's reduction-major vector weight packs.
template <int MaxInputDim, int MaxOutputTile, int AccWidth = 32>
void single_lane_linear_int8(hls::stream<ap_int<8>>& activations,
                             hls::stream<ap_int<8>>& weights,
                             hls::stream<ap_int<AccWidth>>& outputs,
                             int in_dim, int out_channels) {
  ap_int<8> activation_buffer[MaxInputDim];
#pragma HLS bind_storage variable=activation_buffer type=ram_2p impl=bram

  load_activations:
  for (int r = 0; r < in_dim; ++r) {
#pragma HLS pipeline II=1
    activation_buffer[r] = activations.read();
  }

  output_channel_loop:
  for (int out = 0; out < out_channels; ++out) {
#pragma HLS loop_tripcount min=1 max=MaxOutputTile
    ap_int<AccWidth> acc = 0;

    reduction_group_loop:
    for (int r = 0; r < in_dim; r += 3) {
#pragma HLS pipeline II=1
      const ap_int<8> a0 = activation_buffer[r];
      const ap_int<8> a1 = (r + 1 < in_dim) ? activation_buffer[r + 1] : ap_int<8>(0);
      const ap_int<8> a2 = (r + 2 < in_dim) ? activation_buffer[r + 2] : ap_int<8>(0);

      // Never read past the valid weight stream. Missing tail operands are zero.
      const ap_int<8> w0 = weights.read();
      const ap_int<8> w1 = (r + 1 < in_dim) ? weights.read() : ap_int<8>(0);
      const ap_int<8> w2 = (r + 2 < in_dim) ? weights.read() : ap_int<8>(0);

      ap_int<24> partial;
      dsp58_dot3_int8(a0, a1, a2, w0, w1, w2, partial);
      // Long accumulation stays outside the DSP58 dot3 blackbox.
      acc += ap_int<AccWidth>(partial);
    }
    outputs.write(acc);
  }
}

}  // namespace single_lane
}  // namespace flexllm

#endif  // FLEXLLM_SINGLE_LANE_LINEAR_HPP_
