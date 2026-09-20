#include "single_lane_linear.hpp"

// Small synthesis top used to validate the lane independently before it is
// composed into the TAPA decode graph.
void single_lane_top(hls::stream<ap_int<8>>& activations,
                     hls::stream<ap_int<8>>& weights,
                     hls::stream<ap_int<32>>& outputs,
                     int in_dim, int out_channels) {
#pragma HLS interface axis port=activations
#pragma HLS interface axis port=weights
#pragma HLS interface axis port=outputs
#pragma HLS interface s_axilite port=in_dim bundle=control
#pragma HLS interface s_axilite port=out_channels bundle=control
#pragma HLS interface s_axilite port=return bundle=control
  flexllm::single_lane::single_lane_linear_int8<8192, 256, 32>(
      activations, weights, outputs, in_dim, out_channels);
}
