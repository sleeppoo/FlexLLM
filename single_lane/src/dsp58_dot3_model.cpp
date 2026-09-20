#include "dsp58_dot3.hpp"

// C simulation model required by the Vitis RTL-blackbox flow. The JSON file
// replaces this function during synthesis and RTL co-simulation. Keeping it in
// a separate translation unit prevents accidental use as the hardware PE.
void dsp58_dot3_int8(ap_int<8> a0, ap_int<8> a1, ap_int<8> a2,
                     ap_int<8> w0, ap_int<8> w1, ap_int<8> w2,
                     ap_int<24>& dot3) {
#pragma HLS inline off
  dot3 = ap_int<24>(a0) * ap_int<24>(w0) +
         ap_int<24>(a1) * ap_int<24>(w1) +
         ap_int<24>(a2) * ap_int<24>(w2);
}
