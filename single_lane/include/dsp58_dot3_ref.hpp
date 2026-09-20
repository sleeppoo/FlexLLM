#ifndef FLEXLLM_SINGLE_LANE_DSP58_DOT3_REF_HPP_
#define FLEXLLM_SINGLE_LANE_DSP58_DOT3_REF_HPP_

#include <cstdint>

namespace flexllm {
namespace single_lane {

// Software-only arithmetic model. This header is deliberately independent of
// ap_int and is never used as the synthesis implementation of the RTL PE.
inline std::int32_t dot3_ref(std::int8_t a0, std::int8_t a1, std::int8_t a2,
                            std::int8_t w0, std::int8_t w1, std::int8_t w2) {
  return static_cast<std::int32_t>(a0) * static_cast<std::int32_t>(w0) +
         static_cast<std::int32_t>(a1) * static_cast<std::int32_t>(w1) +
         static_cast<std::int32_t>(a2) * static_cast<std::int32_t>(w2);
}

}  // namespace single_lane
}  // namespace flexllm

#endif  // FLEXLLM_SINGLE_LANE_DSP58_DOT3_REF_HPP_
