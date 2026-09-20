#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "dsp58_dot3_ref.hpp"

using flexllm::single_lane::dot3_ref;

namespace {

std::vector<std::int32_t> lane_ref(const std::vector<std::int8_t>& x,
                                   const std::vector<std::int8_t>& weights,
                                   std::size_t out_channels) {
  const std::size_t in_dim = x.size();
  std::vector<std::int32_t> result(out_channels, 0);
  for (std::size_t out = 0; out < out_channels; ++out) {
    for (std::size_t r = 0; r < in_dim; r += 3) {
      const std::int8_t a1 = r + 1 < in_dim ? x[r + 1] : 0;
      const std::int8_t a2 = r + 2 < in_dim ? x[r + 2] : 0;
      const std::int8_t w1 = r + 1 < in_dim ? weights[out * in_dim + r + 1] : 0;
      const std::int8_t w2 = r + 2 < in_dim ? weights[out * in_dim + r + 2] : 0;
      result[out] += dot3_ref(x[r], a1, a2, weights[out * in_dim + r], w1, w2);
    }
  }
  return result;
}

std::vector<std::int32_t> naive(const std::vector<std::int8_t>& x,
                                const std::vector<std::int8_t>& weights,
                                std::size_t out_channels) {
  std::vector<std::int32_t> result(out_channels, 0);
  for (std::size_t out = 0; out < out_channels; ++out)
    for (std::size_t r = 0; r < x.size(); ++r)
      result[out] += static_cast<std::int32_t>(x[r]) *
                     static_cast<std::int32_t>(weights[out * x.size() + r]);
  return result;
}

void expect(bool condition, const std::string& name) {
  if (!condition) {
    std::cerr << "FAIL: " << name << '\n';
    std::exit(1);
  }
  std::cout << "PASS: " << name << '\n';
}

void compare_case(const std::string& name, const std::vector<std::int8_t>& x,
                  const std::vector<std::int8_t>& weights,
                  std::size_t out_channels) {
  expect(weights.size() == x.size() * out_channels, name + " shape");
  expect(lane_ref(x, weights, out_channels) == naive(x, weights, out_channels), name);
}

}  // namespace

int main() {
  expect(dot3_ref(0, 0, 0, 0, 0, 0) == 0, "dot3 all zeros");
  expect(dot3_ref(1, 2, 3, 4, 5, 6) == 32, "dot3 positive");
  expect(dot3_ref(-1, -2, -3, 4, 5, 6) == -32, "dot3 negative");
  expect(dot3_ref(-3, 4, -5, -6, -7, 8) == -50, "dot3 mixed signs");
  expect(dot3_ref(-128, 127, -128, -128, 127, 127) == 16257,
         "dot3 INT8 boundaries");

  compare_case("in_dim divisible by three", {1, -2, 3, 4, 5, -6},
               {2, 3, 4, -1, 2, -3}, 1);
  compare_case("in_dim tail of one", {7, -8, 9, -10},
               {1, 2, 3, 4}, 1);
  compare_case("in_dim tail of two", {7, -8, 9, -10, 11},
               {1, 2, 3, 4, 5}, 1);
  compare_case("multiple output channels",
               {1, -2, 3, -4, 5, -6, 7},
               {1, 1, 1, 1, 1, 1, 1,
                -1, 2, -3, 4, -5, 6, -7,
                -128, 127, 0, 1, -1, 2, -2}, 3);

  std::cout << "All single-lane software reference tests passed.\n";
  return 0;
}
