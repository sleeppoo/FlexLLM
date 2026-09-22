#include <ap_int.h>

#include <cstdint>
#include <iostream>
#include <random>

#include "dot3_lane.hpp"

#ifndef PE_PER_LANE
#define PE_PER_LANE 4
#endif

#ifndef ACC_WIDTH
#define ACC_WIDTH 32
#endif

void dot3_lane_top(const ap_int<8> activation[PE_PER_LANE * 3],
                   const ap_int<8> weight[PE_PER_LANE * 3], bool acc_clear,
                   bool acc_valid, bool acc_last, ap_int<ACC_WIDTH>& result,
                   bool& result_valid);

namespace {

template <int N>
bool test_lane() {
  std::mt19937 generator(0x5800u + N);
  std::uniform_int_distribution<int> distribution(-128, 127);
  ap_int<8> activation[N][3];
  ap_int<8> weight[N][3];

  for (int trial = 0; trial < 1000; ++trial) {
    std::int64_t golden = 0;
    for (int pe = 0; pe < N; ++pe) {
      for (int element = 0; element < 3; ++element) {
        const int a = distribution(generator);
        const int w = distribution(generator);
        activation[pe][element] = a;
        weight[pe][element] = w;
        golden += a * w;
      }
    }
    ap_int<flexllm::single_lane::Dot3LaneWidths<N>::kOutputWidth> result;
    bool valid = false;
    flexllm::single_lane::dot3_lane<N>(activation, weight, false, true, false,
                                          result, valid);
    if (!valid || result.to_int64() != golden) return false;
  }
  return true;
}

template <int N>
bool test_accumulation() {
  ap_int<8> activation[N][3];
  ap_int<8> weight[N][3];
  std::int64_t golden = 0;
  ap_int<32> result = 0;
  bool valid = false;
  for (int chunk = 0; chunk < 7; ++chunk) {
    for (int pe = 0; pe < N; ++pe) {
      for (int element = 0; element < 3; ++element) {
        const int a = chunk - 3 * pe - element;
        const int w = 2 * element - pe + 1;
        activation[pe][element] = a;
        weight[pe][element] = w;
        golden += a * w;
      }
    }
    const bool last = chunk == 6;
    flexllm::single_lane::dot3_lane<N, true, 32>(
        activation, weight, chunk == 0, true, last, result, valid);
    if (valid != last) return false;
  }
  return result.to_int64() == golden;
}

bool test_configured_top() {
  ap_int<8> activation[PE_PER_LANE * 3];
  ap_int<8> weight[PE_PER_LANE * 3];
  std::int64_t golden = 0;
  for (int i = 0; i < PE_PER_LANE * 3; ++i) {
    const int a = (17 * i + 3) % 256 - 128;
    const int w = (29 * i + 11) % 256 - 128;
    activation[i] = a;
    weight[i] = w;
    golden += a * w;
  }
  ap_int<ACC_WIDTH> result = 0;
  bool valid = false;
  dot3_lane_top(activation, weight, false, true, false, result, valid);
  return valid && result.to_int64() == golden;
}

}  // namespace

int main() {
  const bool passed = test_lane<1>() && test_lane<2>() && test_lane<4>() &&
                      test_lane<8>() && test_accumulation<1>() &&
                      test_accumulation<2>() && test_accumulation<4>() &&
                      test_accumulation<8>() && test_configured_top();
  if (!passed) {
    std::cerr << "HLS PE/lane test failed\n";
    return 1;
  }
  std::cout << "HLS PE/lane tests passed for PE_PER_LANE=1,2,4,8.\n";
  return 0;
}
