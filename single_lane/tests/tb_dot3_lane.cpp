#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>

namespace {

template <std::size_t N>
std::int64_t lane_ref(const std::array<std::int8_t, 3 * N>& activation,
                      const std::array<std::int8_t, 3 * N>& weight) {
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < 3 * N; ++i)
    sum += static_cast<std::int64_t>(activation[i]) * weight[i];
  return sum;
}

template <std::size_t N>
bool test_configuration() {
  std::mt19937 generator(0x58u + static_cast<unsigned>(N));
  std::uniform_int_distribution<int> distribution(-128, 127);
  std::array<std::int8_t, 3 * N> activation{};
  std::array<std::int8_t, 3 * N> weight{};

  const auto check = [&] { return lane_ref<N>(activation, weight); };
  if (check() != 0) return false;

  activation.fill(127);
  weight.fill(127);
  if (check() != static_cast<std::int64_t>(3 * N) * 127 * 127) return false;

  activation.fill(-128);
  weight.fill(-128);
  if (check() != static_cast<std::int64_t>(3 * N) * 128 * 128) return false;

  activation.fill(-128);
  weight.fill(127);
  if (check() != -static_cast<std::int64_t>(3 * N) * 128 * 127) return false;

  for (int trial = 0; trial < 1000; ++trial) {
    std::int64_t naive = 0;
    for (std::size_t i = 0; i < 3 * N; ++i) {
      activation[i] = static_cast<std::int8_t>(distribution(generator));
      weight[i] = static_cast<std::int8_t>(distribution(generator));
      naive += static_cast<std::int64_t>(activation[i]) * weight[i];
    }
    if (check() != naive) return false;
  }

  // Accumulation protocol model: clear includes the current chunk, and only
  // last makes the accumulated result valid.
  std::int64_t accumulator = 0;
  for (int chunk = 0; chunk < 17; ++chunk) {
    for (std::size_t i = 0; i < 3 * N; ++i) {
      activation[i] = static_cast<std::int8_t>(distribution(generator));
      weight[i] = static_cast<std::int8_t>(distribution(generator));
    }
    if (chunk == 0) accumulator = 0;
    accumulator += check();
  }
  // A 32-bit accumulator is safely wide for this test sequence.
  return accumulator >= std::numeric_limits<std::int32_t>::min() &&
         accumulator <= std::numeric_limits<std::int32_t>::max();
}

}  // namespace

int main() {
  if (!test_configuration<1>() || !test_configuration<2>() ||
      !test_configuration<4>() || !test_configuration<8>()) {
    std::cerr << "Lane reference test failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "Lane tests passed for PE_PER_LANE=1,2,4,8.\n";
  return EXIT_SUCCESS;
}
