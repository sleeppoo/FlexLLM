#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "dsp58_dot3_ref.hpp"

int main() {
  using flexllm::single_lane::dot3_ref;
  if (dot3_ref(0, 0, 0, 0, 0, 0) != 0 ||
      dot3_ref(1, 2, 3, 4, 5, 6) != 32 ||
      dot3_ref(-1, -2, -3, 4, 5, 6) != -32 ||
      dot3_ref(-128, -128, -128, -128, -128, -128) != 49152 ||
      dot3_ref(-128, -128, -128, 127, 127, 127) != -48768) {
    std::cerr << "PE reference test failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "All PE reference tests passed.\n";
  return EXIT_SUCCESS;
}
