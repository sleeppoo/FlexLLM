CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror

.PHONY: test rtl-test clean

test: build/test_single_lane rtl-test
	./build/test_single_lane

rtl-test: build/tb_dsp58_dot3_int8
	vvp build/tb_dsp58_dot3_int8

build/tb_dsp58_dot3_int8: tests/tb_dsp58_dot3_int8.v rtl/dsp58_dot3_int8.v
	mkdir -p build
	iverilog -g2012 -Wall -DFLEXLLM_DSP58_BEHAVIORAL_SIM \
		-s tb_dsp58_dot3_int8 -o $@ $^

build/test_single_lane: tests/test_single_lane.cpp include/dsp58_dot3_ref.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude tests/test_single_lane.cpp -o $@

clean:
	rm -rf build
