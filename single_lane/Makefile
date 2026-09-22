CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Werror
VCS ?= vcs

.PHONY: test cpp-test rtl-test rtl-test-iverilog clean

test: cpp-test rtl-test

cpp-test: build/test_single_lane build/tb_dot3_pe build/tb_dot3_lane
	./build/test_single_lane
	./build/tb_dot3_pe
	./build/tb_dot3_lane

rtl-test: build/simv
	./build/simv

build/simv: tests/tb_dsp58_dot3_int8.v rtl/dsp58_dot3_int8.v
	mkdir -p build
	$(VCS) -full64 -sverilog -timescale=1ns/1ps \
		+define+FLEXLLM_DSP58_BEHAVIORAL_SIM \
		-top tb_dsp58_dot3_int8 -Mdir=build/vcs_csrc -o $@ $^

# Optional fallback for environments that have Icarus Verilog.
rtl-test-iverilog: build/tb_dsp58_dot3_int8
	vvp build/tb_dsp58_dot3_int8

build/tb_dsp58_dot3_int8: tests/tb_dsp58_dot3_int8.v rtl/dsp58_dot3_int8.v
	mkdir -p build
	iverilog -g2012 -Wall -DFLEXLLM_DSP58_BEHAVIORAL_SIM \
		-s tb_dsp58_dot3_int8 -o $@ $^

build/test_single_lane: tests/test_single_lane.cpp include/dsp58_dot3_ref.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude tests/test_single_lane.cpp -o $@

build/tb_dot3_pe: tests/tb_dot3_pe.cpp include/dsp58_dot3_ref.hpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

build/tb_dot3_lane: tests/tb_dot3_lane.cpp
	mkdir -p build
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

clean:
	rm -rf build
