# Self-documenting build helper for the Llama-3.2-1B-Instruct U280 flow.
#
# Common usage:
#   make host KERNEL=prefill
#   make c-sim KERNEL=decode
#   make tapa-compile KERNEL=decode
#   make xo-sim KERNEL=decode
#   make rapidstream-opt KERNEL=decode
#   make rapidstream-impl KERNEL=decode IMPL_WORKERS=2
#   make board-run KERNEL=decode BITSTREAM=/path/to/SpinQuant_Decoding.xclbin
#   make demo-build
#   make demo-run PREF_BITSTREAM=/path/to/pref.xclbin DEC_BITSTREAM=/path/to/dec.xclbin

SHELL := /bin/bash

PROJECT_ROOT := $(abspath .)
FLEXLLM_HOME ?= $(abspath ..)
LLAMA_CPP_ROOT ?= $(HOME)/llama.cpp

PLATFORM ?= xilinx_u280_gen3x16_xdma_1_202211_1
CLOCK_PERIOD ?= 3.33
TAPA_JOBS ?= 128
RAPIDSTREAM_JOBS ?= 32
IMPL_WORKERS ?= 2

RUN_DIR ?= $(PROJECT_ROOT)/run
BUILD_DIR ?= $(PROJECT_ROOT)/build_make
XOSIM_WORK_DIR ?= $(BUILD_DIR)/xosim_$(KERNEL)
RAPIDSTREAM_DIR ?= $(PROJECT_ROOT)/RapidStream
RAPIDSTREAM_WORK_DIR ?= $(RAPIDSTREAM_DIR)/build

RAPIDSTREAM_WRAPPER ?= $(FLEXLLM_HOME)/tools/rapidstream_tapaopt_ap_done.py

KERNEL ?= decode

ifeq ($(KERNEL),prefill)
  TOP := SpinQuant_Prefilling
  HOST_SRC := SpinQuant_Prefilling_mem_opt_tb.cpp
  HOST_BIN := $(RUN_DIR)/SpinQuant_Prefilling_mem_opt
  KERNEL_HDR := SpinQuant_Prefilling_mem_opt.h
  XO := $(BUILD_DIR)/SpinQuant_Prefilling.xo
  RS_CONFIG_SRC := RapidStream_pref_u280
  RS_FLOORPLAN := floorplan_config_mem_opt.json
  CONNECTIVITY_INI := pref_link_config_u280_mem_opt.ini
  DEFAULT_BITSTREAM := $(RUN_DIR)/bitstreams/SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
else ifeq ($(KERNEL),decode)
  TOP := SpinQuant_Decoding
  HOST_SRC := SpinQuant_Decoding_mem_opt_logits_tb.cpp
  HOST_BIN := $(RUN_DIR)/SpinQuant_Decoding_mem_opt_logits
  KERNEL_HDR := SpinQuant_Decoding_mem_opt_logits.h
  XO := $(BUILD_DIR)/SpinQuant_Decoding.xo
  RS_CONFIG_SRC := RapidStream_dec_u280
  RS_FLOORPLAN := floorplan_config_mem_opt.json
  CONNECTIVITY_INI := dec_link_config_u280_mem_opt.ini
  DEFAULT_BITSTREAM := $(RUN_DIR)/bitstreams/SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
else
  $(error KERNEL must be either prefill or decode)
endif

BITSTREAM ?= $(DEFAULT_BITSTREAM)
CONNECTIVITY_INI_PATH := $(PROJECT_ROOT)/$(CONNECTIVITY_INI)
PREF_BITSTREAM ?= $(RUN_DIR)/bitstreams/SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
DEC_BITSTREAM ?= $(RUN_DIR)/bitstreams/SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin
GGUF ?= $(RUN_DIR)/llama-3.2-1b-f16.gguf
PROMPT ?= $(RUN_DIR)/my_prompt_0.txt
ANSWER ?= $(RUN_DIR)/my_answer.txt

.PHONY: help env prepare host c-sim tapa-compile xo-sim rapidstream-prepare rapidstream-opt rapidstream-impl board-run demo-build demo-run clean-build

help:
	@echo "FlexLLM Llama-3.2-1B-Instruct U280 Makefile"
	@echo ""
	@echo "Select kernel with KERNEL=prefill or KERNEL=decode. Default: decode."
	@echo ""
	@echo "Build and simulation targets:"
	@echo "  make host KERNEL=decode          Build the TAPA host test binary with tapa g++"
	@echo "  make c-sim KERNEL=decode         Run C simulation using the host test binary"
	@echo "  make tapa-compile KERNEL=decode  Generate XO with tapa compile"
	@echo "  make xo-sim KERNEL=decode        Run TAPA fast cosim using the generated XO"
	@echo ""
	@echo "RapidStream and board targets:"
	@echo "  make rapidstream-opt KERNEL=decode       Run RapidStream export and patch generated XO files"
	@echo "  make rapidstream-impl KERNEL=decode      Run patched RapidStream export + v++ implementation"
	@echo "  make board-run KERNEL=decode BITSTREAM=/path/to/kernel.xclbin"
	@echo ""
	@echo "End-to-end demo targets:"
	@echo "  make demo-build"
	@echo "  make demo-run PREF_BITSTREAM=/path/to/pref.xclbin DEC_BITSTREAM=/path/to/dec.xclbin"
	@echo ""
	@echo "Current selection:"
	@echo "  KERNEL=$(KERNEL)"
	@echo "  HOST_SRC=$(HOST_SRC)"
	@echo "  KERNEL_HDR=$(KERNEL_HDR)"
	@echo "  XO=$(XO)"

env:
	@echo "PROJECT_ROOT=$(PROJECT_ROOT)"
	@echo "FLEXLLM_HOME=$(FLEXLLM_HOME)"
	@echo "LLAMA_CPP_ROOT=$(LLAMA_CPP_ROOT)"
	@echo "PLATFORM=$(PLATFORM)"
	@echo "TOP=$(TOP)"
	@echo "HOST_BIN=$(HOST_BIN)"
	@echo "BITSTREAM=$(BITSTREAM)"

prepare:
	@mkdir -p "$(RUN_DIR)" "$(BUILD_DIR)"

host: prepare
	tapa g++ -- "$(HOST_SRC)" \
		-o "$(HOST_BIN)" \
		-I"$(FLEXLLM_HOME)/Modules"

c-sim: host
	cd "$(PROJECT_ROOT)" && "$(HOST_BIN)"

tapa-compile: prepare
	tapa compile -j "$(TAPA_JOBS)" \
		--top "$(TOP)" \
		--platform "$(PLATFORM)" \
		--clock-period "$(CLOCK_PERIOD)" \
		-f "$(KERNEL_HDR)" \
		-c \
		-I"$(FLEXLLM_HOME)/Modules" \
		-o "$(XO)"

xo-sim: host tapa-compile
	cd "$(PROJECT_ROOT)" && "$(HOST_BIN)" \
		--bitstream "$(XO)" \
		-xosim_work_dir "$(XOSIM_WORK_DIR)" \
		-xosim_save_waveform

rapidstream-prepare:
	@mkdir -p "$(RAPIDSTREAM_DIR)"
	cp "$(RS_CONFIG_SRC)"/*.json "$(RAPIDSTREAM_DIR)/"

rapidstream-opt: tapa-compile rapidstream-prepare
	python "$(RAPIDSTREAM_WRAPPER)" \
		-j "$(RAPIDSTREAM_JOBS)" \
		--work-dir "$(RAPIDSTREAM_WORK_DIR)" \
		--tapa-xo-path "$(XO)" \
		--device-config "$(RAPIDSTREAM_DIR)/u280_device.json" \
		--floorplan-config "$(RAPIDSTREAM_DIR)/$(RS_FLOORPLAN)" \
		--pipeline-config "$(RAPIDSTREAM_DIR)/pipeline_config.json" \
		--connectivity-ini "$(CONNECTIVITY_INI_PATH)"

rapidstream-impl: tapa-compile rapidstream-prepare
	python "$(RAPIDSTREAM_WRAPPER)" \
		--impl-workers "$(IMPL_WORKERS)" \
		-j "$(RAPIDSTREAM_JOBS)" \
		--work-dir "$(RAPIDSTREAM_WORK_DIR)" \
		--tapa-xo-path "$(XO)" \
		--device-config "$(RAPIDSTREAM_DIR)/u280_device.json" \
		--floorplan-config "$(RAPIDSTREAM_DIR)/$(RS_FLOORPLAN)" \
		--pipeline-config "$(RAPIDSTREAM_DIR)/pipeline_config.json" \
		--run-impl \
		--implementation-config "$(RAPIDSTREAM_DIR)/impl_config.json" \
		--connectivity-ini "$(CONNECTIVITY_INI_PATH)"

board-run: host
	cd "$(PROJECT_ROOT)" && "$(HOST_BIN)" --bitstream "$(BITSTREAM)"

demo-build:
	@mkdir -p "$(RUN_DIR)"
	tapa g++ -- SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp \
		-I"$(FLEXLLM_HOME)/Modules" \
		-I"$(LLAMA_CPP_ROOT)" \
		-I"$(LLAMA_CPP_ROOT)/include" \
		-I"$(LLAMA_CPP_ROOT)/ggml/include" \
		-I"$(LLAMA_CPP_ROOT)/ggml/include/ggml" \
		"$(LLAMA_CPP_ROOT)/build/bin/libllama.so" \
		-Wl,-rpath,"$(LLAMA_CPP_ROOT)/build/bin" \
		-lpthread -ldl -lm \
		-o "$(RUN_DIR)/SpinQuant_Prefilling_Decoding_mem_opt_demo"

demo-run: demo-build
	cd "$(RUN_DIR)" && ./SpinQuant_Prefilling_Decoding_mem_opt_demo \
		--bitstream_pref "$(PREF_BITSTREAM)" \
		--bitstream_dec "$(DEC_BITSTREAM)" \
		"$(GGUF)" \
		"$(PROMPT)" \
		"$(ANSWER)"

clean-build:
	rm -rf "$(BUILD_DIR)"
