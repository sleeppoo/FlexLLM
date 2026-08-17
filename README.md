# 🔥 FlexLLM: A Composable HLS Library for Rapid LLM Accelerator Design
[![DOI](https://zenodo.org/badge/1109534356.svg)](https://doi.org/10.5281/zenodo.18793354)

FlexLLM is a **composable High-Level Synthesis (HLS) library** for rapidly building **hybrid temporal–spatial accelerators** for Large Language Models (LLMs).  
It provides parameterized module templates, optimized memory-access/dataflow components, and a complete quantization suite, enabling FPGA-based LLM systems to be built with **minimal manual engineering effort**.

Using FlexLLM, we implemented a **full Llama-3.2-1B inference system**—including prefill, decode, tokenizer integration, and long-context memory—**in under two months with ~1K lines of code**.

---

## ✨ Key Features

- **Composable HLS Library** for LLM accelerator development  
- **Hybrid Temporal–Spatial Architecture**  
- **Hardware-Efficient Quantization Suite**  
- **Hierarchical Memory Transformer (HMT) Plug-In**  
- **FPGA Deployment Ready**

---

## 📊 Performance Summary

### AMD U280 FPGA (16nm) vs. NVIDIA A100 GPU (7nm)
- 1.29× end-to-end speedup  
- 1.64× higher decode throughput  
- 3.14× better energy efficiency  

### Projected V80 FPGA (7nm)
- 4.71× end-to-end speedup  
- 6.55× decode throughput  
- 4.13× energy efficiency  

### Long-Context (with HMT)
- 23.23× reduced prefill latency  
- 64× longer context window  

---

## Current Fix Notes

This branch also contains the current U280 decode fixes used for FPGA draft-model debugging in the heterogeneous FPGA + AMD MI210 speculative decoding project.

- Decode MHA K-cache handling was updated to use the corrected scalar K-cache buffer path. This avoids corrupting the last partially filled K-cache lane when the prefix length is not a multiple of 32.
- Active TAPA tasks that access mmap/cache state were changed from `tapa::detach` to `tapa::join`. This fixes the repeated/continuous decode hang observed when decode is invoked multiple times with advancing `pre_seq_len`.
- `SpinQuant_Decoding_mem_opt_logits.h` and `SpinQuant_Decoding_mem_opt_logits_tb.cpp` add a decode-kernel variant that returns sampled token IDs plus top-k logits for speculative-decoding acceptance experiments.
- `SpinQuant_Prefilling_Decoding_mem_opt_demo.cpp` now calls the logits decode kernel and reads sampled token IDs from the returned logits structure.
- `SpinQuant_Llama_32_1B_Ins/parameters/RoPE_sin_cos.h` is restored to the normal Llama-3.2-1B-Instruct `HEAD_DIM=64` table shape: `PE_sin[2048][32]` and `PE_cos[2048][32]`.
- `tools/patch_rapidstream_ap_done.py` is a reusable RapidStream/TAPA XO patch helper. It patches generated `__global_fsm_*_fsm.v` files so `global_fsm_ap_done` is correctly driven by `ap_done__q0` before running `v++`.
- `tools/rapidstream_tapaopt_ap_done.py` wraps `rapidstream-tapaopt` and automatically applies the ap_done patch to every generated solution XO. If `--run-impl` is present, the wrapper runs RapidStream export first, patches the XO files in place, and then launches the generated `v++` scripts so implementation uses the fixed XO.

Recommended RapidStream flow with automatic XO patching:

```
python tools/rapidstream_tapaopt_ap_done.py \
  -j 32 \
  --work-dir ./RapidStream/build \
  --tapa-xo-path SpinQuant_Decoding.xo \
  --device-config ./RapidStream/u280_device.json \
  --floorplan-config ./RapidStream/floorplan_config_mem_opt.json \
  --pipeline-config ./RapidStream/pipeline_config.json \
  --connectivity-ini dec_link_config_u280_mem_opt.ini
```

With implementation:

```
python tools/rapidstream_tapaopt_ap_done.py \
  --impl-workers 2 \
  -j 32 \
  --work-dir ./RapidStream/build \
  --tapa-xo-path SpinQuant_Decoding.xo \
  --device-config ./RapidStream/u280_device.json \
  --floorplan-config ./RapidStream/floorplan_config_mem_opt.json \
  --pipeline-config ./RapidStream/pipeline_config.json \
  --run-impl \
  --implementation-config ./RapidStream/impl_config.json \
  --connectivity-ini dec_link_config_u280_mem_opt.ini
```

Manual single-XO patch:

```
python tools/patch_rapidstream_ap_done.py \
  path/to/solution_N/updated.xo \
  -o path/to/solution_N/updated_global_done_fix.xo
```

The fixes were host-compiled with TAPA for the mem-opt decode, mem-opt-new decode, logits decode, and prefill+decode demo hosts. The tested U280 decode bitstream is the all-join logits decode build generated from the patched RapidStream XO.

---

## 📁 Repository Layout

```
FlexLLM/
├─ Modules/                          # Core FlexLLM module library (compute, quant, memory, data movement)
│
├─ SpinQuant_Llama_32_1B_Ins/        # Llama-3.2-1B-Instruct accelerator (SpinQuant)
│  ├─ parameters/                    # Downloaded model parameters
│  ├─ RapidStream_pref_u280/         # Prefill RapidStream config (U280)
│  ├─ RapidStream_dec_u280/          # Decode RapidStream config (U280)
│  ├─ run/                           # Bitstreams, hosts, and test scripts
│  │  ├─ bitstreams/                 # FPGA .xclbin files
│  │  ├─ parameters/                 # Downloaded parameters
│  │  ├─ llama-3.2-1b-f16.gguf       # Tokenizer (download required)
│  │  ├─ SpinQuant_Prefilling_Decoding_mem_opt
│  │  ├─ SpinQuant_Prefilling_Decoding_mem_opt_demo
│  │  └─ test files (.py/.txt/.csv)
│  └─ TAPA files                     # TAPA HLS kernels, host code, memory configs
│
├─ SpinQuant_Llama_32_1B/            # Llama-3.2-1B accelerator (SpinQuant)
├─ HMT_SpinQuant_Llama_32_1B/        # Llama-3.2-1B-Instruct + SpinQuant + HMT
└─ README.md
```

---

## 📦 Download Required Files

Download parameters & GGUF from:

https://drive.google.com/drive/folders/1KyEL9gC9Wge9l1m5t2lc79uQhK0jYyq8?usp=sharing

Place them in:

```
FlexLLM/SpinQuant_Llama_32_1B_Ins/parameters/
FlexLLM/SpinQuant_Llama_32_1B_Ins/run/parameters/
FlexLLM/SpinQuant_Llama_32_1B_Ins/run/llama-3.2-1b-f16.gguf
```

---

## 🧰 Requirements

- Ubuntu 20.04 / 22.04  
- XRT installed  
- Vitis 2022.2  
- TAPA CLI  
- Compatible FPGA board  

Check FPGA:

```
xbutil examine
```

---

## 🛠 Build (Host Only)

```
cd SpinQuant_Llama_32_1B_Ins

# Build the fixed logits decode host.
make host KERNEL=decode

# Build the mem-opt prefill host.
make host KERNEL=prefill
```

---

## 🧪 Build, Simulation, and RapidStream Flow

`SpinQuant_Llama_32_1B_Ins/Makefile` wraps the common U280 flows from `Tapa_Command`. Select the kernel with `KERNEL=prefill` or `KERNEL=decode`. The default is `decode`, using the fixed logits decode kernel.

Host C simulation:

```
make c-sim KERNEL=decode
make c-sim KERNEL=prefill
```

Generate TAPA XO:

```
make tapa-compile KERNEL=decode
make tapa-compile KERNEL=prefill
```

Run TAPA fast cosim with the generated XO:

```
make xo-sim KERNEL=decode
make xo-sim KERNEL=prefill
```

Run RapidStream optimization with automatic ap_done FSM patching:

```
make rapidstream-opt KERNEL=decode
make rapidstream-opt KERNEL=prefill
```

Run RapidStream plus patched v++ implementation:

```
make rapidstream-impl KERNEL=decode IMPL_WORKERS=2
make rapidstream-impl KERNEL=prefill IMPL_WORKERS=2
```

Run a generated bitstream on board:

```
make board-run KERNEL=decode BITSTREAM=/path/to/SpinQuant_Decoding.xclbin
make board-run KERNEL=prefill BITSTREAM=/path/to/SpinQuant_Prefilling.xclbin
```

Build and run the prefill+decode demo with existing bitstreams:

```
make demo-build
make demo-run \
  PREF_BITSTREAM=/path/to/SpinQuant_Prefilling.xclbin \
  DEC_BITSTREAM=/path/to/SpinQuant_Decoding.xclbin \
  GGUF=/path/to/llama-3.2-1b-f16.gguf \
  PROMPT=/path/to/my_prompt.txt \
  ANSWER=/path/to/my_answer.txt
```

The RapidStream Makefile targets call `tools/rapidstream_tapaopt_ap_done.py`, so generated solution XO files are patched before manual or automatic v++ implementation.

---

## 🚀 Run on U280

```
cd SpinQuant_Llama_32_1B_Ins
make demo-run \
  PREF_BITSTREAM=run/bitstreams/SpinQuant_Prefilling_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin \
  DEC_BITSTREAM=run/bitstreams/SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.xclbin \
  GGUF=run/llama-3.2-1b-f16.gguf \
  PROMPT=run/my_prompt_0.txt \
  ANSWER=run/my_answer.txt
```

---

## 📝 Notes for V80 Support

V80 results are estimates. Full bitstreams coming soon.

---

## 🙏 Acknowledgments

We thank AMD — **Fraser Nicholas** and **Michaela Blott** — for support and guidance.
