# FlexLLM DSP58 dot3 PE and parameterized lane

## Hardware hierarchy

`dsp58_dot3_int8` is the trusted RTL primitive and is intentionally unchanged.
`dsp58_dot3_pe` is a zero-arithmetic HLS wrapper that gives that primitive its
architectural meaning: one PE. `dot3_lane<PE_PER_LANE, ENABLE_ACCUM, ACC_WIDTH>`
fully unrolls `PE_PER_LANE` PEs, reduces their results in fabric, and optionally
accumulates multiple chunks in fabric. A lane transaction consumes
`3*PE_PER_LANE` signed INT8 activation values and the same number of weights.

The primitive ports and PE output are signed 24-bit, matching the verified RTL.
Mathematically an INT8 product is 16 bits and three products need at most 18
bits using the usual conservative growth rule. Keeping 24 bits preserves the
trusted boundary. The lane uses `24+ceil(log2(PE_PER_LANE))` bits. The caller
sets `ACC_WIDTH`; for at most `C` chunks, use at least
`lane_width+ceil(log2(C))` bits. There is no safe finite default for an
unbounded stream, so the default 32 bits must be validated for the application.

PE contract: signed INT8 inputs, signed 24-bit output, latency 1, II 1, one
DSP58 with `DSP_MODE=INT8`. Lane target: II 1 and exactly `PE_PER_LANE` DSP58s.
The `bind_op ... impl=fabric` directives request fabric implementation for the
reduction/accumulator. HLS directives are not a proof of the final mapped
netlist, so the generated reports and netlist remain authoritative.

Accumulation controls are sampled with each transaction. `acc_valid` qualifies
the chunk; `acc_clear` clears before including the current chunk; and `acc_last`
causes `result_valid` to assert for the final accumulated value. With
`ENABLE_ACCUM=0`, clear/last are ignored and valid simply qualifies the current
lane sum.

The synthesis top is `hls/dot3_lane_top.cpp`. Select 1, 2, 4, or 8 PEs (and the
accumulation variant) with the macros in `hls/hls_config.cfg`, using a separate
work directory for every run. `tests/tb_dot3_lane_hls.cpp` directly exercises
the synthesizable lane for all four sizes and both accumulation modes during
HLS C simulation.

The black-box result uses JSON `c_return`, not an output-reference parameter.
Vitis HLS 2025.1 scheduled the output-reference form at II=2 because it created
a false distance-1 dependence between the next black-box call and the load of
the reused local output scalar. The return-value form represents the result as
a value, removes that false dependence, and gives the non-accumulating lane
II=1 without changing `rtl/dsp58_dot3_int8.v`.

## Legacy streamed linear baseline

`single_lane_linear.hpp` remains as an isolated baseline for one token and one contiguous
output-channel tile. It does not alter the existing multi-block decode graph.

## Dataflow and stream order

`INT8 activation stream -> group by 3 -> RTL dot3 -> HLS ap_int<32> accumulator -> output stream`

The activation stream contains `in_dim` values once. The lane buffers them as
the existing FlexLLM decode linear layer buffers its activation vector. Weights
are scalar and channel-major: all `in_dim` weights for output channel 0, then
all weights for channel 1, and so on. Outputs are emitted in contiguous channel
order. Tail groups read only valid weights and insert zeros locally.

FlexLLM's current decode loader instead emits reduction-major vectors of
`DEC_QKVO_FFN_W_PARALLEL` INT4 weights. A later composition step should add a
small layout/precision adapter or change the loader; this baseline intentionally
does neither while the single lane is being validated.

## Host-only test

```sh
make -C single_lane test
```

This runs the standard-C++ reference and the isolated RTL arithmetic/handshake
with Synopsys VCS. It does not require TAPA, Vitis, Vivado, or AMD headers:

```sh
make -C single_lane cpp-test
make -C single_lane rtl-test VCS=/path/to/vcs
```

`VCS=/path/to/vcs` can be omitted when `vcs` is already in `PATH`. An optional
Icarus fallback remains available as `make -C single_lane rtl-test-iverilog`.
These behavioral RTL tests do not prove DSP58 mapping.

## Vitis HLS blackbox

The JSON follows AMD's `syn.blackbox.file` flow. From `single_lane/hls`, replace
the placeholder part in `hls_config.cfg` with the exact installed XCV80 part and
run (command spelling depends on the installed Vitis release):

```sh
vitis-run --mode hls --config hls_config.cfg --work_dir build
```

The supplied RTL explicitly instantiates DSP58 using the packing and control
configuration from the previously validated `dsp58_dot3_acc` experiment. Its
external accumulator was deliberately removed: the DSP58 C input is tied to
zero, and the long accumulator remains in HLS. The standalone VCS test defines
`FLEXLLM_DSP58_BEHAVIORAL_SIM`, so it does not require a UNISIM DSP58 model.
Vitis/Vivado builds must leave this macro undefined. The blackbox uses ap_ctrl_none: ap_ce advances the fixed-latency, II=1 dot3 pipeline, while the JSON declares latency 1 and requires no per-call control handshake.

Check after synthesis/implementation:

1. Blackbox replacement and RTL co-simulation complete successfully.
2. The dot3 hierarchy contains exactly one DSP58 and `DSP_MODE=INT8`.
3. The HLS accumulator is outside the dot3 hierarchy and does not add a second
   DSP58 to one dot3 operation.
4. C/RTL co-simulation matches `tests/test_single_lane.cpp`, including tails.
5. Record achieved II/latency; the JSON declares PE latency 1 and II 1, which
   must match the final registered RTL.
6. For PE_PER_LANE=1,2,4,8, record LUT, FF, BRAM, URAM, and DSP utilization;
   verify DSP totals are 1,2,4,8 and inspect every primitive's DSP_MODE property.
7. Run synthesis and implementation timing for the exact V80 device/clock.

In Vivado, count the actual primitives with
`get_cells -hier -filter {REF_NAME == DSP58}` and inspect each returned cell with
`get_property DSP_MODE <cell>`. Also check the HLS synthesis report for II and
latency and the post-synthesis utilization report for LUT/FF/DSP counts. If the
DSP count exceeds `PE_PER_LANE`, first confirm the two `bind_op` directives were
accepted (not warned as ignored), then apply equivalent `set_directive_bind_op`
commands to the reduction and accumulator variables in the HLS Tcl flow.

The primitive configuration is based on the user's prior test, but its mapping
inside this blackbox, blackbox behavior in the locally installed Vitis version,
timing, and utilization cannot be re-verified here without AMD tools.
The JSON resource fields are scheduling metadata (DSP=1 is the design intent,
and the other fields are left at zero), not measured utilization results.
