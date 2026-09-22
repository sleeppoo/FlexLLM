# FlexLLM INT8 single linear lane

This directory is an isolated baseline for one token and one contiguous
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
5. Record achieved II/latency; the JSON declares RTL latency 1 and II 1, which
   must match the final registered RTL.
6. Record LUT, FF, BRAM, URAM, and DSP utilization.
7. Run synthesis and implementation timing for the exact V80 device/clock.

The primitive configuration is based on the user's prior test, but its mapping
inside this blackbox, blackbox behavior in the locally installed Vitis version,
timing, and utilization cannot be re-verified here without AMD tools.
The JSON resource fields are scheduling metadata (DSP=1 is the design intent,
and the other fields are left at zero), not measured utilization results.
