`timescale 1ns/1ps

// RTL-blackbox boundary for one signed INT8 three-element dot product.
// There is deliberately no long accumulator in this module: DSP58 C is tied
// to zero and the lane-level accumulation remains in HLS.
module dsp58_dot3_int8 (
    input  wire               ap_clk,
    input  wire               ap_rst,
    input  wire               ap_ce,
    input  wire               ap_start,
    input  wire               ap_continue,
    input  wire signed [7:0]  a0,
    input  wire signed [7:0]  a1,
    input  wire signed [7:0]  a2,
    input  wire signed [7:0]  w0,
    input  wire signed [7:0]  w1,
    input  wire signed [7:0]  w2,
    output wire               ap_idle,
    output wire               ap_done,
    output wire               ap_ready,
    output wire               dot3_ap_vld,
    output reg  signed [23:0] dot3
);
  wire [33:0] dsp_a = {
      7'b1111111,
      a2[7], a2,
      a1[7], a1,
      a0[7], a0
  };
  wire [23:0] dsp_b = {w2, w1, w0};
  wire [57:0] dsp_p;

`ifdef FLEXLLM_DSP58_BEHAVIORAL_SIM
  // Open-source simulator model only. Vitis/Vivado builds must not define this
  // macro; they use the explicit primitive below.
  wire signed [15:0] product0 = $signed(a0) * $signed(w0);
  wire signed [15:0] product1 = $signed(a1) * $signed(w1);
  wire signed [15:0] product2 = $signed(a2) * $signed(w2);
  wire signed [23:0] dot3_sim =
      {{8{product0[15]}}, product0} +
      {{8{product1[15]}}, product1} +
      {{8{product2[15]}}, product2};
  assign dsp_p = {{34{dot3_sim[23]}}, dot3_sim};
`else
  // Configuration and packing adapted from the user's previously validated
  // dsp58_dot3_acc module. Its C/accumulator input is intentionally removed.
  DSP58 #(
      .DSP_MODE("INT8"),
      .AMULTSEL("A"),
      .BMULTSEL("B"),
      .A_INPUT("DIRECT"),
      .B_INPUT("DIRECT"),
      .PREADDINSEL("A"),
      .USE_MULT("MULTIPLY"),
      .USE_SIMD("ONE58"),
      .USE_WIDEXOR("FALSE"),
      .AREG(0),
      .ACASCREG(0),
      .BREG(0),
      .BCASCREG(0),
      .CREG(0),
      .DREG(0),
      .ADREG(0),
      .MREG(0),
      .PREG(0),
      .ALUMODEREG(0),
      .INMODEREG(0),
      .OPMODEREG(0),
      .CARRYINREG(0),
      .CARRYINSELREG(0),
      .RESET_MODE("SYNC")
  ) u_dsp58 (
      .A(dsp_a),
      .B(dsp_b),
      .C(58'b0),
      .D(27'b0),
      .P(dsp_p),
      // P = C + INT8 vector multiplier result; C is zero here.
      .OPMODE(9'b000110101),
      .ALUMODE(4'b0000),
      .INMODE(5'b00000),
      .NEGATE(3'b000),
      .CARRYIN(1'b0),
      .CARRYINSEL(3'b000),
      .ACIN(34'b0),
      .BCIN(24'b0),
      .PCIN(58'b0),
      .CARRYCASCIN(1'b0),
      .MULTSIGNIN(1'b0),
      // All internal DSP registers are disabled. The wrapper below registers P.
      .CLK(1'b0),
      .CEA1(1'b0),
      .CEA2(1'b0),
      .CEB1(1'b0),
      .CEB2(1'b0),
      .CEC(1'b0),
      .CED(1'b0),
      .CEAD(1'b0),
      .CEM(1'b0),
      .CEP(1'b0),
      .CEALUMODE(1'b0),
      .CECTRL(1'b0),
      .CEINMODE(1'b0),
      .CECARRYIN(1'b0),
      .ASYNC_RST(1'b0),
      .RSTA(1'b0),
      .RSTB(1'b0),
      .RSTC(1'b0),
      .RSTD(1'b0),
      .RSTM(1'b0),
      .RSTP(1'b0),
      .RSTALUMODE(1'b0),
      .RSTCTRL(1'b0),
      .RSTINMODE(1'b0),
      .RSTALLCARRYIN(1'b0),
      .ACOUT(),
      .BCOUT(),
      .PCOUT(),
      .CARRYCASCOUT(),
      .MULTSIGNOUT(),
      .CARRYOUT(),
      .XOROUT(),
      .OVERFLOW(),
      .UNDERFLOW(),
      .PATTERNDETECT(),
      .PATTERNBDETECT()
  );
`endif

  reg valid_d1;
  always @(posedge ap_clk) begin
    if (ap_rst) begin
      dot3    <= 24'sd0;
      valid_d1 <= 1'b0;
    end else if (ap_ce) begin
      dot3    <= dsp_p[23:0];
      valid_d1 <= ap_start;
    end
  end

  assign dot3_ap_vld = valid_d1;
  assign ap_done     = valid_d1;
  assign ap_ready    = valid_d1;
  assign ap_idle     = ~ap_start;

  // ap_continue is part of ap_ctrl_chain and is intentionally unused for this
  // fixed-latency, II=1 leaf block.
  wire unused_ap_continue = ap_continue;
endmodule
