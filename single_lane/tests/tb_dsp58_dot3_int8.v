`timescale 1ns/1ps

module tb_dsp58_dot3_int8;
  reg ap_clk = 0;
  reg ap_rst = 1;
  reg ap_ce = 1;
  reg signed [7:0] a0, a1, a2, w0, w1, w2;
  wire signed [23:0] dot3;

  dsp58_dot3_int8 dut(.*);
  always #5 ap_clk = ~ap_clk;

  task check;
    input signed [7:0] ta0, ta1, ta2, tw0, tw1, tw2;
    input signed [23:0] expected;
    begin
      @(negedge ap_clk);
      a0 = ta0; a1 = ta1; a2 = ta2;
      w0 = tw0; w1 = tw1; w2 = tw2;
      @(negedge ap_clk);
      if (dot3 !== expected) begin
        $display("FAIL: got %0d expected %0d", dot3, expected);
        $fatal(1);
      end
    end
  endtask

  initial begin
    a0 = 0; a1 = 0; a2 = 0; w0 = 0; w1 = 0; w2 = 0;
    repeat (2) @(negedge ap_clk);
    ap_rst = 0;
    check(0, 0, 0, 0, 0, 0, 0);
    check(1, 2, 3, 4, 5, 6, 32);
    check(-1, -2, -3, 4, 5, 6, -32);
    check(-3, 4, -5, -6, -7, 8, -50);
    check(-128, 127, -128, -128, 127, 127, 16257);
    $display("All dot3 RTL arithmetic tests passed.");
    $finish;
  end
endmodule
