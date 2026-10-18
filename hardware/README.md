# Single-neuron hardware prototype

`hdl/lif_neuron_fxp.v` is the existing Q3.12 single-neuron Verilog prototype. Its ports are `clk`, active-low `rst_n`, 16-bit `i_current` (interpreted using `$signed` in the arithmetic), `i_valid`, `o_spike`, `o_membrane` and `o_ready`. Parameters specify total/fractional width, threshold and beta (0.9; Q3.12 integer 3686). The embedded testbench is a clock/reset skeleton; 12/16-bit software-generated vectors are in `hdl_test_vectors/`.

This contribution is an unverified single-neuron prototype with software-generated reference vectors.

| Parameter | Default | Interpretation |
|---|---:|---|
| INTEGER_BITS | 3 | Magnitude integer bits |
| FRACTIONAL_BITS | 12 | Fractional bits |
| TOTAL_BITS | 16 | Sign plus integer and fractional bits |
| MAX_VAL | 32767 | Positive Q3.12 storage bound |
| MIN_VAL | -32768 | Negative Q3.12 storage bound |
| THRESHOLD | 4096 | Threshold 1.0 in Q3.12 |
| BETA | 3686 | Quantised membrane decay 0.9 |

The prototype uses IDLE, COMPUTE and DONE states. `i_valid` initiates processing; `o_ready` marks the computed output. Input current is not latched, so it must remain stable through COMPUTE.
