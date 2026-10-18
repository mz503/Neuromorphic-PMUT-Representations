"""
Fixed-point LIF neuron and quantization utilities for hardware-aware modelling.

Supports Q-format fixed-point arithmetic with configurable bit widths.
Used by NB13 (float-vs-fixed comparison), NB14 (quantized pipeline),
NB15 (HDL test vectors), and NB16 (HDL specification).

Q-format: Qm.n = m integer bits + n fractional bits + 1 sign bit.
Total bits = m + n + 1 (signed). Range: [-2^m, 2^m - 2^-n].
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Q-format utilities
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QFormat:
    """Fixed-point Q-format descriptor."""
    integer_bits: int   # m
    fractional_bits: int  # n
    signed: bool = True

    @property
    def total_bits(self) -> int:
        return self.integer_bits + self.fractional_bits + (1 if self.signed else 0)

    @property
    def scale(self) -> float:
        return 2.0 ** self.fractional_bits

    @property
    def min_val(self) -> float:
        if self.signed:
            return -(2.0 ** self.integer_bits)
        return 0.0

    @property
    def max_val(self) -> float:
        if self.signed:
            return (2.0 ** self.integer_bits) - (2.0 ** -self.fractional_bits)
        return (2.0 ** (self.integer_bits + 1)) - (2.0 ** -self.fractional_bits)

    @property
    def epsilon(self) -> float:
        return 2.0 ** -self.fractional_bits

    def __repr__(self) -> str:
        sign = "s" if self.signed else "u"
        return f"Q{sign}{self.integer_bits}.{self.fractional_bits} ({self.total_bits}-bit)"


# Standard formats for bit-width study
STANDARD_FORMATS = {
    "16-bit": QFormat(integer_bits=3, fractional_bits=12),   # Q3.12, range [-8, ~8)
    "12-bit": QFormat(integer_bits=3, fractional_bits=8),    # Q3.8,  range [-8, ~8)
    "8-bit":  QFormat(integer_bits=3, fractional_bits=4),    # Q3.4,  range [-8, ~8)
    "4-bit":  QFormat(integer_bits=1, fractional_bits=2),    # Q1.2,  range [-2, ~2)
}

# Wider internal format for accumulation (prevents overflow in dot products)
ACCUM_FORMAT = QFormat(integer_bits=7, fractional_bits=16)  # Q7.16, 24-bit


def float_to_fixed(value: float | np.ndarray, qfmt: QFormat) -> int | np.ndarray:
    """Convert floating-point to fixed-point integer representation."""
    scaled = value * qfmt.scale
    clamped = np.clip(scaled, qfmt.min_val * qfmt.scale, qfmt.max_val * qfmt.scale)
    return np.round(clamped).astype(np.int64)


def fixed_to_float(fixed_val: int | np.ndarray, qfmt: QFormat) -> float | np.ndarray:
    """Convert fixed-point integer back to floating-point."""
    return fixed_val.astype(np.float64) / qfmt.scale


def quantize(value: float | np.ndarray, qfmt: QFormat) -> float | np.ndarray:
    """Quantize a float to fixed-point and back — simulates quantization error."""
    return fixed_to_float(float_to_fixed(value, qfmt), qfmt)


def quantize_array(arr: np.ndarray, qfmt: QFormat) -> np.ndarray:
    """Quantize a numpy array to fixed-point representation."""
    return quantize(arr, qfmt)


# ---------------------------------------------------------------------------
# Fixed-point LIF neuron
# ---------------------------------------------------------------------------

class FixedPointLIF:
    """LIF neuron using fixed-point integer arithmetic throughout.

    This mirrors the floating-point LIFStateLayer from spiking.py but
    operates entirely in integer domain — suitable for HDL translation.

    Update rule (all in Q-format integers):
      I_q   = clip(W_in·x + W_res·s + bias)
      v_q   = clip(beta * v_q + I_q)
      s     = (v_q >= theta_q) ? 1 : 0
      v_q   = v_q - s * theta_q   [subtractive reset]

    Parameters
    ----------
    qfmt : QFormat
        Primary Q-format for membrane, threshold, and states.
    beta : float
        Membrane decay (0 < beta < 1). Quantized internally.
    threshold : float
        Firing threshold.
    n_input : int
        Number of input channels.
    n_neurons : int
        Number of LIF neurons.
    seed : int
        Random seed for weight initialization.
    """

    def __init__(
        self,
        qfmt: QFormat,
        beta: float = 0.9,
        threshold: float = 1.0,
        n_input: int = 4,
        n_neurons: int = 300,
        seed: int = 42,
    ):
        self.qfmt = qfmt
        self.acc_qfmt = ACCUM_FORMAT  # Wider format for dot products
        self.beta_float = beta
        self.threshold_float = threshold
        self.n_input = n_input
        self.n_neurons = n_neurons

        # Quantize constants
        self.beta_q = float_to_fixed(beta, qfmt)
        self.threshold_q = float_to_fixed(threshold, qfmt)

        # Initialize weights (random, quantized)
        rng = np.random.RandomState(seed)
        self.W_in = float_to_fixed(
            rng.uniform(-1.0, 1.0, (n_neurons, n_input)).astype(np.float64), qfmt
        )
        self.bias = float_to_fixed(
            rng.uniform(0.0, 0.5, n_neurons).astype(np.float64), qfmt
        )

        # State
        self._v_q: np.ndarray | None = None  # membrane in Q-format

    def reset(self) -> None:
        self._v_q = None

    def _dot_product(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute dot product in accumulator format, then requantize."""
        # a, b are in qfmt → product in qfmt*2 precision
        # Use accumulator format for intermediate, then truncate back
        result_float = fixed_to_float(a, self.qfmt) @ fixed_to_float(b, self.qfmt).T
        return float_to_fixed(result_float, self.qfmt)

    def step(self, input_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single LIF time step in fixed-point.

        Parameters
        ----------
        input_current : ndarray of shape (batch, n_input) in float

        Returns
        -------
        spikes : ndarray of shape (batch, n_neurons) — binary (0 or 1)
        membrane : ndarray of shape (batch, n_neurons) — quantized float
        """
        batch = input_current.shape[0]
        if self._v_q is None:
            self._v_q = np.zeros((batch, self.n_neurons), dtype=np.int64)

        # Quantize input
        input_q = float_to_fixed(input_current, self.qfmt)

        # Synaptic current: I = W_in @ x + bias (no recurrence for simplicity)
        I_q = np.zeros((batch, self.n_neurons), dtype=np.int64)
        for i in range(batch):
            for j in range(self.n_neurons):
                dot = np.sum(self.W_in[j].astype(np.int64) * input_q[i].astype(np.int64))
                # Scale down: product of two Q-format numbers needs rescaling
                dot_scaled = dot // (2 ** self.qfmt.fractional_bits)
                I_q[i, j] = np.clip(
                    dot_scaled + self.bias[j],
                    self.qfmt.min_val * self.qfmt.scale,
                    self.qfmt.max_val * self.qfmt.scale,
                ).astype(np.int64)

        # Leak + integrate: v = beta * v_prev + I
        v_q_new = np.zeros_like(self._v_q)
        for i in range(batch):
            for j in range(self.n_neurons):
                # beta * v (beta is quantized)
                leak_term = (self.beta_q * self._v_q[i, j]) // (2 ** self.qfmt.fractional_bits)
                v_q_new[i, j] = np.clip(
                    leak_term + I_q[i, j],
                    self.qfmt.min_val * self.qfmt.scale,
                    self.qfmt.max_val * self.qfmt.scale,
                ).astype(np.int64)

        # Spike: s = (v >= theta)
        spikes_q = (v_q_new >= self.threshold_q).astype(np.int64)

        # Reset: v = v - s * theta
        for i in range(batch):
            for j in range(self.n_neurons):
                if spikes_q[i, j]:
                    v_q_new[i, j] = np.clip(
                        v_q_new[i, j] - self.threshold_q,
                        self.qfmt.min_val * self.qfmt.scale,
                        self.qfmt.max_val * self.qfmt.scale,
                    ).astype(np.int64)

        self._v_q = v_q_new

        # Convert back to float for comparison/readout
        spikes_float = spikes_q.astype(np.float64)
        v_float = fixed_to_float(v_q_new, self.qfmt)

        return spikes_float, v_float


# ---------------------------------------------------------------------------
# Floating-point LIF neuron (reference gold model)
# ---------------------------------------------------------------------------

class FloatLIF:
    """Floating-point LIF neuron — reference model for comparison."""

    def __init__(
        self,
        beta: float = 0.9,
        threshold: float = 1.0,
        n_input: int = 4,
        n_neurons: int = 300,
        seed: int = 42,
    ):
        self.beta = beta
        self.threshold = threshold
        self.n_input = n_input
        self.n_neurons = n_neurons

        rng = np.random.RandomState(seed)
        self.W_in = rng.uniform(-1.0, 1.0, (n_neurons, n_input)).astype(np.float64)
        self.bias = rng.uniform(0.0, 0.5, n_neurons).astype(np.float64)

        self._v: np.ndarray | None = None

    def reset(self) -> None:
        self._v = None

    def step(self, input_current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single LIF time step in floating-point."""
        batch = input_current.shape[0]
        if self._v is None:
            self._v = np.zeros((batch, self.n_neurons), dtype=np.float64)

        # Synaptic current
        I = input_current @ self.W_in.T + self.bias

        # Leak + integrate
        self._v = self.beta * self._v + I

        # Spike
        spikes = (self._v >= self.threshold).astype(np.float64)

        # Reset
        self._v = self._v - spikes * self.threshold

        return spikes, self._v.copy()


# ---------------------------------------------------------------------------
# Comparison utilities
# ---------------------------------------------------------------------------

def compare_lif_neurons(
    float_lif: FloatLIF,
    fixed_lif: FixedPointLIF,
    input_signal: np.ndarray,
    n_steps: int = 200,
) -> dict:
    """Run both float and fixed LIF neurons on the same input, compare outputs.

    Parameters
    ----------
    float_lif, fixed_lif : FloatLIF, FixedPointLIF
        Initialized neuron instances.
    input_signal : ndarray of shape (n_steps, n_input)
        Input current over time.
    n_steps : int
        Number of time steps to simulate.

    Returns
    -------
    dict with keys:
        'float_v' : (n_steps, n_neurons) float membrane traces
        'fixed_v' : (n_steps, n_neurons) fixed-point membrane traces
        'float_spikes' : (n_steps, n_neurons) float spike raster
        'fixed_spikes' : (n_steps, n_neurons) fixed-point spike raster
        'v_mae' : float — mean absolute error in membrane voltage
        'v_rmse' : float — root mean square error in membrane voltage
        'spike_mismatch_rate' : float — fraction of time steps where spikes differ
        'spike_timing_error' : float — mean absolute timing difference (in steps)
    """
    float_lif.reset()
    fixed_lif.reset()

    float_v = np.zeros((n_steps, float_lif.n_neurons), dtype=np.float64)
    fixed_v = np.zeros((n_steps, fixed_lif.n_neurons), dtype=np.float64)
    float_s = np.zeros((n_steps, float_lif.n_neurons), dtype=np.float64)
    fixed_s = np.zeros((n_steps, fixed_lif.n_neurons), dtype=np.float64)

    for t in range(n_steps):
        u = input_signal[t : t + 1, :]
        fs, fv = float_lif.step(u)
        qs, qv = fixed_lif.step(u)

        float_v[t] = fv[0]
        fixed_v[t] = qv[0]
        float_s[t] = fs[0]
        fixed_s[t] = qs[0]

    # Voltage error
    v_diff = np.abs(float_v - fixed_v)
    v_mae = float(v_diff.mean())
    v_rmse = float(np.sqrt((v_diff ** 2).mean()))

    # Spike mismatch
    s_mismatch = (float_s != fixed_s).astype(float)
    mismatch_rate = float(s_mismatch.mean())

    # Spike timing error (per neuron, average over neurons)
    timing_errors = []
    for j in range(fixed_lif.n_neurons):
        float_times = np.where(float_s[:, j] > 0.5)[0]
        fixed_times = np.where(fixed_s[:, j] > 0.5)[0]
        if len(float_times) == 0 and len(fixed_times) == 0:
            continue
        # Pad shorter list
        min_len = min(len(float_times), len(fixed_times))
        if min_len > 0:
            timing_errors.append(np.mean(np.abs(float_times[:min_len] - fixed_times[:min_len])))
    mean_timing_error = float(np.mean(timing_errors)) if timing_errors else 0.0

    return {
        "float_v": float_v,
        "fixed_v": fixed_v,
        "float_spikes": float_s,
        "fixed_spikes": fixed_s,
        "v_mae": v_mae,
        "v_rmse": v_rmse,
        "spike_mismatch_rate": mismatch_rate,
        "spike_timing_error": mean_timing_error,
    }


def bit_width_sensitivity(
    input_signal: np.ndarray,
    n_steps: int = 200,
    n_neurons: int = 100,
    beta: float = 0.9,
    threshold: float = 1.0,
    n_input: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Run LIF neuron at multiple bit widths, measure error vs float reference.

    Returns list of dicts with bit_width, qfmt_name, v_mae, v_rmse,
    spike_mismatch_rate, spike_timing_error.
    """
    results = []
    float_lif = FloatLIF(beta=beta, threshold=threshold, n_input=n_input, n_neurons=n_neurons, seed=seed)

    for name, qfmt in STANDARD_FORMATS.items():
        fixed_lif = FixedPointLIF(qfmt=qfmt, beta=beta, threshold=threshold, n_input=n_input, n_neurons=n_neurons, seed=seed)
        comp = compare_lif_neurons(float_lif, fixed_lif, input_signal, n_steps=n_steps)

        results.append({
            "bit_width": qfmt.total_bits,
            "qfmt_name": name,
            "qfmt": repr(qfmt),
            "v_mae": comp["v_mae"],
            "v_rmse": comp["v_rmse"],
            "spike_mismatch_rate": comp["spike_mismatch_rate"],
            "spike_timing_error": comp["spike_timing_error"],
        })

    return results


# ---------------------------------------------------------------------------
# HDL test-vector generation
# ---------------------------------------------------------------------------

def generate_hdl_test_vectors(
    input_spikes: np.ndarray,        # (n_steps, n_input) binary
    beta: float = 0.9,
    threshold: float = 1.0,
    qfmt: QFormat | None = None,
    W_in: np.ndarray | None = None,  # (n_neurons, n_input)
    bias: np.ndarray | None = None,  # (n_neurons,)
    n_neurons: int = 10,
    seed: int = 42,
) -> dict:
    """Generate HDL-ready test vectors from PMUT-derived spike trains.

    Uses a small LIF neuron (n_neurons=10) for tractable HDL verification.

    Parameters
    ----------
    input_spikes : ndarray of shape (n_steps, n_input)
        PMUT-derived binary spike train.
    beta : float
        Membrane decay.
    threshold : float
        Firing threshold.
    qfmt : QFormat
        Fixed-point format for output values.
    W_in, bias : ndarray
        Input weights and bias. Generated randomly if None.
    n_neurons : int
        Number of LIF neurons to simulate.
    seed : int
        Random seed.

    Returns
    -------
    dict with:
        'input_spikes' : CSV-ready integer array
        'expected_voltage' : CSV-ready Q-format integer array
        'expected_spikes' : CSV-ready binary array
        'params' : dict with threshold_q, beta_q, W_in_q, bias_q, qfmt info
    """
    if qfmt is None:
        qfmt = QFormat(integer_bits=3, fractional_bits=12)  # default Q3.12

    n_steps, n_input = input_spikes.shape
    rng = np.random.RandomState(seed)

    if W_in is None:
        W_in = rng.uniform(-1.0, 1.0, (n_neurons, n_input)).astype(np.float64)
    if bias is None:
        bias = rng.uniform(0.0, 0.5, n_neurons).astype(np.float64)

    # Quantize parameters
    W_in_q = float_to_fixed(W_in, qfmt)
    bias_q = float_to_fixed(bias, qfmt)
    beta_q = float_to_fixed(beta, qfmt)
    threshold_q = float_to_fixed(threshold, qfmt)

    # Simulate fixed-point LIF
    v_q = np.zeros(n_neurons, dtype=np.int64)
    expected_v = np.zeros((n_steps, n_neurons), dtype=np.int64)
    expected_s = np.zeros((n_steps, n_neurons), dtype=np.int64)

    for t in range(n_steps):
        # Current: I = W_in @ s_in[t] + bias
        # Binary spike = gate — NO rescaling. W_in_q is ALREADY Q-format.
        # A spike of 1 delivers the full Q-format weight directly.
        # (Accumulation detail: the old dot // 2**frac_bits
        #  attenuated the input contribution by 4096× for Q3.12.)
        s_in_q = input_spikes[t].astype(np.int64)
        I_q = bias_q.copy()
        for j in range(n_neurons):
            dot = np.int64(0)
            for i in range(n_input):
                if s_in_q[i]:
                    dot += W_in_q[j, i]  # direct Q-format addition
            I_q[j] = np.clip(
                I_q[j] + dot,
                qfmt.min_val * qfmt.scale // 1,
                qfmt.max_val * qfmt.scale // 1,
            ).astype(np.int64)

        # Leak + integrate
        for j in range(n_neurons):
            leak = (beta_q * v_q[j]) // (2 ** qfmt.fractional_bits)
            v_q[j] = np.clip(
                leak + I_q[j],
                qfmt.min_val * qfmt.scale // 1,
                qfmt.max_val * qfmt.scale // 1,
            ).astype(np.int64)

        # Spike
        spike = (v_q >= threshold_q).astype(np.int64)
        expected_s[t] = spike

        # Reset
        v_q = v_q - spike * threshold_q
        v_q = np.clip(v_q, qfmt.min_val * qfmt.scale // 1, qfmt.max_val * qfmt.scale // 1)

        expected_v[t] = v_q.copy()

    return {
        "n_steps": n_steps,
        "n_input": n_input,
        "n_neurons": n_neurons,
        "qfmt_name": repr(qfmt),
        "qfmt_total_bits": qfmt.total_bits,
        "qfmt_integer_bits": qfmt.integer_bits,
        "qfmt_fractional_bits": qfmt.fractional_bits,
        "qfmt_scale": int(qfmt.scale),
        "threshold_float": threshold,
        "threshold_q": int(threshold_q),
        "beta_float": beta,
        "beta_q": int(beta_q),
        "W_in_q": W_in_q.tolist(),
        "bias_q": bias_q.tolist(),
        "input_spikes": input_spikes.astype(np.int8).tolist(),
        "expected_voltage_q": expected_v.tolist(),
        "expected_spikes": expected_s.astype(np.int8).tolist(),
    }


def save_hdl_test_vectors(test_data: dict, output_dir: Path) -> None:
    """Save HDL test vectors to CSV and JSON files.

    Creates:
      pmut_input_spikes.csv    — input spike train
      expected_voltage_q.csv   — expected membrane voltage (Q-format ints)
      expected_spikes.csv      — expected spike output (binary)
      params.json              — all parameters and Q-format info
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_steps = test_data["n_steps"]
    n_input = test_data["n_input"]
    n_neurons = test_data["n_neurons"]

    # Input spikes
    with open(output_dir / "pmut_input_spikes.csv", "w") as f:
        f.write(",".join([f"in_{i}" for i in range(n_input)]) + "\n")
        for t in range(n_steps):
            f.write(",".join(str(test_data["input_spikes"][t][i]) for i in range(n_input)) + "\n")

    # Expected voltage (Q-format integers)
    with open(output_dir / "expected_voltage_q.csv", "w") as f:
        f.write(",".join([f"neuron_{j}" for j in range(n_neurons)]) + "\n")
        for t in range(n_steps):
            f.write(",".join(str(test_data["expected_voltage_q"][t][j]) for j in range(n_neurons)) + "\n")

    # Expected spikes
    with open(output_dir / "expected_spikes.csv", "w") as f:
        f.write(",".join([f"neuron_{j}" for j in range(n_neurons)]) + "\n")
        for t in range(n_steps):
            f.write(",".join(str(test_data["expected_spikes"][t][j]) for j in range(n_neurons)) + "\n")

    # Parameters (without large arrays for readability)
    params_summary = {
        "n_steps": test_data["n_steps"],
        "n_input": test_data["n_input"],
        "n_neurons": test_data["n_neurons"],
        "qfmt": test_data["qfmt_name"],
        "total_bits": test_data["qfmt_total_bits"],
        "integer_bits": test_data["qfmt_integer_bits"],
        "fractional_bits": test_data["qfmt_fractional_bits"],
        "scale": test_data["qfmt_scale"],
        "threshold_float": test_data["threshold_float"],
        "threshold_q": test_data["threshold_q"],
        "beta_float": test_data["beta_float"],
        "beta_q": test_data["beta_q"],
        "W_in_q_shape": [n_neurons, n_input],
        "bias_q_shape": [n_neurons],
    }
    with open(output_dir / "params.json", "w") as f:
        json.dump(params_summary, f, indent=2)

    # Full data (for Python verification)
    with open(output_dir / "test_vectors_full.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"  HDL test vectors saved → {output_dir}")
    print(f"    pmut_input_spikes.csv   : {n_steps} steps × {n_input} inputs")
    print(f"    expected_voltage_q.csv  : {n_steps} steps × {n_neurons} neurons (Q-format)")
    print(f"    expected_spikes.csv     : {n_steps} steps × {n_neurons} neurons")
    print(f"    params.json             : Q-format config + neuron params")


# ---------------------------------------------------------------------------
# IntegerLSM — true fixed-point LSM with quantized readout
# ---------------------------------------------------------------------------

class IntegerLSM:
    """Liquid State Machine using INTEGER arithmetic throughout.

    Unlike the FixedPointLIF (single non-recurrent neuron) and NB14's
    float-roundtrip quantisation, this model operates ENTIRELY in integer
    arithmetic — no float casts in the forward pass. This is the model that
    maps 1:1 to Verilog HDL.

    Update rule (all Q-format integers):
      I_q   = clip(W_in_q @ x_q + W_res_q @ s_q + bias_q)
      v_q   = clip((beta_q * v_q) >> FRAC + I_q)
      s_q   = (v_q >= threshold_q)
      v_q   = v_q - s_q * threshold_q    [subtractive reset]

    Readout (quantized):
      score = W_out_q @ features_q + bias_out_q    [integer dot product]
      pred  = argmax(score)

    Hardware-relevant specifications:
      - Multiplier width: 2 × TOTAL_BITS (product of two Q-format values)
      - Accumulator width: 2 × TOTAL_BITS + ceil(log2(n_input)) for dot product
      - Rescaling: arithmetic right-shift by FRACTIONAL_BITS after multiply
      - Rounding mode: truncation (floor toward negative infinity for signed)
      - Overflow behaviour: saturation to [MIN_VAL, MAX_VAL] on every operation
      - Reset behaviour: subtractive (v_q -= threshold_q on spike), clamped ≥ MIN_VAL
      - Saturation vs wraparound: SATURATION — values clip to range, never wrap

    Parameters
    ----------
    qfmt : QFormat for weights and membrane.
    n_input, n_reservoir, n_classes : dimensions.
    beta, threshold : float — quantized on init.
    input_scaling, spectral_radius, sparsity : reservoir params.
    seed : int.
    """

    def __init__(
        self,
        qfmt: QFormat,
        n_input: int = 4,
        n_reservoir: int = 300,
        n_classes: int = 5,
        beta: float = 0.9,
        threshold: float = 1.0,
        spectral_radius: float = 0.95,
        sparsity: float = 0.9,
        input_scaling: float = 1.0,
        exc_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.qfmt = qfmt
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_classes = n_classes
        self.frac = qfmt.fractional_bits
        self.scale = int(qfmt.scale)

        # Quantize constants
        self.beta_q = int(float_to_fixed(beta, qfmt))
        self.threshold_q = int(float_to_fixed(threshold, qfmt))

        rng = np.random.RandomState(seed)

        # Input weights
        W_in_f = rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_input)).astype(np.float64)
        self.W_in_q = float_to_fixed(W_in_f, qfmt).astype(np.int64)

        # Recurrent weights with Dale's law
        W_res_f = rng.randn(n_reservoir, n_reservoir).astype(np.float64)
        mask = rng.rand(n_reservoir, n_reservoir) < (1.0 - sparsity)
        W_res_f[mask] = 0.0
        neuron_type = np.ones(n_reservoir, dtype=np.float64)
        n_exc = int(n_reservoir * exc_ratio)
        neuron_type[n_exc:] = -1.0
        rng.shuffle(neuron_type)
        W_res_f = W_res_f * neuron_type[np.newaxis, :]
        eig_max = max(abs(np.linalg.eigvals(W_res_f)))
        if eig_max > 1e-10:
            W_res_f *= spectral_radius / eig_max
        self.W_res_q = float_to_fixed(W_res_f, qfmt).astype(np.int64)

        # Bias
        bias_f = rng.uniform(0.0, 0.3, n_reservoir).astype(np.float64)
        self.bias_q = float_to_fixed(bias_f, qfmt).astype(np.int64)

        # Readout (trained later, quantized after fitting)
        self.W_out_q: np.ndarray | None = None
        self.bias_out_q: np.ndarray | None = None

        # State
        self._v_q: np.ndarray | None = None
        self._s_q: np.ndarray | None = None

    def reset(self) -> None:
        self._v_q = np.zeros(self.n_reservoir, dtype=np.int64)
        self._s_q = np.zeros(self.n_reservoir, dtype=np.int64)

    def process_sequence(self, x: np.ndarray, warmup: int = 10) -> dict:
        """Process a continuous-valued sweep through the integer LSM.

        Parameters
        ----------
        x : ndarray of shape (seq_len, n_input) in FLOAT (will be quantized per step)

        Returns
        -------
        dict with 'membrane_q' and 'spikes_q' (integer arrays, shape (seq_len-warmup, n_reservoir))
        """
        if x.ndim == 1:
            x = x.reshape(-1, self.n_input)
        seq_len = x.shape[0]
        self.reset()

        membranes = np.zeros((seq_len, self.n_reservoir), dtype=np.int64)
        spikes_all = np.zeros((seq_len, self.n_reservoir), dtype=np.int64)

        for t in range(seq_len):
            # Quantize input
            x_q = float_to_fixed(x[t], self.qfmt).astype(np.int64)

            # Synaptic current: I = W_in @ x + W_res @ s + bias
            I_q = self.bias_q.copy()
            # Input contribution
            for j in range(self.n_reservoir):
                dot_in = 0
                for i in range(self.n_input):
                    dot_in += int(self.W_in_q[j, i]) * int(x_q[i])
                I_q[j] += dot_in >> self.frac

            # Recurrent contribution
            for j in range(self.n_reservoir):
                dot_res = 0
                for k in range(self.n_reservoir):
                    if self._s_q[k]:
                        dot_res += int(self.W_res_q[j, k])
                I_q[j] += dot_res

            # Clamp current
            I_q = np.clip(I_q, self.qfmt.min_val * self.scale, self.qfmt.max_val * self.scale)

            # Membrane update: v = (beta * v) >> frac + I
            # Use Python int for multiplication to avoid overflow, then shift
            v_new = np.zeros(self.n_reservoir, dtype=np.int64)
            for j in range(self.n_reservoir):
                leak = int(self.beta_q) * int(self._v_q[j]) >> self.frac
                v_new[j] = leak + int(I_q[j])
            v_new = np.clip(v_new, self.qfmt.min_val * self.scale, self.qfmt.max_val * self.scale)

            # Spike
            s_new = (v_new >= self.threshold_q).astype(np.int64)

            # Reset
            v_new = v_new - s_new * self.threshold_q
            v_new = np.clip(v_new, self.qfmt.min_val * self.scale, self.qfmt.max_val * self.scale)

            self._v_q = v_new
            self._s_q = s_new
            membranes[t] = v_new
            spikes_all[t] = s_new

        if warmup > 0:
            membranes = membranes[warmup:]
            spikes_all = spikes_all[warmup:]

        return {"membrane_q": membranes, "spikes_q": spikes_all}

    def collect_features(
        self, x: np.ndarray, warmup: int = 10,
        state_mode: str = "membrane", state_reduction: str = "mean",
    ) -> np.ndarray:
        """Run all samples through LSM, reduce to integer feature vectors.

        Parameters
        ----------
        x : ndarray of shape (n_samples, seq_len, n_input) in FLOAT
        state_mode : 'membrane' or 'spikes'
        state_reduction : 'mean' or 'last'

        Returns
        -------
        features : ndarray of shape (n_samples, n_reservoir) in Q-format int64
        """
        n_samples = x.shape[0]
        feat_dim = self.n_reservoir * (2 if state_mode == "both" else 1)
        features = np.zeros((n_samples, feat_dim), dtype=np.int64)

        for i in range(n_samples):
            result = self.process_sequence(x[i], warmup=warmup)
            if state_mode == "membrane":
                states = result["membrane_q"]
            elif state_mode == "spikes":
                states = result["spikes_q"]
            else:  # both
                states = np.concatenate([result["membrane_q"], result["spikes_q"]], axis=1)

            if state_reduction == "mean":
                feat = states.mean(axis=0).round().astype(np.int64)
            else:  # last
                feat = states[-1].astype(np.int64)
            features[i] = feat

        return features

    def fit_readout(
        self, x_train: np.ndarray, y_train: np.ndarray,
        warmup: int = 10, state_mode: str = "membrane",
        state_reduction: str = "mean", alpha: float = 1.0,
    ) -> float:
        """Fit a FLOAT Ridge readout, then QUANTIZE weights to Q-format.

        This is the honest procedure:
          1. Fit Ridge in float (to find optimal weights)
          2. Quantize the weights for integer inference
          3. Report BOTH float-readout and quantized-readout accuracy

        Returns float-readout training accuracy. Quantized readout accuracy
        is available via .quantized_readout_accuracy.
        """
        from sklearn.linear_model import RidgeClassifier

        features_q = self.collect_features(x_train, warmup=warmup, state_mode=state_mode, state_reduction=state_reduction)
        # Convert to float for sklearn fitting
        features_f = fixed_to_float(features_q, self.qfmt)

        self.readout = RidgeClassifier(alpha=alpha)
        self.readout.fit(features_f, y_train)

        # ── Quantize the readout ──
        W_out_f = self.readout.coef_.astype(np.float64)  # (n_classes, n_features)
        bias_out_f = self.readout.intercept_.astype(np.float64)  # (n_classes,)

        self.W_out_q = float_to_fixed(W_out_f, self.qfmt).astype(np.int64)
        self.bias_out_q = float_to_fixed(bias_out_f, self.qfmt).astype(np.int64)

        # Compute quantized-readout accuracy on training data
        self.quantized_readout_accuracy = float(self._score_quantized(features_q, y_train))

        train_acc = float(self.readout.score(features_f, y_train))
        print(f"  IntegerLSM fit: float_train={train_acc:.4f}  quantized_train={self.quantized_readout_accuracy:.4f}")
        return train_acc

    def _score_quantized(self, features_q: np.ndarray, y: np.ndarray) -> float:
        """Score using PURE INTEGER dot-product readout — no float casts."""
        correct = 0
        for i in range(len(features_q)):
            scores = np.zeros(self.n_classes, dtype=np.int64)
            for c in range(self.n_classes):
                dot = 0
                for j in range(len(features_q[i])):
                    dot += int(self.W_out_q[c, j]) * int(features_q[i, j])
                scores[c] = (dot >> self.frac) + int(self.bias_out_q[c])
            pred = int(np.argmax(scores))
            if pred == y[i]:
                correct += 1
        return correct / len(features_q)

    def predict(self, x: np.ndarray, warmup: int = 10,
                state_mode: str = "membrane", state_reduction: str = "mean",
                quantized: bool = True) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        quantized : if True, use INTEGER readout; if False, use float readout
        """
        features_q = self.collect_features(x, warmup=warmup, state_mode=state_mode, state_reduction=state_reduction)

        if quantized and self.W_out_q is not None:
            preds = []
            for i in range(len(features_q)):
                scores = np.zeros(self.n_classes, dtype=np.int64)
                for c in range(self.n_classes):
                    dot = 0
                    for j in range(len(features_q[i])):
                        dot += int(self.W_out_q[c, j]) * int(features_q[i, j])
                    scores[c] = (dot >> self.frac) + int(self.bias_out_q[c])
                preds.append(int(np.argmax(scores)))
            return np.array(preds, dtype=np.int64)
        else:
            features_f = fixed_to_float(features_q, self.qfmt)
            return self.readout.predict(features_f)
