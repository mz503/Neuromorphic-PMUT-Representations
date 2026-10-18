"""
Echo State Network (ESN) reservoir computing model for PMUT sweep classification.

Each frequency sweep is treated as a sequential ordered signal: the reservoir transforms
the sweep point-by-point (low to high frequency) into a high-dimensional state
frequency-point vectors into a high-dimensional state trajectory, and a linear
readout (trained via ridge regression) performs classification.

Supports both tanh and LIF-style reservoir neurons.
"""

import numpy as np
from sklearn.linear_model import RidgeClassifier


class Reservoir:
    """Echo State Network reservoir with linear readout.

    Parameters
    ----------
    n_input : int
        Number of input channels per time step.
    n_reservoir : int
        Number of reservoir neurons.
    spectral_radius : float
        Scaling factor for the recurrent weight matrix (should be < 1 for ESP).
    sparsity : float
        Fraction of non-zero entries in the recurrent weight matrix.
    leaking_rate : float
        Leaking rate a in h(t) = (1-a)*h(t-1) + a*tanh(W_in*x(t) + W_res*h(t-1)).
    input_scaling : float
        Scaling factor for input weights.
    noise : float
        Standard deviation of Gaussian noise added to reservoir state.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_input: int,
        n_reservoir: int = 300,
        spectral_radius: float = 0.95,
        sparsity: float = 0.9,
        leaking_rate: float = 0.3,
        input_scaling: float = 1.0,
        noise: float = 0.0,
        seed: int = 42,
    ):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.noise = noise
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Input weights: uniform in [-input_scaling, +input_scaling]
        self.W_in = rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_input))

        # Sparse recurrent weights
        W_res_dense = rng.randn(n_reservoir, n_reservoir)
        mask = rng.rand(n_reservoir, n_reservoir) < (1.0 - sparsity)
        W_res_dense[mask] = 0.0

        # Scale to desired spectral radius
        rho = max(abs(np.linalg.eigvals(W_res_dense)))
        if rho > 1e-10:
            W_res_dense *= spectral_radius / rho

        self.W_res = W_res_dense.astype(np.float64)

        # Bias
        self.bias = rng.uniform(-0.5, 0.5, n_reservoir)

        # Readout (trained later)
        self.W_out: np.ndarray | None = None
        self.readout: RidgeClassifier | None = None

        # Warm-up state
        self._state: np.ndarray | None = None

    def reset(self) -> None:
        """Reset internal state."""
        self._state = None

    def _update(self, u: np.ndarray) -> np.ndarray:
        """Single time-step update. u shape: (batch, n_input) or (n_input,)."""
        if u.ndim == 1:
            u = u[np.newaxis, :]
        batch = u.shape[0]

        if self._state is None:
            self._state = np.zeros((batch, self.n_reservoir), dtype=np.float64)

        pre_activation = u @ self.W_in.T + self._state @ self.W_res.T + self.bias
        activated = np.tanh(pre_activation)

        noise_term = 0.0
        if self.noise > 0:
            noise_term = np.random.randn(batch, self.n_reservoir) * self.noise

        self._state = (1.0 - self.leaking_rate) * self._state + self.leaking_rate * activated + noise_term
        return self._state.copy()

    def process_sequence(self, x: np.ndarray, warmup: int = 0) -> np.ndarray:
        """Process a full sequence through the reservoir.

        Parameters
        ----------
        x : ndarray of shape (batch, seq_len, n_input) or (seq_len, n_input)
            Input sequence.
        warmup : int
            Number of initial steps to discard (washout).

        Returns
        -------
        states : ndarray of shape (batch, seq_len - warmup, n_reservoir)
            Reservoir states at each time step after warmup.
        """
        single = x.ndim == 2
        if single:
            x = x[np.newaxis, :, :]
        batch, seq_len, _ = x.shape

        self.reset()
        all_states = np.zeros((batch, seq_len, self.n_reservoir), dtype=np.float64)

        for t in range(seq_len):
            all_states[:, t, :] = self._update(x[:, t, :])

        if warmup > 0:
            all_states = all_states[:, warmup:, :]

        if single:
            return all_states[0]
        return all_states

    def fit_readout(
        self,
        x: np.ndarray,
        y: np.ndarray,
        warmup: int = 10,
        state_reduction: str = "mean",
        alpha: float = 1.0,
    ) -> float:
        """Collect reservoir states and train linear readout.

        Parameters
        ----------
        x : ndarray of shape (n_samples, seq_len, n_input)
            Training input sequences.
        y : ndarray of shape (n_samples,) with integer labels.
        warmup : int
            Washout steps to discard per sample.
        state_reduction : str
            How to reduce sequential states: 'mean' (average over sequence),
            'last' (final state), or 'concat_ends' (concatenate start/mid/end).
        alpha : float
            Ridge regularization strength.

        Returns
        -------
        train_accuracy : float
            Training accuracy of the fitted readout.
        """
        features = self._collect_features(x, warmup=warmup, state_reduction=state_reduction)

        self.readout = RidgeClassifier(alpha=alpha)
        self.readout.fit(features, y)

        train_acc = float(self.readout.score(features, y))
        return train_acc

    def predict(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Predict class labels for input sequences.

        Parameters
        ----------
        x : ndarray of shape (n_samples, seq_len, n_input)
        warmup : int
        state_reduction : str

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
        """
        if self.readout is None:
            raise RuntimeError("Readout not fitted. Call fit_readout() first.")

        features = self._collect_features(x, warmup=warmup, state_reduction=state_reduction)
        return self.readout.predict(features)

    def predict_proba(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Predict class probabilities via decision function.

        Uses the distance to decision boundary as a confidence proxy.
        """
        if self.readout is None:
            raise RuntimeError("Readout not fitted.")
        features = self._collect_features(x, warmup=warmup, state_reduction=state_reduction)
        decision = self.readout.decision_function(features)
        # Normalize via softmax
        decision -= decision.max(axis=1, keepdims=True)
        exp_d = np.exp(decision)
        return exp_d / exp_d.sum(axis=1, keepdims=True)

    def _collect_features(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Run all samples through reservoir and reduce to feature vectors."""
        n_samples = x.shape[0]
        features = []

        for i in range(n_samples):
            states = self.process_sequence(x[i], warmup=warmup)  # (seq_len, n_reservoir)

            if state_reduction == "mean":
                feat = states.mean(axis=0)
            elif state_reduction == "last":
                feat = states[-1]
            elif state_reduction == "concat_ends":
                seq = states.shape[0]
                start = states[: seq // 4].mean(axis=0)
                mid = states[seq // 4 : 3 * seq // 4].mean(axis=0)
                end = states[3 * seq // 4 :].mean(axis=0)
                feat = np.concatenate([start, mid, end])
            else:
                raise ValueError(f"Unknown state_reduction: {state_reduction}")
            features.append(feat)

        return np.stack(features)

    def get_state_trajectory(
        self,
        x: np.ndarray,
        warmup: int = 10,
    ) -> np.ndarray:
        """Get full reservoir state trajectory for visualization.

        Parameters
        ----------
        x : ndarray of shape (seq_len, n_input) — single sample.

        Returns
        -------
        states : ndarray of shape (seq_len - warmup, n_reservoir)
        """
        return self.process_sequence(x, warmup=warmup)


# ---------------------------------------------------------------------------
# Liquid State Machine (LSM) — spiking reservoir with LIF neurons
# ---------------------------------------------------------------------------

class LiquidStateMachine:
    """Liquid State Machine: spiking reservoir with LIF neurons.

    Unlike the ESN (which uses tanh rate-coded neurons), the LSM uses
    spiking LIF neurons. Recurrent connections carry spike events, not
    continuous activations. This is the spiking analog of an ESN.

    Key differences from ESN:
      - ESN:  h(t) = (1-a)·h(t-1) + a·tanh(W_in·x + W_res·h(t-1))
      - LSM:  I(t) = W_in·x(t) + W_res·s(t-1) + bias
              v(t) = β·v(t-1) + I(t)
              s(t) = H(v(t) - θ)
              v(t) ← v(t) - s(t)·θ   (subtractive reset)

    Parameters
    ----------
    n_input : int
        Number of input channels per time step.
    n_reservoir : int
        Number of LIF neurons in the liquid.
    beta : float
        Membrane decay factor (0 < beta < 1). Higher = longer memory.
    threshold : float
        Firing threshold.
    spectral_radius : float
        Scaling for recurrent weight matrix (controls chaos/order).
    sparsity : float
        Fraction of zero entries in recurrent weights.
    input_scaling : float
        Scaling factor for input weights.
    exc_ratio : float
        Fraction of excitatory neurons (remainder are inhibitory).
        Dale's law: W_res sign depends on neuron type.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        n_input: int,
        n_reservoir: int = 300,
        beta: float = 0.9,
        threshold: float = 1.0,
        spectral_radius: float = 0.95,
        sparsity: float = 0.9,
        input_scaling: float = 1.0,
        exc_ratio: float = 0.8,
        seed: int = 42,
    ):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.beta = beta
        self.threshold = threshold
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        self.exc_ratio = exc_ratio
        self.seed = seed

        rng = np.random.RandomState(seed)

        # Input weights: uniform in [-input_scaling, +input_scaling]
        self.W_in = rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_input))

        # Recurrent weights with Dale's law: excitatory (positive) vs inhibitory (negative)
        W_res_dense = rng.randn(n_reservoir, n_reservoir)
        mask = rng.rand(n_reservoir, n_reservoir) < (1.0 - sparsity)
        W_res_dense[mask] = 0.0

        # Apply Dale's law: excitatory neurons have positive outgoing weights,
        # inhibitory neurons have negative outgoing weights
        neuron_type = np.ones(n_reservoir)
        n_exc = int(n_reservoir * exc_ratio)
        neuron_type[n_exc:] = -1.0  # inhibitory
        rng.shuffle(neuron_type)

        # Each column j is the outgoing weights from neuron j
        W_res_dense = W_res_dense * neuron_type[np.newaxis, :]

        # Scale to spectral radius
        eig_max = max(abs(np.linalg.eigvals(W_res_dense)))
        if eig_max > 1e-10:
            W_res_dense *= spectral_radius / eig_max

        self.W_res = W_res_dense.astype(np.float64)
        self.neuron_type = neuron_type

        # Bias current
        self.bias = rng.uniform(0.0, 0.3, n_reservoir)

        # Readout
        self.readout: RidgeClassifier | None = None

        # State variables (reset per sequence)
        self._membrane: np.ndarray | None = None
        self._spikes_prev: np.ndarray | None = None

    def reset(self) -> None:
        """Reset internal LIF state."""
        self._membrane = None
        self._spikes_prev = None

    def _lif_update(self, current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single LIF time step. Returns (spikes, membrane)."""
        batch = current.shape[0]

        if self._membrane is None:
            self._membrane = np.zeros((batch, self.n_reservoir), dtype=np.float64)
        if self._spikes_prev is None:
            self._spikes_prev = np.zeros((batch, self.n_reservoir), dtype=np.float64)

        # Membrane update: v(t) = beta * v(t-1) + I(t)
        self._membrane = self.beta * self._membrane + current

        # Spike: s(t) = H(v(t) - threshold)
        spikes = (self._membrane >= self.threshold).astype(np.float64)

        # Subtract-and-reset
        self._membrane = self._membrane - spikes * self.threshold

        # Clamp membrane to avoid negative runaway
        self._membrane = np.maximum(self._membrane, -self.threshold)

        self._spikes_prev = spikes
        return spikes, self._membrane.copy()

    def _update(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Single time-step update. u shape: (batch, n_input) or (n_input,).

        Returns (spikes, membrane) for this time step.
        """
        if u.ndim == 1:
            u = u[np.newaxis, :]
        batch = u.shape[0]

        if self._spikes_prev is None:
            self._spikes_prev = np.zeros((batch, self.n_reservoir), dtype=np.float64)

        # Current = input drive + recurrent spike input + bias
        current = u @ self.W_in.T + self._spikes_prev @ self.W_res.T + self.bias

        return self._lif_update(current)

    def process_sequence(self, x: np.ndarray, warmup: int = 0) -> dict:
        """Process a full sequence through the LSM.

        Parameters
        ----------
        x : ndarray of shape (seq_len, n_input) — single sample.
        warmup : int
            Number of initial steps to discard.

        Returns
        -------
        dict with keys:
            'membrane' : (seq_len - warmup, n_reservoir) membrane potentials
            'spikes'   : (seq_len - warmup, n_reservoir) binary spike trains
        """
        single = x.ndim == 2
        if single:
            x = x[np.newaxis, :, :]
        batch, seq_len, _ = x.shape

        self.reset()
        membranes = np.zeros((batch, seq_len, self.n_reservoir), dtype=np.float64)
        spikes_all = np.zeros((batch, seq_len, self.n_reservoir), dtype=np.float64)

        for t in range(seq_len):
            spikes, membrane = self._update(x[:, t, :])
            membranes[:, t, :] = membrane
            spikes_all[:, t, :] = spikes

        if warmup > 0:
            membranes = membranes[:, warmup:, :]
            spikes_all = spikes_all[:, warmup:, :]

        if single:
            return {"membrane": membranes[0], "spikes": spikes_all[0]}
        return {"membrane": membranes, "spikes": spikes_all}

    def fit_readout(
        self,
        x: np.ndarray,
        y: np.ndarray,
        warmup: int = 10,
        state_mode: str = "membrane",
        state_reduction: str = "mean",
        alpha: float = 1.0,
    ) -> float:
        """Collect LSM states and train linear readout.

        Parameters
        ----------
        x, y : ndarray
            Training data and labels.
        warmup : int
            Washout steps.
        state_mode : str
            'membrane' — use membrane potentials (continuous, like ESN)
            'spikes'   — use spike trains (binary, hardware-compatible)
            'both'     — concatenate membrane and spike features
        state_reduction : str
            'mean', 'last', or 'concat_ends'.
        alpha : float
            Ridge regularization.

        Returns
        -------
        train_accuracy : float
        """
        features = self._collect_features(x, warmup=warmup, state_mode=state_mode, state_reduction=state_reduction)
        self.readout = RidgeClassifier(alpha=alpha)
        self.readout.fit(features, y)
        return float(self.readout.score(features, y))

    def predict(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_mode: str = "membrane",
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Predict class labels."""
        if self.readout is None:
            raise RuntimeError("Readout not fitted.")
        features = self._collect_features(x, warmup=warmup, state_mode=state_mode, state_reduction=state_reduction)
        return self.readout.predict(features)

    def predict_proba(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_mode: str = "membrane",
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Predict class probabilities via decision_function + softmax.

        Returns ndarray of shape (n_samples, n_classes).
        """
        if self.readout is None:
            raise RuntimeError("Readout not fitted.")
        features = self._collect_features(x, warmup=warmup, state_mode=state_mode, state_reduction=state_reduction)
        decision = self.readout.decision_function(features)
        decision -= decision.max(axis=1, keepdims=True)
        exp_d = np.exp(decision)
        return exp_d / exp_d.sum(axis=1, keepdims=True)

    def _collect_features(
        self,
        x: np.ndarray,
        warmup: int = 10,
        state_mode: str = "membrane",
        state_reduction: str = "mean",
    ) -> np.ndarray:
        """Run all samples through LSM and reduce to feature vectors."""
        n_samples = x.shape[0]
        features = []

        for i in range(n_samples):
            result = self.process_sequence(x[i], warmup=warmup)

            if state_mode == "membrane":
                states = result["membrane"]
            elif state_mode == "spikes":
                states = result["spikes"]
            elif state_mode == "both":
                states = np.concatenate([result["membrane"], result["spikes"]], axis=1)
            else:
                raise ValueError(f"Unknown state_mode: {state_mode}")

            if state_reduction == "mean":
                feat = states.mean(axis=0)
            elif state_reduction == "last":
                feat = states[-1]
            elif state_reduction == "concat_ends":
                seq = states.shape[0]
                start = states[: seq // 4].mean(axis=0)
                mid = states[seq // 4 : 3 * seq // 4].mean(axis=0)
                end = states[3 * seq // 4 :].mean(axis=0)
                feat = np.concatenate([start, mid, end])
            else:
                raise ValueError(f"Unknown state_reduction: {state_reduction}")
            features.append(feat)

        return np.stack(features)

    def get_state_trajectory(self, x: np.ndarray, warmup: int = 10) -> dict:
        """Get full state trajectory for visualization."""
        return self.process_sequence(x, warmup=warmup)

    def get_spike_stats(self, x: np.ndarray, warmup: int = 10) -> dict:
        """Compute spike statistics for a dataset.

        Returns
        -------
        dict with mean_spike_count, spike_rate, fraction_silent.
        """
        result = self.process_sequence(x, warmup=warmup)
        spikes = result["spikes"]
        n_steps = spikes.shape[0]
        n_neurons = spikes.shape[1]
        total_spikes = spikes.sum()
        return {
            "mean_spike_count": float(total_spikes),
            "spike_rate": float(total_spikes / (n_steps * n_neurons)),
            "fraction_silent": float((spikes.sum(axis=0) == 0).mean()),
            "mean_firing_rate": float(spikes.mean()),
        }


def estimate_memory_capacity(
    n_reservoir: int = 200,
    n_input: int = 1,
    seq_len: int = 1000,
    max_delay: int = 50,
    seed: int = 42,
) -> dict:
    """Estimate memory capacity of the reservoir.

    Feeds a random input signal and trains a readout for each delay k.
    MC(k) = correlation^2 between delayed input and readout.
    Total MC = sum(MC(k)) for k=1..max_delay.

    Returns
    -------
    dict with keys: 'mc_per_delay' (list), 'total_mc' (float), 'max_delay' (int)
    """
    rng = np.random.RandomState(seed)
    u = rng.uniform(-0.8, 0.8, seq_len).astype(np.float64)

    reservoir = Reservoir(
        n_input=n_input,
        n_reservoir=n_reservoir,
        spectral_radius=0.95,
        sparsity=0.9,
        leaking_rate=0.3,
        input_scaling=1.0,
        seed=seed,
    )

    # Drive reservoir
    u_batch = u.reshape(-1, 1)
    states = reservoir.process_sequence(u_batch, warmup=100)  # (seq_len-100, n_reservoir)

    effective_len = states.shape[0]
    mc_per_delay = []

    for k in range(1, max_delay + 1):
        # Target: delayed input (clipped to valid range)
        target = u[100 + k : 100 + effective_len]  # align with states after warmup + delay
        state_k = states[: effective_len - k]

        min_len = min(len(target), len(state_k))
        target = target[:min_len]
        state_k = state_k[:min_len]

        if min_len < 10:
            mc_per_delay.append(0.0)
            continue

        # Ridge regression readout
        ridge = RidgeClassifier(alpha=1.0)
        # Treat as regression via Ridge
        from sklearn.linear_model import Ridge

        reg = Ridge(alpha=1.0)
        reg.fit(state_k, target)
        pred = reg.predict(state_k)
        corr = np.corrcoef(target, pred)[0, 1]
        mc_per_delay.append(float(max(0.0, corr**2)))

    total_mc = sum(mc_per_delay)
    return {"mc_per_delay": mc_per_delay, "total_mc": total_mc, "max_delay": max_delay}


def reservoir_grid_search(
    x_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    warmup: int = 10,
    state_reduction: str = "mean",
    n_input: int = 4,
    val_frac: float = 0.2,
    seed: int = 42,
) -> list[dict]:
    """Grid search over reservoir hyperparameters using internal train/val split.

    The training data is split 80/20 (stratified) into train/val. Each HP
    configuration is fitted on the train portion and evaluated on the val
    portion. The test set is NEVER seen during hyperparameter selection.

    Parameters
    ----------
    x_train, y_train : ndarray
        Training data (will be split internally into train/val).
    param_grid : dict
        Keys: 'n_reservoir', 'spectral_radius', 'leaking_rate', 'input_scaling', 'sparsity'.
        Values are lists of parameter values to try.
    val_frac : float
        Fraction of training data to use as validation (default 0.2).
    seed : int
        Random seed for the train/val split.

    Returns
    -------
    results : list of dicts with keys: params, train_acc, val_acc.
        Sorted by val_acc descending.
    """
    from itertools import product

    from sklearn.model_selection import train_test_split

    # ── Split training data into train/val (stratified) ──
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train, test_size=val_frac, random_state=seed, stratify=y_train,
    )

    results = []
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]

    for values in product(*param_values):
        params = dict(zip(param_names, values))
        try:
            reservoir = Reservoir(
                n_input=n_input,
                n_reservoir=params.get("n_reservoir", 300),
                spectral_radius=params.get("spectral_radius", 0.95),
                sparsity=params.get("sparsity", 0.9),
                leaking_rate=params.get("leaking_rate", 0.3),
                input_scaling=params.get("input_scaling", 1.0),
                seed=seed,
            )
            train_acc = reservoir.fit_readout(
                x_tr, y_tr, warmup=warmup, state_reduction=state_reduction,
            )
            preds = reservoir.predict(x_val, warmup=warmup, state_reduction=state_reduction)
            val_acc = float((preds == y_val).mean())

            results.append({"params": params, "train_acc": train_acc, "val_acc": val_acc})
        except Exception as e:
            results.append({"params": params, "train_acc": None, "val_acc": None, "error": str(e)})

    # Sort by val accuracy descending (test set never seen)
    results.sort(key=lambda r: r.get("val_acc") or -1.0, reverse=True)
    return results
