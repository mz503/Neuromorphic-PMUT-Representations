"""
Physical feature extraction: PMUT Feature Extraction
=====================================
Extract physically interpretable features from each PMUT frequency sweep.

Tasks:
  - Detect resonance peaks (4-band hybrid multi-modal detection)
  - Estimate resonance frequency (f0) for each mode using LOCAL-band analysis
  - Estimate peak amplitude at each resonance (detrended within band)
  - Estimate phase transition location and slope
  - Calculate bandwidth (-3dB) and Q-factor using LOCAL-band crossings
    with linear interpolation — NO full-sweep defaults
  - Fallback features when no clean resonance exists in a band
  - Full-band spectral shape features (centroid, variance, skewness, kurtosis)
  - I/Q trajectory features
  - Diagnostic plots showing band boundaries, peaks, and half-power crossings

Run:  python scripts/prepare_data.py
Saves: data/processed/features/{task}_feature_table.csv
       data/processed/features/{task}_feature_distributions.png
       data/processed/features/{task}_feature_correlation.png
       data/processed/features/{task}_diagnostic_bands.png  (per-sweep band plots)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.pmut_models.config import DEFAULT_DATA_ROOT
from src.pmut_models.data import load_task_frame, _label_sort_key

OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "processed" / "features"

# ---------------------------------------------------------------------------
# PMUT mode band definitions (physically justified frequency ranges)
# ---------------------------------------------------------------------------

PMUT_MODE_BANDS = [
    {"name": "b1", "label": "0.2 MHz",  "range_hz": (120_000, 380_000),  "method": "phase"},
    {"name": "b2", "label": "0.45 MHz", "range_hz": (380_000, 700_000),  "method": "amp"},
    {"name": "b3", "label": "0.9 MHz",  "range_hz": (700_000, 1_100_000), "method": "amp"},
    {"name": "b4", "label": "1.2 MHz",  "range_hz": (1_100_000, 2_450_000), "method": "amp"},
]


# ---------------------------------------------------------------------------
# Local-band analysis: bandwidth and Q extraction
# ---------------------------------------------------------------------------

def _local_band_indices(freq: np.ndarray, lo_hz: float, hi_hz: float) -> tuple[int, int]:
    """Return (start, end) slice indices for the local frequency band."""
    start = int(np.searchsorted(freq, lo_hz, side="left"))
    end = int(np.searchsorted(freq, hi_hz, side="right"))
    return max(0, start), min(len(freq), end)


def _interp_crossing(
    freq: np.ndarray, amp: np.ndarray, level: float, start: int, end: int, direction: str,
) -> float | None:
    """Find the frequency where amplitude crosses `level` using linear interpolation.

    Searches within [start, end) in the specified direction.
    Returns interpolated frequency in Hz, or None if no crossing found.
    """
    if direction == "left":
        idx_range = range(end - 1, start, -1)
    else:  # "right"
        idx_range = range(start, end - 1)

    for k in idx_range:
        if direction == "left":
            a0, a1 = amp[k + 1], amp[k]
            f0, f1 = freq[k + 1], freq[k]
        else:
            a0, a1 = amp[k], amp[k + 1]
            f0, f1 = freq[k], freq[k + 1]

        if (a0 - level) * (a1 - level) <= 0:
            if abs(a1 - a0) < 1e-20:
                return float((f0 + f1) / 2)
            t = (level - a0) / (a1 - a0)
            return float(f0 + t * (f1 - f0))

    return None


def _analyse_local_band(
    freq: np.ndarray, amp: np.ndarray, lo_hz: float, hi_hz: float,
    phase: np.ndarray | None = None, method: str = "amp",
) -> dict:
    """Analyse a single PMUT mode band for resonance, bandwidth, and Q.

    This method uses local frequency-band crossings:
      - Local baseline estimated and REMOVED within the band
      - Peak detected on detrended local signal using prominence
      - Half-power computed on detrended signal (NOT raw amplitude)
      - Crossing search RESTRICTED to the local band
      - Linear interpolation at crossing points
      - NaN sentinel when no clean resonance exists (no silent defaults)

    Returns a dict with keys: f0_hz, q, bw_hz, amp_at_peak, peak_found,
    band_energy, band_centroid_hz, local_slope, phase_gradient, prominence.
    Q and bw are NaN when no valid half-power crossings exist.
    """
    ilo, ihi = _local_band_indices(freq, lo_hz, hi_hz)
    if ihi - ilo < 5:
        return _no_resonance_result()

    freq_band = freq[ilo:ihi]
    amp_band = amp[ilo:ihi]

    # 1. Estimate and remove local baseline (linear fit to band edges)
    edge_x = np.array([freq_band[0], freq_band[-1]])
    edge_y = np.array([amp_band[0], amp_band[-1]])
    if edge_x[1] - edge_x[0] > 0:
        slope = (edge_y[1] - edge_y[0]) / (edge_x[1] - edge_x[0])
        intercept = edge_y[0] - slope * edge_x[0]
        local_baseline = slope * freq_band + intercept
    else:
        local_baseline = np.full_like(freq_band, edge_y[0])

    amp_local = amp_band - local_baseline  # detrended within band

    # 2. Detect peak on detrended local signal
    local_range = float(np.ptp(amp_local))
    if local_range < 1e-15:
        return _no_resonance_result()

    min_prominence = max(local_range * 0.1, 1e-15)
    min_distance = max(len(freq_band) // 8, 3)
    peaks, props = find_peaks(amp_local, prominence=min_prominence, distance=min_distance)

    if len(peaks) == 0:
        # Fallback: use phase-derivative peak if available
        if phase is not None and method == "phase":
            return _phase_band_analysis(freq_band, amp_band, amp_local, phase[ilo:ihi], ilo)
        return _no_resonance_result()

    # Select best peak by prominence
    prominences = props.get("prominences", np.ones(len(peaks)))
    best_j = int(np.argmax(prominences))
    p_local = peaks[best_j]
    p_global = ilo + p_local
    prominence = float(prominences[best_j])
    peak_amp_local = float(amp_local[p_local])
    amp_at_peak = float(amp_band[p_local])

    # 3. Half-power on DETRENDED signal (relative to local baseline)
    hp_level = peak_amp_local / np.sqrt(2)

    # 4. Search for crossings WITHIN the local band only
    f_left = _interp_crossing(freq_band, amp_local, hp_level, 0, p_local, "left")
    f_right = _interp_crossing(freq_band, amp_local, hp_level, p_local, len(freq_band) - 1, "right")

    if f_left is not None and f_right is not None and f_right > f_left:
        bw = f_right - f_left
        q = freq[p_global] / bw
        # ── Physical sanity bounds for PMUT in liquid ──
        # Q < 5: bandwidth exceeds 20% of f0 — no meaningful resonance
        # Q > 500: implausibly narrow for MEMS in liquid (requires vacuum)
        # Prominence check: peak must be at least 20% of local range
        if q < 5.0 or q > 500.0:
            bw = np.nan
            q = np.nan
        elif prominence < local_range * 0.20:
            # Peak not prominent enough relative to band variation
            bw = np.nan
            q = np.nan
    else:
        bw = np.nan
        q = np.nan

    # 5. Fallback features (computed for ALL bands, not just those without Q)
    band_energy = float(np.trapezoid(amp_band, freq_band))
    if len(freq_band) > 1:
        centroid = float(np.sum(freq_band * amp_band) / np.sum(amp_band)) if np.sum(amp_band) > 0 else np.nan
    else:
        centroid = np.nan

    # Local amplitude slope (linear fit to band)
    if len(freq_band) > 2:
        coeffs = np.polyfit(freq_band, amp_band, 1)
        local_slope = float(coeffs[0])
    else:
        local_slope = np.nan

    # Phase gradient magnitude at band centre
    if phase is not None and len(freq_band) > 3:
        phase_band = phase[ilo:ihi]
        dphase = np.abs(np.gradient(phase_band, freq_band))
        mid = len(freq_band) // 2
        phase_gradient = float(np.mean(dphase[max(0, mid - 3): min(len(dphase), mid + 4)]))
    else:
        phase_gradient = np.nan

    return {
        "f0_hz": float(freq[p_global]),
        "q": q,
        "bw_hz": bw,
        "amp_at_peak": amp_at_peak,
        "peak_found": True,
        "band_energy": band_energy,
        "band_centroid_hz": centroid,
        "local_slope": local_slope,
        "phase_gradient": phase_gradient,
        "prominence": prominence,
    }


def _no_resonance_result() -> dict:
    """Return a result dict with NaN sentinels for bands with no clean resonance."""
    return {
        "f0_hz": np.nan, "q": np.nan, "bw_hz": np.nan,
        "amp_at_peak": np.nan, "peak_found": False,
        "band_energy": np.nan, "band_centroid_hz": np.nan,
        "local_slope": np.nan, "phase_gradient": np.nan,
        "prominence": np.nan,
    }


def _phase_band_analysis(
    freq_band: np.ndarray, amp_band: np.ndarray, amp_local: np.ndarray,
    phase_band: np.ndarray, offset: int,
) -> dict:
    """Fallback: detect resonance from phase derivative when amplitude peak fails."""
    if len(phase_band) < 5:
        return _no_resonance_result()

    dphase = np.abs(np.gradient(phase_band, freq_band))
    # Exclude edges
    edge = max(len(dphase) // 20, 1)
    dphase[:edge] = 0
    dphase[-edge:] = 0

    std_dp = float(np.std(dphase))
    if std_dp < 1e-20:
        return _no_resonance_result()

    peaks_ph, props_ph = find_peaks(dphase, prominence=std_dp * 0.3, distance=max(len(dphase) // 6, 3))
    if len(peaks_ph) == 0:
        return _no_resonance_result()

    prominences = props_ph.get("prominences", np.ones(len(peaks_ph)))
    best_j = int(np.argmax(prominences))
    p_local = peaks_ph[best_j]

    result = _no_resonance_result()
    result["f0_hz"] = float(freq_band[p_local])
    result["amp_at_peak"] = float(amp_band[p_local])
    result["peak_found"] = True
    result["prominence"] = float(prominences[best_j])
    result["phase_gradient"] = float(dphase[p_local])
    result["band_energy"] = float(np.trapezoid(amp_band, freq_band))
    # Q and bw remain NaN — phase-derivative peaks don't yield meaningful bandwidth
    return result


# ---------------------------------------------------------------------------
# Full-band spectral shape features
# ---------------------------------------------------------------------------

def _compute_spectral_shape_features(
    freq: np.ndarray, amp: np.ndarray,
) -> dict:
    """Compute spectral shape features from the full-band amplitude envelope.

    These are physically meaningful even when individual resonances are not
    well-defined (as in overdamped PMUT sensors).

    Returns: centroid_hz, variance_hz2, skewness, kurtosis, rolloff_hz, slope_low, slope_high
    """
    n = len(freq)
    if n < 4:
        return {}

    # Normalise amplitude as a pseudo-probability distribution
    amp_pos = np.maximum(amp, 0)
    total = amp_pos.sum()
    if total < 1e-20:
        return {}

    prob = amp_pos / total

    # Spectral centroid (mean frequency)
    centroid = float(np.sum(freq * prob))

    # Spectral variance
    variance = float(np.sum(((freq - centroid) ** 2) * prob))

    # Spectral skewness
    if variance > 1e-20:
        skewness = float(np.sum(((freq - centroid) ** 3) * prob) / (variance ** 1.5))
    else:
        skewness = np.nan

    # Spectral kurtosis (excess)
    if variance > 1e-20:
        kurtosis = float(np.sum(((freq - centroid) ** 4) * prob) / (variance ** 2) - 3.0)
    else:
        kurtosis = np.nan

    # Spectral roll-off (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(amp_pos)
    if cumsum[-1] > 0:
        rolloff_idx = int(np.searchsorted(cumsum, 0.85 * cumsum[-1]))
        rolloff_hz = float(freq[min(rolloff_idx, n - 1)])
    else:
        rolloff_hz = np.nan

    # Amplitude slopes (low-half vs high-half of sweep)
    mid = n // 2
    if mid > 2:
        coeffs_low = np.polyfit(freq[:mid], amp[:mid], 1)
        slope_low = float(coeffs_low[0])
    else:
        slope_low = np.nan
    if n - mid > 2:
        coeffs_high = np.polyfit(freq[mid:], amp[mid:], 1)
        slope_high = float(coeffs_high[0])
    else:
        slope_high = np.nan

    return {
        "spectral_centroid_hz": centroid,
        "spectral_variance_hz2": variance,
        "spectral_skewness": skewness,
        "spectral_kurtosis": kurtosis,
        "spectral_rolloff_hz": rolloff_hz,
        "slope_low": slope_low,
        "slope_high": slope_high,
    }


# ---------------------------------------------------------------------------
# I/Q trajectory features
# ---------------------------------------------------------------------------

def _compute_iq_features(xv: np.ndarray, yv: np.ndarray) -> dict:
    """Compute features from the complex I/Q (in-phase / quadrature) trajectory."""
    if len(xv) < 4:
        return {}

    # I/Q ellipse area (approximated as |I × Q| cumulative rotation)
    # Compute the signed area of the polygon formed by the I/Q trajectory
    x = xv - np.mean(xv)
    y = yv - np.mean(yv)
    area = 0.5 * np.abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))

    # Mean radius and its variance
    radius = np.sqrt(xv ** 2 + yv ** 2)
    mean_radius = float(np.mean(radius))
    radius_std = float(np.std(radius))

    # Phase spread (circular std of I/Q angle)
    angles = np.arctan2(yv, xv)
    # Circular mean via complex representation
    cmean = np.mean(np.exp(1j * angles))
    phase_spread = float(np.sqrt(-2 * np.log(max(abs(cmean), 1e-10))))

    return {
        "iq_area": float(area),
        "iq_mean_radius": mean_radius,
        "iq_radius_std": radius_std,
        "iq_phase_spread_rad": phase_spread,
    }


# ---------------------------------------------------------------------------
# Phase transition features
# ---------------------------------------------------------------------------

def _extract_phase_features(freq: np.ndarray, phase: np.ndarray) -> dict:
    """Extract phase transition location and slope.

    The phase changes most rapidly near resonance. We find the frequency
    of steepest negative phase slope (smoothed) and measure the slope there.
    """
    n = len(freq)
    win = max(n // 12 * 2 + 1, 51)
    if win % 2 == 0:
        win += 1
    try:
        phase_sm = savgol_filter(phase, window_length=min(win, n - 1 - (n % 2)), polyorder=3)
    except Exception:
        phase_sm = phase

    dphase_df = np.gradient(phase_sm, freq)
    lo, hi = int(n * 0.05), int(n * 0.95)
    search = dphase_df.copy()
    search[:lo] = 0
    search[hi:] = 0

    idx = int(np.argmin(search))
    f_transition = float(freq[idx])
    slope = float(dphase_df[idx])

    return {
        "f_phase_transition_hz": f_transition,
        "phase_slope_deg_per_hz": abs(slope),
    }


# ---------------------------------------------------------------------------
# Diagnostic band plots (per sweep)
# ---------------------------------------------------------------------------

def plot_diagnostic_bands(df: pd.DataFrame, task: str, out_dir: Path, max_sweeps: int | None = None) -> None:
    """Generate diagnostic plots showing band boundaries, peaks, and half-power crossings.

    For each sweep, produces a figure with 4 subplots — one per
    PMUT mode band. Each subplot shows:
      - Raw amplitude trace (thin grey line)
      - Local band boundaries (vertical dashed lines)
      - Detrended amplitude within band (blue line)
      - Peak marker (red dot)
      - Half-power level (green dashed line)
      - Half-power crossing markers (green triangles) — or "no crossing" annotation
      - Q and bw values (or "Q not measurable")

    Parameters
    ----------
    max_sweeps : int or None
        Maximum number of sweeps to plot. None = all sweeps.
    """
    frame = load_task_frame(task)
    sweeps_plotted = 0
    for (_, sweep_key, label), group in frame.groupby(
        ["source", "sweep_key", "label"], sort=False
    ):
        if max_sweeps is not None and sweeps_plotted >= max_sweeps:
            break
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        freq = group["frequency_hz"].to_numpy(dtype=float)
        amp = group["amplitude"].to_numpy(dtype=float)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for ax, band in zip(axes, PMUT_MODE_BANDS):
            bname = band["name"]
            lo, hi = band["range_hz"]
            ilo, ihi = _local_band_indices(freq, lo, hi)

            ax.plot(freq / 1e6, amp, color="grey", alpha=0.4, linewidth=0.8, label="Raw amp")
            ax.axvline(lo / 1e6, color="orange", linestyle="--", alpha=0.6, linewidth=0.8)
            ax.axvline(hi / 1e6, color="orange", linestyle="--", alpha=0.6, linewidth=0.8)

            info = "Band too narrow"
            if ihi - ilo >= 5:
                freq_band = freq[ilo:ihi]
                amp_band = amp[ilo:ihi]

                edge_x = np.array([freq_band[0], freq_band[-1]])
                edge_y = np.array([amp_band[0], amp_band[-1]])
                if edge_x[1] > edge_x[0]:
                    slope_bl = (edge_y[1] - edge_y[0]) / (edge_x[1] - edge_x[0])
                    intercept_bl = edge_y[0] - slope_bl * edge_x[0]
                    local_baseline = slope_bl * freq_band + intercept_bl
                else:
                    local_baseline = np.full_like(freq_band, edge_y[0])
                amp_local = amp_band - local_baseline

                ax.plot(freq_band / 1e6, local_baseline, color="brown", alpha=0.4, linewidth=0.6, label="Local baseline")
                ax.plot(freq_band / 1e6, amp_local, color="blue", alpha=0.8, linewidth=1.2, label="Detrended")

                local_range = float(np.ptp(amp_local))
                if local_range > 1e-15:
                    min_prom = max(local_range * 0.1, 1e-15)
                    min_dist = max(len(freq_band) // 8, 3)
                    peaks, props = find_peaks(amp_local, prominence=min_prom, distance=min_dist)

                    if len(peaks) > 0:
                        prominences = props.get("prominences", np.ones(len(peaks)))
                        best_j = int(np.argmax(prominences))
                        p_local = peaks[best_j]
                        f_peak = freq_band[p_local]
                        amp_peak_local = amp_local[p_local]

                        ax.plot(f_peak / 1e6, amp_band[p_local], "ro", markersize=8, zorder=5, label="Peak")
                        hp_level = amp_peak_local / np.sqrt(2)
                        ax.axhline(hp_level + local_baseline[p_local], color="green", linestyle=":", alpha=0.7, linewidth=0.8)

                        f_left = _interp_crossing(freq_band, amp_local, hp_level, 0, p_local, "left")
                        f_right = _interp_crossing(freq_band, amp_local, hp_level, p_local, len(freq_band) - 1, "right")

                        if f_left is not None and f_right is not None:
                            ax.plot(f_left / 1e6, local_baseline[p_local] + hp_level, "gv", markersize=7, label="HP crossings")
                            ax.plot(f_right / 1e6, local_baseline[p_local] + hp_level, "gv", markersize=7)
                            bw = f_right - f_left
                            q_val = f_peak / bw
                            info = f"Q={q_val:.1f}, bw={bw/1e3:.1f} kHz"
                        elif f_left is not None:
                            ax.plot(f_left / 1e6, local_baseline[p_local] + hp_level, "gv", markersize=7)
                            info = "1 crossing only"
                        elif f_right is not None:
                            ax.plot(f_right / 1e6, local_baseline[p_local] + hp_level, "gv", markersize=7)
                            info = "1 crossing only"
                        else:
                            info = "No crossings — Q NaN"
                    else:
                        info = "No peak found"
                else:
                    info = "Flat band"

            ax.set_title(f"Band {bname} ({band['label']}, {band['method']}) — {info}", fontsize=10)
            ax.set_xlabel("Frequency (MHz)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Diagnostic Band Analysis — {label}, {sweep_key[:60]}", fontsize=13, y=1.01)
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{task}_diagnostic_bands_{sweeps_plotted:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        sweeps_plotted += 1

    print(f"  Saved {sweeps_plotted} diagnostic band plots → {out_dir}/")


# ---------------------------------------------------------------------------
# 1. Per-sweep feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_sweep(
    freq: np.ndarray, amp: np.ndarray, phase: np.ndarray,
    xv: np.ndarray, yv: np.ndarray,
    label: str, sweep_key: str,
) -> dict:
    """Extract resonance, spectral, and I/Q features from a single PMUT sweep.

    Uses LOCAL-band analysis for bandwidth and Q .
    Each PMUT mode band is analysed independently with its own baseline removal,
    peak detection on detrended signal, and half-power crossings restricted to
    the local band. Bands without clean resonances get NaN sentinels.
    """
    n = len(freq)
    if n < 4:
        return {}

    # ---- 1. Per-band local resonance analysis ---------------------------------
    band_results = {}
    for band in PMUT_MODE_BANDS:
        bname = band["name"]
        lo, hi = band["range_hz"]
        method = band["method"]
        br = _analyse_local_band(freq, amp, lo, hi, phase=phase, method=method)

        band_results[f"f0_{bname}_hz"] = br["f0_hz"]
        band_results[f"q_{bname}"] = br["q"]
        band_results[f"amp_{bname}"] = br["amp_at_peak"]
        band_results[f"bw_{bname}_hz"] = br["bw_hz"]
        band_results[f"peak_{bname}"] = br["peak_found"]
        # Fallback features for every band
        band_results[f"energy_{bname}"] = br["band_energy"]
        band_results[f"centroid_{bname}_hz"] = br["band_centroid_hz"]
        band_results[f"slope_{bname}"] = br["local_slope"]
        band_results[f"phase_grad_{bname}"] = br["phase_gradient"]
        band_results[f"prom_{bname}"] = br["prominence"]

    # ---- 2. Phase transition features ------------------------------------------
    phase_feat = _extract_phase_features(freq, phase)

    # ---- 3. Full-band spectral shape features ----------------------------------
    shape_feat = _compute_spectral_shape_features(freq, amp)

    # ---- 4. I/Q trajectory features --------------------------------------------
    iq_feat = _compute_iq_features(xv, yv)

    # ---- 5. Overall level features ---------------------------------------------
    amp_mean = float(np.mean(amp))
    amp_std = float(np.std(amp))
    amp_auc = float(np.trapezoid(amp, freq))
    peak_idx = int(np.argmax(amp))
    amp_peak = float(amp[peak_idx])
    # Noise floor: mean of lowest 10% of amplitude values
    noise_floor = float(np.mean(np.sort(amp)[: max(1, n // 10)]))

    return {
        "sweep_key": sweep_key,
        "label": label,
        "n_points": n,
        # Resonance features (4 PMUT modes — local-band corrected)
        **band_results,
        # Phase transition features
        "f_phase_transition_hz": phase_feat["f_phase_transition_hz"],
        "phase_slope_deg_per_hz": phase_feat["phase_slope_deg_per_hz"],
        # Full-band spectral shape
        **shape_feat,
        # I/Q trajectory
        **iq_feat,
        # Overall level
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "amp_peak": amp_peak,
        "amp_auc": amp_auc,
        "noise_floor": noise_floor,
    }


# ---------------------------------------------------------------------------
# 2. Baseline-relative features
# ---------------------------------------------------------------------------

def add_baseline_features(df: pd.DataFrame, baseline_label: str = "Air") -> pd.DataFrame:
    """Add frequency shift features relative to a baseline (Air or DI Water).

    Computes f0_shift_hz and f0_shift_pct for each band's resonance frequency
    relative to the baseline median.
    """
    baseline = df[df["label"] == baseline_label]
    if baseline.empty:
        print(f"  Note: no '{baseline_label}' sweeps found; skipping baseline features.")
        return df

    band_names = ["b1", "b2", "b3", "b4"]
    for bname in band_names:
        col = f"f0_{bname}_hz"
        if col not in baseline.columns:
            continue
        ref = baseline[col].median()
        if np.isfinite(ref) and ref > 0:
            for idx in df.index:
                val = df.loc[idx, col]
                if np.isfinite(val):
                    df.loc[idx, f"f0_{bname}_shift_hz"] = val - ref
                    df.loc[idx, f"f0_{bname}_shift_pct"] = (val - ref) / ref * 100

    return df


# ---------------------------------------------------------------------------
# 3. Build feature table
# ---------------------------------------------------------------------------

def build_feature_table(task: str, data_root: Path) -> pd.DataFrame:
    """Load the task frame and extract features for every sweep."""
    frame = load_task_frame(task, data_root)
    rows = []
    for (_, sweep_key, label), group in frame.groupby(
        ["source", "sweep_key", "label"], sort=False
    ):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        freq = group["frequency_hz"].to_numpy(dtype=float)
        amp = group["amplitude"].to_numpy(dtype=float)
        phase = group["phase"].to_numpy(dtype=float)
        xv = group["x_v"].to_numpy(dtype=float)
        yv = group["y_v"].to_numpy(dtype=float)
        row = extract_features_from_sweep(freq, amp, phase, xv, yv, label, sweep_key)
        if row:
            rows.append(row)
    df = pd.DataFrame(rows)
    df["_sort"] = df["label"].apply(
        lambda lb: _label_sort_key(lb)[1]
        if isinstance(_label_sort_key(lb), tuple) and _label_sort_key(lb)[0] == 1
        else 0
    )
    df = df.sort_values(["label", "sweep_key"]).drop(columns=["_sort"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------

def plot_feature_distributions(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Box-plots of resonance and phase features per class."""
    features = [
        ("f0_b2_hz", "Resonance Freq — Mode 2 (0.45M, Hz)"),
        ("q_b2", "Q-Factor — Mode 2 (local-band, NaN = not measurable)"),
        ("bw_b2_hz", "Bandwidth — Mode 2 (Hz, local-band)"),
        ("f0_b3_hz", "Resonance Freq — Mode 3 (0.9M, Hz)"),
        ("q_b3", "Q-Factor — Mode 3 (local-band)"),
        ("energy_b2", "Band Energy — Mode 2"),
        ("centroid_b2_hz", "Spectral Centroid — Mode 2 (Hz)"),
        ("amp_peak", "Peak Amplitude"),
        ("amp_auc", "Amplitude AUC"),
        ("spectral_centroid_hz", "Full-Band Spectral Centroid (Hz)"),
        ("spectral_skewness", "Full-Band Spectral Skewness"),
        ("spectral_kurtosis", "Full-Band Spectral Kurtosis"),
        ("f_phase_transition_hz", "Phase Transition Frequency (Hz)"),
        ("phase_slope_deg_per_hz", "Phase Slope (deg/Hz)"),
        ("iq_area", "I/Q Ellipse Area"),
        ("iq_phase_spread_rad", "I/Q Phase Spread (rad)"),
    ]
    labels = sorted(df["label"].unique(), key=_label_sort_key)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    n = len(features)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n))
    if n == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, features):
        if col not in df.columns:
            continue
        data = [df[df["label"] == lb][col].dropna().values for lb in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        ax.set_title(title, fontsize=12)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"PMUT Feature Distributions — {task}", fontsize=14, y=1.01)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{task}_feature_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature distributions → {out_dir / f'{task}_feature_distributions.png'}")


def plot_feature_correlation(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Feature correlation matrix."""
    feat_cols = [
        "f0_b1_hz", "q_b1", "f0_b2_hz", "q_b2",
        "f0_b3_hz", "q_b3", "f0_b4_hz", "q_b4",
        "energy_b2", "energy_b3", "centroid_b2_hz",
        "amp_mean", "amp_peak", "amp_auc", "noise_floor",
        "spectral_centroid_hz", "spectral_variance_hz2",
        "spectral_skewness", "spectral_kurtosis", "spectral_rolloff_hz",
        "slope_low", "slope_high",
        "f_phase_transition_hz", "phase_slope_deg_per_hz",
        "iq_area", "iq_mean_radius", "iq_phase_spread_rad",
    ]
    available = [c for c in feat_cols if c in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(available, fontsize=8)
    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=6, color="black")
    fig.colorbar(im, ax=ax, shrink=0.75)
    ax.set_title(f"Feature Correlation Matrix — {task}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_feature_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved feature correlation → {out_dir / f'{task}_feature_correlation.png'}")


def plot_class_wise_comparison(df: pd.DataFrame, task: str, out_dir: Path) -> None:
    """Class-wise comparison of key resonance features (grouped bar plot)."""
    labels = sorted(df["label"].unique(), key=_label_sort_key)
    feat_cols = [("f0_b2_hz", "Mode 2 f0 (MHz)"), ("q_b2", "Mode 2 Q (local-band)"), ("spectral_centroid_hz", "Spectral Centroid (Hz)")]

    fig, axes = plt.subplots(1, len(feat_cols), figsize=(5 * len(feat_cols), 4))
    if len(feat_cols) == 1:
        axes = [axes]

    for ax, (col, title) in zip(axes, feat_cols):
        if col not in df.columns:
            continue
        means = [df[df["label"] == lb][col].mean() for lb in labels]
        stds = [df[df["label"] == lb][col].std() for lb in labels]
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.tab10(np.linspace(0, 1, len(labels))))
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Class-Wise Feature Comparison — {task}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{task}_class_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved class-wise comparison → {out_dir / f'{task}_class_comparison.png'}")


def print_feature_summary(df: pd.DataFrame) -> None:
    """Print class-wise feature statistics."""
    print("\n" + "=" * 80)
    print("CLASS-WISE FEATURE SUMMARY (mean ± std)")
    print("=" * 80)
    feats = ["f0_b1_hz", "f0_b2_hz", "f0_b3_hz", "f0_b4_hz",
             "q_b2", "q_b3", "amp_mean", "amp_peak",
             "spectral_centroid_hz", "spectral_skewness",
             "iq_area", "f_phase_transition_hz"]
    for lb in sorted(df["label"].unique(), key=_label_sort_key):
        sub = df[df["label"] == lb]
        parts = [f"{lb:<20s}"]
        for f in feats:
            if f in df.columns:
                parts.append(f"{f}={sub[f].mean():.4g} ± {sub[f].std():.4g}")
        print("  ".join(parts))
    print("=" * 80)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PMUT physical features.")
    parser.add_argument("--task", choices=["high-glucose"], default="high-glucose",
        help="Primary task: high-glucose (5-class, 50–1000 mM, 3 sweeps/class). "
             "glucose-ladder retained in data.py for exploratory visualisation only.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features for task: {args.task}")
    df = build_feature_table(args.task, args.data_root)
    df = add_baseline_features(df, baseline_label="Air")

    csv_path = out_dir / f"{args.task}_feature_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved feature table ({len(df)} sweeps, {len(df.columns)} features) → {csv_path}")

    print_feature_summary(df)

    plot_feature_distributions(df, args.task, out_dir)
    plot_feature_correlation(df, args.task, out_dir)
    plot_class_wise_comparison(df, args.task, out_dir)
    plot_diagnostic_bands(df, args.task, out_dir)

    print("\nDone — Notebook 4 complete.")


if __name__ == "__main__":
    main()
