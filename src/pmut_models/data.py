import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DATA_ROOT


PURE_MEDIUM_MAP = {
    "AIR": "Air",
    "WATER": "DI Water",
    "100ACETONE": "Acetone",
    "100IPA": "IPA",
    "50IPA50DI": "50% IPA",
    "50DI50ACETONE": "50% Acetone",
}

SIGNAL_COLUMNS = ["amplitude", "phase", "x_v", "y_v"]


@dataclass
class DatasetBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    labels: list[str]
    channels: list[str]
    train_groups: np.ndarray
    test_groups: np.ndarray
    metadata: dict
    x_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    val_groups: np.ndarray | None = None


def _file_paths(data_root: Path) -> dict[str, Path]:
    return {
        "liquid": data_root / "baseline_liquids" / "Liquid_Loading_Dataset_Air_DIW_Acetone_IPA.csv",
        "high": data_root / "pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx",
    }


def load_liquid_loading_pure(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_csv(_file_paths(data_root)["liquid"])
    raw = raw[raw["medium"].isin(PURE_MEDIUM_MAP)].copy()
    raw["label"] = raw["medium"].map(PURE_MEDIUM_MAP)
    raw["source"] = "liquid_loading_csv"
    raw["sweep_key"] = raw["medium"].astype(str) + "_" + raw["chunk_timestamp"].astype(str)
    raw["frequency_hz"] = raw["frequency_hz"]
    raw["amplitude"] = raw["amplitude_v"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x_v"]
    raw["y_v"] = raw["y_v"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]


def _normalize_air_di_from_high(condition: str) -> str | None:
    if condition.startswith("Air"):
        return "Air"
    if condition.startswith("DI Water"):
        return "DI Water"
    return None


def load_high_air_di(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_excel(_file_paths(data_root)["high"], sheet_name="AllSamples")
    raw["label"] = raw["condition"].map(_normalize_air_di_from_high)
    raw = raw[raw["label"].notna()].copy()
    raw["source"] = "high_glucose_xlsx"
    raw["sweep_key"] = raw["condition"].astype(str)
    raw["frequency_hz"] = raw["frequency"]
    raw["amplitude"] = raw["r"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x"]
    raw["y_v"] = raw["y"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]




def _normalize_high_condition(condition: str) -> str | None:
    if condition.startswith("Air"):
        return "Air"
    if condition.startswith("DI Water"):
        return "DI Water"
    match = re.search(r"Glucose\s+(\d+)mM", condition)
    if match:
        return f"Glucose {match.group(1)} mM"
    return None


def load_high_glucose(data_root: Path = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    raw = pd.read_excel(_file_paths(data_root)["high"], sheet_name="AllSamples")
    raw["label"] = raw["condition"].map(_normalize_high_condition)
    raw = raw[raw["label"].notna()].copy()
    raw["source"] = "high_glucose_xlsx"
    raw["sweep_key"] = raw["condition"].astype(str)
    raw["frequency_hz"] = raw["frequency"]
    raw["amplitude"] = raw["r"]
    raw["phase"] = raw["phase_deg"]
    raw["x_v"] = raw["x"]
    raw["y_v"] = raw["y"]
    return raw[["source", "sweep_key", "label", "frequency_hz", *SIGNAL_COLUMNS]]






def validate_phase_data(frame: pd.DataFrame, window_size: int = 128) -> dict:
    """Audit phase preprocessing for correctness.

    Checks across all sweeps in the frame:
      1. Phase units (degrees vs radians) — warns if values in [-pi, pi] range
      2. Phase wrapping: counts ±180° discontinuities within a sliding window
      3. Sign convention: verifies x_v ≈ amplitude·cos(phase), y_v ≈ amplitude·sin(phase)
      4. Discontinuity count per window: flags windows containing wraps

    Returns a dict with wrapped_windows, sign_errors, and summary statistics.
    """
    n_total = len(frame)
    results = {
        "n_total_points": n_total,
        "phase_range": (float(frame["phase"].min()), float(frame["phase"].max())),
        "wrapped_windows": 0,
        "total_windows": 0,
        "sign_errors_x": 0,
        "sign_errors_y": 0,
        "summary": "",
    }

    # 1. Phase unit check
    pmin, pmax = results["phase_range"]
    if -4.0 < pmin < 4.0 and -4.0 < pmax < 4.0:
        results["summary"] += (
            f"WARNING: Phase range [{pmin:.2f}, {pmax:.2f}] looks like RADIANS. "
            f"Expected degrees for this dataset.\n"
        )
    elif abs(pmin) < 200 and abs(pmax) < 200:
        results["summary"] += (
            f"Phase range [{pmin:.1f}, {pmax:.1f}] degrees — looks correct.\n"
        )
    else:
        results["summary"] += (
            f"Phase range [{pmin:.1f}, {pmax:.1f}] — unexpected range.\n"
        )

    # 2 + 4. Per-sweep wrapping check
    for (source, sweep_key), group in frame.groupby(["source", "sweep_key"], sort=False):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        phase = group["phase"].to_numpy(dtype=float)
        amp = group["amplitude"].to_numpy(dtype=float)
        xv = group["x_v"].to_numpy(dtype=float)
        yv = group["y_v"].to_numpy(dtype=float)
        n = len(phase)

        # 2. Detect phase wraps within sliding windows of size `window_size`
        if n >= window_size:
            stride = window_size  # non-overlapping
            n_wins = (n - window_size) // stride + 1
            for w in range(n_wins):
                start = w * stride
                end = start + window_size
                win_phase = phase[start:end]

                # Check for ±180° discontinuities (wrap detection)
                diffs = np.abs(np.diff(win_phase))
                wraps = int(np.sum(diffs > 300))  # >300° jump = likely wrap
                results["total_windows"] += 1
                if wraps > 0:
                    results["wrapped_windows"] += 1

        # 3. Sign convention: x_v should be approx amplitude * cos(phase_rad)
        phase_rad = np.deg2rad(phase)
        x_expected = amp * np.cos(phase_rad)
        y_expected = amp * np.sin(phase_rad)

        x_error = np.abs(xv - x_expected)
        mean_x_err = float(np.mean(x_error))
        if mean_x_err > amp.mean() * 0.1:  # >10% of mean amplitude
            results["sign_errors_x"] += 1

        y_error = np.abs(yv - y_expected)
        mean_y_err = float(np.mean(y_error))
        if mean_y_err > amp.mean() * 0.1:
            results["sign_errors_y"] += 1

    # Build summary
    frac_wrapped = results["wrapped_windows"] / max(results["total_windows"], 1)
    results["summary"] += (
        f"Windows: {results['total_windows']} total, "
        f"{results['wrapped_windows']} with phase wraps "
        f"({frac_wrapped*100:.1f}%)\n"
    )
    results["summary"] += (
        f"Sign errors: {results['sign_errors_x']} X, {results['sign_errors_y']} Y "
        f"(sweeps where |measured - A·cos/sin(φ)| > 10% of mean amp)\n"
    )

    if frac_wrapped > 0.1:
        results["summary"] += (
            "WARNING: >10% of windows contain phase wraps. "
            "These appear as step-function artifacts to CNN/LSM models.\n"
        )
    if results["sign_errors_x"] > 0 or results["sign_errors_y"] > 0:
        results["summary"] += (
            "WARNING: x_v/y_v sign convention mismatch detected. "
            "Check whether x/y columns are direct measurements or reconstructed from polar.\n"
        )
    if frac_wrapped <= 0.1 and results["sign_errors_x"] == 0 and results["sign_errors_y"] == 0:
        results["summary"] += "Phase preprocessing: ALL CHECKS PASSED.\n"

    print("\n" + "=" * 60)
    print("PHASE PREPROCESSING AUDIT")
    print("=" * 60)
    print(results["summary"])
    print("=" * 60)

    return results


def make_sequence_samples(frame: pd.DataFrame, window_size: int, stride: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stride is None:
        stride = window_size  # non-overlapping windows by default
    sequences = []
    labels = []
    groups = []
    for (_, sweep_key, label), group in frame.groupby(["source", "sweep_key", "label"], sort=False):
        group = group.sort_values("frequency_hz").reset_index(drop=True)
        values = group[SIGNAL_COLUMNS].to_numpy(dtype=np.float32)
        for start in range(0, len(values) - window_size + 1, stride):
            sequences.append(values[start : start + window_size])
            labels.append(label)
            groups.append(sweep_key)
    return np.stack(sequences), np.array(labels), np.array(groups)


def encode_labels(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    classes = sorted(set(labels.tolist()), key=_label_sort_key)
    lookup = {label: idx for idx, label in enumerate(classes)}
    return np.array([lookup[label] for label in labels], dtype=np.int64), classes
## sorts the unique labels alphabetically

def _label_sort_key(label: str) -> tuple[int, float | str]:
    match = re.search(r"Glucose\s+(\d+)", label)
    if match:
        return (1, float(match.group(1)))
    return (0, label)

##Decides which windows go to training and which to testing
def split_indices(labels: np.ndarray, groups: np.ndarray, split: str, test_every: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if split == "window":
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            mask = np.arange(len(idx)) % test_every == test_every - 1
            if not mask.any():
                mask[-1] = True
            test_idx.extend(idx[mask].tolist())
            train_idx.extend(idx[~mask].tolist())
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    if split == "random_window":
        rng = np.random.RandomState(42)
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            perm = rng.permutation(len(idx))
            n_test = max(1, len(idx) // test_every)
            test_positions = set(perm[:n_test].tolist())
            for j, global_idx in enumerate(idx):
                if j in test_positions:
                    test_idx.append(int(global_idx))
                else:
                    train_idx.append(int(global_idx))
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    if split == "grouped":
        train_idx = []
        test_idx = []
        for label in sorted(set(labels.tolist()), key=_label_sort_key):
            idx = np.where(labels == label)[0]
            label_groups = sorted(set(groups[idx].tolist()))
            if len(label_groups) < 2:
                continue
            holdout = label_groups[-1]
            test_mask = groups[idx] == holdout
            test_idx.extend(idx[test_mask].tolist())
            train_idx.extend(idx[~test_mask].tolist())
        if not train_idx:
            raise ValueError("Grouped split needs at least two groups for at least one class.")
        return np.array(train_idx, dtype=np.int64), np.array(test_idx, dtype=np.int64)
    raise ValueError(f"Unknown split: {split}")

##Computes z-score normalization 
def standardize_from_train(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.reshape(-1, x_train.shape[-1]).mean(axis=0)
    std = x_train.reshape(-1, x_train.shape[-1]).std(axis=0)
    std[std == 0.0] = 1.0
    return (x_train - mean) / std, (x_test - mean) / std, mean, std

##
def build_dataset(
    task: str,
    split: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    window_size: int = 256,
    stride: int | None = None,
) -> DatasetBundle:
    frame = load_task_frame(task, data_root)
    x, raw_labels, groups = make_sequence_samples(frame, window_size=window_size, stride=stride)
    y, labels = encode_labels(raw_labels)
    train_idx, test_idx = split_indices(raw_labels, groups, split=split)
    x_train_full, x_test, mean, std = standardize_from_train(x[train_idx], x[test_idx])
    y_train_full = y[train_idx]
    y_test = y[test_idx]
    train_groups_full = groups[train_idx]

    # ── Create validation split from training data (stratified) ──
    #   Training data → model weight updates
    #   Validation data → epoch/hyperparameter selection (used once per epoch)
    #   Test data       → final evaluation (used ONCE after training completes)
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val, tg_train, tg_val = train_test_split(
        x_train_full, y_train_full, train_groups_full,
        test_size=0.2, random_state=42, stratify=y_train_full,
    )

    metadata = {
        "task": task,
        "split": split,
        "window_size": window_size,
        "stride": stride or window_size,
        "n_samples": int(len(x)),
        "n_train": int(len(train_idx)),
        "n_train_final": int(len(x_train)),
        "n_val": int(len(x_val)),
        "n_test": int(len(test_idx)),
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    return DatasetBundle(
        x_train=x_train.astype(np.float32),
        y_train=y_train,
        x_val=x_val.astype(np.float32),
        y_val=y_val,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
        labels=labels,
        channels=SIGNAL_COLUMNS,
        train_groups=tg_train,
        test_groups=groups[test_idx],
        val_groups=tg_val,
        metadata=metadata,
    )


def build_dataset_folds(
    task: str,
    data_root: Path = DEFAULT_DATA_ROOT,
    window_size: int = 256,
    stride: int | None = None,
) -> list[DatasetBundle]:
    """Build 3-fold sweep-level rotation for high-glucose grouped split.

    Each concentration (class) has exactly 3 physical sweeps. This function
    produces 3 folds where train/validation/test each contain windows from a
    DIFFERENT sweep — NO leakage between splits at the physical-sweep level.

      Fold 0: train=sweep1, val=sweep2, test=sweep3
      Fold 1: train=sweep2, val=sweep3, test=sweep1
      Fold 2: train=sweep3, val=sweep1, test=sweep2

    Normalization is fitted on training windows only per fold, then applied to
    val and test. This prevents the final result from depending on which sweep
    was arbitrarily chosen as the test acquisition.

    Returns a list of 3 DatasetBundle objects, one per fold.
    """
    frame = load_task_frame(task, data_root)
    x, raw_labels, groups = make_sequence_samples(frame, window_size=window_size, stride=stride)
    y, labels = encode_labels(raw_labels)

    # Identify sweep keys per class, sorted for deterministic rotation
    class_sweeps: dict[str, list[str]] = {}
    for label_name in labels:
        idx = np.where(raw_labels == label_name)[0]
        sweeps = sorted(set(groups[idx].tolist()))
        class_sweeps[label_name] = sweeps

    # Verify every class has exactly 3 sweeps (required for 3-fold rotation)
    for label_name, sweeps in class_sweeps.items():
        if len(sweeps) != 3:
            raise ValueError(
                f"build_dataset_folds requires exactly 3 sweeps per class. "
                f"Class '{label_name}' has {len(sweeps)} sweeps: {sweeps}. "
                f"Use build_dataset() for non-3-sweep tasks."
            )

    folds: list[DatasetBundle] = []
    # Rotation patterns: (train_sweep, val_sweep, test_sweep) per fold
    rotations = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]

    for fold_idx, (ti, vi, tsti) in enumerate(rotations):
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []

        for label_name in labels:
            idx = np.where(raw_labels == label_name)[0]
            sweeps = class_sweeps[label_name]

            train_sweep = sweeps[ti]
            val_sweep = sweeps[vi]
            test_sweep = sweeps[tsti]

            train_idx.extend(idx[groups[idx] == train_sweep].tolist())
            val_idx.extend(idx[groups[idx] == val_sweep].tolist())
            test_idx.extend(idx[groups[idx] == test_sweep].tolist())

        # Normalize: fit on training windows only
        x_train_full = x[np.array(train_idx, dtype=np.int64)]
        x_val_raw = x[np.array(val_idx, dtype=np.int64)]
        x_test_raw = x[np.array(test_idx, dtype=np.int64)]
        y_train_full = y[np.array(train_idx, dtype=np.int64)]
        y_val = y[np.array(val_idx, dtype=np.int64)]
        y_test = y[np.array(test_idx, dtype=np.int64)]

        mean = x_train_full.reshape(-1, x_train_full.shape[-1]).mean(axis=0)
        std = x_train_full.reshape(-1, x_train_full.shape[-1]).std(axis=0)
        std[std == 0.0] = 1.0

        x_train_norm = (x_train_full - mean) / std
        x_val_norm = (x_val_raw - mean) / std
        x_test_norm = (x_test_raw - mean) / std

        metadata = {
            "task": task,
            "split": "grouped_folds",
            "fold": fold_idx,
            "train_sweeps": [class_sweeps[lb][ti] for lb in labels],
            "val_sweeps": [class_sweeps[lb][vi] for lb in labels],
            "test_sweeps": [class_sweeps[lb][tsti] for lb in labels],
            "window_size": window_size,
            "stride": stride or window_size,
            "n_samples": int(len(x)),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "n_test": len(test_idx),
            "normalization_mean": mean.tolist(),
            "normalization_std": std.tolist(),
        }

        folds.append(DatasetBundle(
            x_train=x_train_norm.astype(np.float32),
            y_train=y_train_full,
            x_val=x_val_norm.astype(np.float32),
            y_val=y_val,
            x_test=x_test_norm.astype(np.float32),
            y_test=y_test,
            labels=labels,
            channels=SIGNAL_COLUMNS,
            train_groups=groups[np.array(train_idx, dtype=np.int64)],
            test_groups=groups[np.array(test_idx, dtype=np.int64)],
            val_groups=groups[np.array(val_idx, dtype=np.int64)],
            metadata=metadata,
        ))

    return folds


def save_dataset(bundle: DatasetBundle, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_test=bundle.x_test,
        y_test=bundle.y_test,
        labels=np.array(bundle.labels),
        channels=np.array(bundle.channels),
        train_groups=bundle.train_groups,
        test_groups=bundle.test_groups,
        metadata=json.dumps(bundle.metadata),
    )
    if bundle.x_val is not None:
        save_kwargs["x_val"] = bundle.x_val
        save_kwargs["y_val"] = bundle.y_val
        save_kwargs["val_groups"] = bundle.val_groups
    np.savez_compressed(output_path, **save_kwargs)


def load_npz_dataset(path: Path) -> DatasetBundle:
    data = np.load(path, allow_pickle=False)
    return DatasetBundle(
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        labels=data["labels"].tolist(),
        channels=data["channels"].tolist(),
        train_groups=data["train_groups"],
        test_groups=data["test_groups"],
        metadata=json.loads(str(data["metadata"])),
        x_val=data.get("x_val") if "x_val" in data else None,
        y_val=data.get("y_val") if "y_val" in data else None,
        val_groups=data.get("val_groups") if "val_groups" in data else None,
    )


def load_task_frame(task, data_root=DEFAULT_DATA_ROOT):
    if task == 'high-glucose':
        frame=load_high_glucose(data_root)
        return frame[frame['label'].str.startswith('Glucose')].copy()
    if task == 'pure':
        return load_liquid_loading_pure(data_root)
    raise ValueError(f'Unknown task: {task}')
