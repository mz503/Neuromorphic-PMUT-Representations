import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DatasetBundle
from .spiking import build_model, SpikeActivationHook


DEFAULT_SEEDS = [7, 42, 123, 999, 2024]

# ── Named encoders for the --encode flag ──
_ENCODER_REGISTRY: dict[str, str] = {
    "rate": "rate_encode",
    "latency": "latency_encode",
    "threshold": "threshold_crossing_encode",
    "delta": "delta_encode",
    "population": "population_encode",
}




def aggregate_to_sweep(
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    targets: np.ndarray,
    groups: np.ndarray,
    labels: list[str],
    method: str = "mean_logits",
) -> dict:
    """Aggregate window-level predictions to physical-sweep level.

    Groups per-window predictions by sweep_key, produces one prediction
    per held-out physical sweep. This makes window-level models (CNN, SNN,
    ESN, LSM) directly comparable to sweep-level models (RF on extracted
    features).

    Parameters
    ----------
    predictions : ndarray of shape (n_windows,) — integer class predictions
    probabilities : ndarray of shape (n_windows, n_classes) or None
        If None and method is "mean_logits", falls back to majority_vote.
    targets : ndarray of shape (n_windows,) — integer class labels
    groups : ndarray of shape (n_windows,) — sweep_key per window
    labels : list[str] — class label names
    method : str — "mean_logits" (avg probabilities → argmax) or
                    "majority_vote" (mode of window predictions)

    Returns
    -------
    dict with:
        sweep_preds : ndarray of shape (n_sweeps,) — sweep-level class index
        sweep_targets : ndarray of shape (n_sweeps,) — sweep-level class index
        sweep_keys : list[str] — sweep identifiers
        window_counts : list[int] — windows per sweep
        method : str — method actually used
    """
    unique_groups = sorted(set(groups.tolist()))
    sweep_preds_list: list[int] = []
    sweep_targets_list: list[int] = []
    sweep_keys: list[str] = []
    window_counts: list[int] = []

    use_proba = probabilities is not None and method == "mean_logits"

    for sweep_key in unique_groups:
        idx = np.where(groups == sweep_key)[0]
        window_counts.append(len(idx))

        # All windows from the same sweep have the same label
        sweep_target = int(targets[idx[0]])
        sweep_targets_list.append(sweep_target)
        sweep_keys.append(str(sweep_key))

        if use_proba and probabilities is not None:
            # Average class probabilities across windows
            avg_proba = probabilities[idx].mean(axis=0)
            sweep_pred = int(np.argmax(avg_proba))
        else:
            # Majority vote
            window_preds = predictions[idx]
            counts = np.bincount(window_preds.astype(int), minlength=len(labels))
            sweep_pred = int(np.argmax(counts))

        sweep_preds_list.append(sweep_pred)

    actual_method = "mean_logits" if use_proba else "majority_vote"

    return {
        "sweep_preds": np.array(sweep_preds_list, dtype=np.int64),
        "sweep_targets": np.array(sweep_targets_list, dtype=np.int64),
        "sweep_keys": sweep_keys,
        "window_counts": window_counts,
        "method": actual_method,
    }


# ---------------------------------------------------------------------------
# Encoder→SNN diagnostics
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(bundle: DatasetBundle, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    train_ds = TensorDataset(torch.from_numpy(bundle.x_train), torch.from_numpy(bundle.y_train))
    test_ds = TensorDataset(torch.from_numpy(bundle.x_test), torch.from_numpy(bundle.y_test))
    val_loader = None
    if bundle.x_val is not None and bundle.y_val is not None:
        val_ds = TensorDataset(torch.from_numpy(bundle.x_val), torch.from_numpy(bundle.y_val))
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        val_loader,
    )


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == targets).float().mean().item())


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, collect_spike_maps: bool = False, return_proba: bool = False) -> tuple[float, np.ndarray, np.ndarray, dict | None, np.ndarray | None]:
    model.eval()
    correct = 0
    total = 0
    predictions = []
    targets_all = []
    probas_all = []
    spike_maps = None
    if collect_spike_maps and hasattr(model, "register_spike_hooks"):
        hooks = model.register_spike_hooks()
    else:
        hooks = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            predictions.extend(pred.cpu().numpy().tolist())
            targets_all.extend(y.cpu().numpy().tolist())
            if return_proba:
                probas_all.append(torch.softmax(logits, dim=1).cpu().numpy())
    if hooks:
        spike_maps = {hook.name: hook.get_activation_map().numpy() for hook in hooks}
    proba_array = np.concatenate(probas_all) if return_proba and probas_all else None
    return correct / max(total, 1), np.array(predictions), np.array(targets_all), spike_maps, proba_array


def train_single_run(
    bundle: DatasetBundle,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    output_dir: Path,
    seed: int = 7,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, val_loader = make_loaders(bundle, batch_size=batch_size)
    input_shape = (bundle.x_train.shape[1], bundle.x_train.shape[2])

    # Use validation set if available, otherwise fall back to train set size
    n_val = len(bundle.y_val) if bundle.y_val is not None else 0
    has_val = val_loader is not None

    # encoded_spike_snn needs to know the encoder output dimensionality
    n_features: int | None = None
    if model_name == "encoded_spike_snn":
        n_features = bundle.x_train.shape[2]  # encoder output channels

    model = build_model(
        model_name,
        input_shape=input_shape,
        num_classes=len(bundle.labels),
        n_features=n_features,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = {"epoch": 0, "val_accuracy": -1.0}
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += float(loss.item())
            train_acc += accuracy_from_logits(logits.detach(), y)
            n_batches += 1

        # ── Validate: use val set if available, else (legacy) use test set ──
        if has_val:
            val_acc, _, _, _, _ = evaluate(model, val_loader, device)
        else:
            val_acc, _, _, _, _ = evaluate(model, test_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(n_batches, 1),
            "train_accuracy": train_acc / max(n_batches, 1),
            "val_accuracy": val_acc,
        }
        history.append(row)

        # Model selection on VALIDATION accuracy (not test)
        if val_acc > best["val_accuracy"]:
            best = row.copy()
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        eval_label = "val" if has_val else "test"
        print(
            f"seed={seed:04d} epoch={epoch:03d} loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} {eval_label}_acc={val_acc:.4f}"
        )

    # ── Final evaluation: load best-val checkpoint, evaluate on TEST once ──
    if (output_dir / "best_model.pt").exists():
        model.load_state_dict(torch.load(output_dir / "best_model.pt", weights_only=True))

    test_acc, predictions, targets, spike_maps, proba = evaluate(
        model, test_loader, device, collect_spike_maps=True, return_proba=True,
    )

    # ── Per-class + aggregate metrics ──
    from sklearn.metrics import (
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
    )
    bal_acc = float(balanced_accuracy_score(targets, predictions))
    cm = confusion_matrix(targets, predictions)
    report = classification_report(
        targets, predictions, target_names=bundle.labels, output_dict=True, zero_division=0,
    )
    per_class_recall = {
        lbl: float(report[lbl]["recall"]) for lbl in bundle.labels if lbl in report
    }
    macro = report.get("macro avg", {})
    macro_precision = float(macro.get("precision", 0))
    macro_recall = float(macro.get("recall", 0))
    macro_f1 = float(macro.get("f1-score", 0))

    # ── Sweep-level aggregation (if test_groups available) ──
    sweep_metrics = None
    test_groups = bundle.test_groups
    if test_groups is not None and len(test_groups) == len(predictions):
        sweep_metrics = aggregate_to_sweep(
            predictions=predictions,
            probabilities=proba,
            targets=targets,
            groups=test_groups,
            labels=bundle.labels,
            method="mean_logits" if proba is not None else "majority_vote",
        )
        sweep_acc = float((sweep_metrics["sweep_preds"] == sweep_metrics["sweep_targets"]).mean())
        sweep_metrics["sweep_accuracy"] = sweep_acc
        print(f"  Sweep-level accuracy ({sweep_metrics['method']}): {sweep_acc:.4f} "
              f"({len(sweep_metrics['sweep_keys'])} sweeps)")

    metrics = {
        "seed": seed,
        "model": model_name,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "labels": bundle.labels,
        "dataset": bundle.metadata,
        "final_test_accuracy": test_acc,
        "balanced_accuracy": bal_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
        "n_val": n_val,
        "has_val": has_val,
        "best": best,
    }
    if sweep_metrics is not None:
        metrics["sweep_accuracy"] = sweep_metrics["sweep_accuracy"]
        metrics["sweep_aggregation_method"] = sweep_metrics["method"]
        metrics["n_sweeps"] = len(sweep_metrics["sweep_keys"])

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        hdr = ["epoch", "train_loss", "train_accuracy", "val_accuracy"]
        writer = csv.DictWriter(f, fieldnames=hdr)
        writer.writeheader()
        writer.writerows(history)
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        pred_fields = ["target", "prediction", "target_label", "prediction_label"]
        has_sweep_key = test_groups is not None and len(test_groups) == len(predictions)
        if has_sweep_key:
            pred_fields.append("sweep_key")
        writer = csv.DictWriter(f, fieldnames=pred_fields)
        writer.writeheader()
        for i, (target, pred) in enumerate(zip(targets.tolist(), predictions.tolist())):
            row = {
                "target": target,
                "prediction": pred,
                "target_label": bundle.labels[target],
                "prediction_label": bundle.labels[pred],
            }
            if has_sweep_key:
                row["sweep_key"] = str(test_groups[i])
            writer.writerow(row)
    if sweep_metrics is not None:
        with (output_dir / "sweep_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            sw_fields = ["sweep_key", "target", "prediction", "target_label", "prediction_label"]
            writer = csv.DictWriter(f, fieldnames=sw_fields)
            writer.writeheader()
            for i in range(len(sweep_metrics["sweep_keys"])):
                t = int(sweep_metrics["sweep_targets"][i])
                p = int(sweep_metrics["sweep_preds"][i])
                writer.writerow({
                    "sweep_key": sweep_metrics["sweep_keys"][i],
                    "target": t,
                    "prediction": p,
                    "target_label": bundle.labels[t],
                    "prediction_label": bundle.labels[p],
                })
    if spike_maps:
        np.savez_compressed(output_dir / "spike_maps.npz", **spike_maps)
    return metrics



