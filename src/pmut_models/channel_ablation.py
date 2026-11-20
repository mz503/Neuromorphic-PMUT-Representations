"""Matched channel ablation, without changing the primary four-channel pipeline."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits

from .data import DatasetBundle
from .reservoir import Reservoir, LiquidStateMachine

ROOT = Path(__file__).resolve().parents[2]
CHANNEL_SETS = {"four": [0, 1, 2, 3], "polar": [0, 1], "cartesian": [2, 3]}
DISPLAY = {"four": "[A, phi, X, Y]", "polar": "[A, phi]", "cartesian": "[X, Y]"}
SEEDS = [7, 42, 123, 999, 2024]


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(a.tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_fold(fold):
    path = ROOT / f"data/processed/windows/fold{fold}.npz"
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    metadata = json.loads(path.with_name(f"fold{fold}_metadata.json").read_text(encoding="utf-8"))
    metadata["source_npz_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for part in ["train", "val", "test"]:
        assert arrays[f"x_{part}"].shape == (230, 128, 4)
        assert len(set(arrays[f"{part}_groups"])) == 5
    sets = [set(arrays[f"{part}_groups"]) for part in ["train", "val", "test"]]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    return DatasetBundle(
        x_train=arrays["x_train"], y_train=arrays["y_train"],
        x_val=arrays["x_val"], y_val=arrays["y_val"],
        x_test=arrays["x_test"], y_test=arrays["y_test"],
        train_groups=arrays["train_groups"], val_groups=arrays["val_groups"],
        test_groups=arrays["test_groups"], labels=arrays["labels"].tolist(),
        channels=arrays["channels"].tolist(), metadata=metadata,
    )


def select_channels(bundle, representation):
    """Subset already training-normalised channels; preserve every sample/split."""
    indices = CHANNEL_SETS[representation]
    metadata = dict(bundle.metadata)
    metadata.update(representation=representation, channel_indices=indices,
                    channels=[bundle.channels[i] for i in indices])
    for name in ["normalization_mean", "normalization_std"]:
        metadata[name] = [bundle.metadata[name][i] for i in indices]
    return replace(bundle,
                   x_train=np.ascontiguousarray(bundle.x_train[:, :, indices]),
                   x_val=np.ascontiguousarray(bundle.x_val[:, :, indices]),
                   x_test=np.ascontiguousarray(bundle.x_test[:, :, indices]),
                   channels=metadata["channels"], metadata=metadata)


def paired_reservoir(kind, seed, indices):
    """Keep the original four-input draw, recurrence, biases and selected columns."""
    cls = Reservoir if kind == "esn" else LiquidStateMachine
    model = cls(n_input=4, seed=seed)
    model.W_in = model.W_in[:, indices].copy()
    model.n_input = len(indices)
    return model


def collect_features(model, x, state_mode="spikes", batch_size=115):
    """Batch independent windows using the existing recurrence and reset rules."""
    result = []
    for start in range(0, len(x), batch_size):
        states = model.process_sequence(x[start:start + batch_size], warmup=10)
        if isinstance(states, dict):
            states = states[state_mode]
        result.append(states.mean(axis=1))
    return np.concatenate(result)


def score_predictions(predictions, targets, groups, n_classes=5):
    keys = sorted(set(groups.tolist()))
    sweep_rows = []
    for key in keys:
        idx = np.flatnonzero(groups == key)
        assert len(set(targets[idx])) == 1
        vote = int(np.argmax(np.bincount(predictions[idx], minlength=n_classes)))
        sweep_rows.append({"sweep_key": str(key), "target": int(targets[idx[0]]),
                           "prediction": vote, "n_windows": int(len(idx))})
    return {
        "window_accuracy": float(np.mean(predictions == targets)),
        "sweep_accuracy_majority": float(np.mean([r["target"] == r["prediction"] for r in sweep_rows])),
        "sweep_predictions": sweep_rows,
    }


def source_fingerprint():
    sources = [Path(__file__), *[ROOT / f"src/pmut_models/{name}.py" for name in
                              ["data", "reservoir", "spiking", "train_utils"]]]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


def run_one(kind, fold, seed, representation, base, output, sources):
    indices = CHANNEL_SETS[representation]
    bundle = select_channels(base, representation)
    folder = output / "runs" / kind / representation / f"fold{fold}_seed{seed}"
    record_path = folder / "ablation_metrics.json"
    identity = {"model": kind, "fold": fold, "seed": seed, "representation": representation,
                "source_hashes": sources, "data_sha256": base.metadata["source_npz_sha256"]}
    if record_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["identity"] != identity:
            raise ValueError(f"Existing run has different provenance: {record_path}; use a new output directory.")
        return record
    folder.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if kind == "cnn1d":
        from .train_utils import train_single_run
        with (folder / "training.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            metrics = train_single_run(bundle, model_name="cnn1d", epochs=150,
                                      batch_size=32, lr=1e-3, output_dir=folder,
                                      seed=seed, paired_input_indices=indices)
        with (folder / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        predictions = np.array([int(r["prediction"]) for r in rows])
        targets = np.array([int(r["target"]) for r in rows])
        groups = np.array([r["sweep_key"] for r in rows])
        assert np.array_equal(targets, bundle.y_test)
        assert np.array_equal(groups, bundle.test_groups)
        extras = {"best_epoch": metrics["best"]["epoch"],
                  "best_validation_accuracy": metrics["best"]["val_accuracy"],
                  "sweep_accuracy_mean_probabilities": metrics["sweep_accuracy"]}
    else:
        model = paired_reservoir(kind, seed, indices)
        train_features = collect_features(model, bundle.x_train)
        test_features = collect_features(model, bundle.x_test)
        readout = RidgeClassifier(alpha=1.0).fit(train_features, bundle.y_train)
        predictions = readout.predict(test_features)
        targets, groups = bundle.y_test, bundle.test_groups
        np.savez_compressed(folder / "model_and_predictions.npz", W_in=model.W_in,
                            W_res=model.W_res, bias=model.bias,
                            readout_coef=readout.coef_, readout_intercept=readout.intercept_,
                            readout_classes=readout.classes_, predictions=predictions,
                            targets=targets, groups=groups)
        extras = {"recurrent_weights_sha256": digest(model.W_res),
                  "bias_sha256": digest(model.bias), "ridge_alpha": 1.0,
                  "state_mode": "continuous" if kind == "esn" else "spikes"}
    scores = score_predictions(predictions, targets, groups)
    record = {"identity": identity, **scores, **extras,
              "elapsed_seconds": time.perf_counter() - started, "dataset": bundle.metadata}
    write_json(record_path, record)
    print(f"{kind:5s} {representation:9s} fold={fold} seed={seed:4d}: "
          f"window={scores['window_accuracy']:.4f} sweep={scores['sweep_accuracy_majority']:.4f} "
          f"({record['elapsed_seconds']:.1f}s)", flush=True)
    return record


def summarise(records, output):
    grouped = defaultdict(list)
    for row in records:
        key = (row["identity"]["model"], row["identity"]["representation"])
        grouped[key].append(row)
    summary = []
    for (kind, representation), rows in grouped.items():
        item = {"model": kind, "representation": representation, "n_runs": len(rows)}
        for metric in ["window_accuracy", "sweep_accuracy_majority"]:
            folds = sorted({r["identity"]["fold"] for r in rows})
            means = [float(np.mean([r[metric] for r in rows if r["identity"]["fold"] == f])) for f in folds]
            item[metric + "_mean"] = float(np.mean(means))
            item[metric + "_sd_fold_means"] = float(np.std(means, ddof=0))
            item[metric + "_fold_means"] = means
        summary.append(item)
    write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        cols = [k for k in summary[0] if not isinstance(summary[0][k], list)]
        writer = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        writer.writeheader(); writer.writerows(summary)
    cells = {(r["model"], r["representation"]): r for r in summary}
    lines = ["# Matched channel ablation", "", "Window accuracy (%), mean ± population SD across the three fold means; seeds averaged within folds.",
             "LSM uses the spike-state readout. These are fresh matched runs, separate from historical manuscript results.", "",
             "| Model | [A, phi] | [X, Y] | [A, phi, X, Y] |", "|---|---:|---:|---:|"]
    for kind, title in [("cnn1d", "CNN1D"), ("esn", "ESN"), ("lsm", "LSM (spikes)")]:
        if all((kind, rep) in cells for rep in ["polar", "cartesian", "four"]):
            vals = [f"{100*cells[kind,rep]['window_accuracy_mean']:.1f} ± {100*cells[kind,rep]['window_accuracy_sd_fold_means']:.1f}" for rep in ["polar", "cartesian", "four"]]
            lines.append("| " + " | ".join([title, *vals]) + " |")
    (output / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["cnn1d", "esn", "lsm"], default=["cnn1d", "esn", "lsm"])
    parser.add_argument("--folds", nargs="+", type=int, choices=range(3), default=[0, 1, 2])
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--representations", nargs="+", choices=CHANNEL_SETS, default=list(CHANNEL_SETS))
    parser.add_argument("--output", type=Path, default=ROOT / "results/channel_ablation")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sources = source_fingerprint()
    write_json(args.output / "protocol.json", {
        "folds": args.folds, "seeds": args.seeds, "representations": args.representations,
        "models": args.models, "source_hashes": sources,
        "channels": CHANNEL_SETS, "primary_pipeline": "four channels, unchanged",
        "window_length": 128, "stride": 128, "train_val_test_windows_each": 230,
        "cnn_epochs": 150, "cnn_batch_size": 32, "cnn_lr": 0.001,
        "initialisation": "Original four-channel initialisation, selected input weights retained; no rescaling. Recurrence/bias/downstream CNN weights and random streams paired.",
        "lsm_readout": "mean post-reset spikes after 10-step warmup; original lower membrane clamp retained",
        "preprocessing": "Select columns of existing training-only per-channel z scores; no phase unwrapping or new preprocessing",
        "primary_metric": "window accuracy; equal fold weighting after seed averaging",
        "secondary_metric": "sweep majority vote; smallest class index on ties",
        "software": {"numpy": np.__version__, "torch": torch.__version__},
    })
    records = []
    with threadpool_limits(limits=1):
        for fold in args.folds:
            base = load_fold(fold)
            for seed in args.seeds:
                for kind in args.models:
                    for representation in args.representations:
                        records.append(run_one(kind, fold, seed, representation, base, args.output, sources))
                        summarise(records, args.output)
    write_json(args.output / "completed_runs.json", records)
    print((args.output / "table.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
