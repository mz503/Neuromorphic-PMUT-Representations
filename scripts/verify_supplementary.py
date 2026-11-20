"""Check the archived evidence and the numerical values in Tables S1–S12."""
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "results/supplementary"
MODELS = {"CNN1D": "cnn1d", "ESN": "esn", "LSM (spikes)": "lsm",
          "Direct spectrum matching": "matching", "Physical-feature RF": "rf_physical"}


def csv(name):
    return pd.read_csv(ROOT / name, encoding="utf-8-sig")


def table(number):
    path, = (ROOT / "tables").glob(f"S{number:02d}_*.csv")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def compare(cell, actual, context):
    shown = [token.strip() for token in cell.split("±")]
    actual = list(actual)
    if len(shown) != len(actual):
        raise AssertionError((context, shown, actual))
    for token, number in zip(shown, actual):
        digits = len(token.partition(".")[2])
        if abs(float(token) - number) > 0.5 * 10**-digits + 1e-9:
            raise AssertionError((context, token, number))


def summary(row, *keys):
    return (100 * float(row[key]) for key in keys)


def verify_ablation():
    records = csv("S01_channel_ablation/window_predictions.csv.gz")
    archive = pd.DataFrame(json.loads((ROOT / "S01_channel_ablation/completed_runs.json")
                                      .read_text(encoding="utf-8")))
    archive = pd.concat([pd.json_normalize(archive.identity), archive], axis=1)
    keys = ["model", "representation", "fold", "seed"]
    scores = (records.assign(correct=records.target.eq(records.prediction))
              .groupby(keys).correct.agg(["size", "mean"]))
    assert len(scores) == 135 and scores["size"].eq(230).all()
    saved = archive.set_index(keys).window_accuracy.sort_index()
    assert saved.index.equals(scores.index)
    np.testing.assert_allclose(saved, scores["mean"], rtol=0, atol=1e-15)
    folds = scores["mean"].groupby(level=keys[:3]).mean()
    for model, *cells in table(1).itertuples(index=False, name=None):
        for representation, cell in zip(("polar", "cartesian", "four"), cells, strict=True):
            values = folds.loc[MODELS[model], representation].to_numpy()
            assert len(values) == 3
            compare(cell, (100 * values.mean(), 100 * values.std()), ("S1", model, representation))


def verify_search():
    folder = "S02_S04_cnn_search/"
    widths = csv(folder + "channel_width_comparison.csv")
    for channels, parameters, accuracy in table(2).itertuples(index=False, name=None):
        rows = widths[widths.channels.eq(channels)]
        scores = rows.validation_accuracy_mean_pct.to_numpy()
        assert len(scores) == 3 and rows.parameters.eq(int(parameters.replace(",", ""))).all()
        compare(accuracy, (scores.mean(), scores.std()), ("S2", channels))
    kernels = csv(folder + "kernel_comparison.csv")
    for kernel, _, accuracy in table(3).itertuples(index=False, name=None):
        scores = kernels.loc[kernels.kernels.eq(kernel), "validation_accuracy_mean_pct"]
        assert len(scores) == 3
        compare(accuracy, (scores.mean(),), ("S3", kernel))
    selected = csv(folder + "selected_configurations.csv").set_index("fold")
    for fold, channels, kernels, lr, batch, accuracy in table(4).itertuples(index=False, name=None):
        row = selected.loc[int(fold)]
        assert lr == "10^{-3}"
        assert (row.channels, row.kernels, row.learning_rate, row.batch_size) == (
            channels, kernels, 1e-3, int(batch)
        )
        compare(accuracy, (row.validation_accuracy_mean_pct,), ("S4", fold))


def verify_windows():
    folder = "S05_S06_window_and_matching/"
    windows = csv(folder + "window_summary.csv").set_index(["model", "window_length"])
    for model, *cells in table(5).itertuples(index=False, name=None):
        for length, cell in zip((64, 128, 256), cells, strict=True):
            row = windows.loc[MODELS[model], length]
            compare(cell, summary(row, "test_sweep_mean", "test_sweep_sd"), ("S5", model, length))
    matching = csv(folder + "comparison_summary.csv").set_index("method")
    for model, cell in table(6).itertuples(index=False, name=None):
        columns = ["sweep_accuracy_mean"] if "Random guessing" in model else ["sweep_accuracy_mean", "sweep_accuracy_sd"]
        compare(cell, summary(matching.loc[model], *columns), ("S6", model))


def verify_partitions():
    folder = "S07_S12_partition_and_readout/"
    partitions = json.loads((ROOT / folder / "partitions.json").read_text(encoding="utf-8"))
    displayed = table(7)
    assert len(partitions) == len(displayed) == 18
    for partition, row in zip(partitions, displayed.itertuples(index=False, name=None), strict=True):
        train, validation, test = (partition[k] for k in ("train", "val", "test"))
        assert sorted(train + validation + test) == list(range(15))
        repeats = ["/".join(str(sweep % 3 + 1) for sweep in group) for group in (train, validation, test)]
        assert row == (f"P{partition['partition']}", *repeats)
    scores = csv(folder + "partition_summary.csv").set_index(["model", "partitions", "metric"])
    for model, original, additional in table(8).itertuples(index=False, name=None):
        for group, cell in (("original", original), ("additional", additional)):
            row = scores.loc[MODELS[model], group, "sweep_accuracy"]
            compare(cell, summary(row, "mean", "sd"), ("S8", model, group))
    concentrations = csv(folder + "per_concentration_errors.csv").set_index(["model", "concentration_mM"])
    for concentration, rf, lsm in table(9).itertuples(index=False, name=None):
        concentration = float(concentration.split()[0])
        for model, cell in (("rf_physical", rf), ("lsm", lsm)):
            row = concentrations.loc[model, concentration]
            assert row.evaluations == 90
            compare(cell, (100 * row.correct / row.evaluations,), ("S9", model, concentration))


def verify_readouts():
    folder = "S07_S12_partition_and_readout/"
    recipes = (
        ("lsm", "majority"), ("lsm_scaled", "majority"), ("hybrid", "majority"),
        ("lsm_rf", "majority"), ("lsm", "score_mean"),
        ("lsm", "validation_temperature_score_mean"), ("lsm", "five_seed_score_ensemble"),
    )
    scores = csv(folder + "gap_bridging_summary.csv").set_index(["model", "variant"])
    for (name, cell), key in zip(table(10).itertuples(index=False, name=None), recipes, strict=True):
        compare(cell, summary(scores.loc[key], "mean", "sd"), ("S10", name))
    nested = csv(folder + "nested_summary.csv").set_index("model")
    order = ("rf_physical_10", "fixed_lsm_10", "nested_lsm_10")
    for (name, cell), model in zip(table(11).itertuples(index=False, name=None), order, strict=True):
        compare(cell, summary(nested.loc[model], "mean", "sd"), ("S11", name))
    loso = csv(folder + "loso_summary.csv").set_index("model")
    for model, majority, average in table(12).itertuples(index=False, name=None):
        row = loso.loc[MODELS[model]]
        compare(majority, summary(row, "mean", "sd_across_seeds"), ("S12", model))
        if model != "Physical-feature RF":
            compare(average, summary(row, "score_mean_mean", "score_mean_sd_across_seeds"), ("S12", model, "mean scores"))
        else:
            assert average == "--"


if __name__ == "__main__":
    verify_ablation()
    verify_search()
    verify_windows()
    verify_partitions()
    verify_readouts()
    print("Verified Tables S1–S12 against the saved experiment records.")
