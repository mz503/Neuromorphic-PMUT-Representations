# Neuromorphic PMUT representations

Code, measurements and fitted models for *Physical and Neuromorphic Representations of PMUT Electrical Spectra for Glucose Loading Classification*, submitted to the Journal of Micromechanics and Microengineering (IOP Publishing)..

The workbook contains 126,000 samples across 22 recorded sweeps. The five-class benchmark uses 90,000 samples from 15 glucose sweeps, with three acquisitions at each of 50, 100, 250, 500 and 1000 mM. Each benchmark sweep has 6,000 samples with frequency, amplitude, phase, X and Y. Four signal channels form non-overlapping 128-sample windows; frequency is retained as provenance. Complete acquisitions are separated in the main benchmark folds.

## Installation

Use Python 3.13 and the pinned scientific environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
```

Activate the environment or use its Python executable for the commands below. The pinned versions describe the environment used to regenerate and validate the supplied exports; they are not asserted to be the historical training environment.

## Repository structure

| Folder | Contents |
|---|---|
| `data/raw/` | Original glucose workbook and separate baseline-liquid acquisitions |
| `data/metadata/` | Dataset manifest, channels, acquisition settings, labels and window provenance |
| `data/processed/` | 17-feature table and processed four-channel windows |
| `data/splits/` | Exact physical-feature CV, sweep-rotation and precision manifests |
| `src/` | Reusable data, model, training, inference, aggregation and evaluation code |
| `configs/` | Recorded experiment settings, seeds, feature bands and precision protocol |
| `notebooks/` | Numbered executable workflow for data preparation, evaluation and reporting |
| `scripts/` | Data preparation, training, evaluation and reporting entry points |
| `models/` | Original neural checkpoints and regenerated fitted feature/reservoir/precision models |
| `results/` | Predictions, metrics, confusion matrices, logs, and archived evidence for [Tables S1–S14](results/supplementary/) |
| `hardware/` | Single-neuron Q3.12 Verilog prototype and existing software test vectors |
| `docs/` | Dataset/protocol definitions and results index |

## Reproduction

```powershell
python scripts/prepare_data.py
python scripts/evaluate.py
python scripts/report.py
python scripts/run_notebooks.py
python scripts/verify_supplementary.py
```

The first command regenerates features, window datasets and benchmark split provenance from the original workbook. Evaluation reloads the saved models and checks their supporting predictions. Reporting regenerates numerical tables and confusion matrices. The [cross-model comparison](results/tables/cross_model_comparison.csv) includes SpikingCNN1D across all three folds. Reservoir sweep results use majority voting, with ties resolved by the smallest class index.

To regenerate fitted models:

```powershell
python scripts/train.py --family features
python scripts/train.py --family reservoir
python scripts/train.py --family precision
```

To retrain a neural model using its recorded settings:

```powershell
python scripts/train.py --family cnn1d --fold 0 --seed 7
python scripts/train.py --family spiking_cnn1d --fold 0 --seed 7
```

New neural training runs are written under `results/new_runs/`; selected original checkpoints are retained. CNN1D's selected manuscript comparison uses seeds 7 and 42 over three folds; other saved CNN seeds remain available. Reservoir comparisons use five seeds per fold. SpikingCNN1D uses five seeds in each of all three sweep folds. Its mean window accuracy is 78.9%, with a population SD of 4.1 percentage points across fold means.

The precision comparison has its own grouped split and window-level evaluation. Its Ridge readout is independently retrained at each precision, including the float reference; preprocessing and readout fitting/scoring remain floating point. See [the exact precision protocol](docs/PRECISION.md).

## Supplementary experiments

The [supplementary results](results/supplementary/) include each published table, the archived run records, and the split and selection records. `python scripts/verify_supplementary.py` checks the reported statistics in Tables S1–S12 against this evidence. To rerun the matched three-fold channel ablation, use `python scripts/channel_ablation.py --output results/new_runs/channel_ablation`. The additional CNN search, window-length and readout comparisons are supplied as archived runs and table evidence. The fixed-point and HDL tables are linked to the [precision outputs](results/precision/) and [prototype](hardware/).

## Licensing and citation

 If you use or build upon this work, please cite the associated publication(s) and this repository, including the release or commit used. Third-party materials remain subject to their original licences and attribution requirements.
