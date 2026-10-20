# Neuromorphic PMUT representations

Code, measurements and fitted models for *Physical and Neuromorphic Representations of PMUT Electrical Spectra for Glucose Loading Classification*.

The primary dataset comprises frequency sweeps at 50, 100, 250, 500 and 1000 mM glucose concentrations. Each sweep has 6,000 samples with frequency, amplitude, phase, X and Y. Four signal channels form non-overlapping 128-sample windows; frequency is retained as provenance. Complete acquisitions are separated in the main benchmark folds.

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
| `notebooks/` | Numbered executable workflow from inspection to figures |
| `scripts/` | Data preparation, training, evaluation and reporting entry points |
| `models/` | Original neural checkpoints and regenerated fitted feature/reservoir/precision models |
| `results/` | Predictions, metrics, confusion matrices, logs and numerical tables |
| `figures/` | Final manuscript panels, regenerated plots and underlying numerical data |
| `hardware/` | Single-neuron Q3.12 Verilog prototype and existing software test vectors |
| `docs/` | Dataset/protocol definitions and results index |

## Reproduction

```powershell
python scripts/prepare_data.py
python scripts/evaluate.py
python scripts/report.py
python scripts/run_notebooks.py
```

The first command regenerates features, window datasets and benchmark split provenance from the original workbook. Evaluation reloads the saved models and checks their supporting predictions. Reporting regenerates tables, confusion matrices and Figures 4–7, using majority-vote reservoir aggregation; tied votes select the smallest class index.

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

New neural training runs are written under `results/new_runs/`; selected original checkpoints are retained. CNN1D's selected manuscript comparison uses seeds 7 and 42 over three folds; other saved CNN seeds remain available. Reservoir comparisons use five seeds per fold. SpikingCNN1D is exploratory and reported for fold 0 only.

The precision comparison has its own grouped split and window-level evaluation. Its Ridge readout is independently retrained at each precision, including the float reference; preprocessing and readout fitting/scoring remain floating point. See [the exact precision protocol](docs/PRECISION.md).

## Licensing and citation

 If you use or build upon this work, please cite the associated publication(s) and this repository, including the release or commit used. Third-party materials remain subject to their original licences and attribution requirements.
