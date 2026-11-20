# Dataset and features

The drive voltage is 10 V peak-to-peak.

The original workbook's `AllSamples` sheet contains 126,000 samples across 22 recorded sweeps under seven conditions. The five-class glucose benchmark uses 90,000 samples from 15 sweeps, with three acquisitions at each of five concentrations. Every benchmark acquisition contains 6,000 samples. The manifest gives each benchmark sweep's condition, repeat and source. Additional baseline-liquid recordings are retained under `data/raw/baseline_liquids/`.

Class IDs are 0: 50 mM, 1: 100 mM, 2: 250 mM, 3: 500 mM, 4: 1000 mM. Channels are amplitude (V), phase (degrees), X (V) and Y (V); the raw frequency column is in Hz. The four-channel loader preserves recorded phase. Per-sweep phase/IQ checks are available in `data/metadata/phase_units_audit.csv`.

The 17 physical predictors are four resonance-frequency/Q/peak-amplitude triplets, phase-transition frequency, phase slope, mean amplitude, peak amplitude and amplitude-frequency area. Exact names, frequency bands and extraction methods are in `configs/feature_parameters.json` and the reusable feature extraction module.

Window length and stride are both 128 samples. Each 6,000-sample sweep contributes 46 full windows (5,888 samples), with 112 incomplete trailing samples excluded. All 690 windows have stable IDs, parent sweeps, frequency bounds and workbook row provenance. In each benchmark fold, one repeat per concentration trains, one validates and one tests: 230 windows in each partition. Z-score mean/SD use only that fold's training windows and are saved in its NPZ.
