# Dataset and features

The drive voltage is 10 V peak-to-peak.

The original high-glucose workbook is retained unchanged. `AllSamples` contains the primary 15 glucose sweeps and additional air/DI records; only the named glucose sweeps enter the five-class benchmark. The dataset manifest lists each sweep's condition, repeat and source. Separate baseline-liquid measurements support Figure 2 and are excluded from the classifier dataset.

Class IDs are 0: 50 mM, 1: 100 mM, 2: 250 mM, 3: 500 mM, 4: 1000 mM. Channels are amplitude (V), phase (degrees), X (V) and Y (V); the raw frequency column is in Hz. The four-channel loader preserves recorded phase. Per-sweep phase/IQ checks are available in `data/metadata/phase_units_audit.csv`.

The 17 physical predictors are four resonance-frequency/Q/peak-amplitude triplets, phase-transition frequency, phase slope, mean amplitude, peak amplitude and amplitude-frequency area. Exact names, frequency bands and extraction methods are in `configs/feature_parameters.json` and the reusable feature extraction module.

Window length and stride are both 128 samples. Each 6,000-sample sweep contributes 46 full windows (5,888 samples), with 112 incomplete trailing samples excluded. All 690 windows have stable IDs, parent sweeps, frequency bounds and workbook row provenance. In each benchmark fold, one repeat per concentration trains, one validates and one tests: 230 windows in each partition. Z-score mean/SD use only that fold's training windows and are saved in its NPZ.
