# data/processed

The selected 17-feature table and full feature extraction table are in `features/`. `windows/fold0.npz`–`fold2.npz` store the training-normalised four-channel arrays, parent sweeps, class labels and training-only normalisation parameters. `high_glucose_90000_samples.csv` retains the raw sample-to-sweep mapping. Windows have length/stride 128: 46 full windows per sweep, 690 total; 112 trailing samples per sweep are excluded. `precision/` holds the separate historical grouped-split dataset, not a main-benchmark fold.
