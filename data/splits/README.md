# data/splits

`window_rotation_sweeps.csv` and `window_fold_assignments.csv` describe the 230/230/230 training/validation/test rotations with whole-sweep separation. `physical_feature_cv_assignments.csv` records the independent three-fold stratified physical-feature CV. `precision_window_assignments.csv` records the separate precision-study protocol; its fitting/validation windows share non-test parent sweeps, as documented in `docs/PRECISION.md`. Do not replace the benchmark splits with the precision split.
