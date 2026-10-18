# Precision-study protocol and supporting outputs

This is a separate **single grouped split, single reservoir seed, window-level** study. It is not the main three-fold comparison and does not measure sweep-level or end-to-end integer-classifier accuracy.

- **Test split:** repeat 3 for each concentration; five held-out physical sweeps, 46 windows each, giving 230 test windows.
- **Fitting/validation:** the 460 windows from repeats 1 and 2 undergo a stratified 80:20 window split with `random_state=42`: 368 fitting windows and 92 validation windows. Their ten parent sweeps overlap across fitting and validation. Neither set shares a parent sweep with the test set. The validation windows are not used for Ridge fitting or precision selection.
- **Seeds:** reservoir seed 42 for the floating-point reference and every reduced-precision condition; internal validation split seed 42. There is one reservoir run per condition, with the same initial reservoir configuration and connectivity.
- **Readout representation:** mean post-reset membrane states, after discarding ten of the 128 spectral samples. The remaining 118 states are averaged; this is a membrane-state readout, not a spike-state readout.
- **Readout training:** a separate floating-point Ridge classifier with alpha 1 is fitted on the 368 fitting windows for the floating-point reference and independently at each precision. Weights from the floating-point readout are not frozen and reused at lower precision.
- **Preprocessing:** z-score calculations remain floating point. Normalisation statistics are fitted on the 460 non-test windows before the internal fitting/validation window split, preserving the source procedure.
- **Readout arithmetic:** Ridge fitting, fitted coefficients, intercepts, class scores and probabilities remain floating point. Reduced-precision reservoir effects are simulated using quantise/dequantise operations in floating-point software. These outputs do not demonstrate integer-only inference or verified hardware.
- **Evaluation unit:** each 128-sample test window; the denominator is 230. No sweep aggregation is used for these accuracy values.

The exact split is saved in `split_assignments.csv`, which maps every array row to its stable window ID and original sweep. Frequency bounds and source workbook row numbers are in `../../data/metadata/window_provenance.csv`. Common configuration is in `protocol.json`; the arrays and normalisation metadata are in `../../data/processed/precision/grouped_split_dataset.npz` and `../../data/processed/precision/grouped_split_metadata.json`. All nine corresponding array fields match the archived grouped NPZ exactly.

## Separate evidence for every condition

| Condition | Correct windows | Accuracy | Output directory |
|---|---:|---:|---|
| Float_reference | 193/230 | 83.91% | [Float_reference](Float_reference/) |
| 16-bit | 195/230 | 84.78% | [16-bit](16-bit/) |
| 12-bit | 193/230 | 83.91% | [12-bit](12-bit/) |
| 8-bit | 173/230 | 75.22% | [8-bit](8-bit/) |
| 4-bit | 165/230 | 71.74% | [4-bit](4-bit/) |

Each directory above, including the floating-point reference, has its **own** `parameters.json`, `metrics.json` and `predictions_with_probabilities.csv`, with its fitted `regenerated_model.npz` under `models/precision/<condition>/`. The model export contains that condition's fitted Ridge coefficients/intercept/classes, fitted train/test features, reservoir parameters, preprocessing statistics, labels and test probabilities. Reduced-precision exports additionally include quantised parameter arrays.

These are newly regenerated supporting outputs. All five aggregate accuracy values reproduce the archived precision summary.
