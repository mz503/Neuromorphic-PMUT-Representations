# Validation and aggregation

Physical-feature models use three-fold StratifiedKFold over 15 sweep rows, with shuffle=True and random_state=42. The archived comparison imputes medians before CV and scales inside each fold. The separate training-fold-only sensitivity rerun fits both median imputation and scaling within each training fold; it preserves Random Forest accuracy (93.3%) and macro-F1 (93.1%) but changes some other classifier scores. Both evaluations have separately identified fitted models and outputs.

CNN and reservoir benchmarks use the explicit whole-sweep train/validation/test rotation in `data/splits/`. No parent acquisition crosses partitions. Reservoir results use seeds 7, 42, 123, 999 and 2024. CNN's selected manuscript comparison uses 7 and 42; saved additional seeds are tagged separately. The exploratory SpikingCNN study covers fold 0 only, with all five seeds.

Reservoir sweep predictions use the majority of the 46 window predictions; ties choose the smallest class index. Per-sweep class scores are vote counts. Neural archived aggregation averages class probabilities (the original implementation calls this `mean_logits`). Window-level scores are softmax probabilities. Physical classifiers provide predict_proba where available and decision_function scores otherwise. No sweep metric is substituted for window accuracy.

Individual run metrics and confusion matrices are in `results/metrics/`. Fold means average seeds; within-fold seed SD and overall SD across the three fold means use population SD (ddof=0). Fold-0 SpikingCNN variation is seed SD only. The LSM membrane-decay coefficient is fixed at beta=0.9.

The source models are preserved for numerical reproducibility. SpikingCNN uses fresh zero membrane states per forward activation. Nominal E/I reservoir labels do not enforce sign-constrained outgoing weights. The archived reduced-precision collector omits the floating reference's negative membrane clamp, so the precision comparison also changes that update rule.
