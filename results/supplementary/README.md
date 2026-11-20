# Supplementary experiment records

The CSVs under [tables](tables/) transcribe Tables S1–S14 in the JMM supplementary material. The grouped directories contain the corresponding selection records, split assignments, saved predictions and numerical summaries.

| Tables | Evidence |
| --- | --- |
| S1 | [Channel ablation](S01_channel_ablation/) contains all 135 matched runs and their ordered window predictions. |
| S2–S4 | [CNN search](S02_S04_cnn_search/) contains the validation grid, fold-level means, frozen choices and selected test results. |
| S5–S6 | [Window length and spectrum matching](S05_S06_window_and_matching/) contains the validation and test comparisons, selected lengths and direct-matching sweep predictions. |
| S7–S12 | [Partition and readout comparisons](S07_S12_partition_and_readout/) contains the 18 partitions, split assignments, readout and ensemble results, nested selections and leave-one-sweep-out records. |
| S13–S14 | [Precision results](../precision/) and the [HDL prototype](../../hardware/) support the fixed-point mapping and declared interface. |

Run this command from the repository root.

~~~console
python scripts/verify_supplementary.py
~~~

The check recomputes S1 accuracy from the saved window predictions, verifies disjoint partitions and their repeat assignments in S7, and checks the other reported accuracy statistics against the saved experiment summaries. The archived records support inspection and reaggregation of the supplementary comparisons. The main evaluation command reloads the benchmark models and checks their predictions. It also verifies the five conditions in the precision experiment associated with S13.

For a fresh channel ablation run, use the command in the [repository README](../../README.md). The HDL file associated with S14 is a single-neuron prototype with software test vectors.
