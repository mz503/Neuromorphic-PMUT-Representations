"""Document reporting scope and condition-specific precision evidence; no model fitting."""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/precision'
dataset=np.load(ROOT/'data/processed/precision/grouped_split_dataset.npz',allow_pickle=False)
metadata=json.loads((ROOT/'data/processed/precision/grouped_split_metadata.json').read_text())
protocol={
    'study':'Single grouped-split, single-seed, window-level precision evaluation',
    'evidence_origin':'Regenerated supporting outputs; archived aggregate accuracies reproduced',
    'task':'Five glucose concentrations: 50, 100, 250, 500, 1000 mM',
    'window_length':128,'stride':128,
    'test_selection':'Repeat 3 of each class; deterministic group holdout',
    'pre_validation_training_windows':460,'fitting_windows':368,'validation_windows':92,'test_windows':230,
    'training_parent_sweeps':sorted(set(dataset['train_groups'].tolist())),
    'validation_parent_sweeps':sorted(set(dataset['val_groups'].tolist())),
    'test_parent_sweeps':sorted(set(dataset['test_groups'].tolist())),
    'validation_split':'Stratified random 80:20 window split of repeats 1 and 2',
    'validation_split_seed':42,'reservoir_seed':42,'number_of_reservoir_seeds':1,
    'validation_role':'Not used to fit the Ridge readout or select a precision',
    'normalization_fitting_scope':'All 460 non-test windows, before the 368/92 split',
    'normalization_mean':metadata['normalization_mean'],'normalization_std':metadata['normalization_std'],
    'train_validation_parent_overlap':True,'test_parent_overlap_with_fitting_or_validation':False,
    'reservoir_neurons':300,'warmup_spectral_samples':10,
    'readout_representation':'Mean post-reset membrane states over the 118 spectral samples remaining after warm-up',
    'readout':'RidgeClassifier, alpha=1.0',
    'readout_retrained_at_every_precision':True,'floating_reference_readout_fitted_separately':True,
    'preprocessing_arithmetic':'Floating point',
    'readout_fit_arithmetic':'Floating point',
    'readout_score_arithmetic':'Floating point, including coefficients, intercept, class scores and probabilities',
    'reduced_precision_simulation':'Quantise/dequantise floating-point simulation of parameters, inputs, currents and membrane updates; not integer-only inference',
    'evaluation_unit':'Individual 128-sample test window','accuracy_denominator':230,
    'comparison_scope':'Within this grouped-split precision study; not the three-fold reservoir comparison',
}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')

# Resolve each split row to a frequency-traceable window identifier.
raw=pd.read_csv(ROOT/'data/processed/high_glucose_90000_samples.csv',float_precision='round_trip')
lookup={}
mean=np.asarray(metadata['normalization_mean'],dtype=np.float32)
std=np.asarray(metadata['normalization_std'],dtype=np.float32)
for sweep,group in raw.groupby('sweep_key',sort=False):
    group=group.sort_values('frequency_hz')
    signal=group[['amplitude','phase','x_v','y_v']].to_numpy(np.float32)
    for n,start in enumerate(range(0,len(group)-127,128)):
        window=(signal[start:start+128]-mean)/std
        key=(sweep,hashlib.sha256(window.tobytes()).hexdigest())
        assert key not in lookup
        lookup[key]=f'{group.sweep_id.iloc[0]}_w{n:03d}'
assignments=[]
for partition in ['train','val','test']:
    for n,(window,group,target) in enumerate(zip(dataset['x_'+partition],dataset[partition+'_groups'],dataset['y_'+partition])):
        key=(str(group),hashlib.sha256(window.tobytes()).hexdigest())
        assignments.append(dict(partition=partition,array_row_zero_based=n,window_id=lookup[key],sweep_key=group,target=int(target)))
assert len(assignments)==690
pd.DataFrame(assignments).to_csv(P/'split_assignments.csv',index=False)

rows=[]
for folder in ['Float_reference','16-bit','12-bit','8-bit','4-bit']:
    d=P/folder;z=np.load(ROOT/'models/precision'/folder/'regenerated_model.npz',allow_pickle=False)
    csv=pd.read_csv(d/'predictions_with_probabilities.csv')
    assert np.array_equal(z['test_targets'],dataset['y_test'])
    assert np.array_equal(z['test_predictions'],csv.prediction.to_numpy())
    assert np.allclose(z['test_probabilities'],csv[[f'probability_class_{i}' for i in range(5)]].to_numpy(),atol=1e-14,rtol=1e-14)
    confusion=np.zeros((5,5),dtype=int)
    for t,pred in zip(z['test_targets'],z['test_predictions']):confusion[int(t),int(pred)]+=1
    correct=int(np.trace(confusion));accuracy=correct/230
    metrics={'origin':'Regenerated, condition-specific supporting output','condition':folder,
             'evaluation_unit':'test window','n_test_windows':230,'n_correct':correct,
             'accuracy':accuracy,'confusion_matrix':confusion.tolist(),'protocol_file':'../protocol.json'}
    (d/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    cfg=json.loads((d/'parameters.json').read_text())
    cfg['protocol']=protocol
    cfg['supporting_outputs']={'model':f'../../../models/precision/{folder}/regenerated_model.npz','predictions':'predictions_with_probabilities.csv',
        'metrics':'metrics.json','dataset':'../../../data/processed/precision/grouped_split_dataset.npz','split_assignment_manifest':'../split_assignments.csv'}
    (d/'parameters.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    rows.append(f'| {folder} | {correct}/230 | {accuracy*100:.2f}% | [{folder}]({folder}/) |')

text='''# Precision-study protocol and supporting outputs

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
'''+ '\n'.join(rows)+'''

Each directory above, including the floating-point reference, has its **own** `parameters.json`, `metrics.json` and `predictions_with_probabilities.csv`, with its fitted `regenerated_model.npz` under `models/precision/<condition>/`. The model export contains that condition's fitted Ridge coefficients/intercept/classes, fitted train/test features, reservoir parameters, preprocessing statistics, labels and test probabilities. Reduced-precision exports additionally include quantised parameter arrays.

These are newly regenerated supporting outputs. All five aggregate accuracy values reproduce the archived precision summary.
'''
(P/'README.md').write_text(text,encoding='utf-8')
