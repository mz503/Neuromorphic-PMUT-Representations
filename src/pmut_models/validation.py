"""Reload saved models and check their stored supporting outputs."""
from pathlib import Path
import json
import hashlib
import numpy as np
from .reproduction import neural_validation_stage, reload_validation_stage, module
from .reservoir import LiquidStateMachine
from .fixed_point import STANDARD_FORMATS

ROOT=Path(__file__).resolve().parents[2]

def validate_precision():
    d=np.load(ROOT/'data/processed/precision/grouped_split_dataset.npz',allow_pickle=False)
    nb=module('nb14_fixed_point_pmut');checks=[]
    for folder in sorted((ROOT/'models/precision').iterdir()):
        z=np.load(folder/'regenerated_model.npz',allow_pickle=False)
        config=json.loads((ROOT/'results/precision'/folder.name/'parameters.json').read_text(encoding='utf-8'))['configuration']
        model=LiquidStateMachine(**config);model.W_in=z['W_in'];model.W_res=z['W_res'];model.bias=z['bias']
        if folder.name=='Float_reference':
            features=model._collect_features(d['x_test'],warmup=10,state_mode='membrane',state_reduction='mean')
        else:features=nb.collect_quantized_features(d['x_test'],model,STANDARD_FORMATS[folder.name],warmup=10)
        scores=features@z['readout_coef'].T+z['readout_intercept']
        pred=z['readout_classes'][scores.argmax(axis=1)]
        assert np.allclose(features,z['test_features'],rtol=1e-8,atol=1e-9),folder.name
        assert np.array_equal(pred,z['test_predictions']),folder.name
        checks.append(dict(condition=folder.name,predictions_match=True,accuracy=float(np.mean(pred==d['y_test']))))
    (ROOT/'results/validation/precision_reload.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Reloaded all five precision-specific models and reproduced their test predictions.',flush=True)
    return checks

def validate_splits():
    import pandas as pd
    m=pd.read_csv(ROOT/'data/splits/window_fold_assignments.csv')
    assert len(m)==2070 and m.window_id.nunique()==690
    for fold,g in m.groupby('fold'):
        assert (g.groupby('sweep_id').partition.nunique()==1).all()
        assert g.groupby('partition').size().to_dict()=={'test':230,'train':230,'val':230}
        d=np.load(ROOT/f'data/processed/windows/fold{fold}.npz',allow_pickle=False)
        for part in ['train','val','test']:
            groups=set(d[part+'_groups'])
            assert groups==set(g.loc[g.partition==part,'sweep_key'])
    print('All three benchmark folds keep complete parent sweeps together.',flush=True)

def main():
    validate_splits();reload_validation_stage();neural_validation_stage();validate_precision()

if __name__=='__main__':main()
