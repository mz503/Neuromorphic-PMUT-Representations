"""Reload a transparent regenerated reservoir export and run window inference.

Example:
python -m src.pmut_models.inference models/reservoir/ESN/fold0_seed7 \
    --dataset data/processed/windows/fold0.npz --already-normalized
"""
from pathlib import Path
import argparse
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import softmax
from src.pmut_models.reservoir import Reservoir,LiquidStateMachine

def predict(export_directory,windows,already_normalized=False):
    folder=Path(export_directory)
    meta=json.loads((folder/'parameters.json').read_text())
    z=np.load(folder/'fitted_parameters.npz',allow_pickle=False)
    kind=meta['kind'];params=meta['params'];Class=Reservoir if kind=='ESN' else LiquidStateMachine
    keys=(['n_reservoir','spectral_radius','sparsity','leaking_rate','input_scaling','noise'] if kind=='ESN'
          else ['n_reservoir','beta','threshold','spectral_radius','sparsity','input_scaling','exc_ratio'])
    model=Class(n_input=4,seed=meta['seed'],**{k:v for k,v in params.items() if k in keys})
    model.W_in=z['W_in'];model.W_res=z['W_res'];model.bias=z['bias']
    x=np.asarray(windows,dtype=np.float32)
    if not already_normalized:
        x=(x-z['normalization_mean'].astype(np.float32))/z['normalization_std'].astype(np.float32)
    opts=dict(warmup=params.get('warmup',10),state_reduction=params.get('state_reduction','mean'))
    if kind!='ESN':opts['state_mode']='spikes' if kind=='LSM_spikes' else 'membrane'
    features=model._collect_features(x,**opts)
    scores=features@z['readout_coef'].T+z['readout_intercept']
    probabilities=softmax(scores,axis=1)
    labels=z['readout_classes'][np.argmax(scores,axis=1)]
    return labels,probabilities

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('export_directory');parser.add_argument('--dataset',required=True)
    parser.add_argument('--key',default='x_test');parser.add_argument('--already-normalized',action='store_true')
    args=parser.parse_args();d=np.load(args.dataset,allow_pickle=False)
    pred,prob=predict(args.export_directory,d[args.key],args.already_normalized)
    print(json.dumps(dict(predictions=pred.tolist(),probabilities=prob.tolist())))
