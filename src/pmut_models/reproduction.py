"""Reproduce the completed PMUT analyses from repository-relative inputs."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT
RAW=REPO/'data/raw'
OUT=ROOT/'results/validation'
sys.path.insert(0,str(REPO))

import numpy as np
import pandas as pd
from src.pmut_models.data import load_task_frame,build_dataset_folds,SIGNAL_COLUMNS

def module(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'src/pmut_models/experiments'/f'{name}.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def load_fold(n):
    d=np.load(ROOT/f'data/processed/windows/fold{n}.npz',allow_pickle=False)
    return {k:d[k] for k in d.files}

def data_stage():
    from src.pmut_models.data import _label_sort_key
    feat=module('nb4_feature_extraction')
    frame=load_task_frame('high-glucose',RAW)
    assert len(frame)==90000 and frame.sweep_key.nunique()==15
    assert all(frame.groupby('sweep_key').size()==6000)
    frame['source_excel_row']=frame.index+2
    frame['source_sheet']='AllSamples'
    frame['source_file']='pmut_rx_tx_Air_Glucose_50mM_to_1000mM.xlsx'
    window_rows=[];sweep_rows=[];feature_rows=[];tidy=[];phase_rows=[]
    for (_,key,label),group in frame.groupby(['source','sweep_key','label'],sort=False):
        g=group.sort_values('frequency_hz').reset_index(drop=True)
        concentration=int(label.split()[1]);repeat=int(key.split('sweep ')[1].split()[0])
        sid=f'g{concentration:04d}_r{repeat:02d}'
        g['sweep_id']=sid;g['sample_index_zero_based']=np.arange(len(g));tidy.append(g)
        freq=g.frequency_hz.to_numpy(float)
        assert np.all(np.diff(freq)>0)
        sweep_rows.append(dict(sweep_id=sid,sweep_key=key,label=label,concentration_mM=concentration,
                               repeat=repeat,n_points=len(g),frequency_min_hz=float(freq[0]),
                               frequency_max_hz=float(freq[-1]),complete_windows=46,
                               retained_points=5888,omitted_trailing_points=112))
        for w,start in enumerate(range(0,len(g)-127,128)):
            stop=start+127
            window_rows.append(dict(window_id=f'{sid}_w{w:03d}',sweep_id=sid,sweep_key=key,label=label,
                                    window_index_zero_based=w,sample_start_zero_based=start,
                                    sample_end_zero_based_inclusive=stop,frequency_start_hz=float(freq[start]),
                                    frequency_end_hz=float(freq[stop]),
                                    source_excel_row_start=int(g.source_excel_row.iloc[start]),
                                    source_excel_row_end=int(g.source_excel_row.iloc[stop])))
        row=feat.extract_features_from_sweep(freq,g.amplitude.to_numpy(float),g.phase.to_numpy(float),
                                             g.x_v.to_numpy(float),g.y_v.to_numpy(float),label,key)
        feature_rows.append(row)
        # Test both phase conventions against measured I/Q; do not alter the saved pipeline.
        amp=g.amplitude.to_numpy(float);phase=g.phase.to_numpy(float)
        x=g.x_v.to_numpy(float);y=g.y_v.to_numpy(float)
        def residual(angle):
            return float(np.sqrt(np.mean((x-amp*np.cos(angle))**2+(y-amp*np.sin(angle))**2))/np.mean(amp))
        phase_rows.append(dict(sweep_id=sid,phase_min=float(phase.min()),phase_max=float(phase.max()),
                               normalized_iq_rmse_if_degrees=residual(np.deg2rad(phase)),
                               normalized_iq_rmse_if_radians=residual(phase)))
    for name,rows in [('sweep_inventory',sweep_rows),('window_provenance',window_rows),('phase_units_audit',phase_rows)]:
        path=ROOT/'data/metadata'/f'{name}.csv';path.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(rows).to_csv(path,index=False)
    pd.concat(tidy,ignore_index=True).to_csv(ROOT/'data/processed/high_glucose_90000_samples.csv',index=False)
    fresh=pd.DataFrame(feature_rows).sort_values(['label','sweep_key']).reset_index(drop=True)
    stored=pd.read_csv(REPO/'data/processed/features/high-glucose_feature_table.csv').sort_values(['label','sweep_key']).reset_index(drop=True)
    predictors=json.loads((REPO/'data/metadata/feature_columns_17.json').read_text())
    checks={}
    for col in predictors:
        aa=fresh[col].to_numpy(float);bb=stored[col].to_numpy(float)
        checks[col]=dict(matches=bool(np.allclose(aa,bb,rtol=1e-10,atol=1e-12,equal_nan=True)),
                         max_abs_difference=float(np.nanmax(np.abs(aa-bb))),missing=int(np.isnan(aa).sum()))
    (ROOT/'data/processed/features').mkdir(exist_ok=True)
    fresh.to_csv(ROOT/'data/processed/features/high_glucose_features_regenerated.csv',index=False)
    assert all(x['matches'] for x in checks.values()),checks
    # Generate source-defined, sweep-separated, training-only-normalised rotation arrays.
    folds=build_dataset_folds('high-glucose',RAW,window_size=128,stride=128)
    all_assignments=[];fold_checks=[]
    windows=pd.DataFrame(window_rows)
    for n,b in enumerate(folds):
        target=ROOT/f'data/processed/windows/fold{n}.npz';target.parent.mkdir(parents=True,exist_ok=True)
        arrays={}
        for partition in ['train','val','test']:
            xx=getattr(b,f'x_{partition}'); yy=getattr(b,f'y_{partition}');gg=getattr(b,f'{partition}_groups')
            assert xx.shape==(230,128,4)
            arrays[f'x_{partition}']=xx;arrays[f'y_{partition}']=yy;arrays[f'{partition}_groups']=gg.astype(str)
            for key in dict.fromkeys(gg):
                for r in windows.loc[windows.sweep_key==key].to_dict('records'):
                    all_assignments.append(dict(fold=n,partition=partition,**r))
        assert not set(b.train_groups)&set(b.val_groups)
        assert not set(b.train_groups)&set(b.test_groups)
        assert not set(b.val_groups)&set(b.test_groups)
        arrays['normalization_mean']=np.array(b.metadata['normalization_mean'])
        arrays['normalization_std']=np.array(b.metadata['normalization_std'])
        arrays['labels']=np.array(b.labels);arrays['channels']=np.array(b.channels)
        np.savez_compressed(target,**arrays)
        write_json(ROOT/f'data/processed/windows/fold{n}_metadata.json',b.metadata)
        recorded=json.loads((REPO/f'results/archive/runs/high-glucose_folds_w128/cnn1d/fold{n}/metrics_summary.json').read_text())['dataset']
        split_match=all(recorded[f'{part}_sweeps']==b.metadata[f'{part}_sweeps'] for part in ['train','val','test'])
        norm_match=all(np.array_equal(recorded[f'normalization_{key}'],b.metadata[f'normalization_{key}']) for key in ['mean','std'])
        assert split_match and norm_match
        fold_checks.append(dict(fold=n,split_matches_recorded=split_match,normalization_exactly_matches_recorded=norm_match))
    pd.DataFrame(all_assignments).to_csv(ROOT/'data/splits/window_fold_assignments.csv',index=False)
    write_json(OUT/'data_validation.json',dict(raw_samples=len(frame),physical_sweeps=15,windows=690,
                                               assignments=len(all_assignments),feature_checks=checks,fold_checks=fold_checks,
                                               phase_units='See phase_units_audit.csv; original source field is phase_deg.'))
    print('DATA: 90,000 samples; 15 sweeps; 690 windows; 3 disjoint rotation datasets; all 17 features match archived values.',flush=True)

def feature_models_stage():
    import joblib
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score,f1_score
    nb=module('nb5_feature_ml_baselines')
    df=pd.read_csv(REPO/'data/processed/features/high-glucose_feature_table.csv')
    xx=df[nb.FEATURE_COLS];classes=sorted(df.label.unique(),key=nb._label_sort_key)
    yy=np.array([classes.index(x) for x in df.label])
    medians=xx.median();cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    stored=pd.read_csv(REPO/'results/archive/feature_ml/high-glucose_baseline_summary.csv').set_index('model')
    results=[];assignments=[]
    for fold,(train,test) in enumerate(cv.split(xx,yy)):
        for i in test:assignments.append(dict(fold=fold,sweep_key=df.sweep_key.iloc[i],partition='test'))
        for i in train:assignments.append(dict(fold=fold,sweep_key=df.sweep_key.iloc[i],partition='train'))
    for mode in ['legacy_full_table_imputation','training_fold_only_imputation']:
        for name,estimator in nb.MODELS.items():
            pred=np.full(len(yy),-1);fold_results=[]
            for fold,(train,test) in enumerate(cv.split(xx,yy)):
                if mode.startswith('legacy'):
                    fit_x=xx.fillna(medians).to_numpy(float)
                    pipe=make_pipeline(StandardScaler(),clone(estimator))
                else:
                    fit_x=xx.to_numpy(float)
                    pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),clone(estimator))
                pipe.fit(fit_x[train],yy[train]);pred[test]=pipe.predict(fit_x[test])
                out=ROOT/'models/features'/mode/name.replace(' ','_').replace('(','').replace(')','')/f'fold{fold}'
                out.mkdir(parents=True,exist_ok=True)
                joblib.dump(dict(pipeline=pipe,predictors=nb.FEATURE_COLS,labels=classes,
                                 external_full_table_medians=medians.to_dict() if mode.startswith('legacy') else None),out/'fitted_model.joblib')
                write_json(out/'provenance.json',dict(regenerated=True,mode=mode,fold=fold,
                    train_sweeps=df.sweep_key.iloc[train].tolist(),test_sweeps=df.sweep_key.iloc[test].tolist(),
                    test_accuracy=float(np.mean(pred[test]==yy[test]))))
                fold_results.append(float(np.mean(pred[test]==yy[test])))
            acc=float(accuracy_score(yy,pred));f1=float(f1_score(yy,pred,average='macro'))
            matches=bool(np.isclose(acc,stored.loc[name,'accuracy']) and np.isclose(f1,stored.loc[name,'f1_macro']))
            results.append(dict(mode=mode,model=name,accuracy=acc,macro_f1=f1,
                                matches_archived_metrics=matches,fold_accuracies=fold_results))
            pd.DataFrame(dict(sweep_key=df.sweep_key,target=yy,prediction=pred)).to_csv(
                ROOT/'models/features'/mode/(name.replace(' ','_').replace('(','').replace(')','')+'_predictions.csv'),index=False)
            print(f'FEATURE: {mode}: {name}: {acc:.6f}; archive match={matches}',flush=True)
    pd.DataFrame(assignments).to_csv(ROOT/'data/splits/physical_feature_cv_assignments.csv',index=False)
    write_json(OUT/'feature_model_validation.json',results)
    assert all(r['matches_archived_metrics'] for r in results if r['mode'].startswith('legacy'))

def reservoir_models_stage():
    from sklearn.linear_model import RidgeClassifier
    from scipy.special import softmax
    from src.pmut_models.reservoir import Reservoir,LiquidStateMachine
    seeds=[7,42,123,999,2024];checks=[]
    for fold in range(3):
        d=load_fold(fold)
        base=REPO/f'results/archive/reservoir/high-glucose/grouped_w128/fold{fold}'
        for kind,folder,Class in [('ESN',base,Reservoir),('LSM_membrane',base/'lsm',LiquidStateMachine),
                                  ('LSM_spikes',base/'lsm',LiquidStateMachine)]:
            state_mode='spikes' if kind=='LSM_spikes' else 'membrane'
            if kind=='LSM_spikes':
                targets=[json.loads((folder/'metrics_spikes.json').read_text())];run_seeds=[42]
            else:
                targets=json.loads((folder/'metrics_summary.json').read_text())['individual_results'];run_seeds=seeds
            for seed,archived in zip(run_seeds,targets):
                params=archived['params'].copy()
                init={k:v for k,v in params.items() if k in (['n_reservoir','spectral_radius','sparsity','leaking_rate','input_scaling','noise']
                    if kind=='ESN' else ['n_reservoir','beta','threshold','spectral_radius','sparsity','input_scaling','exc_ratio'])}
                model=Class(n_input=4,seed=seed,**init)
                collect=dict(warmup=params.get('warmup',10),state_reduction=params.get('state_reduction','mean'))
                if kind!='ESN':collect['state_mode']=state_mode
                train_features=model._collect_features(d['x_train'],**collect)
                test_features=model._collect_features(d['x_test'],**collect)
                readout=RidgeClassifier(alpha=params.get('alpha',1)).fit(train_features,d['y_train'])
                predictions=readout.predict(test_features);probs=softmax(readout.decision_function(test_features),axis=1)
                oldprobs=np.asarray(archived['probabilities']);oldpred=np.asarray(archived['predictions'])
                maxdiff=float(np.max(np.abs(probs-oldprobs)))
                matched=bool(np.array_equal(predictions,oldpred) and np.allclose(probs,oldprobs,rtol=1e-7,atol=1e-8))
                out=ROOT/f'models/reservoir/{kind}/fold{fold}_seed{seed}';out.mkdir(parents=True,exist_ok=True)
                arrays=dict(W_in=model.W_in,W_res=model.W_res,bias=model.bias,
                            readout_coef=readout.coef_,readout_intercept=readout.intercept_,readout_classes=readout.classes_,
                            normalization_mean=d['normalization_mean'],normalization_std=d['normalization_std'],
                            labels=d['labels'],test_probabilities=probs,test_predictions=predictions,
                            test_targets=d['y_test'],test_groups=d['test_groups'])
                if hasattr(model,'neuron_type'):arrays['neuron_type']=model.neuron_type
                np.savez_compressed(out/'fitted_parameters.npz',**arrays)
                write_json(out/'parameters.json',dict(kind=kind,fold=fold,seed=seed,regenerated=True,
                                                      params=params,archived_predictions_and_probabilities_match=matched,
                                                      max_probability_difference=maxdiff))
                checks.append(dict(model=kind,fold=fold,seed=seed,matches_archived=matched,
                                   max_probability_difference=maxdiff,accuracy=float(np.mean(predictions==d['y_test']))))
                write_json(OUT/'reservoir_model_validation.json',checks)
                print(f'RESERVOIR: {kind} fold={fold} seed={seed}; archive match={matched}; max probability delta={maxdiff:.3g}',flush=True)
    assert all(x['matches_archived'] for x in checks),'Regenerated reservoir outputs differ from archived values; inspect validation JSON.'

def neural_validation_stage():
    import torch
    from src.pmut_models.spiking import build_model
    torch.set_num_threads(1)
    checks=[]
    for name,prefix in [('cnn1d','results/archive/runs'),('spiking_cnn1d','results/archive/exploratory/runs')]:
        for fold in ([0] if name=='spiking_cnn1d' else range(3)):
            d=load_fold(fold)
            for seed in [7,42,123,999,2024]:
                run=f'high-glucose_folds_w128/{name}/fold{fold}_seed{seed}'
                model=build_model(name,input_shape=(128,4),num_classes=5)
                checkpoint=REPO/'models'/run/'best_model.pt'
                model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
                model.eval();probabilities=[]
                with torch.inference_mode():
                    for start in range(0,230,32):
                        logits=model(torch.from_numpy(d['x_test'][start:start+32]))
                        probabilities.append(torch.softmax(logits,dim=1).numpy())
                probs=np.concatenate(probabilities);pred=probs.argmax(axis=1)
                saved=pd.read_csv(REPO/prefix/run/'predictions.csv')
                assert np.array_equal(d['y_test'],saved.target.to_numpy())
                matched=bool(np.array_equal(pred,saved.prediction.to_numpy()))
                out=ROOT/'results/neural_inference'/run;out.mkdir(parents=True,exist_ok=True)
                rows=pd.DataFrame(dict(sweep_key=d['test_groups'],target=d['y_test'],prediction=pred))
                for c in range(5):rows[f'probability_class_{c}']=probs[:,c]
                rows.to_csv(out/'predictions_with_probabilities.csv',index=False)
                checks.append(dict(model=name,fold=fold,seed=seed,predictions_match_archived=matched,
                                    accuracy=float(np.mean(pred==d['y_test']))))
                print(f'NEURAL: {name} fold={fold} seed={seed}; checkpoint predictions match={matched}',flush=True)
                write_json(OUT/'neural_checkpoint_validation.json',checks)
    assert all(x['predictions_match_archived'] for x in checks)

def precision_stage():
    from sklearn.linear_model import RidgeClassifier
    from scipy.special import softmax
    from src.pmut_models.data import build_dataset
    from src.pmut_models.reservoir import LiquidStateMachine
    from src.pmut_models.fixed_point import STANDARD_FORMATS
    nb=module('nb14_fixed_point_pmut')
    bundle=build_dataset('high-glucose',data_root=RAW,split='grouped',window_size=128,stride=128)
    out=ROOT/'results/precision';out.mkdir(parents=True,exist_ok=True)
    write_json(ROOT/'data/processed/precision/grouped_split_metadata.json',bundle.metadata)
    np.savez_compressed(ROOT/'data/processed/precision/grouped_split_dataset.npz',
        x_train=bundle.x_train,y_train=bundle.y_train,train_groups=bundle.train_groups.astype(str),
        x_val=bundle.x_val,y_val=bundle.y_val,val_groups=bundle.val_groups.astype(str),
        x_test=bundle.x_test,y_test=bundle.y_test,test_groups=bundle.test_groups.astype(str))
    archived_dataset=np.load(REPO/'data/processed/precision/archived_grouped_w128.npz',allow_pickle=False)
    regenerated_dataset=np.load(ROOT/'data/processed/precision/grouped_split_dataset.npz',allow_pickle=False)
    dataset_checks={k:bool(np.array_equal(archived_dataset[k],regenerated_dataset[k])) for k in regenerated_dataset.files}
    assert all(dataset_checks.values())
    write_json(out/'dataset_matches_archived_npz.json',dataset_checks)
    archived=json.loads((REPO/'results/archive/fixed_point/high-glucose/bit_width_study.json').read_text())
    config=dict(n_input=4,n_reservoir=300,beta=0.9,threshold=1.0,spectral_radius=0.95,sparsity=0.9,input_scaling=1.0,seed=42)
    model=LiquidStateMachine(**config);checks=[]
    for i,name in enumerate(['Float (reference)','16-bit','12-bit','8-bit','4-bit']):
        target=out/name.replace(' ','_').replace('(','').replace(')','');target.mkdir(exist_ok=True)
        if i==0:
            opts=dict(warmup=10,state_mode='membrane',state_reduction='mean')
            ff=model._collect_features(bundle.x_train,**opts);tt=model._collect_features(bundle.x_test,**opts)
            q=None
        else:
            q=STANDARD_FORMATS[name]
            ff=nb.collect_quantized_features(bundle.x_train,model,q,warmup=10)
            tt=nb.collect_quantized_features(bundle.x_test,model,q,warmup=10)
        readout=RidgeClassifier(alpha=1.0).fit(ff,bundle.y_train)
        pred=readout.predict(tt);probs=softmax(readout.decision_function(tt),axis=1)
        acc=float(np.mean(pred==bundle.y_test));expected=archived[i]['accuracy']
        matched=bool(np.isclose(acc,expected))
        arrays=dict(W_in=model.W_in,W_res=model.W_res,bias=model.bias,
                    readout_coef=readout.coef_,readout_intercept=readout.intercept_,readout_classes=readout.classes_,
                    train_features=ff,test_features=tt,test_predictions=pred,test_probabilities=probs,
                    test_targets=bundle.y_test,test_groups=bundle.test_groups.astype(str),
                    normalization_mean=np.array(bundle.metadata['normalization_mean']),
                    normalization_std=np.array(bundle.metadata['normalization_std']))
        if q is not None:arrays.update(nb.quantize_lsm_weights(model,q))
        model_folder=ROOT/'models/precision'/target.name;model_folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(model_folder/'regenerated_model.npz',**arrays)
        write_json(target/'parameters.json',dict(regenerated=True,configuration=config,
            format='float' if q is None else repr(q),warmup=10,state_mode='membrane',state_reduction='mean',
            readout='independently fitted floating-point Ridge, alpha=1',accuracy=acc,archived_accuracy=expected,
            accuracy_matches_archive=matched,
            evidence_origin='Regenerated condition-specific supporting outputs'))
        rows=pd.DataFrame(dict(sweep_key=bundle.test_groups,target=bundle.y_test,prediction=pred))
        for c in range(5):rows[f'probability_class_{c}']=probs[:,c]
        rows.to_csv(target/'predictions_with_probabilities.csv',index=False)
        checks.append(dict(format=name,regenerated_accuracy=acc,archived_accuracy=expected,matches=matched))
        write_json(OUT/'precision_regeneration_validation.json',checks)
        print(f'PRECISION: {name}; accuracy={acc:.6f}; archived accuracy match={matched}',flush=True)
    assert all(x['matches'] for x in checks),'Regenerated precision study differs from archived summary.'
    import runpy
    runpy.run_path(str(ROOT/'scripts/document_precision.py'),run_name='__main__')

def reload_validation_stage():
    import joblib
    from src.pmut_models.inference import predict
    checks=[]
    for folder in sorted((ROOT/'models/reservoir').glob('*/*')):
        meta=json.loads((folder/'parameters.json').read_text())
        d=load_fold(meta['fold']);z=np.load(folder/'fitted_parameters.npz',allow_pickle=False)
        pred,probs=predict(folder,d['x_test'],already_normalized=True)
        matched=bool(np.array_equal(pred,z['test_predictions']) and np.allclose(probs,z['test_probabilities'],atol=1e-9,rtol=1e-7))
        checks.append(dict(artifact=str(folder.relative_to(ROOT)),reload_predictions_match=matched))
    features=pd.read_csv(REPO/'data/processed/features/high-glucose_feature_table.csv').set_index('sweep_key')
    for path in sorted((ROOT/'models/features').glob('*/*/fold*/fitted_model.joblib')):
        bundle=joblib.load(path)
        meta=json.loads((path.parent/'provenance.json').read_text())
        xx=features.loc[meta['test_sweeps'],bundle['predictors']].copy()
        if bundle['external_full_table_medians'] is not None:xx=xx.fillna(bundle['external_full_table_medians'])
        pred=bundle['pipeline'].predict(xx.to_numpy(float))
        expected=pd.read_csv(path.parents[1].parent/(path.parents[1].name+'_predictions.csv')).set_index('sweep_key')
        matched=bool(np.array_equal(pred,expected.loc[meta['test_sweeps'],'prediction'].to_numpy()))
        checks.append(dict(artifact=str(path.relative_to(ROOT)),reload_predictions_match=matched))
    assert len(checks)>=69 and all(x['reload_predictions_match'] for x in checks)
    write_json(OUT/'reloaded_model_validation.json',checks)
    print(f'RELOAD: all {len(checks)} saved feature/reservoir exports reproduce their predictions.',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['data','features','reservoir','neural','precision','reload'],default='data')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    {'data':data_stage,'features':feature_models_stage,'reservoir':reservoir_models_stage,
     'neural':neural_validation_stage,'precision':precision_stage,'reload':reload_validation_stage}[args.stage]()

if __name__=='__main__':main()
