"""Export identified predictions, evaluation tables and reproducible figure data."""
from pathlib import Path
import json
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT=Path(__file__).resolve().parents[2]
SEEDS=[7,42,123,999,2024]

def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def complete_spike_exports():
    """Fit the missing spike readouts for the manuscript's recorded seed set."""
    from sklearn.linear_model import RidgeClassifier
    from scipy.special import softmax
    from .reservoir import LiquidStateMachine
    for fold in range(3):
        d=np.load(ROOT/f'data/processed/windows/fold{fold}.npz',allow_pickle=False)
        for seed in SEEDS:
            out=ROOT/f'models/reservoir/LSM_spikes/fold{fold}_seed{seed}'
            if (out/'fitted_parameters.npz').exists():continue
            base=ROOT/f'models/reservoir/LSM_membrane/fold{fold}_seed{seed}'
            z=np.load(base/'fitted_parameters.npz',allow_pickle=False)
            meta=json.loads((base/'parameters.json').read_text(encoding='utf-8'))
            m=LiquidStateMachine(n_input=4,seed=seed)
            m.W_in=z['W_in'];m.W_res=z['W_res'];m.bias=z['bias']
            opts=dict(warmup=10,state_mode='spikes',state_reduction='mean')
            train=m._collect_features(d['x_train'],**opts);test=m._collect_features(d['x_test'],**opts)
            r=RidgeClassifier(alpha=1).fit(train,d['y_train'])
            pred=r.predict(test);probs=softmax(r.decision_function(test),axis=1)
            out.mkdir(parents=True,exist_ok=True)
            arrays={k:z[k] for k in ['W_in','W_res','bias','normalization_mean','normalization_std','labels','neuron_type'] if k in z}
            arrays.update(readout_coef=r.coef_,readout_intercept=r.intercept_,readout_classes=r.classes_,test_predictions=pred,test_probabilities=probs,test_targets=d['y_test'],test_groups=d['test_groups'])
            np.savez_compressed(out/'fitted_parameters.npz',**arrays)
            save_json(out/'parameters.json',dict(kind='LSM_spikes',fold=fold,seed=seed,regenerated=True,params=meta['params'],origin='Regenerated spike readout using the recorded membrane-reservoir connectivity; not a recovered historical fitted model'))
        print(f'Prepared five-seed spike readouts for fold {fold}.',flush=True)

def collect_results():
    import joblib
    inventory=pd.read_csv(ROOT/'data/metadata/dataset_manifest.csv').set_index('sweep_key')
    provenance=pd.read_csv(ROOT/'data/metadata/window_provenance.csv')
    lookup={key:g.sort_values('window_index_zero_based').to_dict('records') for key,g in provenance.groupby('sweep_key')}
    labelmap={r['class_id']:r['label'] for r in json.loads((ROOT/'data/metadata/class_label_mapping.json').read_text(encoding='utf-8'))}
    all_rows=[];metrics=[]

    def emit(model,fold,seed,pred,target,groups,probabilities,unit='window',aggregation='none',variant='main',precision='float',origin='saved fitted-model inference'):
        counts={};local=[]
        for i,(p,t,key) in enumerate(zip(pred,target,groups)):
            key=str(key);n=counts.get(key,0);counts[key]=n+1
            row=dict(model=model,variant=variant,fold=fold,seed=seed,precision=precision,evaluation_unit=unit,aggregation=aggregation,sweep_key=key,sweep_id=inventory.loc[key,'sweep_id'],target=int(t),prediction=int(p),origin=origin)
            row.update(target_label=labelmap[int(t)],prediction_label=labelmap[int(p)])
            if unit=='window':
                evidence=lookup[key][n]
                row.update({k:evidence[k] for k in ['window_id','frequency_start_hz','frequency_end_hz']})
            if probabilities is not None:
                for c in range(5):row[f'class_score_{c}']=float(probabilities[i,c])
            local.append(row)
        all_rows.extend(local)
        cm=confusion_matrix(target,pred,labels=range(5))
        record=dict(model=model,variant=variant,fold=fold,seed=seed,precision=precision,evaluation_unit=unit,aggregation=aggregation,n=len(pred),accuracy=float(accuracy_score(target,pred)),macro_f1=float(f1_score(target,pred,average='macro',zero_division=0)),confusion_matrix=cm.tolist())
        metrics.append(record)

    def windows_and_sweeps(model,fold,seed,pred,target,groups,probs,aggregation='majority_vote',variant='main'):
        emit(model,fold,seed,pred,target,groups,probs,variant=variant)
        pp=[];tt=[];gg=[];scores=[]
        for key in sorted(set(groups)):
            ix=np.flatnonzero(groups==key)
            vote=np.bincount(np.asarray(pred)[ix].astype(int),minlength=5)
            score=vote if aggregation=='majority_vote' else np.asarray(probs)[ix].mean(axis=0)
            pp.append(int(np.argmax(score)));tt.append(int(np.asarray(target)[ix[0]]));gg.append(key);scores.append(score)
        emit(model,fold,seed,pp,tt,gg,np.asarray(scores),unit='sweep',aggregation=aggregation,variant=variant)

    for folder in sorted((ROOT/'models/reservoir').glob('*/*')):
        meta=json.loads((folder/'parameters.json').read_text(encoding='utf-8'));z=np.load(folder/'fitted_parameters.npz',allow_pickle=False)
        windows_and_sweeps(meta['kind'],meta['fold'],meta['seed'],z['test_predictions'],z['test_targets'],z['test_groups'],z['test_probabilities'])
    for path in sorted((ROOT/'results/neural_inference').glob('*/*/*/predictions_with_probabilities.csv')):
        run=path.parent.name;fold=int(run.split('_')[0][4:]);seed=int(run.split('seed')[1]);model=path.parents[1].name
        d=pd.read_csv(path);probs=d[[f'probability_class_{c}' for c in range(5)]].to_numpy()
        variant='selected' if (model=='cnn1d' and seed in [7,42]) or model=='spiking_cnn1d' else 'additional_saved_seed'
        windows_and_sweeps(model,fold,seed,d.prediction.to_numpy(),d.target.to_numpy(),d.sweep_key.to_numpy(),probs,aggregation='mean_probability',variant=variant)
    for folder in sorted((ROOT/'models/features').glob('*/*/fold*')):
        bundle=joblib.load(folder/'fitted_model.joblib');meta=json.loads((folder/'provenance.json').read_text(encoding='utf-8'))
        frame=pd.read_csv(ROOT/'data/processed/features/high-glucose_feature_table.csv').set_index('sweep_key')
        groups=np.asarray(meta['test_sweeps']);xx=frame.loc[groups,bundle['predictors']]
        if bundle['external_full_table_medians'] is not None:xx=xx.fillna(bundle['external_full_table_medians'])
        pipe=bundle['pipeline'];x=xx.to_numpy(float);pred=pipe.predict(x)
        if hasattr(pipe,'predict_proba'):scores=pipe.predict_proba(x);kind='predict_proba'
        else:scores=pipe.decision_function(x);kind='decision_function'
        target=np.asarray([bundle['labels'].index(x) for x in frame.loc[groups,'label']])
        emit(folder.parent.name,meta['fold'],42,pred,target,groups,scores,unit='sweep',variant=meta['mode'],origin=f'Regenerated fitted classifier; class scores: {kind}')
    assignments=pd.read_csv(ROOT/'data/splits/precision_window_assignments.csv')
    for folder in sorted((ROOT/'models/precision').iterdir()):
        z=np.load(folder/'regenerated_model.npz',allow_pickle=False)
        begin=len(all_rows)
        emit('LSM_precision','grouped',42,z['test_predictions'],z['test_targets'],z['test_groups'],z['test_probabilities'],precision=folder.name,origin='Regenerated condition-specific fitted readout')
        ids=assignments.loc[assignments.partition=='test'].sort_values('array_row_zero_based').window_id.tolist()
        assert len(ids)==len(all_rows)-begin
        for row,wid in zip(all_rows[begin:],ids):
            evidence=provenance.loc[provenance.window_id==wid].iloc[0]
            row.update(window_id=wid,frequency_start_hz=float(evidence.frequency_start_hz),frequency_end_hz=float(evidence.frequency_end_hz))
    destination=ROOT/'results/predictions';destination.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(all_rows).to_csv(destination/'identified_predictions.csv',index=False)
    save_json(ROOT/'results/metrics/individual_runs.json',metrics)
    flat=pd.DataFrame([{k:v for k,v in m.items() if k!='confusion_matrix'} for m in metrics])
    flat.to_csv(ROOT/'results/metrics/individual_runs.csv',index=False)
    (ROOT/'results/tables').mkdir(exist_ok=True)
    reservoir=flat[flat.model.isin(['ESN','LSM_membrane','LSM_spikes'])]
    folds=reservoir.groupby(['model','evaluation_unit','fold']).accuracy.agg(mean='mean',sd_seeds=lambda x:np.std(x,ddof=0),n_seeds='size').reset_index()
    folds.to_csv(ROOT/'results/tables/reservoir_fold_means.csv',index=False)
    summary=folds.groupby(['model','evaluation_unit'])['mean'].agg(mean='mean',sd_fold_means=lambda x:np.std(x,ddof=0)).reset_index()
    summary.to_csv(ROOT/'results/tables/reservoir_summary.csv',index=False)
    selected=flat[((flat.model=='cnn1d') & (flat.variant=='selected')) | (flat.model=='spiking_cnn1d')]
    selected.groupby(['model','evaluation_unit','fold']).accuracy.agg(mean='mean',sd_seeds=lambda x:np.std(x,ddof=0)).reset_index().to_csv(ROOT/'results/tables/neural_fold_means.csv',index=False)
    model_rows=[]
    for p in sorted((ROOT/'models').rglob('*')):
        if p.suffix not in {'.npz','.pt','.joblib'}:continue
        import re
        relative=p.relative_to(ROOT).as_posix();part=p.relative_to(ROOT/'models').parts
        foldmatch=re.search(r'fold(\d+)',relative);seedmatch=re.search(r'seed(\d+)',relative)
        model_rows.append(dict(file=relative,family=part[0],model=part[1],fold=int(foldmatch[1]) if foldmatch else '',seed=int(seedmatch[1]) if seedmatch else 42,precision=part[1] if part[0]=='precision' else 'floating',bytes=p.stat().st_size,origin='original saved neural checkpoint' if p.suffix=='.pt' else 'regenerated fitted export'))
    pd.DataFrame(model_rows).to_csv(ROOT/'models/MODEL_INDEX.csv',index=False)
    print(f'Exported {len(all_rows)} identified predictions and {len(metrics)} individual evaluations.',flush=True)
    return flat

def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    final=ROOT/'figures/regenerated';final.mkdir(parents=True,exist_ok=True)
    feat=pd.read_csv(ROOT/'results/archive/feature_ml/high-glucose_baseline_summary.csv')
    feat.to_csv(final/'Figure_04_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,3.8));positions=np.arange(len(feat));width=.38
    ax.bar(positions-width/2,100*feat.accuracy,width,label='Accuracy')
    ax.bar(positions+width/2,100*feat.f1_macro,width,label='Macro-F1')
    ax.set_xticks(positions,feat.model,rotation=25,ha='right');ax.set_ylabel('Sweep-level score (%)');ax.set_ylim(0,105);ax.legend();fig.tight_layout();fig.savefig(final/'Figure_04.png',dpi=300);plt.close(fig)
    p=pd.read_csv(ROOT/'results/predictions/identified_predictions.csv')
    rf=p[(p.model=='Random_Forest') & (p.variant=='legacy_full_table_imputation')]
    cm=confusion_matrix(rf.target,rf.prediction,labels=range(5))
    np.savetxt(final/'Figure_05_data.csv',cm,fmt='%d',delimiter=',')
    fig,ax=plt.subplots(figsize=(4.4,4));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    for i in range(5):
        for j in range(5):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    ax.set_xticks(range(5),[50,100,250,500,1000]);ax.set_yticks(range(5),[50,100,250,500,1000]);ax.set_xlabel('Predicted concentration (mM)');ax.set_ylabel('True concentration (mM)');fig.tight_layout();fig.savefig(final/'Figure_05.png',dpi=300);plt.close(fig)
    folds=pd.read_csv(ROOT/'results/tables/reservoir_fold_means.csv');folds=folds[folds.evaluation_unit=='sweep']
    folds.to_csv(final/'Figure_06_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,3.8))
    for i,(name,g) in enumerate(folds.groupby('model')):
        ax.bar(g.fold.to_numpy()+i*.25-.25,g['mean']*100,.25,yerr=g.sd_seeds*100,capsize=3,label=name)
    ax.set_xticks(range(3),['Fold 0','Fold 1','Fold 2']);ax.set_ylabel('Sweep accuracy (%)');ax.set_ylim(0,115);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(final/'Figure_06.png',dpi=300);plt.close(fig)
    comparison=[]
    comparison.append(dict(model='Random Forest',evaluation_unit='sweep',mean=float(feat.loc[feat.model=='Random Forest','accuracy'].iloc[0]),sd=float(feat.loc[feat.model=='Random Forest','cv_std'].iloc[0])))
    neural=pd.read_csv(ROOT/'results/tables/neural_fold_means.csv')
    neural=neural[(neural.model=='cnn1d') & (neural.evaluation_unit=='window')]
    comparison.append(dict(model='CNN1D',evaluation_unit='window',mean=float(neural['mean'].mean()),sd=float(neural['mean'].std(ddof=0))))
    for row in pd.read_csv(ROOT/'results/tables/reservoir_summary.csv').to_dict('records'):
        comparison.append(dict(model=row['model'],evaluation_unit=row['evaluation_unit'],mean=row['mean'],sd=row['sd_fold_means']))
    c=pd.DataFrame(comparison);c.to_csv(final/'Figure_07_data.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4));bars=ax.bar(range(len(c)),100*c['mean'],yerr=100*c.sd,capsize=3)
    for bar,unit in zip(bars,c.evaluation_unit):
        if unit=='sweep':bar.set_hatch('///')
    ax.set_xticks(range(len(c)),[f'{m}\n{u}' for m,u in zip(c.model,c.evaluation_unit)],rotation=25,ha='right');ax.set_ylim(0,115);ax.set_ylabel('Accuracy (%)');fig.tight_layout();fig.savefig(final/'Figure_07.png',dpi=300);plt.close(fig)
    precision=json.loads((ROOT/'results/metrics/individual_runs.json').read_text(encoding='utf-8'))
    precision=[m for m in precision if m['model']=='LSM_precision']
    pd.DataFrame([{k:v for k,v in m.items() if k!='confusion_matrix'} for m in precision]).to_csv(final/'precision_data.csv',index=False)
    print('Regenerated Figures 4-7 and numerical figure data.',flush=True)

def main():
    complete_spike_exports();collect_results();make_figures()

if __name__=='__main__':main()
