from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import argparse
import json
def main():
    parser=argparse.ArgumentParser(description='Regenerate fitted models or retrain a recorded neural run.')
    parser.add_argument('--family',choices=['features','reservoir','precision','cnn1d','spiking_cnn1d'],required=True)
    parser.add_argument('--fold',type=int,choices=range(3),default=0)
    parser.add_argument('--seed',type=int,choices=[7,42,123,999,2024],default=7)
    args=parser.parse_args()
    from src.pmut_models import reproduction as r
    if args.family=='features':r.feature_models_stage()
    elif args.family=='reservoir':
        r.reservoir_models_stage()
        from src.pmut_models.reporting import complete_spike_exports
        complete_spike_exports()
    elif args.family=='precision':r.precision_stage()
    else:
        if args.family=='spiking_cnn1d' and args.fold!=0:parser.error('SpikingCNN1D reported scope is fold 0 only.')
        from src.pmut_models.data import build_dataset_folds
        from src.pmut_models.train_utils import train_single_run
        recorded=json.loads((ROOT/'configs/recorded_neural_runs.json').read_text(encoding='utf-8'))
        record=next(x for x in recorded if x['model']==args.family and x['fold']==args.fold and x['seed']==args.seed)
        settings=record['recorded_settings']
        bundle=build_dataset_folds('high-glucose',data_root=ROOT/'data/raw',window_size=128,stride=128)[args.fold]
        output=ROOT/f'results/new_runs/{args.family}/fold{args.fold}_seed{args.seed}'
        train_single_run(bundle,model_name=args.family,epochs=settings['epochs'],batch_size=settings['batch_size'],lr=settings['learning_rate'],output_dir=output,seed=args.seed)
if __name__=='__main__':main()
