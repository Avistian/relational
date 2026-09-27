"""Full population GCN reproduction runner. Default is a read-only plan/preflight."""
import argparse,hashlib,json,time,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from relkit.cvitkovic_l118 import fit_fold,released_folds
P=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,help='author-format preprocessed_datapoints directory')
    parser.add_argument('--output',type=Path,default=P/'results/l118/full')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--budget-usd',type=float,default=10)
    parser.add_argument('--reserve-usd',type=float,default=2)
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    meta=P/'sources/l118/data/homecreditdefaultrisk/homecreditdefaultrisk.db_info.json'
    info=json.loads(meta.read_text());pilot=json.loads((P/'_pilot_l118_results.json').read_text())
    projection=pilot.get('projected_five_fold_300_epoch_usd');rate=pilot.get('resource_rate_usd_per_second',.00022572)
    prior=json.loads((P/'_budget_l118.json').read_text())['pilot_maximum_worker_reservation_usd']
    required=set(map(str,info['train_dp_ids']+info['test_dp_ids']))
    present={p.name for p in args.prepared.iterdir() if p.is_file()} if args.prepared and args.prepared.exists() else set()
    blockers=[]
    if not required<=present:blockers.append(f'Missing {len(required-present)} prepared graphs of {len(required)}')
    if projection is None or projection+args.reserve_usd+prior>args.budget_usd:blockers.append('Measured 300-epoch projection plus reserve exceeds budget')
    report={'status':'BLOCKED' if blockers else 'READY','blockers':blockers,'selected_experiment':'Home Credit GCN, Table 4, five full folds','target_mean_auroc':.780,'target_sd':.004,'mean_tolerance':.01,'projected_resource_usd':projection,'budget_usd':args.budget_usd,'reserve_usd':args.reserve_usd,'prior_pilot_reservation_usd':prior,'available_prepared_graphs':len(required&present),'required_prepared_graphs':len(required),'source_runtime':'modern port; historical DGL runtime NOT_CHECKED','learner_status':'PENDING_WRITTEN_DEFENSE'}
    print(json.dumps(report,indent=2))
    if not args.execute:return
    if blockers:raise SystemExit('Full execution refused: resolve preflight blockers without relabeling a subset.')
    args.output.mkdir(parents=True,exist_ok=False)
    # Exact data fingerprint: hashes every required prepared graph before training.
    hashes={}
    for name in sorted(required):
        with (args.prepared/name).open('rb') as f:hashes[name]=hashlib.file_digest(f,'sha256').hexdigest()
    (args.output/'input_hashes.json').write_text(json.dumps(hashes))
    (args.output/'protocol.json').write_text(json.dumps(report|{'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),'implementation_sha256':hashlib.sha256((P/'relkit/cvitkovic_l118.py').read_bytes()).hexdigest()},indent=2))
    deadline=time.monotonic()+(args.budget_usd-args.reserve_usd-prior)/rate
    records=[]
    for fold in range(5):records.append(fit_fold(info,args.prepared,fold,args.output/f'fold-{fold}',args.device,deadline))
    # Rescore saved predictions and verify every held-out identity before aggregation.
    all_ids=[];scores=[]
    for fold,(_,_,expected) in enumerate(released_folds(info['train_dp_ids'])):
        data=np.load(args.output/f'fold-{fold}/predictions.npz')
        assert np.array_equal(data['ids'],expected)
        score=roc_auc_score(data['y'],data['p']);assert abs(score-records[fold]['test_auroc'])<1e-12
        scores.append(float(score));all_ids.extend(data['ids'].tolist())
    assert len(all_ids)==len(set(all_ids))==307511
    summary={'status':'COMPLETE_SELECTED_PORT','fold_auroc':scores,'mean':float(np.mean(scores)),'sample_sd':float(np.std(scores,ddof=1)),'target_mean':.780,'mean_verdict':'CLOSE' if abs(np.mean(scores)-.780)<=.01 else 'OUTSIDE_TOLERANCE','historical_identity':'NOT_ESTABLISHED','whole_paper':'NOT_ESTABLISHED'}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
