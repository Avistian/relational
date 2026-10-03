"""Portable independent scorer: NumPy and scalar sums, no training-library imports."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np

def audit(directory):
    root=Path(directory)
    manifest=json.loads((root/'compact-lock.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Evidence hash mismatch: '+name)
    y=np.load(root/'data/y.npy');x=np.load(root/'data/x_num.npy')
    split={part:np.load(root/f'data/{part}.npy') for part in ['train','val','test']}
    assert {k:len(v) for k,v in split.items()}=={'train':13209,'val':3303,'test':4128}
    joined=np.concatenate(list(split.values()))
    assert len(joined)==len(y)==len(x)==20640 and len(np.unique(joined))==len(y)
    assert np.array_equal(np.sort(joined),np.arange(len(y)))
    main=json.loads((root/'runs/main/report.json').read_text())
    search=json.loads((root/'runs/main/config.json').read_text())
    assert search['seed']==0 and search['n_models']==64 and search['n_epochs']==-1 and search['patience']==16
    assert search['amp_dtype']=='bfloat16' and search['batch_size']==256
    configs=json.loads((root/'runs/main/experiments.json').read_text())
    selected_ids=sorted(set(main['online_ensembles']['greedy']['report']['ids']))
    evaluation=json.loads((root/'runs/evaluation/config.json').read_text())
    assert evaluation['n_seeds']==5
    assert evaluation['base_config']['configs']==[configs[i]['config'] for i in selected_ids]
    assert evaluation['base_config']['n_models']==len(selected_ids)
    report=json.loads((root/'runs/evaluation/report.json').read_text())
    assert [a['config']['seed'] for a in report['experiments']]==list(range(5))
    records=[];max_error=0.;member_predictions=0
    for name,r in [('main',main)]+[(str(i),e['report']) for i,e in enumerate(report['experiments'])]:
        er=r['online_ensembles']['greedy']['report']
        with np.load(root/f'runs/{name}/observed_ensemble.npz',allow_pickle=False) as z:
            assert z['ids'].tolist()==er['ids'] and z['steps'].tolist()==er['steps']
            scores={}
            for part in ['val','test']:
                p=z[part];assert p.shape==(len(er['ids']),len(split[part])) and np.isfinite(p).all()
                # Preserve float32 averaging in released prediction path; score independently in float64.
                prediction=p.mean(axis=0).astype(np.float64);target=y[split[part]].astype(np.float64)
                rmse=math.sqrt(math.fsum(float(a-b)**2 for a,b in zip(prediction,target))/len(target))
                oracle=float(np.sqrt(np.mean((prediction-target)**2)))
                assert abs(oracle-rmse)<1e-12
                error=abs(rmse-er['metrics'][part]['rmse']);assert error<1e-6,(name,part,error)
                max_error=max(max_error,error);scores[part]=rmse;member_predictions+=p.size
            records.append(dict(run=name,selected_member_occurrences=len(er['ids']),unique_members=len(set(er['ids'])),**scores))
    observed=[r['test'] for r in records[1:]]
    original=json.loads((root/'released-report.json').read_text())
    released=[e['report']['online_ensembles']['greedy']['report']['metrics']['test']['rmse'] for e in original['experiments']]
    assert len(released)==5
    return dict(status='COMPLETE_SELECTED_RELEASE_PROTOCOL',paper_parity='UNRESOLVED_SEED_COUNT',source_commit='05a89e21b955f12de84889d662e15ca534019aaa',search_members=64,selected_configurations=len(selected_ids),evaluation_seeds=list(range(5)),rows={k:len(v) for k,v in split.items()},runs=records,test_mean=statistics.mean(observed),test_sample_sd=statistics.stdev(observed),released_mean=statistics.mean(released),released_sample_sd=statistics.stdev(released),max_score_error=max_error,max_released_seed_score_difference=max(abs(a-b) for a,b in zip(observed,released)),scored_prediction_rows=6*(3303+4128),retained_member_prediction_values=member_predictions,source_serialization_bug='Original online_ensemble_predictions.npz stores a name; read-only observer saves original in-memory arrays',full_benchmarks='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import sys
    E=Path(__file__).resolve().parent/'evidence/b02'
    result=audit(Path(sys.argv[1]) if len(sys.argv)>1 else E/'compact')
    (E/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
