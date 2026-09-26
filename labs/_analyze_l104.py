"""Independent reconstruction of every AP and paired comparison from saved scores."""
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent

def independent_ap(y,scores):
    # Threshold-grouped precision-recall integration, independently of sklearn.
    order=np.argsort(-scores,kind='stable');y=y[order];scores=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(scores)),len(scores)-1]
    tp=np.cumsum(y)[ends];fp=ends+1-tp
    return float(np.sum(np.diff(np.r_[0,tp])/y.sum()*tp/(tp+fp)))

def read_records(z,mode,lane):return {k:z[f'{mode}_{lane}_{k}'] for k in ['e','negative','p','n','batch']}
def ap(r):return independent_ap(np.r_[np.ones(len(r['p'])),np.zeros(len(r['n']))],np.r_[r['p'],r['n']])
def analyze(root):
    records=[];checked=0;max_error=0;replayed=0
    frozen=json.loads((P/'_inputs_l104.json').read_text())
    for seed in range(10):
        path=Path(root)/f'seed-{seed}'
        if not (path/'result.json').exists():continue
        r=json.loads((path/'result.json').read_text());assert r['status']=='COMPLETE' and r['identity']['seed']==seed
        assert hashlib.sha256((path/'predictions.npz').read_bytes()).hexdigest()==r['predictions_sha256']
        expected_input={**{k:v for k,v in frozen.items() if k!='seeds'},'seed_files':frozen['seeds'][str(seed)]}
        assert r['identity']['input_identity']==expected_input,'Input identity mismatch'
        assert r['identity']['checkpoint_sha256']==frozen['seeds'][str(seed)]['selected.pt']
        for name,digest in r['identity']['code'].items():
            assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,'Executed source drift: '+name
        assert r['identity']['intervention_rng']==104+seed and r['identity']['lookahead_seconds']==86400
        z=np.load(path/'predictions.npz')
        for lane in ['all','new']:
            base=read_records(z,'strict',lane);assert r['access_audit']['strict'][lane]['nonpast_records']==0
            for mode in ['inclusive','lookahead']:
                changed=read_records(z,mode,lane)
                for k in ['e','negative','batch']:np.testing.assert_array_equal(base[k],changed[k])
                comparison=r['comparisons'][lane][mode]
                np.testing.assert_allclose([comparison['baseline_ap'],comparison['changed_ap'],comparison['delta_pp']],[ap(base),ap(changed),100*(ap(changed)-ap(base))],rtol=0,atol=1e-11)
                assert r['access_audit'][mode][lane]['nonpast_records']>0
                checked+=1
            release=read_records(z,'release',lane);batch_values=[]
            for batch in np.unique(release['batch']):
                mask=release['batch']==batch;batch_values.append(ap({k:v[mask] for k,v in release.items()}))
            np.testing.assert_allclose(np.mean(batch_values),r['release_replay'][lane]['batch_mean_ap'],atol=1e-12,rtol=0)
            assert len(release['e'])==r['events'][lane]==r['release_replay'][lane]['events']
            replayed+=len(release['e']);max_error=max(max_error,r['release_replay'][lane]['max_prediction_error'])
        records.append({k:r[k] for k in ['comparisons','release_replay','access_audit','events','elapsed_seconds','predictions_sha256']}|{'seed':seed,'checkpoint_sha256':r['identity']['checkpoint_sha256']})
    if not records:raise RuntimeError('No complete results; no aggregate may be claimed')
    summary={}
    for lane,target in [('all',95.34),('new',93.99)]:
        release=statistics.mean(100*r['release_replay'][lane]['batch_mean_ap'] for r in records)
        summary[lane]={'strict_mean_ap_percent':statistics.mean(100*r['comparisons'][lane]['inclusive']['baseline_ap'] for r in records),'release_mean_ap_percent':release,'release_sample_sd_pp':statistics.stdev(100*r['release_replay'][lane]['batch_mean_ap'] for r in records) if len(records)>1 else None,'paper_target_percent':target,'paper_reported_sd_pp':.1 if lane=='all' else .3,'numerical_verdict':('CLOSE' if abs(release-target)<=.5 else 'OUTSIDE_TOLERANCE') if len(records)==10 else 'INCOMPLETE'}
        for mode in ['inclusive','lookahead']:
            values=[r['comparisons'][lane][mode]['delta_pp'] for r in records]
            summary[lane][mode]={'mean_delta_pp':statistics.mean(values),'sample_sd_pp':statistics.stdev(values) if len(values)>1 else None,'min_delta_pp':min(values),'max_delta_pp':max(values)}
    result={'status':'COMPLETE' if len(records)==10 else 'INCOMPLETE','planned_seeds':list(range(10)),'included_seeds':[r['seed'] for r in records],'missing_seeds':sorted(set(range(10))-{r['seed'] for r in records}),'replayed_positive_events':replayed,'max_prediction_error':max_error,'independent_paired_metric_checks':checked,'summary':summary,'records':records,'training':'REUSED_L103','historical_identity':'INCOMPARABLE','full_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
    (P/'_analysis_l104_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','included_seeds','missing_seeds','summary']},indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default=str(P/'evidence/l104/full'));analyze(p.parse_args().directory)
