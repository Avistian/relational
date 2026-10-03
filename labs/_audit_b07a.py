"""Independent identity, head, nearest-neighbor and scalar-metric reconstruction."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np

def audit_course(lab,run_dir):
    lab=Path(lab);run_dir=Path(run_dir)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    protocol=json.loads((lab/'evidence/b07a/course-protocol.json').read_text());report=json.loads((run_dir/'results.json').read_text())
    assert report['protocol_sha256']==digest(lab/'evidence/b07a/course-protocol.json')
    assert report['checkpoint_sha256']==json.loads((lab/'sources/b07a/checkpoint.json').read_text())['sha256']
    expected={(d,s,a) for d in protocol['datasets'] for s in protocol['seeds'] for a in protocol['arms']}
    records=report['records'];keys=[(r['dataset'],r['seed'],r['arm']) for r in records]
    assert len(keys)==len(set(keys)) and set(keys)==expected,'Run coverage mismatch'
    rows=[];probabilities={};arrays={};total=0
    for r in records:
        d,s,a=r['dataset'],r['seed'],r['arm'];spec=next(x for x in protocol['splits'] if x['dataset']==d and x['seed']==s)
        path=lab/f'data/b07a/{d}.npz';assert digest(path)==protocol['datasets'][d]['sha256'];data=np.load(path)
        tr,te=spec['train'],spec['test'];assert set(tr).isdisjoint(te) and set(tr)|set(te)==set(range(len(data['y'])))
        assert r['file']==f'{d}-{s}-{a}.npz'
        path=run_dir/r['file'];assert digest(path)==r['sha256'],'Prediction bytes changed';v=np.load(path)
        assert v['probability'].shape==(len(te),2) and np.isfinite(v['probability']).all()
        assert np.isfinite(v['logits']).all() and np.isfinite(v['raw_logits']).all()
        assert (v['probability']>=0).all() and (v['probability']<=1).all()
        np.testing.assert_allclose(v['probability'].sum(1),1,atol=2e-7,rtol=0)
        assert np.array_equal(v['row_id'],te) and np.array_equal(v['target'],data['y'][te]),'Row or label mismatch'
        assert len(set(r['support_ids']))==512 and set(r['support_ids'])<=set(tr)
        mean=data['X'][tr].mean(0);scale=data['X'][tr].std(0);scale[scale==0]=1
        np.testing.assert_allclose(r['scaler_mean'],mean,rtol=0,atol=1e-12);np.testing.assert_allclose(r['scaler_scale'],scale,rtol=0,atol=1e-12)
        path=run_dir/r['feature_file'];assert digest(path)==r['feature_sha256'];f=np.load(path)
        support=((data['X'][r['support_ids']]-mean)/scale).astype(np.float32).repeat(2,axis=0)
        np.testing.assert_array_equal(f['support_x'],support);np.testing.assert_array_equal(f['support_y'],data['y'][r['support_ids']].repeat(2))
        raw=f['query_hidden']@v['output_matrix']+v['output_bias']
        np.testing.assert_allclose(raw,v['raw_logits'],atol=2e-4,rtol=2e-5)
        logits=raw.copy()
        if a=='retrieval':
            query=((data['X'][te]-mean)/scale).astype(np.float32)
            for q,support_rep,amount in [(query,support,r['bias_parameters'][0]),(f['query_hidden'],f['support_hidden'],r['bias_parameters'][1])]:
                for start in range(0,len(te),16):
                    dist=np.sqrt(np.sum((q[start:start+16,None,:]-support_rep[None,:,:])**2,axis=2))
                    nearest=f['support_y'][np.argmin(dist,axis=1)]
                    logits[np.arange(start,min(start+16,len(te))),nearest]+=amount
        np.testing.assert_allclose(logits,v['logits'],atol=2e-4,rtol=2e-5)
        pp=np.exp(v['logits'].astype(np.float64)-v['logits'].max(1,keepdims=True));pp/=pp.sum(1,keepdims=True)
        np.testing.assert_allclose(pp,v['probability'],atol=2e-7,rtol=1e-6)
        pred=np.argmax(v['probability'],axis=1);target=v['target'];counts=[sum(int(t==k) for t in target) for k in [0,1]]
        ba=sum(sum(int(p==k and t==k) for p,t in zip(pred,target))/counts[k] for k in [0,1])/2
        loss=-sum(math.log(max(float(v['probability'][i,int(t)]),1e-15)) for i,t in enumerate(target))/len(target)
        for entry in r['timings']:assert len(entry['seconds'])==3 and all(math.isfinite(v) and v>0 for v in entry['seconds'])
        assert [t['query_rows'] for t in r['timings']]==[1,32,128]
        rows.append(dict(dataset=d,seed=s,arm=a,balanced_accuracy=ba,log_loss=loss));total+=len(target)
        arrays[d,s,a]=v['raw_logits'];probabilities[d,s,a]=r
    paired=[];aggregate=[];timings=[]
    for d in protocol['datasets']:
        delta=[]
        for s in protocol['seeds']:
            left,right=probabilities[d,s,'retrieval'],probabilities[d,s,'weights_only']
            assert left['predictor_sha256']==right['predictor_sha256'] and left['support_ids']==right['support_ids']
            np.testing.assert_array_equal(arrays[d,s,'retrieval'],arrays[d,s,'weights_only'])
            delta.append(next(x['balanced_accuracy'] for x in rows if (x['dataset'],x['seed'],x['arm'])==(d,s,'retrieval'))-next(x['balanced_accuracy'] for x in rows if (x['dataset'],x['seed'],x['arm'])==(d,s,'weights_only')))
        paired.append(dict(dataset=d,retrieval_minus_weights=delta,mean=statistics.mean(delta),sd=statistics.stdev(delta)))
        for a in protocol['arms']:
            values=[x['balanced_accuracy'] for x in rows if x['dataset']==d and x['arm']==a]
            aggregate.append(dict(dataset=d,arm=a,mean=statistics.mean(values),sd=statistics.stdev(values)))
            for q in [1,32,128]:
                samples=[v for r in records if r['dataset']==d and r['arm']==a for t in r['timings'] if t['query_rows']==q for v in t['seconds']]
                timings.append(dict(dataset=d,arm=a,query_rows=q,median_seconds=statistics.median(samples),min_seconds=min(samples),max_seconds=max(samples)))
    assert {(x['dataset'],x['seed']) for x in report['source_prediction_parity']}=={(d,s) for d in protocol['datasets'] for s in protocol['seeds']}
    assert all(math.isfinite(v) and v<=2e-5 for x in report['source_prediction_parity'] for v in x['max_abs_difference'].values())
    assert report['refresh']['original_sha256']!=report['refresh']['refreshed_sha256']
    return dict(status='PASS',runs=len(records),predictions=total,rows=rows,aggregate=aggregate,paired=paired,timings=timings,source_prediction_parity=report['source_prediction_parity'],refresh=report['refresh'],claim='Complete selected course inference; original Table7 NOT_REPRODUCED; no model-family ranking')

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=audit_course(p,p/'evidence/b07a/runs');(p/'evidence/b07a/course-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','runs','predictions','aggregate','paired']},indent=2))
