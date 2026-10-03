"""Independent identity, state, selection and scalar-scoring audit of B07 evidence."""
import hashlib,json,math,statistics
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata,friedmanchisquare,studentized_range


def audit_course(lab,runs,report=None):
    lab=Path(lab);runs=Path(runs);protocol=lab/'evidence/b07/course-protocol.json';cfg=json.loads(protocol.read_text())
    for name,digest in {**cfg['portable_inputs'],**cfg['implementation_hashes']}.items():
        if hashlib.sha256((lab/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input authentication failed: '+name)
    report=json.loads((runs/'results.json').read_text()) if report is None else report
    if report['protocol_sha256']!=hashlib.sha256(protocol.read_bytes()).hexdigest():raise ValueError('Protocol mismatch')
    if report['encoder_updated'] is not False:raise ValueError('Unexpected encoder update')
    expected={(d,s,a) for d in cfg['datasets'] for s in cfg['seeds'] for a in cfg['arms']}
    keys=[(r['dataset'],r['seed'],r['arm']) for r in report['records']]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('Incomplete or duplicate run identities')
    splitkeys=[(r['dataset'],r['seed']) for r in report['splits']]
    if len(splitkeys)!=9 or set(splitkeys)!={(d,s) for d,s,a in expected}:raise ValueError('Split coverage')
    splits={(r['dataset'],r['seed']):r for r in report['splits']}
    meta=json.loads((lab/'data/b07/manifest.json').read_text());rows=[];allpred=0
    for r in report['records']:
        d,s,a=r['dataset'],r['seed'],r['arm'];split=splits[d,s]
        ids=np.random.default_rng(s+74).permutation(384);tr,va,te=ids[:64],ids[64:128],ids[128:]
        for name,ix in [('train',tr),('validation',va),('test',te)]:
            if split[name]!=ix.tolist():raise ValueError('Split identity mismatch')
        if r['test_ids']!=te.tolist():raise ValueError('Query identity mismatch')
        source=[meta['datasets'][d]['source_rows'][int(i)] for i in te]
        if r['source_ids']!=source or len(set(source))!=256:raise ValueError('Original source identity mismatch')
        data=pd.read_parquet(lab/'data/b07'/f'{d}.parquet');target=meta['datasets'][d]['target'];y=data.pop(target).to_numpy(dtype=float)
        if not np.array_equal(np.array(r['target']),y[te]):raise ValueError('Label identity mismatch')
        labels=[str(i) for i in range(6,19)]+['20','0'];mapping=dict(zip(data.columns,labels))
        expectedcols=list(data.columns) if a=='meaningful' else [mapping[c] for c in (data.select_dtypes(include='number').columns if a=='numeric_only' else data.columns)]
        if r['columns']!=expectedcols:raise ValueError('Intervention schema mismatch')
        path=runs/r['feature_file']
        if path.parent.resolve()!=runs.resolve():raise ValueError('Feature path escapes run directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['feature_sha256']:raise ValueError('Feature artifact mismatch')
        x=np.load(path,allow_pickle=False)['features']
        if x.shape!=(384,300) or hashlib.sha256(x.tobytes()).hexdigest()!=r['features_sha256']:raise ValueError('Feature content mismatch')
        h=r['head'];mu=x[tr].astype(float).mean(0);scale=x[tr].astype(float).std(0);scale[scale==0]=1
        np.testing.assert_allclose(h['mean'],mu,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(h['scale'],scale,rtol=1e-12,atol=1e-12)
        z=(x.astype(float)-mu)/scale;zt=z[tr]-z[tr].mean(0);yt=y[tr]-y[tr].mean()
        losses=[];betas=[];biases=[]
        # Solve the dual ridge system independently of sklearn's fitted object.
        for alpha in cfg['head']['alphas']:
            beta=zt.T@np.linalg.solve(zt@zt.T+alpha*np.eye(64),yt)
            intercept=float(y[tr].mean()-z[tr].mean(0)@beta)
            losses.append(float(np.mean((z[va]@beta+intercept-y[va])**2)));betas.append(beta);biases.append(intercept)
        best=min(range(3),key=lambda i:losses[i])
        if h['alpha']!=cfg['head']['alphas'][best]:raise ValueError('Invalid validation selection')
        np.testing.assert_allclose(h['validation_mse'],losses,rtol=2e-8,atol=1e-9)
        np.testing.assert_allclose(h['coefficient'],betas[best],rtol=2e-7,atol=1e-8)
        np.testing.assert_allclose(h['intercept'],biases[best],rtol=1e-8,atol=1e-9)
        pred=np.asarray(h['prediction'],dtype=float)
        if pred.shape!=(256,) or not np.isfinite(pred).all():raise ValueError('Prediction coverage or nonfinite values')
        np.testing.assert_allclose(pred,z[te]@betas[best]+biases[best],rtol=2e-8,atol=1e-8)
        yy=r['target'];mean=math.fsum(yy)/len(yy)
        sse=math.fsum((float(v)-float(t))**2 for v,t in zip(pred,yy));sst=math.fsum((float(t)-mean)**2 for t in yy)
        rows.append(dict(dataset=d,seed=s,arm=a,r2=1-sse/sst,rmse=math.sqrt(sse/256),selected_alpha=h['alpha']))
        allpred+=len(pred)
    aggregate=[];paired=[];ranks=[]
    bykey={(r['dataset'],r['seed'],r['arm']):r for r in rows}
    for d in cfg['datasets']:
        means=[]
        for a in cfg['arms']:
            values=[bykey[d,s,a]['r2'] for s in cfg['seeds']];means.append(statistics.mean(values))
            aggregate.append(dict(dataset=d,arm=a,r2_mean=statistics.mean(values),r2_sd=statistics.stdev(values)))
        ranks.append(rankdata(-np.array(means)).tolist())
        for left,right in [('anonymous','meaningful'),('numeric_only','anonymous')]:
            values=[bykey[d,s,left]['r2']-bykey[d,s,right]['r2'] for s in cfg['seeds']]
            paired.append(dict(dataset=d,contrast=left+' minus '+right,seed_deltas=values,mean=statistics.mean(values),sd=statistics.stdev(values)))
    statistic,pvalue=friedmanchisquare(*np.array(ranks).T)
    return dict(status='COMPLETE_COURSE_EXPERIMENT',fits=len(rows),test_predictions=allpred,rows=rows,aggregate=aggregate,paired=paired,exploratory_ranks=dict(arms=cfg['arms'],mean=np.mean(ranks,axis=0).tolist(),friedman_p=float(pvalue),nemenyi_cd=float(studentized_range.ppf(.95,3,np.inf)/np.sqrt(2)*np.sqrt(3*4/(6*3))),warning='Only three related wine tables; these are descriptive checks, not independent-domain significance evidence.'),head_reconstruction='Independent dual ridge solution PASS',pretraining='NOT_RUN',paper_result='INCOMPLETE_SOURCE_PROTOCOL',learner='PENDING_WRITTEN_DEFENSE')


if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=audit_course(p,p/'evidence/b07/runs');(p/'evidence/b07/course-audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
