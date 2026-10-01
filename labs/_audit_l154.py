"""Independent oracles plus mutation checks; no fitting or paid services."""
import copy
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, mean_absolute_error
from _replay_l154 import aligned_score, verify_inputs
from _check_l154 import rejects


def check_scorer():
    keys=[(1,10),(1,20),(2,10),(3,10)]
    labels=[0,1,1,0];values=[.2,.5,.5,.5]
    assert aligned_score(keys,labels,keys[::-1],values[::-1],'AUROC')==roc_auc_score(labels,values)
    assert aligned_score(keys,[1,2,4,8],keys[::-1],[7,3,1,2],'MAE')==1
    ranking=[[0,1],[2,0],[1,2],[1,0]];truth=[[0,1],[0],[2],[2]]
    # Four APs: 1, .5, .5, 0; MAP=.5 (k inferred from ranking width for fixtures).
    assert aligned_score(keys,truth,keys[::-1],ranking[::-1],'MAP@10',3)==.5
    rejects(lambda:aligned_score(keys,labels,keys[:3],values[:3],'AUROC'))
    rejects(lambda:aligned_score(keys,labels,[(1,10)]*4,values,'AUROC'))
    rejects(lambda:aligned_score(keys,labels,keys,[.2,.3,float('nan'),.4],'AUROC'))
    rejects(lambda:aligned_score(keys,truth,keys,[[0,0]]*4,'MAP@10',3))
    rejects(lambda:aligned_score(keys,truth,keys,[[0,3]]*4,'MAP@10',3))
    rejects(lambda:aligned_score(keys,[[],[0],[2],[2]],keys,ranking,'MAP@10',3))
    rejects(lambda:aligned_score(keys,[1,1,1,1],keys,values,'AUROC'))
    rejects(lambda:aligned_score(keys,labels,keys,values,'mystery'))


def check_integrity():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp);(p/'a').write_bytes(b'correct')
        m={'files':{'a':hashlib.sha256(b'correct').hexdigest()}}
        assert verify_inputs(p,m)==1
        (p/'a').write_bytes(b'corrupt')
        rejects(lambda:verify_inputs(p,m))
        (p/'a').unlink();rejects(lambda:verify_inputs(p,m))
        rejects(lambda:verify_inputs(p,{'files':{'../escape':'x'}}))


def audit_real():
    import gzip
    from _replay_l154 import replay
    from relkit.portfolio_l154 import summarize_runs,compare_entries,portfolio_verdict
    from _check_l154 import check_summary,check_comparison,check_verdict
    p=Path(__file__).resolve().parent
    m=json.loads((p/'evidence/l154/input-manifest.json').read_text())
    report=replay(p,m);count=0;max_error=0.
    for name,bucket in report['runs'].items():
        for split,rows in bucket.items():
            scores=[]
            for row in rows:
                seed=row['seed']
                if name.startswith('classification'):
                    prefix='ref-' if name.endswith('reference') else 'selected-'
                    file=p/f'evidence/l151/{prefix}{seed}/predictions.npz'
                    a=np.load(file);value=roc_auc_score(a[split+'_target'],a[split+'_pred'])
                else:
                    a=np.load(p/f'evidence/l152/paper/seed-{seed}/predictions.npz')
                    value=mean_absolute_error(a[split+'_target'],a[split+'_pred'])
                max_error=max(max_error,abs(value-row['score']));scores.append(value)
                count+=len(a[split+'_pred'])
            summary=report['summaries'][name][split]
            assert abs(float(np.mean(scores))-summary['mean'])<1e-12
            assert abs(float(np.std(scores,ddof=1))-summary['sample_sd'])<1e-12
    a=np.load(p/'evidence/l153/pilot/predictions.npz')
    truth=json.loads(gzip.decompress((p/'evidence/l153/prepared/val-truth.json.gz').read_bytes()))
    lookup={(t['entity'],t['time']):set(t['positives']) for t in truth}
    sets=[lookup[(int(e),int(t))] for e,t in zip(a['val_entity'],a['val_time'])]
    hits=np.array([[int(i) in relevant for i in ranking] for relevant,ranking in zip(sets,a['val_pred'])],dtype=float)
    aps=(hits*np.cumsum(hits,axis=1)/np.arange(1,11)).sum(axis=1)/np.minimum(10,[len(r) for r in sets])
    max_error=max(max_error,abs(float(aps.mean())-report['pilot']['mean']));count+=len(aps)
    assert max_error<1e-12 and count==report['total_prediction_rows']==61148
    assert report['verdict']['complete_tasks']==2 and report['verdict']['matched_baseline_tasks']==0
    mutants=[]
    # Check that each learner check suite catches an attractive wrong shortcut.
    for label,check,fn in [
        ('missing_seed_zero_fill',check_summary,lambda rows,seeds,contract:dict(mean=3,sample_sd=1,n_seeds=3,seeds=[0,1,2],status='COMPLETE')),
        ('ignore_metric_direction',check_comparison,lambda a,b:dict(oriented_gap=a['mean']-b['mean'],scope='LOCAL_MATCHED_DESCRIPTIVE',winner='MODEL',relative_percent=25)),
        ('count_pilot_as_complete',check_verdict,lambda *args:dict(complete_tasks=3,required_tasks=3))]:
        try:check(fn)
        except (AssertionError,KeyError,ValueError):mutants.append(label)
        else:raise AssertionError('Surviving mutation: '+label)
    # Cross-check the new freeze against inherited prediction digests, not only new hashes.
    inherited151=json.loads((p/'evidence/l151/summary.json').read_text())['hashes']
    inherited152=json.loads((p/'evidence/l152/summary.json').read_text())['hashes']
    inherited_checks=0
    for phase in [f'ref-{s}' for s in range(5)]+[f'selected-{s}' for s in range(10,15)]:
        file=p/f'evidence/l151/{phase}/predictions.npz'
        assert hashlib.sha256(file.read_bytes()).hexdigest()==inherited151[phase]['predictions']
        inherited_checks+=1
    for seed in range(5):
        name=f'evidence/l152/paper/seed-{seed}/predictions.npz'
        assert hashlib.sha256((p/name).read_bytes()).hexdigest()==inherited152[name]
        inherited_checks+=1
    from bs4 import BeautifulSoup
    soup=BeautifulSoup((p/'sources/l151/paper.html').read_text(),'html.parser')
    for table,task,baseline_col,model_col,scale in [('A2.T6','rel-trial/study-outcome',1,2,100),('A2.T7','rel-f1/driver-position',6,7,1),('A2.T8','rel-trial/site-sponsor-run',2,4,100)]:
        rows=soup.find(id=table).find_all('tr')
        i=next(i for i,row in enumerate(rows) if task.split('/')[1] in row.get_text())
        cells=rows[i+1].find_all(['td','th']);assert cells[0].get_text(strip=True)=='Test'
        published=report['published_context']['tasks'][task]
        assert abs(float(cells[baseline_col].get_text(' ',strip=True).split()[0])/scale-published['baseline_mean'])<1e-12
        assert abs(float(cells[model_col].get_text(' ',strip=True).split()[0])/scale-published['model_mean'])<1e-12
    result=dict(status='PASS',independent_oracle_rows=count,maximum_metric_error=max_error,
                mutation_checks=mutants,inherited_prediction_hashes=inherited_checks,published_test_rows=3,input_files=len(m['files']),new_training='NOT_RUN',cloud_usd=0)
    (p/'_audit_l154_results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    check_scorer();check_integrity()
    print(audit_real())
