"""Reconstruct L062 saved predictions/metrics without importing its model operators."""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from relkit.data import load_tier_a
ROOT=Path(__file__).resolve().parent

def check(path,report_path=None):
    path=Path(path);result=json.loads(path.read_text());cfg=result['config'];datasets={}
    for name in cfg['datasets']:
        if name=='wdbc':data=load_breast_cancer();x,y=data.data,data.target
        else:x,y=load_tier_a(name)
        x=np.asarray(x,dtype=np.float32);_,y=np.unique(np.asarray(y),return_inverse=True)
        got=result['datasets'][name]
        assert got['shape']==list(x.shape)
        assert got['x_sha256']==hashlib.sha256(x.tobytes()).hexdigest()
        assert got['y_sha256']==hashlib.sha256(y.tobytes()).hexdigest()
        datasets[name]=(x,y)
    expected={(d,s,v) for d in cfg['datasets'] for s in cfg['seeds'] for v in cfg['views']}
    seen=set();deltas=[]
    for row in result['records']:
        key=(row['dataset'],row['seed'],row['views']);assert key not in seen;seen.add(key)
        x,y=datasets[row['dataset']];ids=np.arange(len(y))
        if cfg['cap'] is not None:ids=train_test_split(ids,train_size=cfg['cap'],stratify=y,random_state=620)[0]
        tr,te=train_test_split(ids,test_size=.5,stratify=y[ids],random_state=row['seed'])
        assert row['train_ids']==tr.tolist() and row['test_ids']==te.tolist()
        assert not set(tr)&set(te) and set(tr)|set(te)==set(ids)
        targets=y[te];assert row['targets']==targets.tolist()
        p=np.asarray(row['probabilities']);assert p.shape==(len(te),2)
        assert np.isfinite(p).all() and (p>=0).all() and np.max(np.abs(p.sum(-1)-1))<2e-7
        trace=row['trace'];logits=np.asarray(trace['logits']);configs=trace['configurations']
        assert len(logits)==len(configs)==min(row['views'],2*2*x.shape[1])
        aligned=np.stack([z[:,[(j+c[0])%2 for j in range(2)]] for z,c in zip(logits,configs)])
        wanted=softmax(aligned.mean(0)/trace['temperature'],axis=-1)
        probability_delta=float(np.max(np.abs(p-wanted)));assert probability_delta<1e-6
        clipped=np.clip(p,np.finfo(np.float32).eps,1-np.finfo(np.float32).eps)
        loss=float(-np.log(clipped[np.arange(len(te)),targets]).mean())
        ranks=rankdata(p[:,1]);positive=targets==1;np_=positive.sum();nn=(~positive).sum()
        auc=float((ranks[positive].sum()-np_*(np_+1)/2)/(np_*nn))
        loss_delta=abs(loss-row['log_loss']);auc_delta=abs(auc-row['auc'])
        assert loss_delta<1e-6 and auc_delta<1e-12
        deltas.append(dict(dataset=key[0],seed=key[1],views=key[2],probability_delta=probability_delta,loss_delta=loss_delta,auc_delta=auc_delta))
    assert seen==expected,(seen,expected)
    intervention_audit=None
    if 'intervention' in result:
        row=result['intervention'];x,y=datasets['diabetes'];tr=np.array(row['train_ids']);te=np.array(row['test_ids'])
        assert row['context_labels']==y[tr][np.random.default_rng(row['shuffle_seed']).permutation(len(tr))].tolist()
        assert row['targets']==y[te].tolist()
        base=next(r for r in result['records'] if r['views']==4)
        assert row['train_ids']==base['train_ids'] and row['test_ids']==base['test_ids']
        p=np.asarray(row['probabilities']);trace=row['trace']
        aligned=np.stack([np.asarray(z)[:,[(j+c[0])%2 for j in range(2)]] for z,c in zip(trace['logits'],trace['configurations'])])
        probability_delta=float(np.abs(p-softmax(aligned.mean(0)/trace['temperature'],axis=-1)).max())
        assert probability_delta<1e-6
        clipped=np.clip(p,np.finfo(np.float32).eps,1-np.finfo(np.float32).eps)
        loss=float(-np.log(clipped[np.arange(len(te)),y[te]]).mean())
        positives=y[te]==1;np_=positives.sum();nn=(~positives).sum()
        auc=float((rankdata(p[:,1])[positives].sum()-np_*(np_+1)/2)/(np_*nn))
        assert abs(loss-row['log_loss'])<1e-6 and abs(auc-row['auc'])<1e-12
        intervention_audit=dict(status='PASS',predictions=len(te),log_loss=loss,auc=auc,
            probability_delta=probability_delta,loss_delta=abs(loss-row['log_loss']),auc_delta=abs(auc-row['auc']),
            paired_log_loss_change=loss-base['log_loss'])
    report=dict(status='PASS',records=len(seen),predictions=sum(len(r['targets']) for r in result['records']),
        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=deltas,
        intervention=intervention_audit,
        scope='Reloaded dataset bytes and regenerated local split IDs; NumPy/SciPy class-aligned logits, probabilities, log loss and rank-based AUC. Not full paper benchmark reproduction.')
    if report_path:Path(report_path).write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('path',nargs='?',default=ROOT/'_verify_l062_v2_results.json');p.add_argument('--report',default=ROOT.parent/'reviews/lesson-quality-audit-047-070/062-evidence.json');a=p.parse_args()
    print(json.dumps(check(a.path,a.report),indent=2))
