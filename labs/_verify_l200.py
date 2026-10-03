"""Independent rank-sum AUROC and adversarial evidence/learner verification."""
import hashlib,json,tempfile,shutil
from pathlib import Path
import numpy as np
from _audit_l200 import audit
from _test_l200 import checks
from relkit.exit_l200 import keyed_auc,complete_grid,exit_gate

def rank_auc(y,p):
    # Independent of pairwise scorer: sort, average tied ranks, Mann-Whitney statistic.
    y=np.asarray(y);p=np.asarray(p);order=np.argsort(p,kind='stable');ranks=np.empty(len(p));i=0
    while i<len(p):
        j=i+1
        while j<len(p) and p[order[j]]==p[order[i]]:j+=1
        ranks[order[i:j]]=(i+1+j)/2;i=j
    positives=int(y.sum());negatives=len(y)-positives
    return float((ranks[y==1].sum()-positives*(positives+1)/2)/(positives*negatives))

def verify(e):
    e=Path(e);report=audit(e);assert checks(keyed_auc,complete_grid,exit_gate)=='PASS';errors=[]
    for phase in ['pilot-1','full-1']:
        receipt=json.loads((e/phase/'receipt.json').read_text())
        for row in receipt['records']:
            x=np.load(e/phase/f"{row['arm']}-{row['seed']}.npz")
            value=rank_auc(x['label'],x['probability']);errors.append(abs(value-row['auc']));assert abs(value-row['auc'])<1e-12
    rng=np.random.default_rng(200)
    for i in range(100):
        y=np.r_[0,1,rng.integers(0,2,18)];p=rng.integers(0,8,20)/7;k=np.array([[j%5,j//5] for j in range(20)]);order=rng.permutation(20)
        assert abs(rank_auc(y,p)-keyed_auc(k,y,k[order],p[order]))<1e-12
    wrong=[(lambda *x:.5,complete_grid,exit_gate),(keyed_auc,lambda x:dict(runs=30,predictions=21060),exit_gate),(keyed_auc,complete_grid,lambda *x:dict(state='READY_FOR_TEACHER_REVIEW',blockers=[]))]
    for fs in wrong:
        try:checks(*fs)
        except (AssertionError,ValueError):pass
        else:raise AssertionError('Wrong learner function accepted')
    # Mutations happen only in disposable copies. Hash corruption and semantically
    # corrupt but rehashed records must both fail; never edit real predictions.
    rejected=[]
    for mode in ['bytes','missing_run','duplicate_run','keys','supports','labels','probability']:
        with tempfile.TemporaryDirectory() as td:
            q=Path(td)/'evidence';shutil.copytree(e,q)
            path=q/'pilot-1/RDBPFN-0.npz';receipt_path=q/'pilot-1/receipt.json'
            receipt=json.loads(receipt_path.read_text())
            if mode=='bytes':path.write_bytes(path.read_bytes()+b'corrupt')
            elif mode in ['missing_run','duplicate_run']:
                if mode=='missing_run':receipt['records'].pop()
                else:receipt['records'].append(receipt['records'][0])
                receipt_path.write_text(json.dumps(receipt))
            else:
                with np.load(path) as x:arrays={k:x[k].copy() for k in x.files}
                if mode=='keys':arrays['keys'][0]=arrays['keys'][1]
                if mode=='supports':arrays['support_keys'][0]=arrays['support_keys'][1]
                if mode=='labels':arrays['label'][0]=1-arrays['label'][0]
                if mode=='probability':arrays['probability'][0]=np.nan
                np.savez_compressed(path,**arrays);receipt['records'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest();receipt_path.write_text(json.dumps(receipt))
            if mode!='bytes':
                manifest=json.loads((q/'packet-manifest.json').read_text())
                for name in manifest['files']:manifest['files'][name]=hashlib.sha256((q/name).read_bytes()).hexdigest()
                (q/'packet-manifest.json').write_text(json.dumps(manifest))
            try:audit(q)
            except (ValueError,AssertionError):rejected.append(mode)
            else:raise AssertionError('Corruption accepted '+mode)
    return dict(status='PASS',independent_rank_auc_runs=len(errors),maximum_error=max(errors),randomized_keyed_cases=100,wrong_learner_functions_rejected=3,corruptions_rejected=rejected,report_parity=report==json.loads((e/'report.json').read_text()))

if __name__=='__main__':
    p=Path(__file__).resolve().parent;r=verify(p/'evidence/l200');assert r['report_parity'];(p/'_verify_l200_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
