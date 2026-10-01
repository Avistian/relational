"""Behavioral contracts for contribution integrity, coverage and claim scope."""
import hashlib,json,tempfile
from pathlib import Path
from relkit.contribution_l157 import verify_manifest,summarize_runs,review_claim

def rejects(call):
    try:call()
    except (ValueError,FileNotFoundError):return
    raise AssertionError('Invalid contribution accepted')

def check_integrity(fn):
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'result').write_bytes(b'original')
        digest=hashlib.sha256(b'original').hexdigest();m={'result':digest}
        assert fn(root,m)==1
        (root/'result').write_bytes(b'changed');rejects(lambda:fn(root,m))
        rejects(lambda:fn(root,{'../escape':digest}))
        rejects(lambda:fn(root,{'/absolute':digest}))
        rejects(lambda:fn(root,{}))
        (root/'alias').symlink_to(root/'result');rejects(lambda:fn(root,{'alias':hashlib.sha256(b'changed').hexdigest()}))

def fixtures():
    return [dict(lane=lane,seed=s,epochs=10,train_queries=7453,val_rows=499,test_rows=760,status='COMPLETE',test_mae=4+s/10) for lane in ['paper','fit_horizon'] for s in range(5)]

def check_runs(fn):
    rows=fixtures();r=fn(rows)
    assert r['status']=='COMPLETE' and r['fits']==10 and r['predictions']==12590
    assert abs(r['lanes']['paper']['mean']-4.2)<1e-12
    assert abs(r['lanes']['paper']['sd']-0.15811388300841897)<1e-12
    assert fn(rows[:-1])['status']=='INCOMPLETE'
    rejects(lambda:fn(rows+[rows[0]]))
    for key,value in [('epochs',9),('train_queries',100),('test_rows',0),('test_mae',float('nan')),('seed',9),('lane','unknown'),('status','PILOT')]:
        bad=[dict(x) for x in rows];bad[0][key]=value
        rejects(lambda:fn(bad))

def check_claims(fn):
    evidence=dict(integrity='PASS',reproduction='COMPLETE',availability='NOT_ESTABLISHED',public_url=None)
    assert fn('selected_reproduction',evidence)=='SUPPORTED'
    assert fn('historically_leak_free',evidence)=='NOT_ESTABLISHED'
    assert fn('public_contribution',evidence)=='PENDING_PUBLICATION'
    assert fn('whole_paper',evidence)=='NOT_RUN'
    assert fn('upstream_bug',evidence)=='NOT_ESTABLISHED'
    assert fn('selected_reproduction',dict(evidence,reproduction='INCOMPLETE'))=='INCOMPLETE'
    assert fn('selected_reproduction',dict(evidence,integrity='FAIL'))=='REJECTED'
    rejects(lambda:fn('magic',evidence))
    assert fn('public_contribution',dict(evidence,public_url='https://example.org/release'))=='NOT_CHECKED'

if __name__=='__main__':
    check_integrity(verify_manifest);check_runs(summarize_runs);check_claims(review_claim)
    print('PASS: integrity, complete experiment, bounded claims')
