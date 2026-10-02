"""Independent scalar context oracle, exact money oracle, and negative controls."""
import copy,hashlib,json,subprocess,sys,tempfile
from decimal import Decimal
from pathlib import Path
import numpy as np
from _audit_l180 import audit180
from _check_l180 import checks
from relkit.checkpoint_l180 import temporal_counts,full_run_cost,checkpoint_decision
P=Path(__file__).resolve().parent;E=P/'evidence/l180';packet=E/'packet'
manifest=json.loads((E/'input-manifest.json').read_text());expected=json.loads((E/'report.json').read_text())
assert checks(temporal_counts,full_run_cost,checkpoint_decision)=='PASS'
def scalar(ts,pad,q,lab,mask,cutoff,horizon):
    d=dict(future_cells=0,unmasked_query_targets=0,unavailable_labels=0,unknown_time_cells=0)
    for t,p,target,label,m in zip(ts,pad,q,lab,mask):
        if p:continue
        t=int(t);known=t!=-2147483648
        d['future_cells']+=int(known and t>cutoff)
        d['unmasked_query_targets']+=int(target and not m)
        d['unavailable_labels']+=int(label and not m and known and t+horizon>cutoff)
        d['unknown_time_cells']+=int(not known)
    return d
assert audit180(packet,manifest,scalar,full_run_cost,checkpoint_decision)==expected
# Integer micro-dollar arithmetic independently checks every price; no float tolerance.
assert int(8*5400*583)==int(Decimal(expected['costs_gpu_only_usd']['A100_40GB'])*1000000)
assert int(8*5400*694)==int(Decimal(expected['costs_gpu_only_usd']['A100_80GB'])*1000000)
rng=np.random.default_rng(180)
for _ in range(100):
    ts=rng.integers(-100,200,32);ts[0]=-2147483648
    flags=rng.integers(0,2,(4,32)).astype(bool)
    assert temporal_counts(ts,*flags,100,30)==scalar(ts,*flags,100,30)
wrong=[(lambda *a:dict(future_cells=0,unmasked_query_targets=0,unavailable_labels=0,unknown_time_cells=0),full_run_cost,checkpoint_decision),(temporal_counts,lambda *a,**kw:'3.148200',checkpoint_decision),(temporal_counts,full_run_cost,lambda e:dict(admission='READY_FOR_SEPARATELY_AUTHORIZED_RUN',blockers=[],practical_exit='PASS'))]
for fns in wrong:
    try:checks(*fns)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('Wrong learner accepted')
# Tampering is rejected before scientific interpretation, even for metadata-only changes.
for name in ['config.json','context-audit.json','contexts-0.npz','example_finetune.py']:
    bad=copy.deepcopy(manifest);bad['files'][name]='0'*64
    try:audit180(packet,bad,temporal_counts,full_run_cost,checkpoint_decision)
    except ValueError as e:assert 'hash mismatch' in str(e)
    else:raise AssertionError('Corrupt input accepted')
run=subprocess.run([sys.executable,str(P/'_run_l180.py')],capture_output=True,text=True)
assert run.returncode==2 and json.loads(run.stdout)['decision']==expected['decision']
# Source-backed architecture, objective and validation selection boundaries.
source=P/'sources/l180/upstream/rt';model=(source/'model.py').read_text();trainer=(source/'main.py').read_text()
for s in ['for l in ["col", "feat", "nbr", "full"]','binary_cross_entropy_with_logits','loss_out = loss_out / masks.sum()']:assert s in model
assert 'metrics["val"].items()' in trainer and 'if save_ckpt_dir is not None:' in trainer
for file,digest in json.loads((P/'sources/l180/source-ledger-l180.json').read_text())['files'].items():assert hashlib.sha256((P/file).read_bytes()).hexdigest()==digest
out=dict(status='PASS',scalar_contexts=2106,scalar_cell_slots=2156544,raw_result_rows=expected['raw_result_rows'],random_context_cases=100,wrong_learners_rejected=3,corrupt_packets_rejected=4,entrypoint='BLOCKED_BEFORE_DISPATCH',independent_cost='INTEGER_MICRO_DOLLARS',fresh_training='NOT_RUN')
(P/'_verify_l180_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
