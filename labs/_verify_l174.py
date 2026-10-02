"""Independent scores, full identities, trainability and all checkpoint audits."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from relkit.finetune_l174 import load_adaptation_packet,AdaptationModel,configure_trainable,adaptation_split,select_epoch,state_hash
from relkit.multitask_l173 import tensor_packet,predict
from _check_l174 import check174
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=P/'evidence/l174/runs');args=parser.parse_args();E=args.evidence
# Reconstruct every inherited target, preprocessing fit, and capped context from raw tables.
subprocess.run([sys.executable,str(P/'_verify_l173.py'),'--population-only'],check=True)
a,t,states,m=load_adaptation_packet(E)
for n,h in {**m['inputs'],**m['source_files']}.items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
old=np.load(P/'evidence/l173/population.npz')
for k in old.files:assert np.array_equal(old[k],a[k])
years=a['dates'][a['cell_ids']].astype('datetime64[s]').astype('datetime64[Y]').astype(int)+1970
train=np.flatnonzero(years==2006);valid=np.flatnonzero(years==2007);test=np.flatnonzero(years>=2008)
assert [len(train),len(valid),len(test)]==[6429,6074,106757]
assert len(set(a['cell_ids'][test]))==len(test)
expected_counts=np.bincount(a['task'][train],minlength=len(t)).tolist()
torch.set_num_threads(1);data=tensor_packet(a)

def oracle(output,target,ids):
    rows=[]
    for i,s in enumerate(t):
        mask=ids==i;x=output[mask].astype(np.float64);y=target[mask].astype(np.float64)
        if s['kind']=='number':
            err=np.abs(x[:,0]-y);loss=np.where(err<1,.5*err**2,err-.5);extra=dict(mae_raw=float(err.mean()*s['scale']))
        else:
            z=x[:,:s['classes']];z=z-z.max(1,keepdims=True)
            loss=np.log(np.exp(z).sum(1))-z[np.arange(len(y)),y.astype(int)]
            extra=dict(accuracy=float((z.argmax(1)==y).mean()),unknown_targets=int((y==0).sum()))
        rows.append(dict(task=s['name'],count=int(mask.sum()),loss=float(loss.mean()),**extra))
    return dict(macro_loss=float(np.mean([x['loss'] for x in rows])),tasks=rows)

def compare(actual,saved):
    assert abs(actual['macro_loss']-saved['macro_loss'])<2e-6
    for x,y in zip(actual['tasks'],saved['tasks']):
        assert x['task']==y['task'] and x['count']==y['count']
        for k in ['loss','mae_raw','accuracy','unknown_targets']:
            if k in x:assert np.isclose(x[k],y[k],rtol=4e-6,atol=2e-6),(k,x,y)

def packet(path):
    p=np.load(path)
    for key in ['cell_ids','task','target']:assert np.array_equal(p[key],a[key][test]),key
    assert np.isfinite(p['output']).all()
    return p['output']

r=json.loads((E/'report.json').read_text());assert r['status']=='COMPLETE' and len(r['runs'])==12
assert {(x['arm'],x['seed']) for x in r['runs']}=={(arm,seed) for arm in ['freeze','full','adapter','scratch'] for seed in range(3)}
for run in r['runs']:
    d=E/f'{run["arm"]}-{run["seed"]}';assert json.loads((d/'result.json').read_text())==run
    initial=torch.load(d/'initial.pt',weights_only=True);assert state_hash(initial)==run['initial_sha256']
    model=AdaptationModel(t,run['arm']);configure_trainable(model,run['arm'])
    assert [n for n,p in model.named_parameters() if p.requires_grad]==run['trainable_names']
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==run['trainable_parameters']
    assert sum(p.numel() for p in model.parameters())==run['total_parameters']
    if run['arm']!='scratch':
        for k,v in states[run['seed']].items():assert torch.equal(initial[k],v),k
    else:
        torch.manual_seed(run['seed']);fresh=AdaptationModel(t,'scratch')
        assert state_hash(fresh.state_dict())==run['initial_sha256']
    rng=np.random.default_rng(run['seed']);val_scores=[]
    for epoch in range(1,11):
        expected_order=rng.permutation(train)
        assert hashlib.sha256(expected_order.tobytes()).hexdigest()==run['permutation_sha256'][epoch-1]
        e=run['epochs'][epoch-1];assert e['training_counts']==expected_counts
        state=torch.load(d/f'epoch-{epoch}.pt',weights_only=True)
        for k,v in state.items():
            if k not in run['trainable_names']:assert torch.equal(v,initial[k]),('Frozen weight changed',run['arm'],k,epoch)
        model.load_state_dict(state);sc=oracle(predict(model,data,valid),a['target'][valid],a['task'][valid]);compare(sc,e['validation']);val_scores.append(e['validation']['macro_loss'])
    assert run['selected_epoch']==int(np.argmin(val_scores))+1
    model.load_state_dict(torch.load(d/f'epoch-{run["selected_epoch"]}.pt',weights_only=True))
    assert state_hash(model.state_dict())==run['selected_sha256']
    out=packet(d/'test.npz');assert np.array_equal(predict(model,data,test),out)
    compare(oracle(out,a['target'][test],a['task'][test]),run['test'])
    changed=[k for k,v in model.state_dict().items() if not torch.equal(v,initial[k])]
    assert changed==run['changed_names']
    if run['arm']=='adapter':assert any(n.startswith('adapter.') for n in changed)
for seed in range(3):
    pair=[x for x in r['runs'] if x['seed']==seed]
    assert all(x['permutation_sha256']==pair[0]['permutation_sha256'] for x in pair)
    assert len({x['initial_backbone_sha256'] for x in pair if x['arm']!='scratch'})==1
baselines=json.loads((E/'baselines.json').read_text())
for b in baselines:
    out=packet(E/(f'unchanged-{b["seed"]}.npz' if b['name']=='unchanged' else 'constant.npz'))
    compare(oracle(out,a['target'][test],a['task'][test]),b['test'])
    if b['name']=='unchanged':
        model=AdaptationModel(t,'freeze');model.load_state_dict(states[b['seed']]);assert np.array_equal(out,predict(model,data,test))
    else:
        for i,s in enumerate(t):
            y=a['target'][train[a['task'][train]==i]];mask=a['task'][test]==i
            if s['kind']=='number':assert np.allclose(out[mask,0],sum(map(float,y))/len(y),atol=1e-6)
            else:
                counts=np.array([1+sum(y==k) for k in range(s['classes'])]);assert np.allclose(out[mask,:s['classes']],np.log(counts/counts.sum()),atol=1e-6)
assert check174(configure_trainable,adaptation_split,select_epoch,AdaptationModel)=='PASS'
rejected=0
for policy,splitter,selector in [(lambda model,arm:[p.requires_grad_(True) for p in model.parameters()],adaptation_split,select_epoch),(configure_trainable,lambda dates:np.zeros(len(dates),dtype=int),select_epoch),(configure_trainable,adaptation_split,lambda scores:len(scores)-1)]:
    try:check174(policy,splitter,selector,AdaptationModel)
    except (AssertionError,ValueError):rejected+=1
assert rejected==3
receipt=dict(status='PASS',fresh_fits=12,test_predictions=12*len(test),baseline_predictions=4*len(test),all_epoch_checkpoints=120,independent_metrics='NUMPY_FLOAT64',raw_population_and_context='FULL_RECONSTRUCTION',frozen_weights='BYTE_EXACT_ALL_EPOCHS',selected_checkpoint_logits='EXACT',wrong_learner_functions_rejected=rejected,paired_orders='PASS',test_exposure='RETROSPECTIVE_L173')
(E/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt)
