"""Independent pair-count scoring, model/source checks and evidence corruption tests."""
import copy,hashlib,importlib.util,itertools,json,tempfile
from pathlib import Path
import numpy as np
import torch
from relkit.curriculum_b20 import *
P=Path(__file__).resolve().parent;E=P/'evidence/b20'

def audit(root,report):
    rows=report['rows'];expected=set(itertools.product(['single','relational'],['staged','shuffled'],[0,1,2]))
    assert len(rows)==12 and {(r['family'],r['mode'],r['seed']) for r in rows}==expected,'Incomplete grid'
    ev=np.load(root/'evaluation.npz');assert ev['x'].shape==(24,32,6)
    result=[]
    for row in rows:
        pool=np.load(root/f"pool-{row['family']}-{row['seed']}.npz")
        assert array_hash(pool['x'],pool['y'])==row['pool'],'Pool hash'
        assert array_hash(ev['x'],ev['y'])==row['evaluation'],'Evaluation hash'
        order=np.array(row['order']);assert len(order)==192 and np.all((order>=0)&(order<96))
        np.testing.assert_array_equal(np.bincount(order,minlength=96),np.full(96,2))
        assert row['exposures']==np.bincount(order,minlength=96).tolist()
        if row['mode']=='staged':assert np.all(np.diff(pool['levels'][order])>=0)
        assert row['optimizer_steps']==192 and len(row['losses'])==192
        assert row['generated_cells']==int(pool['cells'].sum())
        pred=np.load(root/f"predictions-{row['name']}.npz");p=pred['p'];y=pred['y']
        np.testing.assert_array_equal(y,ev['y'][:,16:]);assert p.shape==y.shape==(24,16)
        assert np.isfinite(p).all() and np.all((p>0)&(p<1))
        auc=[]
        for a,b in zip(y,p):
            pos=b[a==1];neg=b[a==0];assert len(pos) and len(neg)
            auc.append(float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg))))
        score=float(np.mean(auc));loss=float(-(y*np.log(p.astype(float))+(1-y)*np.log1p(-p.astype(float))).mean())
        assert abs(score-row['auc'])<1e-12 and abs(loss-row['logloss'])<1e-7,'Metric mismatch'
        weights=torch.load(root/f"weights-{row['name']}.pt",weights_only=True)
        assert array_hash(*[v.numpy() for v in weights.values()])==row['final']
        model=RDBPFN();model.load_state_dict(weights);np.testing.assert_array_equal(evaluate(model,dict(ev)),p)
        result.append(dict(name=row['name'],auc=score,logloss=loss))
    for family,seed in itertools.product(['single','relational'],[0,1,2]):
        pair=[r for r in rows if r['family']==family and r['seed']==seed];paired_contract(*pair)
    return result

def model_checks():
    torch.set_num_threads(1);torch.manual_seed(81)
    spec=importlib.util.spec_from_file_location('inherited',P/'sources/b20/rdbpfn_visible.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    model=RDBPFN();source=mod.RDBPFN(width=16,heads=2,hidden=32,layers=2);source.load_state_dict(model.state_dict())
    x=torch.randn(2,32,6);y=torch.randint(2,(2,16)).float()
    a=model((x,y),16);b=source((x,y),16);torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-6)
    a.square().sum().backward();b.square().sum().backward()
    for p,q in zip(model.parameters(),source.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)
    # Other queries cannot be keys; adding/removing them cannot change a query result.
    one=torch.cat([x[:,:16],x[:,16:17]],1)
    torch.testing.assert_close(a[:,0],model((one,y),16)[:,0],atol=2e-6,rtol=1e-5)
    perm=torch.randperm(16);xp=torch.cat([x[:,perm],x[:,16:]],1)
    torch.testing.assert_close(a,model((xp,y[:,perm]),16),atol=2e-6,rtol=1e-5)
    try:model((x,torch.cat([y,y],1)),16)
    except ValueError:pass
    else:raise AssertionError('Query labels accepted')
    # A query-only change cannot affect support-fit statistics.
    changed=x.clone();changed[:,16:]=1000
    torch.testing.assert_close(normalize_support(x,16)[:,:16],normalize_support(changed,16)[:,:16],rtol=0,atol=0)
    return dict(output_gradient_source_parity=True,support_permutation=True,query_batching=True,query_labels_rejected=True,support_only_normalization=True)

def main():
    torch.set_num_threads(1);report=json.loads((E/'diagnostic.json').read_text());scored=audit(E,report)
    rejected=0
    for mutation in ['missing','duplicate','init','order','metric','pool','exposures','steps']:
        bad=copy.deepcopy(report)
        if mutation=='missing':bad['rows'].pop()
        elif mutation=='duplicate':bad['rows'][0]=copy.deepcopy(bad['rows'][1])
        elif mutation=='order':bad['rows'][0]['order'][0]=bad['rows'][0]['order'][1]
        elif mutation=='metric':bad['rows'][0]['auc']+=.02
        elif mutation=='steps':bad['rows'][0]['optimizer_steps']=1
        elif mutation=='exposures':bad['rows'][0]['exposures'][0]=99
        else:bad['rows'][0][mutation]='corrupt'
        try:audit(E,bad)
        except (AssertionError,ValueError):rejected+=1
        else:raise AssertionError('Accepted '+mutation)
    out=dict(status='PASS',independent_predictions=4608,fits=12,pairs=6,corruptions_rejected=rejected,model=model_checks(),scores=scored)
    (P/'_verify_b20_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='scores'})
if __name__=='__main__':main()
