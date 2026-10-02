"""Independent population/context reconstruction and saved-logit scoring."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from relkit.multitask_l173 import load_packet,MaskedCellModel,tensor_packet,predict,erase_target,task_weights,select_epoch
from _check_l173 import check173
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,default=P/'evidence/l173');parser.add_argument('--population-only',action='store_true');args=parser.parse_args();E=args.evidence.resolve()
torch.set_num_threads(1)
a,t,m=load_packet(E)
for n,h in m['inputs'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
for n,h in m['source_files'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
assert check173(erase_target,task_weights,select_epoch)=='PASS'
rejected=0
for f,w,s in [(lambda c,q:c,task_weights,select_epoch),(erase_target,lambda ids,n,arm:torch.ones(len(ids)),select_epoch),(erase_target,task_weights,lambda values:len(values)-1)]:
    try:check173(f,w,s)
    except (AssertionError,ValueError):rejected+=1
assert rejected==3
schema=json.loads((P/'evidence/l172/schema.json').read_text())
frames={name:pd.read_parquet(P/'evidence/l171/db'/f'{name}.parquet').reset_index(drop=True) for name in schema}
cell_map={};expected_y=[];expected_splits=[];expected_dates=[-1];expected_rows=[-1];expected_cols=[0];expected_numeric=[0.];expected_cat=[0];counter=1
for ti,spec in enumerate(t):
    frame=frames[spec['table']];col=spec['column'];date=pd.to_datetime(frame[schema[spec['table']]['time_col']]);v=frame[col]
    admitted=v[(date<pd.Timestamp('2005-01-01'))&v.notna()]
    if spec['kind']=='number':
        mean=sum(float(x) for x in admitted)/len(admitted)
        sd=(sum((float(x)-mean)**2 for x in admitted)/len(admitted))**.5 or 1.
        assert np.isclose(mean,spec['mean'],rtol=1e-12,atol=1e-12)
        assert np.isclose(sd,spec['scale'],rtol=1e-10,atol=1e-12)
    else:
        vocab=sorted(set(str(x) for x in admitted));assert vocab==spec['vocabulary'];codes={x:i+1 for i,x in enumerate(vocab)}
    for row,value in enumerate(v):
        if pd.isna(value) or pd.isna(date.iloc[row]):continue
        cell_map[spec['table'],row,col]=counter;counter+=1
        y=(float(value)-mean)/sd if spec['kind']=='number' else codes.get(str(value),0)
        expected_y.append(y);expected_splits.append(0 if date.iloc[row].year<2005 else 1 if date.iloc[row].year==2005 else 2)
        expected_dates.append(int(date.iloc[row].timestamp()));expected_rows.append(row);expected_cols.append(ti+1)
        expected_numeric.append(y if spec['kind']=='number' else 0.)
        expected_cat.append(int(y)+spec['category_offset'] if spec['kind']=='category' else 0)
assert np.allclose(a['target'],np.asarray(expected_y,dtype=np.float32),rtol=2e-6,atol=1e-6)
for key,expected in [('split',expected_splits),('dates',expected_dates),('rowids',expected_rows),('columns',expected_cols),('categories',expected_cat)]:assert np.array_equal(a[key],expected),key
assert np.allclose(a['numeric'],np.asarray(expected_numeric,dtype=np.float32),rtol=2e-6,atol=1e-6)
assert np.array_equal(a['cell_ids'],np.arange(1,counter))
lookups={}
for name,sc in schema.items():
    if sc['time_col'] is not None:
        pk=next(k for k,v in sc['columns'].items() if v['role']=='primary_key');lookups[name]={str(v):i for i,v in enumerate(frames[name][pk])}
# Independent dictionary reconstruction enumerates all allowed cells, then the declared slot cap.
for (table,row,col),cid in cell_map.items():
    own=sorted((key[2],v) for key,v in (( (table,row,c), cell_map.get((table,row,c),0)) for c in schema[table]['columns']) if key[2]!=col and v)
    parent=[];date=frames[table].at[row,schema[table]['time_col']]
    for fk,desc in sorted(schema[table]['columns'].items()):
        if desc['role']!='foreign_key' or desc['target_table'] not in lookups:continue
        value=frames[table].at[row,fk]
        if pd.isna(value):continue
        pt=desc['target_table'];pr=lookups[pt][str(value)];pdte=frames[pt].at[pr,schema[pt]['time_col']]
        if pd.isna(pdte) or pdte>date:continue
        parent.extend(cell_map[pt,pr,c] for c in sorted(schema[pt]['columns']) if (pt,pr,c) in cell_map and cell_map[pt,pr,c]!=cid)
    expected=[v for _,v in own[:4]];expected += [0]*(4-len(expected));expected+=parent[:4];expected += [0]*(8-len(expected))
    assert np.array_equal(a['context'][cid-1],expected),(table,row,col)
nonzero=a['context']!=0
assert not (a['context']==a['cell_ids'][:,None]).any()
assert np.all((a['dates'][a['context']]<=a['dates'][a['cell_ids'],None])|~nonzero)
# Direct model intervention: change the raw payload of a hidden cell, retain context.
data=tensor_packet(a);torch.manual_seed(19);model=MaskedCellModel(t).eval()
probe=np.linspace(0,len(a['target'])-1,80,dtype=int)
for j in probe:
    ix=torch.tensor([j]);args=(data['context'][ix],data['task'][ix],data['columns'])
    with torch.no_grad():
        before=model(*args,data['numeric'],data['categories'])
        nums=data['numeric'].clone();cats=data['categories'].clone();cid=a['cell_ids'][j]
        nums[cid]=98765.;cats[cid]=0
        after=model(*args,nums,cats)
    assert torch.equal(before,after)
if '--population-only' in sys.argv:
    print(dict(status='PASS',targets=len(a['target']),tasks=len(t),context='FULL_INDEPENDENT_RECONSTRUCTION',hidden_value_interventions=80));raise SystemExit(0)
report=json.loads((E/'report.json').read_text());test=np.flatnonzero(a['split']==2);train=np.flatnonzero(a['split']==0);validation=np.flatnonzero(a['split']==1)

def oracle(outputs,targets,ids):
    rows=[]
    for i,spec in enumerate(t):
        ix=ids==i;x=outputs[ix].astype(np.float64);y=targets[ix].astype(np.float64)
        if spec['kind']=='number':
            err=np.abs(x[:,0]-y);loss=np.where(err<=1,.5*err**2,err-.5)
            extra=dict(mae_raw=float(err.mean()*spec['scale']))
        else:
            logits=x[:,:spec['classes']];shift=logits-logits.max(1,keepdims=True)
            loss=np.log(np.exp(shift).sum(1))-shift[np.arange(len(y)),y.astype(int)]
            extra=dict(accuracy=float((logits.argmax(1)==y).mean()),unknown_targets=int((y==0).sum()))
        rows.append(dict(task=spec['name'],count=int(ix.sum()),loss=float(loss.mean()),**extra))
    return dict(macro_loss=float(np.mean([x['loss'] for x in rows])),tasks=rows)

baseline=np.zeros((len(test),max(x['classes'] for x in t)),dtype=np.float32)
for i,spec in enumerate(t):
    if spec['kind']=='category':
        y=a['target'][train][a['task'][train]==i].astype(int);freq=np.bincount(y,minlength=spec['classes'])+1
        baseline[a['task'][test]==i,:spec['classes']]=np.log(freq/freq.sum())
np.savez_compressed(E/'baseline.npz',cell_ids=a['cell_ids'][test],output=baseline)
base=oracle(baseline,a['target'][test],a['task'][test]);checks=[]
for run in report['runs']:
    d=E/f'{run["arm"]}-{run["seed"]}';p=dict(np.load(d/'test.npz'))
    assert np.array_equal(p['cell_ids'],a['cell_ids'][test]);assert np.array_equal(p['target'],a['target'][test]);assert np.array_equal(p['task'],a['task'][test])
    scores=[e['validation']['macro_loss'] for e in run['epochs']]
    assert run['selected_epoch']==min(range(3),key=lambda k:(scores[k],k))+1
    observed=oracle(p['output'],p['target'],p['task'])
    assert abs(observed['macro_loss']-run['test']['macro_loss'])<1e-6
    for actual,saved in zip(observed['tasks'],run['test']['tasks']):
        for key in ['loss','mae_raw','accuracy']:
            if key in actual:assert np.isclose(actual[key],saved[key],rtol=3e-6,atol=1e-6),(key,actual,saved)
    model=MaskedCellModel(t);model.load_state_dict(torch.load(d/f'epoch-{run["selected_epoch"]}.pt',weights_only=True))
    assert np.array_equal(predict(model,data,test),p['output'])
    for epoch in range(1,4):
        model.load_state_dict(torch.load(d/f'epoch-{epoch}.pt',weights_only=True))
        val=oracle(predict(model,data,validation),a['target'][validation],a['task'][validation])
        assert abs(val['macro_loss']-scores[epoch-1])<1e-6
        assert run['epochs'][epoch-1]['training_task_counts']==[x['counts'][0] for x in t]
    checks.append(dict(arm=run['arm'],seed=run['seed'],macro_loss=observed['macro_loss'],selected_epoch=run['selected_epoch']))
for seed in range(3):
    pair=[r for r in report['runs'] if r['seed']==seed]
    assert pair[0]['initial_sha256']==pair[1]['initial_sha256']
    assert pair[0]['permutation_sha256']==pair[1]['permutation_sha256']
(E/'baseline.json').write_text(json.dumps(base,indent=2)+'\n')
receipt=dict(status='PASS',population_targets=len(a['target']),test_predictions=len(test)*6,validation_checkpoints=18,context='FULL_INDEPENDENT_RECONSTRUCTION',hidden_value_interventions=80,wrong_learner_functions_rejected=rejected,checkpoint_predictions='EXACT',metric_oracle='NUMPY_FLOAT64',baseline=base,checks=checks,paired_initialization_and_batches='PASS')
(P/'_verify_l173_results.json' if E==(P/'evidence/l173').resolve() else E/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print({k:v for k,v in receipt.items() if k not in ['baseline','checks']})
