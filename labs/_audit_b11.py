"""Independent complete-key saved-evidence audit. No historical reports are mutated."""
import hashlib,json,math
from pathlib import Path
import numpy as np
from relkit.baselines_b11 import first_validation_min
P=Path(__file__).resolve().parent;S=P/'sources/b11';A=S/'archive';E=P/'evidence/b11'
def keyed_metric(keys,y,pkeys,p):
    keys=[tuple(k) for k in keys];pkeys=[tuple(k) for k in pkeys]
    if not keys or len(set(keys))!=len(keys) or len(set(pkeys))!=len(pkeys) or set(keys)!=set(pkeys):raise ValueError('Complete unique entity/cutoff identities required')
    if len(y)!=len(keys) or len(p)!=len(pkeys) or not all(math.isfinite(float(v)) for v in list(y)+list(p)):raise ValueError('Finite aligned scalar values required')
    lookup=dict(zip(pkeys,p));return math.fsum(abs(float(t)-float(lookup[k])) for k,t in zip(keys,y))/len(keys)
def audit():
    ledger=json.loads((S/'source-ledger.json').read_text())
    for n,h in ledger['archive'].items():assert hashlib.sha256((A/n).read_bytes()).hexdigest()==h,n
    for n,h in ledger['extracts'].items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n
    old=json.loads((A/'evidence/l143/training.json').read_text())
    for n,h in old['source_artifact_hashes'].items():
        f=A/'evidence/l143'/n
        if f.exists():assert hashlib.sha256(f.read_bytes()).hexdigest()==h,n
    rows=[];count=0
    refs={split:np.load(A/f'evidence/l146/prepared/{split}.npz') for split in ['val','test']}
    for group,seeds,arms in [('l143',range(5),['RelGNN reconstructed']),('l146',range(3),['gnn','relgt'])]:
        for arm in arms:
            for seed in seeds:
                root=A/'evidence'/group/(f'seed-{seed}' if group=='l143' else f'fit-{arm}-{seed}')
                r=json.loads((root/'result.json').read_text());z=np.load(root/'predictions.npz')
                assert r['seed']==seed and len(r['history'])==10 and all(h['queries']==7453 for h in r['history'])
                selected=first_validation_min([h['val_mae'] for h in r['history']])+1
                assert selected==r['best_epoch' if group=='l143' else 'selected_epoch']
                row=dict(group=group,arm=arm,seed=seed,selected_epoch=selected)
                for split,n in [('val',499),('test',760)]:
                    ref=refs[split];keys=list(zip(ref['entity'],ref['cutoff']))
                    pkeys=list(zip(z[split+'_entity'],z[split+('_time' if group=='l143' else '_cutoff')]))
                    assert len(pkeys)==n
                    targets=dict(zip(keys,ref['target']))
                    assert all(float(t)==float(targets[k]) for k,t in zip(pkeys,z[split+'_target']))
                    m=keyed_metric(keys,ref['target'],pkeys,z[split+'_pred'])
                    recorded=r['scores'][split]['mae'] if group=='l143' else r['scores'][split]
                    assert abs(m-recorded)<1e-10
                    assert abs(keyed_metric(keys,ref['target'],pkeys[::-1],z[split+'_pred'][::-1])-m)<1e-12
                    row[split]=m;count+=n
                if group=='l143':row['nonfinite_gradient_entries']=sum(r['nonfinite_gradients'].values())
                else:
                    for split in ['val','test']:
                        assert r['context_sha256'][split]==hashlib.sha256((A/f'evidence/l146/prepared/{split}.npz').read_bytes()).hexdigest()
                rows.append(row)
    assert len(rows)==11 and count==13849
    summaries={}
    for arm in ['RelGNN reconstructed','gnn','relgt']:
        rr=[r for r in rows if r['arm']==arm]
        summaries[arm]={s:dict(mean=float(np.mean([r[s] for r in rr])),sample_sd=float(np.std([r[s] for r in rr],ddof=1))) for s in ['val','test']}
    # Sanity tests reject missing/duplicate/nonfinite identities rather than silently joining rows.
    rejected=0
    for keys,p in [([(1,10),(1,10)],[1.,2.]),([(1,10)],[1.]),([(1,10),(1,20)],[float('nan'),2.])]:
        try:keyed_metric([(1,10),(1,20)],[1.,2.],keys,p)
        except ValueError:rejected+=1
    assert rejected==3
    cost=json.loads((A/'evidence/l145/cost-decision.json').read_text())
    assert cost['temporal_status']=='FAIL' and cost['projected_nine_at_shallow_speed_usd']>10
    result=dict(status='COMPLETE_SAVED_EVIDENCE_REPLAY',runs=rows,summaries=summaries,verified_predictions=count,checkpoint_selections=11,invalid_packets_rejected=rejected,labels='Reused L146 task targets; agreement with L143 prediction targets checked. No new raw-database label reconstruction.',RelGT_full_gate='INCOMPLETE_TEMPORAL_AND_BUDGET_GATE',inherited_projection_usd=cost['projected_nine_at_shallow_speed_usd'],fresh_B11_benchmark_training='NOT_RUN',matched_RelGNN_RelGT_benchmark='NOT_RUN',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    E.mkdir(parents=True,exist_ok=True);(E/'replay.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2));return result
if __name__=='__main__':audit()
