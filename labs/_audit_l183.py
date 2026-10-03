"""Offline replay. Receives the learner's live functions; never substitutes a solution."""
def audit183(root, manifest, temporal_mask, keyed_mae, factorial_effect, transfer_gate):
    import hashlib,json,math,statistics
    from pathlib import Path
    import numpy as np
    root=Path(root)
    for name,digest in manifest['files'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('Input identity mismatch: '+name)
    def read(name): return json.loads((root/'labs/evidence'/name).read_text())
    runs=[];total=0
    for arm in ['gnn','relgt']:
        for seed in range(3):
            name=f'l146/fit-{arm}-{seed}';r=read(name+'/result.json')
            if r['arm']!=arm or r['seed']!=seed or r['pilot']: raise ValueError('Wrong run identity')
            history=r['history']
            if [h['epoch'] for h in history]!=list(range(1,11)) or any(h['queries']!=7453 for h in history):
                raise ValueError('Incomplete full-data training history')
            selected=min(history,key=lambda h:h['val_mae'])['epoch']
            if r['selected_epoch']!=selected: raise ValueError('Checkpoint selection differs')
            scores={}
            with np.load(root/'labs/evidence'/name/'predictions.npz',allow_pickle=False) as z:
                for split,n in [('val',499),('test',760)]:
                    with np.load(root/f'labs/evidence/l146/prepared/{split}.npz',allow_pickle=False) as ref:
                        truth=list(zip(ref['entity'].tolist(),ref['cutoff'].tolist(),ref['target'].tolist()))
                    preds=list(zip(z[split+'_entity'].tolist(),z[split+'_cutoff'].tolist(),z[split+'_pred'].tolist()))
                    labels=list(zip(z[split+'_entity'].tolist(),z[split+'_cutoff'].tolist(),z[split+'_target'].tolist()))
                    if len(preds)!=n or len(truth)!=n: raise ValueError('Wrong population size')
                    if keyed_mae(truth,labels)!=0: raise ValueError('Saved target mismatch')
                    scores[split]=keyed_mae(truth,preds)
                    if abs(scores[split]-r['scores'][split])>1e-10: raise ValueError('Recorded score differs')
                    total+=n
            runs.append(dict(arm=arm,seed=seed,scores=scores,selected_epoch=selected))
    summary={}
    for arm in ['gnn','relgt']:
        summary[arm]={}
        for split in ['val','test']:
            values=[r['scores'][split] for r in runs if r['arm']==arm]
            summary[arm][split]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),values=values)
    paired={}
    for split in ['val','test']:
        diffs=[a-b for a,b in zip(summary['gnn'][split]['values'],summary['relgt'][split]['values'])]
        paired[split]=dict(gnn_minus_relgt=diffs,mean=statistics.mean(diffs),sample_sd=statistics.stdev(diffs))
    # Recompute inherited cost scenarios, preserving their original resource rates.
    gp=read('l164/pilot.json');gb=read('l164/budget.json');gc=read('l164/cost-decision.json')
    step=max(e['sample_seconds']+e['compute_seconds'] for e in gp['events'] if e['phase']=='train')
    valid=math.fsum(e['sample_seconds']+e['compute_seconds'] for e in gp['events'] if e['phase']=='valid')
    seconds=sum(10*(200*(size//256)*step+100*valid+101*valid*702/566) for size in [512,4096])
    griffin_cost=seconds*gb['rate_usd_per_second']*1.25
    if abs(griffin_cost-gc['safety_adjusted_compute_usd'])>1e-9: raise ValueError('Griffin cost mismatch')
    rp=read('l145/pilot-1/result.json');rc=read('l145/cost-decision.json');ra=read('l145/prepared/audit.json')
    h=rp['history'][0];seconds_per_epoch=h['train_seconds']/3*math.ceil(7453/256)+h['val_seconds']*math.ceil(499/256)
    relgt_cost=seconds_per_epoch*100*9*.00022572
    if abs(relgt_cost-rc['projected_nine_at_shallow_speed_usd'])>1e-9: raise ValueError('RelGT cost mismatch')
    if rc['decision']!='STOP' or gc['decision']!='STOP' or ra['temporal_status']!='FAIL': raise ValueError('Inherited stop changed')
    # Audit a saved violation witness only: no full token-cache rescan is claimed.
    witness=ra['split_audits']['train']['examples'][0]
    if witness['actual_time']<=witness['key'][1]: raise ValueError('Temporal witness is not future-dated')
    fixture=[dict(event_time=5,available_at=6,is_label=False,label_end=None,query_target=False),
             dict(event_time=12,available_at=6,is_label=False,label_end=None,query_target=False),
             dict(event_time=5,available_at=6,is_label=True,label_end=12,query_target=False),
             dict(event_time=5,available_at=None,is_label=False,label_end=None,query_target=False),
             dict(event_time=5,available_at=6,is_label=False,label_end=None,query_target=True)]
    visible=temporal_mask(fixture,10)
    if visible!=[True,False,False,False,False]: raise ValueError('Visibility contract failed')
    arms={'mp_scratch':{0:4.,1:4.1,2:3.9},'mp_pretrained':{0:3.6,1:3.8,2:3.4},'gt_scratch':{0:3.9,1:4.,2:3.8},'gt_pretrained':{0:3.2,1:3.4,2:3.0}}
    interaction=factorial_effect(arms)
    if abs(interaction['interaction_mean']-.3)>1e-12: raise ValueError('Factorial contrast failed')
    evidence=dict(checkpoint_architecture='FAIL',parameter_shapes='NOT_CHECKED',schema_semantics='NOT_CHECKED',task_decoder='NOT_CHECKED',temporal_access='NOT_CHECKED',database_holdout='NOT_CHECKED',finite_gradients='NOT_CHECKED',validation_selection='NOT_CHECKED',full_cost='NOT_CHECKED')
    gate=transfer_gate(evidence)
    if gate['status']!='BLOCKED' or gate['transfer_gain']!='NOT_ESTABLISHED': raise ValueError('Unproved transfer admitted')
    return dict(status='COMPLETE_SELECTED_SAVED_PREDICTION_REPLAY',verified_predictions=total,runs=runs,summary=summary,paired=paired,
                inherited_cost_scenarios_usd=dict(relgt=relgt_cost,griffin=griffin_cost),cost_basis='Inherited pilot timings and original rates; not current prices or new measurements',
                inherited_temporal_counts={s:{k:a[k] for k in ['queries','future_token_occurrences','affected_queries']} for s,a in ra['split_audits'].items()},
                saved_temporal_witness=witness,full_token_cache_rescan='NOT_RUN',raw_label_reconstruction='NOT_RUN',
                synthetic_visibility=visible,synthetic_factorial=interaction,hybrid_evidence=evidence,hybrid_gate=gate,
                relgt_full_reproduction='INCOMPLETE_TEMPORAL_AND_BUDGET_GATES',griffin_full_reproduction='INCOMPLETE_BUDGET_GATE',
                fresh_training='NOT_RUN',fresh_inference='NOT_RUN',hybrid_pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.pretraining_l183 import temporal_mask,keyed_mae,factorial_effect,transfer_gate
    P=Path(__file__).resolve().parent;E=P/'evidence/l183'
    report=audit183(E/'packet',json.loads((E/'input-manifest.json').read_text()),temporal_mask,keyed_mae,factorial_effect,transfer_gate)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['verified_predictions'],report['inherited_cost_scenarios_usd'])
