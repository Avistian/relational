"""Full saved-context replay, raw-label reconstruction and source protocol accounting."""
def audit180(root,manifest,count_fn,cost_fn,decision_fn):
    import ast,hashlib,json
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import ml_dtypes
    root=Path(root)
    for name,digest in manifest['files'].items():
        p=root/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch: '+name)
    cfg=json.loads((root/'config.json').read_text());ref=json.loads((root/'context-audit.json').read_text())
    meta=json.loads((root/'table_info.json').read_text());cols=json.loads((root/'column_index.json').read_text())
    raw=dict(np.load(root/'label-oracle.npz')); results=pd.read_parquet(root/'results.parquet');drivers=pd.read_parquet(root/'drivers.parquet')
    rawcols=dict(driver=results.driverId.to_numpy(),time=(results.date.astype('datetime64[ns]').astype('int64')//10**9).to_numpy(),status=results.statusId.to_numpy(),driver_ids=drivers.driverId.to_numpy())
    for k,v in rawcols.items():
        if not np.array_equal(v,raw[k]):raise ValueError('Raw oracle differs: '+k)
    totals={k:0 for k in ['future_cells','unmasked_query_targets','unavailable_labels','unknown_time_cells']};seeds=[];identities=None
    fields=set(totals);names={v:k for k,v in cols.items()};groups={};witness=None
    for seed in cfg['context_seeds']:
        a=dict(np.load(root/f'contexts-{seed}.npz')); assert a['node_idxs'].shape==(702,1024)
        keys=[];nodes=[];labels=[];future_queries=0;future_cells=0
        for i in range(702):
            q=a['is_targets'][i];assert q.sum()==1
            ts=a['timestamps'][i].astype('int64');pad=a['is_padding'][i];mask=a['masks'][i];node=a['node_idxs'][i];col=a['col_name_idxs'][i]
            cutoff=int(ts[q][0]);qid=int(node[q][0]);driver_row=int(a['f2p_nbr_idxs'][i][q][0,0])-meta['drivers:Db']['node_idx_offset']
            driver=int(raw['driver_ids'][driver_row]);keys.append((driver,cutoff));nodes.append(qid)
            eligible=(raw['driver']==driver)&(raw['time']>cutoff)&(raw['time']<=cutoff+30*86400)
            assert eligible.any()
            y=int((raw['status'][eligible]!=1).any());actual=int(a['boolean_values'][i][q].view(ml_dtypes.bfloat16).astype(float)[0]>0)
            assert y==actual;labels.append(y)
            counts=count_fn(ts,pad,q,col==cols['did_not_finish of driver-dnf'],mask,cutoff,30*86400)
            if set(counts)!=fields:raise ValueError('Incomplete learner audit fields')
            original=ref['records'][seed*702+i]
            assert (original['seed'],original['node_idx'],original['cutoff'])==(seed,qid,cutoff)
            for k,v in counts.items():
                if v!=original[k]:raise ValueError('Context audit differs: '+k)
                totals[k]+=int(v)
            future_queries+=counts['future_cells']>0;future_cells+=counts['future_cells']
            for j in np.flatnonzero((~pad)&(ts>cutoff)):
                table=next(k for k,v in meta.items() if v['node_idx_offset']<=node[j]<v['node_idx_offset']+v['num_nodes'])
                name=table+' / '+names[int(col[j])];groups[name]=groups.get(name,0)+1
                if witness is None:witness=dict(seed=seed,driver_id=driver,query_cutoff=cutoff,context_time=int(ts[j]),table=table,column=names[int(col[j])])
        assert len(set(keys))==702 and sorted(nodes)==list(range(97606,98308))
        keyed=dict(zip(keys,labels))
        if identities is None:identities=keyed
        assert keyed==identities
        seeds.append(dict(seed=seed,queries=702,positive_labels=sum(labels),future_cells=int(future_cells),future_queries=int(future_queries)))
    # Parse, never execute, the original source example.
    tree=ast.parse((root/'example_finetune.py').read_text())
    call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='main')
    kw={k.arg:k.value for k in call.keywords};values={}
    for k in ['seed','batch_size','seq_len','lr','wd','eval_splits','eval_freq','max_eval_steps','save_ckpt_dir','load_ckpt_path','train_tasks','num_blocks','d_model','num_heads','d_ff']:
        values[k]=ast.literal_eval(kw[k])
    expr=kw['max_steps'];assert isinstance(expr,ast.BinOp) and isinstance(expr.op,ast.Add)
    assert isinstance(expr.left,ast.BinOp) and isinstance(expr.left.op,ast.Pow)
    values['max_steps']=ast.literal_eval(expr.left.left)**ast.literal_eval(expr.left.right)+ast.literal_eval(expr.right)
    assert values['max_steps']==cfg['source_steps'] and values['batch_size']*cfg['world_size']==cfg['global_batch']
    assert values['seq_len']==cfg['seq_len'] and values['lr']==cfg['lr'] and values['wd']==cfg['wd']
    values=json.loads(json.dumps(values))
    trainer=(root/'trainer.py').read_text()
    assert 'if metric > best_metric:' in trainer and 'metrics["val"].items()' in trainer
    costs={hardware:cost_fn(8,cfg['hours_per_run'],rate) for hardware,rate in cfg['rates'].items()}
    evidence=dict(temporal='FAIL_EVENT_TIME',training_health='NOT_CHECKED',checkpoint_bytes='NOT_CHECKED',selection='NOT_CHECKED',full_population='NOT_CHECKED',source_protocol='NOT_CHECKED',gpu_only_usd=costs['A100_40GB'],cap_usd=cfg['cap_usd'],all_in_upper_usd=None,fresh_fit='NOT_RUN',written_defense='PENDING_WRITTEN_DEFENSE')
    decision=decision_fn(evidence)
    return dict(experiment=cfg['name'],audit_status='COMPLETE_SELECTED_SAVED_CONTEXT_REPLAY',contexts=2106,cell_slots=2106*1024,raw_result_rows=len(results),raw_labels_reconstructed=2106,unique_query_keys=702,per_seed=seeds,totals=totals,future_groups=groups,witness=witness,source_example=values,source_findings=dict(global_batch=256,source_steps=values['max_steps'],paper_rounded_steps=33000,example_task='rel-amazon/user-churn',target_task=cfg['task'],validation_comparison='HIGHER_IS_BETTER',checkpoint_saving_default='DISABLED',test_evaluated_during_training=True,full_split_coverage='NOT_CHECKED',trained_checkpoint_bytes='NOT_CHECKED'),costs_gpu_only_usd=costs,evidence=evidence,decision=decision,boundaries=dict(fresh_sampling='NOT_RUN',fresh_finetuning='NOT_RUN',fresh_pretraining='NOT_RUN',fresh_inference='NOT_RUN',whole_paper='NOT_RUN',historical_lineage='NOT_ESTABLISHED',historical_availability='NOT_ESTABLISHED',post_gate_training='UNVALIDATED',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED'))

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.checkpoint_l180 import temporal_counts,full_run_cost,checkpoint_decision
    E=Path(__file__).resolve().parent/'evidence/l180'
    r=audit180(E/'packet',json.loads((E/'input-manifest.json').read_text()),temporal_counts,full_run_cost,checkpoint_decision)
    (E/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['audit_status'],r['contexts'],r['totals'],r['decision'])
