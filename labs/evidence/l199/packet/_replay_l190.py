"""Complete offline replay. Learner functions are explicit dependencies, never replaced."""
def replay190(packet, manifest, score, gate, rank):
    import hashlib,json,math,statistics,xml.etree.ElementTree as ET
    from pathlib import Path
    import numpy as np
    q=Path(packet).resolve()
    actual={str(p.relative_to(q)) for p in q.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):raise ValueError('Input file set differs')
    for name,digest in manifest['files'].items():
        p=(q/name).resolve()
        if not p.is_relative_to(q) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:
            raise ValueError('Input identity differs: '+name)
    read=lambda name:json.loads((q/name).read_text())
    expected={(a,s) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)}
    seen=set();runs=[];by={a:{} for a,s in expected}
    with np.load(q/'model/prepared.npz',allow_pickle=False) as z:
        truth=[(int(a),int(b),int(y)) for (a,b),y in zip(z['test_keys'],z['y_test'])]
        if len(truth)!=702 or len(set((a,b) for a,b,_ in truth))!=702:raise ValueError('Wrong released query population')
        support_ref={s:z['train_keys'][z['support'][s]].tolist() for s in range(10)}
    for folder in ['pilot-1','full-1']:
        receipt=read('model/'+folder+'/receipt.json')
        for record in receipt['records']:
            arm,seed=record['arm'],record['seed'];identity=(arm,seed)
            if identity not in expected or identity in seen:raise ValueError('Duplicate or unexpected run')
            seen.add(identity);path=q/'model'/folder/f'{arm}-{seed}.npz'
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Prediction receipt mismatch')
            with np.load(path,allow_pickle=False) as x:
                keys=x['keys'].tolist();labels=x['label'].tolist();prob=x['probability'].tolist();support=x['support_keys'].tolist()
            if keys!=[[a,b] for a,b,y in truth] or labels!=[y for a,b,y in truth]:raise ValueError('Released labels/keys differ')
            if support!=support_ref[seed] or len(support)!=512 or len(set(map(tuple,support)))!=512:raise ValueError('Support draw differs')
            if set(map(tuple,keys))&set(map(tuple,support)):raise ValueError('Support/test overlap')
            if max(b for a,b in support)>=min(b for a,b in keys):raise ValueError('Support cutoff reaches test queries')
            if len(prob)!=702 or record['rows']!=702 or record['support']!=512:raise ValueError('Incomplete population')
            auc=score(truth,[(a,b,p) for (a,b),p in zip(keys,prob)])
            if not math.isfinite(auc) or abs(auc-record['auc'])>1e-12:raise ValueError('Metric differs from receipt')
            by[arm][seed]=auc;runs.append(dict(arm=arm,seed=seed,auc=auc,queries=len(keys),support=len(support)))
    if seen!=expected:raise ValueError('Incomplete evaluation grid')
    inherited=read('reports/l182.json');models={}
    for arm,values in sorted(by.items()):
        v=[values[s] for s in range(10)];mean=statistics.mean(v);sd=statistics.stdev(v)
        if abs(mean-inherited['models'][arm]['mean'])>1e-12:raise ValueError('Inherited mean differs')
        if abs(sd-inherited['models'][arm]['sample_sd'])>1e-12:raise ValueError('Inherited dispersion differs')
        models[arm]=dict(mean=mean,sample_sd=sd,per_seed=v)
    diff=[by['RDBPFN'][s]-by['TabICLv1.1'][s] for s in range(10)]
    paired=dict(mean=statistics.mean(diff),sample_sd=statistics.stdev(diff),positive=sum(d>0 for d in diff),per_seed=diff)
    # Independently read raw Atom, using the frozen first-submission window.
    config=read('literature/config.json');receipt=read('literature/collection.json')
    atom='{http://www.w3.org/2005/Atom}';op='{http://a9.com/-/spec/opensearch/1.1/}'
    all_ids=set();queries={};received=0
    import re
    for query in config['queries']:
        pages=[]
        for attempt in receipt['attempts']:
            if attempt['kind']!=query['name'] or not attempt.get('accepted'):continue
            raw=(q/'literature'/attempt['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=attempt['sha256']:raise ValueError('Raw response receipt mismatch')
            root=ET.fromstring(raw)
            if root.tag!=atom+'feed':raise ValueError('Not an Atom feed')
            total=int(root.findtext(op+'totalResults'));offset=int(root.findtext(op+'startIndex'))
            if offset!=attempt['start']:raise ValueError('Request/response offset differs')
            ids=[];dates_ok=True
            for e in root.findall(atom+'entry'):
                ident=e.findtext(atom+'id').rsplit('/',1)[-1];ident=re.sub(r'v[1-9][0-9]*$','',ident)
                if not re.fullmatch(r'\d{4}\.\d{4,5}',ident):raise ValueError('Unexpected modern arXiv identity')
                ids.append(ident);all_ids.add(ident)
                date=e.findtext(atom+'published');dates_ok &= '2026-07-01T00:00:00Z'<=date<'2026-10-01T00:00:00Z'
            pages.append((offset,total,ids,dates_ok));received+=len(ids)
        cursor=0;unique=set();complete=bool(pages);total=pages[0][1] if pages else None
        for offset,t,ids,dates_ok in sorted(pages):
            complete &= offset==cursor and t==total and dates_ok and len(ids)==len(set(ids)) and not bool(unique.intersection(ids))
            if not ids and total!=0:complete=False
            unique.update(ids);cursor+=len(ids)
        complete &= cursor==total
        queries[query['name']]=dict(status='COMPLETE' if complete else 'INCOMPLETE',received=cursor,total=total)
    lit=dict(queries=queries,complete_queries=sum(v['status']=='COMPLETE' for v in queries.values()),query_count=len(queries),received_records=received,unique_papers=len(all_ids),failed_attempts=sum(not a.get('ok') for a in receipt['attempts']),collection_status='COMPLETE' if all(v['status']=='COMPLETE' for v in queries.values()) else 'INCOMPLETE')
    old=read('reports/l188.json')
    if lit['unique_papers']!=old['unique_papers'] or lit['collection_status']!=old['collection_status']:raise ValueError('Literature report differs')
    cases=read('cases.json')
    if any(c['novelty']!='NOT_ESTABLISHED' or c['cost']!='NOT_ESTABLISHED' for c in cases):raise ValueError('Unestablished proposal promoted')
    ranking=rank(cases)
    checks=dict(authenticated=True,complete=True,metric_checked=True,comparable=True,novelty_reviewed=False)
    decisions={kind:gate(kind,checks) for kind in ['saved_metric','fresh_training','novelty','general_superiority']}
    required=dict(saved_metric='SUPPORTED_REPLAY',fresh_training='NOT_RUN',novelty='NOT_ESTABLISHED',general_superiority='NOT_ESTABLISHED')
    if decisions!=required:raise ValueError('Claim gate promoted evidence')
    keys={181:['selected_experiment','gnn_status','historical_identity'],182:['status','hybrid_training','whole_paper'],183:['relgt_full_reproduction','griffin_full_reproduction'],184:['selected_experiment','fresh_training'],185:['status','real_world_causality'],186:['status'],187:['production_dp','erasure'],188:['collection_status','replay_status']}
    status_receipts={str(n):{k:read(f'reports/l{n}.json')[k] for k in fields} for n,fields in keys.items()}
    return dict(experiment='L190 Q3 Research-Gap Evidence Replay',replay='COMPLETE',authenticated_files=len(actual),runs=sorted(runs,key=lambda r:(r['arm'],r['seed'])),prediction_rows=sum(r['queries'] for r in runs),models=models,paired_rdbpfn_minus_tabicl=paired,literature=lit,ranking=ranking,claims=decisions,inherited_status_receipts=status_receipts,other_lesson_numerics='NOT_REPLAYED',source_labels_and_DFS='RELEASED_ARTIFACTS_ONLY',historical_availability='NOT_ESTABLISHED',novelty='NOT_ESTABLISHED',future_experiment='NOT_RUN',checkpoint='INCOMPLETE',learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.checkpoint_l190 import keyed_auc,claim_gate,rank_cases
    p=Path(__file__).resolve().parent/'evidence/l190'
    result=replay190(p/'packet',json.loads((p/'input-manifest.json').read_text()),keyed_auc,claim_gate,rank_cases)
    (p/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['replay'],result['prediction_rows'],result['literature'],result['paired_rdbpfn_minus_tabicl']['mean'])
