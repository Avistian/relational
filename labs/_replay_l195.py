"""Replay every approved run; report limitations alongside each numerical claim."""
def replay195(packet, manifest, pair, verdict, scope, auc, interval):
    import hashlib,json,math,statistics
    from pathlib import Path
    import numpy as np
    q=Path(packet).resolve()
    actual={str(p.relative_to(q)) for p in q.rglob('*') if p.is_file()}
    if actual!=set(manifest['files']):raise ValueError('Incomplete or extra input files')
    for name,h in manifest['files'].items():
        p=(q/name).resolve()
        if not p.is_relative_to(q) or hashlib.sha256(p.read_bytes()).hexdigest()!=h:raise ValueError('Input hash mismatch: '+name)
    read=lambda name:json.loads((q/name).read_text())
    origins=read('origins.json')
    for name,o in origins.items():
        if manifest['files'].get(name)!=o['sha256']:raise ValueError('Original input identity changed')
    old=read('l149/training.json');fe_summary=read('l149/fe/summary.json');errors=read('l149/errors.json')
    regress={};count=0;max_target_difference=0
    for split,n in [('val',499),('test',760)]:
        runs=[];deltas=[];truth_ref=None
        for seed in range(5):
            arrays={}
            for arm in ['gnn','fe']:
                folder=q/f'l149/{arm}/{seed}';r=read(f'l149/{arm}/{seed}/result.json')
                if r['seed']!=seed:raise ValueError('Wrong seed')
                file=folder/'predictions.npz';h=hashlib.sha256(file.read_bytes()).hexdigest()
                expected=old['files'][f'labs/evidence/l149/final/lr005-full/seed-{seed}/predictions.npz'] if arm=='gnn' else r['files']['predictions.npz']
                if h!=expected:raise ValueError('Prediction receipt mismatch')
                if arm=='gnn':
                    if len(r['trace'])!=10 or any(t['train_queries']!=7453 for t in r['trace']):raise ValueError('Incomplete training receipt')
                    if r['selected_epoch']!=min(r['trace'],key=lambda t:t['val_mae'])['epoch']:raise ValueError('Selection changed')
                elif r['trials']!=10:raise ValueError('Incomplete FE search receipt')
                with np.load(file,allow_pickle=False) as z:
                    arrays[arm]={k:z[split+'_'+k].tolist() for k in ['pred','target','entity','time']}
                a=arrays[arm]
                if any(len(v)!=n for v in a.values()):raise ValueError('Missing regression queries')
                own=math.fsum(abs(y-p) for y,p in zip(a['target'],a['pred']))/n
                if abs(own-r['scores'][split])>1e-6:raise ValueError('Original score differs')
            g,f=arrays['gnn'],arrays['fe'];keys=list(zip(f['entity'],f['time']));gkeys=list(zip(g['entity'],g['time']))
            if set(keys)!=set(gkeys) or len(set(keys))!=n:raise ValueError('Query population changed')
            gy=dict(zip(gkeys,g['target']));difference=max(abs(y-gy[k]) for k,y in zip(keys,f['target']))
            if difference>1e-6:raise ValueError('Target meanings differ')
            max_target_difference=max(max_target_difference,difference)
            truth=[(a,b,y) for (a,b),y in zip(keys,f['target'])]
            if truth_ref is None:truth_ref=truth
            if truth!=truth_ref:raise ValueError('Targets/keys change across seeds')
            p=pair(truth,[(a,b,v) for (a,b),v in zip(gkeys,g['pred'])],[(a,b,v) for (a,b),v in zip(keys,f['pred'])])
            runs.append(dict(seed=seed,gnn_mae=p['candidate_mae'],fe_mae=p['baseline_mae'],gnn_advantage=statistics.mean(p['advantage'])))
            deltas.append(p['advantage']);count+=2*n
        mean_delta=np.mean(deltas,axis=0);ci=interval(mean_delta,[a for a,b,y in truth_ref],draws=2000,seed=137)
        # Original L149 reports the opposite sign: GNN loss penalty.
        previous=errors['splits'][split]['slices']['all']['interval']
        if max(abs(ci['low']+previous['high']),abs(ci['high']+previous['low']))>1e-12:raise ValueError('Bootstrap interval changed')
        for arm,ref in [('gnn',old),('fe',fe_summary)]:
            if abs(statistics.mean(r[arm+'_mae'] for r in runs)-ref['metrics'][split]['mean'])>1e-6:raise ValueError('Inherited mean differs')
        regress[split]=dict(queries=n,runs=runs,gnn_mean=statistics.mean(r['gnn_mae'] for r in runs),fe_mean=statistics.mean(r['fe_mae'] for r in runs),gnn_sd=statistics.stdev(r['gnn_mae'] for r in runs),fe_sd=statistics.stdev(r['fe_mae'] for r in runs),gnn_advantage=float(mean_delta.mean()),conditional_interval=ci,zero_margin_verdict=verdict(ci['low'],ci['high'],0),scope='Conditional descriptive driver bootstrap; fixed models/split; repeated race/time dependence not covered')
    m190=read('l182/l190-input-manifest.json')
    for name,h in m190['files'].items():
        if name.startswith('model/') or name=='reports/l182.json':
            if manifest['files'].get('l182/'+name)!=h:raise ValueError('L190 origin chain mismatch')
    with np.load(q/'l182/model/prepared.npz',allow_pickle=False) as z:
        keys=z['test_keys'].tolist();labels=z['y_test'].tolist();supports={s:z['train_keys'][z['support'][s]].tolist() for s in range(10)}
    if len(keys)!=702 or len(set(map(tuple,keys)))!=702 or len(labels)!=702:raise ValueError('Released identity incomplete')
    truth=[(a,b,y) for (a,b),y in zip(keys,labels)]
    arms=['RDBPFN','RDBPFN_single','TabICLv1.1'];by={a:{} for a in arms};runs=[];seen=set()
    for folder in ['pilot-1','full-1']:
        receipt=read(f'l182/model/{folder}/receipt.json')
        if hashlib.sha256((q/'l182/original-input-manifest.json').read_bytes()).hexdigest()!=receipt['input_manifest_sha256']:raise ValueError('Original checkpoint/data manifest differs')
        for record in receipt['records']:
            arm,seed=record['arm'],record['seed'];ident=(arm,seed)
            if arm not in arms or seed not in range(10) or ident in seen:raise ValueError('Duplicate or extra ICL run')
            seen.add(ident);p=q/f'l182/model/{folder}/{arm}-{seed}.npz'
            if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('ICL receipt hash differs')
            with np.load(p,allow_pickle=False) as z:
                k=z['keys'].tolist();y=z['label'].tolist();prob=z['probability'].tolist();support=z['support_keys'].tolist()
            if k!=keys or y!=labels or len(prob)!=702:raise ValueError('Released queries or labels changed')
            if support!=supports[seed] or len(support)!=512 or len(set(map(tuple,support)))!=512:raise ValueError('Support draw differs')
            if set(map(tuple,k))&set(map(tuple,support)) or max(b for a,b in support)>=min(b for a,b in k):raise ValueError('Support contamination')
            score=auc(truth,[(a,b,v) for (a,b),v in zip(k,prob)])
            if abs(score-record['auc'])>1e-12 or record['rows']!=702:raise ValueError('ICL score differs')
            by[arm][seed]=score;runs.append(dict(arm=arm,seed=seed,auc=score));count+=702
    if seen!={(a,s) for a in arms for s in range(10)}:raise ValueError('Incomplete ICL grid')
    models={a:dict(mean=statistics.mean(by[a].values()),sample_sd=statistics.stdev(by[a].values()),per_seed=[by[a][s] for s in range(10)]) for a in arms}
    inherited=read('l182/reports/l182.json')
    for a in arms:
        if abs(models[a]['mean']-inherited['models'][a]['mean'])>1e-12:raise ValueError('ICL aggregate differs')
    differences=[by['RDBPFN'][s]-by['TabICLv1.1'][s] for s in range(10)]
    icl=dict(queries=702,support=512,runs=sorted(runs,key=lambda x:(x['arm'],x['seed'])),models=models,advantage=dict(mean=statistics.mean(differences),sample_sd=statistics.stdev(differences),per_seed=differences,positive=sum(d>0 for d in differences)),scope='Ten support draws on one task, not independent databases; no confidence interval for cross-database generalization')
    m194=read('l194/input-manifest.json')
    for name,h in m194['files'].items():
        if manifest['files'].get('l194/packet/'+name)!=h:raise ValueError('L194 origin chain mismatch')
    r194=read('l194/report.json');tasks=read('l194/packet/tasks.json');ids={t['id'] for t in tasks};rows=r194['task_results']
    if len(ids)!=21 or len(rows)!=21 or {r['task'] for r in rows}!=ids:raise ValueError('Incomplete L194 task report')
    if r194['status']!='INCOMPLETE_SOURCE_PREPROCESSING_GATE' or r194['executed_model_evaluations']!=0:raise ValueError('Unrun status changed')
    if any(r['fresh_mean'] is not None or r['fresh_sd'] is not None or r['completed_seeds']!=0 or r['fresh_gap'] is not None or r['status']!='NOT_RUN' for r in rows):raise ValueError('Invented fresh results')
    if r194['comparator']!='AutoGluon+DFS':raise ValueError('Comparator changed')
    for r in rows:
        d=(r['paper_model']-r['paper_comparator'])*(1 if r['metric']=='AUROC' else -1)
        if abs(d-r['gap'])>1e-12:raise ValueError('Reference gap differs')
    signs=[sum(r['gap']>0 for r in rows),sum(r['gap']<0 for r in rows),sum(r['gap']==0 for r in rows)]
    if signs!=[17,3,1]:raise ValueError('Reference signs differ')
    evidence=dict(authenticated=True,complete=True,measured=True,comparable=True)
    claims={c:scope(c,evidence) for c in ['pipeline','relational_signal','architecture_cause','general_superiority','undervaluation','fresh_training']}
    if claims!={'pipeline':'SCOPED_COMPARISON','relational_signal':'NOT_ESTABLISHED','architecture_cause':'NOT_ESTABLISHED','general_superiority':'NOT_ESTABLISHED','undervaluation':'NOT_ESTABLISHED','fresh_training':'NOT_RUN'}:raise ValueError('Unsupported claim promotion')
    return dict(experiment='L195 complete thesis stress-test replay',status='COMPLETE_SELECTED_REPLAY',authenticated_files=len(actual),prediction_rows=count,regression=regress,icl=icl,rdblearn=dict(status=r194['status'],tasks=rows,reference_signs=signs,fresh_tasks=0,planned_validation=567,planned_test=63),precision=dict(scoring='float64 from stored predictions',maximum_saved_target_difference=max_target_difference,gnn_internal_target_cast='Original training used float32; common scoring uses saved FE float64 archive targets'),claims=claims,whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',independent_replication=False,learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)


def brief_markdown(r):
    g=r['regression']['test'];c=g['conditional_interval'];i=r['icl']['advantage']
    lines=['# Lesson 195 — falsification brief','', '**Verdict: narrow the thesis; broad superiority and undervaluation remain unestablished.** This is an author reference defense, not the learner submission.','', '## Replayed evidence','',f"All {r['prediction_rows']:,} stored predictions were rescored: L149 five runs per pipeline on 499 validation and 760 test queries, and L182 three arms × ten support draws × 702 test queries. Hash chains retain original receipts. Reusing these results is not an independent replication.",'',f"L149 test GNN MAE {g['gnn_mean']:.6f}; engineered-feature LightGBM MAE {g['fe_mean']:.6f}. GNN advantage (FE minus GNN loss) {g['gnn_advantage']:+.6f}, conditional 95% driver-cluster interval [{c['low']:.6f}, {c['high']:.6f}]. This interval crosses zero. It establishes neither superiority nor equivalence. Both pipelines use relational information. The basic GNN is not the boosted GNN in the paper's Figure 3.",'',f"L182 RDB-PFN minus TabICL mean {i['mean']:+.6f} AUROC; positive on {i['positive']}/10 support draws. Both consume released DFS features. These observations compare saved systems, not the presence versus absence of relational data. Same task, reused test population and uncertain historical feature availability limit generalization.",'','L194 retains all 21 tasks: 17 favorable, 3 unfavorable and 1 equal published reference signs against AutoGluon+DFS; every fresh result remains missing. The preprocessing stop is a reproducibility obstacle, not a measured performance defeat. Model reproduction INCOMPLETE_SOURCE_PREPROCESSING_GATE.','', '## Strongest objections, replies and revision conditions','', '| Claim | Strongest objection from this evidence | What survives | What would change the verdict |','|---|---|---|---|', '| C1: flattening can lose useful signal | A collision in one feature map does not show every feasible feature map loses task-relevant signal. | Information loss can occur; predictive impact is task-dependent. | Predeclared same-backend comparison of target-only versus legal relational features on new tasks; label access and budget matched. |', '| C2: learned structure recovers value | Engineered relational features have lower mean error in L149; RDBLearn offers a strong aggregation-plus-tabular alternative. | A learned prior can still help; L182 has a positive mean against TabICL on this one task. | Compare strong relational FE and learned systems with matched information access, validation search and declared compute; separate architecture and prior interventions. |', '| C3: the advantage is fair and robust | L149 uncertainty crosses zero; L182 support draws are not databases; L194 has no fresh scores. | Scoped replay is reproducible. | New held-out databases and temporal shifts, point-in-time features, complete runs, predeclared practical margin and uncertainty unit. |', '| C4: the field undervalues it | Predictive scores contain no adoption, total engineering cost or economic valuation measurement. | Undervaluation remains a research hypothesis. | Measure total human/compute/serving cost and downstream utility against a specified adoption or investment benchmark. |','', '## Guard against a moving target','','Do not repair an unfavorable result by redefining the comparator, switching the unit of analysis, dropping a task, or moving a practical margin after seeing the score. The interactive margin sweep is explicitly retrospective sensitivity, not a preregistered test. A future confirmatory study must specify its target population, minimally useful benefit, measurement unit and stop rule before test access.','','Observational rows may share entities, events and time. One-table serialization does not create independent observations. The driver bootstrap preserves within-driver dependence, but not shared race/time or new-database uncertainty. Legal event cutoffs alone do not recover unknown historical arrival times.','','## Next decisive study — NOT_RUN','','On a new untouched task set, freeze three information/representation conditions: target-only features; legal relational summaries; learned relational processing with the same available information. Tune only on validation with a declared search budget. Measure task-specific predictive benefit, uncertainty at the appropriate group/database level, and total engineering/serving cost. Predeclare the practical margin in domain units. A valid interval wholly below the required benefit challenges a practical-superiority claim; missing runs never count as a loss. This is a study design, not an executed intervention.','','## Complete published reference inventory','','| Task | Metric | RDBLearn | AutoGluon+DFS | Fresh score |','|---|---|---:|---:|---|']
    for row in r['rdblearn']['tasks']:lines.append(f"| {row['task']} | {row['metric']} | {row['paper_model']:.4f} | {row['paper_comparator']:.4f} | NOT_RUN |")
    lines+=['','## Sources and limits','','[RelBench v1](https://arxiv.org/html/2407.20060v1) · [RDB-PFN v5](https://arxiv.org/html/2603.03805v5) · [RDBLearn v1](https://arxiv.org/html/2602.18495v1). Source revisions, original training protocols and numerical deviations are preserved in packet/contracts/. Complete selected replay; no fresh training, raw database reconstruction, whole-paper reproduction, causal architecture attribution, economic valuation or learner mastery. $0 cloud/API; 1800 seconds aggregate local execution cap.','']
    return '\n'.join(lines)

if __name__=='__main__':
    import json
    from pathlib import Path
    from relkit.stress_l195 import paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval
    e=Path(__file__).resolve().parent/'evidence/l195'
    r=replay195(e/'packet',json.loads((e/'input-manifest.json').read_text()),paired_mae,interval_verdict,claim_scope,keyed_auc,cluster_interval)
    (e/'report.json').write_text(json.dumps(r,indent=2)+'\n');(e/'falsification-brief.md').write_text(brief_markdown(r))
    print(r['status'],r['prediction_rows'],r['regression']['test']['gnn_advantage'],r['icl']['advantage']['mean'])
