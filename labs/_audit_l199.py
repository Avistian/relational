"""Execute the full selected replay; make learner decisions live dependencies."""
def audit199(root, replay, score, claim, rank, priority, gate, readiness):
    import copy,hashlib,itertools,json
    from pathlib import Path
    e=Path(root);q=e/'packet';m=json.loads((e/'input-manifest.json').read_text())
    actual={str(p.relative_to(q)) for p in q.rglob('*') if p.is_file()}
    if actual!=set(m['files']):raise ValueError('Missing or extra frozen input')
    for name,digest in m['files'].items():
        p=(q/name).resolve()
        if not p.is_relative_to(q.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=digest or m['origins'][name]['sha256']!=digest:
            raise ValueError('Input identity differs: '+name)
    read=lambda name:json.loads((q/name).read_text())
    old=replay(q/'evidence/l190/packet',read('evidence/l190/input-manifest.json'),score,claim,rank)
    if old!=read('evidence/l190/report.json'):raise ValueError('Complete L190 report differs')
    cases=read('evidence/l190/packet/cases.json');policy=read('decision-policy.json')
    weights=[]
    for w in itertools.product([1,2,3],repeat=3):
        result=priority(cases,list(w));weights.append(dict(weights=list(w),**result))
    if weights!=old['ranking']:raise ValueError('Decision weights differ from complete inherited grid')
    sensitivity=[]
    for i,c in enumerate(cases):
        for field in range(4):
            before=([c['impact']]+c['feasibility'])[field]
            for delta in [-1,1]:
                if not 1<=before+delta<=5:continue
                changed=copy.deepcopy(cases)
                if field==0:changed[i]['impact']=before+delta
                else:changed[i]['feasibility'][field-1]=before+delta
                sensitivity.append(dict(candidate=c['id'],field=['impact','data','implementation','compute'][field],before=before,after=before+delta,**priority(changed,[1,1,1])))
    receipt194=read('receipts/l194.json');receipt197=read('receipts/l197.json')
    tasks=receipt194['task_results']
    if len(tasks)!=21 or len({r['task'] for r in tasks})!=21:raise ValueError('Receipt task inventory differs')
    if any(r['fresh_mean'] is not None or r['completed_seeds']!=0 for r in tasks):raise ValueError('Missing fresh results promoted')
    if receipt194['status']!='INCOMPLETE_SOURCE_PREPROCESSING_GATE' or receipt197['fresh_training']!='NOT_RUN':raise ValueError('Inherited boundary differs')
    return dict(experiment='L199-DIRECTION-SELECTION-AUDIT',status='COMPLETE_SELECTED_EVIDENCE_AUDIT',authenticated_files=len(actual),replay=old,weight_sensitivity=weights,rating_sensitivity=sensitivity,default_priority=priority(cases,[1,1,1]),launch=gate(policy['gates']),policy=policy,memo=readiness(dict.fromkeys(['direction','hypothesis','contrast','baselines','metric','threshold','uncertainty','cost','stop','deferred','evidence','revision'],'')),later_receipts=dict(l194=dict(status=receipt194['status'],tasks=21,completed_seeds=0,fresh_scores=0),l197=dict(status=receipt197['status'],full_model_reproduction=receipt197['full_model_reproduction'],fresh_training=receipt197['fresh_training']),numerics='NOT_REPLAYED'),full_model_reproduction='INCOMPLETE_SOURCE_PREPROCESSING_GATE',fresh_training='NOT_RUN',novelty='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE',cloud_usd=0)

if __name__=='__main__':
    import json,sys
    from pathlib import Path
    e=Path(__file__).resolve().parent/'evidence/l199';sys.dont_write_bytecode=True
    import relkit
    relkit.__path__.insert(0,str(e/'packet/relkit'));sys.path.insert(0,str(e/'packet'))
    from _replay_l190 import replay190
    from relkit.checkpoint_l190 import keyed_auc,claim_gate,rank_cases
    from relkit.direction_l199 import priority,launch_gate,memo_readiness
    r=audit199(e,replay190,keyed_auc,claim_gate,rank_cases,priority,launch_gate,memo_readiness)
    (e/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'],r['replay']['prediction_rows'],r['default_priority'],r['launch'])
