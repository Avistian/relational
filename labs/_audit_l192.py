"""Portable audit: all official query keys, conservative clocks, recorded source failure."""
import csv,hashlib,io,json,math
from pathlib import Path
from relkit.setup_l192 import keyed_rows,available_history,select_candidate

def audit(packet,manifest,keyed_rows,available_history,select_candidate):
    packet=Path(packet)
    for name,digest in manifest['files'].items():
        if hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:raise ValueError('Corrupt packet: '+name)
    rows=json.loads((packet/'queries.json').read_text());protocol=json.loads((packet/'protocol.json').read_text())
    by_split={s:[r for r in rows if r['split']==s] for s in ['train','val','test']}
    indexed={s:keyed_rows(rs) for s,rs in by_split.items()}
    if sum(map(len,indexed.values()))!=len(rows):raise ValueError('Unknown split')
    all_keys=keyed_rows(rows)
    stats={s:dict(rows=len(rs),unique_entities=len({r['entity'] for r in rs}),positives=sum(r['label'] for r in rs)) for s,rs in by_split.items()}
    times=sorted({r['cutoff'] for r in rows});clock=[]
    for t in times:
        past=[r for r in by_split['train'] if r['cutoff']<t]
        admitted=available_history(by_split['train'],t)
        clock.append(dict(cutoff=t,past_train_rows=len(past),available_train_rows=len(admitted),unavailable_past_rows=len(past)-len(admitted)))
    # This schedule audit does not materialize full relational features or infer arrival times.
    before=list(csv.DictReader(io.StringIO((packet/'before.csv').read_text())))
    after=list(csv.DictReader(io.StringIO((packet/'after.csv').read_text())))
    observed=json.loads((packet/'preprocessing.json').read_text())
    old=[int(x['category']) for x in before];new=[int(x['category']) for x in after]
    original_map={c:i for i,c in enumerate(sorted(set(observed['train_categories'])))}
    expanded_map={c:i for i,c in enumerate(sorted(set(observed['train_categories']+observed['unseen_categories'])))}
    expected_before=[original_map[c] for c in observed['known_categories']]
    expected_after=[expanded_map[c] for c in observed['known_categories']]
    if old!=expected_before or new!=expected_after:raise ValueError('Independent mapping audit failed')
    changed=sum(a!=b for a,b in zip(old,new));assert changed==3
    if any(a['number']!=b['number'] for a,b in zip(before,after)):raise ValueError('Numeric control changed')
    candidates=[f'd{d}-{b}' for d in protocol['depths'] for b in protocol['backends']]
    # Authored selection exercise; these values are not model results.
    toy=[dict(id='example-a',split='val',auc=.62),dict(id='example-b',split='val',auc=.71)]
    toy_winner=select_candidate(toy,['example-a','example-b'])
    assert toy_winner=='example-b'
    return dict(status='INCOMPLETE_SOURCE_PREPROCESSING_GATE',task='rel-trial/study-outcome',paper_target_auc=protocol['target_auc'],measured_auc=None,task_queries=len(all_keys),splits=stats,
                complete_key_collisions=0,clock_audit=clock,unavailable_past_train_rows=sum(r['unavailable_past_rows'] for r in clock),
                availability_scope='Window-end policy only; no raw database reconstruction or historical arrival proof',
                preprocessing=dict(status='FAIL',before=old,after=new,changed_known_codes=changed,numeric_control='UNCHANGED',scope='Synthetic original-source diagnostic; task effect NOT_ESTABLISHED'),
                planned_validation_candidates=len(candidates)*len(protocol['seeds']),planned_test_evaluations=len(protocol['seeds']),executed_model_evaluations=0,forecast='NOT_ESTABLISHED',cloud_usd=0,
                selection_exercise=dict(scope='AUTHORED_NOT_MODEL_SCORES',winner=toy_winner),full_feature_audit='NOT_RUN',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    P=Path(__file__).resolve().parent;E=P/'evidence/l192'
    report=audit(E/'packet',json.loads((E/'input-manifest.json').read_text()),keyed_rows,available_history,select_candidate)
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(report['status'],report['task_queries'],report['unavailable_past_train_rows'])
