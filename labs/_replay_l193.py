"""Portable full-ledger audit; reference arithmetic is not fresh model inference."""
import hashlib,json,math,statistics
from pathlib import Path
from relkit.multitask_l193 import choose_config,summarize_task,aggregate_suite

def replay193(packet,manifest,choose_config,summarize_task,aggregate_suite):
    packet=Path(packet)
    expected=set(manifest['files']);actual={p.name for p in packet.iterdir() if p.is_file()}
    if expected!=actual:raise ValueError('Packet membership mismatch')
    for name,digest in manifest['files'].items():
        if Path(name).name!=name or hashlib.sha256((packet/name).read_bytes()).hexdigest()!=digest:raise ValueError('Corrupt input: '+name)
    tasks=json.loads((packet/'tasks.json').read_text());protocol=json.loads((packet/'protocol.json').read_text());runs=json.loads((packet/'runs.json').read_text());probe=json.loads((packet/'preprocessing.json').read_text())
    candidates=[f'd{d}-{b}' for d in protocol['depths'] for b in protocol['backends']]
    schedule=json.loads((packet/'validation-schedule.json').read_text());tests=json.loads((packet/'test-schedule.json').read_text())
    wanted={(t['id'],s,c) for t in tasks for s in protocol['seeds'] for c in candidates}
    got=[(r['task'],r['seed'],r['candidate']) for r in schedule]
    if len(got)!=len(set(got)) or set(got)!=wanted or any(r['status']!='NOT_RUN' or r['phase']!='validation' for r in schedule):raise ValueError('Incomplete validation schedule')
    expected_tests={(t['id'],s) for t in tasks for s in protocol['seeds']}
    if len(tests)!=len(expected_tests) or {(r['task'],r['seed']) for r in tests}!=expected_tests or any(r['status']!='NOT_RUN' or r['selected_candidate'] is not None for r in tests):raise ValueError('Invalid test schedule')
    if len(runs)!=len(expected_tests) or {(r['task'],r['seed']) for r in runs}!=expected_tests:raise ValueError('Incomplete run ledger')
    changed=[]
    before_map={c:i for i,c in enumerate(sorted(probe['train_categories']))}
    for obs in probe['observations']:
        after_map={c:i for i,c in enumerate(sorted(probe['train_categories']+[obs['unseen']]))}
        old=[before_map[c] for c in probe['known_categories']];new=[after_map[c] for c in probe['known_categories']]
        count=sum(a!=b for a,b in zip(old,new))
        if obs['before']!=old or obs['after']!=new or obs['changed_codes']!=count or obs['numeric_before']!=obs['numeric_after']:raise ValueError('Diagnostic arithmetic mismatch')
        if obs['query_alone']!=[before_map['b']] or obs['query_with_other']!=[after_map[obs['unseen']],after_map['b']]:raise ValueError('Query batch control mismatch')
        changed.append(count)
    if not any(changed) or probe['status']!='FAIL':raise ValueError('Expected recorded source gate failure')
    if hashlib.sha256((packet/'preprocessing.py').read_bytes()).hexdigest()!=probe['source_sha256']:raise ValueError('Wrong diagnostic source')
    if any(r['status']!='NOT_RUN' or r['score'] is not None for r in runs):raise ValueError('Model result fabricated after failed gate')
    summaries=[summarize_task(t,[r for r in runs if r['task']==t['id']],protocol['seeds']) for t in tasks]
    aggregates=aggregate_suite(tasks,summaries)
    reference={g:dict(tasks=sum(t['group']==g for t in tasks),mean_auc=statistics.mean(t['paper_score'] for t in tasks if t['group']==g),scope='Arithmetic on rounded published scalars, not a model replay') for g in ['relbench_classification','4dbinfer_classification']}
    # This is a labeled exercise, not nine observed model outputs.
    toy=[dict(id=c,split='val',score=.60+i*.01) for i,c in enumerate(candidates)]
    winner=choose_config(toy,candidates,'AUROC')
    return dict(status='INCOMPLETE_SOURCE_PREPROCESSING_GATE',task_count=len(tasks),validation_slots=len(schedule),test_slots=len(tests),executed_model_evaluations=0,task_results=summaries,groups=aggregates,published_reference=reference,regression_normalization='NOT_ESTABLISHED',diagnostic=dict(cases=len(changed),changed_known_codes=changed,numeric_controls='UNCHANGED',source='FRESH_ORIGINAL_PREPROCESSOR',task_effect='NOT_ESTABLISHED'),selection_exercise=dict(scope='SYNTHETIC',winner=winner),fresh_inference='NOT_RUN',pretraining='NOT_RUN',whole_paper='NOT_RUN',historical_identity='NOT_ESTABLISHED',forecast='NOT_ESTABLISHED',cloud_usd=0,learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    E=Path(__file__).resolve().parent/'evidence/l193'
    result=replay193(E/'packet',json.loads((E/'input-manifest.json').read_text()),choose_config,summarize_task,aggregate_suite)
    (E/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],result['task_count'],result['published_reference'])
