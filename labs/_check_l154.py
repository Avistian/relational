"""Adversarial contracts for the three live reporting functions."""
import copy
import math
from relkit.portfolio_l154 import summarize_runs, compare_entries, portfolio_verdict


def rejects(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid evidence accepted')


def fixture():
    contract = dict(task='toy', metric='MAE', unit='position', direction='lower',
                    split='test', population='keys:abc', protocol='recipe:a',
                    evaluation='full-population', lane='reference', epochs=10, n_queries=2, origin='LOCAL')
    records = [dict(contract, seed=s, status='COMPLETE', score=v)
               for s, v in [(0, 2.), (1, 4.), (2, 3.)]]
    return contract, records


def check_summary(fn):
    contract, rows = fixture()
    result = fn(list(reversed(rows)), [0, 1, 2], contract)
    assert result['mean'] == 3 and result['sample_sd'] == 1 and result['n_seeds'] == 3
    assert result['seeds'] == [0, 1, 2] and result['status'] == 'COMPLETE'
    rejects(lambda: fn(rows[:2], [0, 1, 2], contract))
    rejects(lambda: fn(rows + [rows[0]], [0, 1, 2], contract))
    rejects(lambda: fn(rows, [0, 0, 1], contract))
    for field, bad in [('protocol','other'),('lane','selected'),('epochs',1),
                       ('population','wrong'),('evaluation','sampled-candidates'),('split','val'),('n_queries',1),
                       ('status','PARTIAL_TIMING_PILOT'),('score',float('nan')),
                       ('score',float('inf')),('origin','PUBLISHED')]:
        changed = copy.deepcopy(rows); changed[0][field] = bad
        rejects(lambda: fn(changed, [0, 1, 2], contract))
    changed = dict(contract, metric='AUROC',unit='fraction',direction='higher')
    rejects(lambda: fn([dict(r,**changed) for r in rows], [0,1,2], changed))


def check_comparison(fn):
    contract, _ = fixture()
    model = dict(contract, mean=3., status='COMPLETE', n_seeds=3)
    baseline = dict(model, mean=4., protocol='baseline:b', model='tree')
    r = fn(model, baseline)
    assert r['oriented_gap'] == 1 and r['scope'] == 'LOCAL_MATCHED_DESCRIPTIVE'
    assert r['winner'] == 'MODEL' and r['relative_percent'] == 25
    high = dict(model, metric='AUROC',unit='fraction',direction='higher',mean=.7)
    r = fn(high, dict(high,mean=.6))
    assert math.isclose(r['oriented_gap'],.1)
    assert fn(model,dict(baseline,mean=3))['winner']=='TIE'
    assert fn(model,dict(baseline,mean=0))['relative_percent'] is None
    paper = dict(baseline,origin='PUBLISHED',population='historical:unknown')
    r = fn(model,paper)
    assert r['scope']=='PUBLISHED_CONTEXT_ONLY' and r['winner']=='NOT_ESTABLISHED'
    for field,bad in [('task','other'),('metric','RMSE'),('unit','percent'),('split','val'),('direction','higher')]:
        rejects(lambda: fn(model,dict(baseline,**{field:bad})))
    rejects(lambda: fn(model,dict(baseline,population='different')))
    rejects(lambda: fn(model,dict(baseline,evaluation='sampled-candidates')))
    rejects(lambda: fn(model,dict(baseline,status='INCOMPLETE')))
    rejects(lambda: fn(model,dict(baseline,mean=float('nan'))))


def check_verdict(fn):
    entries=[dict(task=t,status='COMPLETE',origin='LOCAL',split='test') for t in ['a','b']]
    entries.append(dict(task='c',status='INCOMPLETE',origin='LOCAL',split='test'))
    comparisons=[dict(task='a',scope='PUBLISHED_CONTEXT_ONLY',winner='NOT_ESTABLISHED')]
    r=fn(entries,comparisons,['a','b','c'])
    assert r['complete_tasks']==2 and r['required_tasks']==3
    assert r['benchmark_coverage']=='INCOMPLETE' and r['matched_baseline_tasks']==0
    assert r['local_superiority']=='NOT_ESTABLISHED' and r['learner']=='PENDING_WRITTEN_DEFENSE'
    r=fn(entries[:2],[],['a','b','c'])
    assert r['missing_tasks']==['c']
    complete=copy.deepcopy(entries);complete[2]['status']='COMPLETE'
    comparisons=[dict(task=t,scope='LOCAL_MATCHED_DESCRIPTIVE',winner='MODEL') for t in ['a','b','c']]
    r=fn(complete,comparisons,['a','b','c'])
    assert r['benchmark_coverage']=='COMPLETE' and r['local_superiority']=='DESCRIPTIVE_WINS_ON_SELECTED_TASKS'
    assert r['learner']=='PENDING_WRITTEN_DEFENSE'
    rejects(lambda: fn(entries+[entries[0]],[],['a','b','c']))
    rejects(lambda: fn(entries,comparisons,['a','b','c']))  # pilot cannot support comparison
    rejects(lambda: fn(complete,comparisons+[comparisons[0]],['a','b','c']))
    rejects(lambda: fn([dict(entries[0],split='val')],[],['a']))
    rejects(lambda: fn(entries,[],['a','a','b']))


if __name__ == '__main__':
    for check,fn in [(check_summary,summarize_runs),(check_comparison,compare_entries),(check_verdict,portfolio_verdict)]:
        check(fn)
    print('PASS: seed completeness, comparability and evidence-bounded verdicts')
