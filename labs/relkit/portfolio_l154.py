"""Pure portfolio contracts; learner functions also generate the real report."""
import math
import statistics


def summarize_runs(records, expected_seeds, contract):
    """Aggregate one exact local lane; missing runs are errors, never zeros."""
    fields = ('task','metric','unit','direction','split','population','protocol',
              'lane','epochs','n_queries','origin','evaluation')
    if any(k not in contract for k in fields):
        raise ValueError('Incomplete contract')
    if contract['origin'] != 'LOCAL' or contract['n_queries'] <= 0 or contract['epochs'] <= 0:
        raise ValueError('Expected a positive-sized local experiment')
    allowed = {'AUROC':('fraction','higher'), 'MAP@10':('fraction','higher'),
               'MAE':('position','lower')}
    if allowed.get(contract['metric']) != (contract['unit'],contract['direction']):
        raise ValueError('Metric, unit and direction disagree')
    seeds = [r.get('seed') for r in records]
    if len(expected_seeds)<2 or len(set(expected_seeds))!=len(expected_seeds):
        raise ValueError('Declare at least two distinct seeds for sample SD')
    if len(seeds)!=len(set(seeds)) or set(seeds)!=set(expected_seeds):
        raise ValueError('Missing, duplicate or extra seed')
    for row in records:
        if row.get('status')!='COMPLETE' or any(row.get(k)!=contract[k] for k in fields):
            raise ValueError('Incomplete run or mixed contract')
        value = row.get('score')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid metric')
        if contract['metric']!='MAE' and value>1:
            raise ValueError('Store AUROC/MAP as fractions, not percentages')
    scores = [r['score'] for r in records]
    return dict(contract,status='COMPLETE',seeds=sorted(seeds),n_seeds=len(seeds),
                mean=statistics.mean(scores),sample_sd=statistics.stdev(scores))


def compare_entries(model, baseline):
    """Orient a within-task difference; paper context cannot establish local wins."""
    for key in ('task','metric','unit','direction','split'):
        if key not in model or model[key]!=baseline.get(key):
            raise ValueError('Incompatible '+key)
    for row in (model,baseline):
        if row.get('status')!='COMPLETE' or row.get('origin') not in ('LOCAL','PUBLISHED'):
            raise ValueError('Comparison requires completed, labeled evidence')
        value=row.get('mean')
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('Invalid comparison metric')
        if row['metric'] in ('AUROC','MAP@10') and (row['unit']!='fraction' or value>1):
            raise ValueError('Invalid metric scale')
    if model['direction'] not in ('higher','lower'):
        raise ValueError('Unknown metric direction')
    local = model['origin']==baseline['origin']=='LOCAL'
    if local and (not model.get('evaluation') or model['evaluation']!=baseline.get('evaluation')):
        raise ValueError('Local comparators need identical evaluation protocols')
    if local and (not model.get('population') or model['population']!=baseline.get('population')):
        raise ValueError('Local comparators need identical query population')
    gap = (model['mean']-baseline['mean'])*(1 if model['direction']=='higher' else -1)
    return dict(task=model['task'],oriented_gap=gap,unit=model['unit'],
                relative_percent=100*gap/baseline['mean'] if baseline['mean'] else None,
                scope='LOCAL_MATCHED_DESCRIPTIVE' if local else 'PUBLISHED_CONTEXT_ONLY',
                winner=('MODEL' if gap>0 else 'BASELINE' if gap<0 else 'TIE') if local else 'NOT_ESTABLISHED')


def portfolio_verdict(entries, comparisons, required_tasks):
    """Count task coverage once; absent comparisons prevent a superiority claim."""
    tasks=[e['task'] for e in entries]
    if not required_tasks or len(set(required_tasks))!=len(required_tasks):
        raise ValueError('Required tasks must be unique and nonempty')
    if len(tasks)!=len(set(tasks)) or not set(tasks)<=set(required_tasks):
        raise ValueError('Duplicate or undeclared task')
    if any(e.get('origin')!='LOCAL' or e.get('split')!='test' or
           e.get('status') not in ('COMPLETE','INCOMPLETE','NOT_RUN') for e in entries):
        raise ValueError('Coverage requires local test entries with explicit status')
    complete={e['task'] for e in entries if e['status']=='COMPLETE'}
    compared=[c['task'] for c in comparisons]
    if len(compared)!=len(set(compared)) or not set(compared)<=complete:
        raise ValueError('Duplicate comparison or incomplete task compared')
    if any(c.get('scope') not in ('LOCAL_MATCHED_DESCRIPTIVE','PUBLISHED_CONTEXT_ONLY') for c in comparisons):
        raise ValueError('Unknown comparison scope')
    matched=[c for c in comparisons if c['scope']=='LOCAL_MATCHED_DESCRIPTIVE']
    if any(c.get('winner') not in ('MODEL','BASELINE','TIE') for c in matched):
        raise ValueError('Invalid matched comparison')
    all_covered=complete==set(required_tasks)
    all_wins=all_covered and len(matched)==len(required_tasks) and all(c['winner']=='MODEL' for c in matched)
    return dict(complete_tasks=len(complete),required_tasks=len(required_tasks),
                missing_tasks=sorted(set(required_tasks)-complete),
                benchmark_coverage='COMPLETE' if all_covered else 'INCOMPLETE',
                matched_baseline_tasks=len(matched),
                local_superiority='DESCRIPTIVE_WINS_ON_SELECTED_TASKS' if all_wins else 'NOT_ESTABLISHED',
                aggregate_metric='NOT_DEFINED_ACROSS_METRIC_FAMILIES',
                learner='PENDING_WRITTEN_DEFENSE')
