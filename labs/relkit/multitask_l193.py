"""Visible contracts for honest full-task-set reporting. Python standard library only."""
import math,statistics

def choose_config(rows,order,metric):
    """Require the complete declared candidate set; use validation only, stable ties."""
    if metric not in ['AUROC','MAE'] or not order or len(set(order))!=len(order):
        raise ValueError('Invalid metric or candidate order')
    scores={}
    for row in rows:
        value=row['score']
        if row['split']!='val' or row['id'] in scores or not math.isfinite(value):
            raise ValueError('Duplicate, nonfinite or non-validation candidate')
        if value<0 or (metric=='AUROC' and value>1):raise ValueError('Invalid score')
        scores[row['id']]=value
    if set(scores)!=set(order):raise ValueError('Incomplete candidate set')
    direction=-1 if metric=='AUROC' else 1
    return min(order,key=lambda name:direction*scores[name])

def summarize_task(task,runs,seeds):
    """A missing run is an explicit row with null score, never a discarded seed."""
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Invalid seed set')
    seen=set();values=[]
    for row in runs:
        if row['task']!=task['id'] or row['seed'] in seen:raise ValueError('Run identity mismatch')
        seen.add(row['seed'])
        if row['status']=='COMPLETE':
            score=row['score']
            if score is None or not math.isfinite(score) or score<0:raise ValueError('Invalid completed score')
            if task['metric']=='AUROC' and score>1:raise ValueError('Invalid AUROC')
            values.append(score)
        elif row['status'] not in ['NOT_RUN','FAILED'] or row['score'] is not None:
            raise ValueError('Unrun/failed scores must be null')
    if seen!=set(seeds):raise ValueError('Missing or extra seed records')
    complete=len(values)==len(seeds)
    return dict(task=task['id'],status='COMPLETE' if complete else 'INCOMPLETE',expected_seeds=len(seeds),completed_seeds=len(values),mean=statistics.mean(values) if complete else None,sample_sd=statistics.stdev(values) if complete and len(values)>1 else None)

def aggregate_suite(tasks,summaries):
    """Equal task weights within each benchmark/metric; no incomplete-suite mean."""
    ids=[t['id'] for t in tasks];got=[s['task'] for s in summaries]
    if not ids or len(set(ids))!=len(ids) or len(set(got))!=len(got) or set(ids)!=set(got):
        raise ValueError('Incomplete or duplicate task set')
    by_id={s['task']:s for s in summaries};groups={}
    for group in sorted({t['group'] for t in tasks}):
        members=[t for t in tasks if t['group']==group]
        metrics={t['metric'] for t in members}
        if len(metrics)!=1 or not metrics<=set(['AUROC','MAE']):raise ValueError('Mixed/unknown units')
        metric=next(iter(metrics));values=[];completed=0;normalizers=True
        for task in members:
            row=by_id[task['id']]
            if row['status']=='COMPLETE':
                value=row['mean']
                if value is None or not math.isfinite(value) or value<0 or (metric=='AUROC' and value>1):raise ValueError('Invalid task mean')
                completed+=1
                if metric=='MAE':
                    scale=task.get('normalizer')
                    if scale is None:normalizers=False;continue
                    if not math.isfinite(scale) or scale<=0:raise ValueError('Invalid denominator')
                    value/=scale
                values.append(value)
            elif row['status']!='INCOMPLETE' or row['mean'] is not None:
                raise ValueError('Invalid incomplete task')
        complete=completed==len(members)
        status='INCOMPLETE' if not complete else ('COMPLETE' if normalizers else 'NORMALIZER_NOT_ESTABLISHED')
        groups[group]=dict(status=status,expected_tasks=len(members),complete_tasks=completed,metric='AUROC' if metric=='AUROC' else 'NORMALIZED_MAE',mean=statistics.mean(values) if status=='COMPLETE' else None)
    return groups
