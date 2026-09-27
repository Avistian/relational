"""Visible budget, validation-only selection and paired evidence operators."""
import math
import statistics


def select_configuration(rows, configurations, seeds):
    """Require the exact frozen search; rank validation means, then stable ID."""
    if not configurations or not seeds or len(set(configurations))!=len(configurations) or len(set(seeds))!=len(seeds):
        raise ValueError('Nonempty unique configuration and seed lists required')
    expected={(c,s) for c in configurations for s in seeds}
    values={}
    for row in rows:
        key=(row['config'],row['seed']);value=float(row['selection_mae'])
        if key not in expected or key in values or not math.isfinite(value) or value<0:
            raise ValueError('Unplanned, duplicate or invalid validation record')
        values[key]=value
    if set(values)!=expected:raise ValueError('Finish every planned trial before selecting')
    ranking=sorted((dict(config=c,mean_validation_mae=statistics.mean(values[c,s] for s in seeds)) for c in configurations),key=lambda r:(r['mean_validation_mae'],r['config']))
    return dict(winner=ranking[0]['config'],ranking=ranking,selection='mean of first-best validation checkpoint MAEs; stable ID tie-break')


def reserve_budget(prior_reservations, workers, timeout_seconds, rate, overhead, cap):
    """Reserve all workers at their maximum lifetime before starting any."""
    vals=[*prior_reservations,timeout_seconds,rate,overhead,cap]
    if any(not math.isfinite(float(x)) or x<0 for x in vals) or not isinstance(workers,int) or workers<=0 or timeout_seconds<=0 or rate<=0:
        raise ValueError('Finite nonnegative costs and positive worker count/runtime required')
    total=math.fsum(prior_reservations)+workers*timeout_seconds*rate
    if total+overhead>cap+1e-12:raise ValueError('Dispatch would exceed the aggregate budget')
    return total


def paired_differences(default_rows, tuned_rows):
    """Align by seed; negative tuned-minus-default MAE favors tuning."""
    groups=[]
    for rows in [default_rows,tuned_rows]:
        group={}
        for row in rows:
            seed=row['seed'];value=float(row['mae'])
            if seed in group or not math.isfinite(value) or value<0:raise ValueError('Invalid seed metric')
            group[seed]=value
        groups.append(group)
    a,b=groups
    if not a or set(a)!=set(b):raise ValueError('Both conditions require identical seed identities')
    seeds=sorted(a);diffs=[b[s]-a[s] for s in seeds]
    return dict(seeds=seeds,differences=diffs,mean_difference=statistics.mean(diffs),sample_sd=statistics.stdev(diffs) if len(diffs)>1 else None)
