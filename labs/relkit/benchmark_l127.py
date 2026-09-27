"""Learner-owned experiment contracts for the selected RelBench v1 replay."""
import math
import statistics

def experiment_contract(overrides=None):
    """Return a fresh exact release contract; changes require a separate experiment."""
    contract=dict(dataset='rel-f1',task='driver-position',epochs=10,seeds=[0,1,2,3,4],
        fanouts=[128,64],layers=2,channels=128,batch_size=512,learning_rate=.005,
        loss='L1',optimizer='Adam',selection='first_minimum_val_mae',
        preprocessing_seed=42,temporal_strategy='uniform',clamp_percentiles=[2,98])
    for key,value in (overrides or {}).items():
        if key not in contract or value!=contract[key]:
            raise ValueError(f'{key} changes the frozen release contract')
    return contract

def select_checkpoint(trace, expected_epochs=10, queries_per_epoch=7453):
    """Require full ordered epochs and select the first validation minimum."""
    if len(trace)!=expected_epochs or expected_epochs<1:
        raise ValueError('Incomplete epoch trace')
    for epoch,row in enumerate(trace,1):
        try:
            valid=(row['epoch']==epoch and row['train_queries']==queries_per_epoch
                and math.isfinite(row['val_mae']) and row['val_mae']>=0)
        except (KeyError,TypeError):valid=False
        if not valid:raise ValueError('Invalid epoch identity, coverage or validation MAE')
    return min(trace,key=lambda row:row['val_mae'])['epoch']

def summarize_seeds(records, expected_seeds=(0,1,2,3,4)):
    """Aggregate only complete unique runs; SD describes seed dispersion."""
    try:
        if len(expected_seeds)<2 or len(set(expected_seeds))!=len(expected_seeds):
            raise ValueError('Need at least two distinct expected seeds')
        if len(records)!=len(expected_seeds) or {r['seed'] for r in records}!=set(expected_seeds):
            raise ValueError('Missing, duplicate or unexpected seeds')
        if len({r['run_uuid'] for r in records})!=len(records):
            raise ValueError('Reused run identity')
        for r in records:
            if r['status']!='COMPLETE' or r['epochs']!=10 or not r['run_uuid']:
                raise ValueError('Incomplete run')
            if any(not math.isfinite(r[s]) or r[s]<0 for s in ['val','test']):
                raise ValueError('Invalid MAE')
        ordered=sorted(records,key=lambda r:r['seed'])
        return {s:dict(mean=statistics.mean(r[s] for r in ordered),
            sample_sd=statistics.stdev(r[s] for r in ordered),n=len(ordered)) for s in ['val','test']}
    except (KeyError,TypeError) as exc:
        raise ValueError('Malformed run evidence') from exc
