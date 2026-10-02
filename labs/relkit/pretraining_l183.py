"""Visible evidence contracts. These audit observations; they do not train a model."""
def temporal_mask(cells, cutoff):
    """Strict course policy; unknown event/arrival times deny access, not prove leakage."""
    import math
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int,float)) or not math.isfinite(cutoff):
        raise ValueError('Finite numeric cutoff required')
    visible=[]
    for cell in cells:
        required={'event_time','available_at','is_label','label_end','query_target'}
        if not required <= cell.keys(): raise ValueError('Missing access metadata')
        if type(cell['is_label']) is not bool or type(cell['query_target']) is not bool:
            raise ValueError('Flags must be boolean')
        for name in ['event_time','available_at','label_end']:
            value=cell[name]
            if value is not None and (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)):
                raise ValueError('Times must be finite numbers or None')
        event,arrival,end=cell['event_time'],cell['available_at'],cell['label_end']
        if cell['is_label'] and event is not None and end is not None and end<event:
            raise ValueError('Label window ends before its reference time')
        allowed=event is not None and arrival is not None and event<=cutoff and arrival<=cutoff
        if cell['is_label']: allowed=allowed and end is not None and end<=cutoff
        visible.append(bool(allowed and not cell['query_target']))
    return visible


def keyed_mae(reference, predictions):
    """One finite target/prediction per (entity, cutoff); exact population equality."""
    import math
    def index(rows):
        out={}
        for entity,cutoff,value in rows:
            key=(entity,cutoff)
            if key in out: raise ValueError('Duplicate query key')
            if not math.isfinite(float(value)): raise ValueError('Nonfinite value')
            out[key]=float(value)
        if not out: raise ValueError('Empty population')
        return out
    truth,pred=index(reference),index(predictions)
    if truth.keys()!=pred.keys(): raise ValueError('Query populations differ')
    return math.fsum(abs(value-pred[key]) for key,value in truth.items())/len(truth)


def factorial_effect(arms, higher_is_better=False):
    """Matched seed blocks; positive interaction means extra pretraining benefit for GT."""
    import math,statistics
    required={'mp_scratch','mp_pretrained','gt_scratch','gt_pretrained'}
    if set(arms)!=required or type(higher_is_better) is not bool:
        raise ValueError('Exactly four arms and a boolean metric direction required')
    seeds=set(arms['mp_scratch'])
    if len(seeds)<2 or any(set(arm)!=seeds for arm in arms.values()):
        raise ValueError('At least two complete matched seed blocks required')
    if any(not math.isfinite(float(v)) for arm in arms.values() for v in arm.values()):
        raise ValueError('Nonfinite score')
    seeds=sorted(seeds);direction=1 if higher_is_better else -1
    mp=[direction*(arms['mp_pretrained'][s]-arms['mp_scratch'][s]) for s in seeds]
    gt=[direction*(arms['gt_pretrained'][s]-arms['gt_scratch'][s]) for s in seeds]
    interaction=[b-a for a,b in zip(mp,gt)]
    return dict(seeds=seeds,mp_gain_mean=statistics.mean(mp),gt_gain_mean=statistics.mean(gt),
                interaction_per_seed=interaction,interaction_mean=statistics.mean(interaction),
                interaction_sample_sd=statistics.stdev(interaction))


def transfer_gate(evidence):
    """Evidence inventory only: a PASS flag must be backed by an external audit."""
    required=['checkpoint_architecture','parameter_shapes','schema_semantics','task_decoder',
              'temporal_access','database_holdout','finite_gradients','validation_selection','full_cost']
    blockers=[name.upper() for name in required if evidence.get(name)!='PASS']
    return dict(status='BLOCKED' if blockers else 'READY_FOR_SEPARATELY_AUTHORIZED_PROBE',
                blockers=blockers,transfer_gain='NOT_ESTABLISHED')
