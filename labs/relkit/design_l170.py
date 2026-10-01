"""Visible, portable contracts for an evidence-backed FM design checkpoint."""
def paired_design(records):
    import math
    databases=['rel-f1','rel-trial'];contexts=[64,128,256,512,1024]
    arms=['RDBPFN','RDBPFN_single','TabICLv1.1'];values={}
    for r in records:
        db,k,arm,s=r['database'],r['context'],r['arm'],r['seed']
        value=float(r['auc']);key=(db,k,arm,s)
        if db not in databases or type(k) is not int or k not in contexts or arm not in arms or type(s) is not int or s not in range(10) or key in values or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Invalid or duplicate experiment cell')
        values[key]=value
    expected={(db,k,arm,s) for db in databases for k in contexts for arm in arms for s in range(10)}
    if set(values)!=expected:raise ValueError('Require all 300 experiment cells before comparing')
    grouped={}
    for db in databases:
        for k in contexts:
            delta=[values[db,k,'RDBPFN',s]-values[db,k,'TabICLv1.1',s] for s in range(10)]
            mean=math.fsum(delta)/10
            grouped[db+'/'+str(k)]=dict(per_seed=delta,mean=mean,sample_sd=math.sqrt(math.fsum((v-mean)**2 for v in delta)/9),positive_seeds=sum(v>0 for v in delta))
    macro={str(k):math.fsum(grouped[db+'/'+str(k)]['mean'] for db in databases)/2 for k in contexts}
    return dict(by_task_context=grouped,macro_by_context=macro,uncertainty_unit='SUPPORT_DRAWS_WITHIN_FIXED_TASK')

def adaptation_mode(target_labels, gradient_steps):
    if type(target_labels) is not int or type(gradient_steps) is not int or min(target_labels,gradient_steps)<0 or (gradient_steps>0 and target_labels==0):
        raise ValueError('Declare supervised target adaptation with nonnegative integer counts')
    if gradient_steps>0:return 'TARGET_WEIGHT_UPDATES'
    if target_labels>0:return 'LABELED_CONTEXT'
    return 'NO_TARGET_LABELS_OR_UPDATES'

def claim_gate(claim, evidence):
    keys={'artifacts','metrics','dfs','temporal','exposure','fresh_training','matched_pipeline','heldout_databases','repeatability'}
    required={
        'saved_replay':['artifacts','metrics'],
        'full_pipeline':['artifacts','metrics','dfs','temporal'],
        'fresh_pretraining':['artifacts','metrics','fresh_training','exposure'],
        'general_advantage':['artifacts','metrics','matched_pipeline','heldout_databases','temporal','exposure'],
        'exact_repeatability':['artifacts','metrics','repeatability']}
    if claim not in required or set(evidence)!=keys or any(type(v) is not bool for v in evidence.values()):
        raise ValueError('Use a known claim and the complete Boolean evidence declaration')
    missing=[key for key in required[claim] if not evidence[key]]
    return dict(status='NOT_ESTABLISHED' if missing else 'READY_FOR_REVIEW',missing=missing)
