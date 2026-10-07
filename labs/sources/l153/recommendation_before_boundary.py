"""Query-keyed ranking, sampled-negative diagnostics and evidence contracts."""
import math, statistics

def ranking_metrics(truth, predictions, k, num_candidates):
    """Macro query MAP, Hit and Recall; AP divides by min(k, |positives|)."""
    if not 0<k<=num_candidates or not truth:raise ValueError('Nonempty population and valid k required')
    def index(rows):
        out={}
        for row in rows:
            key=(int(row['entity']),int(row['time']))
            if key in out:raise ValueError('Duplicate query key')
            out[key]=row
        return out
    labels=index(truth);rankings=index(predictions)
    if labels.keys()!=rankings.keys():raise ValueError('Exact query-key coverage required')
    aps=[];hits=[];recalls=[]
    for key,row in labels.items():
        positives=set(row['positives']);ranked=list(rankings[key]['ranking'])
        if not positives or len(ranked)!=k or len(set(ranked))!=k:raise ValueError('Nonempty truth and k distinct recommendations required')
        if any(not isinstance(v,int) or not 0<=v<num_candidates for v in positives|set(ranked)):raise ValueError('Invalid candidate ID')
        count=0;terms=[]
        for rank,node in enumerate(ranked,1):
            if node in positives:count+=1;terms.append(count/rank)
        aps.append(math.fsum(terms)/min(k,len(positives)));hits.append(float(count>0));recalls.append(count/len(positives))
    return dict(n=len(aps),map=math.fsum(aps)/len(aps),hit=math.fsum(hits)/len(hits),recall=math.fsum(recalls)/len(recalls))

def negative_audit(cutoffs, negative_ids, positives, num_candidates, available_since=None):
    """Audit B×B shared comparisons. Collisions remain source behavior, not proof of leakage."""
    if not cutoffs or len(cutoffs)!=len(positives) or len(negative_ids)!=len(cutoffs):raise ValueError('Aligned nonempty batch required')
    if len(set(cutoffs))!=1:raise ValueError('Shared negatives require a common cutoff')
    if any(not 0<=v<num_candidates for v in negative_ids):raise ValueError('Candidate outside catalog')
    if any(not 0<=v<num_candidates for p in positives for v in p):raise ValueError('Positive outside catalog')
    if available_since is not None and len(available_since)!=num_candidates:raise ValueError('Complete availability vector required')
    collisions=sum(node in set(p) for p in positives for node in negative_ids)
    future=None if available_since is None else sum(available_since[node]>t for t in cutoffs for node in negative_ids)
    return dict(comparisons=len(cutoffs)*len(negative_ids),positive_collisions=collisions,future_comparisons=future,availability='NOT_OBSERVED' if available_since is None else 'OBSERVED')

def portfolio_entry(records, protocol_hash, expected_counts):
    """Reject incomplete/mixed evidence before comparing the five-run mean with Table8."""
    if len(records)!=5 or sorted(r['seed'] for r in records)!=list(range(5)):raise ValueError('Exactly seeds0–4 required')
    for r in records:
        if r['status']!='COMPLETE' or r['protocol_hash']!=protocol_hash or r['counts']!=expected_counts or r['temporal_violations']!=0:raise ValueError('Incomplete, mixed or temporally invalid evidence')
        if any(not math.isfinite(r['scores'][s]) or not 0<=r['scores'][s]<=1 for s in ['val','test']):raise ValueError('Invalid MAP')
    out={s:dict(mean=statistics.mean(r['scores'][s] for r in records),sample_sd=statistics.stdev(r['scores'][s] for r in records)) for s in ['val','test']}
    out.update(status='COMPLETE',paper_target=.107,tolerance=.02,paper_comparison='CLOSE' if abs(out['test']['mean']-.107)<=.02 else 'OUTSIDE_TOLERANCE',historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
    return out
