"""Small decision operations for L079; benchmark auditing remains in comparison_l060."""
import math


def select_feasible(candidates, latency_budget_ms):
    """Filter by measured p95, then minimize validation loss; input order breaks ties."""
    if not math.isfinite(latency_budget_ms) or latency_budget_ms < 0:
        raise ValueError('Budget must be finite and nonnegative')
    names=[r['model'] for r in candidates]
    if len(names)!=len(set(names)):
        raise ValueError('Candidate names must be unique')
    for r in candidates:
        if not all(math.isfinite(r[k]) and r[k]>=0 for k in ['validation_loss','p95_ms']):
            raise ValueError('Loss and measured latency must be finite and nonnegative')
    feasible=[r for r in candidates if r['p95_ms']<=latency_budget_ms]
    return min(feasible,key=lambda r:r['validation_loss'])['model'] if feasible else None


def matched_datasets(panels):
    """Match underlying dataset names before comparing random and temporal ranks."""
    if set(panels)!={'random','temporal'}:
        raise ValueError('Require random and temporal panels')
    bases={}
    for regime,ids in panels.items():
        if len(ids)!=len(set(ids)) or any(not x.endswith('/'+regime) for x in ids):
            raise ValueError('Require unique dataset/regime IDs')
        bases[regime]={x.rsplit('/',1)[0] for x in ids}
    common=sorted(bases['random'] & bases['temporal'])
    if not common:
        raise ValueError('No paired underlying datasets')
    return {regime:[name+'/'+regime for name in common] for regime in panels}
