"""Evidence contracts: positive gap always favors RDL; compare like with like."""
import math

def oriented_gap(rdl, fe, metric):
    """AUROC inputs are fractions; MAE stays in its original target units."""
    if metric not in ('AUROC','MAE'): raise ValueError('Unknown metric')
    if not all(isinstance(v,(int,float)) and math.isfinite(v) for v in [rdl,fe]):
        raise ValueError('Finite scores required')
    if metric=='AUROC' and not all(0<=v<=1 for v in [rdl,fe]):
        raise ValueError('AUROC must be a fraction')
    if metric=='MAE' and min(rdl,fe)<0: raise ValueError('Negative loss')
    return rdl-fe if metric=='AUROC' else fe-rdl

def rank_catalog(rows):
    """Rank weakest first inside one metric/variant/split/evidence family."""
    if not rows: return []
    fields=('metric','units','variant','split','source')
    contract=tuple(rows[0][f] for f in fields)
    out=[];seen=set()
    for row in rows:
        if tuple(row[f] for f in fields)!=contract: raise ValueError('Incomparable evidence')
        if row['task'] in seen: raise ValueError('Duplicate task')
        if row['status'] not in ('EXACT','PLOT_DERIVED'): raise ValueError('Unavailable evidence')
        seen.add(row['task'])
        out.append(dict(row,gap=oriented_gap(row['rdl'],row['fe'],row['metric'])))
    return sorted(out,key=lambda r:(r['gap'],r['task']))
