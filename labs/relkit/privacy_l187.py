"""L187 course privacy mechanisms; public seeded demonstrations, not production DP."""
import numpy as np
import pandas as pd


def owned_counts(db):
    """Count declared person records, not just graph nodes or incoming edges."""
    ids=db['drivers']['driverId']
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('Unique nonmissing driver IDs required')
    out=pd.DataFrame({'driverId':sorted(ids.tolist())})
    for table in ['results','qualifying','standings']:
        owners=db[table]['driverId']
        if owners.isna().any() or not owners.isin(ids).all():
            raise ValueError('Unknown or missing owner')
        out[table]=out.driverId.map(owners.value_counts()).fillna(0).astype(int)
    out['owned_rows']=1+out[['results','qualifying','standings']].sum(axis=1)
    out['driver_incident_edges']=out.owned_rows-1
    # Row nodes have one edge for each FK, regardless of edge direction in GNN code.
    out['owned_incident_edges']=3*out.results+3*out.qualifying+2*out.standings
    return out


def bounded_histogram(events, domain, cap):
    """Whole-vector L1 bound C under add/remove-owner adjacency; fixed public domain."""
    if isinstance(cap,bool) or not isinstance(cap,(int,np.integer)) or cap<1:
        raise ValueError('Positive integer cap required')
    domain=list(domain)
    if not domain or len(set(domain))!=len(domain) or pd.isna(domain).any():
        raise ValueError('Unique nonmissing public domain required')
    cols=['resultId','driverId','constructorId']
    if events[cols].isna().any().any() or events.resultId.duplicated().any():
        raise ValueError('Stable unique row IDs and complete ownership required')
    if not events.constructorId.isin(domain).all():
        raise ValueError('Category outside fixed public domain')
    kept=events.sort_values('resultId').groupby('driverId',sort=False).head(cap)
    return kept.constructorId.value_counts().reindex(domain,fill_value=0).to_numpy(dtype=np.int64)


def release_scale(cap, epsilon, repeats=1):
    """Ideal Laplace scale; basic sequential composition for independent releases."""
    for value in [cap,repeats]:
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<1:
            raise ValueError('Positive integer cap/repeats required')
    if not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('Positive finite epsilon required')
    return {'scale':float(cap/epsilon),'epsilon_total':float(repeats*epsilon)}
