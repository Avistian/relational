"""L105: visible event-to-snapshot representation audit; no trained model.

A source ID and destination ID inhabit separate type namespaces. We never
union raw IDs or infer ingestion order from equal timestamps.
"""
# %% Imports and immutable data identity
from pathlib import Path
import hashlib
import urllib.request
import numpy as np

DATA_URL='https://snap.stanford.edu/jodie/wikipedia.csv'
DATA_SHA256='a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09'
WIDTHS=(3600,86400,604800)

# %% PROVIDED: validate the event contract

def validate_events(events):
    required=('u','v','t','a','e')
    if any(k not in events for k in required):
        raise ValueError('Require u, v, t, a (availability), e (event identity)')
    n=len(events['t'])
    if any(np.asarray(events[k]).shape!=(n,) for k in required):
        raise ValueError('Event columns must be aligned one-dimensional arrays')
    t,a=np.asarray(events['t']),np.asarray(events['a'])
    if not np.isfinite(t).all() or not np.isfinite(a).all() or np.any(a<t):
        raise ValueError('Finite timestamps and availability >= event time required')
    if np.any(np.diff(t)<0):
        raise ValueError('Sort events by time; ties remain simultaneous')
    for k in ('u','v','e'):
        if np.asarray(events[k]).dtype.kind not in 'iu' or np.any(events[k]<0):
            raise ValueError('IDs must be nonnegative integers')
    if len(np.unique(events['e']))!=n:
        raise ValueError('Event identities must be unique; repeated edges are legal')

# %% TODO 1: assign half-open time windows

def bin_index(times,width,origin=0.0):
    """Return k for origin+k*width <= time < origin+(k+1)*width."""
    t=np.asarray(times,dtype=np.float64)
    if t.ndim!=1 or not np.isfinite(t).all():
        raise ValueError('Require a finite one-dimensional timestamp array')
    if not np.isfinite(width) or width<=0 or not np.isfinite(origin):
        raise ValueError('Width must be positive and finite; origin must be finite')
    if np.any(t<origin):
        raise ValueError('This lesson requires events at or after the fixed origin')
    bins=np.floor((t-origin)/width)
    if np.any(bins>=np.iinfo(np.int64).max):
        raise ValueError('Bin index exceeds int64 range')
    return bins.astype(np.int64)

# %% TODO 2: preserve multiplicity while aggregating endpoints

def aggregate_events(events,width,origin=0.0):
    """Count edges by (bin,user,item); release only when complete and available.

    first/last times are AUDIT metadata, not input features of the binary or
    weighted snapshot representations under study. Keeping them would define
    a richer representation, still insufficient to reconstruct all events.
    """
    validate_events(events)
    b=bin_index(events['t'],width,origin)
    keys=np.rec.fromarrays([b,events['u'],events['v']],names='bin,u,v')
    unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
    n=len(unique)
    first=np.full(n,np.inf);last=np.full(n,-np.inf)
    np.minimum.at(first,inverse,events['t'])
    np.maximum.at(last,inverse,events['t'])
    occupied,window_inverse=np.unique(b,return_inverse=True)
    arrival=np.full(len(occupied),-np.inf)
    np.maximum.at(arrival,window_inverse,events['a'])
    window_arrival=arrival[np.searchsorted(occupied,unique['bin'])]
    start=origin+unique['bin']*width;end=start+width
    return {'bin':unique['bin'],'u':unique['u'],'v':unique['v'],
            'count':counts,'start':start,'end':end,'first':first,'last':last,
            'release':np.maximum(end,window_arrival)}

# %% TODO 3: enforce the snapshot's publication boundary

def visible_snapshot_mask(snapshots,query):
    """A half-open window ending at query is legal if all its rows arrived."""
    if not np.isfinite(query):
        raise ValueError('Query time must be finite')
    return snapshots['release']<=query

# %% PROVIDED: compute the complete representation audit

def choose2(counts):
    c=np.asarray(counts,dtype=np.int64)
    return int(np.sum(c*(c-1)//2))


def audit_stream(events,width,origin=0.0):
    """All event timestamps are queries, including repeats; no labels or training.

    Access counts concern the entire stream, not a node-local model sampler.
    The real-data audit assumes a=t. Delayed-arrival behavior is tested separately.
    """
    snapshots=aggregate_events(events,width,origin)
    t=np.asarray(events['t']);n=len(t)
    if not np.array_equal(events['a'],t):
        raise ValueError('Global access audit requires availability == event time')
    b=bin_index(t,width,origin);start=origin+b*width;end=start+width
    _,bin_counts=np.unique(b,return_counts=True)
    _,tie_counts=np.unique(t,return_counts=True)
    ties=choose2(tie_counts)
    # searchsorted(left) excludes the query's entire tie group.
    strict_history=np.searchsorted(t,t,side='left')
    before_window=np.searchsorted(t,start,side='left')
    through_window=np.searchsorted(t,end,side='left')
    withheld=strict_history-before_window
    nonpast=through_window-strict_history
    delay=end-t
    return {'width_seconds':int(width),'origin_seconds':float(origin),'events':n,
            'snapshot_edges':len(snapshots['count']),
            'collapsed_events':n-len(snapshots['count']),
            'collapsed_percent':100*(n-len(snapshots['count']))/n if n else 0.0,
            'weighted_count_sum':int(snapshots['count'].sum()),
            'occupied_windows':len(bin_counts),
            'total_windows':int(b.max()+1) if n else 0,
            'empty_windows':int(b.max()+1)-len(bin_counts) if n else 0,
            'tied_timestamp_pairs':ties,
            'hidden_strict_pairs':choose2(bin_counts)-ties,
            'withheld_past_sum':int(withheld.sum()),
            'nonpast_exposure_sum':int(nonpast.sum()),
            'queries_with_withheld_past':int(np.count_nonzero(withheld)),
            'queries_with_nonpast_exposure':int(np.count_nonzero(nonpast)),
            'mean_withheld_past':float(withheld.mean()) if n else 0.0,
            'mean_nonpast_exposure':float(nonpast.mean()) if n else 0.0,
            'delay_mean_seconds':float(delay.mean()) if n else 0.0,
            'delay_p95_seconds':float(np.quantile(delay,.95)) if n else 0.0,
            'delay_max_seconds':float(delay.max()) if n else 0.0,
            'unreleased_events_at_last_query':int(snapshots['count'][~visible_snapshot_mask(snapshots,float(t[-1]))].sum()) if n else 0}

# %% PROVIDED: authenticate original bytes and parse the event projection

def load_wikipedia(path):
    """Load only event identity/endpoints/time. Features and labels are excluded.

    Download is ~560 MB. The CSV header describes its variable-width trailing
    features; parsing only columns 0..2 avoids treating that header as a schema.
    No processed cache is trusted. The complete raw file is authenticated each run.
    """
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temp=path.with_suffix('.download')
        urllib.request.urlretrieve(DATA_URL,temp)
        with temp.open('rb') as f:
            if hashlib.file_digest(f,'sha256').hexdigest()!=DATA_SHA256:
                temp.unlink();raise ValueError('Downloaded dataset checksum mismatch')
        temp.replace(path)
    with path.open('rb') as f:
        actual=hashlib.file_digest(f,'sha256').hexdigest()
    if actual!=DATA_SHA256:
        raise ValueError('Raw dataset checksum mismatch; do not reuse these bytes')
    raw=np.loadtxt(path,delimiter=',',skiprows=1,usecols=(0,1,2),dtype=np.float64)
    if not np.equal(raw[:,:2],np.floor(raw[:,:2])).all():
        raise ValueError('Raw endpoint IDs are not integers')
    events={'u':raw[:,0].astype(np.int64),'v':raw[:,1].astype(np.int64),
            't':raw[:,2],'a':raw[:,2].copy(),'e':np.arange(len(raw),dtype=np.int64)}
    validate_events(events)
    if len(raw)!=157474 or len(np.unique(events['u']))!=8227 or len(np.unique(events['v']))!=1000:
        raise ValueError('Unexpected Wikipedia release population')
    return events
