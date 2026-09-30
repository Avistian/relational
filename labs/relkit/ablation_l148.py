"""Three live experiment contracts; NumPy-only for the portable lab."""
import numpy as np
ARMS=['full','encoder','messages','history','combined']

def history_mask(cutoffs,times,owners,undated,roots,window):
    cutoffs=np.asarray(cutoffs);times=np.asarray(times);owners=np.asarray(owners,dtype=int)
    undated=np.asarray(undated,dtype=bool);roots=np.asarray(roots,dtype=bool)
    if window<0 or any(x.shape!=times.shape for x in [owners,undated,roots]):raise ValueError('Invalid window or shapes')
    if np.any(owners<0) or np.any(owners>=len(cutoffs)):raise ValueError('Invalid query owner')
    upper=cutoffs[owners]
    return undated | ((times<=upper)&(roots|(times>=upper-window)))

def keyed_mae(reference_keys,targets,prediction_keys,predictions):
    a=[tuple(k) for k in reference_keys];b=[tuple(k) for k in prediction_keys]
    y=np.asarray(targets,dtype=float);p=np.asarray(predictions,dtype=float)
    if len(set(a))!=len(a) or len(set(b))!=len(b) or set(a)!=set(b):raise ValueError('Require unique complete keys')
    if y.shape!=(len(a),) or p.shape!=(len(b),) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Invalid values')
    lookup=dict(zip(b,p));return float(np.mean(np.abs(y-np.array([lookup[k] for k in a]))))

def interaction_summary(full,encoder,messages,combined):
    seeds=sorted(full)
    if len(seeds)<2 or any(set(x)!=set(full) for x in [encoder,messages,combined]):raise ValueError('Require matching paired seeds')
    values=np.array([[x[s] for s in seeds] for x in [full,encoder,messages,combined]],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite score')
    diff=values[3]-values[1]-values[2]+values[0]
    return dict(seeds=seeds,differences=diff.tolist(),mean=float(diff.mean()),sample_sd=float(diff.std(ddof=1)))
