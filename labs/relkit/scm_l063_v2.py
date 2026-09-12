"""L063 numerical historical-prior key parts. Source pinned in _sources_l063_v2.json.

Different RNG and bounded architecture distribution; copied-state operator parity is
checked separately. No TabPFN predictor, causal discovery, or full prior fitting.
"""
import base64, copy, hashlib, inspect, json, platform, time, types, zlib
from pathlib import Path
import numpy as np
import scipy
from scipy.special import logsumexp
from scipy.stats import truncnorm, norm
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

VERSIONS={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}
PRESETS={'smoke':dict(tasks=2,rows=128,context=64,seed=630),
         'lab':dict(tasks=12,rows=256,context=128,seed=630),
         'closer':dict(tasks=64,rows=1024,context=512,seed=6300)}

def activation_value(x, activation):
    if activation=='tanh':return np.tanh(x)
    if activation=='identity':return x
    if activation=='leaky_relu':return np.where(x>=0,x,.01*x)
    if activation=='elu':return np.expm1(np.minimum(x,0))+np.maximum(x,0)
    raise ValueError('Unknown activation')

def sparse_weights(raw, mask, dropout, first=False):
    """Independent-edge source path, scale_weights_sqrt=True; raw already has init_std.
    Historical denominator is 1-sqrt(p), not sqrt(1-p). First layer is undropped.
    """
    raw=np.asarray(raw,float);mask=np.asarray(mask)
    if raw.ndim!=2 or mask.shape!=raw.shape or not np.isin(mask,[0,1]).all() or not 0<=dropout<1:
        raise ValueError('Finite matrix, binary mask and 0<=dropout<1 required')
    if not np.isfinite(raw).all():raise ValueError('Nonfinite weight')
    return raw.copy() if first else raw*mask/(1-np.sqrt(min(dropout,.99)))

def block_weights(raw,n_blocks):
    """Released block dropout: floor-sized diagonal blocks; leftover coordinates stay zero."""
    raw=np.asarray(raw,float);h,w=raw.shape
    if not 1<=n_blocks<=min(h,w):raise ValueError('Invalid block count')
    bh,bw=h//n_blocks,w//n_blocks;mask=np.zeros_like(raw)
    for b in range(n_blocks):mask[b*bh:(b+1)*bh,b*bw:(b+1)*bw]=1
    keep=n_blocks*bh*bw/raw.size
    return raw*mask/np.sqrt(keep)

def propagate(causes, weights, biases, noises, activation):
    """Return every affine/noise output. W indexes child,parent. Layer0 has no noise.
    Later layer: a(previous) @ W.T + b + epsilon. Noise arrays are frozen per call.
    """
    h=np.asarray(causes,float)
    if h.ndim!=2 or len(weights)<2 or len(weights)!=len(biases) or len(noises)!=len(weights)-1:
        raise ValueError('Incompatible layered SCM')
    outputs=[]
    for i,(w,b) in enumerate(zip(weights,biases)):
        w=np.asarray(w,float);b=np.asarray(b,float)
        if w.ndim!=2 or w.shape[1]!=h.shape[1] or b.shape!=(w.shape[0],):raise ValueError('Layer shape')
        h=(h if i==0 else activation_value(h,activation))@w.T+b
        if i:
            eps=np.asarray(noises[i-1],float)
            if eps.shape!=h.shape:raise ValueError('Noise must have one value per row and node')
            h=h+eps
        if not np.isfinite(h).all():raise ValueError('Nonfinite generated world')
        outputs.append(h.copy())
    return outputs

def select_nodes(values, feature_nodes, target_node):
    """Distinct observed nodes, target excluded. All other generated nodes remain latent."""
    v=np.asarray(values,float);f=np.asarray(feature_nodes)
    if v.ndim!=2 or f.ndim!=1 or not len(f) or not np.issubdtype(f.dtype,np.integer):raise ValueError('Feature indices required')
    if not isinstance(target_node,(int,np.integer)) or not 0<=target_node<v.shape[1]:raise ValueError('Target index')
    if len(set(f.tolist()))!=len(f) or target_node in f or np.any(f<0) or np.any(f>=v.shape[1]):raise ValueError('Invalid or overlapping observed nodes')
    return v[:,f].copy(),v[:,target_node].copy()

def rank_labels(target, bound_indices, permutation):
    """Paper4.5/release MulticlassRank: sampled target bounds, strict >, class relabeling.
    Duplicate sampled bounds may leave empty classes. No forced balancing.
    """
    target=np.asarray(target,float);idx=np.asarray(bound_indices);perm=np.asarray(permutation)
    if target.ndim!=1 or not np.isfinite(target).all() or idx.ndim!=1 or not np.issubdtype(idx.dtype,np.integer):raise ValueError('Finite vector and integer bounds')
    if np.any(idx<0) or np.any(idx>=len(target)) or sorted(perm.tolist())!=list(range(len(idx)+1)):raise ValueError('Bounds/permutation invalid')
    bounds=target[idx];ranks=(target[:,None]>bounds[None,:]).sum(1)
    return perm[ranks].astype(np.int64),bounds.copy()

def posterior_weights(log_prior, log_context, log_query_x):
    """Finite joint-generative posterior over worlds conditional on observed query x."""
    a,b,c=[np.asarray(v,float) for v in (log_prior,log_context,log_query_x)]
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=c.shape or np.isnan(a+b+c).any():raise ValueError('One log likelihood per world')
    score=a+b+c
    if not np.isfinite(logsumexp(score)):raise ValueError('No possible world')
    return np.exp(score-logsumexp(score))

def sample_meta_scale(rng,low,high,minimum=0,rounded=False):
    """Release meta_trunc_norm_log_scaled: sigma = mu * relative_std; upper1e6.
    This is distinct from the independent mu,sigma LogUniform in paper Table5.
    """
    mu=np.exp(rng.uniform(np.log(low),np.log(high)))
    relative=np.exp(rng.uniform(np.log(.01),np.log(1.)))
    sigma=mu*relative
    value=truncnorm.rvs(-mu/sigma,(1e6-mu)/sigma,loc=mu,scale=sigma,random_state=rng)
    return minimum+(int(round(value)) if rounded else float(value))

def sample_world(rng,family='SCM',features=3):
    """One fixed world per dataset. Lab caps are explicit, not original prior sampling."""
    if family not in ['SCM','BNN']:raise ValueError('SCM or BNN')
    raw_layers=sample_meta_scale(rng,1,6,2,True);raw_width=sample_meta_scale(rng,5,130,4,True)
    layers=min(raw_layers,5);width=min(max(raw_width,1+2*features) if family=='SCM' else raw_width,32)
    causes=sample_meta_scale(rng,1,12,1,True) if family=='SCM' else features
    causes=min(causes,16)
    dropout=.9*rng.beta(rng.uniform(.1,5),rng.uniform(.1,5))
    init_std=sample_meta_scale(rng,.01,10);noise_std=sample_meta_scale(rng,.0001,.3)
    activation=str(rng.choice(['tanh','leaky_relu','elu','identity']))
    block=bool(rng.integers(2));pre_sample=bool(rng.integers(2));shared_causes=bool(rng.integers(2))
    dims=[causes]+[width]*layers+([1] if family=='BNN' else [])
    weights=[];biases=[];stds=[];blocks=[]
    for i,(din,dout) in enumerate(zip(dims[:-1],dims[1:])):
        raw=rng.normal(0,init_std,(dout,din))
        nblocks=int(rng.integers(1,int(np.ceil(np.sqrt(min(din,dout))))+1)) if block else 0
        if block:w=block_weights(raw,nblocks)
        else:w=sparse_weights(raw,rng.random(raw.shape)>dropout,dropout,first=i==0)
        weights.append(w);biases.append(rng.uniform(-1/np.sqrt(din),1/np.sqrt(din),dout));blocks.append(nblocks)
        if i:stds.append(np.abs(rng.normal(0,noise_std,dout)) if pre_sample else np.full(dout,noise_std))
    mean=rng.normal(size=causes) if shared_causes else np.zeros(causes)
    std=np.abs(rng.normal(size=causes)*mean) if shared_causes else np.ones(causes)
    return dict(family=family,features=features,activation=activation,weights=weights,biases=biases,noise_stds=stds,cause_mean=mean,cause_std=std,
                config=dict(raw_layers=raw_layers,raw_width=raw_width,layers=layers,width=width,causes=causes,dropout=dropout,init_std=init_std,noise_std=noise_std,block_dropout=block,blocks=blocks,pre_sample_noise=pre_sample,pre_sample_causes=shared_causes))

def sample_episode(world,rng,rows):
    causes=rng.normal(size=(rows,len(world['cause_mean'])))*world['cause_std']+world['cause_mean']
    noises=[rng.normal(size=(rows,len(std)))*std for std in world['noise_stds']]
    outputs=propagate(causes,world['weights'],world['biases'],noises,world['activation'])
    if world['family']=='SCM':
        values=np.concatenate(outputs[1:],1) # matches source outputs[2:] after prepended causes
        count=values.shape[1];k=world['features'];clique=bool(rng.integers(2));effect=bool(rng.integers(2))
        # Prevent the upstream in_clique/y_is_effect overlap; disclose this local rejection.
        rejected=0
        while True:
            p=(int(rng.integers(count-k))+rng.permutation(k+1)) if clique else rng.permutation(count-1)
            target=count-1 if effect else int(p[0]);features=p[1:k+1]
            if target not in features:break
            rejected+=1
        if rng.integers(2):features=np.sort(features)
        x,z=select_nodes(values,features,target)
        roles=dict(feature_nodes=features.tolist(),target_node=target,clique=clique,y_is_effect=effect,overlap_rejections=rejected)
    else:
        # Index this joint matrix through the same learner observation operator.
        values=np.column_stack([causes,outputs[-1][:,0]])
        x,z=select_nodes(values,list(range(world['features'])),world['features'])
        roles=dict(feature_nodes=list(range(world['features'])),target_node=world['features'],clique=False,y_is_effect=True,overlap_rejections=0)
    # release class_sampler_f: half binary, otherwise rounded Uniform(2,10)
    classes=2 if rng.random()<=.5 else int(round(rng.uniform(2,10)))
    idx=rng.integers(0,rows,classes-1)
    perm=np.arange(classes) if rng.random()<=.5 else rng.permutation(classes)
    if rng.random()>.5:perm=classes-1-perm
    y,bounds=rank_labels(z,idx,perm)
    return dict(x=x,y=y,continuous_target=z,values=values,causes=causes,noises=noises,outputs=outputs,roles=roles,bound_indices=idx,class_permutation=perm,bounds=bounds,classes=classes)

def fitted_probe(x,y,n,classes,shuffle_seed=None):
    """Fixed C=1 logistic diagnostic. Fit scaling+coefficients only on context rows."""
    target=y[:n].copy()
    if shuffle_seed is not None:target=target[np.random.default_rng(shuffle_seed).permutation(n)]
    unique=np.unique(target);p=np.zeros((len(y)-n,classes))
    if len(unique)==1:p[:,unique[0]]=1
    else:
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000))
        with threadpool_limits(limits=1):model.fit(x[:n],target);q=model.predict_proba(x[n:])
        p[:,model[-1].classes_]=q
    p=np.clip(p,1e-12,1);p/=p.sum(1,keepdims=True)
    return p

def paired_intervention(world,episode):
    """Delete largest last-layer edge; hold causes, all noises, biases and other weights fixed.
    Also contrast structural-noise removal against feature-only measurement corruption.
    """
    base=episode['outputs'];weights=[w.copy() for w in world['weights']]
    child,parent=np.unravel_index(np.abs(weights[-1]).argmax(),weights[-1].shape)
    old=weights[-1][child,parent];weights[-1][child,parent]=0
    changed=propagate(episode['causes'],weights,world['biases'],episode['noises'],world['activation'])
    zero_noises=[np.zeros_like(e) for e in episode['noises']]
    noiseless=propagate(episode['causes'],world['weights'],world['biases'],zero_noises,world['activation'])
    # This exact oracle is for the last continuous latent node with known parents/noise.
    expected=-activation_value(base[-2][:,parent],world['activation'])*old
    delta=changed[-1][:,child]-base[-1][:,child]
    return dict(layer=len(weights)-1,child=int(child),parent=int(parent),removed_weight=float(old),nonzero_edge=bool(old!=0),operation='edge_deletion' if old!=0 else 'no_op_zero_edge_control',
                unchanged_upstream=all(np.array_equal(a,b) for a,b in zip(base[:-1],changed[:-1])),
                last_node_mae=float(np.abs(delta).mean()),oracle_max_error=float(np.max(np.abs(delta-expected))),
                structural_noise_last_layer_mae=float(np.abs(noiseless[-1]-base[-1]).mean()),
                delta=delta.tolist(),oracle_delta=expected.tolist(),note='No label thresholds or roles are resampled. Selected edge may not affect observed y; this measures the named latent child.')

def finite_oracle_experiment(seed=639,queries=500):
    """Exactly specified two-world joint model, not the intractable full TabPFN prior.
    phi0/phi1 equally likely; X|phi Normal(-1/+1,1); Y|phi Bernoulli(.1/.9), X independent Y given phi.
    """
    rng=np.random.default_rng(seed);means=np.array([-1.,1.]);rates=np.array([.1,.9]);records=[]
    for _ in range(queries):
        phi=int(rng.integers(2));cx=rng.normal(means[phi],1,2);cy=(rng.random(2)<rates[phi]).astype(int);qx=rng.normal(means[phi]);qy=int(rng.random()<rates[phi])
        context=norm.logpdf(cx[:,None],means).sum(0)+(cy[:,None]*np.log(rates)+(1-cy[:,None])*np.log1p(-rates)).sum(0)
        qlik=norm.logpdf(qx,means)
        weights=posterior_weights(np.log([.5,.5]),context,qlik)
        ignored=posterior_weights(np.log([.5,.5]),context,np.zeros(2))
        records.append(dict(phi=phi,cx=cx.tolist(),cy=cy.tolist(),qx=float(qx),qy=qy,log_context=context.tolist(),log_query_x=qlik.tolist(),weights=weights.tolist(),p=float(weights@rates),ignored_p=float(ignored@rates)))
    loss=lambda key:float(np.mean([-np.log(r[key] if r['qy'] else 1-r[key]) for r in records]))
    return dict(seed=seed,model='phi~Bernoulli(.5); X|phi~N(2phi-1,1); Y|phi~Bernoulli(.1+.8phi)',records=records,correct_nll=loss('p'),ignored_query_x_nll=loss('ignored_p'))

def serializable(value):
    if isinstance(value,np.ndarray):
        if value.size<1000:return value.tolist()
        a=np.ascontiguousarray(value);return {'encoding':'zlib-base64','dtype':a.dtype.str,'shape':list(a.shape),'data':base64.b64encode(zlib.compress(a.tobytes(),9)).decode()}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serializable(v) for v in value]
    return value

def decode_array(value):
    if isinstance(value,dict) and value.get('encoding')=='zlib-base64':
        return np.frombuffer(zlib.decompress(base64.b64decode(value['data'])),dtype=value['dtype']).reshape(value['shape']).copy()
    return np.asarray(value)

def summarize(records):
    result=[]
    for family in ['SCM','BNN']:
        rows=[r for r in records if r['family']==family];gaps=np.array([r['shuffled_nll']-r['nll'] for r in rows])
        result.append(dict(family=family,tasks=len(rows),nll=float(np.mean([r['nll'] for r in rows])),shuffled_nll=float(np.mean([r['shuffled_nll'] for r in rows])),paired_gap_mean=float(gaps.mean()),paired_gap_sd=float(gaps.std(ddof=1)) if len(gaps)>1 else None,positive_gaps=int((gaps>0).sum())))
    return result

def run_experiment(preset='lab',output=None,config=None):
    if preset not in PRESETS:raise ValueError('smoke/lab/closer only; full prior-fitting not implemented')
    cfg=copy.deepcopy(PRESETS[preset] if config is None else config)
    if output is not None and Path(output).exists():raise FileExistsError('Fresh outputs only; no resume or overwrite')
    if not 1<=cfg['context']<cfg['rows'] or cfg['tasks']<1:raise ValueError('Invalid evaluation budget')
    start=time.perf_counter();records=[]
    for family in ['SCM','BNN']:
        for task in range(cfg['tasks']):
            seed=cfg['seed']+task; rng=np.random.default_rng(seed)
            world=sample_world(rng,family);episode=sample_episode(world,rng,cfg['rows']);n=cfg['context'];p=fitted_probe(episode['x'],episode['y'],n,episode['classes']);ps=fitted_probe(episode['x'],episode['y'],n,episode['classes'],seed+100000)
            truth=episode['y'][n:];loss=lambda a:float(-np.log(a[np.arange(len(truth)),truth]).mean())
            records.append(serializable(dict(family=family,seed=seed,n_context=n,world=world,episode={k:v for k,v in episode.items() if k not in ['values','outputs']},probabilities=p,shuffled_probabilities=ps,targets=truth,nll=loss(p),shuffled_nll=loss(ps),intervention=paired_intervention(world,episode))))
    result=dict(status='COMPLETE',operator='scm_l063_v2',config=cfg,versions=VERSIONS,records=records,summary=summarize(records),finite_oracle=finite_oracle_experiment(cfg['seed']+900,500 if preset!='closer' else 5000),elapsed_seconds=time.perf_counter()-start,paper_reproduction='INCOMPARABLE',full_pretraining='NOT_RUN',scope='Bounded numerical historical-generator key parts plus fresh logistic probes and exact two-world oracle; no pretrained PFN inference')
    if output is not None:
        Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
