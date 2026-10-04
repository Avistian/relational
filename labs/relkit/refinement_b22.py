"""B22 course mechanism: random fixed weights, never the pretrained RefineICL model."""
import numpy as np


def attention_read(query, keys, values):
    """Rows in query read support keys/values with stable scaled dot-product attention."""
    q,k,v=[np.asarray(x,dtype=np.float64) for x in (query,keys,values)]
    if any(x.ndim!=2 or not np.isfinite(x).all() for x in (q,k,v)) or not k.shape[0] or not q.shape[1] or q.shape[1]!=k.shape[1] or k.shape[0]!=v.shape[0]:
        raise ValueError('Finite matrices with compatible nonempty keys required')
    score=q@k.T/np.sqrt(q.shape[1]);score-=score.max(axis=1,keepdims=True)
    weight=np.exp(score);weight/=weight.sum(axis=1,keepdims=True)
    return weight@v


def replace_support(before, after, n_support, mode):
    """Intervene after a block. Query rows retain their actual block outputs."""
    b,a=[np.asarray(x,dtype=np.float64) for x in (before,after)]
    if b.ndim!=2 or b.shape!=a.shape or not isinstance(n_support,(int,np.integer)) or not 0<n_support<len(b) or mode not in ('normal','identity','skip','permute'):
        raise ValueError('Invalid states, split or intervention')
    out=a.copy()
    if mode=='identity':out[:n_support]=a[:n_support].copy()
    if mode=='skip':out[:n_support]=b[:n_support]
    if mode=='permute':
        updates=a[:n_support]-b[:n_support]
        out[:n_support]=b[:n_support]+np.roll(updates,1,axis=0)
    return out


def paired_effect(baseline_logits, changed_logits, labels):
    """Changed minus normal: positive CE is worse; positive accuracy is better."""
    a,b=[np.asarray(x,dtype=np.float64) for x in (baseline_logits,changed_logits)];y=np.asarray(labels)
    if a.ndim!=2 or a.shape!=b.shape or y.shape!=(len(a),) or not len(a) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=a.shape[1]) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Aligned finite logits and integer class labels required')
    ce=[];accuracy=[]
    for logits in (a,b):
        z=logits-logits.max(axis=1,keepdims=True)
        ce.append(float(np.mean(np.log(np.exp(z).sum(axis=1))-z[np.arange(len(y)),y])))
        accuracy.append(float(np.mean(logits.argmax(axis=1)==y)))
    return dict(baseline_ce=ce[0],changed_ce=ce[1],delta_ce=ce[1]-ce[0],baseline_accuracy=accuracy[0],changed_accuracy=accuracy[1],delta_accuracy_pp=100*(accuracy[1]-accuracy[0]))


def parameters(seed):
    rng=np.random.default_rng(seed)
    shapes={'embed':(2,8),'labels':(2,8),'head':(8,2)}
    for layer in range(3):
        for name in ['q','k','v','gate','out']:shapes[f'{layer}_{name}']=(8,8)
    return {key:rng.normal(0,.25,shape) for key,shape in shapes.items()}


def episodes():
    data=[]
    for family in ['linear','xor','radial']:
        for replicate in range(8):
            identity=len(data);rng=np.random.default_rng(22000+identity)
            def sample(each):
                xs=[];ys=[]
                for cls in range(2):
                    for _ in range(each):
                        while True:
                            x=rng.uniform(-1,1,2)
                            label=int(x[0]>0) if family=='linear' else int(x[0]*x[1]>0) if family=='xor' else int(x@x>.5)
                            if label==cls:break
                        xs.append(x.tolist());ys.append(cls)
                order=rng.permutation(len(ys))
                return np.asarray(xs)[order].tolist(),np.asarray(ys)[order].tolist()
            sx,sy=sample(6);qx,qy=sample(8)
            data.append(dict(id=identity,family=family,replicate=replicate,generator_seed=22000+identity,support_x=sx,support_y=sy,query_x=qx,query_y=qy))
    return data


def forward(weights, support_x, support_y, query_x, mode='normal'):
    """Query answers are deliberately absent from this prediction interface."""
    w={k:np.asarray(v,dtype=np.float64) for k,v in weights.items()}
    sx,qx=np.asarray(support_x),np.asarray(query_x);sy=np.asarray(support_y,dtype=int);n=len(sx)
    h=np.concatenate([sx,qx])@w['embed'];h[:n]+=w['labels'][sy]
    states=[h.tolist()];block_inputs=[];block_outputs=[]
    def norm(a):return (a-a.mean(axis=1,keepdims=True))/np.sqrt(a.var(axis=1,keepdims=True)+1e-5)
    for layer in range(3):
        before=h.copy();z=norm(h)
        read=attention_read(z@w[f'{layer}_q'],z[:n]@w[f'{layer}_k'],z[:n]@w[f'{layer}_v'])
        gate=z@w[f'{layer}_gate'];gate=gate/(1+np.exp(-gate))
        after=h+(norm(read)*gate)@w[f'{layer}_out']
        block_inputs.append(before.tolist());block_outputs.append(after.tolist())
        h=replace_support(before,after,n,mode) if layer==1 else after
        states.append(h.tolist())
    return dict(logits=(h[n:]@w['head']).tolist(),states=states,block_inputs=block_inputs,block_outputs=block_outputs)


def experiment():
    data=episodes();conditions=[];paired=[];weights={}
    for seed in [0,1,2]:
        w=parameters(seed);weights[str(seed)]={k:v.tolist() for k,v in w.items()}
        for d in data:
            runs={}
            for mode in ['normal','identity','skip','permute']:
                runs[mode]=forward(w,d['support_x'],d['support_y'],d['query_x'],mode)
                conditions.append(dict(seed=seed,episode=d['id'],mode=mode,**runs[mode]))
            for mode in ['identity','skip','permute']:
                paired.append(dict(seed=seed,episode=d['id'],mode=mode,**paired_effect(runs['normal']['logits'],runs[mode]['logits'],d['query_y'])))
    return dict(experiment='B22-SUPPORT-WRITE',status='COMPLETE_COURSE_EXPERIMENT',training='NONE_FIXED_RANDOM_WEIGHTS',seeds=[0,1,2],modes=['normal','identity','skip','permute'],data=data,weights=weights,conditions=conditions,paired=paired)
