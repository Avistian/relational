"""L067 full historical pretrained LoCalPFN, numeric paper procedure and bounded adaptation.

Official source ff8803c57cd277380b2444f0f5ed4856b46f47f5 validates, never predicts for us.
Paper anchor exclusion is intentional; official released training retains the anchor.
"""
import copy,hashlib,inspect,importlib.metadata,json,math,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from relkit.tabpfn_l062_v2 import (TabPFNv1,V1Block,attention_mix,postnorm_update,
    load_pretrained,ensure_checkpoint,load_dataset,CHECKPOINT_SHA,CHECKPOINT_URL,MODEL_CONFIG)

PROTOCOL=dict(version='l067-v2',outer_split='stratified80/10/10',metric='validation AUC',temperature=1.,
    retrieval='train StandardScaler ddof0 then clip10; exact squared Euclidean stable row order',
    model_input='local context sample std ddof1 epsilon1e-6 clamp100 then100/F pad100',
    anchor='excluded by original identity per final paper footnote2',optimizer='AdamW',weight_decay=.01,
    lr_authority={'.01':'final paper AppendixA.2.2','1e-5':'released config.py default'},
    query_labels='loss only; global training class axis remains fixed',paper_reproduction='INCOMPARABLE')
PRESETS67={
 'smoke':dict(datasets=['diabetes'],seeds=[7],steps=2,eval_every=2,max_context=32,queries=4,batch=2,lrs=[.01,1e-5],inference_batch=4),
 'lab':dict(datasets=['diabetes'],seeds=[7],steps=30,eval_every=30,max_context=1000,queries=16,batch=2,lrs=[.01,1e-5],inference_batch=4),
 'closer':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[7,17,27],steps=30,eval_every=30,max_context=1000,queries=16,batch=2,lrs=[.01,1e-5],inference_batch=4)}

def neighbor_ids(memory,queries,k,memory_ids=None,exclude_ids=None):
    """Q×k local indices. Exclude identity, not every zero distance or first tie."""
    memory=np.asarray(memory);queries=np.asarray(queries)
    if memory.ndim!=2 or queries.ndim!=2 or memory.shape[1]!=queries.shape[1]:raise ValueError('Feature shape')
    ids=np.arange(len(memory)) if memory_ids is None else np.asarray(memory_ids)
    d=((queries[:,None,:]-memory[None,:,:])**2).sum(-1)
    if exclude_ids is not None:d=np.where(ids[None,:]==np.asarray(exclude_ids)[:,None],np.inf,d)
    if k<1 or np.any(np.isfinite(d).sum(1)<k):raise ValueError('Insufficient eligible rows')
    return np.argsort(d,axis=1,kind='stable')[:,:k]

def episode_indices(neighbors,context_size,rng):
    """One shared permutation per batch, like release; context/query disjoint per episode."""
    neighbors=np.asarray(neighbors)
    if neighbors.ndim!=2 or not 2<=context_size<neighbors.shape[1]:raise ValueError('Episode shape')
    order=rng.permutation(neighbors.shape[1]);shuffled=neighbors[:,order]
    return shuffled[:,:context_size],shuffled[:,context_size:]

def local_normalize(x,context_size,max_features=100):
    """Source-exact T×B×100 masked reductions; preserve float32 operation order."""
    if x.ndim!=3 or not 2<=context_size<x.shape[1] or not 1<=x.shape[-1]<=max_features:raise ValueError('Context/feature shape')
    features=x.shape[-1]
    z=nn.functional.pad(x,(0,max_features-features)).transpose(0,1).contiguous()
    c=z[:context_size];mask=~torch.isnan(c);count=mask.sum(0,keepdim=True)
    mean=torch.where(mask,c,0).sum(0,keepdim=True)/count
    differences=torch.where(mask,mean-c,0)
    std=(differences.square().sum(0,keepdim=True)/(count-1)).sqrt()
    z=((z-mean)/(std+1e-6)).clamp(-100,100)/(features/max_features)
    return z.transpose(0,1)

def query_loss(logits,targets,classes):
    """Only query logits/targets, mean over every B×Q query; full training class axis."""
    if logits.shape[:-1]!=targets.shape or not 2<=classes<=logits.shape[-1]:raise ValueError('Query/class shape')
    return nn.functional.cross_entropy(logits[...,:classes].reshape(-1,classes),targets.reshape(-1).long())

def validation_choice(scores):
    """Return earliest maximum AUC; step0 must be a candidate and ties retain it."""
    if not scores or not np.isfinite(scores).all():raise ValueError('Finite validation scores required')
    return int(np.argmax(scores))

def episode_logits(model,x,y,context,query):
    ids=np.concatenate([context,query],axis=1)
    xx=torch.as_tensor(x[ids],dtype=torch.float32);yy=torch.as_tensor(y[context],dtype=torch.float32)
    return model(local_normalize(xx,context.shape[1]),yy)

def predict_contexts(model,x,y,q,contexts,classes,batch_size=4):
    """Exact per-query contexts, vectorized B datasets, Q=1; never accept query labels."""
    if len(contexts)!=len(q):raise ValueError('One context per query')
    outputs=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(q),batch_size):
            c=contexts[start:start+batch_size];xx=np.concatenate([x[c],q[start:start+batch_size,None,:]],axis=1)
            logits=model(local_normalize(torch.as_tensor(xx,dtype=torch.float32),c.shape[1]),torch.as_tensor(y[c],dtype=torch.float32))[:,0,:classes]
            outputs.append(logits.softmax(-1).cpu().numpy())
    return np.concatenate(outputs)

def predict_shared(model,x,y,q,context,classes):
    """One shared context, all held-out queries; exact same numeric wrapper."""
    model.eval()
    with torch.no_grad():
        xx=torch.as_tensor(np.concatenate([x[context],q])[None],dtype=torch.float32)
        yy=torch.as_tensor(y[context][None],dtype=torch.float32)
        return model(local_normalize(xx,len(context)),yy)[0,:,:classes].softmax(-1).cpu().numpy()

def fit_geometry(train,other):
    """Official numeric dataset.py transform: train moments, population std, clip10."""
    scaler=StandardScaler().fit(train)
    return [np.clip(scaler.transform(z),-10,10).astype(np.float32) for z in [train,*other]],dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())

def model_digest(model):
    h=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        h.update(name.encode());h.update(str((tuple(tensor.shape),tensor.dtype)).encode());h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def stable_value(v):
    if v is None or isinstance(v,(str,int,float,bool)):return v
    if v is Ellipsis:return 'Ellipsis'
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,(list,tuple)):return [stable_value(i) for i in v]
    if isinstance(v,dict):return {str(k):stable_value(i) for k,i in v.items()}
    if isinstance(v,(set,frozenset)):return sorted([stable_value(i) for i in v],key=repr)
    if isinstance(v,bytes):return v.hex()
    if isinstance(v,Path):return str(v)
    if isinstance(v,types.CodeType):return {k:stable_value(getattr(v,k)) for k in ['co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags']}
    if inspect.isfunction(v):return dict(code=stable_value(v.__code__),defaults=stable_value(v.__defaults__),kwdefaults=stable_value(v.__kwdefaults__),closure=stable_value([c.cell_contents for c in v.__closure__ or []]))
    if inspect.isclass(v):return v.__module__+'.'+v.__qualname__
    raise TypeError('Unbound identity value '+str(type(v)))

def model_runtime_identity(model):
    """Bind actual module types, forward methods and non-weight runtime settings.

    Instance overrides and hooks are unsupported by this certified inference path;
    reject them rather than pretending an arbitrary callback has been audited.
    """
    records={}
    for name,module in model.named_modules():
        if 'forward' in vars(module):raise RuntimeError('Instance forward override invalidates source certification')
        if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:raise RuntimeError('Module hooks invalidate source certification')
        for parameter in module.parameters(recurse=False):
            if parameter._backward_hooks or getattr(parameter,'_post_accumulate_grad_hooks',None):raise RuntimeError('Parameter gradient hooks invalidate source certification')
        method=module.forward.__func__
        settings={k:stable_value(v) for k,v in vars(module).items() if not k.startswith('_')}
        records[name]=dict(type=type(module).__module__+'.'+type(module).__qualname__,forward=stable_value(method),settings=settings)
    return hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()


def kernel_identity(namespace,root):
    """Bind callable roots, nested CodeTypes and their globals/default dependencies."""
    roots=['model_runtime_identity','neighbor_ids','episode_indices','local_normalize','query_loss','validation_choice','TabPFNv1','load_pretrained','ensure_checkpoint','fit_geometry','predict_contexts','predict_shared','load_dataset','run_experiment']
    seen={};allowed={namespace['TabPFNv1'].__module__,namespace['run_experiment'].__module__}
    def visit(name,obj):
        if name in seen:return
        if inspect.isclass(obj):
            seen[name]={'class':obj.__qualname__}
            for k,v in vars(obj).items():
                if isinstance(v,(staticmethod,classmethod)):v=v.__func__
                if inspect.isfunction(v):visit(name+'.'+k,v)
        elif inspect.isfunction(obj):
            seen[name]=stable_value(obj)
            def dependency(v,n):
                if (inspect.isfunction(v) or inspect.isclass(v)) and (v.__module__ in allowed or v.__module__.startswith('relkit')):visit(n,v)
            def scan(code):
                for n in code.co_names:dependency(obj.__globals__.get(n),n)
                for c in code.co_consts:
                    if isinstance(c,types.CodeType):scan(c)
            scan(obj.__code__)
            for i,v in enumerate(obj.__defaults__ or []):dependency(v,name+'.default'+str(i))
            for k,v in (obj.__kwdefaults__ or {}).items():dependency(v,name+'.kwdefault.'+k)
    for name in roots:visit(name,namespace[name])
    from relkit.data import load_tier_a
    data_globals={k:stable_value(load_tier_a.__globals__[k]) for k in ['SPECS','CACHE']}
    result=dict(operators={k:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest() for k,v in seen.items()},constants={k:stable_value(namespace[k]) for k in ['MODEL_CONFIG','PROTOCOL','PRESETS67','CHECKPOINT_SHA','CHECKPOINT_URL']},loader_globals=data_globals,loader_source_sha256=hashlib.sha256(Path(inspect.getfile(load_tier_a)).read_bytes()).hexdigest(),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scipy','scikit-learn']})
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest();return result

def run_experiment(root,model,config=None,namespace=None):
    """Fresh paired evidence; immutable initial checkpoint restored for every arm and seed."""
    cfg=copy.deepcopy(PRESETS67['lab'] if config is None else config);ns=globals() if namespace is None else namespace
    identity=kernel_identity(ns,root);initial=copy.deepcopy(model.state_dict());initial_digest=model_digest(model)
    result=dict(config=cfg,protocol=PROTOCOL,kernel_identity=identity,initial_weights_sha256=initial_digest,
        initial_runtime_sha256=model_runtime_identity(model),checkpoint_sha256=CHECKPOINT_SHA,records=[],datasets={},
        paper_reproduction='INCOMPARABLE',scope='Complete 25.8M pretrained model; fresh local numeric subset adaptation, not 95-task 10-fold paper results')
    artifact_dir=Path(root)/'data/cache/l067-weights'/str(time.time_ns());artifact_dir.mkdir(parents=True)
    result['weight_artifact_directory']=str(artifact_dir)
    started=time.perf_counter()
    for name in cfg['datasets']:
        raw,y=load_dataset(name);classes=len(np.unique(y))
        result['datasets'][name]=dict(shape=list(raw.shape),x_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),y_sha256=hashlib.sha256(y.tobytes()).hexdigest())
        for seed in cfg['seeds']:
            begin=time.perf_counter();ids=np.arange(len(y));train,hold=train_test_split(ids,test_size=.2,random_state=seed,stratify=y)
            valid,test=train_test_split(hold,test_size=.5,random_state=seed,stratify=y[hold])
            (x,v,t),geometry=fit_geometry(raw[train],[raw[valid],raw[test]]);yt=y[train];classes=len(np.unique(yt))
            assert np.array_equal(np.unique(yt),np.arange(classes))
            k=min(int(10*np.sqrt(len(train))),cfg['max_context'],len(train)-1)
            queries=min(cfg['queries'],len(train)-k-1)
            if queries<1:raise ValueError('Training pool too small for excluded anchor plus context/query')
            vc=neighbor_ids(x,v,k);tc=neighbor_ids(x,t,k)
            model.load_state_dict(initial);model.eval()
            random_context=np.random.default_rng(seed).permutation(len(train))[:k]
            row=dict(dataset=name,seed=seed,train_ids=train.tolist(),valid_ids=valid.tolist(),test_ids=test.tolist(),
                k=k,queries_per_episode=queries,geometry=geometry,valid_context_ids=train[vc].tolist(),test_context_ids=train[tc].tolist(),
                random_context_ids=train[random_context].tolist(),targets=y[test].tolist(),valid_targets=y[valid].tolist(),arms={},adaptation={})
            for arm,context in [('global_all',np.arange(len(train))),('random_k',random_context)]:
                p=predict_shared(model,x,yt,t,context,classes);row['arms'][arm]=dict(probabilities=p.tolist())
            p=predict_contexts(model,x,yt,t,tc,classes,cfg['inference_batch']);row['arms']['local_frozen']=dict(probabilities=p.tolist())
            pv=predict_contexts(model,x,yt,v,vc,classes,cfg['inference_batch']);baseline_auc=roc_auc_score(y[valid],pv[:,1])
            # Generate once; both learning rates receive identical ordered training episodes.
            rng=np.random.default_rng(seed+67000);episodes=[]
            for step in range(cfg['steps']):
                anchor=rng.choice(len(train),size=cfg['batch'],replace=False)
                neighbors=neighbor_ids(x,x[anchor],k+queries,memory_ids=train,exclude_ids=train[anchor])
                c,q=episode_indices(neighbors,k,rng)
                assert all(not set(a)&set(b) for a,b in zip(c,q))
                assert all(train[a] not in train[n] for a,n in zip(anchor,neighbors))
                episodes.append((c,q,anchor))
            for lr in cfg['lrs']:
                model.load_state_dict(initial);model.eval();torch.manual_seed(seed)
                opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
                best_state=copy.deepcopy(initial);scores=[baseline_auc];selected=0
                trace=[dict(step=0,validation_auc=float(baseline_auc),weights_sha256=initial_digest)]
                losses=[];steps_trace=[];train_start=time.perf_counter()
                for step,(c,q,anchor) in enumerate(episodes,1):
                    opt.zero_grad(set_to_none=True);logits=episode_logits(model,x,yt,c,q)
                    loss=query_loss(logits,torch.as_tensor(yt[q]),classes)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite adaptation loss')
                    loss.backward();opt.step();losses.append(float(loss.detach()))
                    steps_trace.append(dict(step=step,anchor_ids=train[anchor].tolist(),context_ids=train[c].tolist(),query_ids=train[q].tolist()))
                    if step%cfg['eval_every']==0 or step==cfg['steps']:
                        pv=predict_contexts(model,x,yt,v,vc,classes,cfg['inference_batch']);score=roc_auc_score(y[valid],pv[:,1]);scores.append(score)
                        trace.append(dict(step=step,validation_auc=float(score),validation_probabilities=pv.tolist(),weights_sha256=model_digest(model)))
                        choice=validation_choice(scores)
                        if choice!=selected:selected=choice;best_state=copy.deepcopy(model.state_dict())
                weight_file=artifact_dir/f'{name}-{seed}-{lr}-final.pt';torch.save(model.state_dict(),weight_file)
                final_file_sha=hashlib.sha256(weight_file.read_bytes()).hexdigest()
                final_digest=model_digest(model)
                delta=math.sqrt(sum(float((v.detach().double()-initial[n].double()).square().sum()) for n,v in model.state_dict().items()))
                model.load_state_dict(best_state);selected_digest=model_digest(model)
                selected_file=artifact_dir/f'{name}-{seed}-{lr}-selected.pt';torch.save(best_state,selected_file)
                pred=predict_contexts(model,x,yt,t,tc,classes,cfg['inference_batch'])
                arm='local_ft_paper_lr' if lr==.01 else 'local_ft_release_lr'
                row['arms'][arm]=dict(probabilities=pred.tolist(),selected_step=trace[selected]['step'],selected_weights_sha256=selected_digest)
                row['adaptation'][arm]=dict(lr=lr,losses=losses,validation=trace,episodes=steps_trace,final_weights_sha256=final_digest,
                    final_parameter_l2_change=delta,final_weight_file=str(weight_file),final_weight_file_sha256=final_file_sha,
                    selected_weight_file=str(selected_file),selected_weight_file_sha256=hashlib.sha256(selected_file.read_bytes()).hexdigest(),seconds=time.perf_counter()-train_start,selected_step=trace[selected]['step'])
            for arm,a in row['arms'].items():
                pp=np.asarray(a['probabilities'],dtype=np.float32);a.update(log_loss=float(log_loss(y[test],pp,labels=np.arange(classes))),auc=float(roc_auc_score(y[test],pp[:,1])),accuracy=float(accuracy_score(y[test],pp.argmax(1))))
            model.load_state_dict(initial);model.eval()
            assert model_digest(model)==initial_digest
            row['seconds']=time.perf_counter()-begin;result['records'].append(row)
            print(name,seed,k,{a:round(v['auc'],5) for a,v in row['arms'].items()},round(row['seconds'],1),flush=True)
    result['seconds']=time.perf_counter()-started
    assert kernel_identity(ns,root)['sha256']==identity['sha256'];assert model_digest(model)==initial_digest
    result['final_restored_weights_sha256']=model_digest(model)
    result['final_runtime_sha256']=model_runtime_identity(model)
    assert result['final_runtime_sha256']==result['initial_runtime_sha256']
    return result
