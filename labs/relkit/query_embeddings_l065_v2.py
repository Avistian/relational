"""L065 v2: pretrained query extraction, explicit ten-fold and validation protocol.

Backbone lives unchanged in tabpfn_l064_v2. This module is independently written;
exhaustive layer/C selection and full-training evaluation contexts are local choices.
"""
import hashlib,importlib.metadata,inspect,itertools,json,time,types
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,log_loss,roc_auc_score
from .tabpfn_l064_v2 import TabPFNv2,MODEL_CONFIG,CHECKPOINT_SHA,load_pretrained,ensure_checkpoint,load_dataset
PROTOCOL=dict(folds=10,split=[.64,.16,.20],C=[.1,1.,10.],max_layers=3,selection='validation accuracy; ties fewer layers, smaller C, lexicographic layers',evaluation_context='all outer-training rows; validation and test separate fixed query batches',seed_policy='same group-identity seed in all folds and evaluation',recipe='one numeric view, context constant removal, no external transform/fingerprint/outlier transform',head='StandardScaler fit on head-training representations; binary liblinear L2 logistic regression, max_iter=2000, tol=1e-6')
PRESETS65={'lab':dict(datasets=['diabetes'],seeds=[7]),'closer':dict(datasets=['diabetes','blood_transfusion','wdbc'],seeds=[0,1,2])}

def extract_target_states(hidden,n_context,role='query'):
    """B,N,G+1,D -> B,Q,D or B,C,D, selecting the final (target) token."""
    if hidden.ndim!=4 or not 0<n_context<hidden.shape[1] or role not in ('query','context'):raise ValueError('Expected complete hidden state and legal role')
    rows=slice(n_context,None) if role=='query' else slice(0,n_context)
    return hidden[:,rows,-1,:]

def scatter_fold_embeddings(n_rows,folds,extractor):
    """extractor(context_indices,query_indices) returns Q,12,192; scatter once/row."""
    folds=np.asarray(folds)
    if folds.shape!=(n_rows,) or len(np.unique(folds))<2:raise ValueError('At least two nonempty folds')
    result=None;writes=np.zeros(n_rows,dtype=int);traces=[]
    for fold in np.unique(folds):
        query=np.flatnonzero(folds==fold);context=np.flatnonzero(folds!=fold)
        values=np.asarray(extractor(context,query))
        if values.ndim!=3 or len(values)!=len(query) or not np.isfinite(values).all():raise ValueError('Malformed embeddings')
        if result is None:result=np.empty((n_rows,*values.shape[1:]),dtype=values.dtype)
        if values.shape[1:]!=result.shape[1:]:raise ValueError('Layer or width changed across folds')
        result[query]=values;writes[query]+=1
        traces.append(dict(fold=int(fold),context_positions=context.tolist(),query_positions=query.tolist()))
    if not np.all(writes==1):raise ValueError('Every row must be written exactly once')
    return result,traces

def concatenate_layers(embeddings,layers):
    """N,L,D and distinct one-based layer IDs -> N,(len(layers)*D)."""
    if embeddings.ndim!=3 or not 1<=len(layers)<=3 or len(set(layers))!=len(layers) or any(l<1 or l>embeddings.shape[1] for l in layers):raise ValueError('Select one to three distinct available layers')
    return np.concatenate([embeddings[:,layer-1,:] for layer in layers],axis=1)

def choose_candidate(candidates):
    """Validation-only deterministic selection. A candidate never contains test metrics."""
    if not candidates or any(not np.isfinite(c['validation_accuracy']) for c in candidates):raise ValueError('Finite validation scores required')
    return min(candidates,key=lambda c:(-c['validation_accuracy'],len(c['layers']),c['C'],tuple(c['layers'])))

def extract_embeddings(model,context_x,context_y,query_x,seed=0,namespace=None):
    ns=globals() if namespace is None else namespace
    cx=np.asarray(context_x,dtype=np.float32);qx=np.asarray(query_x,dtype=np.float32)
    classes,y=np.unique(context_y,return_inverse=True)
    if len(classes)!=2 or np.isinf(cx).any() or np.isinf(qx).any():raise ValueError('Local lane supports binary finite-or-NaN numeric data')
    keep=(cx==cx[:1]).mean(0)<1
    if not keep.any():raise ValueError('No variable columns')
    x=torch.tensor(np.concatenate([cx[:,keep],qx[:,keep]]))[None];labels=torch.tensor(y,dtype=x.dtype)[None]
    states=[];handles=[]
    def capture(module,args,output):states.append(output.detach())
    try:
        for block in model.blocks:handles.append(block.register_forward_hook(capture))
        model.eval()
        with torch.no_grad():logits=model(x,labels,seed=seed)[0,:,:len(classes)]
    finally:
        for handle in handles:handle.remove()
    query=torch.stack([ns['extract_target_states'](h,len(cx),'query')[0] for h in states],1).numpy()
    context=torch.stack([ns['extract_target_states'](h,len(cx),'context')[0] for h in states],1).numpy()
    return dict(query=query,context=context,logits=logits.numpy(),probabilities=(logits/.9).softmax(-1).numpy(),classes=classes,kept_columns=np.flatnonzero(keep))

def fit_head(train,y,C):
    scaler=StandardScaler().fit(train);scaled=scaler.transform(train)
    head=LogisticRegression(C=C,solver='liblinear',max_iter=2000,tol=1e-6,random_state=0).fit(scaled,y)
    if int(head.n_iter_.max())>=2000:raise RuntimeError('Head did not converge')
    return scaler,head

def fit_candidates(train,y,valid,valid_y,layer_sets,namespace=None):
    ns=globals() if namespace is None else namespace;candidates=[]
    for layers in layer_sets:
        a=ns['concatenate_layers'](train,layers);b=ns['concatenate_layers'](valid,layers)
        for C in PROTOCOL['C']:
            scaler,head=fit_head(a,y,C);p=head.predict_proba(scaler.transform(b))
            candidates.append(dict(layers=list(layers),C=C,validation_accuracy=float(accuracy_score(valid_y,p.argmax(1))),validation_log_loss=float(log_loss(valid_y,p))))
    selected=ns['choose_candidate'](candidates)
    scaler,head=fit_head(ns['concatenate_layers'](train,selected['layers']),y,selected['C'])
    return selected,candidates,scaler,head

def metric_record(y,p):
    return dict(accuracy=float(accuracy_score(y,p.argmax(1))),log_loss=float(log_loss(y,p)),auc=float(roc_auc_score(y,p[:,1])),probabilities=p.tolist())

def model_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def stable_value(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if value is Ellipsis:return 'Ellipsis'
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,(list,tuple)):return [stable_value(v) for v in value]
    if isinstance(value,dict):return {str(k):stable_value(v) for k,v in value.items()}
    if isinstance(value,(set,frozenset)):return sorted([stable_value(v) for v in value],key=repr)
    if isinstance(value,bytes):return value.hex()
    if isinstance(value,types.CodeType):return {k:stable_value(getattr(value,k)) for k in ['co_code','co_consts','co_names','co_varnames','co_freevars','co_cellvars','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_flags']}
    if inspect.isfunction(value):return dict(code=stable_value(value.__code__),defaults=stable_value(value.__defaults__),kwdefaults=stable_value(value.__kwdefaults__),closure=stable_value([c.cell_contents for c in (value.__closure__ or [])]))
    if isinstance(value,Path):return str(value)
    if inspect.isclass(value):return value.__module__+'.'+value.__qualname__
    raise TypeError(str(type(value)))

def kernel_identity(namespace,root):
    """Explicit roots plus recursive globals in ALL nested code; stable across processes."""
    from relkit import tabpfn_l064_v2 as backbone
    roots=['extract_target_states','scatter_fold_embeddings','concatenate_layers','choose_candidate','extract_embeddings','fit_head','fit_candidates','metric_record','run_experiment']
    seen={}
    def visit(name,obj):
        if name in seen:return
        if inspect.isclass(obj):
            seen[name]={k:stable_value(v) for k,v in vars(obj).items() if inspect.isfunction(v)}
            for k,v in vars(obj).items():
                if inspect.isfunction(v):visit(name+'.'+k,v)
        elif inspect.isfunction(obj):
            seen[name]=stable_value(obj)
            def scan(code):
                for n in code.co_names:
                    v=obj.__globals__.get(n)
                    if inspect.isfunction(v) and (v.__module__.startswith('relkit') or v.__globals__ is namespace):visit(n,v)
                    elif inspect.isclass(v) and v.__module__ in ('relkit.tabpfn_l064_v2','__main__'):visit(n,v)
                for v in code.co_consts:
                    if isinstance(v,types.CodeType):scan(v)
            scan(obj.__code__)
    for name in roots:visit(name,namespace[name])
    visit('TabPFNv2',namespace.get('TabPFNv2',backbone.TabPFNv2))
    from relkit.data import load_tier_a
    result=dict(operators=seen,config=namespace.get('PROTOCOL',PROTOCOL),presets=namespace.get('PRESETS65',PRESETS65),checkpoint_sha256=CHECKPOINT_SHA,backbone_file_sha256=hashlib.sha256(Path(backbone.__file__).read_bytes()).hexdigest(),loader_file_sha256=hashlib.sha256(Path(inspect.getfile(load_tier_a)).read_bytes()).hexdigest(),versions={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','scipy']})
    result['loader_globals']={k:stable_value(load_tier_a.__globals__[k]) for k in ['SPECS','CACHE']}
    result['backbone_constants']={k:stable_value(namespace.get(k,getattr(backbone,k))) for k in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','RECIPE']}
    result['sha256']=hashlib.sha256(json.dumps(result,sort_keys=True).encode()).hexdigest();return result

def run_experiment(root,model,config=None,namespace=None):
    ns=globals() if namespace is None else namespace;config=dict(PRESETS65['lab'] if config is None else config);records=[]
    identity=kernel_identity(ns,root);weights=model_digest(model)
    for dataset in config['datasets']:
        x,y=load_dataset(dataset);ids=np.arange(len(y));data_sha=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()
        for seed in config['seeds']:
            start=time.perf_counter();trainvalid,test=train_test_split(ids,test_size=.2,stratify=y,random_state=seed)
            train,valid=train_test_split(trainvalid,test_size=.2,stratify=y[trainvalid],random_state=seed)
            fold_ids=np.empty(len(train),int)
            for k,(_,q) in enumerate(StratifiedKFold(PROTOCOL['folds'],shuffle=True,random_state=seed).split(x[train],y[train])):fold_ids[q]=k
            forward_traces=[]
            def encoder(c,q):
                z=ns['extract_embeddings'](model,x[train[c]],y[train[c]],x[train[q]],seed,ns)
                forward_traces.append(dict(context_ids=train[c].tolist(),query_ids=train[q].tolist(),classes=z['classes'].tolist(),kept_columns=z['kept_columns'].tolist(),query_logits=z['logits'].tolist()))
                return z['query']
            ztrain,scatter=ns['scatter_fold_embeddings'](len(train),fold_ids,encoder)
            # Full training context for each evaluation split. No validation/test labels enter.
            zv=ns['extract_embeddings'](model,x[train],y[train],x[valid],seed,ns)
            zt=ns['extract_embeddings'](model,x[train],y[train],x[test],seed,ns)
            extraction_seconds=time.perf_counter()-start;select_start=time.perf_counter()
            layer_sets=list(itertools.chain.from_iterable(itertools.combinations(range(1,13),r) for r in range(1,PROTOCOL['max_layers']+1)))
            chosen,candidates,scaler,head=fit_candidates(ztrain,y[train],zv['query'],y[valid],layer_sets,ns)
            methods={};heads={}
            def score(name,a,b,c,layersets):
                choice,trials,scale,linear=fit_candidates(a,y[train],b,y[valid],layersets,ns)
                features=ns['concatenate_layers'](c,choice['layers']);p=linear.predict_proba(scale.transform(features))
                train_p=linear.predict_proba(scale.transform(ns['concatenate_layers'](a,choice['layers'])))
                methods[name]=metric_record(y[test],p);methods[name]['fitted_train_accuracy']=float(accuracy_score(y[train],train_p.argmax(1)))
                heads[name]=dict(selected=choice,candidates=trials,scaler_mean=scale.mean_.tolist(),scaler_scale=scale.scale_.tolist(),coefficients=linear.coef_.tolist(),intercept=linear.intercept_.tolist(),classes=linear.classes_.tolist())
            p=head.predict_proba(scaler.transform(ns['concatenate_layers'](zt['query'],chosen['layers'])))
            methods['combined']=metric_record(y[test],p)
            heads['combined']=dict(selected=chosen,candidates=candidates,scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),coefficients=head.coef_.tolist(),intercept=head.intercept_.tolist(),classes=head.classes_.tolist())
            for layer in [6,9,12]:score('layer_'+str(layer),ztrain,zv['query'],zt['query'],[(layer,)])
            score('vanilla',zv['context'],zv['query'],zt['query'],[(12,)])
            imputer=SimpleImputer().fit(x[train]);raw=[imputer.transform(x[v])[:,None,:] for v in [train,valid,test]]
            score('raw',*raw,[(1,)])
            methods['native']=metric_record(y[test],zt['probabilities'])
            record=dict(dataset=dataset,seed=seed,rows=len(y),features=x.shape[1],data_sha256=data_sha,train_ids=train.tolist(),valid_ids=valid.tolist(),test_ids=test.tolist(),train_targets=y[train].tolist(),valid_targets=y[valid].tolist(),test_targets=y[test].tolist(),folds=fold_ids.tolist(),forward_traces=forward_traces,scatter=scatter,embedding_shape=list(ztrain.shape),validation_logits=zv['logits'].tolist(),test_logits=zt['logits'].tolist(),evaluation_classes=zt['classes'].tolist(),evaluation_kept_columns=zt['kept_columns'].tolist(),methods=methods,heads=heads,extraction_seconds=extraction_seconds,selection_seconds=time.perf_counter()-select_start)
            records.append(record);print(dataset,seed,'combined',methods['combined']['accuracy'],'native',methods['native']['accuracy'],'layers',chosen['layers'],flush=True)
    if model_digest(model)!=weights or kernel_identity(ns,root)['sha256']!=identity['sha256']:raise RuntimeError('Kernel or weights changed during measurement')
    return dict(status='MEASURED_PRETRAINED_QUERY_EMBEDDINGS',paper_reproduction='INCOMPARABLE',paper_29_dataset_benchmark='NOT_RUN',protocol=PROTOCOL,config=config,kernel_identity=identity,weights_sha256=weights,checkpoint_sha256=CHECKPOINT_SHA,records=records,timing_boundary='Extraction includes all ten fold forwards and separate validation/test forwards; model/checkpoint loading and source checks excluded. Selection includes all fitted heads and metrics.')
