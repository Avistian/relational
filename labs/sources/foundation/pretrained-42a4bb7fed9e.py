"""Explicit historical checkpoint experiments. Use the isolated requirements files.

Examples: python _run_pretrained_foundation.py --lesson 62
          python _run_pretrained_foundation.py --lesson 65
Pretrained package implementations are reference/measurement paths, not the
from-scratch architecture exercise. Their identities are in _sources_foundation.json.
"""
import argparse,copy,hashlib,importlib.metadata,json,os,time,typing,warnings
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from relkit.foundation_benchmark import random_task,encode_train,fit_candidate,error,ROOT
from relkit.foundation_core import crossfit_embeddings,nearest_context,local_episode,open_class_loss
from relkit.benchmark_core import paired_summary


def v1(seed=0,grad=False):
    # Legacy package imports a typing alias removed from torch 2.13. No tensor math changes.
    import torch.nn.modules.transformer as module
    if not hasattr(module,'Optional'):module.Optional=typing.Optional
    from tabpfn import TabPFNClassifier
    if importlib.metadata.version('tabpfn')!='0.1.11':raise RuntimeError('L062/L067 require tabpfn==0.1.11 in a separate environment')
    os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
    base=ROOT/'data/cache/foundation/v1'
    path=base/'models_diff/prior_diff_real_checkpoint_n_0_epoch_42.cpkt'
    if hashlib.sha256(path.read_bytes()).hexdigest()!='3c9aadaeddbf51462af8c0ee4b3ca3c697890f77e92318abbb0821b75261c392':raise ValueError('Unexpected v1 checkpoint')
    return TabPFNClassifier(device='cpu',base_path=base,N_ensemble_configurations=1,seed=seed,
        no_grad=not grad,no_preprocess_mode=grad,feature_shift_decoder=not grad)


def v2(seed=0):
    from tabpfn import TabPFNClassifier
    if importlib.metadata.version('tabpfn')!='2.0.9':raise RuntimeError('This historical-v2 lab requires tabpfn==2.0.9')
    return TabPFNClassifier(n_estimators=1,model_path=ROOT/'data/cache/foundation/tabpfn-v2.ckpt',
                           device='cpu',random_state=seed,n_jobs=1)


def datasets():
    for name in ['diabetes','blood_transfusion','phoneme']:
        raw,y,reg,audit=random_task(name,cap=300,seed=62)
        yield name,encode_train(raw),y,audit


def run_v1():
    records=[];audits={};isolation=[]
    for name,x,y,audit in datasets():
        audits[name]=audit
        for seed in [0,1,2]:
            begin=time.perf_counter();model=v1(seed);model.fit(x['train'],y['train']);p=model.predict_proba(x['test'])[:,1]
            joined=model.predict_proba(np.concatenate([x['test'][:1],x['test'][-2:]]))[:,1]
            alone=model.predict_proba(x['test'][:1])[:,1]
            isolation.append(float(abs(joined[0]-alone[0])))
            records.append(dict(dataset=name,arm='TabPFN-v1',seed=seed,error=error(y['test'],p,False),
                seconds=time.perf_counter()-begin,predictions=p.tolist(),targets=y['test'].tolist()))
            print(name,seed,records[-1]['error'],flush=True)
    return dict(records=records,datasets=audits,summary=paired_summary(records),query_batch_max_delta=max(isolation),
        scope='Historical v1 pretrained inference, one ensemble view, 180 context/60 test rows per task',verdict='INCOMPARABLE')


def run_embeddings():
    records=[];audits={}
    for name,x,y,audit in datasets():
        audits[name]=audit
        for seed in [0,1,2]:
            begin=time.perf_counter();fold=np.empty(len(y['train']),int)
            for i,(_,ix) in enumerate(StratifiedKFold(3,shuffle=True,random_state=65).split(x['train'],y['train'])):fold[ix]=i
            fitted=[]
            def embed(cx,cy,qx):
                m=v2(seed);m.fit(cx,cy);h=m.get_embeddings(qx,data_source='test')[0]
                fitted.append(m);return h
            train_h=crossfit_embeddings(x['train'],y['train'],fold,embed)
            val_h=np.mean([m.get_embeddings(x['val'],data_source='test')[0] for m in fitted],axis=0)
            test_h=np.mean([m.get_embeddings(x['test'],data_source='test')[0] for m in fitted],axis=0)
            candidates=[]
            for c in [.01,1.,100.]:
                head=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=1000))
                head.fit(train_h,y['train']);candidates.append((log_loss(y['val'],head.predict_proba(val_h)),head,c))
            validation,head,c=min(candidates,key=lambda item:item[0]);p=head.predict_proba(test_h)[:,1]
            elapsed=time.perf_counter()-begin
            records.append(dict(dataset=name,arm='crossfit-linear-head',seed=seed,error=error(y['test'],p,False),seconds=elapsed,
                predictions=p.tolist(),targets=y['test'].tolist(),selected_C=c,validation_errors=[float(a[0]) for a in candidates],
                fold_ids=fold.tolist(),embedding_shape=list(train_h.shape)))
            begin=time.perf_counter();m=v2(seed);m.fit(x['train'],y['train']);p=m.predict_proba(x['test'])[:,1]
            records.append(dict(dataset=name,arm='native-v2',seed=seed,error=error(y['test'],p,False),seconds=time.perf_counter()-begin,predictions=p.tolist(),targets=y['test'].tolist()))
            print(name,seed,'embedding/native',records[-2]['error'],records[-1]['error'],flush=True)
    return dict(records=records,datasets=audits,summary=paired_summary(records),verdict='INCOMPARABLE',
        scope='Three-fold final-layer query-role embeddings and validation-selected logistic head vs native v2',
        deviations=['Three rather than ten folds','Final layer only; no paper layer selection','Fold-context test embeddings averaged','Three small binary tasks rather than 29','Seed intervals conditional on fixed rows/folds'])


def run_scale():
    from tabicl import TabICLClassifier
    raw,y,reg,audit=random_task('phoneme',cap=900,seed=66);x=encode_train(raw)
    records=[]
    for seed in [0,1,2]:
        order=np.random.default_rng(seed+66).permutation(len(y['train']))
        for size in [60,180,540]:
            ix=order[:size];m=TabICLClassifier(n_estimators=1,device='cpu',random_state=seed,n_jobs=1,use_amp=False,
                model_path=str(ROOT/'data/cache/foundation/tabicl-v1.1.ckpt'),checkpoint_version='tabicl-classifier-v1.1-0506.ckpt',allow_auto_download=False)
            begin=time.perf_counter();m.fit(x['train'][ix],y['train'][ix]);fit=time.perf_counter()-begin
            begin=time.perf_counter();p=m.predict_proba(x['test'][:60])[:,1];predict=time.perf_counter()-begin
            records.append(dict(seed=seed,context=size,log_loss=error(y['test'][:60],p,False),fit_seconds=fit,predict_seconds=predict,
                context_ids=ix.tolist(),predictions=p.tolist(),targets=y['test'][:60].tolist()))
            print(seed,size,records[-1]['log_loss'],predict,flush=True)
    return dict(scope='Nested context-size intervention on one real task; not a cross-dataset superiority claim',records=records,dataset=audit,
        verdict='INCOMPARABLE',paper_500k_run='NOT_RUN',checkpoint='TabICL v1.1 (paper originally v1)')


def run_open():
    records=[];audits={};cfg=dict(trees=100,epochs=24)
    for name,x,y,audit in datasets():
        audits[name]=audit
        # Drop one column at deployment and replace using its train mean. No test fitting.
        center=x['train'].mean(0);column=int(np.argmax(x['train'].std(0)))
        variants={'clean':x['test'].copy(),'missing-column':x['test'].copy(),'scaled-column':x['test'].copy()}
        variants['missing-column'][:,column]=center[column]
        variants['scaled-column'][:,column]=center[column]+3*(variants['scaled-column'][:,column]-center[column])
        for seed in [0,1,2]:
            for arm in ['XGBoost','TabPFN-v2']:
                begin=time.perf_counter();predict,valid,epoch=fit_candidate(arm,x['train'],y['train'],x['val'],y['val'],False,seed,0,cfg)
                for condition,xt in variants.items():
                    p=predict(xt);records.append(dict(dataset=name,arm=arm,seed=seed,condition=condition,column=column,
                        log_loss=error(y['test'],p,False),predictions=p.tolist(),targets=y['test'].tolist(),seconds=time.perf_counter()-begin))
                print(name,seed,arm,flush=True)
    unknown=open_class_loss([0,1,2],[[.8,.2],[.2,.8],[.1,.9]],[0,1])
    return dict(records=records,datasets=audits,unseen_class_log_loss_example=unknown,unseen_class_epsilon=1e-12,
        scope='Held-fixed models under two test-only feature corruptions, plus a synthetic unsupported-class diagnostic',
        verdict='INCOMPARABLE',boundary='Corruption sensitivity is not equivalent to natural temporal shift; no new-class training or full open-environment benchmark')


def run_local():
    records=[];audits={}
    for name,x,y,audit in datasets():
        audits[name]=audit
        scale=StandardScaler().fit(x['train']);z={s:scale.transform(a).astype('float32') for s,a in x.items()}
        for seed in [0,1,2]:
            # Deepcopy avoids mutating the legacy package's shared in-memory checkpoint.
            model=v1(seed,True);model.model=(model.model[0],model.model[1],copy.deepcopy(model.model[2]))
            network=model.model[2];initial=copy.deepcopy(network.state_dict());rng=np.random.default_rng(seed+67)
            def predict(q,local):
                probabilities=[]
                for start in range(0,len(q),8):
                    qq=q[start:start+8]
                    if local:
                        ids=nearest_context(z['train'],qq,32)
                        for row,context in zip(qq,ids):
                            if len(np.unique(y['train'][context]))<2:
                                probabilities.append(np.array([float(y['train'][context][0])]))
                                continue
                            model.fit(torch.tensor(z['train'][context]),y['train'][context])
                            with torch.no_grad():probabilities.append(model.predict_proba(torch.tensor(row[None]))[:,1].detach().numpy())
                    else:
                        model.fit(torch.tensor(z['train']),y['train'])
                        with torch.no_grad():probabilities.append(model.predict_proba(torch.tensor(qq))[:,1].detach().numpy())
                return np.concatenate(probabilities)
            for arm,local in [('global',False),('local',True)]:
                begin=time.perf_counter();p=predict(z['test'][:24],local)
                records.append(dict(dataset=name,arm=arm,seed=seed,error=error(y['test'][:24],p,False),seconds=time.perf_counter()-begin,predictions=p.tolist(),targets=y['test'][:24].tolist()))
            optimizer=torch.optim.Adam(network.parameters(),lr=1e-5);history=[];best=float('inf');state=None
            begin=time.perf_counter()
            for step in range(6):
                anchor=int(rng.integers(len(y['train'])));cx,cy,qx,qy,ci,qi=local_episode(z['train'],y['train'],anchor,32,8,rng)
                if len(np.unique(cy))<2:continue
                model.fit(torch.tensor(cx,requires_grad=True),cy)
                probs=model.predict_proba(torch.tensor(qx,requires_grad=True))
                loss=nn.functional.nll_loss(probs.clamp_min(1e-7).log(),torch.tensor(qy))
                optimizer.zero_grad();loss.backward();optimizer.step()
                val=predict(z['val'][:12],True);score=error(y['val'][:12],val,False)
                history.append(dict(step=step,loss=float(loss.detach()),validation=score,context_ids=ci.tolist(),query_ids=qi.tolist()))
                if score<best:best=score;state=copy.deepcopy(network.state_dict())
            if state is None:raise RuntimeError('No valid adaptation episode')
            network.load_state_dict(state);p=predict(z['test'][:24],True)
            changed=max(float((initial[k]-network.state_dict()[k]).abs().max()) for k in initial if initial[k].is_floating_point())
            records.append(dict(dataset=name,arm='local-finetuned',seed=seed,error=error(y['test'][:24],p,False),seconds=time.perf_counter()-begin,predictions=p.tolist(),targets=y['test'][:24].tolist(),history=history,max_parameter_change=changed))
            print(name,seed,'local adaptation',changed,flush=True)
    return dict(records=records,datasets=audits,summary=paired_summary(records),scope='Actual v1 checkpoint retrieval and six-step gradient adaptation; three tasks, 24 test rows each',
        verdict='INCOMPARABLE',deviations=['Small context and six steps','24 test rows; very weak uncertainty','Scaled Euclidean retrieval','One-view no-preprocess gradient path','No full LoCalPFN benchmark or hyperparameter search'])


if __name__=='__main__':
    from torch import nn
    parser=argparse.ArgumentParser();parser.add_argument('--lesson',type=int,required=True);args=parser.parse_args()
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore',category=FutureWarning)
    with threadpool_limits(limits=1):
        result={62:run_v1,65:run_embeddings,66:run_scale,67:run_local,69:run_open}[args.lesson]()
    result['operator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['versions']={p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn']}
    (ROOT/f'_verify_l{args.lesson:03}_results.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
