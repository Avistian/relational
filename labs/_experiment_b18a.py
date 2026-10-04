"""Pinned release inference; no training, HPO, or implicit protocol reduction."""
import gc,hashlib,json,os,platform,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,log_loss
from taco.model.tabpfn_arch.taco_classifier import TACOClassifier
from state_b18a import state_key,replacement_contrast

HF='d38ed9517764698a0b0064a7a8cb4197016349a3'
HASHES={'TACO':'bb1735869c5e995370ac6c1d8d7e2d27701fe35dbdb5c328565278d9d024471e','POT':'230330a87a629d0f505557863489341f50b0d059fb8cc241c2cca725cdb14963'}

def run(phase):
    import sklearn
    from huggingface_hub import hf_hub_download
    start=time.monotonic();out=dict(phase=phase,status='INCOMPLETE',records=[],updates=[],explanations=[])
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        out['environment']=dict(torch=torch.__version__,sklearn=sklearn.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),python=platform.python_version())
        paths={}
        for name in ['POT','TACO']:
            p=hf_hub_download('zabergjg/TabPFN-TACO',f'TabPFN-{name}-classifier.ckpt',revision=HF)
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==HASHES[name]
            paths[name]=p
            ck=torch.load(p,map_location='cpu',weights_only=False)
            out.setdefault('checkpoint_metadata',{})[name]={k:repr(v)[:12000] for k,v in ck.items() if k in ['config','step','epoch','iteration','train_config']}
            del ck
        X,y=load_breast_cancer(return_X_y=True)
        train,test=train_test_split(np.arange(len(y)),test_size=.5,random_state=42,stratify=y)
        selected,_=train_test_split(train,train_size=.25,random_state=42,stratify=y[train])
        out['data']=dict(x=X.tolist(),y=y.tolist(),train=train.tolist(),test=test.tolist(),selected=selected.tolist(),sha256=hashlib.sha256(X.tobytes()+y.tobytes()).hexdigest())
        def sync():torch.cuda.synchronize()
        def make(arm,seed,mode,ids,values=X,labels=y):
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();sync();t=time.monotonic()
            model=TACOClassifier(use_compressor=arm=='TACO4',row_compression_percentage=4,
                checkpoint_path=paths['TACO' if arm=='TACO4' else 'POT'],device='cuda',
                fit_mode=mode,n_estimators=8,random_state=seed,inference_precision=torch.float32,n_jobs=1)
            model.fit(values[ids],labels[ids]);sync()
            rec=dict(arm=arm,seed=seed,mode=mode,fit_seconds=time.monotonic()-t,support_ids=ids.tolist(),engine=type(model.executor_).__name__)
            rec['state_key']=state_key(ids=ids.tolist(),x=values[ids],y=labels[ids],weights=HASHES['TACO' if arm=='TACO4' else 'POT'],preprocessing={'implementation':'pinned release','fit_rows':ids.tolist()},recipe={'seed':seed,'mode':mode,'estimators':8,'compression':4,'dtype':'float32','batch':50})
            rec['effective_compressor']=model.model_.core.use_compressor
            return model,rec
        def predict(model,queries,repeats=1):
            predictions=[];times=[]
            for rep in range(repeats):
                pieces=[];rt=[]
                for batch in range(0,len(queries),50):
                    sync();t=time.monotonic();p=model.predict_proba(queries[batch:batch+50])[:,1];sync()
                    rt.append(time.monotonic()-t);pieces.extend(p.astype(float).tolist())
                predictions.append(pieces);times.append(rt)
            return predictions,times
        pairs=[(s,a,m) for s in [0,1,2] for a in ['POT-full','POT-selected','TACO4'] for m in ['fit_preprocessors','fit_with_cache']]
        chosen=[p for p in pairs if (p[0]==0 and p[1]=='POT-full')==(phase=='pilot')] if phase in ['pilot','matrix'] else []
        for seed,arm,mode in chosen:
            ids=selected if arm=='POT-selected' else train
            model,rec=make(arm,seed,mode,ids)
            pred,times=predict(model,X[test],4)
            rec.update(predictions=pred,batch_seconds=times,auc=roc_auc_score(y[test],pred[0]),log_loss=log_loss(y[test],pred[0]),peak_allocated_bytes=torch.cuda.max_memory_allocated(),post_predict_allocated_bytes=torch.cuda.memory_allocated(),K_values=getattr(model,'K_values_',None))
            out['records'].append(rec);print(phase,seed,arm,mode,rec['auc'],sum(map(sum,times)),flush=True)
            del model;gc.collect();torch.cuda.empty_cache()
        if phase=='updates':
            # Baseline reserves one SUPPORT row for the addition test.
            base=train[1:];add_id=int(train[0])
            variants=[('baseline',base,X,y),('add',train,X,y),('delete',base[1:],X,y)]
            flip=y.copy();flip[base[0]]=1-flip[base[0]];variants.append(('label',base,X,flip))
            scaled=X.copy();scaled[:,0]*=10;variants.append(('units',base,scaled,y))
            for arm in ['POT-full','TACO4']:
                for label,ids,values,labels in variants:
                    model,rec=make(arm,0,'fit_with_cache',ids,values,labels)
                    pred,times=predict(model,values[test]);rec.update(intervention=label,predictions=pred[0],batch_seconds=times,auc=roc_auc_score(y[test],pred[0]),peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    out['updates'].append(rec)
                    if label=='baseline':
                        masked=X[test].copy();masked[:57,0]=np.nan
                        mp,mt=predict(model,masked);out['updates'].append(dict(arm=arm,intervention='query_missing',predictions=mp[0],batch_seconds=mt,support_ids=ids.tolist(),state_key=rec['state_key'],auc=roc_auc_score(y[test],mp[0])))
                        for bgname,bg in [('first8',base[:8]),('last8',base[-8:])]:
                            calls=[]
                            def prob(q):
                                p,t=predict(model,q);calls.append(dict(x=q.tolist(),p=p[0],seconds=t));return np.array(p[0])
                            effects=replacement_contrast(prob,X[test[:16]],X[bg],0)
                            out['explanations'].append(dict(arm=arm,background=bgname,background_ids=bg.tolist(),query_ids=test[:16].tolist(),feature=0,effects=effects.tolist(),calls=calls))
                    del model;gc.collect();torch.cuda.empty_cache()
        out['status']='COMPLETE'
    except Exception:
        out['error']=traceback.format_exc();print(out['error'],flush=True)
    out['seconds']=time.monotonic()-start
    return out
