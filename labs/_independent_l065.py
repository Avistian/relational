"""Reconstruct L065 extraction and selected heads using the historical release.

Run with the prepared TabPFN 2.0.9 / sklearn 1.6.1 source directory on PYTHONPATH.
No L064 or L065 teaching operator is imported. Prediction metrics, data/fold
identities, candidate completeness and head optimality are separate checks.
"""
import argparse,hashlib,itertools,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.special import expit,softmax
from scipy.stats import rankdata
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split,StratifiedKFold
from threadpoolctl import threadpool_limits
import torch,tabpfn
from tabpfn.model.loading import load_model
ROOT=Path(__file__).resolve().parent

def data(name):
    if name=='wdbc':bundle=load_breast_cancer();x=bundle.data.astype('float32');y=bundle.target
    else:
        df=pd.read_parquet(ROOT/f'data/cache/{name}.parquet');raw=df.pop('class' if name=='diabetes' else 'Class');x=df.to_numpy(dtype='float32')
        y=(raw.astype(str)==sorted(raw.astype(str).unique())[-1]).astype(int).to_numpy() if raw.dtype==object or str(raw.dtype)=='category' else raw.astype(int).to_numpy()
    _,y=np.unique(y,return_inverse=True);return x,y

def extract(model,x,y,q):
    keep=(x[:1]==x).mean(0)<1;states=[]
    handles=[b.register_forward_hook(lambda m,a,o:states.append(o[0,:,-1].detach().clone().numpy())) for b in model.transformer_encoder.layers]
    try:
        with torch.no_grad():logits=model(torch.tensor(np.concatenate([x[:,keep],q[:,keep]]))[:,None],torch.tensor(y,dtype=torch.float32)[:,None],single_eval_pos=len(x)).numpy()[:,0,:2]
    finally:
        for h in handles:h.remove()
    z=np.stack(states,1);return z[:len(x)],z[len(x):],logits,np.flatnonzero(keep)

def score(y,p):
    pos=y==1;n1=pos.sum();n0=len(y)-n1
    return dict(accuracy=float(np.mean(p.argmax(1)==y)),log_loss=float(-np.log(p[np.arange(len(y)),y]).mean()),auc=float((rankdata(p[:,1])[pos].sum()-n1*(n1+1)/2)/(n1*n0)))

def check(path,output):
    assert tabpfn.__version__=='2.0.9';torch.set_num_threads(1)
    result=json.loads(path.read_text());checkpoint=ROOT/'data/cache/foundation/tabpfn-v2.ckpt';assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==result['checkpoint_sha256']
    records=[];predictions=0
    for record in result['records']:
        name=record['dataset'];seed=record['seed'];x,y=data(name)
        assert hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()==record['data_sha256']
        tv,te=train_test_split(np.arange(len(y)),test_size=.2,stratify=y,random_state=seed);tr,va=train_test_split(tv,test_size=.2,stratify=y[tv],random_state=seed)
        for ids,key in [(tr,'train'),(va,'valid'),(te,'test')]:
            assert np.array_equal(ids,record[key+'_ids']) and np.array_equal(y[ids],record[key+'_targets'])
        folds=np.empty(len(tr),int)
        for fold,(_,q) in enumerate(StratifiedKFold(10,shuffle=True,random_state=seed).split(x[tr],y[tr])):folds[q]=fold
        assert np.array_equal(folds,record['folds'])
        model,_,_=load_model(path=checkpoint,model_seed=seed);model.eval();model.cache_trainset_representation=False
        ztrain=np.empty((len(tr),12,192),dtype='float32');max_logit=0.
        for fold in range(10):
            c=np.flatnonzero(folds!=fold);q=np.flatnonzero(folds==fold);_,z,logits,keep=extract(model,x[tr[c]],y[tr[c]],x[tr[q]])
            source=record['forward_traces'][fold];assert np.array_equal(tr[c],source['context_ids']) and np.array_equal(tr[q],source['query_ids']) and not set(tr[c])&set(tr[q])
            assert np.array_equal(keep,source['kept_columns']);ztrain[q]=z
            max_logit=max(max_logit,float(abs(logits-np.array(source['query_logits'])).max()))
        context,zvalid,vlogits,keep=extract(model,x[tr],y[tr],x[va]);_,ztest,tlogits,testkeep=extract(model,x[tr],y[tr],x[te])
        max_logit=max(max_logit,float(abs(vlogits-np.array(record['validation_logits'])).max()),float(abs(tlogits-np.array(record['test_logits'])).max()));assert max_logit<2e-4,(name,seed,max_logit)
        assert np.array_equal(testkeep,record['evaluation_kept_columns']) and record['evaluation_classes']==[0,1]
        head_checks=[]
        for method,head in record['heads'].items():
            candidate=head['candidates'];chosen=head['selected'];actual={(tuple(v['layers']),v['C']) for v in candidate}
            if method=='combined':expected={(layers,C) for size in range(1,4) for layers in itertools.combinations(range(1,13),size) for C in [.1,1.,10.]}
            else:expected={(tuple(chosen['layers']),C) for C in [.1,1.,10.]}
            assert actual==expected and len(candidate)==len(expected)
            for c in candidate:
                a=c['validation_accuracy'];assert 0<=a<=1 and abs(a*len(va)-round(a*len(va)))<1e-8 and np.isfinite(c['validation_log_loss'])
            winners=[c for c in candidate if c['validation_accuracy']==max(v['validation_accuracy'] for v in candidate)]
            fewest=min(len(v['layers']) for v in winners);winners=[v for v in winners if len(v['layers'])==fewest]
            smallest=min(v['C'] for v in winners);winners=[v for v in winners if v['C']==smallest]
            assert chosen==sorted(winners,key=lambda v:v['layers'])[0]
            layers=chosen['layers']
            if method=='raw':
                assert np.isfinite(x).all(),'This independent raw lane expects the complete observed numeric panel'
                a,b,c=x[tr],x[va],x[te]
            else:
                a=context if method=='vanilla' else ztrain
                a,b,c=[np.concatenate([v[:,layer-1] for layer in layers],1) for v in [a,zvalid,ztest]]
            mean=np.array(head['scaler_mean']);scale=np.array(head['scaler_scale']);w=np.array(head['coefficients'])[0];bias=head['intercept'][0];assert head['classes']==[0,1]
            mean_delta=float(abs(a.mean(0,dtype=np.float64)-mean).max());std=a.std(0,dtype=np.float64);std[std==0]=1;scale_delta=float(abs(std-scale).max())
            assert mean_delta<5e-5 and scale_delta<5e-5,(name,seed,method,mean_delta,scale_delta)
            def standard(v):return ((v-mean).astype(np.float32)/scale).astype(np.float32)
            test_p=expit(standard(c)@w+bias);p=np.column_stack([1-test_p,test_p]);saved=np.array(record['methods'][method]['probabilities']);delta=float(abs(p-saved).max());assert delta<3e-4,(name,seed,method,delta)
            val_p=expit(standard(b)@w+bias);val_accuracy=float(np.mean((val_p>.5)==y[va]));assert val_accuracy==chosen['validation_accuracy'],(name,seed,method,val_accuracy,chosen)
            # Binary liblinear L2 objective: (||w||²+b²)/2 + C*sum NLL.
            # The default synthetic intercept has scaling1 and is regularized.
            design=np.column_stack([standard(a),np.ones(len(a))]);coef=np.r_[w,bias]
            gradient=coef+chosen['C']*(design.T@(expit(design@coef)-y[tr]));g0=chosen['C']*(design.T@(.5-y[tr]));relative=float(np.linalg.norm(gradient)/max(1,np.linalg.norm(g0)))
            assert relative<5e-4,(name,seed,method,relative)
            head_checks.append(dict(method=method,candidates=len(candidate),probability_max_delta=delta,scaler_mean_max_delta=mean_delta,scaler_scale_max_delta=scale_delta,head_gradient_relative_norm=relative))
        native=softmax(tlogits.astype(float)/.9,axis=1);assert np.max(abs(native-np.array(record['methods']['native']['probabilities'])))<3e-5
        for method,values in record['methods'].items():
            p=np.array(values['probabilities']);calculated=score(y[te],p)
            assert np.allclose(p.sum(1),1,atol=1e-6)
            for metric,value in calculated.items():assert abs(value-values[metric])<2e-6,(name,seed,method,metric,value,values[metric])
            predictions+=len(te)
        records.append(dict(dataset=name,seed=seed,source_logit_max_delta=max_logit,heads=head_checks));print(name,seed,'PASS',flush=True)
    report=dict(status='PASS',evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),records=records,predictions=predictions,
        scope='Original historical network re-extracts all ten folds and validation/test states. Independent data/split/fold/class identities, complete candidate sets and validation tie rules, selected scaler/head predictions, convex-head gradient residuals and all saved prediction metrics. No teaching model/extraction/selection operator imported; no full paper reproduction.')
    output.write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path,nargs='?',default=ROOT/'_verify_l065_v2_results.json');p.add_argument('--report',type=Path,default=ROOT.parent/'reviews/lesson-quality-audit-047-070/065-independent.json');a=p.parse_args()
    with threadpool_limits(limits=1):r=check(a.evidence,a.report)
    print(json.dumps(dict(status=r['status'],records=len(r['records']),predictions=r['predictions'])))
