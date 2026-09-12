"""Isolated tabpfn2.0.9 reference worker. The main checker never changes sklearn."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model

def simple_config():
 return dict(PREPROCESS_TRANSFORMS=[dict(name='none',categorical_name='numeric',append_original=False,subsample_features=-1,global_transformer_name=None)],FINGERPRINT_FEATURE=False,FEATURE_SHIFT_METHOD=None,CLASS_SHIFT_METHOD=None,OUTLIER_REMOVAL_STD=None)

def run(checkpoint,destination):
 torch.set_num_threads(1);torch.manual_seed(64);result={};model,_,_=load_model(path=checkpoint,model_seed=0);model.cache_trainset_representation=False;model.eval()
 x=torch.randn(1,11,5,dtype=torch.float64);y=torch.tensor([[0.,1,2,0,1,2,0]],dtype=torch.float64);model.double();xx=x.transpose(0,1).clone().requires_grad_()
 out=model(xx,y.T,single_eval_pos=7);out.square().mean().backward()
 result['full']={'x':x,'y':y,'out':out.detach().transpose(0,1),'xgrad':xx.grad.transpose(0,1),'grads':{k:p.grad for k,p in model.named_parameters()}}
 cases=[]
 for kind,f,k in [('binary',4,2),('multiclass',7,3),('constant',5,2),('missing',6,2),('extreme',4,2)]:
  rng=np.random.default_rng(f);x=rng.normal(size=(35,f)).astype('float32');y=np.arange(24)%k
  if kind=='constant':x[:,1]=3
  if kind=='missing':x[3,2]=np.nan;x[27,4]=np.nan
  if kind=='extreme':x[-1,1]=100000
  m=TabPFNClassifier(n_estimators=1,model_path=checkpoint,device='cpu',n_jobs=1,inference_config=simple_config(),inference_precision=torch.float32,random_state=0);m.fit(x[:24],y)
  p=m.predict_proba(x[24:]);cases.append(dict(kind=kind,x=x,y=y,p=p))
 x=np.column_stack([np.linspace(-2,2,12),np.ones(12)]).astype('float32');x[1,1]=np.nan;y=(x[:,0]>0).astype(int)
 m=TabPFNClassifier(n_estimators=1,model_path=checkpoint,device='cpu',n_jobs=1,inference_config=simple_config(),inference_precision=torch.float32,random_state=0);m.fit(x,y)
 result['coupling']=dict(alone=m.predict_proba(np.array([[.3,1.]],dtype='float32')),joined=m.predict_proba(np.array([[.3,1.],[0.,2.]],dtype='float32'))[:1])
 result['wrappers']=cases
 # Real default 4-view wrapper: record every transformed input and compare later
 # through the learner's full model, independent of reference output tensors.
 rng=np.random.default_rng(99);x=rng.normal(size=(43,5)).astype('float32');y=np.arange(31)%3;x[4,1]=np.nan;x[36,0]=np.nan
 m=TabPFNClassifier(model_path=checkpoint,device='cpu',n_jobs=1,inference_precision=torch.float32,random_state=0);m.fit(x[:31],y);calls=[]
 def capture(module,args,kwargs):
  calls.append(dict(x=args[-2].detach().clone(),y=args[-1].detach().clone(),n=kwargs['single_eval_pos']))
 handle=m.model_.register_forward_pre_hook(capture,with_kwargs=True)
 p=m.predict_proba(x[31:]);handle.remove()
 configs=m.executor_.ensemble_configs
 result['default']=dict(x=x,y=y,p=p,calls=calls,permutations=[c.class_permutation.tolist() if c.class_permutation is not None else None for c in configs],temperature=m.softmax_temperature,outlier_sigma=m.model_.encoder[3].remove_outliers_sigma,model_seed=m.model_.seed,average_before_softmax=m.average_before_softmax,balance_probabilities=m.balance_probabilities,classes=m.n_classes_)
 torch.save(result,destination)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('checkpoint',type=Path);p.add_argument('destination',type=Path);a=p.parse_args();run(a.checkpoint,a.destination)
