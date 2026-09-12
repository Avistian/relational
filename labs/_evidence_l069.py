"""Independent source-data, partition, metric and original-model audit for L069.

Run with PYTHONPATH=labs/data/cache/l064-source/official to use pinned TabPFN
2.0.9. No lesson model or evaluator is imported. The original attention kernel
is replaced by an independently checked SDPA adapter to bound CPU memory.
"""
import argparse,hashlib,itertools,json,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).resolve().parent

def auc(y,s):
 y=np.asarray(y,dtype=bool);s=np.asarray(s);n=int(y.sum());m=len(y)-n
 return float((rankdata(s)[y].sum()-n*(n+1)/2)/(n*m))

def ap(y,s):
 y=np.asarray(y,dtype=int);s=np.asarray(s);order=np.argsort(-s,kind='stable');yy=y[order];ss=s[order]
 ends=np.r_[np.flatnonzero(ss[:-1]!=ss[1:]),len(ss)-1];tp=np.cumsum(yy)[ends];recall=tp/yy.sum();precision=tp/(ends+1)
 return float(np.sum(np.diff(np.r_[0.,recall])*precision))

def metrics(y,p,classes,epsilon):
 classes=np.asarray(classes);y=np.asarray(y);p=np.asarray(p);pred=classes[p.argmax(1)];support=np.isin(y,classes)
 target=np.array([p[i,np.flatnonzero(classes==v)[0]] if v in classes else 0 for i,v in enumerate(y)])
 loss=-np.log(np.maximum(target,epsilon));labels=np.union1d(y,pred);recalls=[];f1=[]
 for v in labels:
  tp=np.sum((y==v)&(pred==v));fn=np.sum((y==v)&(pred!=v));fp=np.sum((y!=v)&(pred==v))
  if tp+fn:recalls.append(tp/(tp+fn))
  f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
 result=dict(n=len(y),unsupported_n=int((~support).sum()),unsupported_fraction=float((~support).mean()),accuracy=float((pred==y).mean()),known_accuracy=float((pred[support]==y[support]).mean()) if support.any() else None,clipped_log_loss=float(loss.mean()),known_log_loss=float(loss[support].mean()) if support.any() else None,balanced_accuracy=float(np.mean(recalls)),macro_f1=float(np.mean(f1)))
 if support.all():result['auc']=float(np.mean([auc(y==v,p[:,j]) for j,v in enumerate(classes) if len(classes)>2 or j==1])) if len(np.unique(y))==len(classes) else None
 return result

def original_model():
 import torch,tabpfn
 from tabpfn.model.loading import load_model
 from tabpfn.model.multi_head_attention import MultiHeadAttention
 assert tabpfn.__version__=='2.0.9';torch.set_num_threads(1)
 saved=MultiHeadAttention.compute_attention_heads
 def compact(q,k,v,kv,qkv,dropout_p=None,softmax_scale=None):
  if qkv is not None:q,k,v=qkv.unbind(-3)
  elif kv is not None:k,v=kv.unbind(-3)
  k=k.repeat_interleave(q.shape[2]//k.shape[2],dim=2);v=v.repeat_interleave(q.shape[2]//v.shape[2],dim=2)
  scale=float(np.sqrt(np.float32(1/q.shape[-1]))) if softmax_scale is None else softmax_scale
  return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),dropout_p=dropout_p or 0.,scale=scale).transpose(1,2)
 deltas=[]
 for shared in [False,True]:
  torch.manual_seed(69);q=torch.randn(2,7,6,32,dtype=torch.float64,requires_grad=True);k=torch.randn(2,9,1 if shared else 6,32,dtype=torch.float64,requires_grad=True);v=torch.randn_like(k,requires_grad=True)
  a=saved(q,k,v,None,None);ga=torch.autograd.grad(a.square().sum(),(q,k,v));b=compact(q,k,v,None,None);gb=torch.autograd.grad(b.square().sum(),(q,k,v));ds=[float((a-b).abs().max().detach())]+[float((x-y).abs().max()) for x,y in zip(ga,gb)];assert max(ds)<1e-10;deltas.append(ds)
 MultiHeadAttention.compute_attention_heads=staticmethod(compact)
 model,_,_=load_model(path=ROOT/'data/cache/foundation/tabpfn-v2.ckpt',model_seed=0);model.eval();model.cache_trainset_representation=False
 return model,deltas

def source_predict(model,cx,cy,qx):
 import torch
 classes,labels=np.unique(cy,return_inverse=True)
 keep=np.mean(cx[:1]==cx,axis=0)<1
 x=np.concatenate([cx[:,keep],qx[:,keep]]).astype('float32');y=torch.tensor(labels,dtype=torch.float32)[:,None]
 with torch.no_grad():logits=model(torch.from_numpy(x)[:,None],y,single_eval_pos=len(cx))[:,0,:len(classes)];p=(logits/.9).softmax(-1).numpy()
 return p,classes

def check(path,output,model_check=True):
 r=json.loads(Path(path).read_text());cfg=r['config'];datasets={};splits={v['id']:v for v in r['splits']};assert len(splits)==len(r['splits'])
 model,adapter=original_model() if model_check else (None,None);checks=[];total=0
 for name,meta in r['datasets'].items():
  file=ROOT/'data/cache/l069-source'/name/(name+'.csv');assert hashlib.sha256(file.read_bytes()).hexdigest()==meta['sha256'];df=pd.read_csv(file);target='label' if 'label' in df else df.columns[-1];y=pd.Categorical(df[target]).codes.astype(int);x=df.drop(columns=target).to_numpy(dtype='float32');datasets[name]=(x,y)
 assert r['source_commit']=='744c010457f68284faa7ae6ded793a8b3f3e03a4'
 assert r['protocol']['seeds']==[42,2023,789] and r['protocol']['levels']==[0,.2,.4,.6,.8,1.] and r['protocol']['model_seed']==0 and r['protocol']['temperature']==.9
 expected=set()
 for name in cfg['novelty']:
  for seed in cfg['seeds']:
   yy=datasets[name][1]
   for arm in ['v2','xgboost']:
    for held in np.unique(yy):expected.add(('novelty',name,seed,arm,'leave-one-class-out',int(held)))
    expected.add(('novelty',name,seed,arm,'natural-prevalence-class0',0))
 for name in cfg['features']:
  for seed in cfg['seeds']:
   for arm in ['v2','xgboost']:
    for level in [0,.2,.4,.6,.8,1.]:expected.add(('features',name,seed,arm,f'{level:.0%}',None))
 for seed in cfg['seeds']:
  for arm in ['v2','xgboost']:
   for kind in ['iid','covariate','concept']:expected.add(('shift','generated-threshold',seed,arm,kind,None))
 actual=[(v['axis'],v['dataset'],v['seed'],v['arm'],v['condition'],v.get('held_label')) for v in r['records']]
 assert len(actual)==len(set(actual)) and set(actual)==expected, 'Missing or duplicate predeclared measurements'
 groups={}
 for row in r['records']:groups.setdefault(tuple(row[k] for k in ['axis','dataset','arm','condition']),[]).append(row)
 assert len(groups)==len(r['summary'])
 for summary in r['summary']:
  key=tuple(summary[k] for k in ['axis','dataset','arm','condition']);rows=groups[key]
  for field,value in summary.items():
   if field.endswith('_mean'):
    metric=field[:-5];values=[np.mean([q[metric] for q in rows if q['seed']==seed and q.get(metric) is not None]) for seed in sorted({q['seed'] for q in rows}) if any(q['seed']==seed and q.get(metric) is not None for q in rows)]
    assert abs(np.mean(values)-value)<1e-12,(key,field)
    if metric+'_n' in summary:assert summary[metric+'_n']==len(values)
    sd=summary.get(metric+'_sd')
    assert (sd is None and len(values)==1) or abs(sd-np.std(values,ddof=1))<1e-12
 for row in r['records']:
  name=row['dataset'];seed=row['seed'];split=splits[row['split_id']];tr=np.array(split['context_ids']);te=np.array(split['query_ids']);assert len(set(tr))==len(tr) and len(set(te))==len(te) and not set(tr)&set(te)
  if row['axis']=='shift':
   rng=np.random.default_rng(seed);cx=rng.normal(size=(256,2));qx=rng.normal(size=(512,2));cy=(cx[:,0]>0).astype(int)
   if row['condition']=='covariate':qx[:,0]+=1.5
   y=(qx[:,0]>0).astype(int)
   if row['condition']=='concept':y=1-y
   assert np.array_equal(cx,split['context_x']) and np.array_equal(qx,split['query_x']) and np.array_equal(cy,split['context_y'])
  else:
   x,all_y=datasets[name];ids=np.arange(len(x))
   if cfg.get('cap') and len(ids)>cfg['cap']:ids=train_test_split(ids,train_size=cfg['cap'],random_state=seed,stratify=all_y)[0]
   if row['axis']=='novelty' and row['condition']=='leave-one-class-out':
    held=row['held_label'];novel=ids[all_y[ids]==held];known=ids[all_y[ids]!=held];strata=all_y[known] if len(novel)>=len(np.unique(all_y[known])) else None
    known_test,expected_train=train_test_split(known,train_size=len(novel),random_state=seed,stratify=strata);expected_test=np.r_[novel,known_test]
    assert np.array_equal(tr,expected_train) and np.array_equal(te,expected_test)
   elif row['axis']=='novelty':
    train,expected_test=train_test_split(ids,test_size=.2,random_state=seed,stratify=all_y[ids]);expected_train=train[all_y[train]!=row['held_label']]
    assert np.array_equal(te,expected_test) and np.array_equal(tr,expected_train);assert np.array_equal(split['excluded_train_ids'],train[all_y[train]==row['held_label']])
   else:
    expected_train,expected_test=train_test_split(ids,test_size=.2,random_state=seed);assert np.array_equal(tr,expected_train) and np.array_equal(te,expected_test)
   cx=x[tr];cy=all_y[tr];qx=x[te].astype(float);y=all_y[te]
   if row['axis']=='features':
    level=float(row['condition'].strip('%'))/100;columns=np.random.default_rng(seed).permutation(x.shape[1])[:math.floor(level*x.shape[1])];assert np.array_equal(columns,row['columns']);qx[:,columns]=cx.astype(float)[:,columns].mean(0)
  assert np.array_equal(y,split['targets']);p=np.asarray(row['probabilities']);classes=np.asarray(row['classes']);assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(1),1)
  if row['axis']=='features':
   baseline=next(v for v in r['records'] if v['axis']=='features' and v['dataset']==name and v['seed']==seed and v['arm']==row['arm'] and v['condition']=='0%')
   gap=row['accuracy']-baseline['accuracy'];assert abs(gap-row['delta_accuracy'])<1e-12 and abs(gap/baseline['accuracy']-row['relative_accuracy_gap'])<1e-12
   if row['condition']=='100%':assert np.ptp(p,axis=0).max()<4e-6, 'Identical query features must have identical predictions'
  if row['axis']=='shift' and row['condition']=='concept':
   baseline=next(v for v in r['records'] if v['axis']=='shift' and v['seed']==seed and v['arm']==row['arm'] and v['condition']=='iid')
   assert np.array_equal(p,baseline['probabilities']) and abs(row['accuracy']+baseline['accuracy']-1)<1e-12 and abs(row['auc']+baseline['auc']-1)<1e-12
  m=metrics(y,p,classes,r['protocol']['epsilon'])
  if row['axis']=='novelty':
   novel=(y==row['held_label']);assert np.array_equal(novel,split['novel']);continuous=1-p.max(1);binary=((p.max(1)>=.4)&(p.max(1)<=.6)).astype(int)
   m.update(auc=auc(novel,continuous),ap=ap(novel,continuous),interval_auc=auc(novel,binary),interval_ap=ap(novel,binary))
   assert abs(m['interval_auc']-(binary[novel].mean()+1-binary[~novel].mean())/2)<1e-12
  for key,value in m.items():
   if key in row:
    if value is None:assert row[key] is None,(row['split_id'],key,'must be undefined')
    else:assert abs(row[key]-value)<1e-10,(row['split_id'],key,row[key],value)
  delta=None
  if row['arm']=='v2' and model_check:
   ref,cs=source_predict(model,cx,cy,qx);assert np.array_equal(cs,classes);delta=float(np.max(abs(ref-p)));assert np.allclose(ref,p,atol=8e-5,rtol=8e-5),(row['split_id'],delta)
  elif row['arm']=='xgboost':
   from xgboost import XGBClassifier
   cs,encoded=np.unique(cy,return_inverse=True);est=XGBClassifier(**r['protocol']['xgboost'],random_state=seed);est.fit(cx,encoded);ref=est.predict_proba(qx);delta=float(np.max(abs(ref-p)));assert np.array_equal(cs,classes) and delta==0
  total+=len(y);checks.append(dict(split=row['split_id'],condition=row['condition'],arm=row['arm'],rows=len(y),probability_delta=delta));print('CHECK',row['split_id'],row['condition'],row['arm'],delta,flush=True)
 report=dict(status='PASS',evidence_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),records=len(checks),predictions=total,source_attention_adapter_forward_and_gradient_deltas=adapter,model_predictions_checked=model_check,scope='Independent CSV identities, splits, class alignment, all-row metrics, rank AUC and grouped AP; fresh XGBoost fits. '+('Original TabPFN 2.0.9 predictions checked with independently checked compact CPU attention.' if model_check else 'TabPFN predictions NOT_CHECKED in metrics-only mode.'),checks=checks)
 Path(output).write_text(json.dumps(report,indent=2)+'\n');return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('output');p.add_argument('--metrics-only',action='store_true');a=p.parse_args();check(a.evidence,a.output,not a.metrics_only)
