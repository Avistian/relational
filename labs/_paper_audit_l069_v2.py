"""Independently inspect and execute isolated released protocol functions."""
import ast,hashlib,itertools,json,math,os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parent

def check(namespace=None):
 from relkit import openenv_l069_v2 as c
 s=vars(c) if namespace is None else namespace;folder=ROOT/'sources/l069-v2';manifest=json.loads((folder/'manifest.json').read_text())
 for name,digest in manifest['files'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
 raw=(folder/'newclass.py').read_text();tree=ast.parse(raw)
 # Execute the ORIGINAL assignments, avoiding model imports and training.
 body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_model')
 wanted=['pred_probs','aupr','roc_auc'];assign=[n for n in ast.walk(body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in wanted]
 p=np.array([[.50,.25,.25],[.34,.33,.33],[.85,.10,.05],[.55,.30,.15],[.80,.1,.1],[.70,.2,.1]])
 truth=np.array([1,1,0,0,0,1]);ns=dict(np=np,y_pred_proba=p.max(1),y_test=truth,threshold_min=.4,threshold_max=.6,roc_auc_score=roc_auc_score,average_precision_score=average_precision_score)
 exec(compile(ast.fix_missing_locations(ast.Module(body=assign,type_ignores=[])),'released-newclass-fragment','exec'),ns)
 scores=s['novelty_scores'](p);np.testing.assert_array_equal(ns['pred_probs'],scores['interval']);assert ns['roc_auc']==roc_auc_score(truth,scores['interval'])
 tpr=scores['interval'][truth==1].mean();tnr=1-scores['interval'][truth==0].mean();assert abs(ns['roc_auc']-(tpr+tnr)/2)<1e-12
 # Source class-split algorithm on each real label: compare actual row ordering.
 split_errors=[]
 for dataset in ['cmc','winequality-red','winequality-white']:
  x,y,ids,_=s['load_dataset'](ROOT,dataset)
  for held in np.unique(y):
   df=pd.DataFrame({'label':y});novel=df[df.label==held];known=df[df.label!=held];n=len(novel)
   sampled,remaining=train_test_split(known,test_size=len(known)-n,random_state=42,stratify=known.label if n>=known.label.nunique() else None)
   local=s['novel_split'](y,int(held),42);np.testing.assert_array_equal(local['context'],remaining.index);np.testing.assert_array_equal(local['query'],np.r_[novel.index,sampled.index]);split_errors.append([dataset,int(held)])
 # Execute original split_dataset on Iris, including all15 feature subsets.
 raw_features=(folder/'run_experiment.py').read_text();nodes=[n for n in ast.parse(raw_features).body if isinstance(n,ast.FunctionDef) and n.name in ['split_dataset','pearson']]
 ns2=dict(pd=pd,np=np,math=math,itertools=itertools,train_test_split=train_test_split)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'released-feature-split','exec'),ns2)
 data_dir=ROOT/'data/cache/l069-source';old=os.getcwd();os.chdir(data_dir.parent)
 # The source hardcodes ./dataset; redirect read_csv only for this local source audit.
 original=pd.read_csv
 def reader(path,*a,**kw):
  frame=original(data_dir/str(path).removeprefix('./dataset/'),*a,**kw)
  # Pandas3 infers dedicated string dtype; the release explicitly tests object.
  for column in frame:
   if pd.api.types.is_string_dtype(frame[column]):frame[column]=frame[column].astype(object)
  return frame
 pd.read_csv=reader
 try:
  train,tests=ns2['split_dataset']('iris','random','all')
  _,least=ns2['split_dataset']('iris','least','all');_,most=ns2['split_dataset']('iris','most','all')
  assert least[1] is least[-1] and most[1] is most[-1], 'Original mutable-level alias expected'
  try:ns2['split_dataset']('iris','random','1')
  except ValueError as e:assert 'No objects to concatenate' in str(e)
  else:raise AssertionError('Expected source degree1 target-column counting failure')
 finally:pd.read_csv=original;os.chdir(old)
 cx=train.iloc[:,:-1].to_numpy();qx=tests[0].iloc[:,:-1].to_numpy();checks=0
 for size in range(1,5):
  blocks=[s['impute_features'](cx,qx,cols) for cols in itertools.combinations(range(4),size)]
  np.testing.assert_allclose(np.concatenate(blocks),tests[size].iloc[:,:-1]);checks+=len(blocks)
 findings=['newclass uses binary interval decisions as ROC/AP scores','newclass integer-casts all features','newclass saves only final held-class result','newclass uses seed42 only','requirements select tabpfn6.4.1 and README v2.5','feature degree uses total columns including target','feature least/most correlations include test labels','feature random/all enumerates every subset','feature least/most saved levels alias one mutable object','numeric degree1 requests target-inflated feature count and fails','distribution subsampling calls omit stratify despite paper wording','paper Table1 Wine-Red MLP exceeds TabPFN','paper AppendixF relative gap differs from Table2 absolute displayed gaps']
 result=dict(status='PASS',commit=manifest['commit'],manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),split_label_cases=len(split_errors),feature_subset_cases=checks,fixture=dict(probabilities=p.tolist(),truth=truth.tolist(),interval=scores['interval'].tolist(),binary_auc=ns['roc_auc'],continuous_auc=float(roc_auc_score(truth,scores['continuous'])),binary_ap=ns['aupr'],continuous_ap=float(average_precision_score(truth,scores['continuous']))),findings=findings,scope='Current released protocol primitive parity; not historical end-to-end or published table parity; isolated least/most CSV reader restores object string dtype for pandas3 compatibility',kernel_identity=s['kernel_identity'](s,ROOT)['sha256'])
 return result
if __name__=='__main__':
 r=check();(ROOT/'_source_check_l069_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
