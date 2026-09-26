"""Complete frozen slice census. No model selection or fitting happens here."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from ogb.nodeproppred import Evaluator
from relkit.ogb_l112 import load_arxiv
from relkit.error_l114 import neighborhood_properties,slice_masks,slice_metrics,choose_failure,TARGETS
P=Path(__file__).resolve().parent;out=P/'evidence/l114';out.mkdir(exist_ok=True)
x,edge,yt,st,audit=load_arxiv(P/'data/l112');y=yt.numpy();split={k:v.numpy() for k,v in st.items()}
with zipfile.ZipFile(P/'data/l112/arxiv.zip') as z:
 with z.open('arxiv/raw/node_year.csv.gz') as f:year=pd.read_csv(f,header=None,compression='gzip').to_numpy().reshape(-1)
props=neighborhood_properties(edge.numpy(),y,split['train']);preds={};summary={};files={};seed_rows=[];resources=0.;ev=Evaluator(name='ogbn-arxiv')
for model,origin in [('gcn','l112'),('mlp','l114')]:
 pred=[];scores={'train':[],'valid':[],'test':[]}
 for seed in range(10):
  d=P/f'evidence/{origin}/paper'/f'seed-{seed}';record=json.loads((d/'result.json').read_text());assert record['seed']==seed and record['epochs']==500 and record['status']=='COMPLETE'
  history=record['history'];assert len(history)==500 and [h['epoch'] for h in history]==list(range(1,501))
  selected=int(np.argmax([h['valid'] for h in history]));assert selected+1==record['selected_epoch']
  digest=hashlib.sha256((d/'predictions.npz').read_bytes()).hexdigest();assert digest==record['predictions_sha256'];files[str((d/'predictions.npz').relative_to(P))]=digest
  data=np.load(d/'predictions.npz');np.testing.assert_array_equal(data['y'],y)
  pred.append(data['pred']);row={'model':model,'seed':seed,'selected_epoch':selected+1}
  for k,ids in split.items():
   np.testing.assert_array_equal(data[k],ids)
   correct=int(np.count_nonzero(data['pred'][ids]==y[ids]));value=correct/len(ids)
   official=ev.eval({'y_true':y[ids,None],'y_pred':data['pred'][ids,None]})['acc'];assert value==official==record['scores'][k]==history[selected][k]
   scores[k].append(100*value);row[k]=100*value
  seed_rows.append(row)
  if model=='mlp':resources+=json.loads((d/'completed.json').read_text())['resource_usd']
 preds[model]=np.stack(pred)
 targets={'valid':73.,'test':71.74} if model=='gcn' else TARGETS
 summary[model]={k:{'mean_percent':float(np.mean(v)),'sample_sd_pp':float(np.std(v,ddof=1)),**({'target_percent':targets[k],'gap_pp':float(np.mean(v)-targets[k]),'verdict':'CLOSE' if abs(np.mean(v)-targets[k])<=.5 else 'OUTSIDE_TOLERANCE'} if k in targets else {})} for k,v in scores.items()}
masks=slice_masks(props,y,year);rows=[]
for population,ids in split.items():
 if population=='train':continue
 for (family,name),mask in masks.items():
  members=ids[mask[ids]];r={'population':population,'family':family,'slice':name,**slice_metrics(y,preds['gcn'],preds['mlp'],members)}
  if r['n']:
   r.update({'gcn_mean_percent':100*float(np.mean(r['gcn'])),'mlp_mean_percent':100*float(np.mean(r['mlp'])),'gcn_sd_pp':100*float(np.std(r['gcn'],ddof=1)),'mlp_sd_pp':100*float(np.std(r['mlp'],ddof=1)),'delta_sd_pp':float(np.std(r['delta_pp'],ddof=1))})
  rows.append(r)
 # Each family is a disjoint exhaustive partition; recover global accuracy by weights.
 for family in ['degree','homophily','class','year']:
  part=[r for r in rows if r['population']==population and r['family']==family and r['n']]
  assert sum(r['n'] for r in part)==len(ids)
  for model in ['gcn','mlp']:
   weighted=sum(np.array(r[model])*r['n'] for r in part)/len(ids)
   np.testing.assert_allclose(weighted,(preds[model][:,ids]==y[ids]).mean(1),rtol=0,atol=1e-15)
selected=choose_failure([r for r in rows if r['population']=='valid' and r['family']!='year'])
heldout=next(r for r in rows if r['population']=='test' and (r['family'],r['slice'])==(selected['family'],selected['slice']))
resources+=json.loads((out/'pilot/seed-100/completed.json').read_text())['resource_usd']
np.savez_compressed(out/'analysis-inputs.npz',labels=y,year=year,edge=edge.numpy(),**split,**props,gcn=preds['gcn'],mlp=preds['mlp'])
summary={'status':'COMPLETE','experiment':'OGB v6 Table6 MLP10x500; GCN10 checkpoints reused from L112','summary':summary,'seed_rows':seed_rows,'selected_validation_slice':selected,'same_rule_on_test':heldout,'slice_rows':rows,'input_files_sha256':files,'analysis_inputs_sha256':hashlib.sha256((out/'analysis-inputs.npz').read_bytes()).hexdigest(),'successful_resource_usd':resources,'data_audit':audit,'historical_identity':'NOT_ESTABLISHED','whole_paper_parity':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','uncertainty':'SD over10 independent initialization runs on this fixed graph; indexed delta pairs are not matched initialization or node IID confidence intervals'}
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
budget=json.loads((P/'_budget_l114.json').read_text());budget.update({'status':'COMPLETE','successful_resource_usd':resources});(P/'_budget_l114.json').write_text(json.dumps(budget,indent=2))
print(json.dumps({k:summary[k] for k in ['summary','selected_validation_slice','same_rule_on_test','successful_resource_usd']},indent=2))
