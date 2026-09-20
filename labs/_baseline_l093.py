"""Training-only field-frequency baseline evaluated on exactly the saved test queries."""
import json,numpy as np
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import load_oag,field_protocol
P=Path(__file__).resolve().parent;m=json.loads((P/'_sources_l093.json').read_text());g=load_oag(P/'data/l093/graph_NN.pk',m['nn_data']['sha256']);c,pairs=field_protocol(g)
counts={f:0 for f in c}
for fields,year in pairs['train'].values():
 for field in fields:counts[field]+=1
order=sorted(c,key=lambda f:-counts[f]);ranks={f:i+1 for i,f in enumerate(order)}
r=json.loads((P/'_teaching_l093_results.json').read_text());runs=[]
for row in [v for v in r['runs'] if v['arm']=='hgt']:
 values=[]
 for q in row['test_queries']:
  rr=np.array([ranks[f] for f in q['true_field_ids']]);den=lambda x:np.where(x==1,1,np.log2(x))
  values.append(((1/den(rr)).sum()/(1/den(np.arange(1,len(rr)+1))).sum(),1/rr.min()))
 runs.append({'seed':row['seed'],'ndcg':float(np.mean([v[0] for v in values])),'mrr':float(np.mean([v[1] for v in values]))})
top=order[0];test_fraction=np.mean([top in fields for fields,_ in pairs['test'].values()]);row=g.node_feature['field'].loc[top]
result={'status':'PASS','scope':'Frequency counts use train labels only; no model fitting or test selection','top_field_id':int(top),'top_field_name':str(row.get('name',row.get('title',row.get('id','unknown')))),'train_fraction':counts[top]/len(pairs['train']),'test_fraction':float(test_fraction),'runs':runs,'summary':{metric:{'mean':float(np.mean([v[metric] for v in runs])),'sample_sd':float(np.std([v[metric] for v in runs],ddof=1))} for metric in ['ndcg','mrr']}}
(P/'_baseline_l093_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
