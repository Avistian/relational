"""Independent sampling oracle, source integrity, edge masking and stored-metric audit."""
import ast,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import load_oag,field_protocol,sample_oag,file_sha256,ranking_metrics
P=Path(__file__).resolve().parent;m=json.loads((P/'_sources_l093.json').read_text())
for name,h in m['files'].items():assert file_sha256(P/'sources/hgt-l093'/name)==h,name
g=load_oag(P/'data/l093/graph_NN.pk',m['nn_data']['sha256']);c,pairs=field_protocol(g)
# Execute the archived source sampling and conversion functions, with only NumPy aliases repaired.
ns={'np':np,'torch':torch,'defaultdict':defaultdict}
for name,functions in [('utils.py',['feature_OAG']),('data.py',['sample_subgraph','to_torch'])]:
 text=(P/'sources/hgt-l093/OAG/pyHGT'/name).read_text();lines=text.splitlines()
 for node in ast.parse(text).body:
  if isinstance(node,ast.FunctionDef) and node.name in functions:
   code='\n'.join(lines[node.lineno-1:node.end_lineno]).replace('dtype=np.float)','dtype=float)').replace('dtype=np.str)','dtype=str)')
   exec(compile(code,name,'exec'),ns)
checks=[]
for seed in [7,31]:
 size=8;depth=2;width=8
 ours,a=sample_oag(g,pairs['train'],c,seed,size,depth,width,2014)
 np.random.seed(seed);ids=np.random.choice(list(pairs['train']),size,replace=False)
 inputs=np.array([[i,pairs['train'][i][1]] for i in ids])
 feature,times,edges,index,_=ns['sample_subgraph'](g,{t:True for t in g.times if t is not None and t<2015},depth,width,{'paper':inputs})
 edges['paper']['field']['rev_PF_in_L2']=[e for e in edges['paper']['field']['rev_PF_in_L2'] if e[0]>=size]
 edges['field']['paper']['PF_in_L2']=[e for e in edges['field']['paper']['PF_in_L2'] if e[1]>=size]
 # Old to_torch uses torch.FloatTensor(list_of_arrays), compatible but slow.
 x,nt,et,ei,er,_,_=ns['to_torch'](feature,times,edges,g)
 torch.testing.assert_close(x,ours[0],atol=0,rtol=0);assert torch.equal(nt,ours[1])
 canon=lambda e,r,t:sorted(zip(e[0].tolist(),e[1].tolist(),r.tolist(),t.tolist()))
 assert canon(ei,er,et)==canon(ours[2],ours[3],ours[4])
 assert ids.tolist()==a['paper_ids']
 for t,values in index.items():assert values.tolist()==a['node_ids'][t]
 rel=a['relation_ids'];source,target=ours[2];seeds=ours[5]
 assert not ((ours[3]==rel['PF_in_L2'])&torch.isin(source,seeds)).any()
 assert not ((ours[3]==rel['rev_PF_in_L2'])&torch.isin(target,seeds)).any()
 checks.append({'seed':seed,'nodes':a['nodes'],'edges':a['edges'],'features_node_ids_typed_timed_edges':'EXACT'})
# Independent metric example: method0 discounts first two ranks equally.
log=torch.tensor([[0.,2.,1.,-1.],[3.,2.,1.,0.]])
y=torch.tensor([[.5,0.,.5,0.],[0.,0.,0.,1.]])
nd,mrr=ranking_metrics(log,y)
assert abs(nd[0].item()-(.5+.5/np.log2(3)))<1e-6
assert abs(nd[1].item()-.5)<1e-6
assert mrr.tolist()==[.5,.25]
result={'status':'PASS','source_hashes':len(m['files']),'data_sha256':m['nn_data']['sha256'],'graph_nodes':{k:len(v) for k,v in g.node_feature.items()},'split_sizes':{k:len(v) for k,v in pairs.items()},'candidate_fields':len(c),'upstream_sampler_parity':checks,'both_label_edge_directions_removed':'PASS','release_dcg_method0':'PASS','loader_compatibility':'Old serialized defaultdict factories replaced by dict; stored keys/values retained. Original-runtime loader identity NOT_CHECKED.'}
path=P/'_teaching_l093_results.json'
if path.exists():
 r=json.loads(path.read_text())
 for run in r['runs']:
  for q in run['test_queries']:
   ranks=np.array(q['relevant_ranks']);den=lambda x:np.where(x==1,1.,np.log2(x))
   expected=(1/den(ranks)).sum()/(1/den(np.arange(1,len(ranks)+1))).sum();mr=1/ranks.min()
   assert abs(expected-q['ndcg'])<2e-6 and abs(mr-q['mrr'])<2e-6
  criterion='valid_kl' if run['config']['selection']=='valid_loss' else 'valid_ndcg'
  values=[v[criterion] for v in run['trace']];expected=int(np.argmin(values) if criterion=='valid_kl' else np.argmax(values))+1
  assert expected==run['selected_epoch']
 result['stored_runs_audited']=len(r['runs'])
 # Same seed must have exactly the same sampled train/valid/test IDs across arms.
 for seed in set(v['seed'] for v in r['runs']):
  rows=[v for v in r['runs'] if v['seed']==seed]
  for row in rows[1:]:
   assert [q['paper_id'] for q in row['test_queries']]==[q['paper_id'] for q in rows[0]['test_queries']]
   assert [e['train_sampling_seeds'] for e in row['trace']]==[e['train_sampling_seeds'] for e in rows[0]['trace']]
 result['paired_sampling_streams']='PASS'
(P/'_audit_l093_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
