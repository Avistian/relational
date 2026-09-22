"""Independent released preprocessing/split and stronger state-boundary checks."""
import ast,hashlib,importlib.util,json,os,random,sys,tempfile,types
from pathlib import Path
import numpy as np
import torch
P=Path(__file__).resolve().parent
from relkit.tgn_l102 import *
torch.set_num_threads(1)
def main():
 nodes,edges,data,audit=load_wikipedia(P/'data/l102')
 # Parse the original released source without executing its CLI entrypoint.
 raw=(P/'sources/l102/utils/preprocess_data.py').read_text();tree=ast.parse(raw)
 tree.body=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.FunctionDef))]
 env={};exec(compile(tree,'original-preprocess','exec'),env)
 frame,features=env['preprocess'](str(P/'data/l102/wikipedia.csv'));frame=env['reindex'](frame,True)
 assert np.array_equal(frame.u.to_numpy(),data['full']['u'])
 assert np.array_equal(frame.i.to_numpy(),data['full']['v'])
 assert np.array_equal(frame.ts.to_numpy(),data['full']['t'])
 assert np.array_equal(features.astype('float32'),edges[1:])
 spec=importlib.util.spec_from_file_location('released_data',P/'sources/l102/utils/data_processing.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 rng=random.Random();module.random=types.SimpleNamespace(seed=rng.seed,sample=lambda pop,k:rng.sample(tuple(pop),k))
 cwd=Path.cwd()
 with tempfile.TemporaryDirectory(prefix='l102-source-data-') as tmp:
  d=Path(tmp)/'data';d.mkdir();frame.to_csv(d/'ml_wikipedia.csv');np.save(d/'ml_wikipedia.npy',edges);np.save(d/'ml_wikipedia_node.npy',nodes)
  try:
   os.chdir(tmp);result=module.get_data('wikipedia')
  finally:os.chdir(cwd)
 for name,original in zip(['full','train','val','test','new_val','new_test'],result[2:]):
  for k,attr in [('u','sources'),('v','destinations'),('t','timestamps'),('e','edge_idxs')]:
   assert np.array_equal(data[name][k],getattr(original,attr)),(name,k)
 # Independent sort/filter oracle across graph histories, including equal-time events.
 rng=np.random.RandomState(29);cases=0
 for repeat in range(20):
  u=rng.randint(1,5,40);v=rng.randint(5,9,40);t=np.sort(rng.randint(0,15,40)).astype(float);ei=np.arange(1,41)
  f=temporal_neighbors(u,v,t,ei,9)
  for node in range(9):
   for cutoff in [0,3,7,15]:
    n,e,tt=f(np.array([node]),np.array([cutoff]),3)
    expected=[(b if a==node else a,j,time) for a,b,time,j in zip(u,v,t,ei) if (a==node or b==node) and time<cutoff][-3:]
    assert n[0].tolist()==[0]*(3-len(expected))+[x[0] for x in expected]
    assert e[0].tolist()==[0]*(3-len(expected))+[x[1] for x in expected];cases+=1
 # Round-trip snapshots include queued messages and reproduce the next prediction.
 model=TGN(np.zeros((9,4),np.float32),np.ones((41,4),np.float32),dropout=0);model.finder=f;model.eval()
 b=(np.array([1,2]),np.array([5,6]),np.array([7,8]),np.array([1.,2.]),np.array([1,2]))
 model.probabilities(*b);model.detach_state();state=model.snapshot()
 next_b=(np.array([1,3]),np.array([6,8]),np.array([7,5]),np.array([5.,6.]),np.array([3,4]))
 a=model.probabilities(*next_b);model.restore(state);c=model.probabilities(*next_b)
 assert all(torch.equal(x,y) for x,y in zip(a,c))
 # Negative-only nodes can receive candidate updates from real prior history, but no new messages.
 fresh=TGN(np.zeros((9,4),np.float32),np.ones((41,4),np.float32),dropout=0);fresh.finder=f
 fresh.probabilities(*b);assert set(fresh.pending)=={1,2,5,6}
 out={'status':'PASS','released_preprocessing':'Exact IDs/times and float32 feature equality on all 157474 rows',
      'released_split':'Exact event arrays for all six partitions; Python set-to-tuple compatibility adaptation',
      'temporal_oracle_cases':cases,'snapshot_replay':'EXACT','negative_message_exclusion':'PASS',
      'processed_cache_sha256':hashlib.file_digest((P/'data/l102/processed.npz').open('rb'),'sha256').hexdigest(),
      'counts':audit['counts']}
 (P/'_audit_l102_results.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
