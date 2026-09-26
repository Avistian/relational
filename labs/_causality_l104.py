"""Metamorphic causal checks and notebook-task mutation tests on live model paths."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
from relkit import leakage_l104 as audit
from relkit.tgat_l103 import TGAT,compact_weights
P=Path(__file__).resolve().parent

def causal_check():
 torch.set_num_threads(1);rng=np.random.default_rng(104)
 events={'u':np.array([1,2,1,3]),'v':np.array([3,4,5,5]),'t':np.array([1.,3.,6.,9.]),'e':np.arange(1,5)}
 nodes=np.zeros((6,4),np.float32);edges=np.vstack([np.zeros((1,4)),rng.normal(size=(4,4))]).astype(np.float32)
 changed=edges.copy();changed[3:]+=40 # only feature rows at times 6 and 9
 queries={'u':np.array([1,2]),'v':np.array([3,4]),'t':np.array([5.,5.]),'e':np.array([10,11])}
 questions={'e':queries['e'],'negative':np.array([5,5]),'batch':np.array([0,0])}
 results={}
 torch.manual_seed(104);first=TGAT(audit.AuditFinder(events,6),nodes,edges)
 for mode in ['strict','lookahead']:
  outputs=[]
  for features in [edges,changed]:
   finder=audit.AuditFinder(events,6,mode,lookahead=2)
   model=TGAT(finder,nodes,features);model.load_state_dict(compact_weights(first),strict=False)
   pred=audit.score_fixed_questions(model,queries,questions,rng_seed=18)
   outputs.append(np.r_[pred['p'],pred['n']])
  error=float(np.max(np.abs(outputs[0]-outputs[1])));results[mode]=error
 assert results['strict']==0,'Forbidden future features affected a strict prediction'
 assert results['lookahead']>1e-7,'Positive-control leak was not detected'
 # Test the actual task checks with deliberately wrong implementations.
 spec=importlib.util.spec_from_file_location('builder_checks',P/'_check_l104.py');checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)
 mutants={
  'drop_availability':('eligible_history',lambda e,a,c,inclusive=False:np.asarray(e)<=c if inclusive else np.asarray(e)<c),
  'include_ties':('eligible_history',lambda e,a,c,inclusive=False:(np.asarray(e)<=c)&(np.asarray(a)<=c)),
  'ignore_label_maturity':('admissible_training',lambda q,a,f:np.asarray(q)<f),
  'ignore_pairing':('paired_ap',lambda a,b:{'delta_pp':0}),
 }
 killed=[]
 for name,(function,replacement) in mutants.items():
  original=getattr(checks,function);setattr(checks,function,replacement)
  try:
   try:checks.check()
   except (AssertionError,ValueError):killed.append(name)
  finally:setattr(checks,function,original)
 assert len(killed)==len(mutants)
 return {'status':'PASS','future_feature_perturbation_max_probability_change':results,'strict_invariance':'EXACT','lookahead_positive_control':'DETECTED','killed_mutations':killed,'scope':'Small synthetic graph checks a causal property; not paper-result evidence'}
if __name__=='__main__':
 r=causal_check();(P/'_causality_l104_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
