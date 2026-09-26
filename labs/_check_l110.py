"""Behavioral contract: a complete checkpoint must recover predictions after future use."""
import copy,json
from pathlib import Path
import numpy as np
import torch
from relkit import tgn_l102 as base
try:
 from relkit import checkpoint_l110 as c
except ImportError:
 c=base

def check_restore(restore):
 torch.manual_seed(18);torch.set_num_threads(1)
 m=base.TGN(np.zeros((5,4),np.float32),np.random.RandomState(3).normal(size=(6,4)).astype('float32'),dropout=0,neighbors=2)
 u=np.array([1,1,2,1,2]);v=np.array([3,4,3,3,4]);t=np.array([1.,3.,4.,9.,10.]);e=np.arange(1,6)
 m.finder=base.temporal_neighbors(u,v,t,e,5);m.eval()
 def score(i):return m.probabilities(u[i:i+1],v[i:i+1],np.array([4]),t[i:i+1],e[i:i+1])
 score(0);m.detach_state();score(1);m.detach_state()
 saved={'weights':copy.deepcopy(m.state_dict()),'temporal_state':m.snapshot()}
 expected=[x.detach().clone() for x in score(2)]
 score(3);m.detach_state()
 with torch.no_grad():next(m.parameters()).add_(.3)
 before=copy.deepcopy(saved)
 restore(m,saved)
 got=score(2)
 assert all(torch.equal(a,b) for a,b in zip(expected,got)), 'Checkpoint restoration must recover queued messages as well as weights and memory'
 assert saved['temporal_state'][2].keys()==before['temporal_state'][2].keys(), 'Restoration must not alias the saved queue'
 for i in [0,1]:assert torch.equal(saved['temporal_state'][i],before['temporal_state'][i]), 'Saved memory/clocks mutated after restore'
 for node,values in before['temporal_state'][2].items():
  assert all(torch.equal(a,b) for a,b in zip(saved['temporal_state'][2][node],values)), 'Saved queued message mutated'
 for key,value in before['weights'].items():assert torch.equal(saved['weights'][key],value), 'Saved weights aliased'

def check_batches(fn):
 for times,size in [([1,2,2,2,3,4],2),([1,1,1,1],2),([1,2,3,4],2),([],2)]:
  ev={'t':np.array(times,dtype=float),'e':np.arange(len(times)),'u':np.ones(len(times),dtype=int)};parts=list(fn(ev,size))
  assert sum(len(x['t']) for x in parts)==len(times)
  assert all(a['t'][-1]<b['t'][0] for a,b in zip(parts,parts[1:])), 'Equal-time events cross a memory-update boundary'
  if parts:np.testing.assert_array_equal(np.concatenate([x['e'] for x in parts]),ev['e'])

def check_legal(fn):
 assert fn(np.array([2.,5.,4.]),np.array([7.,5.,4.]),5.).tolist()==[False,False,True], 'Strict history needs both clocks; a tie is not past'

def aliasing_restore(model,checkpoint):
 model.load_state_dict(checkpoint['weights'])
 model.memory,model.last_update,model.pending=checkpoint['temporal_state']

if __name__=='__main__':
 restore=getattr(c,'restore_checkpoint',lambda m,s:m.load_state_dict(s['weights']))
 check_restore(restore)
 check_batches(c.strict_batches)
 check_legal(c.legal_history)
 rejected=[]
 for name,check,fn in [('weights_only',check_restore,lambda m,s:m.load_state_dict(s['weights'])),('alias_saved_state',check_restore,aliasing_restore),('split_ties',check_batches,base.batches),('event_only',check_legal,lambda t,a,q:t<q),('inclusive_ties',check_legal,lambda t,a,q:(t<=q)&(a<=q))]:
  try:check(fn)
  except (AssertionError,RuntimeError):rejected.append(name)
  else:raise AssertionError('Mutant survived: '+name)
 out={'status':'PASS','rejected_mutants':rejected,'atomic_prediction_recovery':'EXACT','strict_ties':'PASS','two_clock_eligibility':'PASS'}
 Path(__file__).with_name('_check_l110_results.json').write_text(json.dumps(out,indent=2));print(out)
