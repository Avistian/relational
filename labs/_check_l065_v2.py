"""Behavior, actual release hidden states, and live evidence identity for L065."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent

def check(namespace=None,save=True):
 from relkit import query_embeddings_l065_v2 as core
 ns=vars(core) if namespace is None else namespace;torch.set_num_threads(1)
 from _check_l064_v2 import reference_result,WHEEL_SHA
 # Sets up hash-verified historical wheel and old sklearn without replacing runtime sklearn.
 checkpoint=ns['ensure_checkpoint'](ROOT);reference_result(checkpoint)
 cache=ROOT/'data/cache/l065-source';cache.mkdir(parents=True,exist_ok=True);path=cache/'embeddings.pt'
 env=dict(os.environ,PYTHONPATH=str(ROOT/'data/cache/l064-source/official'),OMP_NUM_THREADS='1')
 subprocess.run([sys.executable,str(ROOT/'_reference_l065_v2.py'),str(checkpoint),str(path)],env=env,check=True)
 ref=torch.load(path,weights_only=False);model,_=ns['load_pretrained'](checkpoint,ns['TabPFNv2'](**ns['MODEL_CONFIG']))
 z=ns['extract_embeddings'](model,ref['x'][:22],ref['y'],ref['x'][22:],0,ns)
 expected=np.stack([h[0,:,-1].numpy() for h in ref['states']],1)
 qdelta=float(abs(z['query']-expected[22:]).max());cdelta=float(abs(z['context']-expected[:22]).max())
 assert qdelta<5e-5 and cdelta<5e-5,(qdelta,cdelta)
 assert np.allclose(z['query'][:,-1],ref['query'][0],atol=5e-5)
 assert np.allclose(z['context'][:,-1],ref['context'][0],atol=5e-5)
 # Fixed row/fold/seed/query-batch intervention; changing an excluded label changes no input.
 x=ref['x'];labels=ref['y'].copy();c=np.arange(1,22);q=np.array([0]);original=ns['extract_embeddings'](model,x[c],labels[c],x[q],0,ns)['query'];labels[0]=1-labels[0]
 changed=ns['extract_embeddings'](model,x[c],labels[c],x[q],0,ns)['query'];own_delta=float(abs(original-changed).max());assert own_delta==0
 # Context-role uses own label: same original contexts/features/query batch, just label0 flips.
 before=ns['extract_embeddings'](model,x[:22],ref['y'],x[22:],0,ns)['context'][0]
 after=ns['extract_embeddings'](model,x[:22],labels,x[22:],0,ns)['context'][0]
 role_deltas=abs(before-after).max(1);assert role_deltas[-1]>1e-3
 folds=np.array([1,0,1,0,2,2]);seen=[]
 def probe(c,q):
  assert not set(c)&set(q);seen.extend(q.tolist());return np.broadcast_to(q[:,None,None],(len(q),12,3)).copy()
 scattered,trace=ns['scatter_fold_embeddings'](6,folds,probe);assert np.array_equal(scattered[:,0,0],np.arange(6)) and sorted(seen)==list(range(6))
 fixture=np.arange(3*12*2).reshape(3,12,2);combined=ns['concatenate_layers'](fixture,[6,9,12]);assert np.array_equal(combined,np.concatenate([fixture[:,5],fixture[:,8],fixture[:,11]],1))
 candidates=[dict(layers=[6,9],C=1.,validation_accuracy=.8),dict(layers=[12],C=1.,validation_accuracy=.8),dict(layers=[6],C=.1,validation_accuracy=.7)]
 assert ns['choose_candidate'](candidates)==candidates[1]
 # Independent changed nested helper must invalidate identity, including genexpr globals.
 ident=core.kernel_identity(ns,ROOT)['sha256'];original_task=ns['extract_target_states']
 def changed_task(*a,**k):return original_task(*a,**k)*0
 ns['extract_target_states']=changed_task
 try:assert core.kernel_identity(ns,ROOT)['sha256']!=ident
 finally:ns['extract_target_states']=original_task
 assert core.kernel_identity(ns,ROOT)['sha256']==ident
 result=dict(status='PASS',kernel_sha256=ident,checkpoint_sha256=core.CHECKPOINT_SHA,wheel_sha256=WHEEL_SHA,query_all_layers_max_delta=qdelta,context_all_layers_max_delta=cdelta,actual_get_embeddings_final_layer=True,own_label_fixed_context_delta=own_delta,context_own_label_per_layer_delta=role_deltas.tolist(),scatter_original_order=True,layer_order=True,validation_tie_policy=True,nested_helper_identity_mutation=True,reference_worker_sha256=hashlib.sha256((ROOT/'_reference_l065_v2.py').read_bytes()).hexdigest(),checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 if save:(ROOT/'_check_l065_v2_results.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':print(json.dumps(check(),indent=2))
