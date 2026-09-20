"""Reconstruct selection/metrics, verify every saved checkpoint and data boundary."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import accuracy_score,f1_score
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'relkit'))
import compare_l099 as m
torch.set_num_threads(2)
r=json.loads((P/'_experiment_l099_results.json').read_text());assert r['status']=='COMPLETE' and len(r['runs'])==24
assert r['implementation_sha256']==hashlib.sha256((P/'relkit/compare_l099.py').read_bytes()).hexdigest()
g=m.load_graph(P/'data/l099');assert g['meta']==r['data']
assert len(set(r['splits']['train'])&set(r['splits']['test']))==0
assert len(set(r['splits']['val'])&set(r['splits']['test']))==0
expected={(a,lr,s) for a in m.ARMS for lr in m.LEARNING_RATES for s in m.SEEDS}
assert {(x['arm'],x['lr'],x['seed']) for x in r['runs']}==expected
for x in r['runs']:
 assert len(x['trace'])==60 and [t['epoch'] for t in x['trace']]==list(range(1,61))
 best=min(x['trace'],key=lambda t:t['val_ce']);assert x['selected_epoch']==best['epoch'] and x['best_val_ce']==best['val_ce']
 path=P/'_experiment_l099_results-checkpoints'/x['checkpoint_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==x['checkpoint_sha256']
 model=m.Model([1903,1,1],x['arm']);model.load_state_dict(torch.load(path,weights_only=True));model.eval()
 with torch.no_grad():z=model(g);v=float(torch.nn.functional.cross_entropy(z[g['val']],g['y'][g['val']]))
 assert abs(v-x['best_val_ce'])<1e-7
 if x['lr']==r['selected_rates'][x['arm']]:
  row=next(v for v in r['selected'] if v['arm']==x['arm'] and v['seed']==x['seed'])
  pred=z[g['test']].argmax(1).tolist();assert pred==row['test_predictions']
  assert abs(accuracy_score(r['test_labels'],pred)-row['accuracy'])<1e-12
  assert abs(f1_score(r['test_labels'],pred,average='macro')-row['macro_f1'])<1e-12
for arm in m.ARMS:
 scores={lr:sum(x['best_val_ce'] for x in r['runs'] if x['arm']==arm and x['lr']==lr)/3 for lr in m.LEARNING_RATES}
 assert min(scores,key=lambda lr:(scores[lr],lr))==r['selected_rates'][arm]
 for metric in ['accuracy','macro_f1']:
  vals=[x[metric] for x in r['selected'] if x['arm']==arm]
  assert abs(np.mean(vals)-r['summary'][arm][metric]['mean'])<1e-12
# Verify the raw release arithmetic independently of load_graph's graph construction.
raw=m.sio.loadmat(P/'data/l099/ACM.mat',spmatrix=True);sel=np.flatnonzero(raw['PvsC'][:,[0,1,9,10,13]].sum(1).A1)
assert sel.tolist()==r['data']['raw_node_ids'][0]
for i,key in enumerate(['PvsA','PvsL']):
 matrix=raw[key][sel].tocsr();keep=np.flatnonzero(matrix.sum(0).A1);assert keep.tolist()==r['data']['raw_node_ids'][i+1]
 assert matrix[:,keep].nnz==g['edges'][2*i].shape[1]
 assert torch.equal(g['edges'][2*i].flip(0),g['edges'][2*i+1])
report={'status':'PASS','complete_fits':24,'complete_epochs':1440,'restored_checkpoints':24,'selected_prediction_sets':12,'independent_sklearn_metrics':'EXACT','validation_only_selection':'EXACT','raw_data_and_reverse_edges':'PASS','paper_parity':'NOT_ESTABLISHED'}
(P/'_audit_l099_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
