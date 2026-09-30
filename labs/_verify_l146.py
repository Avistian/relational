"""Scientific checks and exact evidence boundaries, independent of delivery checks."""
import ast,hashlib,json,sys
from pathlib import Path
import numpy as np
from _check_l146 import check_selection,check_pairs,check_context
from relkit.comparison_l146 import select_config,paired_errors,temporal_context
P=Path(__file__).resolve().parent;E=P/'evidence/l146'
for check,fn in [(check_selection,select_config),(check_pairs,paired_errors),(check_context,temporal_context)]:check(fn)
# Real anti-regressions: last-tie selection, positional pairing, global fallback.
mutants=[(check_selection,lambda names,vals:names[len(vals)-1-int(np.argmin(vals[::-1]))]),
         (check_pairs,lambda k,y,ak,a,bk,b:np.abs(np.asarray(a)-y)-np.abs(np.asarray(b)-y)),
         (check_context,lambda root,cutoff,adj,times,k,seed:[root]+list(adj)[:k-1])]
for check,fn in mutants:
 try:check(fn)
 except (AssertionError,ValueError,IndexError):pass
 else:raise AssertionError('A scientifically invalid learner implementation passed')
source=json.loads((E/'sources.json').read_text())
for name,h in source['reused_files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h
# Confirm the entire training implementation survived finalization changes intact.
old=(E/'executed-trainer-before-eval-check.py').read_text()
expected=old.replace("np.testing.assert_allclose(p2['pred'],pack['val_pred'],atol=0,rtol=0)","np.testing.assert_allclose(p2['pred'],pack['val_pred'],atol=1e-5,rtol=0)").replace('repeat_eval_max_error=0.,seconds=time.perf_counter()-start,',"repeat_eval_max_error=float(np.max(np.abs(p2['pred']-pack['val_pred']))),seconds=time.perf_counter()-start,").replace('# A second fixed-layout eval must reproduce the selected validation predictions.','# A second fixed-layout eval must agree within 1e-5 absolute (CUDA scatter ordering).')
current=(P/'_full_l146.py').read_text();assert current.startswith(expected)
# Only the two declared evaluation changes distinguish course RelGT from original mirror.
model=(P/'relkit/relgt_l145.py').read_text().replace('# Source-adapted RelGT, MIT license in sources/l145/LICENSE.','# Course RelGT, derived from the L145 source mirror; MIT license in sources/l145/LICENSE.\n# Two declared evaluation corrections; full source replay remains in relgt_l145.py.').replace('dropout_p=self.attention_dropout_rate,','dropout_p=self.attention_dropout_rate if self.training else 0.0,').replace('x_input = torch.randn(total_nodes, 1, device=device)','# Fixed normal features at evaluation; stable for a fixed batch layout.\n            generator = None if self.training else torch.Generator(device=device).manual_seed(146)\n            x_input = torch.randn(total_nodes, 1, device=device, generator=generator)')
assert model==(P/'relkit/relgt_course_l146.py').read_text()
contracts=(E/'executed-contracts.py').read_text()
assert contracts.replace('rng=random.Random(seed);chosen=[root]', 'if k==1:return [root]\n    rng=random.Random(seed);chosen=[root]')==(P/'relkit/comparison_l146.py').read_text()
a=json.loads((E/'prepared/audit.json').read_text());assert a['independently_rebuilt_labels']==8712
seen=0
for split,n in [('train',7453),('val',499),('test',760)]:
 z=np.load(E/f'prepared/{split}.npz');assert hashlib.sha256((E/f'prepared/{split}.npz').read_bytes()).hexdigest()==a['splits'][split]['cache_sha256']
 keys=list(zip(z['entity'],z['cutoff']));assert len(keys)==len(set(keys))==n and z['token_times'].shape==(n,32)
 assert np.all(z['indices'][:,0]==z['entity'])
 assert not ((z['token_times']!=-1)&(z['token_times']>z['cutoff'][:,None])).any()
 assert (z['hops']<=2).all() and (z['hops'][:,0]==0).all()
 padded=sum(32-len(set(row)) for row in z['global_ids']);assert padded==a['splits'][split]['padding_occurrences'];seen+=n*32
s=json.loads((E/'summary.json').read_text());assert s['verified_predictions']==7554 and len(s['runs'])==6
for r in s['runs']:
 for split in ['train','val','test']:assert r['context_sha256'][split]==a['splits'][split]['cache_sha256']
assert s['paired']['test']['relgt_wins']==0 and s['paired']['val']['relgt_wins']==3
m=json.loads((E/'mechanism.json').read_text());assert m['status']=='PASS' and m['gradient_tensors']==149
for phase in ['fit-gnn-0','fit-relgt-0','fit-gnn-1','fit-relgt-1','fit-gnn-2']:
 assert (E/(phase+'-failure.txt')).exists();assert 'no retraining' in json.loads((E/phase/'result.json').read_text())['recovery']
checksums=json.loads((E/'checkpoint-audit.json').read_text());assert checksums['verified_checkpoints']==6
execution=json.loads((P/'_execution_l146_results.json').read_text());pinned=json.loads((E/'notebook-final/execution.json').read_text())
assert execution['status']==pinned['status']=='PASS' and execution['executed_code_sha256']==pinned['code_sha256']
budget=json.loads((P/'_budget_l146.json').read_text());reserved=budget['overhead_reserve_usd']+sum(r['upper_usd'] for r in budget['reservations']);assert reserved<=10
costs=list(E.glob('*-cost.json'))+list(E.glob('notebook*/cost.json'));measured=sum(json.loads(p.read_text())['worker_body_usd'] for p in costs)
assert len(costs)==len(budget['reservations']),'Missing cost artifact, including failed attempts'
from _reproduce_l146 import preflight
assert preflight()['temporal']=='FAIL' and preflight()['cost']=='STOP'
r=dict(status='PASS',course_experiment='COMPLETE',full_selected_reproduction='INCOMPLETE',source_full_fits='NOT_RUN',fresh_canonical_rdl='NOT_RUN',
       historical_identity='NOT_ESTABLISHED',whole_paper='NOT_RUN',temporal_course_audit='PASS',audited_token_occurrences=seen,independently_rebuilt_labels=8712,
       independently_scored_predictions=7554,complete_primary_fits=6,verified_checkpoint_hashes=6,rejected_mutants=3,relgt_original_gradient_tensors=149,
       max_cuda_repeat_error=max(r['repeat_eval_max_error'] for r in s['runs']),standalone_notebook='PASS',pinned_notebook='PASS',
       conservative_reserved_usd=reserved,recorded_worker_body_usd=measured,invoice='NOT_ITEMIZED',learner='PENDING_WRITTEN_DEFENSE',live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l146_results.json').write_text(json.dumps(r,indent=2));print(r)
