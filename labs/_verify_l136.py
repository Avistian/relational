"""Independent regressions and provenance checks for the lesson's scientific claims."""
import ast,hashlib,json,math,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from _check_l136 import check_align,check_metric,check_board
from relkit.leaderboard_l136 import align_predictions,regression_score,complete_board
P=Path(__file__).resolve().parent
for test,fn in [(check_align,align_predictions),(check_metric,regression_score),(check_board,complete_board)]:test(fn)
mutants=[('positional score',check_align,lambda e,k,v:list(v)),('entity-only join',check_align,lambda e,k,v:[dict(zip([x[0] for x in k],v))[x[0]] for x in e]),('unnormalized MAE',check_metric,lambda y,p,s:dict(mae=np.mean(np.abs(np.array(y)-p)),nmae=np.mean(np.abs(np.array(y)-p)),count=len(y))),('test-fitted scale',check_metric,lambda y,p,s:regression_score(y,p,np.std(y,ddof=1))),('partial board mean',check_board,lambda s,t:sum(s.values())/len(s)),('extra tasks ignored',check_board,lambda s,t:sum(s[x] for x in t)/len(t))]
rejected=[]
for name,test,fn in mutants:
 try:test(fn)
 except Exception:rejected.append(name)
 else:raise AssertionError('Mutant survived: '+name)
source=json.loads((P/'_sources_l136.json').read_text())
for name,meta in source['files'].items():assert hashlib.sha256((P/'sources/l136'/name).read_bytes()).hexdigest()==meta['sha256']
r=json.loads((P/'evidence/l136/leaderboard.json').read_text())
for name,meta in r['data_files'].items():assert hashlib.sha256((P/'results/l136/leaderboard-data'/name).read_bytes()).hexdigest()==meta['sha256']
assert len(r['canonical_tasks'])==9 and len(r['entries'])==3
for issue in r['entries']:
 assert math.isclose(complete_board({t:r['tasks'][t]['entries'][issue]['nmae'] for t in r['canonical_tasks']},r['canonical_tasks']),r['entries'][issue]['reported'],abs_tol=1e-12)
assert sum(t['rows']*3 for t in r['tasks'].values())==2480739
budget=json.loads((P/'_budget_l136.json').read_text());reserved=sum(x['upper_usd'] for x in budget['reservations']);assert reserved+3<=10
# A private dry-run creates a usable operator without dispatching paid work.
with tempfile.TemporaryDirectory(prefix='l136-rerun-') as tmp:
 target=Path(tmp)/'fresh';subprocess.run([sys.executable,str(P/'_new_run_l136.py'),'--directory',str(target)],check=True,capture_output=True)
 fresh=json.loads((target/'labs/_budget_l136.json').read_text());assert not fresh['reservations']
 for name,digest in fresh['source_hashes'].items():assert hashlib.sha256((target/name).read_bytes()).hexdigest()==digest
report=dict(status='PASS',rejected_mutants=rejected,pinned_sources=len(source['files']),data_files=len(r['data_files']),task_entry_pairs=27,predictions_rescored=r['predictions_rescored'],fresh_operator='DRY_RUN_PASS_NO_DISPATCH',maximum_allocated_worker_usd=reserved,overhead_usd=3,hosted_scale_discrepancy='RECORDED_NOT_SILENTLY_CORRECTED')
(P/'_verify_l136_results.json').write_text(json.dumps(report,indent=2));print(report)
