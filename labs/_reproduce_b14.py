"""Display or dispatch the exact selected release; never substitute a small preset."""
import argparse,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b14'
PLAN=dict(name='B14-TABPFNREL-OSS-F1-DNF',paper='2608.16319v2 Table4',target_auroc=.7145,source_commit='e89002200e18be6d8d7a55f8a5ab50c993ce4d5d',checkpoint_revision='24a16a89d245878b846555110985634aa2e656d7',relbench='2.1.2',seed=0,depths=[2,3,4],support_cap=100000,pool_inflation=4,estimators=8,text=False,inner_snapshot='2005-01-01',outer_snapshot='2010-01-01',refit='train+validation; selected and default',selection='validation AUROC; first candidate on tie',test_queries=702,cloud_cap_usd=10,cloud_stop_usd=8,local_seconds_cap=3600,historical_checkpoint_identity='NOT_ESTABLISHED',unrun_scope=['whole21task benchmark','hosted API','foundation model pretraining'])
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--show-plan',action='store_true');a.add_argument('--phase',choices=['pilot','full']);args=a.parse_args()
 print(json.dumps(PLAN,indent=2))
 if args.phase:
  required=['paper-preflight.json','warm-receipt.json','checkpoint.json','cloud-input-lock.json','cloud-budget.json']+(['admission.json'] if args.phase=='full' else [])
  missing=[n for n in required if not (E/n).exists()]
  if missing:raise SystemExit('INCOMPLETE_PREPARATION: '+', '.join(missing))
  subprocess.run([sys.executable,'-m','modal','run',str(R/'modal/b14_tabpfn_rel.py'),'--phase',args.phase],cwd=R,check=True)
