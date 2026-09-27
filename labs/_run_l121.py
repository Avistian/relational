"""L121 full-data entrypoint: retain source-checked L118 trainer and its cost gate.
The cached pilot is experimental: disk-backed full-cache timing remains unmeasured.
"""
import json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
if __name__=='__main__':
 pilot=json.loads((P/'_pilot_l121_results.json').read_text());budget=json.loads((P/'_budget_l121.json').read_text())
 print(json.dumps({'selected_experiment':'Cvitkovic2020 Table4 Home Credit GCN, five full folds','cached_pilot_projection_usd':pilot['projected_five_fold_300_epoch_usd'],'cap_usd':budget['aggregate_cap_usd'],'full_data_status':'NOT_RUN','cached_full_training':'NOT_INTEGRATED: full disk-cache path not benchmarked','runnable_fallback':'unchanged L118 full trainer, with its conservative uncached projection and input checks'},indent=2),flush=True)
 # L121 approval is bounded at 10USD; command-line budget escalation must not bypass it.
 if any(a.startswith(('--budget-usd','--reserve-usd')) for a in sys.argv[1:]):raise SystemExit('Budget changes require an explicitly revised approved contract.')
 if '--execute' in sys.argv:
  needed=pilot['projected_five_fold_300_epoch_usd']+budget['overhead_retry_reserve_usd']+budget['pilot_reservation_usd']
  if needed>budget['aggregate_cap_usd']:raise SystemExit(f'REFUSED: even cached pilot projection plus reserves is {needed:.2f}USD >10USD. No training launched.')
 command=[sys.executable,str(P/'_run_l118.py'),*sys.argv[1:]]
 if '--output' not in sys.argv:command+=['--output',str(P/'results/l121/full')]
 raise SystemExit(subprocess.call(command))
