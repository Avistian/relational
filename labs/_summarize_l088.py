"""Report completed full-fold candidates without mislabeling an incomplete search."""
import hashlib,json,importlib.metadata as md
from pathlib import Path
import numpy as np
from relkit.gin_l088 import paper_grid,select_epoch,summarize_grid
LAB=Path(__file__).resolve().parent
records=[json.loads(p.read_text()) for p in (LAB/'results/l088/paper').glob('*.json')]
if len(records)==80:
 report=summarize_grid(records)
else:
 candidates=[]
 for config in paper_grid():
  rows=sorted([r for r in records if all(r['config'][k]==v for k,v in config.items())],key=lambda r:r['fold'])
  if len(rows)!=10:continue
  assert [r['fold'] for r in rows]==list(range(10))
  curves=np.array([[e['val_acc'] for e in r['curves']] for r in rows]);epoch=select_epoch(curves);values=curves[:,epoch]
  candidates.append({'config':config,'epoch':epoch+1,'mean':float(values.mean()),'fold_values':values.tolist(),'population_sd':float(values.std(ddof=0)),'sample_sd':float(values.std(ddof=1))})
 assert candidates,'At least one complete ten-fold candidate is required'
 report={'target':'Xu et al. 2019 Table 1 MUTAG GIN-0','paper_mean':.894,'paper_sd':.056,
 'status':'FULL_FOLDS_FIXED_CONFIG_EXECUTED','full_hyperparameter_search':'INCOMPLETE',
 'historical_parity':'INCOMPARABLE','reason':'A full ten-fold, 350-epoch candidate is executed; remaining grid not completed. Modern runtime/splitter and reduction order; historical split IDs and winning config unavailable.',
 'selection':'Common epoch within each completed configuration. No claim that the best of all eight configurations has been found.',
 'candidates':candidates,'fold_runs':len(records),'optimizer_updates':len(records)*350*50,
 'planned_fold_runs':80,'remaining_fold_runs':80-len(records),
 'interrupted_work':'Additional fold jobs may have started before shutdown; no partial-fold outputs are used or retained.',
 'fixed_config_choice':'First declared grid entry, chosen before validation results; not tuned to the published target.',
 'uncompleted_configurations':[c for c in paper_grid() if c not in [r['config'] for r in candidates]]}
report['identity']={'source_sha256':hashlib.sha256((LAB/'relkit/gin_l088.py').read_bytes()).hexdigest(),'runtime':{x:md.version(x) for x in ['torch','numpy','scikit-learn']},'runner_sha256':hashlib.sha256((LAB/'_run_l088.py').read_bytes()).hexdigest()}
(LAB/'_paper_l088_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
