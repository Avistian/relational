"""Run all six frozen fits. Existing evidence must be explicitly archived first."""
import argparse,hashlib,json,platform,shutil,time
from pathlib import Path
import numpy as np
import torch
from relkit.multitask_l173 import load_packet,train_run
P=Path(__file__).resolve().parent;INPUT=P/'evidence/l173'
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
E=args.output.resolve() if args.output else INPUT
a,t,m=load_packet(INPUT)
if args.output:
    E.mkdir(parents=True,exist_ok=False)
    for name in [*m['files'],'manifest.json']:shutil.copy2(INPUT/name,E/name)
for name,h in m['source_files'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==h,name
if any((E/f'{arm}-{seed}').exists() for seed in [0,1,2] for arm in ['cell','task']):
    raise SystemExit('Existing run evidence: use --output with a new directory')
results=[];start=time.monotonic()
(E/'report.json').write_text(json.dumps(dict(status='INCOMPLETE',expected_runs=6,runs=results,whole_paper_reproduction='NOT_RUN'),indent=2)+'\n')
for seed in [0,1,2]:
    for arm in ['cell','task']:
        results.append(train_run(a,t,arm,seed,E/f'{arm}-{seed}'))
        (E/'report.json').write_text(json.dumps(dict(status='INCOMPLETE',expected_runs=6,runs=results,whole_paper_reproduction='NOT_RUN'),indent=2)+'\n')
report=dict(status='COMPLETE_SELECTED_COURSE_PRETRAINING',experiment='L173 F1 Multi-task Masked-cell Pretraining',runs=results,seconds=time.monotonic()-start,python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cloud_usd=0,whole_paper_reproduction='NOT_RUN',transfer='NOT_ESTABLISHED',historical_availability='NOT_ESTABLISHED',learner='PENDING_WRITTEN_DEFENSE')
(E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('Completed six fresh fits')
