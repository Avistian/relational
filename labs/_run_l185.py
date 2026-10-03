"""Persist every input, held-out prediction and paired intervention from all seeds."""
import hashlib,importlib.metadata,json,platform
from pathlib import Path
from relkit.causal_l185 import full_experiment
P=Path(__file__).resolve().parent; E=P/'evidence/l185'
if __name__=='__main__':
 outputs,report=full_experiment()
 for seed,(tables,predictions,potentials,_) in enumerate(outputs):
  d=E/f'seed-{seed}';d.mkdir(exist_ok=True)
  for name,table in {**tables,'predictions':predictions,'potentials':potentials}.items():table.to_parquet(d/(name+'.parquet'),index=False)
 (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 environment={'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','pyarrow','nbformat','nbclient','nbconvert','matplotlib']}}
 (E/'environment.json').write_text(json.dumps(environment,indent=2)+'\n')
 files=[P/'relkit/causal_l185.py',P/'_run_l185.py',P.parent/'docs/plans/2026-10-02-lesson-185-design.md',*sorted(E.glob('seed-*/*.parquet')),E/'report.json',E/'environment.json']
 (E/'run-manifest.json').write_text(json.dumps({str(f.relative_to(P.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2)+'\n')
 print(json.dumps(report['summary'],indent=2))
