import modal,json
from pathlib import Path
v=modal.Volume.from_name('b02-tabpack-california');root=Path('labs/evidence/b02/fresh');count=0
for e in v.iterdir('run-audited/california',recursive=True):
 p=Path(e.path)
 if p.name not in ['config.json','report.json','experiments.json','observed_ensemble.npz','online_ensemble_history.json']:continue
 out=root/p;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('wb') as f:
  for b in v.read_file(e.path):f.write(b)
 count+=1
print('Downloaded',count,'audit files')
for name in ['audited-receipt.json','full-receipt.json','pilot-receipt.json','pilot2-receipt.json','pilot3-receipt.json','audited.log','full.log']:
 with (root/name).open('wb') as f:
  for b in v.read_file(name):f.write(b)
