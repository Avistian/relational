from pathlib import Path
import modal,json,hashlib
E=Path(__file__).resolve().parent/'evidence/l146';v=modal.Volume.from_name('l146-comparison-evidence');rows=[]
for arm in ['gnn','relgt']:
 for seed in [0,1,2]:
  phase=f'fit-{arm}-{seed}';r=json.loads((E/phase/'result.json').read_text());h=hashlib.sha256();size=0
  for chunk in v.read_file(phase+'/selected.pt'):h.update(chunk);size+=len(chunk)
  assert h.hexdigest()==r['checkpoint_sha256'];rows.append(dict(phase=phase,bytes=size,sha256=h.hexdigest()))
(E/'checkpoint-audit.json').write_text(json.dumps(dict(status='PASS',verified_checkpoints=len(rows),volume='l146-comparison-evidence',checkpoints=rows),indent=2))
print('PASS: six remote checkpoint byte streams independently hashed')
