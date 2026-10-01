"""Freeze completed primary predictions, selection traces and provenance once."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l155'
paths=[E/'effort-log.json',E/'label-audit.json',E/'fe/sql-audit.json',E/'fe/preparation.json',E/'summary.json',E/'fe/summary.json']
paths += [E/f'{prefix}paper/seed-{seed}/{name}' for prefix in ['', 'fe/'] for seed in range(5) for name in ['predictions.npz','result.json']]
for p in [P/'_sources_l155.json',P/'_upstream_l155.json',P/'_budget_l155.json']:
    dest=E/('frozen-'+p.name);dest.write_bytes(p.read_bytes());paths.append(dest)
m=dict(scope='Primary full-data computational replay. Frozen budget is primary-only; live ledger includes subsequent validation.',files={str(p.relative_to(E)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
f=E/'input-manifest.json'
if f.exists():assert json.loads(f.read_text())==m,'Existing primary manifest differs'
else:f.write_text(json.dumps(m,indent=2)+'\n')
print('Pinned',len(paths),'inputs')
