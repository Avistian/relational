"""Freeze inputs once; subsequent runs reject drift rather than bless new bytes."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l158';E.mkdir(exist_ok=True)
files=set()
for lesson in [154,155]:
    prefix='' if lesson==154 else 'evidence/l155/'
    name=f'evidence/l{lesson}/input-manifest.json';files.add(name)
    files.update(prefix+n for n in json.loads((P/name).read_text())['files'])
for lesson in [156,157]:
    base=f'evidence/l{lesson}/';files.update(base+n for n in ['input-manifest.json','report.json','val-labels.npz','test-labels.npz'])
    for lane in ['paper','fit_horizon']:
        for seed in range(5):
            files.update(f'{base}{lane}/seed-{seed}/{n}' for n in ['predictions.npz','result.json','completed.json','audit-l156.json','checkpoint-audit.json','diagnostics.json'])
files.update(['evidence/l155/report.json','evidence/l154/report.json'])
# Pin implementations consumed by the standalone packet, separately from historical training identities.
files.update(['_replay_l154.py','_report_l155.py','relkit/portfolio_l154.py','relkit/effort_l155.py'])
m=dict(experiment='L158 Year 4 thesis evidence replay',freeze_date='2026-10-01',files={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in sorted(files)})
f=E/'input-manifest.json'
if f.exists():assert json.loads(f.read_text())==m,'Frozen inputs changed; investigate, do not overwrite'
else:f.write_text(json.dumps(m,indent=2)+'\n')
print('Frozen',len(files),'files;',sum((P/n).stat().st_size for n in files),'bytes')
