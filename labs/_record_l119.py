"""Freeze the delivered source/evidence inventory without asserting learner mastery."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
summary=json.loads((P/'evidence/l119/summary.json').read_text());assert summary['status']=='COMPLETE'
checks={}
for name in ['check','source_check','audit','verify','execution','delivery']:
 p=P/f'_{name}_l119_results.json';assert json.loads(p.read_text())['status']=='PASS';checks[p.name]='PASS'
files=sorted(P.glob('_*l119*.py'))+[P/'relkit/synthesis_l119.py',P/'relkit/rdl_l117.py',P/'_run_l117.py',P/'requirements-l117-runtime.txt',R/'modal/l119_repro.py',P/'_budget_l119.json',P/'l119-reproduction.md',P/'l119-writing-template.md']
files += [p for p in sorted((P/'evidence/l119').rglob('*')) if p.is_file()]
files += [R/'lessons/content/0119-year-3-synthesis.md',R/'assets/l119-lesson.js',R/'assets/representation-collision-viz.js']
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest=dict(status='PASS',files=hashes,checks=checks,upstream='sources/l117/manifest.json',fresh_evidence='evidence/l119/summary.json',source_reuse='Unchanged pinned L117 model/trainer; no L117 trained artifacts reused')
(P/'_sources_l119.json').write_text(json.dumps(manifest,indent=2))
path=P/'reproductions/execution_evidence.json';ledger=json.loads(path.read_text())
ledger['l119']=dict(status='COMPLETE_SELECTED_RELEASE_REPLAY',summary='labs/evidence/l119/summary.json',protocol='labs/l119-reproduction.md',
    checks=checks,validation_mae=summary['metrics']['val'],test_mae=summary['metrics']['test'],fresh_seeds=5,
    historical_identity='NOT_ESTABLISHED',whole_paper='NOT_ESTABLISHED',matched_tabular_comparison='NOT_RUN',
    live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE')
path.write_text(json.dumps(ledger,indent=2)+'\n');print('Recorded L119 source inventory and execution evidence')
