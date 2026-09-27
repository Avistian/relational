"""Freeze the delivered source/evidence inventory without asserting learner mastery."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
summary=json.loads((P/'evidence/l120/summary.json').read_text());assert summary['status']=='COMPLETE'
checks={}
for name in ['check','source_check','audit','verify','execution','delivery']:
 p=P/f'_{name}_l120_results.json';assert json.loads(p.read_text())['status']=='PASS';checks[p.name]='PASS'
files=sorted(P.glob('_*l120*.py'))+[P/'relkit/exam_l120.py',P/'relkit/rdl_l117.py',P/'_run_l117.py',P/'requirements-l117-runtime.txt',R/'modal/l120_repro.py',P/'_budget_l120.json',P/'l120-reproduction.md',P/'l120-submission.md']
files += [p for p in sorted((P/'evidence/l120').rglob('*')) if p.is_file()]
files += [R/'lessons/content/0120-year-3-exit-exam.md',R/'assets/l120-lesson.js']
files += [R/'lessons/0120-year-3-exit-exam.html', R/'reference/year-3-exit-exam.html', P/'0120-year-3-exit-exam.ipynb', P/'solutions/0120-year-3-exit-exam.ipynb']
files += sorted((P/'figures/l120').glob('*'))
hashes={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest=dict(status='PASS',files=hashes,checks=checks,upstream='sources/l117/manifest.json',fresh_evidence='evidence/l120/summary.json',source_reuse='Unchanged pinned L117 model/trainer; no L117 trained artifacts reused')
(P/'_sources_l120.json').write_text(json.dumps(manifest,indent=2))
path=P/'reproductions/execution_evidence.json';ledger=json.loads(path.read_text())
ledger['l120']=dict(status='COMPLETE_SELECTED_RELEASE_REPLAY',summary='labs/evidence/l120/summary.json',protocol='labs/l120-reproduction.md',
    checks=checks,validation_mae=summary['metrics']['val'],test_mae=summary['metrics']['test'],fresh_seeds=5,
    historical_identity='NOT_ESTABLISHED',whole_paper='NOT_ESTABLISHED',matched_tabular_comparison='NOT_RUN',
    live_colab='NOT_CHECKED',deployment='NOT_CHECKED',learner_status='PENDING_WRITTEN_DEFENSE')
path.write_text(json.dumps(ledger,indent=2)+'\n');print('Recorded L120 source inventory and execution evidence')
