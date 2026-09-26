"""Record completed verified artifacts without claiming learner mastery or publication."""
import hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
reports={key:json.loads((P/name).read_text()) for key,name in {'analysis':'_analysis_l106_results.json','audit':'_audit_l106_results.json','checks':'_check_l106_results.json','provenance':'_provenance_l106_results.json','mutation':'_mutation_l106_results.json','execution':'_execution_l106_results.json','delivery':'_delivery_l106_results.json'}.items()}
assert all(x['status']=='PASS' for x in reports.values())
env={'python':sys.version,'platform':platform.platform(),'packages':dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions() if d.metadata['Name']))}
(P/'evidence/l106/environment.json').write_text(json.dumps(env,indent=2)+'\n')
r=reports['analysis'];summaries={s:x['summary'] for s,x in r['conditions'].items()}
entry={'lesson':106,'file':'labs/_analysis_l106_results.json','command':'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python labs/_run_l106.py --raw labs/data/l102/wikipedia.csv','scope':'Complete Wikipedia EdgeBank released-code replay: two memories, three samplers, five original iterations','status':'PASS','numeric_results':summaries,'source_prediction_parity':'PASS every batch','historical_identity':'NOT_ESTABLISHED','full_paper':'NOT_RUN','cost_usd':0,'learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED'}
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['runs']=[x for x in ledger['runs'] if x.get('lesson')!=106];ledger['runs'].append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n')
close=sum(v['numeric_status']=='CLOSE' for x in summaries.values() for v in x.values())
p=R/'NOTES.md';s=p.read_text();marker='## Lesson 106 prepared · 2026-09-26'
if marker not in s:s+='\n\n'+marker+f'\n\n- Full selected Wikipedia EdgeBank source replay completed: 157,474 input events, 23,621 test events, six conditions × five original iterations; {close}/6 numerically CLOSE under predeclared tolerance. All batch predictions match pinned source.\n- Standalone student/executed solution notebooks, visible implementation, primary-source audit, original candidates and predictions, reference and interactive lesson delivered. See `labs/l106-reproduction.md` and machine-readable checks.\n- Preserve source deviations: quantile window, post-exclusion source replacement/collisions, node withholding, batch mean, and global/instance RNG behavior. Full-paper/historical identity not established. No learner completion inferred: PENDING_WRITTEN_DEFENSE. Live Colab/deployment NOT_CHECKED.\n'
p.write_text(s)
p=R/'thesis-dossier.md';s=p.read_text();marker='**BAR · L106 (2026-09-26):**'
if marker not in s:s+='\n- '+marker+' A full Wikipedia EdgeBank replay tests the evaluation before crediting learned relational structure. Candidate distribution changes scores with the model fixed; original-code parity and selected target closeness do not demonstrate general relational-model superiority, operational forecasting quality or full-paper reproduction.\n'
p.write_text(s)
text=f'''# Lesson 106 delivery · 2026-09-26

User-approved scope: complete Wikipedia EdgeBank evaluation, two memories × three samplers × five original iterations. Local CPU, $0 cloud cost.

- Raw input: {r['events']:,} events; released history {r['history']:,}; test {r['test']:,}; 922 held-out nodes. All original source hashes pinned.
- Numeric comparison: {close}/6 conditions CLOSE by predeclared AP/AUROC tolerance. Complete results and deviations are in the reproduction contract and `_analysis_l106_results.json`.
- Every original-model prediction and independent sklearn metric checked across all batches/runs. Original-candidate generation executed freshly by author runner; notebook independently rehashes raw data and recomputes full predictions from authenticated embedded source candidates.
- Three live student tasks; five meaningful mutants rejected. Notebook is standalone and embeds the optional full original sampler regeneration path.
- Browser desktop/mobile, native controls/reset, prediction/teachback, print/no-JS, figure bounds, portable notebook figures, executed source hash, deterministic rebuild and copied Pages navigation checks passed.
- Closest prior visuals: L105 event/snapshot representation views. L106 instead exposes membership scoring and the exact cross-class pair credits underlying AUROC, while holding positives fixed in a candidate intervention.
- Learner PENDING_WRITTEN_DEFENSE. Historical/full-paper identity unestablished; other datasets and neural baselines NOT_RUN. Live Colab and deployment NOT_CHECKED.
'''
(R/'plan/lesson-106-delivery.md').write_text(text)
print('Recorded verified lesson 106 delivery')
