"""Record only the completed, independently audited lesson package."""
import hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
analysis=json.loads((P/'_analysis_l107_results.json').read_text());assert analysis['status']=='COMPLETE'
checks=['initialization','check','source_check','audit','data_check','mutation','history_witness','execution','delivery']
reports={k:json.loads((P/f'_{k}_l107_results.json').read_text()) for k in checks}
assert all(r['status']=='PASS' for r in reports.values())
for variant in ['H','O']:
 source=json.loads((P/f'evidence/l107/sbm/{variant}/source_replay.json').read_text());assert source['status']=='PASS' and source['source_predictions_checked']==10000000
manifest=json.loads((P/'_sources_l107.json').read_text())
for name,h in manifest['files'].items():assert hashlib.sha256((P/'sources/l107/original'/name).read_bytes()).hexdigest()==h
for name,h in analysis['artifacts'].items():assert hashlib.sha256((P/'evidence/l107'/name).read_bytes()).hexdigest()==h
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata['Name']}
env={'scope':'Local authoring and standalone notebook execution; GPU model runtimes recorded per run','python':sys.version,'platform':platform.platform(),'packages':dict(sorted(packages.items()))}
(P/'_environment_l107_results.json').write_text(json.dumps(env,indent=2)+'\n')
budget=json.loads((P/'_budget_l107.json').read_text())
entry={'lesson':107,'file':'labs/_analysis_l107_results.json','commands':['.venv/bin/modal run --detach modal/l107_repro.py::full_sbm','.venv/bin/modal run --detach modal/l107_repro.py::full_wiki'],'scope':'Complete declared Wikipedia three-arm three-seed comparison; full released EvolveGCN H/O SBM training schedules and all-pairs evaluation','status':'PASS','paper_result_status':'CLOSE' if all(x=='CLOSE' for v in analysis['sbm'].values() for x in v['verdict'].values()) else 'NOT_REPRODUCED_WITHIN_TOLERANCE','wiki':analysis['wiki'],'sbm':analysis['sbm'],'independent_metrics':analysis['checks'],'original_model_predictions_checked':20000000,'historical_identity':'NOT_ESTABLISHED','other_datasets_and_paper_LSTM_O':'NOT_RUN','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','budget':budget}
p=P/'reproductions/execution_evidence.json';ledger=json.loads(p.read_text());ledger['runs']=[x for x in ledger['runs'] if x.get('lesson')!=107];ledger['runs'].append(entry);p.write_text(json.dumps(ledger,indent=2)+'\n')
wiki=analysis['wiki'];sbm=analysis['sbm'];scores=', '.join(f"{a} AP {wiki[a]['mean']['ap']:.4f}" for a in ['3600','86400','tgn'])
paper_scores='; '.join(f"{v} MAP {z['test']['map']:.4f} ({z['verdict']['map']}), MRR {z['test']['mrr']:.4f} ({z['verdict']['mrr']})" for v,z in sbm.items())
p=R/'NOTES.md';s=p.read_text();marker='## Lesson 107 prepared · 2026-09-26'
if marker not in s:s+='\n\n'+marker+'\n\n- Approved combined scope: complete Wikipedia comparison (nine fits, 90 epochs) and released EvolveGCN SBM H/O schedules. '+scores+'. SBM: '+paper_scores+'. Full 81,029/23,621/23,621 train/val/test populations; held-out earlier interactions excluded consistently.\n- Every final SBM pair prediction replayed through original model code, 20 million scores; independently reconstructed MAP/MRR. Release O uses GRU rather than paper LSTM; fixed-length resetting removes deterministic dependence on earlier graph content. Preserve stochastic RReLU, future-aware degree schema, worker RNG and runtime deviations.\n- Six portable computation diagrams, two gate controls, three live tasks, seven rejected mutants, standalone executed solution and copied Pages/browser checks. Visible full models/trainers and exact commands in `labs/l107-reproduction.md`. Historical identity unestablished; other paper datasets and LSTM-O NOT_RUN. Learner PENDING_WRITTEN_DEFENSE; live Colab/deployment NOT_CHECKED.\n'
p.write_text(s)
p=R/'thesis-dossier.md';s=p.read_text();marker='**BAR · L107 (2026-09-26):**'
if marker not in s:s+='\n- '+marker+' '+scores+'. These matched-candidate single-dataset system comparisons include architecture, delay and update-budget differences; they do not prove broad temporal or relational superiority. The released O fixed-window recurrence has no deterministic path from earlier graph contents to the final output, showing why a recurrent architecture label is insufficient evidence of learned history. Full selected SBM replay retains source/prose and runtime deviations.\n'
p.write_text(s)
text='''# Lesson 107 delivery · 2026-09-26

Approved combined scope delivered locally. No learner mastery or deployment inferred.

- Complete Wikipedia course experiment: nine fresh T4 fits, ten epochs each, identical questions/candidates; full released populations 81,029 train / 23,621 validation / 23,621 test. All cached raw fields independently re-parsed; all test metrics independently reconstructed.
- Complete released SBM H/O schedules: full 4,870,863-row, 50-snapshot data, released seed and configs, validation selection, all ordered pairs. All 20 million final probabilities checked against original encoder/classifier using saved random states; full MAP/MRR reconstructed independently.
- Numeric summaries and CLOSE/FAIL diagnostics: `labs/_analysis_l107_results.json`. The paper O LSTM, historical worker streams, other datasets, other baselines and ablations are not reproduced.
- Visible source, standalone student/executed solution, three live tasks, seven rejected mutants, six portable diagrams, two independent gate widgets, reference/provenance/protocol and manifest navigation.
- Desktop/mobile, 36 control states, keyboard/reset, prediction/teachback, no-JS/print, copied Pages, code identity and deterministic build checked. Live Colab and deployment NOT_CHECKED.
- Closest prior visual: L105 showed event grouping and availability. L107 adds node-row versus feature-coordinate state lattices, numerical normalization, top-k summary/transposition and a history-dependence witness. These expose operations and counterfactual paths rather than renaming boxes.
- Runtime-based resource estimate is recorded in `_budget_l107.json`, including failed preflight/pilots and original-source checks; it is not a provider invoice.
- Learner status PENDING_WRITTEN_DEFENSE.
'''
text=text.replace('- Numeric summaries', '- '+paper_scores+'. Numeric summaries')
(R/'plan/lesson-107-delivery.md').write_text(text)
(P/'_provenance_l107_results.json').write_text(json.dumps({'status':'PASS','source_files':len(manifest['files']),'authenticated_artifacts':len(analysis['artifacts']),'model_sources_match':True,'original_predictions_checked':20000000,'standalone_notebook':reports['execution']['status'],'historical_identity':'NOT_ESTABLISHED'},indent=2)+'\n')
print('Lesson 107 recorded with completed evidence')
