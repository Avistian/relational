"""Refresh the compact L102 execution ledger only after checks and full runs complete."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
paper=json.loads((P/'_paper_l102_results.json').read_text());assert paper['status']=='COMPLETE'
checks={}
for name in ['check','source_check','state_check','audit','resume_check','gpu_resume','metrics_check','execution','delivery']:
 path=P/f'_{name}_l102_results.json';value=json.loads(path.read_text());assert value['status']=='PASS',(name,value);checks[name]=value
replay=json.loads((P/'_source_replay_l102_results.json').read_text());assert replay['status']=='PASS'
report={'status':'PASS','selected_experiment':'COMPLETE','complete_seeds':paper['complete_seeds'],'total_epochs':paper['total_epochs'],
 'summary':paper['summary'],'primary_runtime':paper['identity'],'checks':checks,'original_source_full_evaluation':replay,
 'historical_identity':'INCOMPARABLE','full_paper_parity':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE',
 'live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED',
 'commands':['OMP_NUM_THREADS=1 .venv/bin/python labs/_check_l102.py','OMP_NUM_THREADS=1 .venv/bin/python labs/_source_check_l102.py','OMP_NUM_THREADS=1 .venv/bin/python labs/_audit_l102.py','OMP_NUM_THREADS=1 .venv/bin/python labs/_state_check_l102.py','OMP_NUM_THREADS=1 .venv/bin/python labs/_verify_l102.py --preset smoke --seeds 0,1,2','.venv/bin/modal run --detach modal/l102_paper_repro.py::main --preset paper','.venv/bin/modal run modal/l102_paper_repro.py::source_check','.venv/bin/python labs/_collect_l102.py labs/results/l102/gpu','.venv/bin/python labs/_execute_l102.py','.venv/bin/python labs/_delivery_l102.py'],
 'rejected_evidence':['Partial CPU pilots excluded from complete GPU means','Three short teaching fits are not Table 2 reproduction','Source parity is not score parity','Two close means are not full-paper or historical identity','Notebook execution does not establish learner mastery or live Colab'],
 'artifact_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [P/'relkit/tgn_l102.py',P/'_paper_l102_results.json',P/'_sources_l102.json',*sorted((P/'evidence/l102').glob('*'))] if p.is_file()}}
(P/'_verify_l102_results.json').write_text(json.dumps(report,indent=2))
ledger=P/'reproductions/execution_evidence.json';root=json.loads(ledger.read_text());root['lesson_102']={k:v for k,v in report.items() if k!='artifact_sha256'};ledger.write_text(json.dumps(root,indent=2)+'\n')
summary=paper['summary'];a=summary['all'];n=summary['new']
notes=f'''\n## Lesson102 created · Temporal Graph Networks · 2026-09-22\n\nApproved combined lesson plus complete Wikipedia TGN-attn replay. Full ten-run T4 experiment: {paper['total_epochs']} epochs, all-event AP {a['mean_ap_percent']:.4f}% (sample SD {a['sample_sd_pp']:.4f}pp), new-node AP {n['mean_ap_percent']:.4f}% (SD {n['sample_sd_pp']:.4f}pp). Numerical verdicts {a['numerical_verdict']}/{n['numerical_verdict']}; original-source full seed-0 evaluation and independent all-run metric reconstruction pass. Complete visible model/trainer, three live tasks, 720 temporal oracle cases, preprocessing/split parity, memory checkpoint counterexample, portable architecture and interactive event trace. Explicit independent seeds, modern runtime and preserved release pending-message checkpoint behavior documented. Partial CPU runs excluded; full-paper parity NOT_ESTABLISHED, historical identity INCOMPARABLE. No learner mastery or deployment inferred.\n'''
p=R/'NOTES.md';s=p.read_text();assert '## Lesson102 created' not in s;p.write_text(s+notes)
p=R/'thesis-dossier.md';s=p.read_text();assert '**BAR · L102' not in s;p.write_text(s+f'''\n- **BAR · L102 (2026-09-22):** Complete ten-run TGN-attn Wikipedia replay reaches {a['mean_ap_percent']:.3f}% all-event and {n['mean_ap_percent']:.3f}% new-node batch-mean AP. Shared temporal state must be audited independently of weights: omitting pending messages changes predictions. This single-dataset sampled-candidate experiment does not establish relational-database superiority, full-catalog retrieval quality or full-paper reproduction.\n''')
print('L102 evidence, notes and thesis ledger recorded')
