"""Publish measured local evidence into course ledgers, without claiming learner mastery."""
import json,re
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;r=json.loads((P/'evidence/l110/summary.json').read_text());d=json.loads((P/'_delivery_l110_results.json').read_text());assert r['status']=='COMPLETE' and d['status']=='PASS'
assert json.loads((P/'_replay_l110_results.json').read_text())['status']=='PASS'
assert json.loads((P/'evidence/l110/seed-0/release/seed-0-source-replay.json').read_text())['status']=='PASS'
a=r['summary']['release'];b=r['summary']['clean']
text=f'''\n\n## Lesson 110 prepared · 2026-09-26\n\n- Approved combined scope completed: twenty fresh full-data TGN-attn Wikipedia fits, ten released-protocol seeds and ten paired clean-state seeds. Release AP {a['all']['mean_ap_percent']:.4f}% / {a['new']['mean_ap_percent']:.4f}% (all/new); paper numerical verdicts {a['all']['numerical_verdict']} / {a['new']['numerical_verdict']}. Clean policy AP {b['all']['mean_ap_percent']:.4f}% / {b['new']['mean_ap_percent']:.4f}%; course intervention, not a published target.\n- Complete selected-state restoration, timestamp-group batching and best-validation selection at the cap are combined changes. Fixed-weight tied-feature counterexample, full split/source audit, original-source seed0 replay, independent metrics and fresh-checkpoint CPU replay are distinct checks. Historical identity INCOMPARABLE; full-paper NOT_ESTABLISHED; real ingestion histories absent.\n- Visible standalone model/trainer, three live TODOs, executed solution, four portable figures, reference, browser desktop/mobile/keyboard/no-JS/print and copied Pages checks. Learner PENDING_WRITTEN_DEFENSE. Live Colab and deployment NOT_CHECKED.\n- Completed-call resource estimate including pilot: USD{r['completed_call_resource_usd']:.4f}, plus unitemized overhead under the USD10 plan. Exact commands and limitations: labs/l110-reproduction.md.\n'''
p=R/'NOTES.md';s=p.read_text();marker='\n\n## Lesson 110 prepared ·';s=re.sub(r'\n\n## Lesson 110 prepared ·.*?(?=\n## |\Z)', '', s, flags=re.S) if marker in s else s;p.write_text(s+text)
p=R/'thesis-dossier.md';s=p.read_text();line=f'\n- **L110 · BAR · 2026-09-26:** twenty fresh full-data temporal-GNN fits and a time-travel audit. Release Wikipedia AP {a["all"]["mean_ap_percent"]:.4f}% / {a["new"]["mean_ap_percent"]:.4f}% is numerically {a["all"]["numerical_verdict"]}/{a["new"]["numerical_verdict"]}; strict sampling alone cannot protect a memory path that crosses timestamp ties. Complete checkpoint consistency and known availability remain prerequisites for relational evidence. This is one interaction benchmark, not database superiority or full-paper parity. [Contract](labs/l110-reproduction.md).\n';
if '**L110 · BAR' not in s:p.write_text(s+line)
p=P/'reproductions/execution_evidence.json';x=json.loads(p.read_text());x['lesson_110']={'status':'COMPLETE_SELECTED_EXPERIMENT','scope':'Fresh TGN-attn Wikipedia ten-seed release reproduction and ten-seed clean-state course intervention','paper_summary':a,'course_summary':b,'epochs':r['epochs'],'independent_event_evaluations':r['event_evaluations'],'historical_identity':'INCOMPARABLE','full_paper_reproduction':'NOT_ESTABLISHED','learner_status':'PENDING_WRITTEN_DEFENSE','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','budget':{'limit_usd':10,'completed_call_resource_estimate_usd':r['completed_call_resource_usd'],'other_overhead':'NOT_ITEMIZED'},'protocol':'labs/l110-reproduction.md','evidence':'labs/evidence/l110/summary.json'};p.write_text(json.dumps(x,indent=2)+'\n')
p=P/'_budget_l110.json';x=json.loads(p.read_text());x['status']='COMPLETE';x['completed_call_resource_estimate_usd']=r['completed_call_resource_usd'];x['unitemized_overhead']='Startup/build/storage; resource estimate is not an invoice';p.write_text(json.dumps(x,indent=2)+'\n')
p=R/'plan/year-3.md';s=p.read_text().replace('- **Approved scope** — [lesson](../lessons/0110-temporal-gnn-checkpoint.html)', '- **Delivered** — [lesson](../lessons/0110-temporal-gnn-checkpoint.html)');p.write_text(s)
p=P/'l110-reproduction.md';s=p.read_text();start='<!-- L110 measured evidence -->';end='<!-- /L110 measured evidence -->'
block=f'''{start}

## Completed fresh execution

Twenty fits completed: release {r['epochs']['release']} epochs across ten seeds, clean {r['epochs']['clean']} epochs across ten seeds. Complete selected release experiment COMPLETE; full-paper NOT_ESTABLISHED.

| Arm | All-event batch AP | New-node batch AP | Comparison |
|---|---:|---:|---|
| Release | {a['all']['mean_ap_percent']:.4f}% ± {a['all']['sample_sd_pp']:.4f} pp | {a['new']['mean_ap_percent']:.4f}% ± {a['new']['sample_sd_pp']:.4f} pp | {a['all']['numerical_verdict']} / {a['new']['numerical_verdict']} |
| Clean | {b['all']['mean_ap_percent']:.4f}% ± {b['all']['sample_sd_pp']:.4f} pp | {b['new']['mean_ap_percent']:.4f}% ± {b['new']['sample_sd_pp']:.4f} pp | COURSE_INTERVENTION |

Uncertainty is sample seed SD. Independent AP reconstruction covers {r['event_evaluations']:,} positive questions and their paired negatives, with maximum AP error {r['max_independent_ap_error']:.3g}. All forty population/checkpoint replay branches pass in the distinct CPU runtime. Original-source full release seed0 AP agrees exactly in both populations. Completed-call resource estimate including pilot: USD{r['completed_call_resource_usd']:.4f}; other overhead is not itemized. Browser and copied-Pages delivery PASS; live Colab/deployment NOT_CHECKED.

{end}'''
if start in s:s=s[:s.index(start)]+block+s[s.index(end)+len(end):]
else:s+='\n'+block+'\n'
p.write_text(s)
print('L110 evidence recorded; learner remains PENDING_WRITTEN_DEFENSE')
