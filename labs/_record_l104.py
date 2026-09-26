"""Record final artifact identity only after complete scientific and delivery checks."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;S='0104-information-leakage-in-time'
if __name__=='__main__':
 analysis=json.loads((P/'_analysis_l104_results.json').read_text())
 assert analysis['status']=='COMPLETE' and analysis['included_seeds']==list(range(10))
 gates={}
 for name in ['check','causality','resume','witness','prepare','execution','delivery']:
  report=json.loads((P/f'_{name}_l104_results.json').read_text());assert report['status']=='PASS';gates[name]='PASS'
 budget_path=P/'_budget_l104.json';budget=json.loads(budget_path.read_text())
 seconds=sum(r['elapsed_seconds'] for r in analysis['records'])+budget['pilot_seconds']
 budget.update({'completed_seeds':list(range(10)),'completed_seed_calls':10,'paid_pilots':1,'paid_retries':0,'observed_successful_call_seconds_including_pilot':seconds,'observed_call_resource_estimate_usd':seconds*budget['total_resource_ceiling_usd_per_second'],'cost_boundary':'Runtime-based conservative resource estimate, not a provider invoice; excludes startup/idle/build overhead covered by the declared reserve. No L103 training costs assigned to L104.'})
 budget_path.write_text(json.dumps(budget,indent=2)+'\n')
 paths=[R/'lessons'/f'{S}.html',R/'lessons/content'/f'{S}.md',R/'reference/temporal-leakage-audit.html',P/f'{S}.ipynb',P/'solutions'/f'{S}.ipynb',P/'html'/f'{S}.html',P/'relkit/leakage_l104.py',P/'relkit/tgat_l103.py',P/'_run_l104.py',P/'_inputs_l104.json',P/'_analysis_l104_results.json',P/'_budget_l104.json',R/'modal/l104_replay.py',R/'.github/workflows/pages.yml']
 paths += [P/'l104-reproduction.md',P/'_build_l104.py',P/'_figures_l104.py',P/'_analyze_l104.py',R/'assets/l104-lesson.js',R/'assets/availability-audit-viz.js',R/'assets/temporal-leakage.css'] + sorted((P/'figures/l104').glob('*'))
 hashes={str(path.relative_to(R)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
 result={'status':'COMPLETE','gates':gates,'artifact_sha256':hashes,'fresh_evaluation_seeds':10,'replayed_positive_events':analysis['replayed_positive_events'],'max_probability_error':analysis['max_prediction_error'],'training':'REUSED_L103','full_training_recipe':'VISIBLE_AND_RUNNABLE_NOT_EXECUTED_FOR_L104','historical_identity':'INCOMPARABLE','full_paper':'NOT_ESTABLISHED','live_colab':'NOT_CHECKED','deployment':'NOT_CHECKED','learner':'PENDING_WRITTEN_DEFENSE'}
 (P/'_provenance_l104_results.json').write_text(json.dumps(result,indent=2)+'\n')
 a=analysis['summary'];text=f'''# Lesson 104 delivery

Prepared 2026-09-26. Learner status: **PENDING_WRITTEN_DEFENSE**.

- Fresh complete evaluation of all ten full-data L103 TGAT checkpoints: {analysis['replayed_positive_events']:,} positive event questions, each paired with a negative; maximum archived-prediction error {analysis['max_prediction_error']:.3g}.
- Independent reconstruction: {analysis['independent_paired_metric_checks']} paired AP checks, plus every released batch AP; all pass.
- All-test pooled AP inflation: same-time {a['all']['inclusive']['mean_delta_pp']:+.4f} pp; lookahead {a['all']['lookahead']['mean_delta_pp']:+.4f} pp. New-node: {a['new']['inclusive']['mean_delta_pp']:+.4f} pp and {a['new']['lookahead']['mean_delta_pp']:+.4f} pp. Sample SDs are in the lesson and ledger.
- Notebook: 26 executed code cells, three student TODOs, visible complete model/trainer and separate fresh-training gate. Four portable figures; three temporal mechanism diagrams and the paired result plot.
- Boundary, mutation, future-feature invariance, witness instrumentation and resume checks pass. Desktop/mobile, keyboard, closed-details print, no-JS, deterministic rebuild and copied Pages navigation pass.
- Conservative resource-time estimate including pilot: USD{budget['observed_call_resource_estimate_usd']:.2f}, plus unitemized startup/idle/build overhead within the USD10 plan. No paid retraining or retries in L104.

## Interpretation

The replay regenerates inference from previously trained checkpoints. The inference interventions are new course experiments, not published leakage results. Published AP targets are numerical checks with the original L103 protocol/population deviations retained. Full-paper parity remains NOT_ESTABLISHED. Live Colab and deployment remain NOT_CHECKED.

## Canonical entry points

- `lessons/content/{S}.md` and generated HTML.
- `labs/_build_l104.py`: notebook pair and reference; `labs/_figures_l104.py`: diagrams and chart.
- `labs/relkit/leakage_l104.py`: audited eligibility, metrics and inference; unchanged `tgat_l103.py`: complete model/trainer.
- `labs/l104-reproduction.md`: local/Modal replay, raw training recipe, isolated independent rerun and deviations.
- `labs/_analysis_l104_results.json`, `labs/_provenance_l104_results.json`: measured evidence and final file identities.

The input-extension gate briefly rejected a local mismatch while final L103 artifacts became available. Subsequent independent SHA checks matched all nine existing pins, then the ten-seed freeze passed. No old pin was replaced and no rejected snapshot entered an experiment.
'''
 (R/'plan/lesson-104-delivery.md').write_text(text)
 print(json.dumps({'status':'COMPLETE','events':analysis['replayed_positive_events'],'estimated_resource_usd':budget['observed_call_resource_estimate_usd']},indent=2))
