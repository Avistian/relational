"""Reconcile measured delivery evidence without promoting replay to fresh training."""
import hashlib,json
from pathlib import Path
from _replay_l154 import replay
P=Path(__file__).resolve().parent
manifest=json.loads((P/'evidence/l154/input-manifest.json').read_text())
report=replay(P,manifest)
assert report==json.loads((P/'evidence/l154/report.json').read_text())
checks={name:json.loads((P/f'_{name}_l154_results.json').read_text()) for name in ['audit','execution','delivery','checkout']}
assert all(c['status']=='PASS' for c in checks.values())
assert report['verdict']['complete_tasks']==2 and report['verdict']['matched_baseline_tasks']==0
assert report['entries'][2]['mean'] is None and report['pilot']['split']=='val'
assert report['additional_cloud_spend_usd']==0
result=dict(status='PASS',experiment=report['experiment'],prediction_rows=report['total_prediction_rows'],
            frozen_inputs=report['input_files_verified'],validation_selections=report['validation_selection_checks'],
            independent_maximum_error=checks['audit']['maximum_metric_error'],
            inherited_prediction_hashes=checks['audit']['inherited_prediction_hashes'],
            code_cells=checks['execution']['code_cells'],browser_states=checks['delivery']['interactive_states'],
            copied_pages_links=checks['delivery']['copied_pages_links'],clean_index_pages='PASS',
            verdict=report['verdict'],full_recommendation='INCOMPLETE',fresh_training='NOT_RUN',
            fresh_manual_fe='NOT_RUN',fresh_tree_baselines='NOT_RUN',additional_cloud_spend_usd=0,
            live_colab='NOT_CHECKED',deployment='NOT_CHECKED')
(P/'_verify_l154_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
