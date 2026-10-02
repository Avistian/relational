"""Executable reproduction admission; stops before paid work on the observed gate.

The complete upstream trainer is archived, but its post-gate cloud integration is
NOT_VALIDATED. This entry point must never interpret a replay as a fresh GNN fit.
"""
import json,sys
from pathlib import Path
from _audit_l181 import audit181
from relkit.autocomplete_l181 import visible_columns,baseline_predictions,keyed_scores
P=Path(__file__).resolve().parent;E=P/'evidence/l181'
report=audit181(E/'packet',json.loads((E/'input-manifest.json').read_text()),visible_columns,baseline_predictions,keyed_scores)
print(json.dumps(dict(admission='BLOCKED',baseline=report['baseline_status'],gnn=report['gnn_status'],failed_components=report['gradient_failures'],cloud_usd=0,post_gate_cloud_training='NOT_VALIDATED'),indent=2))
if report['gradient_failures']:raise SystemExit(2)
raise SystemExit('STOP: changed prerequisites require protocol review; no validated post-gate cloud operator')
