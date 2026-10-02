"""Executable reproduction admission. No paid or unvalidated post-gate dispatch.

Recompute every prerequisite from pinned evidence. The approved L180 scope
ends here; changing a receipt cannot authorize a larger or repaired experiment.
"""
import json
from pathlib import Path
from _audit_l180 import audit180
from relkit.checkpoint_l180 import temporal_counts,full_run_cost,checkpoint_decision
P=Path(__file__).resolve().parent;E=P/'evidence/l180'
r=audit180(E/'packet',json.loads((E/'input-manifest.json').read_text()),temporal_counts,full_run_cost,checkpoint_decision)
print(json.dumps(dict(reproduction='NOT_RUN',decision=r['decision'],cloud_usd=0,reason='Approved scope is audit/replay only; original trainer is archived in sources/l180/upstream/rt/main.py'),indent=2))
raise SystemExit(2 if r['decision']['admission']=='BLOCKED' else 3)
