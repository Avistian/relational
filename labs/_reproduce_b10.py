"""Fail-closed selected paper operator; no mutable PASS receipt can unlock it."""
import argparse,json
from pathlib import Path
from _audit_b10 import audit
P=Path(__file__).resolve().parent
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');p.add_argument('--checkpoints',type=Path);p.add_argument('--out',type=Path);a=p.parse_args()
    report=audit() # Recompute from fixed authenticated inputs, never trust edited status.
    if report['paper_gate']!='PASS':
        if a.run:raise SystemExit('BLOCKED_TEMPORAL_AUDIT: inference NOT_RUN; frozen context audit fails')
        print('Preflight complete: INCOMPLETE_TEMPORAL_GATE; inference NOT_RUN')
    else:
        if not a.run:raise SystemExit('Preflight passed; execution not requested')
        if not a.checkpoints or not a.out:raise SystemExit('--checkpoints and immutable --out required')
        from _infer_b10 import run
        run(P/'sources/b10/upstream',a.checkpoints,P/'evidence/l175/audit-3',a.out)
