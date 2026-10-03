"""Original Table2 preflight. This is a source gate, not a complete benchmark runner."""
import argparse,json
from pathlib import Path
from _source_b07 import audit_sources
p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
r=audit_sources(Path(__file__).resolve().parent/'sources/b07');print(json.dumps(r,indent=2))
if a.run:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: no dispatch; original Table2 inputs/evaluator not authenticated')
