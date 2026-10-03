"""Fail-closed original-paper preflight. No current-checkpoint substitution."""
import json,sys
from pathlib import Path
from _source_b05 import source_gate
if __name__=='__main__':
    r=source_gate(Path(__file__).resolve().parent/'sources/b05');print(json.dumps(r,indent=2))
    if '--run' in sys.argv:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: '+ '; '.join(r['missing']))
