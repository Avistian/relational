"""Executable Figure 3 preflight; no fabricated dispatch or override."""
import json,sys
from pathlib import Path
from _source_b04 import source_gate
if __name__=='__main__':
    report=source_gate(Path(__file__).resolve().parent/'sources/b04');print(json.dumps(report,indent=2))
    if '--run' in sys.argv:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: recover original Figure 3 identities before implementing benchmark dispatch')
