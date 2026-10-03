"""Fail-closed Figure 9 preflight; not a finished benchmark dispatcher."""
import json,sys
from pathlib import Path
from _source_b04a import source_gate
if __name__=='__main__':
    result=source_gate(Path(__file__).resolve().parent/'sources/b04a')
    print(json.dumps(result,indent=2))
    if '--run' in sys.argv:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: no Figure 9 dispatch authorized by authenticated inputs')
