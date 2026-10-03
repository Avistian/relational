"""Original Table12 preflight. Refuses execution while source identities are absent."""
import argparse,json
from pathlib import Path
from _source_b06 import source_gate
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');a=parser.parse_args()
    r=source_gate(Path(__file__).parent/'sources/b06');print(json.dumps(r,indent=2))
    if a.run:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: original six-arm Table12 pipeline unavailable; no run dispatched')
