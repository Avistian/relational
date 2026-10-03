"""Original Table7 source preflight; never silently substitute the course protocol."""
import argparse,json
from pathlib import Path
from _source_b07a import audit_sources
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true');a=p.parse_args()
    r=audit_sources(Path(__file__).resolve().parent/'sources/b07a');print(json.dumps(r,indent=2))
    if a.run:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: original Table7 dispatch refused; see missing input list')
