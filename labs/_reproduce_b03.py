"""Fail-closed entry point for the approved historical benchmark.
An archived runner is preserved, but cannot run without the absent source loader.
"""
import argparse,json
from pathlib import Path
from _audit_b03 import audit_sources
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--sources',type=Path,default=Path(__file__).resolve().parent/'sources/b03');args=parser.parse_args()
    result=audit_sources(args.sources);print(json.dumps(result,indent=2))
    if args.run:raise SystemExit('INCOMPLETE_SOURCE_PROTOCOL: no model dispatched. Recover and authenticate the original loader, task/split/config/checkpoint mapping and reference before implementing benchmark execution. See b03-reproduction.md.')
