"""Safe B11 entry point: audit sealed evidence; never silently dispatch full fits."""
import argparse,json
from pathlib import Path
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',action='store_true');p.add_argument('--run-full',action='store_true');a=p.parse_args()
    if a.run_full:raise SystemExit('BLOCKED: USD0 B11 scope; RelGT temporal audit FAIL and inherited full-search projection exceeds USD10. See labs/b11-reproduction.md.')
    if a.audit:
        from _audit_b11 import audit
        audit()
    else:print((Path(__file__).parent/'b11-reproduction.md').read_text())
