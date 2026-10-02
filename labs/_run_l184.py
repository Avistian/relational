"""Executable admission check for GelGT v2 Table2 driver-position.

No paid process is created when a scientific prerequisite fails. This release
cannot pass this gate: changing it is a new declared reproduction protocol.
"""
import argparse,json
from pathlib import Path
from _audit_l184 import audit184
from relkit.gelgt_l184 import gaussian_features

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preset',choices=['smoke','closer','paper'],default='paper');args=parser.parse_args()
    p=Path(__file__).resolve().parent;e=p/'evidence/l184'
    report=audit184(e/'packet',json.loads((e/'input-manifest.json').read_text()),gaussian_features)
    print(json.dumps({'preset':args.preset,'status':report['selected_experiment'],'dispatch':False,'reason':'Original sampler overwrites query cutoffs; historical configuration and seeds unresolved','cloud_usd':0},indent=2))
    return 2
if __name__=='__main__':raise SystemExit(main())
