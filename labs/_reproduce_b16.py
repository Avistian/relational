"""Portable source-authenticating preflight for AutoGrable Table 1.

This operator cannot launch an unknown experiment. A recovered, reviewed
experiment driver and protocol are prerequisites for replacing this gate.
"""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','paper'],default='audit');args=parser.parse_args()
m=json.loads((P/'sources/b16/manifest.json').read_text())
for name,digest in m['files'].items():
 f=P/'sources/b16'/name
 if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest:
  raise SystemExit('SOURCE_HASH_MISMATCH: '+name)
s=json.loads((P/'evidence/b16/paper-status.json').read_text());print(json.dumps(s,indent=2))
print('Authenticated',len(m['files']),'source files. No model training or paid dispatch.')
if args.phase=='paper':raise SystemExit('NOT_RUN: Table 1 driver/configuration missing and paper/code discrepancies unresolved')
