"""Read-only paper preflight. Never replace unknown protocol with defaults."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['audit','paper'],default='audit');args=parser.parse_args()
m=json.loads((P/'sources/b15/manifest.json').read_text())
for name,digest in m['files'].items():
 if hashlib.sha256((P/'sources/b15'/name).read_bytes()).hexdigest()!=digest:raise SystemExit('SOURCE_HASH_MISMATCH: '+name)
s=json.loads((P/'evidence/b15/paper-status.json').read_text());print(json.dumps(s,indent=2))
if args.phase=='paper':raise SystemExit('NOT_RUN: missing authenticated Table 5 protocol; no paid dispatch implemented or admitted')
