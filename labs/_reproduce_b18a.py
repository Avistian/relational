"""Audit source bytes; fail closed for unresolved historical Figure3 identity."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def audit():
    root=P/'sources/b18a'
    for record in json.loads((root/'manifest.json').read_text()):
        if hashlib.sha256((root/record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('SOURCE_HASH_MISMATCH: '+record['path'])
    return json.loads((root/'paper-contract.json').read_text())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['audit','paper'],default='audit');args=ap.parse_args()
    contract=audit();print(json.dumps(contract,indent=2))
    if args.phase=='paper':raise SystemExit('NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE: '+', '.join(contract['unresolved']))
