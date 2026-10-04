"""Authenticate primary-source bytes; fail closed for unresolved Animus execution."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def audit():
    folder=P/'sources/b18';receipt=json.loads((folder/'receipt.json').read_text())
    for entry in receipt:
        if hashlib.sha256((folder/entry['file']).read_bytes()).hexdigest()!=entry['sha256']:
            raise SystemExit('SOURCE_HASH_MISMATCH: '+entry['file'])
    if not any(e['file']=='paper.html' and e['http_status']==200 for e in receipt):
        raise SystemExit('PRIMARY_SOURCE_UNAVAILABLE')
    print('Authenticated primary-source and search bytes:',len(receipt))
    print('B18-ANIMUS-RT-RAW-AGG: 12 configurations; NOT_RUN / INCOMPLETE_SOURCE_PROTOCOL_GATE')
    print('Missing original generator/data, model/init, seeds/selection and reconciled RT scores.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','paper'],default='audit');args=p.parse_args();audit()
    if args.phase=='paper':
        raise SystemExit('NOT_RUN: source/protocol unresolved; this gate is not a training launcher.')
