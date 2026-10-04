"""Authenticate dated source evidence and refuse the unresolved paper experiment."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent

def audit():
    folder=P/'sources/b17';m=json.loads((folder/'manifest.json').read_text())
    for item in m['sources']:
        if hashlib.sha256((folder/item['file']).read_bytes()).hexdigest()!=item['sha256']:
            raise SystemExit('SOURCE_HASH_MISMATCH: '+item['file'])
    print('Archived source bytes authenticated:',len(m['sources']))
    print('B17-FLEXTAB-MULTI-TABLE5-F1-DNF: INCOMPLETE_SOURCE_PROTOCOL_GATE')
    print('Paper target AUROC0.746; no measured FlexTab result.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','paper'],default='audit');a=p.parse_args();audit()
    if a.phase=='paper':
        raise SystemExit('NOT_RUN: recover original checkpoint, evaluator, population, sampling and complete temporal policy. No benchmark execution authorized by changing a status flag.')
