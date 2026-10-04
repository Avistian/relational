"""Seal final immutable B17 deliverables; verify with --check.

The running budget ledger and Pages receipt are deliberately excluded: they
change when a validation command completes. They report execution, not inputs.
"""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;OUT=P/'evidence/b17/artifact-seal.json'
def files():
    paths=[]
    for root in [P/'sources/b17',P/'figures/b17',P/'evidence/b17']:
        paths.extend(p for p in root.rglob('*') if p.is_file() and p!=OUT and p.name!='local-budget.json' and '__pycache__' not in p.parts)
    paths.extend(P.glob('_*b17*.py'));paths.extend(p for p in P.glob('_*b17*.json') if p.name!='_pages_b17_results.json')
    paths.extend([P/'b17-reproduction.md',P/'b17-reusable-representations.ipynb',P/'solutions/b17-reusable-representations.ipynb',P/'html/b17-reusable-representations.html',P/'relkit/representations_b17.py',R/'lessons/b17-reusable-representations.html',R/'lessons/content/b17-reusable-representations.md',R/'reference/b17-reusable-representations.html'])
    paths.extend(R/'assets'/n for n in ['b17-evidence.js','representation-boundary.js','representation-boundary.css'])
    return sorted(set(paths))
def seal():return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args();actual=seal()
    if a.check:assert actual==json.loads(OUT.read_text())['sha256'],'ARTIFACT_HASH_MISMATCH'
    else:OUT.write_text(json.dumps(dict(sha256=actual,scope='Immutable B17 artifacts; excludes mutable budget and Pages receipt'),indent=2)+'\n')
    print('B17 immutable artifact seal:',len(actual),'files', 'verified' if a.check else 'written')
