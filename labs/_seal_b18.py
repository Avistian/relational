"""Seal final immutable B18 deliverables; verify with --check.

The running budget ledger and Pages receipt are deliberately excluded: they
change when a validation command completes. They report execution, not inputs.
"""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;OUT=P/'evidence/b18/artifact-seal.json'
def files():
    paths=[]
    for root in [P/'sources/b18',P/'figures/b18',P/'evidence/b18']:
        paths.extend(p for p in root.rglob('*') if p.is_file() and p!=OUT and p.name!='local-budget.json' and '__pycache__' not in p.parts)
    paths.extend(P.glob('_*b18*.py'));paths.extend(p for p in P.glob('_*b18*.json') if p.name!='_pages_b18_results.json')
    paths.extend([P/'b18-reproduction.md',P/'b18-context-sufficiency.ipynb',P/'solutions/b18-context-sufficiency.ipynb',P/'html/b18-context-sufficiency.html',P/'relkit/context_b18.py',R/'lessons/b18-context-sufficiency.html',R/'lessons/content/b18-context-sufficiency.md',R/'reference/b18-context-sufficiency.html'])
    paths.extend(R/'assets'/n for n in ['b18-evidence.js','context-budget.js','context-budget.css'])
    return sorted(set(paths))
def seal():return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args();actual=seal()
    if a.check:assert actual==json.loads(OUT.read_text())['sha256'],'ARTIFACT_HASH_MISMATCH'
    else:OUT.write_text(json.dumps(dict(sha256=actual,scope='Immutable B18 artifacts; excludes mutable budget and Pages receipt'),indent=2)+'\n')
    print('B18 immutable artifact seal:',len(actual),'files', 'verified' if a.check else 'written')
