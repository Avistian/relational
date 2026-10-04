"""Seal final immutable B19 deliverables; verify with --check.

The running budget ledger and Pages receipt are deliberately excluded: they
change when a validation command completes. They report execution, not inputs.
"""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;OUT=P/'evidence/b19/artifact-seal.json'
def files():
    paths=[]
    for root in [P/'sources/b19',P/'figures/b19',P/'evidence/b19']:
        paths.extend(p for p in root.rglob('*') if p.is_file() and p!=OUT and p.name!='local-budget.json' and '__pycache__' not in p.parts)
    paths.extend(P.glob('_*b19*.py'));paths.extend(p for p in P.glob('_*b19*.json') if p.name!='_pages_b19_results.json')
    paths.extend([P/'b19-reproduction.md',P/'b19-preregistration.md',R/'docs/plans/2026-10-04-b19-design.md',P/'b19-benchmark-evidence.ipynb',P/'solutions/b19-benchmark-evidence.ipynb',P/'html/b19-benchmark-evidence.html',P/'relkit/evidence_b19.py',R/'lessons/b19-benchmark-evidence.html',R/'lessons/content/b19-benchmark-evidence.md',R/'reference/b19-benchmark-evidence.html'])
    paths.extend(R/'assets'/n for n in ['b19-evidence.js','benchmark-evidence.js','benchmark-evidence.css'])
    return sorted(set(paths))
def seal():return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files()}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',action='store_true');a=parser.parse_args();actual=seal()
    if a.check:assert actual==json.loads(OUT.read_text())['sha256'],'ARTIFACT_HASH_MISMATCH'
    else:OUT.write_text(json.dumps(dict(sha256=actual,scope='Immutable B19 artifacts; excludes mutable budget and Pages receipt'),indent=2)+'\n')
    print('B19 immutable artifact seal:',len(actual),'files', 'verified' if a.check else 'written')
