"""Seal immutable B18a artifact bytes; mutable local budget and Pages result excluded."""
import argparse,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent;E=P/'evidence/b18a';target=E/'artifact-manifest.json'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def files():
    paths=[]
    for root in [E,P/'sources/b18a',P/'figures/b18a']:
        paths.extend(p for p in root.iterdir() if p.is_file() and p not in [target,E/'local-budget.json'])
    paths.extend(P.glob('_*b18a*.py'));paths.extend(P.glob('_*b18a*.json'))
    paths.extend([P/'relkit/state_b18a.py',P/'b18a-reproduction.md',P/'b18a-context-state.ipynb',P/'solutions/b18a-context-state.ipynb',P/'html/b18a-context-state.html',R/'lessons/b18a-context-state.html',R/'lessons/content/b18a-context-state.md',R/'reference/b18a-context-state.html',R/'assets/context-state.css',R/'assets/context-state.js',R/'docs/plans/2026-10-04-b18a-design.md',R/'learning-records/0174-context-state-prepared.md'])
    return sorted(set(p for p in paths if p.name!='_pages_b18a_results.json'))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
    if args.check:
        m=json.loads(target.read_text())
        for name,sha in m['sha256'].items():assert digest(R/name)==sha,'ARTIFACT_HASH_MISMATCH: '+name
        print('PASS',len(m['sha256']),'sealed artifacts')
    else:
        m=dict(experiment='B18A-CONTEXT-STATE',source_git='002f83bdb5b1776ca69b7916f993a82344a3aef7',checkpoint_revision='d38ed9517764698a0b0064a7a8cb4197016349a3',sha256={str(p.relative_to(R)):digest(p) for p in files()})
        target.write_text(json.dumps(m,indent=2)+'\n');print('Sealed',len(m['sha256']),'artifacts')
