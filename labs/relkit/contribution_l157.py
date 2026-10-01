"""Executable learner contracts for a reviewable scientific contribution."""
import hashlib,math,statistics
from pathlib import Path

def verify_manifest(root,files):
    """Verify listed bytes under root; hashes do not authenticate their author."""
    root=Path(root).resolve()
    if not files:raise ValueError('Empty evidence manifest')
    for name,digest in files.items():
        rel=Path(name)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe path')
        path=root/rel
        if any(p.is_symlink() for p in [path,*path.parents] if p!=root and root in p.parents):raise ValueError('Symlink evidence')
        if not path.resolve().is_relative_to(root):raise ValueError('Escaped root')
        if not path.is_file():raise FileNotFoundError(name)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Changed evidence: '+name)
    return len(files)

def summarize_runs(rows):
    """Require all ten full runs; incomplete evidence has no pooled result."""
    seen=set();by_lane={'paper':[],'fit_horizon':[]}
    for row in rows:
        lane=row['lane'];seed=row['seed'];key=(lane,seed)
        if lane not in by_lane or type(seed) is not int or seed not in range(5):raise ValueError('Unexpected run')
        if key in seen:raise ValueError('Duplicate run')
        if row['status']!='COMPLETE' or row['epochs']!=10 or row['train_queries']!=7453 or row['val_rows']!=499 or row['test_rows']!=760:raise ValueError('Incomplete run presented as full')
        if not math.isfinite(row['test_mae']) or row['test_mae']<0:raise ValueError('Invalid MAE')
        seen.add(key);by_lane[lane].append(row['test_mae'])
    complete=len(seen)==10
    return dict(status='COMPLETE' if complete else 'INCOMPLETE',fits=len(seen),predictions=len(seen)*1259,
        lanes={k:dict(mean=statistics.mean(v),sd=statistics.stdev(v)) for k,v in by_lane.items()} if complete else {},
        missing=[dict(lane=k,seed=s) for k in by_lane for s in range(5) if (k,s) not in seen])

def review_claim(claim,evidence):
    """Bind each claim to its own evidence; publication needs a verified URL."""
    allowed={'selected_reproduction','historically_leak_free','public_contribution','whole_paper','upstream_bug'}
    if claim not in allowed:raise ValueError('Unknown claim')
    if evidence.get('integrity')!='PASS':return 'REJECTED'
    if claim=='selected_reproduction':return 'SUPPORTED' if evidence.get('reproduction')=='COMPLETE' else 'INCOMPLETE'
    if claim=='public_contribution':return 'NOT_CHECKED' if evidence.get('public_url') else 'PENDING_PUBLICATION'
    if claim=='whole_paper':return 'NOT_RUN'
    # This package has no historical arrival data or current-upstream defect proof.
    return 'NOT_ESTABLISHED'
