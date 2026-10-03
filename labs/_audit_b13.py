"""New B13 pairwise-AUROC audit of immutable L200 released-checkpoint evidence."""
import hashlib,io,json,tempfile,zipfile
from pathlib import Path
import numpy as np

def pairwise_auc(y,p):
    y=np.asarray(y);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or y.ndim!=1 or set(y.tolist())!={0,1} or not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid binary predictions')
    delta=p[y==1,None]-p[y==0][None,:]
    return float(np.mean((delta>0)+.5*(delta==0)))

def audit_l200(root):
    root=Path(root);manifest=json.loads((root/'packet-manifest.json').read_text())
    for name,digest in manifest['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed inherited bytes: '+name)
    data=np.load(root/'packet/prepared.npz',allow_pickle=False)
    if tuple(data['test_keys'].shape)!=(702,2) or len(set(map(tuple,data['test_keys'])))!=702:raise ValueError('Missing full query identities')
    if len(data['train_keys'])!=11411 or len(set(map(tuple,data['train_keys'])))!=11411:raise ValueError('Training population changed')
    for seed in range(10):
        derived=int.from_bytes(hashlib.sha256(f'rel-f1-dfs-2:driver-dnf:{seed}'.encode()).digest()[:4],'big')
        expected=np.random.default_rng(derived).choice(11411,512,replace=False)
        if not np.array_equal(expected,data['support'][seed]):raise ValueError('Support draw derivation changed')
    arms=['RDBPFN','RDBPFN_single','TabICLv1.1'];records={};count=0
    for phase in ['pilot-1','full-1']:
        receipt=json.loads((root/phase/'receipt.json').read_text())
        if receipt['input_manifest_sha256']!=manifest['input_manifest_sha256']:raise ValueError('Input identity changed')
        for r in receipt['records']:
            key=(r['arm'],r['seed'])
            if key in records:raise ValueError('Duplicate evaluation')
            p=root/phase/f"{r['arm']}-{r['seed']}.npz"
            if hashlib.sha256(p.read_bytes()).hexdigest()!=r['sha256']:raise ValueError('Prediction receipt mismatch')
            x=np.load(p,allow_pickle=False)
            for a,b in [(x['keys'],data['test_keys']),(x['label'],data['y_test']),(x['support_keys'],data['train_keys'][data['support'][r['seed']]])]:
                if not np.array_equal(a,b):raise ValueError('Key, target, or support mismatch')
            if len(set(map(tuple,x['support_keys'])))!=512 or set(map(tuple,x['keys']))&set(map(tuple,x['support_keys'])):raise ValueError('Support/query overlap')
            score=pairwise_auc(x['label'],x['probability'])
            if abs(score-r['auc'])>1e-12:raise ValueError('Receipt AUROC disagreement')
            records[key]=score;count+=len(x['label'])
    if set(records)!={(a,s) for a in arms for s in range(10)} or count!=21060:raise ValueError('Incomplete grid')
    models={}
    for arm,target in zip(arms,[.7219,.6640,.7176]):
        scores=[records[arm,s] for s in range(10)]
        models[arm]=dict(per_seed=scores,mean=float(np.mean(scores)),sample_sd=float(np.std(scores,ddof=1)),paper_target=target,status='CLOSE' if abs(np.mean(scores)-target)<=.02 else 'OUTSIDE_TOLERANCE')
    diffs=[records['RDBPFN',s]-records['TabICLv1.1',s] for s in range(10)]
    return dict(status='COMPLETE_SAVED_EVIDENCE_REPLAY',origin='L200 fresh released-checkpoint inference; B13 performs no new inference',evaluations=30,predictions=count,models=models,paired_rdbpfn_minus_tabicl=dict(mean=float(np.mean(diffs)),sample_sd=float(np.std(diffs,ddof=1)),per_seed=diffs),whole_paper='NOT_RUN',fresh_pretraining='NOT_RUN',historical_identity='NOT_ESTABLISHED',full_dfs_regeneration='NOT_RUN')

def replay_archive(archive):
    with tempfile.TemporaryDirectory(prefix='b13-l200-') as td:
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            if any(Path(n).is_absolute() or '..' in Path(n).parts for n in z.namelist()):raise ValueError('Unsafe archive paths')
            z.extractall(td)
        return audit_l200(Path(td)/'labs/evidence/l200')

if __name__=='__main__':
    e=Path(__file__).resolve().parent/'evidence/b13'
    ledger=json.loads((e.parents[1]/'sources/b13/source-ledger.json').read_text())
    expected=next(x['sha256'] for x in ledger if x.get('inherited')=='evidence/l200/reproducer.zip')
    data=(e/'l200-reproducer.zip').read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected,'Inherited archive identity changed'
    r=replay_archive(data)
    (e/'rdbpfn-replay.json').write_text(json.dumps(r,indent=2)+'\n')
    print(r['status'],r['evaluations'],r['predictions'],{a:x['mean'] for a,x in r['models'].items()})
