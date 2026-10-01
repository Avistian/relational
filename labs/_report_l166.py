"""Rescore complete keyed predictions; means and sample SD describe support draws."""
def score166(keys,labels,scores):
    import numpy as np
    k=np.asarray(keys);y=np.asarray(labels);s=np.asarray(scores)
    if k.ndim!=2 or k.shape[1]!=2 or len(set(map(tuple,k)))!=len(k):raise ValueError('Duplicate or malformed query identities')
    if y.shape!=s.shape or y.ndim!=1 or len(k)!=len(y) or not np.isfinite(s).all() or set(y)!={0,1}:raise ValueError('Invalid score population')
    pos=s[y==1];neg=np.sort(s[y==0])
    lower=np.searchsorted(neg,pos,side='left');upper=np.searchsorted(neg,pos,side='right')
    return float(np.mean((lower+.5*(upper-lower))/len(neg)))

def summarize166(packet):
    import numpy as np
    expected={(a,s) for a in ['RDBPFN','RDBPFN_single','TabICLv1.1'] for s in range(10)}
    seen=set();by={a:[] for a,s in expected};keys=None;labels=None;support={}
    for r in packet:
        identity=(r['arm'],r['seed'])
        if identity not in expected or identity in seen:raise ValueError('Unexpected/duplicate evaluation')
        seen.add(identity)
        if keys is None:keys=r['keys'];labels=r['label']
        if not np.array_equal(keys,r['keys']) or not np.array_equal(labels,r['label']):raise ValueError('Unaligned test population')
        if r['seed'] in support and support[r['seed']]!=r['support_keys']:raise ValueError('Unpaired support')
        support[r['seed']]=r['support_keys']
        if len(set(map(tuple,r['support_keys'])))!=512:raise ValueError('Invalid support count')
        if set(map(tuple,r['support_keys'])) & set(map(tuple,r['keys'])):raise ValueError('Support/test overlap')
        by[r['arm']].append((r['seed'],score166(r['keys'],r['label'],r['probability'])))
    if seen!=expected:raise ValueError('Missing selected runs')
    targets={'RDBPFN':.7219,'RDBPFN_single':.6640,'TabICLv1.1':.7176};summary={};vectors={}
    for arm,rows in sorted(by.items()):
        values=np.array([v for seed,v in sorted(rows)]);vectors[arm]=values
        summary[arm]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),paper=targets[arm],delta=float(values.mean()-targets[arm]),
                          tolerance=.02,status='CLOSE' if abs(values.mean()-targets[arm])<=.02 else 'OUTSIDE_TOLERANCE',per_seed=values.tolist())
    paired={}
    for other in ['RDBPFN_single','TabICLv1.1']:
        d=vectors['RDBPFN']-vectors[other];paired['RDBPFN-minus-'+other]=dict(mean=float(d.mean()),sample_sd=float(d.std(ddof=1)),per_seed=d.tolist())
    return dict(experiment='L166 Table9 driver-dnf 512-context released-checkpoint replay',status='COMPLETE',runs=30,predictions=30*len(keys),unique_test_queries=len(keys),
                models=summary,paired=paired,historical_identity='NOT_ESTABLISHED',fresh_pretraining='NOT_RUN',whole_paper='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')

if __name__=='__main__':
    import hashlib,json
    import numpy as np
    from pathlib import Path
    from sklearn.metrics import roc_auc_score
    P=Path(__file__).resolve().parent;E=P/'evidence/l166';packet=[]
    prepared=np.load(E/'prepared.npz');max_delta=0
    for folder in ['pilot-2','full-1']:
        receipt=json.loads((E/folder/'receipt.json').read_text())
        for record in receipt['records']:
            p=E/folder/f"{record['arm']}-{record['seed']}.npz";assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256']
            a=np.load(p);r={k:a[k].tolist() for k in a.files};r.update(arm=record['arm'],seed=record['seed'])
            np.testing.assert_array_equal(a['keys'],prepared['test_keys']);np.testing.assert_array_equal(a['label'],prepared['y_test'])
            np.testing.assert_array_equal(a['support_keys'],prepared['train_keys'][prepared['support'][record['seed']]])
            value=score166(a['keys'],a['label'],a['probability']);oracle=roc_auc_score(a['label'],a['probability'])
            assert abs(value-oracle)<1e-12 and abs(value-record['auc'])<1e-12
            max_delta=max(max_delta,abs(value-oracle));packet.append(r)
    report=summarize166(packet)
    for bad in [packet[:-1],packet+[packet[0]]]:
        try:summarize166(bad)
        except ValueError:pass
        else:raise AssertionError('Incomplete/duplicate experiment accepted')
    # A separate pairwise definition checks ties and order on small populations.
    for labels,scores in [([0,1,0,1],[.5,.5,0.,1.]),([1,0],[.2,.8])]:
        pairs=[(a>b)+.5*(a==b) for y,a in zip(labels,scores) if y==1 for z,b in zip(labels,scores) if z==0]
        assert score166([[i,0] for i in range(len(labels))],labels,scores)==sum(pairs)/len(pairs)
    (E/'predictions.json').write_text(json.dumps(packet,separators=(',',':'))+'\n')
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    audit=dict(status='PASS',rows_checked=report['predictions'],sklearn_max_delta=max_delta,complete_keys_and_support_pairing='PASS',missing_duplicate_runs_rejected='PASS')
    (E/'evaluation-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(report,indent=2))
