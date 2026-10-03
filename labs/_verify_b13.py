"""Independent scalar feature/MSE and augmented least-squares oracles."""
import copy,itertools,json
from pathlib import Path
import numpy as np
from relkit.synthetic_b13 import SCHEMAS,canonical_schema,generate_database,features,run_experiment

def scalar_features(db,relational,corrupt=False):
    parents=[a for a,b in SCHEMAS[db['family']] if b==3];keys={}
    for a in parents:
        k=db['fk'][(a,3)]
        if corrupt:k=k[np.random.default_rng(np.random.SeedSequence([db['seed'],97,a])).permutation(len(k))]
        keys[a]=k
    rows=[]
    for i in range(len(db['y'])):
        row=list(db['x'][3,i])
        if relational:row.append(sum(float(db['x'][a,keys[a][i],0]) for a in parents)/len(parents))
        rows.append(row)
    return np.array(rows)

def verify(report):
    assert report['fits']==12 and report['intact_predictions']==3072 and report['corrupted_predictions']==3072
    expected={(s,d,a) for s in range(3) for d in [1,3] for a in ['relational','feature-only']}
    if len(report['conditions'])!=12 or {(r['seed'],r['diversity'],r['arm']) for r in report['conditions']}!=expected:raise ValueError('Incomplete course grid')
    errors=[];count=0
    for r in report['conditions']:
        seed=r['seed'];div=r['diversity'];rel=r['arm']=='relational';families=['chain','out-star','in-star']
        dbs=[generate_database(seed*100+i,families[i%div]) for i in range(12)]
        assert r['training_seeds']==[d['seed'] for d in dbs] and r['training_families']==[d['family'] for d in dbs]
        assert r['training_feature_cells']==sum(d['x'].size for d in dbs)==12288
        assert r['training_target_cells']==sum(d['y'].size for d in dbs)==768
        test=[generate_database(100000+seed*100+i,'diamond') for i in range(4)]
        assert r['keys']==[[d['seed'],3,i] for d in test for i in range(64)]
        x=np.concatenate([scalar_features(d,rel) for d in dbs]);y=np.concatenate([d['y'] for d in dbs])
        mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;z=(x-mu)/sd
        # Solve augmented regression with SVD, independent of normal equations.
        coef=np.linalg.lstsq(np.vstack([z,np.eye(z.shape[1])]),np.r_[y-y.mean(),np.zeros(z.shape[1])],rcond=None)[0]
        np.testing.assert_allclose(coef,r['model']['coef'],atol=1e-12,rtol=1e-12)
        for field,value in [('mean',mu),('scale',sd),('intercept',y.mean())]:np.testing.assert_allclose(r['model'][field],value,atol=1e-12,rtol=1e-12)
        truth=np.concatenate([d['y'] for d in test]);np.testing.assert_array_equal(r['labels'],truth)
        for corrupt,pkey,mkey in [(False,'predictions','mse'),(True,'corrupted_predictions','corrupted_mse')]:
            tx=np.concatenate([scalar_features(d,rel,corrupt) for d in test])
            oracle=(tx-mu)/sd@coef+y.mean();saved=np.array(r[pkey])
            np.testing.assert_allclose(saved,oracle,atol=1e-12,rtol=1e-12)
            mse=sum((float(p)-float(t))**2 for p,t in zip(saved,truth))/len(truth)
            assert abs(mse-r[mkey])<1e-12;count+=len(saved)
        if not rel:assert r['predictions']==r['corrupted_predictions']
        for d in test:
            before=features(d,rel);changed=copy.deepcopy(d);changed['y'][:]=1e9
            np.testing.assert_array_equal(before,features(changed,rel))
        errors.append(float(np.max(np.abs(coef-np.array(r['model']['coef'])))))
    assert len({canonical_schema(e) for e in SCHEMAS.values()})==4
    for edges in SCHEMAS.values():
        for perm in itertools.permutations(range(4)):
            assert canonical_schema([(perm[a],perm[b]) for a,b in edges])==canonical_schema(edges)
    return dict(status='PASS',fits=12,predictions_checked=count,svd_coefficient_max_error=max(errors),target_intervention='PASS',feature_only_fk_invariance='PASS',exact_unlabeled_topology_separation='PASS')

if __name__=='__main__':
    p=Path(__file__).resolve().parent;report=json.loads((p/'evidence/b13/diagnostic.json').read_text());result=verify(report)
    mutations=0
    for mode in ['prediction','label','key','budget','missing']:
        r=copy.deepcopy(report)
        if mode=='prediction':r['conditions'][0]['predictions'][0]+=.1
        elif mode=='label':r['conditions'][0]['labels'][0]+=1
        elif mode=='key':r['conditions'][0]['keys'][0][0]+=1
        elif mode=='budget':r['conditions'][0]['training_feature_cells']-=1
        else:r['conditions'].pop()
        try:verify(r)
        except (ValueError,AssertionError):mutations+=1
        else:raise AssertionError('Corruption admitted: '+mode)
    result['rejected_corruptions']=mutations
    (p/'_verify_b13_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
