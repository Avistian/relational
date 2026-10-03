"""B13 course mechanisms; deliberately separate from published model code."""
import itertools
import numpy as np

def canonical_schema(edges, n=4):
    """Exact directed unlabeled graph identity, practical only for tiny graphs."""
    edges=[tuple(e) for e in edges]
    if not 1<=n<=7 or len(set(edges))!=len(edges):raise ValueError('Invalid graph size or duplicates')
    if any(len(e)!=2 or any(type(v) is not int or not 0<=v<n for v in e) or e[0]==e[1] for e in edges):raise ValueError('Invalid edge')
    remaining=set(range(n))
    while remaining:
        roots={v for v in remaining if not any(b==v and a in remaining for a,b in edges)}
        if not roots:raise ValueError('Cyclic schema unsupported')
        remaining-=roots
    return min(tuple(sorted((p[a],p[b]) for a,b in edges)) for p in itertools.permutations(range(n)))

def parent_mean(values, foreign_keys):
    """Mean of direct-parent scalar attributes, one gathered value per FK."""
    if not values or len(values)!=len(foreign_keys):raise ValueError('Aligned nonempty parents required')
    gathered=[]
    for v,k in zip(values,foreign_keys):
        v=np.asarray(v,dtype=float);k=np.asarray(k)
        if v.ndim!=1 or k.ndim!=1 or k.dtype.kind not in 'iu' or not np.isfinite(v).all() or np.any(k<0) or np.any(k>=len(v)):raise ValueError('Invalid foreign key/value')
        if gathered and len(k)!=len(gathered[0]):raise ValueError('Unaligned child rows')
        gathered.append(v[k])
    return np.mean(np.stack(gathered),axis=0)

def fit_ridge(x, y, alpha=1.0):
    """Train-only population normalization; unpenalized target mean/intercept."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=2 or y.shape!=(len(x),) or len(x)==0 or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(alpha) or alpha<=0:raise ValueError('Invalid ridge inputs')
    mean=x.mean(0);scale=x.std(0);scale=np.where(scale==0,1.,scale)
    z=(x-mean)/scale;intercept=float(y.mean())
    coef=np.linalg.solve(z.T@z+alpha*np.eye(x.shape[1]),z.T@(y-intercept))
    return dict(mean=mean,scale=scale,coef=coef,intercept=intercept)

SCHEMAS={'chain':[(0,1),(1,2),(2,3)],'out-star':[(0,1),(0,2),(0,3)],
         'in-star':[(0,3),(1,3),(2,3)],'diamond':[(0,1),(0,2),(1,3),(2,3)]}

def generate_database(seed, family, rows=64):
    """Course SCM: four independent numeric attributes/table plus a relational target.

    4*rows*4 feature cells. PK/FK cells and 64 target labels are additional.
    Separate RNG streams keep exogenous draws paired across schema interventions.
    """
    edges=SCHEMAS[family];canonical_schema(edges)
    x=np.random.default_rng(seed).normal(size=(4,rows,4))
    fk={}
    for a,b in edges:
        rng=np.random.default_rng(np.random.SeedSequence([seed,11,a,b]))
        fk[(a,b)]=rng.integers(rows,size=rows)
    parents=[a for a,b in edges if b==3]
    signal=parent_mean([x[a,:,0] for a in parents],[fk[(a,3)] for a in parents])
    noise=np.random.default_rng(np.random.SeedSequence([seed,29])).normal(0,.1,rows)
    return dict(seed=seed,family=family,x=x,fk=fk,y=.5*x[3,:,0]+signal+noise)

def features(db, relational, corrupt=False):
    """No target reads. Corruption preserves each FK column's multiset."""
    x=db['x'];base=x[3].copy()
    if not relational:return base
    parents=[a for a,b in SCHEMAS[db['family']] if b==3];keys=[]
    for a in parents:
        k=db['fk'][(a,3)].copy()
        if corrupt:
            rng=np.random.default_rng(np.random.SeedSequence([db['seed'],97,a]))
            k=k[rng.permutation(len(k))]
        keys.append(k)
    return np.column_stack([base,parent_mean([x[a,:,0] for a in parents],keys)])

def predict(model,x):
    return (x-model['mean'])/model['scale']@model['coef']+model['intercept']

def run_experiment():
    """12 fixed fits. All test interventions reuse the intact-trained coefficients."""
    training=['chain','out-star','in-star']
    assert canonical_schema(SCHEMAS['diamond']) not in [canonical_schema(SCHEMAS[s]) for s in training]
    results=[]
    for seed in range(3):
        tests=[generate_database(100000+seed*100+i,'diamond') for i in range(4)]
        for diversity in [1,3]:
            dbs=[generate_database(seed*100+i,training[i%diversity]) for i in range(12)]
            assert sum(d['x'].size for d in dbs)==12288
            for relational in [False,True]:
                x=np.concatenate([features(d,relational) for d in dbs]);y=np.concatenate([d['y'] for d in dbs])
                model=fit_ridge(x,y,1.)
                keys=[[d['seed'],3,i] for d in tests for i in range(64)]
                target=np.concatenate([d['y'] for d in tests])
                p=predict(model,np.concatenate([features(d,relational) for d in tests]))
                broken=predict(model,np.concatenate([features(d,relational,True) for d in tests]))
                results.append(dict(seed=seed,diversity=diversity,arm='relational' if relational else 'feature-only',
                    training_seeds=[d['seed'] for d in dbs],training_families=[d['family'] for d in dbs],
                    training_feature_cells=12288,training_target_cells=768,keys=keys,labels=target.tolist(),
                    predictions=p.tolist(),corrupted_predictions=broken.tolist(),
                    mse=float(np.mean((p-target)**2)),corrupted_mse=float(np.mean((broken-target)**2)),
                    model={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in model.items()}))
    return dict(experiment='B13 held-out diamond course diagnostic',status='COMPLETE_COURSE_EXPERIMENT',
                fits=12,intact_predictions=3072,corrupted_predictions=3072,conditions=results,
                paper_model_reproduction=False,held_out_family='diamond')
