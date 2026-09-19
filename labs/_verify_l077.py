"""Independent analytical, SQL and intervention checks for the complete experiment."""
import hashlib,json,platform,sqlite3
from pathlib import Path
import numpy as np
from relkit.ceiling_l077 import make_database,split_pairs,flat_rows,temporal_delta,collision_ceiling,fit_stump,predict_stump,run_experiment
LAB=Path(__file__).resolve().parent

def run():
    customers,events=make_database(30,7)
    ids=[c['customer_id'] for c in customers]
    x=flat_rows(customers,events,10); delta=temporal_delta(customers,events,10)
    y=np.array([c['target'] for c in customers])
    assert x.shape==(60,5) and delta.shape==(60,)
    assert collision_ceiling(x,y)==0.5
    assert collision_ceiling(np.column_stack([x,delta]),y)==1.
    assert collision_ceiling([[0],[0],[0],[1]], [0,0,1,1])==.75
    assert collision_ceiling(np.arange(60).reshape(-1,1),y)==1.
    # Independent SQL grouping and ordered-subquery oracle.
    db=sqlite3.connect(':memory:')
    db.execute('CREATE TABLE events(customer_id INTEGER, event_day INTEGER, available_day INTEGER, amount REAL)')
    db.executemany('INSERT INTO events VALUES (:customer_id,:event_day,:available_day,:amount)',events)
    for i,cid in enumerate(ids):
        got=db.execute('SELECT COUNT(*),SUM(amount),AVG(amount),MAX(amount) FROM events WHERE customer_id=? AND event_day<=10 AND available_day<=10',(cid,)).fetchone()
        assert np.array_equal(x[i,1:],got)
        amounts=db.execute('SELECT amount FROM events WHERE customer_id=? AND event_day<=10 AND available_day<=10 ORDER BY event_day',(cid,)).fetchall()
        assert delta[i]==amounts[-1][0]-amounts[0][0]
    db.close()
    changed=[dict(e,amount=1e9) if e['available_day']>10 or e['event_day']>10 else dict(e) for e in events]
    assert np.array_equal(x,flat_rows(customers,changed,10))
    assert np.array_equal(delta,temporal_delta(customers,changed,10))
    assert np.array_equal(x[::-1],flat_rows(customers[::-1],events[::-1],10))
    assert np.array_equal(delta[::-1],temporal_delta(customers[::-1],events[::-1],10))
    assert flat_rows([customers[0]],[],10)[0,1:].tolist()==[0,0,0,0]
    assert temporal_delta([customers[0]],[],10).tolist()==[0]
    # Every pair stays in one partition; both labels occur at each representation.
    parts=split_pairs(customers,7)
    pairsets=[{customers[i]['pair_id'] for i in p} for p in parts]
    assert not(pairsets[0]&pairsets[1] or pairsets[0]&pairsets[2] or pairsets[1]&pairsets[2])
    assert sorted(np.concatenate(parts).tolist())==list(range(60))
    for p in parts: assert collision_ceiling(x[p],y[p])==.5
    assert np.mean(predict_stump(fit_stump(delta[:,None],y),delta[:,None])==y)==1
    # Independent library learner uses the same information and partition.
    from sklearn.tree import DecisionTreeClassifier
    tr,va,te=parts
    tree_scores={}
    for name,a in [('flat',x),('restored',np.column_stack([x,delta]))]:
        tree=DecisionTreeClassifier(max_depth=3,random_state=7).fit(a[tr],y[tr])
        tree_scores[name]=float(tree.score(a[te],y[te]))
    assert tree_scores=={'flat':.5,'restored':1.}
    runs=[run_experiment(1000,seed) for seed in range(5)]
    for r in runs:
        for arm,target in [('flat',.5),('restored',1.)]:
            assert r['scores'][arm]['test']==target
            assert r['ceilings'][arm]==target
    result={'status':'PASS','implementation_sha256':hashlib.sha256((LAB/'relkit/ceiling_l077.py').read_bytes()).hexdigest(),
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'checks':['SQL aggregation and ordering oracle','exact class-count bound','unique-ID boundary','pair split isolation','future and late-arrival invariance','row permutations','empty history','independent sklearn tree'],
            'library_control':tree_scores,'runs':runs,'paper_result_reproduction':'NOT_APPLICABLE: original synthesis construction; published RelBench experiment remains separate'}
    (LAB/'_verify_l077_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':run()
