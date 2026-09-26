"""Independent graph, boundary, mutation and all-prediction evidence checks."""
import json,sqlite3,random
from pathlib import Path
import numpy as np
import pandas as pd
from relkit import database_l109 as m
from _check_l109 import check,fixture,oracle
P=Path(__file__).resolve().parent

def audit():
    check();rng=random.Random(110);compared=0
    for seed in range(30):
        people=[];facts=[]
        for kind,rows,n in [('p',people,5),('f',facts,9)]:
            for i in range(n):
                for v in range(3):
                    row=dict(id=kind+str(i),valid_from=rng.randrange(9),observed_at=rng.randrange(12),revision=v,deleted=rng.random()<.15,value=v)
                    if kind=='f':row['person']='p'+str(rng.randrange(7))
                    rows.append(row)
        for t in range(13):
            p=oracle(people,t);f=oracle(facts,t)
            con=sqlite3.connect(':memory:');con.execute('create table p(id text,v real,a real)');con.execute('create table f(id text,parent text,v real,a real)')
            con.executemany('insert into p values(?,?,?)',[(r['id'],r['valid_from'],r['observed_at']) for r in p.values()])
            con.executemany('insert into f values(?,?,?,?)',[(r['id'],r['person'],r['valid_from'],r['observed_at']) for r in f.values()])
            expected=con.execute('select f.id,p.id,max(f.v,f.a,p.v,p.a) from f join p on f.parent=p.id order by f.id').fetchall()
            actual=m.graph_at(people,facts,t)
            assert actual['edges']==expected
            assert actual['dangling']==sorted(k for k,r in f.items() if r['person'] not in p)
            con.close();compared+=1
    # Mutation checks must make the same contract suite fail, not simply raise on import.
    original=m.asof_versions;original_legal=m.legal_history;original_ready=m.label_ready
    mutants={
      'ignore_observation':('asof_versions',lambda rows,t:original([dict(r,observed_at=min(r['observed_at'],t)) for r in rows],t)),
      'drop_tombstones_first':('asof_versions',lambda rows,t:original([r for r in rows if not r['deleted']],t)),
      'latest_database_snapshot':('asof_versions',lambda rows,t:original(rows,1000)),
      'event_time_only':('legal_history',lambda rows,t:[i for i,r in enumerate(rows) if r['event_time']<=t]),
      'ignore_label_arrivals':('label_ready',lambda q,h,a:q+h)}
    caught=[]
    for name,(function,mutant) in mutants.items():
        setattr(m,function,mutant)
        try:check()
        except AssertionError:caught.append(name)
        finally:m.asof_versions=original;m.legal_history=original_legal;m.label_ready=original_ready
    assert len(caught)==len(mutants)
    report=json.loads((P/'evidence/l109/reproduction.json').read_text());arrays=np.load(P/'evidence/l109/predictions.npz');metrics=[]
    for record in report['results']:
        split,arm=record['split'],record['arm'];pred=arrays[split+'_'+arm];target=arrays[split+'_target']
        # Scalar absolute-error accumulation, independent of numpy aggregation.
        mae=sum(abs(float(p)-float(y)) for p,y in zip(pred,target))/len(target)
        assert abs(mae-record['mae'])<1e-12;metrics.append(mae)
    out={'status':'PASS','independent_sql_graph_queries':compared,'rejected_mutants':caught,'independent_full_prediction_metrics':len(metrics),'real_ingestion_history':'NOT_AVAILABLE'}
    (P/'_audit_l109_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
if __name__=='__main__':audit()
