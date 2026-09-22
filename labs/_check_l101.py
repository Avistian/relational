"""Independent boundary and path oracles; run before accepting the lesson."""
import sqlite3
import numpy as np
from relkit.temporal_l101 import available, split_queries, incoming_subgraph

def check():
    assert available(4, 7, 5) is False, 'Late arrival must stay hidden'
    assert available(5, 5, 5) is True, 'Inclusive boundary is deliberate'
    assert available(6, 4, 5) is False, 'Future event must stay hidden'
    q=[{'time':t,'label_ready':t+3} for t in [5,8,10,17,20]]
    assert split_queries(q,10,20)==['train','purged','val','val','test']
    nodes={'a':(0,0),'b':(1,1),'c':(8,8),'d':(2,9),'e':(5,5)}
    edges=[('b','a',1),('c','b',8),('d','b',2),('e','b',5)]
    assert incoming_subgraph(nodes,edges,'a',5,2)==({'a','b','e'},{0,3})
    assert incoming_subgraph(nodes,edges,'a',9,2)==(set(nodes),set(range(4)))
    # Independent SQL filtering + recursive reachability, randomized typed-like IDs.
    rng=np.random.default_rng(101)
    for _ in range(32):
        nodes={f'n{i}':tuple(map(int,rng.integers(0,11,2))) for i in range(14)}
        nodes['n0']=(0,0)
        edges=[(f'n{int(a)}',f'n{int(b)}',int(t)) for a,b,t in rng.integers([0,0,0],[14,14,11],(50,3))]
        db=sqlite3.connect(':memory:')
        db.execute('create table nodes(id text,event int,arrival int)')
        db.execute('create table edges(id int,src text,dst text,arrival int)')
        db.executemany('insert into nodes values(?,?,?)',[(k,*v) for k,v in nodes.items()])
        db.executemany('insert into edges values(?,?,?,?)',[(i,*e) for i,e in enumerate(edges)])
        for cutoff in [0,5,10]:
            legal=db.execute('select e.id,e.src,e.dst from edges e join nodes s on s.id=e.src join nodes d on d.id=e.dst where max(s.event,s.arrival,d.event,d.arrival,e.arrival)<=?',(cutoff,)).fetchall()
            for depth in [0,1,2,3]:
                frontier={'n0'};seen=set(frontier);used=set()
                for hop in range(depth):
                    found=[e for e in legal if e[2] in frontier]
                    frontier={e[1] for e in found};seen |= frontier;used|={e[0] for e in found}
                assert incoming_subgraph(nodes,edges,'n0',cutoff,depth)==(seen,used)
        db.close()
    return {'status':'PASS','random_graphs':32,'query_depth_checks':384,'oracle':'SQLite filtering plus independent traversal'}

if __name__=='__main__':
    import json
    from pathlib import Path
    result=check();Path(__file__).with_name('_check_l101_results.json').write_text(json.dumps(result,indent=2));print(result)
