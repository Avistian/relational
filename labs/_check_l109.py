"""Contract tests: selection precedes joins; later knowledge cannot rewrite history."""
import json,sqlite3
from pathlib import Path
from relkit import database_l109 as m

def fixture():
    # immutable event time; append-only revisions, including retroactive corrections
    people=[dict(id='A',valid_from=0,observed_at=0,revision=0,deleted=False,value=1),
            dict(id='A',valid_from=0,observed_at=9,revision=1,deleted=False,value=9),
            dict(id='B',valid_from=0,observed_at=7,revision=0,deleted=False,value=2)]
    facts=[dict(id='r',valid_from=2,observed_at=3,revision=0,deleted=False,person='A',value=5),
           dict(id='r',valid_from=2,observed_at=8,revision=1,deleted=False,person='B',value=6),
           dict(id='r',valid_from=2,observed_at=10,revision=2,deleted=True,person='B',value=6)]
    return people,facts

def oracle(rows,t):
    con=sqlite3.connect(':memory:')
    con.execute('create table versions(id text, valid_from real, observed_at real, revision int, deleted int, payload int)')
    con.executemany('insert into versions values(?,?,?,?,?,?)',[(r['id'],r['valid_from'],r['observed_at'],r['revision'],r['deleted'],i) for i,r in enumerate(rows)])
    ids=con.execute('''select payload from (select *, row_number() over
      (partition by id order by valid_from desc,observed_at desc,revision desc) as n
      from versions where valid_from<=? and observed_at<=?) where n=1 and deleted=0 order by id''',(t,t)).fetchall()
    con.close();return {rows[i]['id']:rows[i] for i, in ids}

def check():
    people,facts=fixture()
    assert m.asof_versions(people,6)['A']['value']==1,'Late correction must not rewrite earlier features'
    assert 'B' not in m.asof_versions(people,6),'Old valid time does not imply known then'
    assert m.asof_versions(people,9)['A']['value']==9,'Inclusive observation tie is available'
    assert not m.asof_versions(facts,10),'Tombstone must suppress the old row, not resurrect it'
    g=m.graph_at(people,facts,6)
    assert g['edges']==[('r','A',3)],'Use the versioned FK, not the newest stored FK'
    assert m.graph_at(people,facts,8)['edges']==[('r','B',8)]
    assert m.graph_at(people,facts,10)['edges']==[]
    assert m.label_ready(5,4,[6,12])==12,'Label waits for both horizon and constituent arrivals'
    assert m.label_ready(5,4,[])==9,'Empty outcome needs a closed horizon'
    assert m.legal_history([{'event_time':2,'observed_at':7},{'event_time':5,'observed_at':5}],5)==[1]
    import random
    rng=random.Random(109);count=0
    for seed in range(40):
        rows=[]
        for entity in range(6):
            for revision in range(4):
                valid=rng.randrange(12);arrival=rng.randrange(12)
                rows.append(dict(id=str(entity),valid_from=valid,observed_at=arrival,revision=revision,deleted=rng.random()<.2,value=rng.randrange(100)))
        for t in range(13):
            assert m.asof_versions(rows,t)==oracle(rows,t)
            count+=1
    # A declared row version key is unique; missing times fail closed.
    for bad in [[people[0],dict(people[0],value=999)],[dict(people[0],observed_at=None)]]:
        try:m.asof_versions(bad,6)
        except ValueError:pass
        else:raise AssertionError('Ambiguous/missing version keys must be rejected')
    return {'status':'PASS','independent_sql_queries':count,'boundary_fk_tombstone_maturity':'PASS'}
if __name__=='__main__':
    result=check();Path(__file__).with_name('_check_l109_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
