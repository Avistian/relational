"""Independent graph, SQL, registry and corruption oracles for L171."""
import copy,hashlib,json,random,sqlite3,tempfile,shutil
from pathlib import Path
import pandas as pd
from relkit.corpus_l171 import validate_corpus,split_corpus,audit_tables
from _check_l171 import check171
from _audit_l171 import load_f1,replay171
P=Path(__file__).resolve().parent;E=P/'evidence/l171'
assert check171(validate_corpus,split_corpus,audit_tables)=='PASS'
manifest=json.loads((E/'input-manifest.json').read_text())
r=replay171(P,manifest,validate_corpus,split_corpus,audit_tables)
assert r==json.loads((E/'report.json').read_text())
# Independent all-pairs Boolean matrix closure rather than the learner's family frontier.
rng=random.Random(171);graph_cases=0
for size in range(2,13):
 for attempt in range(12):
    rows=[dict(database=str(i),source_families=rng.sample(['a','b','c','d','e','f','g'],rng.randint(1,2)),archive_sha256=f'{rng.randrange(size*2):064x}') for i in range(size)]
    reach=[[i==j or bool(set(rows[i]['source_families'])&set(rows[j]['source_families'])) or rows[i]['archive_sha256']==rows[j]['archive_sha256'] for j in range(size)] for i in range(size)]
    for k in range(size):
     for i in range(size):
      for j in range(size):reach[i][j]=reach[i][j] or (reach[i][k] and reach[k][j])
    for held in range(size):
        got=split_corpus(rows,str(held));assert got['train']==sorted(str(j) for j in range(size) if not reach[held][j])
        assert got['quarantine']==sorted(str(j) for j in range(size) if j!=held and reach[held][j]);graph_cases+=1
# Independent SQL counts for all real rows, all PKs and every declared FK.
tables=load_f1(P);db=sqlite3.connect(':memory:')
for name,t in tables.items():t['df'].to_sql(name,db,index=False)
a=r['full_snapshot'];sql_checks=0
for name,t in tables.items():
    pk=t['pkey_col']; row=a['tables'][name]
    n,nulls,dups=db.execute(f'SELECT COUNT(*), COUNT(*)-COUNT("{pk}"), COUNT("{pk}")-COUNT(DISTINCT "{pk}") FROM "{name}"').fetchone()
    assert (n,nulls,dups)==(row['rows'],row['pk_nulls'],row['pk_duplicate_excess']);sql_checks+=1
    if t['time_col']:
        col=t['time_col'];counts=db.execute(f'''SELECT SUM("{col}" < '2005-01-01'), SUM("{col}" >= '2005-01-01' AND "{col}" < '2010-01-01'), SUM("{col}" >= '2010-01-01'), SUM("{col}" IS NULL) FROM "{name}"''').fetchone()
        assert tuple(row['time_windows'].values())==counts;sql_checks+=1
for fk in a['foreign_keys']:
    name,col,target=fk['table'],fk['column'],fk['target'];pk=tables[target]['pkey_col']
    dangling=db.execute(f'SELECT COUNT(*) FROM "{name}" c WHERE c."{col}" IS NOT NULL AND NOT EXISTS (SELECT 1 FROM "{target}" p WHERE p."{pk}"=c."{col}")').fetchone()[0]
    assert dangling==fk['dangling'];sql_checks+=1
db.close()
# Random key populations, including nulls/duplicates; direct row-by-row independent oracle.
key_cases=0
for _ in range(100):
    parents=[rng.choice([None,0,1,2,3]) for _ in range(7)]
    children=[rng.choice([None,0,1,2,3,4,5]) for _ in range(15)]
    tiny={'p':dict(df=pd.DataFrame({'id':parents}),pkey_col='id',time_col=None,fkey_col_to_pkey_table={}),
          'c':dict(df=pd.DataFrame({'parent':children}),pkey_col=None,time_col=None,fkey_col_to_pkey_table={'parent':'p'})}
    got=audit_tables(tiny,'2005-01-01','2010-01-01')
    assert got['foreign_keys'][0]['dangling']==sum(x is not None and x not in parents for x in children)
    assert got['tables']['p']['pk_nulls']==parents.count(None)
    assert got['tables']['p']['pk_duplicate_excess']==len([x for x in parents if x is not None])-len({x for x in parents if x is not None});key_cases+=1
# Independent expected source-declaration populations checked against the source definitions.
expected={'rel-amazon':3,'rel-avito':8,'rel-event':5,'rel-f1':9,'rel-hm':3,'rel-stack':7,'rel-trial':15}
assert {x['database']:len(x['tables']) for x in r['inventory']}==expected
assert all(len(x['train'])==6 and not x['quarantine'] for x in r['splits'].values())
assert r['contamination_demo']['quarantine']==['synthetic-bridge','synthetic-f1-copy','synthetic-tail']
# Verify tests reject plausible wrong learner implementations.
wrong=[(lambda x: x,split_corpus,audit_tables),
       (validate_corpus,lambda rows,h:dict(train=sorted(x['database'] for x in rows if x['database']!=h),heldout=[h],quarantine=[]),audit_tables),
       (validate_corpus,split_corpus,lambda *args:dict(rows=0,foreign_key_columns=0))]
for functions in wrong:
    try:check171(*functions)
    except (AssertionError,KeyError):pass
    else:raise AssertionError('Incorrect learner function survived')
# Hash mismatch and missing file: preserve authoritative originals.
corruptions=0
with tempfile.TemporaryDirectory(prefix='l171-corruption-') as tmp:
    root=Path(tmp)
    for name in manifest['files']:
        dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,dst)
    for name in ['sources/l171/relbench/datasets/f1.py','evidence/l171/db/results.parquet','evidence/l171/rel-f1-db.zip']:
        path=root/name;raw=path.read_bytes();path.write_bytes(raw+b'corrupt')
        try:replay171(root,manifest,validate_corpus,split_corpus,audit_tables)
        except ValueError:corruptions+=1
        else:raise AssertionError('Corrupt packet accepted')
        path.write_bytes(raw)
result=dict(status='PASS',graph_cases=graph_cases,sql_full_population_checks=sql_checks,random_key_cases=key_cases,
    rejected_wrong_implementations=3,rejected_corrupt_packets=corruptions,source_databases=7,source_declared_tables=50,
    real_rows=a['rows'],real_foreign_key_columns=a['foreign_key_columns'],integrity=a['integrity'],availability='NOT_ESTABLISHED')
(P/'_verify_l171_results.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
