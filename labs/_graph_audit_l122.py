"""Full released F1 topology: learner constructor versus SQL and original code."""
import hashlib,importlib.util,json,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from relbench.datasets import get_dataset
from relkit.reg_l122 import construct_reg
P=Path(__file__).resolve().parent
torch.set_num_threads(1)
dataset=get_dataset('rel-f1',download=True)
assert hashlib.sha256((Path(dataset.cache_dir)/'db.zip').read_bytes()).hexdigest()=='ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'
db=dataset.get_db();tables={k:v.df for k,v in db.table_dict.items()}
schema={k:{'pk':v.pkey_col,'fks':v.fkey_col_to_pkey_table} for k,v in db.table_dict.items()}
graph,features=construct_reg(tables,schema)
spec=importlib.util.spec_from_file_location('original_graph_l122',P/'sources/l117/graph.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
import copy
topology_db=copy.deepcopy(db)
for table in topology_db.table_dict.values():table.time_col=None
original,_=mod.make_pkey_fkey_graph(topology_db,{k:{} for k in tables})
assert set(original.edge_types)==set(graph.edge_types)
connection=sqlite3.connect(':memory:')
for name,df in tables.items():
    pk=schema[name]['pk'];cols=list(dict.fromkeys(([pk] if pk else [])+list(schema[name]['fks'])))
    sql=df[cols].copy();sql['__row__']=np.arange(len(df));sql.to_sql(name,connection,index=False)
report=[];payload={}
for name in tables:
    assert graph[name].num_nodes==len(tables[name])==original[name].num_nodes
    payload['nodes_'+name]=np.array([graph[name].num_nodes])
    for fk,dest in schema[name]['fks'].items():
        pk=schema[dest]['pk'];kind=(name,'f2p_'+fk,dest);rev=(dest,'rev_f2p_'+fk,name)
        rows=connection.execute(f'SELECT a.__row__,b.__row__ FROM "{name}" a JOIN "{dest}" b ON a."{fk}"=b."{pk}" ORDER BY a.__row__,b.__row__').fetchall()
        expected=np.asarray(rows,dtype=np.int64).reshape(-1,2).T
        actual=graph[kind].edge_index.numpy();np.testing.assert_array_equal(actual,expected)
        np.testing.assert_array_equal(original[kind].edge_index.numpy(),expected)
        ordered=expected[:,np.lexsort((expected[0],expected[1]))][::-1]
        np.testing.assert_array_equal(original[rev].edge_index.numpy(),ordered)
        np.testing.assert_array_equal(graph[rev].edge_index.numpy(),expected[::-1])
        payload['|'.join(kind)]=actual;payload['|'.join(rev)]=graph[rev].edge_index.numpy()
        report.append({'type':list(kind),'edges':actual.shape[1],'null_foreign_keys':int(tables[name][fk].isna().sum()),'sql':'EXACT','upstream':'EXACT_AFTER_SORT','sha256':hashlib.sha256(actual.tobytes()).hexdigest()})
connection.close()
out=P/'evidence/l122';out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'reg-topology.npz',**payload)
# Standalone input retains every released key row, independently of edge tensors.
import gzip
key_tables={}
for name,df in tables.items():
    pk=schema[name]['pk'];cols=list(dict.fromkeys(([pk] if pk else [])+list(schema[name]['fks'])))
    key_tables[name]={'columns':cols,'data':json.loads(df[cols].to_json(orient='values')),'row_count':len(df)}
fixture={'schema':schema,'tables':key_tables,'archive_sha256':'ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'}
(out/'key-tables.json.gz').write_bytes(gzip.compress(json.dumps(fixture,separators=(',',':')).encode(),mtime=0))
(P/'results/l122').mkdir(parents=True,exist_ok=True);torch.save(graph,P/'results/l122/constructed-reg.pt')
r={'status':'PASS','rows':{k:len(v) for k,v in tables.items()},'total_rows':sum(map(len,tables.values())),
   'forward_relations':report,'directed_edges':sum(graph[k].num_edges for k in graph.edge_types),'feature_columns':features,
   'scope':'Full test-censored released F1 topology; original topology comparison uses constant features and omits time attributes (local pandas read-only array incompatibility in upstream time conversion). Full learned feature path independently executes in every GPU fit.',
   'temporal_safety':'NOT_ESTABLISHED_BY_STATIC_TOPOLOGY','source_sha256':hashlib.sha256((P/'relkit/reg_l122.py').read_bytes()).hexdigest(),
   'topology_sha256':hashlib.sha256((out/'reg-topology.npz').read_bytes()).hexdigest()}
(P/'_graph_audit_l122_results.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
