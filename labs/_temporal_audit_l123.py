"""All real graph timestamps plus independent SQL/PyG two-hop comparisons."""
import gzip,hashlib,json,sqlite3,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from torch_geometric.loader import NeighborLoader
from relkit.reg_l122 import construct_reg
from relkit.temporal_l123 import sample_temporal,audit_sample
from relkit.batch_audit_l123 import audit_batch
P=Path(__file__).resolve().parent;out=P/'evidence/l123';out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
dataset=get_dataset('rel-f1',download=True);db=dataset.get_db();task=get_task('rel-f1','driver-position',download=True)
archive=hashlib.sha256((Path(dataset.cache_dir)/'db.zip').read_bytes()).hexdigest()
assert archive=='ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'
tables={k:v.df for k,v in db.table_dict.items()};schema={k:{'pk':v.pkey_col,'fks':v.fkey_col_to_pkey_table} for k,v in db.table_dict.items()}
graph,_=construct_reg(tables,schema)
nodes={};times={};edges=[]
for kind,table in db.table_dict.items():
    stamp=None
    if table.time_col is not None:
        assert table.df[table.time_col].notna().all()
        stamp=(table.df[table.time_col].astype('int64').to_numpy()//10**9).astype('int64')
        graph[kind].time=torch.tensor(stamp.copy());times[kind]=stamp.tolist()
    for i in range(len(table)):nodes[(kind,i)]=None if stamp is None else (int(stamp[i]),int(stamp[i]))
for kind in graph.edge_types:
    src,rel,dst=kind
    for a,b in graph[kind].edge_index.T.tolist():edges.append(dict(src=(src,a),dst=(dst,b),kind=rel,stamp=None))
# Equal event/availability here is the release's event-time proxy, not measured arrival.
conn=sqlite3.connect(':memory:')
conn.execute('CREATE TABLE node (kind TEXT, id INTEGER, time INTEGER, PRIMARY KEY(kind,id))')
conn.executemany('INSERT INTO node VALUES (?,?,?)',[(k,i,None if s is None else s[0]) for (k,i),s in nodes.items()])
conn.execute('CREATE TABLE edge (id INTEGER PRIMARY KEY, src TEXT, a INTEGER, dst TEXT, b INTEGER)')
conn.executemany('INSERT INTO edge VALUES (?,?,?,?,?)',[(i,*e['src'],*e['dst']) for i,e in enumerate(edges)])
conn.execute('CREATE INDEX incoming ON edge(dst,b)')
records=[];queries=[];oracles={}
for split in ['train','val','test']:
 df=task.get_table(split).df
 for row in [0,len(df)//2,len(df)-1]:
  queries.append((split,int(df.iloc[row][task.entity_col]),int(df.iloc[row][task.time_col].timestamp())))
for split,entity,cutoff in queries:
 root=('drivers',entity);sample=sample_temporal(nodes,edges,root,cutoff,2);audit_sample(nodes,edges,sample)
 # Recursive SQL independently walks eligible incoming relations from this query.
 rows=conn.execute('''WITH RECURSIVE reach(kind,id,depth) AS (
 SELECT ?,?,0 UNION SELECT e.src,e.a,r.depth+1 FROM reach r
 JOIN edge e ON e.dst=r.kind AND e.b=r.id
 JOIN node n ON n.kind=e.src AND n.id=e.a
 WHERE r.depth<2 AND (n.time IS NULL OR n.time<=?)) SELECT DISTINCT kind,id FROM reach''',(*root,cutoff)).fetchall()
 assert set(rows)==sample['nodes']
 loader=NeighborLoader(graph,num_neighbors=[-1,-1],input_nodes=('drivers',torch.tensor([entity])),input_time=torch.tensor([cutoff]),time_attr='time',batch_size=1,num_workers=0)
 batch=next(iter(loader));batch['drivers'].seed_time=torch.tensor([cutoff]);audit_batch(batch,graph,'drivers')
 actual={(kind,int(i)) for kind in batch.node_types for i in batch[kind].n_id}
 assert actual==sample['nodes'],(split,entity,len(actual),len(sample['nodes']))
 # Compare typed endpoint sets; local edge ordering need not match.
 native={(kind,int(batch[kind[0]].n_id[a]),int(batch[kind[2]].n_id[b])) for kind in batch.edge_types for a,b in batch[kind].edge_index.T.tolist()}
 expected={((e['src'][0],e['kind'],e['dst'][0]),e['src'][1],e['dst'][1]) for i in sample['edges'] for e in [edges[i]]}
 assert native==expected,(len(native),len(expected))
 oracles[split+':'+str(entity)+':'+str(cutoff)]={'nodes':sorted([list(n) for n in sample['nodes']]),'edges':sample['edges']}
 records.append(dict(split=split,entity=entity,cutoff=cutoff,nodes=len(actual),edges=len(native)))
# Mixed query times must preserve root-specific duplicates; compare each component.
entity=queries[-1][1];cutoffs=[queries[0][2],queries[-1][2]]
loader=NeighborLoader(graph,num_neighbors=[-1,-1],input_nodes=('drivers',torch.tensor([entity,entity])),input_time=torch.tensor(cutoffs),time_attr='time',batch_size=2,num_workers=0)
batch=next(iter(loader));batch['drivers'].seed_time=torch.tensor(cutoffs);audit_batch(batch,graph,'drivers')
for q,cutoff in enumerate(cutoffs):
 actual={(k,int(i)) for k in batch.node_types for i,b in zip(batch[k].n_id,batch[k].batch) if b==q}
 assert actual==sample_temporal(nodes,edges,('drivers',entity),cutoff,2)['nodes']
# Export compact full graph, times and independent expected query sizes for portable labs.
payload=dict(rows={k:len(v) for k,v in tables.items()},times=times,edges={'|'.join(k):graph[k].edge_index.tolist() for k in graph.edge_types},queries=records,query_oracles=oracles,archive_sha256=archive,availability='NOT_OBSERVED: event-time proxy only')
raw=gzip.compress(json.dumps(payload,separators=(',',':')).encode(),mtime=0);(out/'temporal-graph.json.gz').write_bytes(raw)
report=dict(status='PASS',rows=len(nodes),edges=len(edges),dated_rows=sum(map(len,times.values())),dated_tables=list(times),static_tables=[k for k in tables if k not in times],comparisons=records,mixed_time_same_entity='PASS',sql='EXACT_NODE_SETS',pyg='EXACT_TYPED_NODES_AND_EDGES',graph_payload_sha256=hashlib.sha256(raw).hexdigest(),availability='NOT_OBSERVED',boundary='<= root cutoff; no invented edge timestamps')
(P/'_temporal_audit_l123_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
