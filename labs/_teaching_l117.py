"""Default standalone lab: authentic real rows plus controlled edge cases."""
# NOTEBOOK_RUN
import io,json,zipfile,hashlib,urllib.request
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

def archive(url,expected):
    payload=urllib.request.urlopen(url,timeout=60).read()
    assert hashlib.sha256(payload).hexdigest()==expected,'Archive identity changed'
    return zipfile.ZipFile(io.BytesIO(payload))

with archive('https://relbench.stanford.edu/download/rel-f1/db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482') as z:
    tables={name:pq.read_table(io.BytesIO(z.read('db/'+name+'.parquet')),use_threads=False).to_pandas() for name in ['drivers','results','constructors']}
with archive('https://relbench.stanford.edu/download/rel-f1/tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e') as z:
    train=pq.read_table(io.BytesIO(z.read('driver-position/train.parquet')),use_threads=False).to_pandas()

# Real released key identities; each student function affects the final report.
drivers,results,constructors=(tables[k] for k in ['drivers','results','constructors'])
nd,nr=len(drivers),len(results)
rd=foreign_key_edges(drivers.driverId.tolist(),results.driverId.tolist())
rc=foreign_key_edges(constructors.constructorId.tolist(),results.constructorId.tolist())
edges=[]
for src,dst in rd.T:edges.extend([(int(dst),nd+int(src)),(nd+int(src),int(dst))])
for src,dst in rc.T:edges.extend([(nd+int(src),nd+nr+int(dst)),(nd+nr+int(dst),nd+int(src))])
times=[None]*nd+[int(x.value//10**9) for x in results.date]+[None]*len(constructors)
query=train.iloc[0];cutoff=int(query.date.value//10**9);seed=drivers.driverId.tolist().index(int(query.driverId))
selected=temporal_nodes(edges,times,seed,cutoff,2)
assert seed in selected
future=[i for i in selected if times[i] is not None and times[i]>cutoff]
assert not future
historic=[i-nd for i in selected if nd<=i<nd+nr]
assert len(historic)>0
input_ids=np.array([2,0,1]);attached=query_targets(train.position.to_numpy(),input_ids)
np.testing.assert_array_equal(attached,train.iloc[input_ids].position.to_numpy())
report={'status':'PASS','dataset':'rel-f1 released real rows; three-table learning view',
        'query_driver_key':int(query.driverId),'query_time':str(query.date),
        'nodes_in_learning_view':len(times),'directed_edges_in_learning_view':len(edges),
        'eligible_two_hop_nodes':len(selected),'historic_result_rows':len(historic),
        'future_rows_included':len(future),'query_row_order':input_ids.tolist(),
        'attached_targets':attached.tolist(),'full_benchmark':'Separate author evidence; not run by this cell',
        'learner_status':'PENDING_WRITTEN_DEFENSE'}
Path('l117-task-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
