"""Full released-query cutoff census: pandas REG edges against independent SQL."""
import hashlib,json,sqlite3,time
from pathlib import Path
import numpy as np
import pandas as pd
from relkit import f1_l109 as m
P=Path(__file__).resolve().parent

def audit():
    assert hasattr(m,'graph_snapshot'),'Implement the real-data schema-to-graph handoff'
    tables,tasks,metadata=m.load_inputs(P/'data/l109');con=sqlite3.connect(':memory:');start=time.perf_counter()
    for name,frame in tables.items():
        meta=metadata[name];columns=list(dict.fromkeys([meta['pkey_col'],*meta['fkey_col_to_pkey_table']]))
        df=frame[columns].copy();df['clock']=frame[meta['time_col']].astype('int64') if meta['time_col'] else -9223372036854775807
        # pandas 3 uses microseconds for these timestamps; normalize SQL clocks explicitly.
        if meta['time_col']:df['clock']=frame[meta['time_col']].to_numpy(dtype='datetime64[ns]').astype('int64')
        df.to_sql(name,con,index=False)
    rows=[];comparisons=0
    cutoffs=sorted(set(pd.concat([t.date for t in tasks.values()])))
    for t in cutoffs:
        graph=m.graph_snapshot(tables,metadata,t);stamp=t.value;edge_count=0;digest=hashlib.sha256()
        for name,actual_nodes in graph['nodes'].items():
            pk=metadata[name]['pkey_col']
            expected_nodes=np.asarray([row[0] for row in con.execute(f'SELECT "{pk}" FROM "{name}" WHERE clock<=? ORDER BY "{pk}"',(stamp,))],dtype=np.int64)
            np.testing.assert_array_equal(np.sort(actual_nodes),expected_nodes)
        for (name,fk,parent),actual in graph['edges'].items():
            pk=metadata[name]['pkey_col'];parent_pk=metadata[parent]['pkey_col']
            sql=f'SELECT a."{pk}",b."{parent_pk}" FROM "{name}" a JOIN "{parent}" b ON a."{fk}"=b."{parent_pk}" WHERE a.clock<=? AND b.clock<=? ORDER BY a."{pk}",b."{parent_pk}"'
            expected=np.asarray(con.execute(sql,(stamp,stamp)).fetchall(),dtype=np.int64).reshape(-1,2)
            np.testing.assert_array_equal(actual,expected);digest.update(actual.tobytes());edge_count+=len(actual);comparisons+=1
        rows.append({'cutoff':str(t),'nodes':sum(len(a) for a in graph['nodes'].values()),'edges':edge_count,'dropped_dangling':graph['dropped_dangling'],'edge_sha256':digest.hexdigest()})
    con.close();out={'status':'PASS','query_cutoffs':len(cutoffs),'relation_comparisons':comparisons,'node_table_comparisons':len(cutoffs)*len(tables),'seconds':time.perf_counter()-start,'clock_policy':'Release event-date filter <= cutoff; undated tables assumed always eligible. Real availability/version history unavailable.','snapshots':rows}
    (P/'evidence/l109/graph-census.json').write_text(json.dumps(out,indent=2)+'\n');print({k:v for k,v in out.items() if k!='snapshots'})
if __name__=='__main__':audit()
