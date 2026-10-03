"""Fresh B10 replay of authenticated L175 saved contexts. No native resampling."""
import hashlib,json
from pathlib import Path
import numpy as np
import ml_dtypes
P=Path(__file__).resolve().parent;S=P/'sources/b10';E=P/'evidence/b10'

def audit():
    ledger=json.loads((S/'source-ledger.json').read_text())
    for name,digest in ledger['inherited_contexts'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
    meta=json.loads((S/'table_info.json').read_text());cols=json.loads((S/'column_index.json').read_text())
    truth=np.load(P/'evidence/l175/label-oracle.npz');target=cols['did_not_finish of driver-dnf'];rows=[];witnesses=[];reference=None
    for seed in range(3):
      a=np.load(P/f'evidence/l175/audit-3/contexts-{seed}.npz');assert a['node_idxs'].shape==(702,1024)
      summary=dict(seed=seed,queries=702,slots=702*1024,future_cells=0,future_contexts=0,exposed_query_targets=0,unfinished_labels=0,unknown_time_cells=0,visible_labels=0,positives=0)
      keys=[];query_nodes=[]
      for i in range(702):
        q=a['is_targets'][i];assert q.sum()==1;ts=a['timestamps'][i].astype(np.int64);pad=a['is_padding'][i];hidden=a['masks'][i];node=a['node_idxs'][i]
        cutoff=int(ts[q][0]);driver_row=int(a['f2p_nbr_idxs'][i][q][0,0])-meta['drivers:Db']['node_idx_offset'];driver=int(truth['driver_ids'][driver_row]);keys.append((driver,cutoff));query_nodes.append(int(node[q][0]))
        label=int(a['boolean_values'][i][q].view(ml_dtypes.bfloat16).astype(float)[0]>0)
        hits=(truth['driver']==driver)&(truth['time']>cutoff)&(truth['time']<=cutoff+30*86400)
        assert hits.any() and label==int((truth['status'][hits]!=1).any());summary['positives']+=label
        known=ts!=np.iinfo(np.int32).min;future=(ts>cutoff)&~pad&known
        visible=(a['col_name_idxs'][i]==target)&~pad&~hidden
        summary['future_cells']+=int(future.sum());summary['future_contexts']+=int(future.any())
        summary['exposed_query_targets']+=int((q&~hidden&~pad).sum())
        summary['unfinished_labels']+=int((visible&known&(ts+30*86400>cutoff)).sum())
        summary['unknown_time_cells']+=int((~pad&~known).sum());summary['visible_labels']+=int(visible.sum())
        if future.any() and not any(w['seed']==seed for w in witnesses):
          j=int(np.flatnonzero(future)[0]);witnesses.append(dict(seed=seed,driver=driver,cutoff=cutoff,future_time=int(ts[j]),column=next(k for k,v in cols.items() if v==int(a['col_name_idxs'][i,j]))))
      assert len(set(keys))==702 and sorted(query_nodes)==list(range(97606,98308))
      if reference is None:reference=set(keys)
      assert set(keys)==reference;rows.append(summary)
    totals={k:sum(x[k] for x in rows) for k in rows[0] if k!='seed'}
    assert (totals['future_cells'],totals['future_contexts'],totals['unknown_time_cells'])==(385,77,432050)
    assert totals['exposed_query_targets']==totals['unfinished_labels']==0
    result={'status':'COMPLETE_SAVED_CONTEXT_REPLAY','paper_gate':'INCOMPLETE_TEMPORAL_GATE','per_seed':rows,'totals':totals,'witnesses':witnesses,'benchmark_inference':'NOT_RUN','native_resampling':'NOT_RUN','checkpoint_authentication':'NOT_RUN','interpretation':'Event-time rule fails. Scheduled fields might have been available earlier; historical availability is NOT_ESTABLISHED. Unknown timestamp is not proof of future leakage.'}
    (E/'temporal-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':audit()
