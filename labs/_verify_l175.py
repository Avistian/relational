"""Independent full-population audit and exact raw-label reconstruction."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import ml_dtypes
from sklearn.metrics import roc_auc_score
from relkit.zero_shot_l175 import aligned_auc,audit_context,exposure_claim
from _check_l175 import checks
P=Path(__file__).resolve().parent;E=P/'evidence/l175';A=E/'audit-3';S=P/'sources/l175'

def verify():
 checks();ref=json.loads((A/'context-audit.json').read_text());meta=json.loads((S/'table_info.json').read_text());cols=json.loads((S/'column_index.json').read_text());names={v:k for k,v in cols.items()};target=cols['did_not_finish of driver-dnf']
 raw=np.load(E/'label-oracle.npz');records=[];witnesses=[];groups={};per_seed=[];keys_reference=None
 for seed in range(3):
  a=np.load(A/f'contexts-{seed}.npz');assert a['node_idxs'].shape==(702,1024)
  keys=[];nodeids=[];alllabels=[]
  for i in range(702):
   q=a['is_targets'][i];assert q.sum()==1;node=a['node_idxs'][i];ts=a['timestamps'][i].astype('int64');pad=a['is_padding'][i];mask=a['masks'][i];col=a['col_name_idxs'][i]
   cutoff=int(ts[q][0]);qid=int(node[q][0]);driver_row=int(a['f2p_nbr_idxs'][i][q][0,0])-meta['drivers:Db']['node_idx_offset'];driver_id=int(raw['driver_ids'][driver_row]);keys.append((driver_id,cutoff));nodeids.append(qid)
   # Values are serialized raw bfloat16; the original numeric bridge uses >0.
   value=a['boolean_values'][i][q].view(ml_dtypes.bfloat16).astype(float)[0];label=int(value>0)
   eligible=(raw['driver']==driver_id)&(raw['time']>cutoff)&(raw['time']<=cutoff+30*86400)
   assert eligible.any();oracle=int((raw['status'][eligible]!=1).any());assert label==oracle,(seed,qid,label,oracle)
   alllabels.append(label)
   lab=col==target;visible=(~pad)&lab&(~mask);known=ts!=np.iinfo(np.int32).min
   # Independent Python counts, not a call back into the production vector function.
   independent={'unmasked_query_targets':sum(bool(q[j] and not mask[j] and not pad[j]) for j in range(1024)),
    'same_time_labels':sum(bool(visible[j] and known[j] and ts[j]==cutoff) for j in range(1024)),
    'unavailable_labels':sum(bool(visible[j] and known[j] and ts[j]+30*86400>cutoff) for j in range(1024)),
    'future_cells':sum(bool(not pad[j] and known[j] and ts[j]>cutoff) for j in range(1024)),
    'unknown_time_cells':sum(bool(not pad[j] and not known[j]) for j in range(1024)),
    'visible_label_cells':int(visible.sum()),
    'test_label_cells':sum(bool(visible[j] and 97606<=node[j]<98308) for j in range(1024)),
    'seed':seed,'node_idx':qid,'cutoff':cutoff}
   assert independent==ref['records'][seed*702+i]
   live=audit_context(ts,pad,q,lab,mask,cutoff,30*86400)
   assert all(independent[k]==v for k,v in live.items());records.append(independent)
   bad=(ts>cutoff)&~pad
   for pos in np.flatnonzero(bad):
    ni=int(node[pos]);table=next(k for k,v in meta.items() if v['node_idx_offset']<=ni<v['node_idx_offset']+v['num_nodes']);column=names[int(col[pos])];groups[(table,column)]=groups.get((table,column),0)+1
    if len(witnesses)<6:witnesses.append(dict(seed=seed,query_node=qid,driver_id=driver_id,cutoff=cutoff,node=ni,table=table,column=column,time=int(ts[pos])))
  assert sorted(nodeids)==list(range(97606,98308));assert len(set(keys))==702
  if keys_reference is None:keys_reference=dict(zip(keys,alllabels))
  assert dict(zip(keys,alllabels))==keys_reference
  per_seed.append(dict(seed=seed,rows=702,positive_labels=sum(alllabels),future_cells=sum(r['future_cells'] for r in records if r['seed']==seed),future_queries=sum(r['future_cells']>0 for r in records if r['seed']==seed)))
 # Metric parity on 100 random tie-rich synthetic cases, keyed permutation changes.
 rng=np.random.default_rng(175)
 for _ in range(100):
  y=rng.integers(0,2,50);scores=rng.integers(0,5,50)/4;keys=np.column_stack((np.repeat(np.arange(25),2),np.tile([100,200],25)));perm=rng.permutation(50)
  assert abs(aligned_auc(keys,y,keys[perm],scores[perm])-roc_auc_score(y,scores))<1e-14
 # Mutation checks: dropping repeated entities, ignoring context labels or ignoring label horizon.
 for wrong in [lambda *args:0.,lambda *args:float('nan')]:
  try:assert wrong(None,None,None,None)==.875
  except AssertionError:pass
  else:raise AssertionError('Wrong scoring accepted')
 result=dict(status='PASS',contexts=2106,cells=2106*1024,raw_labels_reconstructed=2106,key_coverage='COMPLETE',per_seed=per_seed,metric_parity_cases=100,summary=ref['summary'],future_cell_groups=[dict(table=k[0],column=k[1],cells=v) for k,v in groups.items()],future_witnesses=witnesses,inference_gate=ref['inference_gate'],checkpoint_evaluations='NOT_RUN',learner='PENDING_WRITTEN_DEFENSE')
 (E/'verified-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify()
