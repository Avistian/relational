"""Run the unmodified original native sampler, without a model or target fitting."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import ml_dtypes  # Registers NumPy bfloat16 for the original Rust bridge.
from zero_shot_l175 import audit_context

def run(out):
 from huggingface_hub import snapshot_download
 from rustler import Sampler
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 start=time.monotonic();rev='e8b48dc2cfb0a3c9171a8fddaaef14b6240f18ee'
 folder=Path.home()/'scratch/pre'
 snapshot_download('hvag976/relational-transformer',repo_type='dataset',revision=rev,allow_patterns=['rel-f1/*'],local_dir=folder)
 db=folder/'rel-f1';meta=json.loads((db/'table_info.json').read_text());cols=json.loads((db/'column_index.json').read_text())
 (out/'data-manifest.json').write_text(json.dumps({p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in db.iterdir() if p.is_file()},indent=2))
 target=cols['did_not_finish of driver-dnf'];offset=meta['driver-dnf:Test']['node_idx_offset'];n=meta['driver-dnf:Test']['num_nodes'];assert n==702
 records=[];witnesses=[];arrays=[]
 for seed in [0,1,2]:
  sampler=Sampler([('rel-f1',offset,n)],8,1024,0,1,256,'all-MiniLM-L12-v2',384,seed,[target],[[]]);sampler.shuffle_py(0)
  seen=[];buffer={k:[] for k in ['node_idxs','timestamps','col_name_idxs','masks','is_targets','is_padding','f2p_nbr_idxs','boolean_values']}
  for batch_idx in range(sampler.len_py()):
   batch=dict(sampler.batch_py(batch_idx));actual=int(batch['true_batch_size'])
   for k in buffer:
    a=np.asarray(batch[k]);shape=(8,1024,5) if k=='f2p_nbr_idxs' else (8,1024)
    buffer[k].append(a.reshape(shape)[:actual].copy())
  merged={k:np.concatenate(v) for k,v in buffer.items()}
  for j in range(n):
   node=merged['node_idxs'][j];ts=merged['timestamps'][j];col=merged['col_name_idxs'][j];mask=merged['masks'][j];tgt=merged['is_targets'][j];pad=merged['is_padding'][j]
   assert tgt.sum()==1
   target_node=int(node[tgt][0]);cutoff=int(ts[tgt][0]);seen.append(target_node)
   audit=audit_context(ts,pad,tgt,col==target,mask,cutoff,30*86400)
   labels=(col==target)&~mask&~pad
   test_labels=labels&(node>=offset)&(node<offset+n)
   audit['test_label_cells']=int(test_labels.sum());audit.update(seed=seed,node_idx=target_node,cutoff=cutoff)
   records.append(audit)
   bad=labels&(ts+30*86400>cutoff)
   if bad.any() and len(witnesses)<12:
    pos=int(np.flatnonzero(bad)[0]);witnesses.append(dict(seed=seed,query_node=target_node,query_cutoff=cutoff,label_node=int(node[pos]),label_cutoff=int(ts[pos]),label_available_after=int(ts[pos])+30*86400,masked=bool(mask[pos]),label_is_test=bool(test_labels[pos])))
  assert sorted(seen)==list(range(offset,offset+n))
  np.savez_compressed(out/f'contexts-{seed}.npz',**merged)
 fields=['unmasked_query_targets','same_time_labels','unavailable_labels','future_cells','unknown_time_cells','visible_label_cells','test_label_cells']
 summary={k:dict(cells=sum(r[k] for r in records),queries=sum(r[k]>0 for r in records)) for k in fields}
 result=dict(status='COMPLETE_CONTEXT_AUDIT',seeds=[0,1,2],queries=len(records),rows_per_seed=n,summary=summary,witnesses=witnesses,records=records,seconds=time.monotonic()-start)
 result['inference_gate']='BLOCKED_TEMPORAL_AUDIT' if any(summary[k]['cells'] for k in ['unmasked_query_targets','unavailable_labels','future_cells']) else 'PASS'
 (out/'context-audit.json').write_text(json.dumps(result,indent=2)+'\n');return {k:v for k,v in result.items() if k!='records'}
