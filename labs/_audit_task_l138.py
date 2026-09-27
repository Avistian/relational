"""Full task audit from narrow parquet columns; independent sorted-event checks."""
import json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score,average_precision_score

def run(root):
    start=time.perf_counter();root=Path(root);paths=list((root/'unpacked').rglob('*.parquet'))
    review=next(p for p in paths if p.stem=='review');events=pq.read_table(review,columns=['customer_id','review_time']).to_pandas()
    events=events.dropna(subset=['customer_id','review_time']);events['customer_id']=events.customer_id.astype('int64')
    day=86400*10**9;window=91*day
    dtype=np.dtype([('id','<i8'),('t','<i8')]);ordered=np.empty(len(events),dtype=dtype)
    ordered['id']=events.customer_id;ordered['t']=events.review_time.astype('int64');ordered.sort(order=['id','t'])
    def position(ids,times):
        x=np.empty(len(ids),dtype=dtype);x['id']=ids;x['t']=times;return np.searchsorted(ordered,x,side='right')
    result={'splits':{},'review_rows':len(events),'label_contract':'eligible (t-91d,t]; churn = no event (t,t+91d]','source':'current downloaded archive; compare manifest separately'}
    samples={};heldout={}
    for split in ['train','val','test']:
        p=next(p for p in paths if p.stem==split);df=pq.read_table(p).to_pandas();ids=df.customer_id.to_numpy(dtype='int64');times=df.timestamp.astype('int64').to_numpy();y=df.churn.to_numpy()
        right=position(ids,times);left=position(ids,times-window);future=position(ids,times+window)
        count=right-left;labels=(future==right).astype('int64')
        # Prevent row positions from crossing a customer's key range for recency.
        idx=np.maximum(right-1,0);same=ordered['id'][idx]==ids
        recency=np.where(same,(times-ordered['t'][idx])/day,np.inf)
        assert (count>0).all(),f'{split}: ineligible queries'
        assert np.array_equal(labels,y),f'{split}: target mismatch'
        assert not df.duplicated(['customer_id','timestamp']).any()
        # Fixed course-only score: more elapsed days since a review -> higher churn.
        auc=roc_auc_score(y,recency);ap=average_precision_score(y,recency)
        result['splits'][split]=dict(rows=len(df),customers=len(np.unique(ids)),timestamps=[str(x) for x in sorted(df.timestamp.unique())],positive_rate=float(y.mean()),recency_auc=float(auc),recency_ap=float(ap),label_mismatches=int(np.sum(labels!=y)),ineligible=int(np.sum(count==0)))
        rng=np.random.default_rng(138);ix=np.sort(rng.choice(len(df),min(256,len(df)),replace=False));s=[]
        for i in ix[:12]:
            # Store only relative event days, avoiding names or review text.
            ev=events.loc[events.customer_id==ids[i],'review_time'].astype('int64').to_numpy();relative=(ev-times[i])/day
            s.append(dict(customer=int(ids[i]),cutoff=str(df.timestamp.iloc[i]),events=relative.tolist(),target=int(y[i]),recent_count=int(count[i]),recency=float(recency[i])))
        samples[split]=s
        if split!='train':heldout.update({split+'_target':y,split+'_score':recency,split+'_customer':ids,split+'_time':times,split+'_count':count})
        print(split,result['splits'][split],flush=True)
    result['seconds']=time.perf_counter()-start;result['status']='PASS'
    (root/'task_audit.json').write_text(json.dumps(result,indent=2));(root/'samples.json').write_text(json.dumps(samples,indent=2));np.savez_compressed(root/'recency_predictions.npz',**heldout)
    return result
