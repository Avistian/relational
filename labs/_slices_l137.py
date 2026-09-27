"""Feature-only diagnostics from frozen source tables, never future labels."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from relkit.fe_experiment_l129 import load_archives
P=Path(__file__).resolve().parent;E=P/'evidence/l137';F=E/'fe'
tables,labels,_=load_archives(P/'sources/l129');results=tables['results']
bydriver={i:g.date.astype('int64').to_numpy() for i,g in results.groupby('driverId')}
arrays={};metadata={}
for split,df in labels.items():
    features=pd.read_parquet(F/f'{split}-features.parquet').set_index(['driverId','date']).loc[pd.MultiIndex.from_frame(df[['driverId','date']])]
    counts=[];lags=[]
    for row in df.itertuples():
        dates=bydriver.get(row.driverId,np.array([],dtype='int64'));past=dates[dates<row.date.value]
        counts.append(len(past));lags.append((row.date.value-past.max())/86400e9 if len(past) else np.inf)
    cols=[c for c in features if c.startswith('past_')]
    missing=features[cols].isna().mean(axis=1).to_numpy()
    arrays.update({split+'_history':np.array(counts),split+'_recency':np.array(lags),split+'_missing':missing,split+'_entity':df.driverId.to_numpy(),split+'_time':df.date.astype('int64').to_numpy()})
threshold=float(np.median(arrays['train_history']))
for split in labels:
    h=arrays[split+'_history'];r=arrays[split+'_recency'];m=arrays[split+'_missing']
    masks={'low_history':h<=threshold,'high_history':h>threshold,'stale_history':r>180,'recent_history':r<=180,'missing_recent_slots':m>=.5,'observed_recent_slots':m<.5}
    for name,mask in masks.items():arrays[split+'_mask_'+name]=mask
    metadata[split]={name:dict(rows=int(mask.sum()),drivers=len(np.unique(arrays[split+'_entity'][mask]))) for name,mask in masks.items()}
np.savez_compressed(E/'diagnostics.npz',**arrays)
report=dict(status='PASS',history_threshold_train_median=threshold,recency_threshold_days=180,missing_threshold=.5,feature_columns=cols,counts=metadata,rule='results dated strictly before query in database snapshot through2010-01-01; missingness from released SQL recent-slot columns',availability='Event-time audit only; actual arrival histories unknown',sha256=hashlib.sha256((E/'diagnostics.npz').read_bytes()).hexdigest())
(E/'diagnostics.json').write_text(json.dumps(report,indent=2));print(report)
