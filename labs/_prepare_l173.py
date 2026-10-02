"""Authenticate complete F1 snapshot, fit training-only encoders, enumerate targets."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from relkit.tokenization_l172 import fit_column,encode_column
from relkit.multitask_l173 import erase_target
P=Path(__file__).resolve().parent;E=P/'evidence/l173';E.mkdir(exist_ok=True,parents=True)
prior=json.loads((P/'evidence/l172/input-manifest.json').read_text())
inputs={k:v for k,v in prior['files'].items() if k.startswith('evidence/')}
for n,h in inputs.items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h,n
schema=json.loads((P/'evidence/l172/schema.json').read_text());frames={n:pd.read_parquet(P/'evidence/l171/db'/f'{n}.parquet').reset_index(drop=True) for n in schema}
tasks=[];lookups={};cells={};excluded=[];records=[]
columns=[0];numeric=[0.];categories=[0];target=[0.];splits=[-1];dates=[-1];rowids=[-1];offset=1
for table,spec in sorted(schema.items()):
    frame=frames[table]
    if spec['time_col'] is None:
        excluded.append(dict(table=table,reason='UNTIMED_TABLE',rows=len(frame),cells=int(frame.size)));continue
    time=pd.to_datetime(frame[spec['time_col']]);admitted=(time<pd.Timestamp('2005-01-01')).to_numpy()
    pk=next(c for c,s in spec['columns'].items() if s['role']=='primary_key')
    lookups[table]={str(v):i for i,v in enumerate(frame[pk])}
    for column,desc in sorted(spec['columns'].items()):
        if desc['kind'] not in {'number','category'} or desc['role']!='feature':
            excluded.append(dict(table=table,column=column,reason='NON_TARGET_ROLE_OR_TYPE',cells=len(frame)));continue
        values=frame[column];fitted=fit_column(values,desc['kind'],admitted)
        if fitted['fit_nonnull']==0:raise ValueError('No training observations: '+table+'.'+column)
        enc=encode_column(values,fitted);valid=values.notna().to_numpy() & time.notna().to_numpy()
        ids=np.zeros(len(frame),dtype=np.int64);rows=np.flatnonzero(valid)
        ids[rows]=np.arange(len(columns),len(columns)+len(rows));cells[table,column]=ids
        ti=len(tasks);classes=len(fitted.get('vocabulary',[]))+1 if desc['kind']=='category' else 1
        tasks.append(dict(name=table+'.'+column,table=table,column=column,classes=classes,category_offset=offset if desc['kind']=='category' else 0,**fitted))
        y=np.asarray(enc['payload'],dtype=np.float32)[rows]
        columns.extend([ti+1]*len(rows));numeric.extend(y if desc['kind']=='number' else np.zeros(len(rows)))
        categories.extend(y.astype(int)+offset if desc['kind']=='category' else np.zeros(len(rows),dtype=int))
        target.extend(y);splits.extend(np.where(time.iloc[rows]<pd.Timestamp('2005-01-01'),0,np.where(time.iloc[rows]<pd.Timestamp('2006-01-01'),1,2)))
        dates.extend(time.iloc[rows].to_numpy(dtype='datetime64[s]').astype(np.int64));rowids.extend(rows)
        if desc['kind']=='category':offset+=classes
        excluded.append(dict(table=table,column=column,reason='NULL_TARGET_OR_TIME',cells=int((~valid).sum())))
        records.append((table,column,rows,ids[rows],ti))
context=np.zeros((len(columns)-1,8),dtype=np.int64)
for table,column,rows,ids,ti in records:
    frame=frames[table];spec=schema[table];cols=sorted(c for t,c in cells if t==table and c!=column)
    for row,cid in zip(rows,ids):
        own=[int(cells[table,c][row]) for c in cols if cells[table,c][row]!=0][:4]
        linked=[]
        for fk,desc in sorted(spec['columns'].items()):
            if desc['role']!='foreign_key' or desc['target_table'] not in lookups or pd.isna(frame.at[row,fk]):continue
            parent=desc['target_table'];pr=lookups[parent].get(str(frame.at[row,fk]))
            if pr is None:raise ValueError('Dangling reference')
            pt=frames[parent].at[pr,schema[parent]['time_col']]
            if pd.isna(pt) or pt>frame.at[row,spec['time_col']]:continue
            for t,c in sorted(cells):
                if t==parent and cells[t,c][pr]!=0:linked.append(int(cells[t,c][pr]))
        context[cid-1,:len(own)]=own
        context[cid-1,4:4+min(4,len(linked))]=linked[:4]
ids=np.arange(1,len(columns),dtype=np.int64);context=erase_target(context,ids)
arrays=dict(context=context,cell_ids=ids,task=np.asarray(columns[1:],dtype=np.int64)-1,target=np.asarray(target[1:],dtype=np.float32),split=np.asarray(splits[1:],dtype=np.int64),columns=np.asarray(columns,dtype=np.int64),numeric=np.asarray(numeric,dtype=np.float32),categories=np.asarray(categories,dtype=np.int64),dates=np.asarray(dates,dtype=np.int64),rowids=np.asarray(rowids,dtype=np.int64))
for i,t in enumerate(tasks):
    t['counts']=[int(((arrays['task']==i)&(arrays['split']==s)).sum()) for s in range(3)]
    if min(t['counts'])==0:raise ValueError('Task missing split: '+t['name'])
np.savez_compressed(E/'population.npz',**arrays)
(E/'tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')
config=dict(name='L173 F1 Multi-task Masked-cell Pretraining',seeds=[0,1,2],arms=['cell','task'],epochs=3,batch_size=1024,optimizer='Adam',learning_rate=.001,weight_decay=0,context_slots=dict(row=4,parent=4),token_width=32,hidden_width=64,selection='earliest_minimum_validation_macro_loss',train_before='2005-01-01',validation_before='2006-01-01',test_from='2006-01-01',unknown_category_class=0)
(E/'config.json').write_text(json.dumps(config,indent=2)+'\n')
coverage=dict(rows=sum(len(f) for f in frames.values()),tables=len(frames),targets=len(ids),tasks=len(tasks),splits=[int((arrays['split']==s).sum()) for s in range(3)],row_context_cells=int((context[:,:4]!=0).sum()),parent_context_cells=int((context[:,4:]!=0).sum()),excluded=excluded)
(E/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
manifest=dict(inputs=inputs,files={n:hashlib.sha256((E/n).read_bytes()).hexdigest() for n in ['population.npz','tasks.json','config.json','coverage.json']},source_files={})
for name in ['relkit/multitask_l173.py','relkit/tokenization_l172.py','_prepare_l173.py']:
    manifest['source_files'][name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
(E/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in coverage.items() if k!='excluded'})
