"""Full rel-stack topology, bounded real-query training; COURSE_ONLY features/model."""
import gc,hashlib,json,time,zipfile,urllib.request,resource
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from relkit.rdl_l117 import HeteroGraphSAGE
from relkit.scale_l134 import inspect_batch,frontier_bound,profile_summary

def fetch(name,root):
    import relbench.datasets as datasets
    import relbench.tasks as tasks
    key='rel-stack/'+name
    historical=datasets.hashes[key] if name=='db.zip' else tasks.hashes[key]
    digest='5a97bf65a926529143e96f6413b2b0550ca55c01247f85156bfb999ee903e94e' if name=='db.zip' else historical
    path=root/Path(name).name
    prior=root.parent/'attempt-1'/Path(name).name
    if name=='db.zip' and prior.exists():path=prior
    url='https://relbench.stanford.edu/download/'+key
    if not path.exists():urllib.request.urlretrieve(url,path)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==digest, f'Archive hash {h.hexdigest()} != {digest}'
    return path,dict(url=url,sha256=digest,historical_registry_sha256=historical,historical_match=digest==historical,bytes=path.stat().st_size)

def run(output):
    torch.set_num_threads(2);seed_everything(134);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();archive,db_source=fetch('db.zip',output)
    # Read only structure/time columns; never materialize millions of text strings.
    tables={};metadata={};raw_rows={}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('.parquet'):continue
            local=output/Path(name).name
            with z.open(name) as src,local.open('wb') as dst:
                import shutil;shutil.copyfileobj(src,dst,8*1024*1024)
            pf=pq.ParquetFile(local);meta={k.decode():json.loads(v) for k,v in pf.schema_arrow.metadata.items() if k in [b'pkey_col',b'fkey_col_to_pkey_table',b'time_col']}
            cols=list(dict.fromkeys([c for c in [meta['pkey_col'],meta['time_col'],*meta['fkey_col_to_pkey_table']] if c]))
            df=pf.read(columns=cols,use_threads=False).to_pandas();kind=local.stem;raw_rows[kind]=len(df)
            if meta['time_col']:df=df[df[meta['time_col']]<=pd.Timestamp('2021-01-01')].copy()
            tables[kind]=df;metadata[kind]=meta;local.unlink()
    # Match the released Dataset correction after the test-time truncation:
    # references to removed future parent rows become null, never clipped/reindexed.
    from relbench.base import Database,Table
    from relbench.datasets import get_dataset
    projected=Database({k:Table(df=df,**metadata[k]) for k,df in tables.items()})
    before_nulls={k:{fk:int(df[fk].isna().sum()) for fk in metadata[k]['fkey_col_to_pkey_table']} for k,df in tables.items()}
    get_dataset('rel-stack').validate_and_correct_db(projected)
    tables={k:t.df for k,t in projected.table_dict.items()}
    corrected={k:{fk:int(df[fk].isna().sum())-before_nulls[k][fk] for fk in metadata[k]['fkey_col_to_pkey_table']} for k,df in tables.items()}
    graph=HeteroData();census={};sql_edges={} 
    import duckdb
    connection=duckdb.connect();connection.execute('SET threads=2')
    for kind,df in tables.items():
        m=metadata[kind];assert np.array_equal(df[m['pkey_col']].to_numpy(),np.arange(len(df)))
        graph[kind].num_nodes=len(df);graph[kind].x=torch.ones((len(df),1),dtype=torch.float32)
        if m['time_col']:graph[kind].time=torch.from_numpy(df[m['time_col']].astype('datetime64[s]').astype('int64').to_numpy().copy())
        connection.register(kind,df);census[kind]=len(df)
    for kind,df in tables.items():
        for fk,dst in metadata[kind]['fkey_col_to_pkey_table'].items():
            mask=df[fk].notna();src=np.flatnonzero(mask);dest=df.loc[mask,fk].to_numpy(dtype=np.int64)
            assert np.all(dest>=0) and np.all(dest<len(tables[dst]))
            edge=torch.from_numpy(np.stack([src,dest]));name=(kind,'f2p_'+fk,dst)
            graph[name].edge_index=edge;graph[(dst,'rev_f2p_'+fk,kind)].edge_index=edge.flip(0)
            pk=metadata[dst]['pkey_col']
            sql=connection.execute(f'SELECT COUNT(*) FROM "{kind}" a JOIN "{dst}" b ON a."{fk}"=b."{pk}"').fetchone()[0]
            assert sql==edge.shape[1];sql_edges['|'.join(name)]=sql
    connection.close();gc.collect();assert sum(census.values())>=1_000_000
    task_archive,task_source=fetch('tasks/user-engagement.zip',output)
    with zipfile.ZipFile(task_archive) as z:
        with z.open('user-engagement/train.parquet') as f:
            table=pq.read_table(f,use_threads=False);meta=table.schema.metadata;train=table.to_pandas()
    from relbench.tasks import get_task
    task=get_task('rel-stack','user-engagement',download=False)
    # Reconstruct labels against THIS pinned database, avoiding any assumed key
    # equivalence between the changed archive and historically released task rows.
    time_col=task.time_col;entity=task.entity_table
    query_time=train[time_col].min()
    original_prefix=train.iloc[:2048]
    original_ids=original_prefix[task.entity_col].to_numpy(dtype=np.int64)
    original_secs=original_prefix[time_col].astype('datetime64[s]').astype('int64').to_numpy()
    archived_root_violations=int((graph[entity].time[original_ids].numpy()>original_secs).sum())
    rebuilt=task.make_table(projected,pd.Series([query_time])).df
    # Independently reconstruct active and future user sets from source rows.
    past=set();future=set()
    for kind,fk in [('posts','OwnerUserId'),('votes','UserId'),('comments','UserId')]:
        df=tables[kind];stamp=df['CreationDate'];valid=df[fk].notna()
        past.update(df.loc[valid&(stamp<=query_time),fk].astype(int).tolist())
        future.update(df.loc[valid&(stamp>query_time)&(stamp<=query_time+task.timedelta),fk].astype(int).tolist())
    assert set(rebuilt[task.entity_col])==past-{-1}
    assert all(int(y)==int(int(u) in future) for u,y in zip(rebuilt[task.entity_col],rebuilt[task.target_col]))
    eligibility=graph[entity].time[torch.from_numpy(rebuilt[task.entity_col].to_numpy(dtype=np.int64))].numpy()<=int(query_time.timestamp())
    excluded=int((~eligibility).sum())
    train=rebuilt[eligibility].sort_values(task.entity_col).iloc[:2048].copy()
    assert len(train)==2048
    query_contract=dict(status='PASS',origin='Original task SQL reconstructed on pinned current database; independent event-set parity for full one-cutoff cohort',cutoff=str(query_time),horizon_days=task.timedelta.days,eligible_cohort=len(rebuilt),excluded_future_created_roots=excluded,archived_prefix_root_violations=archived_root_violations,selection='Created by cutoff; ascending entity ID; first2048; labels not inspected for selection',historical_query_identity='NOT_ESTABLISHED')
    (output/'queries.json').write_text(train.to_json(orient='records',date_format='iso'))
    nodes=torch.from_numpy(train[task.entity_col].to_numpy(dtype=np.int64).copy())
    cutoff=torch.from_numpy(train[task.time_col].astype('datetime64[s]').astype('int64').to_numpy().copy())
    target=torch.from_numpy(train[task.target_col].to_numpy(dtype=np.float32).copy())
    assert (graph[entity].time[nodes]<=cutoff).all(), 'Task root predates entity creation'
    assert ((cutoff>0)&(cutoff<2_000_000_000)).all(), 'Declared UNIX seconds range'
    del tables,projected,rebuilt;gc.collect()
    query_hash=hashlib.sha256(nodes.numpy().tobytes()+cutoff.numpy().tobytes()).hexdigest()
    graph.validate();schema=list(graph.edge_types)
    prep=time.perf_counter()-start
    report=dict(status='RUNNING',scope='COURSE_ONLY full topology; constant+age features, width32, real training-query prefix; no paper metric',database=db_source,task=task_source,raw_rows=raw_rows,corrected_future_foreign_keys=corrected,rows=census,sql_forward_edges=sql_edges,directed_edges=sum(v.shape[1] for v in graph.edge_index_dict.values()),query_count=len(nodes),query_contract=query_contract,query_sha256=query_hash,time_unit='UNIX seconds; explicit datetime64[s] conversion',query_cutoff_range=[int(cutoff.min()),int(cutoff.max())],preprocessing_s=prep,configurations=[],static_time_tables=[k for k in graph.node_types if 'time' not in graph[k]])
    (output/'scale.json').write_text(json.dumps(report,indent=2))
    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__();self.enc=torch.nn.ModuleDict({k:torch.nn.Linear(2,32) for k in graph.node_types});self.gnn=HeteroGraphSAGE(graph.node_types,graph.edge_types,32,'sum',2);self.head=torch.nn.Linear(32,1)
        def forward(self,b):
            x={}
            for k in b.node_types:
                age=(b[entity].seed_time[b[k].batch]-b[k].time).float()/86400 if 'time' in b[k] else torch.zeros(len(b[k].x),device='cuda')
                x[k]=self.enc[k](torch.cat([b[k].x,torch.log1p(age).unsqueeze(1)],dim=1))
            return self.head(self.gnn(x,b.edge_index_dict)[entity][:b[entity].batch_size]).view(-1)
    configs=[dict(batch_size=64,fanouts=[8,4]),dict(batch_size=64,fanouts=[16,8]),dict(batch_size=128,fanouts=[16,8])]
    for config in configs:
        seed_everything(134);torch.cuda.empty_cache();model=Predictor().cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        begin=time.perf_counter();loader=NeighborLoader(graph,num_neighbors=config['fanouts'],input_nodes=(entity,nodes),input_time=cutoff,time_attr='time',temporal_strategy='uniform',batch_size=config['batch_size'],shuffle=False,num_workers=0)
        build_s=time.perf_counter()-begin
        records=[];warmups=[]
        for phase in ['warmup','measured']:
            iterator=iter(loader);limit=2 if phase=='warmup' else len(loader)
            for j in range(limit):
                torch.cuda.synchronize();begin=time.perf_counter();batch=next(iterator);sample=time.perf_counter()-begin
                begin=time.perf_counter();inspect_batch(batch,graph,entity);audit=time.perf_counter()-begin
                n=batch[entity].batch_size;index=batch[entity].input_id.clone()
                actual_nodes=sum(batch.num_nodes_dict.values());actual_edges=sum(e.shape[1] for e in batch.edge_index_dict.values())
                bound=frontier_bound(schema,entity,n,config['fanouts']);assert actual_nodes<=bound['node_occurrences'] and actual_edges<=bound['edge_occurrences']
                torch.cuda.reset_peak_memory_stats();begin=time.perf_counter();batch=batch.to('cuda');y=target[index].cuda();torch.cuda.synchronize();transfer=time.perf_counter()-begin
                begin=time.perf_counter();optimizer.zero_grad(set_to_none=True);pred=model(batch);loss=torch.nn.functional.binary_cross_entropy_with_logits(pred,y);assert torch.isfinite(loss);loss.backward();optimizer.step();torch.cuda.synchronize();step=time.perf_counter()-begin
                rec=dict(queries=n,sample_s=sample,transfer_s=transfer,step_s=step,audit_s=audit,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),nodes=actual_nodes,edges=actual_edges,bound=bound['node_occurrences'],loss=float(loss.detach()),query_indices=index.tolist())
                (warmups if phase=='warmup' else records).append(rec);del batch,pred,loss,y
        assert [i for r in records for i in r['query_indices']]==list(range(2048))
        summary=profile_summary(records);report['configurations'].append(dict(**config,loader_build_s=build_s,warmup=warmups,batches=records,summary=summary))
        (output/'scale.json').write_text(json.dumps(report,indent=2));print(config,summary,flush=True)
        del model,optimizer,loader;gc.collect()
    report.update(status='COMPLETE',gpu=torch.cuda.get_device_name(),total_s=time.perf_counter()-start,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'scale.json').write_text(json.dumps(report,indent=2));return report
