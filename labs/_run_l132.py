"""Execute the pinned full RelBench link scripts with explicit evidence-only instrumentation.
Original source is retained. No objective, optimizer, sampling or selection algorithm is replaced.
"""
import argparse,hashlib,json,os,sys,time,copy
from pathlib import Path
P=Path(__file__).resolve().parent

def prepare(root):
    import torch
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task
    from relbench.modeling.utils import get_stype_proposal
    from relbench.modeling.graph import make_pkey_fkey_graph
    from torch_frame.config import TextEmbedderConfig
    from torch_geometric.seed import seed_everything
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1);seed_everything(42);start=time.perf_counter()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    dataset=get_dataset('rel-trial',download=True);task=get_task('rel-trial','condition-sponsor-run',download=True)
    hashes={}
    for path,expected in [('db.zip','9fb5ba14f7cbca8115f3dfe0800415f98d6ddc15561e56c35ee614da6b89552a'),('tasks/condition-sponsor-run.zip','eeef170e06b6728928116c333f56601e036b24fcbef4f97dc93ec1948700c2f8')]:
        p=Path(dataset.cache_dir)/path;digest=hashlib.sha256(p.read_bytes()).hexdigest();assert digest==expected;hashes[path]=digest
    db=dataset.get_db();types=get_stype_proposal(db)
    spec=json.loads((P/'sources/l117/text_model.json').read_text())
    emb=SentenceTransformer(spec['model'],revision=spec['revision'],device='cuda' if torch.cuda.is_available() else 'cpu')
    def embed(strings):return torch.from_numpy(emb.encode(strings,show_progress_bar=False))
    data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256),cache_dir=str(root/'materialized'))
    torch.save((data,stats),root/'graph.pt')
    (root/'stypes.json').write_text(json.dumps(types,default=str))
    tables={s:task.get_table(s,mask_input_cols=False).df for s in ['train','val','test']}
    for s,df in tables.items():df.to_parquet(root/f'{s}.parquet')
    report=dict(status='COMPLETE',archive_hashes=hashes,rows={k:len(v) for k,v in db.table_dict.items()},edges={'|'.join(k):v.shape[1] for k,v in data.edge_index_dict.items()},queries={k:len(v) for k,v in tables.items()},num_dst=task.num_dst_nodes,eval_k=task.eval_k,source_col=task.src_entity_col,destination_col=task.dst_entity_col,time_col=task.time_col,source_type=task.src_entity_table,destination_type=task.dst_entity_table,seconds=time.perf_counter()-start,text_model=spec,preprocessing_seed=42)
    (root/'prepared.json').write_text(json.dumps(report,indent=2));return report

def instrument_source(source):
    """Only redirect cached preprocessing and add per-epoch evidence / selected epoch."""
    edits=[('text_embedder=GloveTextEmbedding(device=device)','text_embedder=None'),
      ('state_dict = None\nbest_val_metric = 0','state_dict = None\nbest_val_metric = 0\nselected_epoch = None\ntrace = []'),
      ('    train_loss = train()','    epoch_start = time.perf_counter()\n    train_loss = train()'),
      ('            state_dict = copy.deepcopy(model.state_dict())','            state_dict = copy.deepcopy(model.state_dict())\n            selected_epoch = epoch'),
      ('\n\nmodel.load_state_dict(state_dict)','\n    trace.append(dict(epoch=epoch,train_loss=train_loss,val=val_metrics[tune_metric],seconds=time.perf_counter()-epoch_start))\n    output.mkdir(parents=True,exist_ok=True)\n    (output/"progress.json").write_text(json.dumps(trace,indent=2))\n\nmodel.load_state_dict(state_dict)')]
    for old,new in edits:
        assert source.count(old)==1,old
        source=source.replace(old,new)
    return source

def run(variant,seed,epochs,root,output):
    import torch,numpy as np,pandas as pd
    from relkit.identity_l132 import mean_average_precision
    import relbench.modeling.graph as graph
    torch.set_num_threads(1);root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    assert not (output/'result.json').exists(),'Never overwrite a completed run'
    start=time.perf_counter();data,stats=torch.load(root/'graph.pt',weights_only=False)
    source_name='idgnn_link.py' if variant=='idgnn' else 'gnn_link.py'
    original=(P/'sources/l132'/source_name).read_text();source=instrument_source(original)
    (output/'executed.py').write_text(source)
    # Pin installed primitives before using the original Model.
    import relbench.modeling.nn as nn,relbench.modeling.loader as loader
    for name,module in [('nn.py',nn),('loader.py',loader),('graph.py',graph)]:
        assert Path(module.__file__).read_bytes()==(P/'sources/l132'/name).read_bytes(),name
    cache=root.parent/'examples';folder=cache/'rel-trial';folder.mkdir(parents=True,exist_ok=True)
    (folder/'stypes.json').write_bytes((root/'stypes.json').read_bytes())
    old_graph=graph.make_pkey_fkey_graph;graph.make_pkey_fkey_graph=lambda *a,**k:(data,stats)
    argv=sys.argv[:];sys.path.insert(0,str(P/'sources/l132'))
    sys.argv=[source_name,'--dataset','rel-trial','--task','condition-sponsor-run','--seed',str(seed),'--epochs',str(epochs),'--num_layers',str(4 if variant=='idgnn' else 2),'--temporal_strategy','uniform','--cache_dir',str(cache)]
    env={'__name__':'l132_run','time':time,'output':output}
    try:exec(compile(source,source_name,'exec'),env)
    finally:graph.make_pkey_fkey_graph=old_graph;sys.argv=argv
    task=env['task'];result=dict(variant=variant,seed=seed,epochs=epochs,selected_epoch=env['selected_epoch'],trace=env['trace'],scores={},source_sha256=hashlib.sha256(original.encode()).hexdigest(),executed_sha256=hashlib.sha256(source.encode()).hexdigest())
    arrays={}
    for split in ['val','test']:
        df=task.get_table(split,mask_input_cols=False).df
        pred=env[split+'_pred'];truth=df[task.dst_entity_col].tolist()
        metric=mean_average_precision(pred,truth,task.eval_k)
        official=float(env[split+'_metrics']['link_prediction_map']);assert abs(metric-official)<1e-10
        result['scores'][split]=metric
        arrays[split+'_pred']=pred;arrays[split+'_source']=df[task.src_entity_col].to_numpy();arrays[split+'_time']=df[task.time_col].astype('int64').to_numpy()
    np.savez_compressed(output/'predictions.npz',**arrays)
    torch.save(env['model'].state_dict(),output/'checkpoint.pt')
    result['seconds']=time.perf_counter()-start
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--root',default='labs/results/l132/prepared');p.add_argument('--output',default='labs/results/l132/local');p.add_argument('--variant',choices=['sage','idgnn'],default='idgnn');p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=20);a=p.parse_args()
    print(prepare(a.root) if a.prepare else run(a.variant,a.seed,a.epochs,a.root,a.output))
