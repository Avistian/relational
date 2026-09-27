"""Complete released RDL classification lane. Full data/text; no teaching caps.

Requires requirements-l117-runtime.txt plus torch2.5.1 and matching pyg-lib.
Preprocessing and model source are visible in the companion notebook.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_frame.config import TextEmbedderConfig
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.base import Table
from relbench.modeling.utils import get_stype_proposal
from relkit.rdl_l117 import Model,make_pkey_fkey_graph,get_node_train_table_input

def full_run(output,seed=0,epochs=10,cache='/evidence/full-cache',prepared_root=None):
    from sentence_transformers import SentenceTransformer
    start=time.perf_counter();torch.set_num_threads(1);seed_everything(seed)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dataset=get_dataset('rel-amazon',download=prepared_root is None);task=get_task('rel-amazon','user-churn',download=prepared_root is None)
    if prepared_root is not None:
        prepared_root=Path(prepared_root)
        dataset.cache_dir=str(prepared_root/'unpacked');task.cache_dir=str(prepared_root/'unpacked/user-churn')
    manifest={}
    for name,expected in [('db.zip','db71c7701b892a4eb7481ff04d14d25465795501dba3a5931aabee9930805efe'),('tasks/user-churn.zip','cddd511a1f609aea286cf2e20803f489ab9b3158e18cea66e0888d174b1515f0')]:
        h=hashlib.sha256()
        archive=(prepared_root/'cache'/name) if prepared_root is not None else (Path(dataset.cache_dir)/name)
        with archive.open('rb') as f:
            while chunk:=f.read(8*1024*1024):h.update(chunk)
        manifest[name]=dict(sha256=h.hexdigest(),historical=expected,match=h.hexdigest()==expected)
    (output/'data_identity.json').write_text(json.dumps(manifest,indent=2))
    if not all(v['match'] for v in manifest.values()):raise RuntimeError('Archive differs from pinned release; audit a separate replay explicitly')
    if prepared_root is not None:
        data,stats=torch.load(prepared_root/'graph.pt',weights_only=False)
    else:
        db=dataset.get_db();types=get_stype_proposal(db)
        text=SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d',revision='e5e8fec6971be8960cfaa853a77a6ddc62a265d7',device=device)
        def embed(strings):return torch.from_numpy(text.encode(strings,show_progress_bar=False))
        data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256),cache_dir=cache)
        del text,db
    if device=='cuda':torch.cuda.empty_cache()
    # Loading the cached graph makes raw review strings redundant. Read the
    # identical task parquet tables directly to avoid get_table's full-db load.
    tables={}
    for split in ['train','val','test']:
        if prepared_root is None:
            tables[split]=task.get_table(split,mask_input_cols=False)
        else:
            tables[split]=Table.load(prepared_root/'unpacked/user-churn'/f'{split}.parquet')
            ids=tables[split].df[task.entity_col].to_numpy()
            assert ((ids>=0)&(ids<data[task.entity_table].tf.num_rows)).all(), 'Dangling task identity'
    loaders={}
    for split in ['train','val','test']:
        table=tables[split]
        if split=='test':
            table=Table(df=table.df.drop(columns=[task.target_col]),fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,pkey_col=table.pkey_col,time_col=table.time_col)
        q=get_node_train_table_input(table,task)
        loaders[split]=NeighborLoader(data,num_neighbors=[128,64],time_attr='time',input_nodes=q.nodes,input_time=q.time,transform=q.transform,batch_size=512,temporal_strategy='uniform',shuffle=split=='train',num_workers=0)
    model=Model(data,stats,num_layers=2,channels=128,out_channels=1,aggr='sum',norm='batch_norm').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005);loss_fn=torch.nn.BCEWithLogitsLoss()
    audit=dict(query_occurrences=0,timestamped_node_occurrences=0,future_violations=0)
    train_targets=tables['train'].df[task.target_col].to_numpy()
    def audit_batch(batch,split):
        root=batch[task.entity_table];cutoffs=root.seed_time
        audit['query_occurrences']+=len(cutoffs)
        for kind,times in batch.time_dict.items():
            assert bool((times<=cutoffs[batch[kind].batch]).all()), 'Future event entered a query'
            audit['timestamped_node_occurrences']+=len(times)
        if split=='train':
            assert np.array_equal(root.y.numpy(),train_targets[root.input_id.numpy()]), 'Target/query misalignment'
    @torch.no_grad()
    def predict(split):
        model.eval();pred=[];ids=[]
        for batch in loaders[split]:
            audit_batch(batch,split)
            batch=batch.to(device);pred.append(model(batch,task.entity_table).view(-1).sigmoid().cpu());ids.append(batch[task.entity_table].input_id.cpu())
        ids=torch.cat(ids).numpy();assert np.array_equal(ids,np.arange(len(ids)))
        return torch.cat(pred).numpy()
    history=[];best=-float('inf');selected=None
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0;steps=0;clock=time.perf_counter()
        for batch in loaders['train']:
            audit_batch(batch,'train')
            batch=batch.to(device);optimizer.zero_grad();pred=model(batch,task.entity_table).view(-1)
            loss=loss_fn(pred.float(),batch[task.entity_table].y.float());assert bool(torch.isfinite(loss)), 'Nonfinite training loss';loss.backward();optimizer.step()
            total+=float(loss.detach())*len(pred);count+=len(pred);steps+=1
            # Preserve released off-by-one: 2001 batches, not silently corrected to2000.
            if steps>2000:break
        val=predict('val');metrics=task.evaluate(val,tables['val'])
        if metrics['roc_auc']>best:best=metrics['roc_auc'];selected=copy.deepcopy(model.state_dict());best_epoch=epoch
        history.append(dict(epoch=epoch,loss=total/count,queries=count,steps=steps,val=metrics,seconds=time.perf_counter()-clock))
        (output/'progress.json').write_text(json.dumps(history,indent=2));print(history[-1],flush=True)
    model.load_state_dict(selected);torch.save(selected,output/'selected.pt');scores={};saved={}
    for split in ['val','test']:
        p=predict(split);table=tables[split];scores[split]=task.evaluate(p,table)
        saved[split+'_pred']=p;saved[split+'_target']=table.df[task.target_col].to_numpy();saved[split+'_customer']=table.df[task.entity_col].to_numpy();saved[split+'_time']=table.df[task.time_col].astype('int64').to_numpy()
    np.savez_compressed(output/'predictions.npz',**saved)
    result=dict(seed=seed,epochs=epochs,best_epoch=best_epoch,selection_auc=best,history=history,scores=scores,temporal_audit=audit,seconds=time.perf_counter()-start,status='COMPLETE',historical_identity='NOT_ESTABLISHED')
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=10);p.add_argument('--cache',default='labs/results/l138/materialized');a=p.parse_args();full_run(a.output,a.seed,a.epochs,a.cache)
