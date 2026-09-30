"""Full released-checkpoint replay and explicitly reconstructed fresh training.

Reconstruction: five seeds, ten complete epochs, Adam .005, L1, first validation
minimum. These choices are frozen course choices; historical training is unreleased.
"""
import hashlib,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.seed import seed_everything
from torch_frame.config import TextEmbedderConfig
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from relbench.modeling.graph import make_pkey_fkey_graph,get_node_train_table_input
from relkit.relgnn_l143 import RelGNN_Model,get_atomic_routes,keyed_mae
from _parity_l143 import original_modules

ARCHIVES={'db.zip':'ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482','tasks/driver-position.zip':'775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e'}
CHECKPOINT_REVISION='321e6f6e7af5d7546b637f147783fc28ab5d4a7a'
from relkit.reproduction_l143 import first_validation_min

CONFIG=dict(num_model_layers=1,channels=128,aggr='sum',num_heads=4)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()

def materialize(root,source_root):
    """Fresh full test-cutoff database and released GloVe row materialization."""
    import urllib.request,zipfile,importlib.metadata as md
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download
    root=Path(root);root.mkdir(parents=True,exist_ok=False);seed_everything(42);torch.set_num_threads(1)
    for name,digest in ARCHIVES.items():
        target=root/'cache'/name;target.parent.mkdir(parents=True,exist_ok=True)
        urllib.request.urlretrieve('https://relbench.stanford.edu/download/rel-f1/'+name,target)
        assert sha(target)==digest
        with zipfile.ZipFile(target) as z:z.extractall(root/'unpacked')
    dataset=get_dataset('rel-f1');dataset.cache_dir=str(root/'unpacked');dataset.get_db.cache_clear();db=dataset.get_db()
    # Fail closed if installed preprocessing differs from the vendored release.
    import relbench.modeling.graph as graph,relbench.modeling.nn as nn,relbench.modeling.utils as utils
    source_checks={}
    for module,name in [(graph,'graph'),(nn,'nn'),(utils,'utils')]:
        expected=Path(source_root)/f'relbench__modeling__{name}.py'
        source_checks[name]=Path(module.__file__).read_bytes()==expected.read_bytes()
    assert all(source_checks.values()),source_checks
    types=get_stype_proposal(db)
    text=SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d',revision='e5e8fec6971be8960cfaa853a77a6ddc62a265d7',device='cpu')
    def embed(strings):return text.encode(strings,convert_to_tensor=True)
    data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256),cache_dir=str(root/'materialized'))
    # Independent FK reconstruction from table rows, including missing references.
    fk_edges=0
    for table_name,table in db.table_dict.items():
        for column,parent in table.fkey_col_to_pkey_table.items():
            df=db.table_dict[parent].df;pk=db.table_dict[parent].pkey_col
            lookup={int(v):i for i,v in enumerate(df[pk])}
            expected={(i,lookup[int(v)]) for i,v in enumerate(table.df[column]) if not __import__('pandas').isna(v)}
            actual=set(map(tuple,data[(table_name,'f2p_'+column,parent)].edge_index.numpy().T))
            assert actual==expected,(table_name,column);fk_edges+=len(actual)
    torch.save((data,stats),root/'graph.pt')
    checkpoint=hf_hub_download('tianlangchen/RelGNN','rel-f1_driver-position.pth',revision=CHECKPOINT_REVISION)
    import shutil;shutil.copyfile(checkpoint,root/'released.pth')
    info=dict(graph_sha256=sha(root/'graph.pt'),checkpoint_sha256=sha(root/'released.pth'),checkpoint_revision=CHECKPOINT_REVISION,
        preprocessing='FRESH full released snapshot; seed42; pinned GloVe',archives=ARCHIVES,source_checks=source_checks,
        rows={k:len(t) for k,t in db.table_dict.items()},fk_edges=fk_edges,edges=sum(e.shape[1] for e in data.edge_index_dict.values()),
        routes=[list(r) for r in get_atomic_routes(data.edge_types)],packages={k:md.version(k) for k in ['torch','torch-geometric','pytorch-frame','relbench','numpy','pandas','sentence-transformers','pyg-lib']})
    (root/'prepared.json').write_text(json.dumps(info,indent=2));return info

def full_run(output,prepared_root,source_root,seed=0,replay=False,lr=.005,evaluate_test=True,track="reference"):
    """All query rows are used. Test is opened only after validation selection."""
    if lr not in (.001,.003,.005):raise ValueError('Unplanned learning rate')
    if track=='search' and evaluate_test:raise ValueError('Search cannot evaluate test')
    start=time.perf_counter();torch.set_num_threads(1);seed_everything(42 if replay else seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=False);root=Path(prepared_root)
    prepared=json.loads((root/'prepared.json').read_text());assert sha(root/'graph.pt')==prepared['graph_sha256']
    for name,digest in ARCHIVES.items():assert sha(root/'cache'/name)==digest
    data,stats=torch.load(root/'graph.pt',weights_only=False)
    dataset=get_dataset('rel-f1');dataset.cache_dir=str(root/'unpacked');dataset.get_db.cache_clear()
    task=get_task('rel-f1','driver-position');task.cache_dir=str(root/'unpacked'/'driver-position');task.get_table.cache_clear()
    raw_get_table=task.get_table
    def guarded_get_table(split,*args,**kwargs):
        if split=='test' and not evaluate_test:raise RuntimeError('Test access forbidden in search')
        return raw_get_table(split,*args,**kwargs)
    # Local wrapper: no mutation of the cached task object's method.
    train=guarded_get_table('train');clip=np.percentile(train.df[task.target_col].to_numpy(),[2,98]).tolist()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    kwargs=dict(data=data,col_stats_dict=stats,out_channels=1,norm='batch_norm',atomic_routes=get_atomic_routes(data.edge_types),**CONFIG)
    model=RelGNN_Model(**kwargs).to(device)
    # Initial lazy columns must be materialized before a new optimizer is created.
    def loader(split):
        table=guarded_get_table(split,mask_input_cols=False)
        input_table=table
        if split=='test':
            from relbench.base import Table
            input_table=Table(df=table.df.drop(columns=[task.target_col]),fkey_col_to_pkey_table=table.fkey_col_to_pkey_table,pkey_col=table.pkey_col,time_col=table.time_col)
        inp=get_node_train_table_input(table=input_table,task=task)
        return NeighborLoader(data,num_neighbors=[128,64],time_attr='time',input_nodes=inp.nodes,input_time=inp.time,transform=inp.transform,subgraph_type='bidirectional',batch_size=512,temporal_strategy='uniform',shuffle=split=='train',num_workers=0),table
    # Replay construction order matches source: val loader, test loader, model initialization.
    # Constructors do not sample; seed reset below matches original parameter RNG consumption.
    val_loader,val_table=loader('val')
    train_loader,_=loader('train') if not replay else (None,None)
    audit=dict(query_occurrences=0,timestamped_node_occurrences=0,future_violations=0);trace={};grad_norm=None;nonfinite_gradients={}
    def audit_batch(batch):
        cut=batch[task.entity_table].seed_time
        audit['query_occurrences']+=len(cut)
        for name,times in batch.time_dict.items():
            owners=batch[name].batch;bad=int((times>cut[owners]).sum());audit['future_violations']+=bad
            audit['timestamped_node_occurrences']+=len(times)
            assert bad==0,'Future node relative to owning query'
        if not trace:
            trace['cutoffs']=cut.cpu().numpy()
            for name,times in batch.time_dict.items():
                trace[name+'_times']=times.cpu().numpy();trace[name+'_owners']=batch[name].batch.cpu().numpy()
    def evaluate(dl,table,original=None):
        model.eval();parts=[];ids=[];max_error=0.
        with torch.no_grad():
            for batch in dl:
                audit_batch(batch);ids.append(batch[task.entity_table].input_id.cpu().numpy());batch=batch.to(device)
                p=model(batch,task.entity_table).view(-1)
                if original is not None:
                    q=original(batch,task.entity_table).view(-1)
                    error=float((p-q).abs().max());max_error=max(max_error,error)
                    torch.testing.assert_close(p,q,atol=2e-4,rtol=2e-4)
                parts.append(p.clamp(*clip).cpu().numpy())
        ids=np.concatenate(ids);pred=np.concatenate(parts);assert np.array_equal(np.sort(ids),np.arange(len(table)))
        frame=table.df;entity=frame[task.entity_col].to_numpy(dtype=np.int64)
        times=(frame[task.time_col].astype('int64')//10**9).to_numpy();target=frame[task.target_col].to_numpy()
        keys=list(zip(entity,times));predkeys=[keys[i] for i in ids]
        score=keyed_mae(keys,target,predkeys,pred)
        aligned=np.empty_like(pred);aligned[ids]=pred
        assert abs(score-task.evaluate(aligned,table)['mae'])<1e-10
        return dict(mae=score,entity=entity,time=times,target=target,pred=aligned,max_original_error=max_error)
    history=[];best=float('inf');best_epoch=None
    if replay:
        assert sha(root/'released.pth')==prepared['checkpoint_sha256']
        model.load_state_dict(torch.load(root/'released.pth',map_location=device,weights_only=False))
        # Model creation above matches release seed42; checkpoint fills every parameter.
    else:
        model.eval()
        with torch.no_grad():model(next(iter(train_loader)).to(device),task.entity_table)
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        for epoch in range(1,11):
            model.train();total=0.;seen=0;steps=0
            for batch in train_loader:
                audit_batch(batch)
                assert np.array_equal(batch[task.entity_table].y.numpy(),train.df[task.target_col].to_numpy()[batch[task.entity_table].input_id.numpy()])
                batch=batch.to(device);optimizer.zero_grad()
                p=model(batch,task.entity_table).view(-1);y=batch[task.entity_table].y
                loss=torch.nn.functional.l1_loss(p,y);assert bool(torch.isfinite(loss));loss.backward()
                if grad_norm is None:
                    nonfinite_gradients={n:int((~torch.isfinite(v.grad)).sum()) for n,v in model.named_parameters() if v.grad is not None and not torch.isfinite(v.grad).all()}
                    grad_norm=float(sum(v.grad[torch.isfinite(v.grad)].square().sum() for v in model.parameters() if v.grad is not None).sqrt())
                optimizer.step();total+=float(loss.detach())*len(y);seen+=len(y);steps+=1
            assert seen==len(train)
            metric=evaluate(val_loader,val_table)['mae']
            history.append(dict(epoch=epoch,train_l1=total/seen,val_mae=metric,queries=seen,steps=steps))
            if first_validation_min(history)==epoch:
                best=metric;best_epoch=epoch;torch.save(model.state_dict(),out/'selected.pt')
            (out/'progress.json').write_text(json.dumps(history,indent=2))
            print('seed',seed,'epoch',epoch,'val',metric,flush=True)
        model.load_state_dict(torch.load(out/'selected.pt',map_location=device,weights_only=False))
    # Compare all held-out raw outputs with original architecture at identical weights/batches.
    from _parity_l143 import original_modules
    with original_modules(source_root) as (Original,_):
        # Constructing the parity oracle must not change neighborhood sampling RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            original=Original(**kwargs).to(device);original.load_state_dict(model.state_dict());original.eval()
        # Original replay samples test first and never consumes validation sampling RNG.
        evaluation=[('val',val_loader,val_table)]
        if evaluate_test:
            test_loader,test_table=loader('test');evaluation.insert(0,('test',test_loader,test_table))
        scores={};arrays={}
        for split,dl,table in evaluation:
            ev=evaluate(dl,table,original);scores[split]={'mae':ev.pop('mae'),'max_original_error':ev.pop('max_original_error')}
            for k,v in ev.items():arrays[split+'_'+k]=v
    np.savez_compressed(out/'predictions.npz',**arrays);np.savez_compressed(out/'sampled_trace.npz',**trace)
    result=dict(status='COMPLETE',kind='CHECKPOINT_COMPATIBILITY_REPLAY' if replay and 'compatibility' in prepared else ('RELEASED_CHECKPOINT_REPLAY' if replay else 'RECONSTRUCTED_TRAINING'),seed=42 if replay else seed,
        epochs=0 if replay else 10,history=history,best_epoch=best_epoch,selection_mae=None if replay else best,scores=scores,
        count=dict(train=len(train),val=len(val_table),**({'test':len(test_table)} if evaluate_test else {})),lr=lr,track=track,test_access='AFTER_SELECTION' if evaluate_test else 'FORBIDDEN',clip=clip,config=CONFIG,finite_gradient_norm=grad_norm,nonfinite_gradients=nonfinite_gradients,compatibility=prepared.get("compatibility"),
        graph_sha256=prepared['graph_sha256'],checkpoint_sha256=prepared['checkpoint_sha256'] if replay else sha(out/'selected.pt'),
        temporal_audit=audit,seconds=time.perf_counter()-start,paper_target=3.798,historical_training='NOT_ESTABLISHED')
    (out/'result.json').write_text(json.dumps(result,indent=2));return result
