"""Prepare full F1 graph then run a bounded released-protocol experiment."""
import argparse,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
import torch
from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.utils import get_stype_proposal
from torch_frame.config import TextEmbedderConfig
from torch_geometric.seed import seed_everything
from relkit.rdl_l117 import make_pkey_fkey_graph
from relkit.classification_l128 import fit_rdl
import relkit.classification_l128 as model_module
from relkit.batch_audit_l123 import audit_batch
P=Path(__file__).resolve().parent

def run(seed,epochs,output):
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1);start=time.perf_counter();seed_everything(42)
    dataset=get_dataset('rel-f1',download=True);db=dataset.get_db()
    from relkit.historical_task_l128 import historical_task
    task,recovery=historical_task(output)
    hashes={}
    for name,expected in [('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482')]:
        path=Path(dataset.cache_dir)/name
        digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==expected;hashes[name]=digest
    types=get_stype_proposal(db)
    spec=json.loads((P/'sources/l117/text_model.json').read_text())
    text_model=SentenceTransformer(spec['model'],revision=spec['revision'],device='cpu')
    def embed(strings):return torch.from_numpy(text_model.encode(strings,show_progress_bar=False))
    data,stats=make_pkey_fkey_graph(db,types,TextEmbedderConfig(text_embedder=embed,batch_size=256))
    del text_model
    # Compare every edge with an independent SQL-style key lookup from database rows.
    from relkit.rdl_l117 import foreign_key_edges
    for table_name,table in db.table_dict.items():
        for fk,dst in table.fkey_col_to_pkey_table.items():
            parent=db.table_dict[dst];keys=parent.df[parent.pkey_col].tolist()
            values=[None if __import__('pandas').isna(x) else int(x) for x in table.df[fk]]
            edge=foreign_key_edges(keys,values)
            actual=data[(table_name,'f2p_'+fk,dst)].edge_index.numpy()
            assert set(map(tuple,actual.T))==set(map(tuple,edge.T))
    sys.path.insert(0,str(P/'sources/l117'))
    from model import Model as OriginalModel
    # Check installed RelBench primitives match the pinned upstream source exactly.
    import relbench.modeling.nn as nn
    assert Path(nn.__file__).read_bytes()==(P/'sources/l117/nn.py').read_bytes()
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    import importlib.metadata as md
    audit=dict(task_recovery=recovery,database_hashes=hashes,rows={k:len(v) for k,v in db.table_dict.items()},
        edges={'|'.join(k):v.shape[1] for k,v in data.edge_index_dict.items()},
        stypes={k:{c:str(s) for c,s in v.items()} for k,v in types.items()},
        preprocessing='Released full database up to test cutoff; statistics are not train-only',
        preprocessing_seed=42,text_model=spec,packages={k:md.version(k) for k in ['torch','torch-geometric','pytorch-frame','relbench','numpy','pandas','sentence-transformers','pyg-lib']},
        classification_sha256=hashlib.sha256((P/'relkit/classification_l128.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest())
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    counts={};BaseLoader=model_module.NeighborLoader
    class AuditedLoader(BaseLoader):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.original_graph=args[0];self.entity=kwargs['input_nodes'][0]
            self.audit_key=str(len(counts));counts[self.audit_key]=dict(batches=0,queries=0,dated_node_occurrences=0,edge_occurrences=0)
        def __iter__(self):
            for batch in super().__iter__():
                report=audit_batch(batch,self.original_graph,self.entity)
                for name,value in report.items():counts[self.audit_key][name]+=value
                yield batch
    model_module.NeighborLoader=AuditedLoader
    try:result=fit_rdl(data,stats,task,seed,output,epochs,'cuda' if torch.cuda.is_available() else 'cpu',OriginalModel)
    finally:model_module.NeighborLoader=BaseLoader
    for key,n in [('0',epochs*11411),('1',(epochs+1)*566),('2',702)]:assert counts[key]['queries']==n
    report=dict(status='PASS',splits=dict(zip(['train','val','test'],counts.values())),audit_sha256=hashlib.sha256((P/'relkit/batch_audit_l123.py').read_bytes()).hexdigest(),contract='All sampled timestamps, source edges and query isolation')
    (output/'temporal-audit.json').write_text(json.dumps(report,indent=2))
    result['total_seconds']=time.perf_counter()-start
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',default='labs/results/l128/local');a=p.parse_args()
    print(run(a.seed,a.epochs,a.output))
