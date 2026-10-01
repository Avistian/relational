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
from relkit.rdl_l117 import make_pkey_fkey_graph,fit_rdl
P=Path(__file__).resolve().parent

def run(seed,epochs,output):
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1);start=time.perf_counter();seed_everything(42)
    dataset=get_dataset('rel-f1',download=True);db=dataset.get_db()
    task=get_task('rel-f1','driver-position',download=True)
    hashes={}
    for name,expected in [('db.zip','ec31a4e1bc2b2f9c36c05fcd3dfe2a40a506f335dc51ce79c3ec8bb40feb1482'),('tasks/driver-position.zip','775b28a51604169539bbe712a2f0d15158c112bc6abf316cdd0995087a7ae03e')]:
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
    audit=dict(database_hashes=hashes,rows={k:len(v) for k,v in db.table_dict.items()},
        edges={'|'.join(k):v.shape[1] for k,v in data.edge_index_dict.items()},
        stypes={k:{c:str(s) for c,s in v.items()} for k,v in types.items()},
        preprocessing='Released full database up to test cutoff; statistics are not train-only',
        preprocessing_seed=42,text_model=spec,packages={k:md.version(k) for k in ['torch','torch-geometric','pytorch-frame','relbench','numpy','pandas','sentence-transformers','pyg-lib']},
        source_sha256=hashlib.sha256((P/'relkit/rdl_l117.py').read_bytes()).hexdigest())
    (output/'audit.json').write_text(json.dumps(audit,indent=2))
    result=fit_rdl(data,stats,task,seed,output,epochs,'cuda' if torch.cuda.is_available() else 'cpu',OriginalModel)
    result['total_seconds']=time.perf_counter()-start
    (output/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=10);p.add_argument('--output',default='labs/results/l117/local');a=p.parse_args()
    print(run(a.seed,a.epochs,a.output))
