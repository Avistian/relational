"""Prepare the full released products archive; native METIS is provided infrastructure."""
import json,os,time,urllib.request,zipfile,shutil,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData
from relkit.scaling_l113 import file_hash,tensor_hash,mean_adjacency

def prepare(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    if (root/'prepared.json').exists():return json.loads((root/'prepared.json').read_text())
    archive=root/'products.zip';url='https://snap.stanford.edu/ogb/data/nodeproppred/products.zip'
    if not archive.exists():
        # Server supports byte ranges. Read 16 nonoverlapping ranges to local SSD,
        # validate every Content-Range, then verify ZIP CRCs on extraction.
        size=1480993786
        target=Path(tempfile.mkdtemp(prefix='l113-download-'))/'products.zip'
        with target.open('wb') as f:f.truncate(size)
        def part(i):
            lo=size*i//16;hi=size*(i+1)//16-1
            req=urllib.request.Request(url,headers={'Range':f'bytes={lo}-{hi}'})
            with urllib.request.urlopen(req,timeout=60) as response,target.open('r+b') as out:
                assert response.status==206 and response.headers['Content-Range']==f'bytes {lo}-{hi}/{size}'
                out.seek(lo);n=0
                while True:
                    b=response.read(1024*1024)
                    if not b:break
                    out.write(b);n+=len(b)
                assert n==hi-lo+1
            print('Download range complete',i,flush=True)
        with ThreadPoolExecutor(max_workers=16) as pool:list(pool.map(part,range(16)))
        shutil.copyfile(target,root/'products.zip.partial');(root/'products.zip.partial').rename(archive)
        shutil.rmtree(target.parent)
    expected='5ea0a112edaec2141c0a2a612dd4aed58df97ff3e1ab1a0ca8238f43cbbb50a8'
    assert file_hash(archive)==expected,'Products archive identity mismatch'
    print('Downloaded archive',archive.stat().st_size,flush=True)
    if not (root/'products/raw/node-feat.csv.gz').exists():
        with zipfile.ZipFile(archive) as z:
            assert all(not name.startswith('/') and '..' not in Path(name).parts for name in z.namelist())
            z.extractall(root)
    raw=root/'products/raw';read=lambda path,dtype:pd.read_csv(path,header=None,dtype=dtype).values
    x=torch.from_numpy(read(raw/'node-feat.csv.gz',np.float32));y=torch.from_numpy(read(raw/'node-label.csv.gz',np.int64)).view(-1)
    raw_edges=torch.from_numpy(read(raw/'edge.csv.gz',np.int64).T.copy());edge=torch.cat([raw_edges,raw_edges.flip(0)],1);del raw_edges
    split={k:torch.from_numpy(read(root/f'products/split/sales_ranking/{k}.csv.gz',np.int64).reshape(-1)) for k in ['train','valid','test']}
    assert len(y)==2449029 and edge.shape[1]==123718280 and x.shape==(2449029,100)
    assert [len(split[k]) for k in ['train','valid','test']]==[196615,39323,2213091]
    assert torch.unique(torch.cat(list(split.values()))).numel()==len(y)
    data=Data(x=x,y=y,edge_index=edge,num_nodes=len(y))
    data.train_mask=torch.zeros(len(y),dtype=torch.bool);data.train_mask[split['train']]=True
    print('Loaded full graph; preparing adjacency',flush=True)
    adj=mean_adjacency(edge,len(y));torch.save(adj,root/'adj.pt')
    torch.save({'data':data,'split':split},root/'data.pt');del adj
    print('Partitioning into 15000 clusters',flush=True);tick=time.perf_counter();torch.manual_seed(113)
    clusters=ClusterData(data,num_parts=15000,recursive=False,save_dir=str(root/'partitions'))
    torch.save(clusters,root/'clusters.pt')
    audit={'url':url,'archive_sha256':file_hash(archive),'nodes':len(y),'directed_entries':edge.shape[1],'features':100,'classes':47,
           'split_counts':{k:len(v) for k,v in split.items()},'split_hashes':{k:tensor_hash(v) for k,v in split.items()},
           'x_sha256':tensor_hash(x),'y_sha256':tensor_hash(y),'partition_sha256':file_hash(root/'clusters.pt'),
           'partition_seconds':time.perf_counter()-tick,'seconds':time.perf_counter()-start,
           'runtime':{'torch':torch.__version__},'partition_seed_note':'torch seed113; original METIS partition and seed unavailable; fixed saved partition shared by all runs'}
    for name in ['data.pt','adj.pt','clusters.pt']:audit[name+'_sha256']=file_hash(root/name)
    (root/'prepared.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True);return audit
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',default='labs/data/l113');a=p.parse_args();prepare(a.root)
