"""Visible R-GCN reconstruction of tkipf/relational-gcn (MIT).
Source revision 4bec1341dd46b72bf482f7ed26c2dca4533577f6.
Modern PyTorch execution, canonical node order, declared fresh seeds.
"""
from pathlib import Path
import csv,gzip,hashlib,io,json,math,platform,tarfile,time,urllib.request
import numpy as np
import scipy.sparse as sp
import rdflib
import torch
from torch import nn
from torch.nn import functional as F


def normalize_relations(adjacencies):
    """Row = receiving node; normalize each relation independently; zero stays zero."""
    result=[]
    for a in adjacencies:
        a=a.tocsr().astype(np.float32)
        degree=np.asarray(a.sum(1)).ravel()
        inv=np.divide(1.,degree,out=np.zeros_like(degree),where=degree!=0)
        a=(sp.diags(inv)@a).tocoo()
        result.append(torch.sparse_coo_tensor(np.stack([a.row,a.col]),a.data,a.shape).coalesce())
    return result


def compose_weights(bases,coefficients):
    """B×din×dout and R×B -> R×din×dout; no basis activation."""
    return torch.einsum('rb,bio->rio',coefficients,bases)


def relation_sum(supports,features,weights):
    """Sum per-relation means, including the identity support for the self path."""
    return sum(torch.sparse.mm(a,features)@w for a,w in zip(supports,weights))


def masked_loss(logits,labels,train_idx):
    return F.cross_entropy(logits[train_idx],labels[train_idx])


def horizontal_support(supports):
    n=supports[0].shape[0]
    indices=[];values=[]
    for r,a in enumerate(supports):
        ij=a.indices().clone();ij[1]+=r*n
        indices.append(ij);values.append(a.values())
    return torch.sparse_coo_tensor(torch.cat(indices,1),torch.cat(values),(n,n*len(supports))).coalesce()


class RGCN(nn.Module):
    def __init__(self,nodes,relations,classes,hidden=16,bases=0):
        super().__init__();self.bases=bases
        count=bases if bases>0 else relations
        self.v1=nn.Parameter(torch.empty(count,nodes,hidden))
        self.v2=nn.Parameter(torch.empty(count,hidden,classes))
        self.c1=nn.Parameter(torch.empty(relations,bases)) if bases else None
        self.c2=nn.Parameter(torch.empty(relations,bases)) if bases else None
        # Release initializes EACH 2D relation/basis separately, not a 3D tensor.
        for v in [self.v1,self.v2]:
            for matrix in v:nn.init.xavier_uniform_(matrix)
        for c in [self.c1,self.c2]:
            if c is not None:nn.init.xavier_uniform_(c)

    def forward(self,supports,joined=None):
        w1=compose_weights(self.v1,self.c1) if self.bases else self.v1
        w2=compose_weights(self.v2,self.c2) if self.bases else self.v2
        # H0 = I. S_r I W_r = S_r W_r; no dense N×N identity allocation.
        joined=horizontal_support(supports) if joined is None else joined
        h=torch.relu(torch.sparse.mm(joined,w1.reshape(-1,w1.shape[-1])))
        return relation_sum(supports,h,w2)


class KerasAdam:
    """Keras 1.2.1 update placement; avoids modern Adam epsilon discrepancy."""
    def __init__(self,parameters,lr=.01):
        self.parameters=list(parameters);self.lr=lr;self.t=0
        self.m=[torch.zeros_like(p) for p in self.parameters]
        self.v=[torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:p.grad=None

    @torch.no_grad()
    def step(self):
        self.t+=1
        rate=self.lr*math.sqrt(1-.999**self.t)/(1-.9**self.t)
        for p,m,v in zip(self.parameters,self.m,self.v):
            g=p.grad
            m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.addcdiv_(m,v.sqrt().add_(1e-8),value=-rate)


def verified_archive(root,manifest):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'aifb.tgz'
    if not path.exists():
        payload=urllib.request.urlopen(manifest['data']['url'],timeout=90).read()
        if hashlib.sha256(payload).hexdigest()!=manifest['data']['sha256']:raise ValueError('Download hash mismatch')
        path.write_bytes(payload)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['data']['sha256']:raise ValueError('Archive hash mismatch')
    return path


class DocumentBlankNodes(dict):
    """Preserve this one document's blank-node labels, rather than random UUIDs."""
    def get(self,key,default=None):
        return rdflib.BNode(key)


def load_aifb(root,manifest,prune=True):
    """Original triples/split; canonical ordering; release two-layer row pruning."""
    archive=verified_archive(root,manifest)
    with tarfile.open(archive) as tar:
        def member(name):
            matches=[m for m in tar.getmembers() if m.name.lstrip('./')==name]
            if len(matches)!=1 or not matches[0].isfile():raise ValueError(name)
            return tar.extractfile(matches[0]).read()
        graph=rdflib.Graph()
        graph.parse(data=gzip.decompress(member('aifb_stripped.nt.gz')).decode(),format='nt',bnode_context=DocumentBlankNodes())
        tables={name:list(csv.DictReader(io.StringIO(member(name+'.tsv').decode()),delimiter='\t')) for name in ['completeDataset','trainingSet','testSet']}
        for name in tables:
            assert hashlib.sha256(member(name+'.tsv')).hexdigest()==manifest['split_sha256'][name+'.tsv']
    relations=sorted(set(graph.predicates()),key=lambda p:(-sum(1 for _ in graph.triples((None,p,None))),str(p)))
    forbidden={'http://swrc.ontoware.org/ontology#employs','http://swrc.ontoware.org/ontology#affiliation'}
    assert not (set(map(str,relations))&forbidden),'Target relation leakage'
    # n3 preserves RDF term type, language and datatype (str alone need not).
    nodes=sorted(set(graph.subjects())|set(graph.objects()),key=lambda x:x.n3())
    node_ids={v:i for i,v in enumerate(nodes)}
    classes=sorted({r['label_affiliation'] for r in tables['completeDataset']})
    labels=torch.full((len(nodes),),-1,dtype=torch.long)
    splits={}
    for name in ['trainingSet','testSet']:
        ids=[]
        for row in tables[name]:
            idx=node_ids[rdflib.URIRef(row['person'])];ids.append(idx)
            labels[idx]=classes.index(row['label_affiliation'])
        splits[name]=torch.tensor(ids)
    assert not set(splits['trainingSet'].tolist())&set(splits['testSet'].tolist())
    adj=[]
    for rel in relations:
        pairs=sorted((node_ids[s],node_ids[o]) for s,p,o in graph.triples((None,rel,None)))
        row,col=np.asarray(pairs).T
        a=sp.csr_matrix((np.ones(len(row)),(row,col)),shape=(len(nodes),len(nodes)))
        adj.extend([a,a.T.tocsr()])
    adj.append(sp.eye(len(nodes),format='csr'))
    # Release A_r[s,o] receives object into subject; inverse support reverses it.
    # Use identities of all labeled roots, NEVER their target values.
    if prune:
        roots=np.concatenate([splits['trainingSet'].numpy(),splits['testSet'].numpy()])
        keep=set(roots.tolist())
        for a in adj:keep.update(a[roots].indices.tolist())
        mask=np.zeros(len(nodes));mask[list(keep)]=1
        adj=[sp.diags(mask)@a for a in adj]
    meta={'nodes':len(nodes),'raw_relations':len(relations),'triples':len(graph),'supports':len(adj),'train':len(splits['trainingSet']),'test':len(splits['testSet']),'classes':len(classes),'target_relations_absent':True,'pruned_rows':prune,'node_order_sha256':hashlib.sha256('\n'.join(x.n3() for x in nodes).encode()).hexdigest(),'relation_order':[str(r) for r in relations],'class_order':classes}
    assert [meta[k] for k in ['nodes','raw_relations','triples','train','test','classes']]==[8285,45,29043,140,36,4],meta
    return normalize_relations(adj),labels,splits['trainingSet'],splits['testSet'],meta


def train_aifb(data,seed,epochs=50,bases=0):
    supports,labels,train_idx,test_idx,meta=data
    torch.manual_seed(seed)
    model=RGCN(meta['nodes'],meta['supports'],meta['classes'],bases=bases)
    opt=KerasAdam(model.parameters());joined=horizontal_support(supports);trace=[]
    start=time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad();logits=model(supports,joined)
        loss=masked_loss(logits,labels,train_idx);loss.backward();opt.step()
        trace.append(float(loss.detach()))
    # Fixed final epoch, no test-dependent checkpoint or hyperparameter selection.
    with torch.no_grad():
        logits=model(supports,joined);prediction=logits[test_idx].argmax(1)
        correct=int((prediction==labels[test_idx]).sum())
    return {'seed':seed,'epochs':epochs,'bases':bases,'test_correct':correct,'test_total':len(test_idx),'test_accuracy':correct/len(test_idx),'train_loss':trace,'test_predictions':prediction.tolist(),'test_labels':labels[test_idx].tolist(),'test_ids':test_idx.tolist(),'seconds':time.perf_counter()-start}


def run_aifb(root,manifest,seeds=range(10),bases=0,epochs=50):
    torch.set_num_threads(1);data=load_aifb(root,manifest);runs=[]
    for seed in seeds:
        record=train_aifb(data,int(seed),epochs,bases);runs.append(record)
        print(f"seed={seed} bases={bases} accuracy={record['test_accuracy']:.6f} seconds={record['seconds']:.1f}",flush=True)
    scores=np.array([r['test_accuracy'] for r in runs]);mean=float(scores.mean())
    return {'experiment':'R-GCN Table 2 AIFB release-protocol port' if bases==0 else 'AIFB basis-sharing teaching extension','status':'EXECUTED','historical_exact_parity':'INCOMPARABLE','source_revision':manifest['source_revision'],'data_sha256':manifest['data']['sha256'],'data':data[-1],'runs':runs,'mean':mean,'sample_sd':float(scores.std(ddof=1)) if len(scores)>1 else None,'paper_target':.9583 if bases==0 else None,'gap_percentage_points':100*(mean-.9583) if bases==0 else None,'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'rdflib':rdflib.__version__}}
