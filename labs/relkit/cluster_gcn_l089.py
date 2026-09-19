"""Visible Cluster-GCN release reconstruction. See l089-reproduction.md for gaps.

Release recipe: Google Research 89c16e4, cluster_gcn/run_ppi.sh (2019).
Modern PyTorch/PyMetis numerics are not historical TensorFlow/METIS parity.
"""
import hashlib
import json
import math
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pymetis
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler

DATA_URL = 'https://snap.stanford.edu/graphsage/ppi.zip'
DATA_SHA = '53aeb76e54fd41b645e7edb48b62929240b89839495396b048086fd212503fbd'


def load_ppi(root='l089-data'):
    """Full GraphSAGE PPI archive; train-only scaling, legacy JSON endpoint mapping."""
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    archive = root / 'ppi.zip'
    if not archive.exists():
        urllib.request.urlretrieve(DATA_URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DATA_SHA:
        raise ValueError('PPI archive hash mismatch')
    with zipfile.ZipFile(archive) as z:
        def member(suffix):
            names = [n for n in z.namelist() if n.endswith(suffix)]
            assert len(names) == 1
            return z.open(names[0])
        graph = json.load(member('-G.json'))
        ids = {int(k): int(v) for k,v in json.load(member('-id_map.json')).items()}
        labels = json.load(member('-class_map.json'))
        x = np.load(member('-feats.npy'), allow_pickle=False).astype(np.float32)
    n = len(ids)
    masks = {k: np.zeros(n, dtype=bool) for k in ['train','valid','test']}
    y = np.zeros((n,121),dtype=np.float32)
    for node in graph['nodes']:
        if int(node['id']) not in ids: continue
        i = ids[int(node['id'])]
        masks['valid'][i] = node['val']; masks['test'][i] = node['test']
        masks['train'][i] = not(node['val'] or node['test'])
        y[i] = labels[str(node['id'])]
    rows,cols = [],[]
    # networkx 1.11 node-link format stores indices into the nodes list.
    for edge in graph['links']:
        u = int(graph['nodes'][edge['source']]['id'])
        v = int(graph['nodes'][edge['target']]['id'])
        if u in ids and v in ids:
            rows.append(ids[u]); cols.append(ids[v])
    # NetworkX Graph collapses duplicates; preserve doubled diagonal from A + A.T.
    a = sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n),dtype=np.float32)
    a.data[:] = 1
    a = (a + a.T).tocsr()
    scaler = StandardScaler().fit(x[masks['train']])
    x = scaler.transform(x).astype(np.float32)
    assert n == 56944 and x.shape == (n,50) and y.shape == (n,121)
    assert np.all(sum(m.astype(int) for m in masks.values()) == 1)
    return {'a':a,'x':x,'y':y,'masks':masks,'scaler_mean':scaler.mean_}


def induced_batch(a, parts, chosen):
    """TODO: concatenate selected disjoint partitions; slice ORIGINAL A on both axes.

    Return (global node IDs in partition order, CSR adjacency in that local order).
    Preserve edges between selected partitions. Do not concatenate block diagonals.
    """
    nodes = np.concatenate([parts[i] for i in chosen]).astype(np.int64)
    if len(np.unique(nodes)) != len(nodes): raise ValueError('Partitions overlap')
    return nodes, a[nodes][:,nodes].tocsr()


def enhanced_support(a, diag_lambda=1.0):
    """TODO: add I, row-normalize, then add lambda times the resulting diagonal.

    S = (D+I)^(-1)(A+I) + lambda*diag((D+I)^(-1)(A+I)).
    The final row sum need not be one. diag_lambda=-1 is the release's plain D^-1 A.
    """
    if diag_lambda == -1:
        return (sp.diags(1 / np.maximum(1,np.asarray(a.sum(1)).ravel())) @ a).tocsr()
    b = a + sp.eye(a.shape[0],format='csr',dtype=np.float32)
    b = sp.diags(1 / np.asarray(b.sum(1)).ravel()) @ b
    return (b + diag_lambda * sp.diags(b.diagonal())).astype(np.float32).tocsr()


def partition(a, num_parts=50, method='metis', seed=1):
    """Partition eligible nodes only. PyMetis 2025.2.2 is a declared runtime deviation."""
    n = a.shape[0]
    if not 1 <= num_parts <= n: raise ValueError('Invalid partition count')
    if num_parts == 1: return [np.arange(n,dtype=np.int64)]
    if method == 'random':
        return [p.astype(np.int64) for p in np.array_split(np.random.RandomState(seed).permutation(n),num_parts)]
    clean = a.copy(); clean.setdiag(0); clean.eliminate_zeros(); clean.sort_indices()
    # Legacy metis.part_graph defaults to multilevel k-way (not recursive bisection).
    _, membership = pymetis.part_graph(num_parts,
        adjacency=[clean.indices[clean.indptr[i]:clean.indptr[i+1]].tolist() for i in range(n)],
        recursive=False, options=pymetis.Options(seed=seed))
    membership = np.asarray(membership)
    parts = [np.flatnonzero(membership == k) for k in range(num_parts)]
    if any(len(p)==0 for p in parts): raise ValueError('Empty partition')
    return parts


def sparse_tensor(a, device='cpu'):
    """Sparse COO; rows are receivers, columns senders."""
    a = a.tocoo()
    return torch.sparse_coo_tensor(np.stack([a.row,a.col]),a.data,a.shape,
                                   dtype=torch.float32,device=device).coalesce()


def concat_message(h, support):
    """TODO: return [S @ H, H], neighbor branch first, with gradients through both."""
    return torch.cat([torch.sparse.mm(support,h),h],dim=1)


class ClusterGCN(nn.Module):
    """Release GraphConvolution: concat -> dropout -> W -> per-node LN -> ReLU.

    First input is precomputed [A_train X, X], with raw (not normalized) A.
    The final layer omits layer norm and ReLU. No affine bias; LN has scale/offset.
    """
    def __init__(self, features=50, labels=121, hidden=2048, layers=5, dropout=.2):
        super().__init__()
        if layers < 2: raise ValueError('Use at least two layers')
        dims = [features]+[hidden]*(layers-1)+[labels]
        self.weights = nn.ParameterList([nn.Parameter(torch.empty(2*dims[i],dims[i+1])) for i in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden,eps=1e-9) for _ in range(layers-1)])
        self.dropout = dropout
        for w in self.weights: nn.init.xavier_uniform_(w)
    def forward(self, precomputed, support):
        h = precomputed
        for i,w in enumerate(self.weights):
            joined = h if i == 0 else concat_message(h,support)
            h = F.dropout(joined,p=self.dropout,training=self.training) @ w
            if i < len(self.weights)-1: h = F.relu(self.norms[i](h))
        return h


class ReleaseAdam:
    """TF1 Adam epsilon placement: lr_t*m / (sqrt(v)+epsilon), epsilon=1e-8.

    Unlike torch.optim.Adam, epsilon is added before the second-moment bias correction.
    Only dense gradients and zero weight decay are needed by the released PPI recipe.
    """
    def __init__(self, params, lr=.01):
        self.params = list(params); self.lr = lr; self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    def zero_grad(self):
        for p in self.params: p.grad = None
    @torch.no_grad()
    def step(self):
        self.t += 1
        alpha = self.lr * math.sqrt(1-.999**self.t)/(1-.9**self.t)
        for p,m,v in zip(self.params,self.m,self.v):
            if p.grad is None: continue
            m.mul_(.9).add_(p.grad,alpha=.1)
            v.mul_(.999).addcmul_(p.grad,p.grad,value=.001)
            p.addcdiv_(m,v.sqrt().add_(1e-8),value=-alpha)


def micro_f1(logits, labels):
    """Pool binary decisions across nodes AND labels; threshold logits strictly above zero."""
    p = np.asarray(logits)>0; y = np.asarray(labels)>.5
    tp = np.count_nonzero(p & y); fp = np.count_nonzero(p & ~y); fn = np.count_nonzero(~p & y)
    return float(2*tp/max(1,2*tp+fp+fn))


def prepare(data, num_parts=50, method='metis', partition_seed=1):
    """Training graph and first-layer cache never access held-out features/edges/labels."""
    train_ids = np.flatnonzero(data['masks']['train'])
    a = data['a'][train_ids][:,train_ids].tocsr(); x = data['x'][train_ids]
    t = time.perf_counter()
    parts = partition(a,num_parts,method,partition_seed)
    pre = np.concatenate([a @ x,x],axis=1).astype(np.float32)
    return {'a':a,'pre':pre,'y':data['y'][train_ids],'parts':parts,
            'ids':train_ids,'preprocess_seconds':time.perf_counter()-t}


@torch.no_grad()
def evaluate(model, data, mask_name, num_parts=1, device='cpu'):
    """Release inference: precompute on full graph; validation uses 2 parts, test 1.

    Stream partitions from host. PPI tissues are disconnected, but a METIS partition
    can still cut a tissue. Validation/test labels never enter the forward pass.
    """
    model.eval()
    a,x = data['a'],data['x']; mask = data['masks'][mask_name]
    pre = np.concatenate([a@x,x],axis=1).astype(np.float32)
    parts = partition(a,num_parts)
    all_ids,all_logits = [],[]
    total_loss = 0.0
    for nodes in parts:
        selected = mask[nodes]
        if not selected.any(): continue
        block = a[nodes][:,nodes].copy(); block.data[:] = 1  # release partition_graph binarizes
        s = sparse_tensor(enhanced_support(block),device)
        logits = model(torch.as_tensor(pre[nodes],device=device),s)
        z = logits[torch.as_tensor(selected,device=device)]
        target = torch.as_tensor(data['y'][nodes[selected]],device=device)
        total_loss += float(F.binary_cross_entropy_with_logits(z,target))*int(selected.sum())
        all_ids.append(nodes[selected]); all_logits.append(z.cpu().numpy())
    ids = np.concatenate(all_ids); logits = np.concatenate(all_logits)
    order = np.argsort(ids); ids,logits = ids[order],logits[order]
    return {'loss':total_loss/len(ids),'micro_f1':micro_f1(logits,data['y'][ids])},ids,logits


def preset(name):
    """Named paper target never silently shrinks. Other presets are different experiments."""
    configs = {
        'smoke':dict(hidden=32,layers=2,epochs=2,num_parts=50,q=1),
        'closer':dict(hidden=128,layers=3,epochs=30,num_parts=50,q=1),
        'paper':dict(hidden=2048,layers=5,epochs=400,num_parts=50,q=1),
    }
    return dict(configs[name],dropout=.2,lr=.01,partition_method='metis',partition_seed=1)


def train(data, config, seed=1, device='cpu', output=None, validate_every=1, test=True):
    """Fresh complete trainer: full training data, shuffled partition pass per epoch.

    Fixed final epoch, no best-validation checkpoint (the released recipe disables
    early stopping with patience 1000 > 400). Test runs once after training on CPU.
    Config + source hashes + runtime are persisted by the CLI; no resume is implied.
    """
    torch.manual_seed(seed); rng = np.random.RandomState(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    prep = prepare(data,config['num_parts'],config['partition_method'],config['partition_seed'])
    model = ClusterGCN(hidden=config['hidden'],layers=config['layers'],dropout=config['dropout']).to(device)
    opt = ReleaseAdam(model.parameters(),config['lr'])
    parts = prep['parts']; order = np.arange(len(parts)); curves = []
    out = Path(output) if output is not None else None
    if out:
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'partitions.npz',train_ids=prep['ids'],**{f'p{i}':p for i,p in enumerate(parts)})
    # q=1 uses fixed supports, as the release. Keep tensors on CPU until their batch.
    cached = {}
    if config['q']==1:
        for i,p in enumerate(parts):
            block = prep['a'][p][:,p].copy(); block.data[:] = 1
            cached[i] = (p,enhanced_support(block))
    max_nodes,max_nnz,retained = 0,0,[]
    train_seconds = 0.0
    for epoch in range(1,config['epochs']+1):
        if device.startswith('cuda'): torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); model.train(); loss_sum = 0; count = 0; edges_seen = 0
        rng.shuffle(order)
        # The q=1 release shuffles idx_parts twice, preserving this otherwise redundant step.
        if config['q']==1: rng.shuffle(order)
        for begin in range(0,len(order),config['q']):
            chosen = order[begin:begin+config['q']]
            if config['q']==1:
                nodes,s = cached[int(chosen[0])]
            else:
                nodes,a = induced_batch(prep['a'],parts,chosen); s = enhanced_support(a)
            support = sparse_tensor(s,device)
            x = torch.as_tensor(prep['pre'][nodes],device=device)
            y = torch.as_tensor(prep['y'][nodes],device=device)
            opt.zero_grad(); logits = model(x,support)
            loss = F.binary_cross_entropy_with_logits(logits,y)
            loss.backward(); opt.step()
            loss_sum += float(loss.detach())*len(nodes); count += len(nodes)
            max_nodes = max(max_nodes,len(nodes)); max_nnz = max(max_nnz,s.nnz)
            edges_seen += prep['a'][nodes][:,nodes].nnz
        if device.startswith('cuda'): torch.cuda.synchronize()
        elapsed = time.perf_counter()-start; train_seconds += elapsed
        row = {'epoch':epoch,'train_loss':loss_sum/count,'train_seconds':elapsed,
               'edge_fraction':edges_seen/max(1,prep['a'].nnz)}
        if device.startswith('cuda'): row['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        if epoch % validate_every == 0 or epoch == config['epochs']:
            metrics,_,_ = evaluate(model,data,'valid',2,device)
            row.update({'valid_'+k:v for k,v in metrics.items()})
        curves.append(row); retained.append(row['edge_fraction'])
        if out:
            (out/'progress.json').write_text(json.dumps({'status':'INCOMPLETE','config':config,'seed':seed,'curves':curves},indent=2))
        print(f"epoch={epoch}/{config['epochs']} loss={row['train_loss']:.5f} train_s={elapsed:.2f}"+
              (f" val_F1={row['valid_micro_f1']:.5f}" if 'valid_micro_f1' in row else ''),flush=True)
    result = {'status':'COMPLETE','config':config,'seed':seed,'selection':'final_epoch',
              'train_nodes':len(prep['ids']),'train_directed_nnz':prep['a'].nnz,
              'max_batch_nodes':max_nodes,'max_support_nnz':max_nnz,
              'hidden_state_proxy_bytes':4*max_nodes*config['hidden']*(config['layers']-1),
              'mean_retained_edge_fraction':float(np.mean(retained)),
              'preprocess_seconds':prep['preprocess_seconds'],'training_seconds':train_seconds,'curves':curves}
    if test:
        model.cpu()
        metrics,ids,logits = evaluate(model,data,'test',1,'cpu')
        result['test'] = metrics
        if out:
            np.savez_compressed(out/'test_predictions.npz',ids=ids,logits=logits,labels=data['y'][ids])
            result['prediction_sha256'] = hashlib.sha256((out/'test_predictions.npz').read_bytes()).hexdigest()
    if out:
        torch.save(model.cpu().state_dict(),out/'model.pt')
        (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
