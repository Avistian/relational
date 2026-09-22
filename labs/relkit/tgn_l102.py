"""Visible one-layer TGN-attn port for L102 (Rossi et al., 2020).

Apache-2.0 source reference: twitter-research/tgn e38cdf85998c6ca077167610dc4e769a688efa95.
This focused port supports the released Wikipedia TGN-attn setting, not all TGN variants.
"""
# %% Imports and immutable experiment identities
import copy
import hashlib
import json
import math
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import average_precision_score, roc_auc_score

SOURCE_COMMIT = 'e38cdf85998c6ca077167610dc4e769a688efa95'
DATA_URL = 'https://snap.stanford.edu/jodie/wikipedia.csv'
DATA_SHA256 = 'a6b73e09c0d1e5b9db11e7e7aa416f2e87838a745273e0446a79952cf4cfae09'
PAPER_AP = {'all': 98.46, 'new': 97.81}
CLOSE_TOLERANCE_PP = 0.5

# %% Task 1: last-message aggregation

def last_message_indices(node_ids):
    """Last occurrence per node, returned in sorted-node order (chronological input)."""
    last = {}
    for i, node in enumerate(node_ids):
        last[int(node)] = i
    return np.array([last[node] for node in sorted(last)], dtype=np.int64)

# %% Task 2: raw interaction messages

def raw_messages(own, other, edge, elapsed, time_encoder):
    """Eq. 1 identity message: [own state, other state, event features, elapsed encoding]."""
    return torch.cat([own, other, edge, time_encoder(elapsed[:, None]).squeeze(1)], dim=1)

# %% Task 3: differentiable memory update

def update_state(memory, last_update, nodes, messages, timestamps, gru):
    """New candidate state; do not mutate the pre-event state that messages refer to."""
    state, clock = memory.detach().clone(), last_update.detach().clone()
    if len(nodes):
        assert bool((clock[nodes] <= timestamps).all()), 'Memory would travel backward in time'
        state[nodes] = gru(messages, state[nodes])
        clock[nodes] = timestamps
    return state, clock

# %% Strict temporal adjacency and release negative sampling

def temporal_neighbors(u, v, t, edge_ids, n_nodes):
    adjacency = [[] for _ in range(n_nodes)]
    for a, b, when, eid in zip(u, v, t, edge_ids):
        adjacency[int(a)].append((int(b), int(eid), float(when)))
        adjacency[int(b)].append((int(a), int(eid), float(when)))
    adjacency = [np.asarray(sorted(rows, key=lambda x: x[2]), dtype=np.float64).reshape(-1, 3)
                 for rows in adjacency]
    def find(nodes, cutoffs, count):
        neighbors = np.zeros((len(nodes), count), dtype=np.int64)
        ids = np.zeros_like(neighbors)
        times = np.zeros(neighbors.shape, dtype=np.float32)
        for row, (node, cutoff) in enumerate(zip(nodes, cutoffs)):
            history = adjacency[int(node)]
            end = np.searchsorted(history[:, 2], cutoff, side='left')
            chosen = history[max(0, end-count):end]
            if len(chosen):
                neighbors[row, -len(chosen):] = chosen[:, 0]
                ids[row, -len(chosen):] = chosen[:, 1]
                times[row, -len(chosen):] = chosen[:, 2]
        return neighbors, ids, times
    return find


class NegativeSampler:
    """Uniform destinations; consume the unused source draw to preserve release RNG order."""
    def __init__(self, events, seed=None):
        self.sources = np.unique(events['u'])
        self.destinations = np.unique(events['v'])
        self.seed = seed
        self.reset()
    def reset(self):
        self.rng = np.random if self.seed is None else np.random.RandomState(self.seed)
    def sample(self, size):
        self.rng.randint(0, len(self.sources), size)
        return self.destinations[self.rng.randint(0, len(self.destinations), size)]

# %% Time encoder and merge blocks

class TimeEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Linear(1, dim)
        self.w.weight = nn.Parameter(torch.tensor(1 / 10 ** np.linspace(0, 9, dim), dtype=torch.float32)[:, None])
        self.w.bias = nn.Parameter(torch.zeros(dim))
    def forward(self, elapsed):
        return torch.cos(self.w(elapsed.unsqueeze(-1)))


class Merge(nn.Module):
    def __init__(self, left, right, hidden, out):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(left+right, hidden), nn.Linear(hidden, out)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, left, right):
        return self.fc2(torch.relu(self.fc1(torch.cat([left, right], dim=-1))))

# %% Complete one-layer TGN-attn encoder and decoder

class TGN(nn.Module):
    def __init__(self, node_features, edge_features, dropout=0.1, neighbors=10):
        super().__init__()
        d = node_features.shape[1]
        self.dim, self.neighbors = d, neighbors
        self.register_buffer('node_features', torch.as_tensor(node_features, dtype=torch.float32))
        self.register_buffer('edge_features', torch.as_tensor(edge_features, dtype=torch.float32))
        self.time_encoder = TimeEncoder(d)
        self.gru = nn.GRUCell(3*d+edge_features.shape[1], d)
        self.merge = Merge(2*d, d, d, d)
        self.attention = nn.MultiheadAttention(2*d, 2, kdim=2*d+edge_features.shape[1],
                                              vdim=2*d+edge_features.shape[1], dropout=dropout)
        self.decoder = Merge(d, d, d, 1)
        self.register_buffer('memory', torch.zeros_like(self.node_features))
        self.register_buffer('last_update', torch.zeros(len(node_features)))
        self.pending = {}
        self.finder = None
    def reset_state(self):
        self.memory.zero_(); self.last_update.zero_(); self.pending = {}
    def detach_state(self):
        self.memory = self.memory.detach()
        self.pending = {k: (m.detach(), t.detach()) for k, (m, t) in self.pending.items()}
    def snapshot(self):
        return (self.memory.detach().clone(), self.last_update.clone(),
                {k: (m.detach().clone(), t.clone()) for k, (m, t) in self.pending.items()})
    def restore(self, state):
        self.memory, self.last_update, self.pending = copy.deepcopy(state)
    def candidate_state(self):
        nodes = sorted(self.pending)
        messages = torch.stack([self.pending[k][0] for k in nodes]) if nodes else None
        times = torch.stack([self.pending[k][1] for k in nodes]) if nodes else None
        return update_state(self.memory, self.last_update, nodes, messages, times, self.gru)
    def embed(self, nodes, timestamps, state):
        device = self.memory.device
        node_ids = torch.as_tensor(nodes, dtype=torch.long, device=device)
        own = state[node_ids] + self.node_features[node_ids]
        neighbors, edges, times = self.finder(nodes, timestamps, self.neighbors)
        neighbors = torch.as_tensor(neighbors, dtype=torch.long, device=device)
        edges = torch.as_tensor(edges, dtype=torch.long, device=device)
        elapsed = torch.as_tensor(timestamps[:, None] - times, dtype=torch.float32, device=device)
        query = torch.cat([own[:, None], self.time_encoder(torch.zeros((len(nodes), 1), device=device))], dim=-1)
        key = torch.cat([state[neighbors]+self.node_features[neighbors], self.edge_features[edges],
                         self.time_encoder(elapsed)], dim=-1)
        mask = neighbors == 0
        empty = mask.all(dim=1)
        mask[empty, 0] = False  # MHA softmax must not receive an all-masked row.
        output, _ = self.attention(query.transpose(0, 1), key.transpose(0, 1), key.transpose(0, 1),
                                   key_padding_mask=mask)
        output = output.squeeze(0).masked_fill(empty[:, None], 0)
        return self.merge(output, own)
    def probabilities(self, u, v, negative, timestamps, edge_ids):
        """Score first using old messages; queue current positive events only afterward."""
        state, clock = self.candidate_state()
        embeddings = self.embed(np.concatenate([u, v, negative]), np.tile(timestamps, 3), state)
        src, dst, neg = embeddings.chunk(3)
        # Persist consumed messages only for real endpoints, exactly as release TGN does.
        positives = np.unique(np.concatenate([u, v]))
        with torch.no_grad():
            self.memory[positives] = state[positives]
            self.last_update[positives] = clock[positives]
        for node in positives:
            self.pending.pop(int(node), None)
        device = self.memory.device
        when = torch.as_tensor(timestamps, dtype=torch.float32, device=device)
        # Source and destination messages both use the same pre-current-event state.
        for own, other in [(u, v), (v, u)]:
            messages = raw_messages(self.memory[own], self.memory[other], self.edge_features[edge_ids],
                                    when-self.last_update[own], self.time_encoder)
            for idx in last_message_indices(own):
                self.pending[int(own[idx])] = (messages[idx], when[idx])
        scores = self.decoder(torch.cat([src, src]), torch.cat([dst, neg])).flatten().sigmoid()
        return scores.chunk(2)

# %% Complete data preprocessing and frozen release split

def load_wikipedia(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    path = directory/'wikipedia.csv'
    if not path.exists():
        urllib.request.urlretrieve(DATA_URL, path)
    sha = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    assert sha == DATA_SHA256, f'Dataset checksum mismatch: {sha}'
    cache = directory/'processed.npz'
    if cache.exists():
        z = np.load(cache)
        # The raw identity alone cannot authenticate an arbitrary processed cache.
        if str(z['raw_sha']) != sha:
            raise ValueError('Processed cache belongs to other raw bytes')
        u,v,t,x = (z[k] for k in ['u','v','t','x'])
    else:
        # Header has four columns, rows additionally have 172 edge features.
        a = np.loadtxt(path, delimiter=',', skiprows=1)
        u = a[:, 0].astype(np.int64)+1
        v = a[:, 1].astype(np.int64)+int(u.max())+1
        t,x = a[:, 2], a[:, 4:].astype(np.float32)
        np.savez(cache, u=u,v=v,t=t,x=x,raw_sha=sha)
    assert len(u)==157474 and x.shape==(157474,172)
    assert np.all(np.diff(t)>=0) and len(set(u)|set(v))==9227
    edges = np.vstack([np.zeros((1,172),np.float32),x])
    nodes = np.zeros((int(max(u.max(),v.max()))+1,172),np.float32)
    full = {'u':u,'v':v,'t':t,'e':np.arange(1,len(u)+1)}
    val_time,test_time = np.quantile(t,[.70,.85])
    candidates = set(u[t>val_time]) | set(v[t>val_time])
    # Python 3.9 random.sample(set, k) internally converted to tuple. Preserve that order.
    held = set(random.Random(2020).sample(tuple(candidates),int(.1*(len(nodes)-1))))
    train_mask = (t<=val_time)&~np.isin(u,list(held))&~np.isin(v,list(held))
    seen = set(u[train_mask])|set(v[train_mask]); unseen=(set(u)|set(v))-seen
    novel = np.isin(u,list(unseen))|np.isin(v,list(unseen))
    masks={'train':train_mask,'val':(t>val_time)&(t<=test_time),'test':t>test_time}
    masks['new_val']=masks['val']&novel; masks['new_test']=masks['test']&novel
    data={name:{k:a[mask] for k,a in full.items()} for name,mask in masks.items()}
    data['full']=full
    audit={'raw_sha256':sha,'cutoffs':[float(val_time),float(test_time)],
           'counts':{k:len(d['u']) for k,d in data.items()},'held_nodes':sorted(int(i) for i in held),
           'split_sha256':{k:hashlib.sha256(d['e'].tobytes()).hexdigest() for k,d in data.items()}}
    return nodes,edges,data,audit

# %% Full chronological train/evaluation loops and named-target runner

def batches(events, size=200):
    for start in range(0,len(events['u']),size):
        yield {k:v[start:start+size] for k,v in events.items()}


def evaluate(model, events, sampler, batch_size=200):
    model.eval(); sampler.reset(); aps=[]; aucs=[]; saved=[]
    with torch.no_grad():
        for b in batches(events,batch_size):
            neg=sampler.sample(len(b['u']))
            positive,negative=model.probabilities(b['u'],b['v'],neg,b['t'],b['e'])
            p=np.concatenate([positive.cpu().numpy(),negative.cpu().numpy()])
            y=np.concatenate([np.ones(len(positive)),np.zeros(len(negative))])
            aps.append(float(average_precision_score(y,p))); aucs.append(float(roc_auc_score(y,p)))
            saved.append({'edges':b['e'],'negative':neg,'positive':positive.cpu().numpy(),'negative_score':negative.cpu().numpy()})
    return {'ap':float(np.mean(aps)),'auc':float(np.mean(aucs)),'batch_ap':aps,'batch_auc':aucs},saved


def run_training(nodes,edges,data,seed=0,epochs=50,patience=5,batch_size=200,output=None,device='cpu'):
    """Fixed-seed modern port. Release checkpoint/pending-message behavior is preserved."""
    torch.manual_seed(seed); np.random.seed(seed)
    model=TGN(nodes,edges).to(device)
    full_finder=temporal_neighbors(**{'u':data['full']['u'],'v':data['full']['v'],'t':data['full']['t'],
                                    'edge_ids':data['full']['e'],'n_nodes':len(nodes)})
    train=data['train']
    train_finder=temporal_neighbors(train['u'],train['v'],train['t'],train['e'],len(nodes))
    train_sampler=NegativeSampler(train)
    val_sampler=NegativeSampler(data['full'],0)
    test_sampler=NegativeSampler(data['full'],2)
    new_sampler=NegativeSampler(data['new_test'],3)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    trace=[]; best=-float('inf'); rounds=0; best_epoch=0; best_weights=None; stop=False
    for epoch in range(epochs):
        start=time.perf_counter();model.reset_state();model.finder=train_finder;model.train();losses=[]
        for b in batches(train,batch_size):
            optimizer.zero_grad()
            p,n=model.probabilities(b['u'],b['v'],train_sampler.sample(len(b['u'])),b['t'],b['e'])
            loss=nn.functional.binary_cross_entropy(p,torch.ones_like(p))+nn.functional.binary_cross_entropy(n,torch.zeros_like(n))
            loss.backward();optimizer.step();model.detach_state();losses.append(float(loss.detach()))
        model.finder=full_finder
        train_state=model.snapshot()
        val,_=evaluate(model,data['val'],val_sampler,batch_size)
        val_state=model.snapshot();model.restore(train_state)
        # The released trainer intentionally passes the full-node validation sampler here.
        new_val,_=evaluate(model,data['new_val'],val_sampler,batch_size)
        model.restore(val_state)
        improved=best_weights is None or (val['ap']-best)/abs(best)>1e-10
        if improved:
            best=val['ap'];best_epoch=epoch;rounds=0
            best_weights={k:v.detach().clone() for k,v in model.state_dict().items()
                          if k not in ['node_features','edge_features']}
        else: rounds+=1
        row={'epoch':epoch,'loss':float(np.mean(losses)),'val_ap':val['ap'],'new_val_ap':new_val['ap'],
             'seconds':time.perf_counter()-start}
        trace.append(row);print(json.dumps({'seed':seed,**row}),flush=True)
        if output:
            Path(output).mkdir(parents=True,exist_ok=True)
            (Path(output)/f'seed-{seed}-progress.json').write_text(json.dumps(trace,indent=2))
        if rounds>=patience:
            # Release does not serialize pending messages in state_dict. Keep stopping-epoch
            # pending messages when restoring its best weights and persistent memory.
            model.load_state_dict(best_weights,strict=False);stop=True;break
    # Release uses the final epoch if the maximum is reached without early stopping.
    selected=best_epoch if stop else len(trace)-1
    val_state=model.snapshot()
    test,predictions=evaluate(model,data['test'],test_sampler,batch_size)
    model.restore(val_state)
    new_test,new_predictions=evaluate(model,data['new_test'],new_sampler,batch_size)
    model.restore(val_state)
    result={'seed':seed,'epochs_completed':len(trace),'selected_epoch':selected,'early_stopped':stop,
            'test':test,'new_test':new_test,'trace':trace,
            'seed_protocol':'Independent explicit seed per run; release originally seeds once before ten runs',
            'checkpoint_policy':'release: best state_dict on early stop; pending messages from stopping epoch'}
    if output:
        out=Path(output)
        torch.save({'weights':{k:v for k,v in model.state_dict().items() if k not in ['node_features','edge_features']},
                    'temporal_state':model.snapshot(),'result':result},out/f'seed-{seed}.pt')
        arrays={f'{lane}_{k}':np.concatenate([b[k] for b in values])
                for lane,values in [('all',predictions),('new',new_predictions)] for k in values[0]}
        np.savez_compressed(out/f'seed-{seed}-predictions.npz',**arrays)
        (out/f'seed-{seed}.json').write_text(json.dumps(result,indent=2))
    return result
