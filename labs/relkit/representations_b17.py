"""Reduced FlexTab-shaped dependency diagnostic; not pretrained FlexTab.

Float64, single-head attention, numeric tokenization, no dropout/optimization.
Paper-inspired wiring is explicit; no released-source equivalence is claimed.
"""
import hashlib
import json
import math
import torch
from torch import nn


def attention(q, k, v):
    """Scaled dot products → probabilities over keys → weighted values."""
    if k.shape[-2] == 0 or k.shape[-2] != v.shape[-2]:
        raise ValueError('Require a nonempty aligned key/value sequence')
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    return torch.softmax(scores, dim=-1) @ v


def aggregate_layers(rows, projections, final):
    """Each layer has its own projection BEFORE the sum and final map."""
    if not rows or len(rows) != len(projections):
        raise ValueError('One projection per encoder layer is required')
    return final(torch.stack([p(r) for r, p in zip(rows, projections)]).sum(0))


def cache_identity(x, ids, support_size, state, preprocessing):
    """Conservative whole-table key, including row order and encoder state.

    Labels are intentionally absent. State must be encoder state only.
    In production also version implementation, masks, tokenizer and precision.
    """
    if len(ids) != len(x) or len(set(ids)) != len(ids) or not 0 < support_size <= len(x):
        raise ValueError('Unique aligned IDs and nonempty support required')
    h = hashlib.sha256()
    h.update(json.dumps([ids, support_size, preprocessing, 'b17-encoder-v1'], separators=(',', ':')).encode())
    for name, value in [('features', x), *sorted(state.items())]:
        t = value.detach().cpu().contiguous()
        h.update(json.dumps([name, list(t.shape), str(t.dtype)]).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


class Attention(nn.Module):
    """Visible single-head Q/K/V/output projections; no attention library."""
    def __init__(self, width):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, q, kv):
        return self.out(attention(self.q(q), self.k(kv), self.v(kv)))


class EncoderBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.columns = Attention(width)
        self.rows = Attention(width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, t, support_size):
        # t: [rows, columns+1, width]; last position is [ROW].
        a = self.norms[0](t)
        t = t + self.columns(a, a[:, :-1])  # no feature reads [ROW]
        a = self.norms[1](t).transpose(0, 1)  # [columns+1, rows, width]
        t = t + self.rows(a, a[:, :support_size]).transpose(0, 1)
        return t + self.ff(self.norms[2](t))


class Encoder(nn.Module):
    def __init__(self, columns=2, width=16, depth=2):
        super().__init__()
        self.numeric = nn.Linear(1, width)
        self.column = nn.Parameter(torch.randn(columns, width)*.1)
        self.row = nn.Parameter(torch.randn(width)*.1)
        self.blocks = nn.ModuleList([EncoderBlock(width) for _ in range(depth)])
        self.projections = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.final = nn.Linear(width, width)

    def forward(self, x, support_size):
        # No labels argument: targets cannot enter encoder inference.
        t = self.numeric(x.unsqueeze(-1)) + self.column
        t = torch.cat([t, self.row.expand(len(x), 1, -1)], dim=1)
        rows = []
        for block in self.blocks:
            t = block(t, support_size)
            rows.append(t[:, -1])
        return aggregate_layers(rows, self.projections, self.final)


class DecoderBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.features = Attention(width)
        self.targets = Attention(width)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width, width))

    def forward(self, t, z, support_size):
        # Each target reads its own row embedding (one key); target-stream
        # attention then propagates support information to query targets.
        t = t + self.features(self.norms[0](t).unsqueeze(1), z.unsqueeze(1)).squeeze(1)
        a = self.norms[1](t)
        t = t + self.targets(a, a[:support_size])
        return t + self.ff(self.norms[2](t))


class Decoder(nn.Module):
    def __init__(self, width=16, depth=2):
        super().__init__()
        self.label = nn.Embedding(3, width)  # 0, 1, query MASK=2
        self.blocks = nn.ModuleList([DecoderBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, 2)

    def forward(self, z, support_labels):
        n = len(support_labels)
        tokens = torch.cat([support_labels, torch.full((len(z)-n,), 2, dtype=torch.long)])
        t = self.label(tokens)
        for block in self.blocks:
            t = block(t, z, n)
        return self.head(t[n:]).softmax(-1)[:, 1]


def fixture():
    x = torch.tensor([[-1,0],[0,1],[1,0],[0,-1],[.5,.5],[-.5,-.5]], dtype=torch.float64)
    return x, ['s0','s1','s2','s3','q0','q1'], {'A':[0,0,1,1], 'B':[0,1,0,1]}


def initialized(seed):
    torch.manual_seed(seed)
    return Encoder().double().eval(), Decoder().double().eval()


def run_experiment():
    """Run all declared interventions; save enough state for independent replay."""
    torch.set_num_threads(1)
    x, ids, tasks = fixture()
    result = dict(name='B17-REPRESENTATION-BOUNDARY', status='COMPLETE_MECHANISM_DIAGNOSTIC',
                  inputs=x.tolist(), ids=ids, support_size=4, tasks=tasks, seeds=[],
                  precision='float64', training='NOT_RUN', query_labels_reporting_only=[1,0])
    for seed in [0,1,2]:
        enc, dec = initialized(seed)
        state = {name:{k:v.detach().tolist() for k,v in model.state_dict().items()} for name,model in [('encoder',enc),('decoder',dec)]}
        rec = dict(seed=seed, state=state, records=[], checks=[])
        with torch.no_grad():
            z = enc(x,4)
            base_key = cache_identity(x,ids,4,enc.state_dict(),'numeric-v1')
            cached = {base_key:z.clone()}
            for task, values in tasks.items():
                y = torch.tensor(values,dtype=torch.long)
                baseline = dec(z,y)
                for mode in ['baseline','support-labels','support-feature','other-query','support-order','query-order']:
                    changed=x.clone();labels=y.clone();order=list(range(6))
                    if mode=='support-labels':labels=1-labels
                    if mode=='support-feature':changed[0,0]+=2
                    if mode=='other-query':changed[5]=torch.tensor([7.,-3.])
                    if mode=='support-order':order=[2,0,3,1,4,5];labels=labels[order[:4]]
                    if mode=='query-order':order=[0,1,2,3,5,4]
                    changed=changed[order];newids=[ids[i] for i in order]
                    key=cache_identity(changed,newids,4,enc.state_dict(),'numeric-v1')
                    fresh=enc(changed,4);hit=key in cached
                    used=cached[key] if hit else fresh
                    pred=dec(used,labels);direct=dec(fresh,labels)
                    assert torch.allclose(pred,direct,atol=1e-12,rtol=0)
                    aligned_z=fresh[torch.argsort(torch.tensor(order))]
                    q_order=[i-4 for i in order[4:]]
                    aligned_p=pred[torch.argsort(torch.tensor(q_order))]
                    dz=float((aligned_z-z).abs().max());dp=float((aligned_p-baseline).abs().max())
                    if mode in ['baseline','support-labels']:assert dz==0 and hit
                    if mode=='support-labels':assert dp>1e-9
                    if mode=='support-feature':assert dz>1e-9 and not hit
                    if mode=='other-query':
                        assert torch.allclose(fresh[:5],z[:5],atol=1e-12,rtol=0)
                        assert abs(float(pred[0]-baseline[0]))<1e-12
                    if mode in ['support-order','query-order']:assert dz<1e-12 and dp<1e-12
                    rec['records'].append(dict(task=task,mode=mode,inputs=changed.tolist(),ids=newids,
                        support_labels=labels.tolist(),embedding=fresh.tolist(),probability=pred.tolist(),
                        cache_key=key,cache_hit=hit,embedding_delta=dz,prediction_delta=dp,
                        q0_probability=float(aligned_p[0]),cache_fresh_delta=float((pred-direct).abs().max())))
                # All four possible externally held query-label assignments.
                for external in [[0,0],[0,1],[1,0],[1,1]]:
                    assert torch.equal(dec(enc(x,4),y),baseline)
                    rec['checks'].append(dict(task=task,external_query_labels=external,prediction=baseline.tolist(),unchanged=True))
            rec['cache_invalidation']={}
            for mode in ['weights','preprocessing']:
                st={k:v.clone() for k,v in enc.state_dict().items()}
                if mode=='weights':st['numeric.weight'][0,0]+=.1
                key=cache_identity(x,ids,4,st,'numeric-v2' if mode=='preprocessing' else 'numeric-v1')
                rec['cache_invalidation'][mode]=key!=base_key
                assert key!=base_key
        result['seeds'].append(rec)
    return result
