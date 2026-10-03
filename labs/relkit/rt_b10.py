"""RT-v1 dense teaching mirror. CPU math backend; no checkpoint or benchmark claim.

Shapes: batch B, sampled cells S, hidden width D, heads H, per-head width d.
IDs encode relationships, never ordinal position. Values are finite preprocessed inputs.
"""
import torch
from torch import nn
import torch.nn.functional as F
TYPES = ('number', 'text', 'datetime', 'boolean')

def relational_masks(batch):
    """Return B×S×S Boolean permissions; rows are readers, columns are sources."""
    node = batch['node_idxs']; parents = batch['f2p_nbr_idxs']
    valid = ~batch['is_padding']
    pair = valid[:, :, None] & valid[:, None, :]
    same_row = node[:, :, None] == node[:, None, :]
    to_parent = (node[:, None, :, None] == parents[:, :, None, :]).any(-1)
    from_child = to_parent.transpose(1, 2)
    same_column = ((batch['table_name_idxs'][:, :, None] == batch['table_name_idxs'][:, None, :]) &
                   (batch['col_name_idxs'][:, :, None] == batch['col_name_idxs'][:, None, :]))
    return {'col': same_column & pair, 'feat': (same_row | to_parent) & pair,
            'nbr': from_child & pair, 'full': pair}

def safe_attention(q, k, v, allowed):
    """Dense scaled dot product; empty permission rows have exactly zero output.

    q/k/v are B×H×S×d; allowed is B×S×S. No softmax over a row of -inf.
    """
    scores = q @ k.transpose(-2, -1) / q.shape[-1] ** 0.5
    permit = allowed[:, None]
    has_key = permit.any(-1, keepdim=True)
    scores = scores.masked_fill(~permit, -torch.inf)
    scores = torch.where(has_key, scores, torch.zeros_like(scores))
    weights = torch.softmax(scores, dim=-1) * permit
    return weights @ v

def eligible_rows(event_time, available_at, cutoff, is_label, horizon):
    """Strict COURSE visibility policy, not a repair of RT's released sampler.

    Units must match. Unknown event/arrival times fail closed. Labels become usable
    only when both their outcome window has closed and their availability time passed.
    """
    known = torch.isfinite(event_time) & torch.isfinite(available_at)
    return known & (event_time <= cutoff) & (available_at <= cutoff) & (~is_label | (event_time + horizon <= cutoff))

class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__(); self.num_heads = num_heads
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
    def forward(self, x, block_mask):
        b, s, d = x.shape; h = self.num_heads
        q, k, v = [layer(x).reshape(b,s,h,d//h).transpose(1,2) for layer in (self.wq,self.wk,self.wv)]
        mixed = safe_attention(q,k,v,block_mask)
        return self.wo(mixed.transpose(1,2).reshape(b,s,d))

class FFN(nn.Module):
    """SwiGLU: one projected stream gates another; project back to D."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1=nn.Linear(d_model,d_ff,bias=False)
        self.w2=nn.Linear(d_ff,d_model,bias=False)
        self.w3=nn.Linear(d_model,d_ff,bias=False)
    def forward(self,x):return self.w2(F.silu(self.w1(x))*self.w3(x))

class RelationalBlock(nn.Module):
    def __init__(self,d_model,num_heads,d_ff):
        super().__init__()
        self.norms=nn.ModuleDict({k:nn.RMSNorm(d_model) for k in ('feat','nbr','col','full','ffn')})
        self.attns=nn.ModuleDict({k:MaskedAttention(d_model,num_heads) for k in ('feat','nbr','col','full')})
        self.ffn=FFN(d_model,d_ff)
    def forward(self,x,masks):
        for kind in ('col','feat','nbr','full'):
            x=x+self.attns[kind](self.norms[kind](x),masks[kind])
        return x+self.ffn(self.norms['ffn'](x))

class RelationalTransformer(nn.Module):
    """Same parameter names and operations as pinned RT-v1; dense CPU attention."""
    def __init__(self,num_blocks,d_model,d_text,num_heads,d_ff):
        super().__init__()
        dims={'number':1,'text':d_text,'datetime':1,'col_name':d_text,'boolean':1}
        self.enc_dict=nn.ModuleDict({t:nn.Linear(n,d_model) for t,n in dims.items()})
        self.dec_dict=nn.ModuleDict({t:nn.Linear(d_model,dims[t]) for t in TYPES})
        self.norm_dict=nn.ModuleDict({t:nn.RMSNorm(d_model) for t in dims})
        self.mask_embs=nn.ParameterDict({t:nn.Parameter(torch.randn(d_model)) for t in TYPES})
        self.blocks=nn.ModuleList([RelationalBlock(d_model,num_heads,d_ff) for _ in range(num_blocks)])
        self.norm_out=nn.RMSNorm(d_model);self.d_model=d_model
    def encode(self,batch):
        valid=~batch['is_padding'];hidden=batch['masks']
        x=self.norm_dict['col_name'](self.enc_dict['col_name'](batch['col_name_values']))*valid[...,None]
        for i,t in enumerate(TYPES):
            active=(batch['sem_types']==i)&valid
            values=self.norm_dict[t](self.enc_dict[t](batch[t+'_values']))
            x=x+values*(active&~hidden)[...,None]+self.mask_embs[t]*(active&hidden)[...,None]
        return x
    def forward(self,batch):
        if (batch['masks'] & batch['is_padding']).any():raise ValueError('Cannot supervise padding')
        if not batch['masks'].any():raise ValueError('At least one masked target required')
        if (batch['masks'] & (batch['sem_types']==1)).any():raise ValueError('masking text not supported')
        x=self.encode(batch);masks=relational_masks(batch)
        for block in self.blocks:x=block(x,masks)
        x=self.norm_out(x);loss=x.new_zeros(());pred={}
        for i,t in enumerate(TYPES):
            pred[t]=self.dec_dict[t](x)
            selected=(batch['sem_types']==i)&batch['masks']
            if not selected.any():
                loss=loss+pred[t].sum()*0;continue
            if t=='boolean':
                error=F.binary_cross_entropy_with_logits(pred[t],(batch[t+'_values']>0).float(),reduction='none').mean(-1)
            else:error=F.huber_loss(pred[t],batch[t+'_values'],reduction='none').mean(-1)
            loss=loss+(error*selected).sum()
        return loss/batch['masks'].sum(),pred

def train_step(model,batch,optimizer):
    """Visible local optimization step, not RT's distributed pretraining recipe."""
    model.train();optimizer.zero_grad(set_to_none=True)
    loss,_=model(batch);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
    optimizer.step();return float(loss.detach())
