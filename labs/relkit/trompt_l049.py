"""L049 complete numeric Trompt mirror, paper v2 Fig.2–4 and Eq.1–9.

Zero initial state follows §5.1. Dense expansion is a bias-free 1→P map
per scalar and GroupNorm uses two groups, following pinned PyTorch Frame's
independent implementation; these underspecified paper details are choices,
not author-code parity. No categorical encoder or benchmark reproduction.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .claim_models import prompt_weights, prompt_reduce


class TromptCell(nn.Module):
    """Separate column identities, row values, and recurrent prompt state."""
    def __init__(self, features, d, prompts, groups=2):
        super().__init__()
        if prompts % groups: raise ValueError('prompts must divide into GroupNorm groups')
        self.prompts=nn.Parameter(torch.randn(prompts,d)*.01)
        self.columns=nn.Parameter(torch.randn(features,d)*.01)
        self.prompt_norm=nn.LayerNorm(d)
        self.column_norm=nn.LayerNorm(d)
        self.fusion=nn.Linear(2*d,d)
        self.value_weight=nn.Parameter(torch.empty(features,d))
        self.value_bias=nn.Parameter(torch.zeros(features,d))
        nn.init.normal_(self.value_weight,std=.01)
        self.value_norm=nn.LayerNorm(d)
        self.expansion_weight=nn.Parameter(torch.randn(prompts)*.01)
        self.group_norm=nn.GroupNorm(groups,prompts)

    def forward(self,x,previous,return_trace=False):
        p=self.prompt_norm(self.prompts)[None].expand(len(x),-1,-1)
        fused=self.fusion(torch.cat([p,previous],dim=-1))+p+previous
        weights=prompt_weights(fused,self.column_norm(self.columns))
        values=self.value_norm(F.relu(x[...,None]*self.value_weight+self.value_bias))
        # Each scalar becomes P learned scalings. GroupNorm normalizes within
        # each row over (P/groups, C, d), with learned affine per prompt.
        expanded=F.relu(values[:,None]*self.expansion_weight[None,:,None,None])
        expanded=self.group_norm(expanded)+values[:,None]
        output=prompt_reduce(weights,expanded)
        if return_trace:return output,dict(weights=weights,expanded=expanded,output=output)
        return output


class TromptHead(nn.Module):
    """One shared downstream head; softmax now reduces prompts P, not columns C."""
    def __init__(self,d,classes):
        super().__init__()
        self.prompt_score=nn.Linear(d,1)
        self.hidden=nn.Linear(d,d)
        self.norm=nn.LayerNorm(d)
        self.output=nn.Linear(d,classes)

    def forward(self,state):
        weights=self.prompt_score(state).softmax(dim=1)
        pooled=(weights*state).sum(dim=1)
        return self.output(self.norm(F.relu(self.hidden(pooled))))


class Trompt(nn.Module):
    """Return [B,L,T] logits; every cell rereads x and passes state to the next."""
    def __init__(self,features,d=16,prompts=8,layers=2,classes=2,groups=2):
        super().__init__()
        if layers<1:raise ValueError('at least one cell is required')
        self.d,self.num_prompts=d,prompts
        self.cells=nn.ModuleList([TromptCell(features,d,prompts,groups) for _ in range(layers)])
        self.head=TromptHead(d,classes)

    def forward(self,x,return_trace=False):
        state=x.new_zeros(len(x),self.num_prompts,self.d)
        logits,trace=[],[]
        for cell in self.cells:
            state,detail=cell(x,state,return_trace=True)
            logits.append(self.head(state));trace.append(detail)
        result=torch.stack(logits,dim=1)
        return (result,trace) if return_trace else result


def trompt_loss(cell_logits,y):
    """Eq.9: sum L mean-over-row CE losses (do not first average logits)."""
    return sum(F.cross_entropy(cell_logits[:,i],y) for i in range(cell_logits.shape[1]))


def train_trompt(model,x,y,train,valid,*,seed=49,epochs=20,batch=64,lr=.001,patience=8,device='cpu'):
    """Local fixed recipe; checkpoint chosen by validation CE of mean logits."""
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=model.to(device)
    xt=torch.as_tensor(x,dtype=torch.float32,device=device)
    yt=torch.as_tensor(y,dtype=torch.long,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0)
    best,best_state,stale,history=float('inf'),None,0,[]
    for epoch in range(epochs):
        model.train()
        for ids in np.array_split(rng.permutation(train),max(1,math.ceil(len(train)/batch))):
            loss=trompt_loss(model(xt[ids]),yt[ids])
            optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():score=float(F.cross_entropy(model(xt[valid]).mean(1),yt[valid]))
        history.append({'epoch':epoch+1,'valid_log_loss':score})
        if score<best:best,best_state,stale=score,copy.deepcopy(model.state_dict()),0
        else:stale+=1
        if stale>=patience:break
    model.load_state_dict(best_state);model.eval()
    return model,history


@torch.no_grad()
def predict_trompt(model,x,batch=256):
    """Chosen Eq.9 convention: average cell logits, then class softmax."""
    device=next(model.parameters()).device
    return np.concatenate([model(torch.as_tensor(a,dtype=torch.float32,device=device)).mean(1).softmax(-1).cpu().numpy()
        for a in np.array_split(x,max(1,math.ceil(len(x)/batch)))])
