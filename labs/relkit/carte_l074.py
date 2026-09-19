"""Visible CARTE readout architecture, matching release f54690d (dropout disabled).

The lab uses initial_x, initial_e and read_out_block from the YAGO checkpoint.
It intentionally loads initial_x, unlike the release estimator's renamed-key path.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import PowerTransformer,StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import rankdata,friedmanchisquare,studentized_range
from catboost import CatBoostRegressor


def make_graph(row, vectors):
    """A source-oriented star: index[0] receives, index[1] supplies each message."""
    leaves,edges=[],[]
    for column,value in row.items():
        if pd.isna(value):continue
        edge=np.asarray(vectors[column],dtype=np.float32)
        node=np.asarray(vectors[str(value).lower()],dtype=np.float32) if isinstance(value,str) else float(value)*edge
        leaves.append(node);edges.append(edge)
    if not leaves:raise ValueError('All-missing row: no observed leaf; choose an explicit fallback')
    leaves=torch.tensor(np.stack(leaves));edges=torch.tensor(np.stack(edges))
    center=(leaves*edges).mean(0,keepdim=True)
    x=torch.cat([center,leaves]);n=len(leaves);ids=torch.arange(1,n+1)
    index=torch.stack([torch.cat([torch.zeros(n,dtype=torch.long),ids]),torch.cat([ids,ids])])
    attrs=torch.cat([edges,torch.ones_like(edges)])
    return x,index,attrs


def grouped_attention(edge_index, query, key, value):
    """Normalize over incoming messages separately for each receiving node."""
    receiver=edge_index[0]
    logits=(query[receiver]*key).sum(-1)/math.sqrt(query.shape[-1])
    maxima=torch.full((len(query),),-torch.inf,dtype=query.dtype,device=query.device)
    maxima=maxima.scatter_reduce(0,receiver,logits,reduce='amax',include_self=True)
    weights=(logits-maxima[receiver]).exp()
    totals=torch.zeros_like(maxima).index_add(0,receiver,weights)
    weights=weights/totals[receiver]
    output=torch.zeros_like(query).index_add(0,receiver,weights[:,None]*value)
    return output,weights


class Attention(nn.Module):
    def __init__(self,d=300,heads=12):
        super().__init__();self.heads=heads
        self.lin_query=nn.Linear(d,d,bias=False)
        self.lin_key=nn.Linear(d,d,bias=False)
        self.lin_value=nn.Linear(d,d,bias=False)
    def forward(self,x,index,edge):
        z=edge*x[index[1]]
        q,k,v=self.lin_query(x),self.lin_key(z),self.lin_value(z)
        width=q.shape[1]//self.heads
        return torch.cat([grouped_attention(index,q[:,i:i+width],k[:,i:i+width],v[:,i:i+width])[0] for i in range(0,q.shape[1],width)],1)


class Readout(nn.Module):
    def __init__(self):
        super().__init__();self.g_attn=Attention()
        self.linear_net_x=nn.Sequential(nn.Linear(300,300),nn.Dropout(0),nn.GELU(),nn.Linear(300,300))
        self.norm1_x=nn.LayerNorm(300);self.norm2_x=nn.LayerNorm(300)
    def forward(self,x,index,edge):
        # The pinned code has no node residual addition in this block.
        x=self.norm1_x(self.g_attn(x,index,edge))
        return self.norm2_x(self.linear_net_x(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_x=nn.Sequential(nn.Linear(300,300),nn.GELU(),nn.LayerNorm(300))
        self.initial_e=nn.Sequential(nn.Linear(300,300),nn.GELU(),nn.LayerNorm(300))
        self.read_out_block=Readout()
    def forward(self,x,index,edge):
        return self.read_out_block(self.initial_x(x),index,self.initial_e(edge))
    def rows(self,x,index,edge,heads):
        # Only center outputs are consumed. Leaf inputs still supply keys/values.
        receiver_map=torch.full((len(x),),-1,dtype=torch.long)
        receiver_map[heads]=torch.arange(len(heads))
        keep=receiver_map[index[0]]>=0
        row_ids=receiver_map[index[0,keep]]
        x=self.initial_x(x);edge=self.initial_e(edge[keep])
        attn=self.read_out_block.g_attn
        z=edge*x[index[1,keep]]
        q,k,v=attn.lin_query(x[heads]),attn.lin_key(z),attn.lin_value(z)
        row_index=torch.stack([row_ids,torch.zeros_like(row_ids)])
        width=q.shape[1]//attn.heads
        output=torch.cat([grouped_attention(row_index,q[:,i:i+width],k[:,i:i+width],v[:,i:i+width])[0] for i in range(0,q.shape[1],width)],1)
        output=self.read_out_block.norm1_x(output)
        return self.read_out_block.norm2_x(self.read_out_block.linear_net_x(output))


def load_encoder(checkpoint,pretrained,seed):
    torch.manual_seed(seed);model=Encoder()
    if pretrained:
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        selected={k:state['ft_base.'+k] for k in model.state_dict()}
        model.load_state_dict(selected,strict=True)
    return model


def batch_graphs(graphs):
    xs,es,attrs,heads=[],[],[],[];offset=0
    for x,index,edge in graphs:
        heads.append(offset);xs.append(x);es.append(index+offset);attrs.append(edge);offset+=len(x)
    return torch.cat(xs),torch.cat(es,1),torch.cat(attrs),torch.tensor(heads)


def embed(model,graphs):
    model.eval();parts=[]
    with torch.no_grad():
        for i in range(0,len(graphs),64):
            x,index,edge,heads=batch_graphs(graphs[i:i+64]);parts.append(model.rows(x,index,edge,heads).numpy())
    return np.concatenate(parts)


def ridge_predict(train,valid,test,ytrain,yvalid):
    """Student TODO: train-only scaling and validation-only alpha selection."""
    scaler=StandardScaler().fit(train)
    train,valid,test=[scaler.transform(z) for z in [train,valid,test]]
    best=None
    for alpha in [1.,10.,100.]:
        model=Ridge(alpha=alpha).fit(train,ytrain)
        loss=float(np.mean((model.predict(valid)-yvalid)**2))
        if best is None or loss<best[0]:best=(loss,alpha,model)
    return best[2].predict(test),best[1]


def tune(model,graphs,y,train,valid,test,epochs,seed):
    torch.manual_seed(seed+900)
    head=nn.Linear(300,1);optimizer=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=1e-4,weight_decay=1e-3)
    mean=float(y[train].mean());scale=float(y[train].std());target=torch.tensor((y-mean)/scale,dtype=torch.float32)
    tensors=[batch_graphs([graphs[i] for i in ids]) for ids in [train,valid,test]]
    best=None
    for epoch in range(epochs):
        model.train();head.train();x,index,edge,h=tensors[0]
        loss=((head(model.rows(x,index,edge,h)).flatten()-target[train])**2).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        model.eval();head.eval()
        with torch.no_grad():
            x,index,edge,h=tensors[1];vl=float(((head(model.rows(x,index,edge,h)).flatten()-target[valid])**2).mean())
        if best is None or vl<best[0]:best=(vl,copy.deepcopy(model.state_dict()),copy.deepcopy(head.state_dict()),epoch+1)
    model.load_state_dict(best[1]);head.load_state_dict(best[2])
    with torch.no_grad():
        x,index,edge,h=tensors[2];prediction=head(model.rows(x,index,edge,h)).flatten().numpy()*scale+mean
    return prediction,best[3]


def normalize_numeric(frame,train):
    """Fit per-column maps using train only; constant columns need no power fit."""
    out=frame.copy();policy={}
    for col in frame.select_dtypes(include='number'):
        observed=frame.iloc[train][col].dropna()
        if len(observed)==0:
            out[col]=np.nan;policy[col]='unobserved_train_omit';continue
        constant=observed.nunique()==1
        transformer=StandardScaler() if constant else PowerTransformer()
        transformer.fit(frame.iloc[train][[col]])
        out[col]=transformer.transform(frame[[col]]).flatten()
        policy[col]='standard_constant' if constant else 'power'
    return out,policy


def run_transfer(data_dir='data/l074',seeds=(0,1,2),epochs=40,train_size=64):
    torch.set_num_threads(1);root=Path(data_dir)
    manifest=json.loads((root/'manifest.json').read_text())
    payload=np.load(root/'vectors.npz');vectors=dict(zip(payload['names'].tolist(),payload['vectors']))
    records=[];splits=[]
    for dataset in manifest['datasets']:
        frame=pd.read_parquet(root/(dataset+'.parquet'));target=manifest['datasets'][dataset]['target']
        y=frame.pop(target).to_numpy(dtype=float);numeric=frame.select_dtypes(include='number').columns.tolist()
        for seed in seeds:
            ids=np.random.default_rng(seed+74).permutation(len(frame));tr=ids[:train_size];va=ids[train_size:train_size+64];te=ids[train_size+64:]
            if len(te)<32:raise ValueError('Need at least 32 test rows')
            X,numeric_policy=normalize_numeric(frame,tr)
            graphs=[make_graph(row,vectors) for row in X.to_dict('records')]
            splits.append({'dataset':dataset,'seed':seed,'train':tr.tolist(),'validation':va.tolist(),'test':te.tolist(),'numeric_policy':numeric_policy})
            raw=np.stack([g[0][0].numpy() for g in graphs]);arms={}
            arms['fasttext_ridge']=ridge_predict(raw[tr],raw[va],raw[te],y[tr],y[va])
            for pre,name in [(False,'random'),(True,'pretrained')]:
                model=load_encoder(root/'kg_pretrained.pt',pre,seed)
                features=embed(model,graphs)
                arms[name+'_probe']=ridge_predict(features[tr],features[va],features[te],y[tr],y[va])
                arms['scratch' if not pre else 'pretrained_ft']=tune(model,graphs,y,tr,va,te,epochs,seed)
            cats=[c for c in frame if c not in numeric];treeX=frame.copy()
            for c in cats:treeX[c]=treeX[c].fillna('<missing>').astype(str)
            for c in numeric:treeX[c]=treeX[c].fillna(float(treeX.iloc[tr][c].median()))
            tree=CatBoostRegressor(iterations=150,depth=4,learning_rate=.05,loss_function='RMSE',random_seed=seed,thread_count=1,verbose=False,allow_writing_files=False)
            tree.fit(treeX.iloc[tr],y[tr],cat_features=cats)
            arms['catboost']=(tree.predict(treeX.iloc[te]),150)
            for arm,(pred,selected) in arms.items():
                records.append({'dataset':dataset,'seed':seed,'arm':arm,'r2':float(r2_score(y[te],pred)),'selected':float(selected),'prediction':np.asarray(pred).tolist(),'target':y[te].tolist()})
            print(dataset,seed,'done',flush=True)
    names=list(arms);summary=[];ranks=[]
    for dataset in manifest['datasets']:
        means=[]
        for arm in names:
            vals=[r['r2'] for r in records if r['dataset']==dataset and r['arm']==arm];means.append(np.mean(vals))
            summary.append({'dataset':dataset,'arm':arm,'mean':float(np.mean(vals)),'sd':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.})
        ranks.append(rankdata(-np.array(means)))
    stat,p=friedmanchisquare(*np.array(ranks).T)
    return {'config':{'seeds':list(seeds),'epochs':epochs,'train_size':train_size,'validation_size':64,'datasets':list(manifest['datasets']),'arms':names},'records':records,'splits':splits,'summary':summary,'ranks':{'mean':dict(zip(names,np.array(ranks).mean(0).tolist())),'friedman_p':float(p),'nemenyi_cd':float(studentized_range.ppf(.95,len(names),np.inf)/np.sqrt(2)*np.sqrt(len(names)*(len(names)+1)/(6*len(ranks))))},'benchmark_verdict':'INCOMPARABLE'}
