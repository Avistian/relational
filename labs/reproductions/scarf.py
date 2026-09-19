"""SCARF paper Tables 2 and 4: complete visible implementation, no packaged model.

Source: https://arxiv.org/html/2106.15147v2, Algorithm 1 and Section 4.
Unspecified choices (seed list, split stratification, initialization, scaling population)
are recorded as reconstruction choices, never claimed as author-code parity.
"""
import copy
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def paper_config():
    return dict(width=256,pre_epochs=1000,epochs=200,patience=3,batch=128,
                lr=.001,corruption=.6,tau=1.,validation_passes=10,seeds=list(range(30)))


class ScarfEncoder(nn.Sequential):
    """f: four affine/ReLU layers, including a ReLU on the representation."""
    def __init__(self,d):
        super().__init__(nn.Linear(d,256),nn.ReLU(),*[m for _ in range(3)
                         for m in (nn.Linear(256,256),nn.ReLU())])


class ScarfProjection(nn.Sequential):
    """g: two-layer ReLU head; the loss normalizes its output."""
    def __init__(self):
        super().__init__(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU())


def scarf_objective(a,b,tau=1.):
    scores=F.normalize(a,dim=1)@F.normalize(b,dim=1).T/tau
    # Paper denominator is the MEAN of N exponentials, hence the -log(N).
    return F.cross_entropy(scores,torch.arange(len(a),device=a.device))-math.log(len(a))


def corrupt_columns(x,bank,groups,rate,generator):
    """Uniformly choose floor(c*M) original columns, then independent marginal donors.

    A categorical block is replaced together; this equals corruption before one-hot
    encoding. Drawing separate donors for one-hot bits would create invalid categories.
    """
    mask=torch.zeros((len(x),len(groups)),dtype=torch.bool)
    count=int(rate*len(groups))
    chosen=torch.rand((len(x),len(groups)),generator=generator).argsort(1)[:,:count]
    mask.scatter_(1,chosen,True)
    donors=torch.randint(len(bank),(len(x),len(groups)),generator=generator)
    changed=x.clone()
    for j,group in enumerate(groups):
        rows=mask[:,j].nonzero().flatten().to(x.device)
        changed[rows,group]=bank[donors[mask[:,j],j].to(bank.device),group]
    return changed,mask


def load_openml(data_id,seed):
    """70/10/20 random split; published full-table missing-value imputation.

    Scaling is fit on training rows. The paper does not specify its fitting population.
    Use full-table categories to reproduce the explicitly transductive categorical
    vocabulary/imputation; expose that choice instead of calling this leak-free.
    """
    bunch=fetch_openml(data_id=int(data_id),as_frame=True,parser='auto')
    frame=bunch.data.copy();y=pd.factorize(bunch.target,sort=True)[0].astype('int64')
    ids=np.arange(len(y))
    development,test=train_test_split(ids,test_size=.2,random_state=seed)
    train,valid=train_test_split(development,test_size=.125,random_state=seed)
    blocks=[];groups=[];offset=0;columns=[]
    for name in frame:
        col=frame[name]
        if col.isna().all():continue
        numeric=pd.api.types.is_numeric_dtype(col)
        if numeric:
            values=pd.to_numeric(col).fillna(col.mean()).to_numpy(dtype='float64')
            if int(data_id) not in (4134,28,1468):
                sd=values[train].std();values=(values-values[train].mean())/(sd if sd else 1.)
            block=values[:,None]
        else:
            values=col.astype(object).fillna(col.mode().iloc[0])
            codes,levels=pd.factorize(values,sort=True)
            block=np.eye(len(levels),dtype='float32')[codes]
        groups.append(slice(offset,offset+block.shape[1]));offset+=block.shape[1]
        blocks.append(block);columns.append(str(name))
    X=np.concatenate(blocks,axis=1).astype('float32')
    assert np.isfinite(X).all() and (y>=0).all()
    provenance=dict(openml_id=int(data_id),version=int(bunch.details['version']),
                    rows=len(y),original_columns=columns,encoded_dim=X.shape[1],
                    data_sha256=hashlib.sha256(pd.util.hash_pandas_object(frame,index=True).values.tobytes()+y.tobytes()).hexdigest(),
                    train=train.tolist(),validation=valid.tolist(),test=test.tolist())
    return torch.from_numpy(X),torch.from_numpy(y),groups,train,valid,test,provenance


def pretrain_scarf(x,xv,groups,seed,cfg,device):
    torch.manual_seed(seed);encoder=ScarfEncoder(x.shape[1]).to(device)
    model=nn.Sequential(encoder,ScarfProjection().to(device))
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    generator=torch.Generator().manual_seed(seed+10000)
    x=x.to(device);xv=xv.to(device)
    # Freeze both candidates and batch membership for ten full validation passes.
    validation=[]
    for _ in range(cfg['validation_passes']):
        for idx in torch.randperm(len(xv),generator=generator).split(cfg['batch']):
            if len(idx)<2:continue
            clean=xv[idx.to(device)]
            changed,_=corrupt_columns(clean,xv,groups,cfg['corruption'],generator)
            validation.append((clean,changed))
    best=float('inf');state=None;wait=0;trace=[]
    for epoch in range(cfg['pre_epochs']):
        model.train();training=[]
        for idx in torch.randperm(len(x),generator=generator).split(cfg['batch']):
            if len(idx)<2:continue
            clean=x[idx.to(device)]
            changed,_=corrupt_columns(clean,x,groups,cfg['corruption'],generator)
            loss=scarf_objective(model(clean),model(changed),cfg['tau'])
            optimizer.zero_grad();loss.backward();optimizer.step();training.append(loss.item())
        model.eval()
        with torch.no_grad():
            value=sum(len(a)*scarf_objective(model(a),model(b),cfg['tau']).item() for a,b in validation)/sum(len(a) for a,b in validation)
        trace.append(dict(epoch=epoch+1,train=float(np.mean(training)),validation=value))
        if value<best:
            best=value;state=copy.deepcopy(encoder.state_dict());wait=0
        else:wait+=1
        if wait>=cfg['patience']:break
    encoder.load_state_dict(state)
    return encoder,trace


def finetune_scarf(encoder,x,y,xv,yv,classes,seed,cfg,device):
    """Both f and newly initialized h receive gradients; validate classification error."""
    torch.manual_seed(seed+20000)
    head=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,classes))
    model=nn.Sequential(copy.deepcopy(encoder).cpu(),head).to(device)
    before=copy.deepcopy(model[0].state_dict())
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr'])
    gen=torch.Generator().manual_seed(seed+30000)
    x,y,xv,yv=[v.to(device) for v in (x,y,xv,yv)]
    best=float('inf');state=None;wait=0;trace=[]
    for epoch in range(cfg['epochs']):
        model.train()
        for idx in torch.randperm(len(x),generator=gen).split(cfg['batch']):
            idx=idx.to(device);loss=F.cross_entropy(model(x[idx]),y[idx])
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():value=float((model(xv).argmax(1)!=yv).float().mean())
        trace.append(dict(epoch=epoch+1,validation_error=value))
        if value<best:
            best=value;state=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if wait>=cfg['patience']:break
    model.load_state_dict(state)
    delta=sum(float((v-before[k]).abs().sum()) for k,v in model[0].state_dict().items())
    return model,dict(trace=trace,encoder_update_l1=delta,validation_error=best)


def run_scarf(data_ids=(37,1464,40975),fractions=(1.,.25),smoke=False,device='cpu',output=None):
    """Default reproduces six named table rows; --all-data selects all 69.

    Smoke alone changes rows/epochs/seeds and is categorically INCOMPARABLE.
    Save after every arm so a long 30-split run retains completed measurements.
    """
    torch.set_num_threads(1);start=time.time();cfg=paper_config()
    if smoke:cfg.update(pre_epochs=2,epochs=3,seeds=[0],validation_passes=2)
    targets=json.loads((Path(__file__).parent/'scarf_targets.json').read_text())
    result=dict(paper='SCARF Tables 2 and 4',config=cfg,records=[],splits=[],smoke=smoke,
                verdict='INCOMPARABLE' if smoke else 'REPLICATION_MEASURED',
                remaining=['Author seeds and code were not released; independent reconstruction.',
                           'Unspecified split stratification, initialization and scaling population use documented choices.',
                           'Pretraining head interpreted as two affine/ReLU layers.',
                           'Selected dataset rows only unless all 69 IDs are requested.'])
    for data_id in data_ids:
        if str(data_id) not in targets['1.0']:raise ValueError('Dataset is not a published SCARF row')
        for seed in cfg['seeds']:
            X,y,groups,tr,va,te,split=load_openml(data_id,seed)
            if smoke:tr=tr[:256];va=va[:64];te=te[:128];split.update(train=tr.tolist(),validation=va.tolist(),test=te.tolist())
            split.update(seed=seed);result['splits'].append(split)
            encoder,pretrace=pretrain_scarf(X[tr],X[va],groups,seed,cfg,device)
            for fraction in fractions:
                if fraction not in (1.,.25):raise ValueError('Paper tables use 100% or 25%; extensions belong in the teaching lane')
                labeled=np.random.RandomState(seed+40000).permutation(tr)[:max(1,int(len(tr)*fraction))]
                for arm in ['control','scarf']:
                    torch.manual_seed(seed);initial=ScarfEncoder(X.shape[1]) if arm=='control' else encoder
                    model,details=finetune_scarf(initial,X[labeled],y[labeled],X[va],y[va],int(y.max())+1,seed,cfg,device)
                    with torch.no_grad():pred=torch.cat([model(batch.to(device)).argmax(1).cpu() for batch in X[te].split(1024)])
                    accuracy=float((pred==y[te]).float().mean())
                    target=targets[str(fraction)][str(data_id)][arm]/100
                    record=dict(data_id=data_id,seed=seed,fraction=fraction,arm=arm,accuracy=accuracy,
                                paper_mean=target,gap_to_paper=accuracy-target,train_labels=len(labeled),
                                validation_labels=len(va),labeled=labeled.tolist(),pretrain_trace=pretrace if arm=='scarf' else [],
                                prediction=pred.tolist(),target=y[te].tolist(),**details)
                    result['records'].append(record);result['seconds']=time.time()-start
                    if output:Path(output).write_text(json.dumps(result,indent=2))
                    print(f'SCARF id={data_id} seed={seed} labels={fraction} {arm} accuracy={accuracy:.4f}',flush=True)
    result['summary']=[]
    for data_id in data_ids:
        for fraction in fractions:
            for arm in ['control','scarf']:
                values=[r['accuracy'] for r in result['records'] if (r['data_id'],r['fraction'],r['arm'])==(data_id,fraction,arm)]
                target=targets[str(fraction)][str(data_id)][arm]/100
                mean=float(np.mean(values))
                result['summary'].append(dict(data_id=data_id,fraction=fraction,arm=arm,n=len(values),mean=mean,
                    sd=float(np.std(values,ddof=1)) if len(values)>1 else None,paper_mean=target,gap=mean-target,
                    comparison='INCOMPARABLE' if smoke else ('WITHIN_1PP' if abs(mean-target)<=.01 else 'OUTSIDE_1PP')))
    if output:Path(output).write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--all-data',action='store_true');parser.add_argument('--data-ids',default='37,1464,40975')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='scarf-paper-results.json')
    args=parser.parse_args()
    ids=list(map(int,json.loads((Path(__file__).parent/'scarf_targets.json').read_text())['1.0'])) if args.all_data else list(map(int,args.data_ids.split(',')))
    run_scarf(ids,smoke=args.smoke,device=args.device,output=args.output)
