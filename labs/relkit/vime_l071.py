"""L071: visible VIME numeric-data implementation and matched experiment.

Source semantics: jsyoon0823/VIME @ 996c58cf4c570061b30c38ecf2a754a9af85aafd.
PyTorch port, not a numerical port of historical Keras/TensorFlow optimizers.
"""
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split


def corrupt(x, mask, donors):
    """Eq. 3 + released pretext_generator: donors[i,j] indexes a row for column j.

    Return actual-change targets and corrupted values, each [B,d]. Inputs stay intact.
    """
    cols = torch.arange(x.shape[1], device=x.device)
    replacement = x[donors, cols]
    xt = (1 - mask) * x + mask * replacement
    return (xt != x).to(x.dtype), xt


def draw_corruption(x, p, generator):
    """One independent permutation per column, as in the released source."""
    mask = (torch.rand(x.shape, generator=generator) < p).to(x.dtype)
    donors = torch.stack([torch.randperm(len(x), generator=generator)
                          for _ in range(x.shape[1])], dim=1)
    return corrupt(x, mask, donors)


def pretext_loss(mask_logits, reconstruction, x, changed, alpha=2., mask_weight=1.):
    """Eqs. 4–6: mean BCE plus alpha times MSE over ALL coordinates."""
    lm = F.binary_cross_entropy_with_logits(mask_logits, changed)
    lr = ((reconstruction - x) ** 2).mean()
    return mask_weight * lm + alpha * lr, lm, lr


def consistency_loss(logits):
    """Released vime_semi.py: population variance of logits across K views.

    logits: [K,B,C]. Mean over batch and classes AFTER variance over views.
    """
    return ((logits - logits.mean(dim=0, keepdim=True)) ** 2).mean()


class VIME(nn.Module):
    """Released numeric architecture: d→d ReLU encoder; two d-dimensional heads."""
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.mask_head = nn.Linear(d, d)
        self.feature_head = nn.Sequential(nn.Linear(d, d), nn.Sigmoid())

    def forward(self, x):
        z = self.encoder(x)
        return self.mask_head(z), self.feature_head(z)


class Predictor(nn.Module):
    """Encoder → 100-ReLU → 100-ReLU → C logits. All arms share this shape."""
    def __init__(self, encoder, d, classes):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        self.head = nn.Sequential(nn.Linear(d, 100), nn.ReLU(),
                                  nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, classes))

    def forward(self, x):
        return self.head(self.encoder(x))


def pretrain(x, seed, epochs=30, p=.3, alpha=2., mask_weight=1., device='cpu'):
    """Fixed corruption bank as in vime_self.py; no labels or test rows accepted."""
    torch.manual_seed(seed)
    model = VIME(x.shape[1]).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=.001, alpha=.9, eps=1e-7)
    g = torch.Generator().manual_seed(seed + 7100)
    changed, xt = draw_corruption(x.cpu(), p, g)
    x, changed, xt = x.to(device), changed.to(device), xt.to(device)
    history = []
    for epoch in range(epochs):
        losses=[]
        for idx in torch.randperm(len(x), generator=g).split(128):
            idx=idx.to(device)
            ml, xr = model(xt[idx])
            loss, lm, lr = pretext_loss(ml, xr, x[idx], changed[idx], alpha, mask_weight)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append([loss.item(), lm.item(), lr.item()])
        history.append(np.mean(losses, axis=0).tolist())
    return model.cpu(), {'loss_trace':history, 'actual_change_rate':changed.mean().item(),
                         'sampled_probability':p, 'mask_weight':mask_weight}


def make_split(y, budgets, seed, external_test=False):
    """No feature-dependent fitting. Nested class-balanced label budgets include validation.

    A fixed 60% of development rows is unlabeled. Remaining rows form the label reserve.
    Original class labels are used ONLY by the benchmark designer to balance selection.
    """
    ids=np.arange(len(y))
    if external_test:
        dev, test = ids, np.array([],dtype=int)
    else:
        dev, test = train_test_split(ids,test_size=.25,stratify=y,random_state=seed)
    reserve, unlab=train_test_split(dev,test_size=.6,stratify=y[dev],random_state=seed+100)
    rng=np.random.default_rng(seed+200)
    buckets=[rng.permutation(reserve[y[reserve]==c]).tolist() for c in np.unique(y)]
    order=[]
    while any(buckets):
        for bucket in buckets:
            if bucket: order.append(bucket.pop())
    result={}
    for b in budgets:
        if b>len(order): raise ValueError('Label budget exceeds reserve')
        selected=np.array(order[:b])
        tr,va=train_test_split(selected,test_size=.2,stratify=y[selected],random_state=seed+300)
        result[b]=(tr,va)
    return {'unlabeled':unlab,'test':test,'budgets':result}


def fit_predictor(encoder, x, y, xv, yv, xu, seed, epochs=60, frozen=False, beta=0., K=3, device='cpu'):
    """Select by validation cross-entropy; test arrays cannot enter this function.

    The same supervised batches and head initialization are used across paired arms.
    """
    torch.manual_seed(seed+1000)
    model=Predictor(encoder,x.shape[1],int(max(y.max(),yv.max()).item())+1).to(device)
    for parameter in model.encoder.parameters(): parameter.requires_grad_(not frozen)
    before={k:v.detach().clone() for k,v in model.encoder.state_dict().items()}
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    g=torch.Generator().manual_seed(seed+2000)
    ug=torch.Generator().manual_seed(seed+3000)
    x,y,xv,yv=x.to(device),y.to(device),xv.to(device),yv.to(device)
    best_loss=float('inf'); best=None; chosen=0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x),generator=g).split(64):
            idx=idx.to(device)
            loss=F.cross_entropy(model(x[idx]),y[idx])
            if beta:
                ux=xu[torch.randperm(len(xu),generator=ug)[:64]]
                views=torch.stack([draw_corruption(ux,.3,ug)[1] for _ in range(K)])
                logits=model(views.to(device))
                loss=loss+beta*consistency_loss(logits)
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad(): value=F.cross_entropy(model(xv),yv).item()
        if value<best_loss:
            best_loss=value;chosen=epoch+1;best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    delta=sum((v-before[k]).abs().sum().item() for k,v in model.encoder.state_dict().items())
    return model.cpu(),{'best_epoch':chosen,'validation_loss':best_loss,'encoder_delta':delta}


def load_data(dataset='digits'):
    if dataset=='digits':
        d=load_digits()
        # 16 is the documented measurement range, not an estimated dataset maximum.
        return d.data.astype('float32')/16.,d.target.astype('int64'),None,None
    if dataset=='mnist':
        import urllib.request
        cache=Path.home()/'.cache'/'l071';cache.mkdir(parents=True,exist_ok=True)
        path=cache/'mnist.npz'
        if not path.exists():
            urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',path)
        # Archive SHA-256 published by keras/src/datasets/mnist.py.
        if hashlib.sha256(path.read_bytes()).hexdigest()!='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':
            raise ValueError('MNIST archive checksum mismatch')
        with np.load(path) as d:
            return (d['x_train'].reshape(-1,784).astype('float32')/255.,d['y_train'].astype('int64'),
                    d['x_test'].reshape(-1,784).astype('float32')/255.,d['y_test'].astype('int64'))
    raise ValueError(dataset)


def summarize(records):
    rows=[]
    for budget in sorted({r['budget'] for r in records}):
        scratch={r['seed']:r['accuracy'] for r in records if r['budget']==budget and r['arm']=='scratch'}
        for arm in sorted({r['arm'] for r in records}):
            rr=[r for r in records if r['budget']==budget and r['arm']==arm]
            scores=[r['accuracy'] for r in rr]
            gaps=[r['accuracy']-scratch[r['seed']] for r in rr]
            rows.append(dict(budget=budget,arm=arm,accuracy=float(np.mean(scores)),
                             sd=float(np.std(scores,ddof=1)) if len(scores)>1 else None,
                             paired_gain=float(np.mean(gaps)),seed_gains=gaps))
    return rows


def run_experiment(seeds=(0,1,2),budgets=(50,150,500),pre_epochs=30,epochs=60,dataset='digits',device='cpu'):
    torch.set_num_threads(1)
    start=time.time()
    X,Y,Xtest,Ytest=load_data(dataset)
    x=torch.from_numpy(X);y=torch.from_numpy(Y)
    records=[];splits=[];pretexts=[]
    for seed in seeds:
        split=make_split(Y,budgets,seed,external_test=Xtest is not None)
        u=split['unlabeled'];xu=x[u]
        vime,trace=pretrain(xu,seed,pre_epochs,device=device)
        recon,rtrace=pretrain(xu,seed,pre_epochs,mask_weight=0.,device=device)
        torch.manual_seed(seed);random=VIME(x.shape[1])
        pretexts.append(dict(seed=seed,vime=trace,reconstruction=rtrace))
        split_record={'seed':seed,'unlabeled':u.tolist(),'test':split['test'].tolist(),'budgets':{}}
        for budget in budgets:
            tr,va=split['budgets'][budget]
            split_record['budgets'][str(budget)]={'train':tr.tolist(),'validation':va.tolist()}
            for arm,enc,frozen,beta in [('scratch',random.encoder,False,0.),
                                       ('vime_finetune',vime.encoder,False,0.),
                                       ('vime_frozen',vime.encoder,True,0.),
                                       ('recon_finetune',recon.encoder,False,0.),
                                       ('vime_semi',vime.encoder,True,1.)]:
                model,info=fit_predictor(enc,x[tr],y[tr],x[va],y[va],xu,seed,
                                         epochs,frozen,beta,device=device)
                tx=x[split['test']] if Xtest is None else torch.from_numpy(Xtest)
                ty=Y[split['test']] if Ytest is None else Ytest
                with torch.no_grad(): probs=model(tx).softmax(-1).numpy()
                records.append(dict(seed=seed,budget=budget,train_labels=len(tr),validation_labels=len(va),
                                    unlabeled_rows=len(u),arm=arm,accuracy=float(accuracy_score(ty,probs.argmax(1))),
                                    log_loss=float(log_loss(ty,probs)),**info))
        splits.append(split_record)
    return dict(dataset=dataset,config=dict(seeds=list(seeds),budgets=list(budgets),pre_epochs=pre_epochs,epochs=epochs,
                p=.3,alpha=2.,beta=1.,K=3,device=device),records=records,summary=summarize(records),
                pretexts=pretexts,splits=splits,data_sha256=hashlib.sha256(X.tobytes()+Y.tobytes()).hexdigest(),
                environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__),
                seconds=time.time()-start,verdict='INCOMPARABLE',
                scope='Local label-efficiency experiment; no paper-table reproduction. Seeds are not datasets.')
