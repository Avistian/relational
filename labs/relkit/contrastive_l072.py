"""L072 numeric teaching implementations. See l072-reproduction.md for deviations."""
import copy
import hashlib
import itertools
import platform
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.stats import rankdata, friedmanchisquare, studentized_range


def corrupt(x, bank, mask, donors):
    """Independent donor row per cell; each value stays in its original column."""
    columns = torch.arange(x.shape[1], device=x.device)[None, :]
    return torch.where(mask, bank[donors, columns], x)


def scarf_loss(clean, changed, tau=1.):
    """N clean anchors × N corrupt candidates; CE differs from paper by +log(N)."""
    if tau <= 0 or len(clean) < 2:
        raise ValueError('Positive temperature and at least two rows required')
    scores = F.normalize(clean, dim=1) @ F.normalize(changed, dim=1).T / tau
    return F.cross_entropy(scores, torch.arange(len(clean), device=clean.device))


def subtab_loss(a, b, tau=1.):
    """Symmetric 2N NT-Xent: exclude self, retain cross-view positive."""
    n = len(a)
    z = F.normalize(torch.cat([a, b]), dim=1)
    scores = z @ z.T / tau
    scores = scores.masked_fill(torch.eye(2*n, dtype=torch.bool, device=z.device), -torch.inf)
    targets = (torch.arange(2*n, device=z.device) + n) % (2*n)
    return F.cross_entropy(scores, targets)


def aggregate_views(h):
    """[views, rows, hidden] -> [rows, hidden]. Never aggregate distinct rows."""
    return h.mean(dim=0)


def subsets(d, count=3, overlap=.75):
    """Released SubTab contiguous layout, including its floor-division tail behavior."""
    width = d // count
    extra = int(overlap * width)
    return [list(range(0 if i == 0 else i*width-extra,
                       width+extra if i == 0 else (i+1)*width)) for i in range(count)]


def draw_view(x, bank, rate=.6):
    q = int(rate * x.shape[1])
    mask = torch.zeros_like(x, dtype=torch.bool)
    chosen = torch.rand_like(x).argsort(dim=1)[:, :q]
    mask.scatter_(1, chosen, True)
    donors = torch.randint(len(bank), x.shape, device=x.device)
    return corrupt(x, bank, mask, donors)


class SCARF(nn.Module):
    """Local compact f: d->64->64; g: 64->64->32; ReLU after hidden layers."""
    def __init__(self, d, width=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU())
        self.projector = nn.Sequential(nn.Linear(width,width),nn.ReLU(),nn.Linear(width,32))
    def forward(self, x):
        return self.projector(self.encoder(x))
    def represent(self, x):
        return self.encoder(x)


class SubTab(nn.Module):
    """Shared subset encoder, full-row decoder and projector; local compact widths."""
    def __init__(self, d, width=64):
        super().__init__()
        self.columns = subsets(d)
        self.encoder = nn.Sequential(nn.Linear(len(self.columns[0]),width),nn.LeakyReLU(),nn.Linear(width,width))
        self.projector = nn.Sequential(nn.Linear(width,width),nn.LeakyReLU(),nn.Linear(width,32))
        self.decoder = nn.Sequential(nn.Linear(width,width),nn.LeakyReLU(),nn.Linear(width,d))
    def forward(self, x, noise=False):
        hs=[]
        for cols in self.columns:
            view=x[:,cols]
            if noise:
                view=view + .1*torch.randn_like(view)*(torch.rand_like(view)<.3)
            hs.append(self.encoder(view))
        h=torch.stack(hs)
        z=F.normalize(self.projector(h),dim=-1)
        return h,z,self.decoder(h)
    def represent(self, x):
        return aggregate_views(self(x,noise=False)[0])


def pretrain(x, kind, seed, epochs=40, device='cpu'):
    """No labels or held-out rows enter this function. Fixed epochs; no selection."""
    torch.manual_seed(seed)
    net=(SubTab(x.shape[1]) if kind.startswith('subtab') else SCARF(x.shape[1])).to(device)
    x=torch.as_tensor(x,dtype=torch.float32,device=device)
    initial=copy.deepcopy(net.state_dict())
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    history=[]
    if kind=='random':
        return net, {'loss': [], 'parameter_delta': 0.}
    for epoch in range(epochs):
        values=[]
        for ix in torch.randperm(len(x),device=device).split(64):
            if len(ix)<2: continue
            batch=x[ix]
            if kind.startswith('subtab'):
                _,z,reconstruction=net(batch,noise=True)
                # Released getMSEloss sums features and averages rows, not all cells.
                recon=(reconstruction-batch[None]).square().sum(-1).mean()
                loss=recon
                if kind=='subtab_joint':
                    pair_losses=[subtab_loss(z[a],z[b])+(z[a]-z[b]).square().sum(-1).mean()
                                 for a,b in itertools.combinations(range(3),2)]
                    loss=loss+torch.stack(pair_losses).mean()
            else:
                changed=draw_view(batch,x,rate=0. if kind=='scarf_c0' else .6)
                loss=scarf_loss(net(batch),net(changed))
            opt.zero_grad();loss.backward();opt.step()
            values.append(float(loss.detach()))
        history.append(float(np.mean(values)))
    delta=sum(float((net.state_dict()[k]-v).abs().sum()) for k,v in initial.items())
    return net, {'loss': history, 'parameter_delta': delta}


def load_split(name, seed):
    loaders={'wine':load_wine,'breast_cancer':load_breast_cancer,'digits':load_digits}
    data=loaders[name]()
    x=data.data.astype('float32');y=data.target
    indices=np.arange(len(x))
    train,held=train_test_split(indices,test_size=.3,stratify=y,random_state=seed)
    validation,test=train_test_split(held,test_size=2/3,stratify=y[held],random_state=seed)
    labeled,_=train_test_split(train,train_size=.25,stratify=y[train],random_state=seed)
    scaler=StandardScaler().fit(x[train])
    return scaler.transform(x).astype('float32'),y,{
        'train':train.tolist(),'labeled':labeled.tolist(),'validation':validation.tolist(),'test':test.tolist(),
        'feature_mean':scaler.mean_.tolist(),
        'data_sha256':hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()}


def probe(features, y, split):
    """Fit only labeled rows. Select regularization on validation; score test once."""
    tr=split['labeled'];va=split['validation'];te=split['test']
    scale=StandardScaler().fit(features[tr])
    f=scale.transform(features)
    best=None;trials=[]
    for c in [.1,1.,10.]:
        model=LogisticRegression(C=c,max_iter=2000).fit(f[tr],y[tr])
        score=accuracy_score(y[va],model.predict(f[va]))
        trials.append({'C':c,'validation_accuracy':float(score)})
        if best is None or score>best[0]:best=(score,c,model)
    pred=best[2].predict(f[te])
    return {'accuracy':float(accuracy_score(y[te],pred)), 'C':best[1], 'trials':trials,
            'prediction':pred.tolist(),'target':y[te].tolist()}


def summarize(records):
    rows=[]
    for dataset in sorted({r['dataset'] for r in records}):
        for arm in sorted({r['arm'] for r in records}):
            rr=[r for r in records if r['dataset']==dataset and r['arm']==arm]
            values=[r['accuracy'] for r in rr]
            base={r['seed']:r['accuracy'] for r in records if r['dataset']==dataset and r['arm']=='random'}
            rows.append({'dataset':dataset,'arm':arm,'mean':float(np.mean(values)),
                         'sd':float(np.std(values,ddof=1)) if len(values)>1 else 0.,
                         'seed_values':values,'gain_vs_random':[r['accuracy']-base[r['seed']] for r in rr]})
    arms=sorted({r['arm'] for r in records});datasets=sorted({r['dataset'] for r in records})
    scores=np.array([[next(r['mean'] for r in rows if r['dataset']==d and r['arm']==a) for a in arms] for d in datasets])
    ranks=np.array([rankdata(-s) for s in scores]);k=len(arms);n=len(datasets)
    p=float(friedmanchisquare(*scores.T).pvalue) if n>=3 else None
    cd=float(studentized_range.ppf(.95,k,np.inf)/np.sqrt(2)*np.sqrt(k*(k+1)/(6*n)))
    return rows, {'arms':arms,'datasets':datasets,'ranks':ranks.tolist(),'mean_ranks':ranks.mean(0).tolist(),'friedman_p':p,'nemenyi_cd':cd,
                  'interpretation':'Exploratory only: three datasets; seeds are averaged within dataset, not independent datasets.'}


def run_experiment(seeds=(0,1,2), datasets=('wine','breast_cancer','digits'), epochs=40, device='cpu'):
    started=time.time();torch.set_num_threads(1)
    records=[];splits=[]
    for name in datasets:
        for seed in seeds:
            x,y,s=load_split(name,seed);splits.append({'dataset':name,'seed':seed,**s})
            for arm in ['raw','random','scarf','scarf_c0','subtab_recon','subtab_joint']:
                if arm=='raw':f=x;trace={'loss':[],'parameter_delta':0.};frozen_delta=0.
                else:
                    net,trace=pretrain(x[s['train']],arm,seed,epochs,device)
                    net.eval()
                    for p in net.parameters():p.requires_grad_(False)
                    before=copy.deepcopy(net.state_dict())
                    with torch.no_grad():f=net.represent(torch.tensor(x,device=device)).cpu().numpy()
                result=probe(f,y,s)
                if arm!='raw':frozen_delta=sum(float((net.state_dict()[k]-v).abs().sum()) for k,v in before.items())
                records.append({'dataset':name,'seed':seed,'arm':arm,**result,**trace,'frozen_delta':frozen_delta,
                                'train_labels':len(s['labeled']),'validation_labels':len(s['validation'])})
    summary,ranks=summarize(records)
    return {'config':{'seeds':list(seeds),'datasets':list(datasets),'epochs':epochs,'batch':64,'rate':.6,'tau':1.,'device':device},
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn_version},
            'elapsed_seconds':time.time()-started,'records':records,'splits':splits,'summary':summary,'rank_audit':ranks,
            'verdict':'INCOMPARABLE to paper benchmark tables; numeric local mechanism experiment'}
