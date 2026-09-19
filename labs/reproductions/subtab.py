"""Full MNIST SubTab release reconstruction, not the compact L072 teaching model.

Source pin: AstraZeneca/SubTab aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc.
Config: config/mnist.yaml and runtime.yaml. Comparison: supplement Table A3 mean.
The release does not identify which C produced Table A3: retain the entire sweep.
"""
import copy
import hashlib
import itertools
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression


def mnist_arrays():
    path=Path.home()/'.cache/relational-paper/mnist.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1':raise ValueError('MNIST digest differs')
    with np.load(path) as data:
        return [data[k].reshape(-1,784).astype('float32')/255 if k.startswith('x_') else data[k].astype('int64')
                for k in ['x_train','y_train','x_test','y_test']]


class SubTabMNIST(nn.Module):
    """343→784 LeakyReLU encoder, 784→784 decoder, two-layer normalized projector."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(343,784),nn.LeakyReLU())
        self.decoder=nn.Linear(784,784)
        self.project1=nn.Linear(784,784);self.project2=nn.Linear(784,784)
    def forward(self,x):
        h=self.encoder(x)
        z=F.normalize(self.project2(F.leaky_relu(self.project1(h))),dim=1)
        return z,h,self.decoder(h)


def subtab_views(x,rng,training=True):
    """Release overlap windows; corruption is disabled by both official evaluation entrypoints."""
    order=rng.permutation(4) if training else range(4);views=[]
    for i in order:
        start=0 if i==0 else i*196-147;stop=343 if i==0 else (i+1)*196
        view=x[:,start:stop]
        if training:
            donor=np.column_stack([view[rng.permutation(len(x)),j] for j in range(view.shape[1])])
            mask=rng.binomial(1,.3,view.shape)
            view=view*(1-mask)+donor*mask
        views.append(view.astype('float32'))
    return views


def subtab_joint(z,reconstruction,target):
    """Exactly the released reductions: sum features / rows for R and D; 2B InfoNCE."""
    n=len(z)//2
    scores=torch.tensordot(z.unsqueeze(1),z.T.unsqueeze(0),dims=2)
    labels=(torch.arange(2*n,device=z.device)+n)%(2*n)
    rows=torch.arange(2*n,device=z.device)
    positive=scores[rows,labels,None]
    negative_mask=~torch.eye(2*n,dtype=torch.bool,device=z.device)
    negative_mask[rows,labels]=False
    # Preserve the release's positive-first candidate order, including roundoff.
    candidates=torch.cat([positive,scores[negative_mask].reshape(2*n,-1)],dim=1)
    contrast=F.cross_entropy(candidates/.1,torch.zeros(2*n,dtype=torch.long,device=z.device),reduction="sum")/(2*n)
    recon=(reconstruction-target).square().sum()/len(target)
    distance=(z[:n]-z[n:]).square().sum()/n
    return recon+contrast+distance


def train_subtab(X,seed=57,epochs=15,device='cpu'):
    torch.manual_seed(seed);rng=np.random.RandomState(seed)
    # Released loader randomly orders all 60k training rows before making shuffled batches.
    X=X[rng.permutation(len(X))]
    model=SubTabMNIST().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-7,weight_decay=.01)
    loader=torch.utils.data.DataLoader(X,batch_size=32,shuffle=True,drop_last=True)
    trace=[]
    for epoch in range(epochs):
        total=0.;batches=0;model.train()
        for x in loader:
            views=subtab_views(x.numpy(),rng)
            target=torch.from_numpy(np.concatenate([x.numpy(),x.numpy()])).to(device)
            losses=[]
            for a,b in itertools.combinations(views,2):
                z,h,reconstruction=model(torch.from_numpy(np.concatenate([a,b])).to(device))
                losses.append(subtab_joint(z,reconstruction,target))
            loss=torch.stack(losses).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=loss.item();batches+=1
        trace.append(total/batches);print(f'SubTab epoch {epoch+1}/{epochs}: {trace[-1]:.5f}',flush=True)
    return model,trace


def subtab_represent(model,X,rng,device,shuffle=False):
    """Mean of four CLEAN latent vectors, as train.py/eval.py set add_noise=False."""
    order=rng.permutation(len(X)) if shuffle else np.arange(len(X));values=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(order),32):
            views=subtab_views(X[order[start:start+32]],rng,training=False)
            h=[model(torch.from_numpy(v).to(device))[1] for v in views]
            values.append(torch.stack(h).mean(0).cpu().numpy())
    return np.concatenate(values),order


def run_subtab(smoke=False,device='cpu',output=None,threads=1):
    torch.set_num_threads(threads);start=time.time();X,y,Xt,yt=mnist_arrays()
    if smoke:X,y,Xt,yt=X[:256],y[:256],Xt[:128],yt[:128]
    model,trace=train_subtab(X,epochs=1 if smoke else 15,device=device)
    if output:
        torch.save({'state_dict':model.cpu().state_dict(),'seed':57,'epochs':1 if smoke else 15},Path(output).with_suffix('.pt'))
        model.to(device)
    # The release's eval.py resets seed to 57 in a new process. Recreate the evaluation stream.
    rng=np.random.RandomState(57)
    train,order=subtab_represent(model,X,rng,device,shuffle=True)
    test,_=subtab_represent(model,Xt,rng,device)
    records=[]
    for c in ([1.] if smoke else [.01,.1,1.,10.,100.,1e3,1e4,1e5,1e6]):
        clf=LogisticRegression(C=c,max_iter=1200,solver='lbfgs').fit(train,y[order])
        prediction=clf.predict(test);score=float(np.mean(prediction==yt))
        records.append(dict(C=c,accuracy=score,prediction=prediction.tolist(),target=yt.tolist(),
                            gap_to_table_A3=score-.9786,converged=bool(clf.n_iter_.max()<1200)))
        print(f'SubTab C={c}: {score:.5f}',flush=True)
    out=dict(target='SubTab supplement Table A3 mean aggregation: 97.86%',paper_accuracy=.9786,
             release='aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc',smoke=smoke,
             config=dict(seed=57,epochs=1 if smoke else 15,batch=32,subsets=4,overlap=.75,
                         noise=.3,evaluation_noise=False,tau=.1,lr=.001,adamw_eps=1e-7,weight_decay=.01,threads=threads),
             train_rows=len(X),test_rows=len(Xt),loss_trace=trace,records=records,seconds=time.time()-start,
             verdict='INCOMPARABLE' if smoke else 'RELEASE_REPLICATION_MEASURED',
             remaining=['The paper does not tie Table A3 to a released checkpoint or selected C.',
                        'Released clean evaluation reports every C on test; no post-test selection is declared a held-out score.',
                        'Modern torch/sklearn numerical kernels and random-stream ordering can differ from 2021.',
                        'MNIST release only; other paper datasets and full ablations are not reproduced.'])
    if output:Path(output).write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--device',default='cpu');p.add_argument('--output',default='subtab-paper-results.json');a=p.parse_args()
    run_subtab(a.smoke,a.device,a.output)
