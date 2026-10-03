"""Visible numeric-only single-member course adapter, with original NN compute path."""
import hashlib,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from relkit.hyper_b07a import class_weights,retrieval_bias
from relkit.hyperfast_b07a import HyperFast,forward_main_network,seed_everything,transform_data_for_main_network

CONFIG=dict(hn_n_layers=4,hn_hidden_size=1024,clip_data_value=27.6041,rf_size=32768,n_dims=784,main_n_layers=3,max_categories=46,lr=.0001,torch_pca=True,device='cpu',nn_bias=False)

def load_hypernetwork(path,head_fn=class_weights):
    """Meta allocation + mmap avoids holding two5GB copies; exact float32 weights."""
    cfg=SimpleNamespace(**CONFIG)
    with torch.device('meta'):model=HyperFast(cfg,head_fn=head_fn)
    state=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
    model.load_state_dict(state,strict=True,assign=True)
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    return model,cfg

class GeneratedPredictor:
    def __init__(self,model,cfg,bias_fn=retrieval_bias):
        self.model=model;self.cfg=cfg;self.bias_fn=bias_fn
    def fit(self,X,y,seed,batch_size=512):
        seed_everything(seed)
        self.scaler=StandardScaler().fit(X)
        x=torch.tensor(self.scaler.transform(X),dtype=torch.float32)
        labels=torch.tensor(y,dtype=torch.long)
        self.classes=np.unique(y);assert np.array_equal(self.classes,np.arange(len(self.classes)))
        indices=torch.randperm(len(x))[:batch_size]
        self.support_indices=indices.numpy().copy()
        self.support=x[indices];self.labels=labels[indices]
        if len(self.support)<self.cfg.n_dims:
            n=int(np.ceil(self.cfg.n_dims/len(self.support)))
            self.support=self.support.repeat_interleave(n,dim=0);self.labels=self.labels.repeat_interleave(n)
        with torch.no_grad():self.rf,self.pca,self.layers,self.bias=self.model(self.support,self.labels,len(self.classes))
        return self
    def predict(self,X,retrieval=False,return_trace=False):
        x=torch.tensor(self.scaler.transform(X),dtype=torch.float32)
        with torch.no_grad():
            transformed=transform_data_for_main_network(x,self.cfg,self.rf,self.pca)
            raw,hidden=forward_main_network(transformed,self.layers)
            logits=raw.clone()
            if retrieval:
                # Match release: recompute support activations on every call.
                sup=transform_data_for_main_network(self.support,self.cfg,self.rf,self.pca)
                _,support_hidden=forward_main_network(sup,self.layers)
                logits=self.bias_fn(logits,x,self.support,self.labels,self.bias[0])
                logits=self.bias_fn(logits,hidden,support_hidden,self.labels,self.bias[1])
            probs=torch.softmax(logits,dim=1).numpy()
        if return_trace:return probs,raw.numpy(),logits.numpy()
        return probs
    def identity(self):
        h=hashlib.sha256()
        arrays=[self.rf[0].weight,self.pca.mean_,self.pca.components_,self.bias]+[a for layer in self.layers for a in layer]
        for a in arrays:h.update(a.detach().numpy().tobytes())
        return h.hexdigest()
    def retained_bytes(self):
        arrays=[self.rf[0].weight,self.pca.mean_,self.pca.components_,self.bias]+[a for layer in self.layers for a in layer]
        core=sum(a.numel()*a.element_size() for a in arrays)+self.scaler.mean_.nbytes+self.scaler.scale_.nbytes
        support=self.support.numel()*self.support.element_size()+self.labels.numel()*self.labels.element_size()
        return dict(weights_only_minimum=core,retrieval_minimum=core+support,adapter_actual_both=core+support,scope='Array storage only; excludes shared hypernetwork, allocator/workspace, Python overhead. Course adapter keeps support in both arms; weights-only can discard it.')
