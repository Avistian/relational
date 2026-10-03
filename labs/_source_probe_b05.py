"""Execute the unchanged released sampler method with an exact L2 index oracle."""
import ast,json,random,types
from pathlib import Path
import numpy as np
import torch
from relkit.retrieval_b05 import episode
P=Path(__file__).resolve().parent;E=P/'evidence/b05';S=P/'sources/b05'
def probe_sampler(source):
    tree=ast.parse(source)
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='use_knn')
    module=ast.Module(body=[method],type_ignores=[])
    class FixedRandom:
        def __init__(self):self.calls=0
        def randint(self,a,b):
            self.calls+=1
            return 0 if self.calls==1 else (1 if self.calls==2 else b)
    class ExactIndex:
        def __init__(self,x):self.x=x;self.last=None
        def get_knn_indices(self,q,k):
            self.last=np.argsort(((self.x-q)**2).sum(axis=1),kind='stable')[:k]
            return self.last[None,:]
    tables=[np.array([[0,0],[.1,100],[.2,0],[.3,100],[1,0],[2,100]],float),np.array([[0,0],[.1,0],[.2,100],[.3,0],[1,100],[2,100]],float)]
    observed=[]
    for table in tables:
        x=(table-table.mean(axis=0))/table.std(axis=0)
        index=ExactIndex(x);namespace=dict(np=np,torch=torch,random=FixedRandom())
        exec(compile(module,'pinned-use_knn','exec'),namespace)
        obj=types.SimpleNamespace(context_length=3,max_feat=100,transform_target=lambda y,**kw:(y,'reg'))
        # Seed irrelevant column choice too; retain the original global state.
        state=np.random.get_state();np.random.seed(0)
        try:namespace['use_knn'](obj,x,index,len(x),x.shape[1])
        finally:np.random.set_state(state)
        observed.append(index.last.tolist())
    safe=[episode(t,1,0,3,2,0)['selected_ids'] for t in tables]
    assert observed[0]!=observed[1] and safe[0]==safe[1]
    return dict(status='CONFIRMED_RELEASED_METHOD_DEPENDENCE',released_neighbors=observed,paper_order_neighbors=safe,scope='Unchanged use_knn AST; exact L2 index oracle replaces FAISS; normalized six-row counterexample. No model training or benchmark effect measured.',historical_training_identity='NOT_ESTABLISHED')
if __name__=='__main__':
    r=probe_sampler((S/'training/dataset.py').read_text());(E/'sampler-probe.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
