"""Historical 2.0.9 release embeddings and every intermediate target state."""
import sys
from pathlib import Path
import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.model.loading import load_model
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _reference_l064_v2 import simple_config

def run(checkpoint,destination):
 torch.set_num_threads(1);rng=np.random.default_rng(650);x=rng.normal(size=(31,5)).astype('float32');x[2,1]=np.nan;y=np.arange(22)%2
 clf=TabPFNClassifier(n_estimators=1,model_path=checkpoint,device='cpu',n_jobs=1,inference_config=simple_config(),inference_precision=torch.float32,random_state=0);clf.fit(x[:22],y)
 states=[]
 def capture(module,args,output):states.append(output.detach().clone())
 handles=[b.register_forward_hook(capture) for b in clf.model_.transformer_encoder.layers]
 query=clf.get_embeddings(x[22:],data_source='test');query_states=states.copy();states.clear()
 context=clf.get_embeddings(x[22:],data_source='train')
 for h in handles:h.remove()
 torch.save(dict(x=x,y=y,query=query,context=context,states=query_states),destination)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
