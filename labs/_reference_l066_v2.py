"""Run immutable 0.1.4 reference only; no teaching-model imports."""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'sources/l066-v2'))
from tabicl import TabICL,TabICLClassifier,InferenceConfig

def run(checkpoint,destination):
 torch.set_num_threads(1);torch.manual_seed(660)
 ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
 model=TabICL(**ck['config']).eval();model.load_state_dict(ck['state_dict'])
 cfg=InferenceConfig();cfg.update_from_dict({k:dict(device='cpu',use_amp=False,offload=False) for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
 fixtures=[]
 with torch.no_grad():
  for n,c,f,k in [(13,9,5,3),(9,5,2,2),(18,13,9,10)]:
   x=torch.randn(1,n,f);y=(torch.arange(c)%k).float()[None]
   e=model.col_embedder(x,train_size=c,mgr_config=cfg.COL_CONFIG)
   r=model.row_interactor(e.clone(),mgr_config=cfg.ROW_CONFIG)
   out=model(x,y,inference_config=cfg)
   fixtures.append(dict(x=x,y=y,column=e[:,:,4:],row=r,logits=out))
 rng=np.random.default_rng(661);x=rng.normal(size=(38,5)).astype('float32');x[:,4]=7.;x[0,0]=35.;y=np.arange(27)%3
 clf=TabICLClassifier(n_estimators=1,norm_methods='none',model_path=checkpoint,allow_auto_download=False,device='cpu',n_jobs=1,use_amp=False,inference_config={k:dict(offload=False) for k in ['COL_CONFIG','ROW_CONFIG','ICL_CONFIG']})
 clf.fit(x[:27],y);prob=clf.predict_proba(x[27:]);data=clf.ensemble_generator_.transform(x[27:])['none'][0]
 torch.save(dict(fixtures=fixtures,wrapper=dict(x=x,y=y,preprocessed=data,probabilities=prob),checkpoint_config=ck['config']),destination)
if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
