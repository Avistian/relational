"""Use the original monitor to oracle the port's shifted checkpoint behavior."""
import importlib.util,json,tempfile
from pathlib import Path
import numpy as np,torch
from torch import nn
from _fetch_l103 import fetch
import relkit.tgat_l103 as port
P=Path(__file__).resolve().parent;source,_=fetch()
spec=importlib.util.spec_from_file_location('original_utils',source/'utils.py');utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils)
sequence=[.8,.81,.809,.80,.79];monitor=utils.EarlyStopMonitor()
for epoch,value in enumerate(sequence):
 if monitor.early_stop_check(value):break
assert epoch==4 and monitor.best_epoch==2 # the best score is at zero-based epoch 1
class Tiny(nn.Module):
 def __init__(self,*args,**kw):super().__init__();self.weight=nn.Parameter(torch.tensor(0.));self.ngh_finder=None
 def contrast(self,u,v,neg,t,count):return self.weight.sigmoid().expand(len(u)),(-self.weight).sigmoid().expand(len(u))
ev={'u':np.ones(8,dtype=int),'v':np.full(8,2,dtype=int),'t':np.arange(8,dtype=float),'e':np.arange(1,9)}
data={name:ev for name in ['train','val','new_val','test','new_test','full']};counter=0
original_model,original_eval=port.TGAT,port.evaluate

def controlled_evaluation(model,events,sampler,*args,**kw):
 global counter
 value=sequence[counter//2] if counter<10 else .75;counter+=1
 saved=[{'e':ev['e'][:-1],'negative':ev['v'][:-1],'p':np.full(7,.8),'n':np.full(7,.2),'batch':np.zeros(7,dtype=int)}]
 return {'ap':value,'auc':.8,'accuracy':.7},saved
try:
 port.TGAT=Tiny;port.evaluate=controlled_evaluation
 with tempfile.TemporaryDirectory(prefix='l103-selection-') as tmp:
  result=port.run_training(np.zeros((3,4),np.float32),np.zeros((9,4),np.float32),data,epochs=10,output=tmp)
  selected=torch.load(Path(tmp)/'selected.pt',weights_only=False)['weights'];expected=torch.load(Path(tmp)/'epoch-2.pt',weights_only=True)
  assert result['selected_epoch']==monitor.best_epoch and result['epochs_completed']==5 and result['early_stopped']
  torch.testing.assert_close(selected['weight'],expected['weight'],rtol=0,atol=0)
  assert selected['weight']!=torch.load(Path(tmp)/'epoch-1.pt',weights_only=True)['weight']
finally:port.TGAT,port.evaluate=original_model,original_eval
report={'status':'PASS','validation_sequence':sequence,'true_best_epoch':1,'released_selected_checkpoint':2,'port_selected_checkpoint':result['selected_epoch'],'selected_weights':'EXACT with source-monitor index','stop_after_epochs':5}
(P/'_selection_check_l103_results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
