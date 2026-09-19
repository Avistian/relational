"""Declared low-budget readout intervention; not the published hyperparameter search."""
import json
from pathlib import Path
import numpy as np
from relkit.gin_l088 import load_mutag,train_fold,select_epoch
LAB=Path(__file__).resolve().parent
if __name__=='__main__':
 graphs=load_mutag(LAB/'data/l088');rows=[]
 out=LAB/'results/l088/teaching';out.mkdir(parents=True,exist_ok=True)
 for mode in ['sum','mean','max']:
  folds=[]
  for fold in range(10):
   r,z=train_fold(graphs,fold=fold,hidden=16,batch_size=32,dropout=0,epochs=30,iters=10,readout=mode)
   (out/f'{mode}-{fold}.json').write_text(json.dumps(r,indent=2)+'\n');folds.append(r)
  curves=[[e['val_acc'] for e in r['curves']] for r in folds];epoch=select_epoch(curves);values=np.array(curves)[:,epoch]
  rows.append({'readout':mode,'selected_epoch':epoch+1,'mean':float(values.mean()),'sample_sd':float(values.std(ddof=1)),'fold_values':values.tolist()});print(rows[-1],flush=True)
 (LAB/'_teaching_l088_results.json').write_text(json.dumps({'status':'EXECUTED','scope':'Teaching extension: 30 epochs x 10 updates, fixed h16/b32/d0; 10 folds; seed0; common epoch selected per readout; no independent test.', 'rows':rows},indent=2)+'\n')
