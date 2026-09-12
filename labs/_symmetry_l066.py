"""Exact-distribution symmetry fixture through the original TabICL source."""
import hashlib,json,sys,types
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
package=types.ModuleType('tabicl');package.__path__=[str(ROOT/'sources/l066-v2/tabicl')];sys.modules['tabicl']=package
from tabicl.model.tabicl import TabICL

def check():
    torch.set_num_threads(1)
    path=ROOT/'data/cache/foundation/tabicl-v1-0208.ckpt';c=torch.load(path,map_location='cpu',weights_only=True)
    model=TabICL(**c['config']);model.load_state_dict(c['state_dict'],strict=True);model.train()
    assert model.dropout==0
    x=torch.tensor([[[0.,0.],[0.,1.],[1.,0.],[1.,1.],[0.,1.],[1.,0.]]]);y=torch.tensor([[0.,0.,1.,1.]])
    records=[];rope=model.row_interactor.tf_row.rope
    with torch.no_grad():
        for enabled in [True,False]:
            model.row_interactor.tf_row.rope=rope if enabled else None
            e=model.col_embedder(x,train_size=4);r=model.row_interactor(e.clone());out=model.icl_predictor(r.clone(),y)[...,:2]
            p=torch.softmax(out/.9,-1)
            e2=model.col_embedder(x,train_size=4);r2=model.row_interactor(e2.clone());changed=model.icl_predictor(r2.clone(),1-y)[...,:2]
            records.append(dict(rope=enabled,query_row_max_difference=float((r[0,4]-r[0,5]).abs().max()),query_probabilities=p[0].tolist(),swapped_label_pre_icl_delta=float((r-r2).abs().max()),swapped_label_logit_delta=float((out-changed).abs().max())))
    model.row_interactor.tf_row.rope=rope
    assert records[1]['query_row_max_difference']<1e-5 and records[0]['query_row_max_difference']>1e-2
    assert all(v['swapped_label_pre_icl_delta']==0 and v['swapped_label_logit_delta']>1e-2 for v in records)
    r=dict(status='PASS',checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),context_x=x[0,:4].tolist(),query_x=x[0,4:].tolist(),context_y=y[0].tolist(),records=records,scope='Original source and original v1 checkpoint, fixed 4-row Cartesian context with identical feature marginals. Disabling row RoPE is a forward intervention in already trained weights, not retraining or reproduction of paper Figure4. Pre-ICL rows cannot inspect labels; final logits can. No preprocessing or score claim.')
    (ROOT/'_symmetry_l066_results.json').write_text(json.dumps(r,indent=2)+'\n');return r
if __name__=='__main__':print(json.dumps(check(),indent=2))
