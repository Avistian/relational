"""Refresh visible RDB-PFN checkpoint parity without modifying inherited evidence."""
import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
from relkit.rdbpfn_l166 import RDBPFN
P=Path(__file__).resolve().parent;torch.set_num_threads(1)
spec=importlib.util.spec_from_file_location('pfn182',P/'sources/l166/upstream/model_pretrain/src/models.py');original=importlib.util.module_from_spec(spec);sys.modules[spec.name]=original;spec.loader.exec_module(original)
rows=[]
for arm in ['RDBPFN','RDBPFN_single']:
 ref=original.build_model(original.ModelConfig(num_layers=6)).eval()
 original.load_checkpoint(ref,P/'evidence/l166/checkpoints'/(arm+'.pt'),'cpu')
 course=RDBPFN().eval();course.load_state_dict(ref.state_dict(),strict=True);ref.double();course.double()
 torch.manual_seed(182);x=torch.randn(1,7,3,dtype=torch.float64);y=torch.tensor([[0.,1.,0.,1.]],dtype=torch.float64)
 with torch.no_grad():a=ref((x,y),4);b=course((x,y),4)
 torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
 rows.append(dict(arm=arm,max_logit_error=float((a-b).abs().max()),input=x.tolist(),support_labels=y.tolist(),reference_logits=a.tolist()))
r=dict(status='PASS',scope='Both released checkpoints; base numeric float64 forward on fixed7row/3feature input',rows=rows)
(P/'evidence/l200/pfn-parity.json').write_text(json.dumps(r,indent=2)+'\n');print([(x['arm'],x['max_logit_error']) for x in rows])
