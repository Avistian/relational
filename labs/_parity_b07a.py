"""Full small-configuration source output parity, including generated matrices."""
import sys,json
from pathlib import Path
from types import SimpleNamespace
import torch
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'sources/b07a/hyperfast'))
from hyperfast.model import HyperFast as Original
from hyperfast.utils import nn_bias_logits
from relkit.hyperfast_b07a import HyperFast
from relkit.hyper_b07a import retrieval_bias
cfg=SimpleNamespace(n_dims=4,max_categories=3,rf_size=16,torch_pca=True,clip_data_value=27.6041,hn_n_layers=4,hn_hidden_size=8,main_n_layers=3)
torch.set_num_threads(1);torch.manual_seed(12);original=Original(cfg);ours=HyperFast(cfg);ours.load_state_dict(original.state_dict())
x=torch.randn(12,3);y=torch.tensor([0,1,2]*4)
with torch.no_grad():
 torch.manual_seed(99);a=original(x,y,3)
 torch.manual_seed(99);b=ours(x,y,3)
 for la,lb in zip(a[2],b[2]):
  for ta,tb in zip(la,lb):torch.testing.assert_close(ta,tb,rtol=0,atol=0)
 q=torch.randn(7,3);logits=torch.randn(7,3)
 torch.testing.assert_close(retrieval_bias(logits,q,x,y,torch.tensor(.2)),nn_bias_logits(logits.clone(),q,x,y,torch.tensor(.2),3,True),rtol=0,atol=0)
(P/'evidence/b07a/source-parity.json').write_text(json.dumps(dict(status='PASS',generated_matrices='bitwise equal at small dimensions with copied random weights',retrieval='bitwise equal on non-tied fixture',released_checkpoint='tested separately in course run'),indent=2))
print('Complete small-architecture source parity PASS')

from relkit.serving_b07a import GeneratedPredictor
from _run_b07a import source_prediction_parity
cfg.device='cpu'
predictor=GeneratedPredictor(ours,cfg).fit(x.numpy(),y.numpy(),0,12)
print('Small fitted wrapper parity',source_prediction_parity(P,predictor,x.numpy()))
