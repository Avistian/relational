"""A minimal mechanism diagnostic, separate from the selected experiment."""
import json
from pathlib import Path
import torch
w=torch.tensor(2.,requires_grad=True);x=torch.tensor(float('nan'))
y=torch.nan_to_num(w*x,nan=0.);y.backward()
assert y.item()==0 and torch.isnan(w.grad)
r=dict(status='PASS',scope='Minimal autograd mechanism, not a fix or a second benchmark',output=0,weight_gradient='NaN',mechanism='Replacing a nonfinite encoded output with zero does not guarantee a finite upstream parameter gradient',source='Pinned StypeEncoder.forward applies nan_to_num after encode_forward')
Path(__file__).with_name('_missing_gradient_l131_results.json').write_text(json.dumps(r,indent=2));print(r)
