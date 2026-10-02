"""Independent numerical-path oracle and a diagnostic early-imputation control."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch,torch_frame
from torch_frame.nn.encoder import LinearEncoder
from torch_frame.data.stats import StatType
P=Path(__file__).resolve().parent;E=P/'evidence/l178';torch.set_num_threads(1)
r=json.loads((E/'gradient-preflight.json').read_text());df=pd.read_parquet(E/'gradient-input.parquet');columns=r['numerical_columns']
assert len(df)==r['history_rows'] and (df.date<pd.Timestamp(r['query']['cutoff'])).all()
assert df.driverId.eq(r['query']['driverId']).all()
# Independent NumPy derivation: zero upstream gradient times NaN remains NaN.
x=df[columns].to_numpy(dtype=np.float32);means=np.nanmean(x,axis=0);std=np.nanstd(x,axis=0);normalized=(x-means)/(std+1e-6)
upstream=np.where(np.isnan(normalized),0.,1.)
expected=(~np.isfinite((normalized*upstream).sum(axis=0))).sum()*128
assert expected==r['nonfinite_gradient_elements']==256
# Execute the actual pinned LinearEncoder, then controlled input imputation.
dataset=torch_frame.data.Dataset(df[columns],{c:torch_frame.numerical for c in columns}).materialize();tf=dataset.tensor_frame
stats=[dataset.col_stats[c] for c in columns]
probe=LinearEncoder(out_channels=128,stats_list=stats,stype=torch_frame.numerical);probe.init_modules()
original=probe(tf.feat_dict[torch_frame.numerical]);assert torch.isfinite(original).all();original.sum().backward()
original_bad=int((~torch.isfinite(probe.weight.grad)).sum());assert original_bad==256
probe.zero_grad(set_to_none=True)
features=tf.feat_dict[torch_frame.numerical];clean=torch.where(torch.isnan(features),probe.mean[None,:],features)
control=probe(clean);control.sum().backward();control_bad=int((~torch.isfinite(probe.weight.grad)).sum());assert control_bad==0
result=dict(status='PASS',numpy_expected_nonfinite=int(expected),source_linear_encoder_nonfinite=original_bad,early_imputation_control_nonfinite=control_bad,
 meaning='Confirms NaN-before-affine mechanism; control is not a repaired RelGNN training run',torch_version=torch.__version__,torch_frame_version=torch_frame.__version__)
(E/'gradient-independent.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
