"""Diagnostic oracle restricted to the same fixed bars; not a trained predictor."""
import base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
import torch
from relkit.pfn_l061_v2 import riemann_nll
ROOT=Path(__file__).resolve().parent

def decode(item):
 raw=zlib.decompress(base64.b64decode(item['zlib_base64']))
 assert hashlib.sha256(raw).hexdigest()==item['sha256']
 return torch.from_numpy(np.frombuffer(raw,dtype=item['dtype']).copy().reshape(item['shape']))

def fixed_head_diagnostic(evidence):
 borders=torch.tensor(evidence['borders'],dtype=torch.float64);output=[]
 # Same tasks across training seeds; one copy suffices for the analytic head diagnostic.
 for row in [r for r in evidence['records'] if r['seed']==evidence['config']['seeds'][0]]:
  losses=[]
  for batch in row['audit_batches']:
   mu,var,y=decode(batch['gp_mean']),decode(batch['gp_variance']),decode(batch['y'])[:,-1:]
   cdf=torch.distributions.Normal(mu.unsqueeze(-1),var.sqrt().unsqueeze(-1)).cdf(borders[1:-1])
   edges=torch.cat([torch.zeros_like(cdf[...,:1]),cdf,torch.ones_like(cdf[...,:1])],-1)
   mass=(edges[...,1:]-edges[...,:-1]).clamp_min(1e-300)
   losses+=riemann_nll(mass.log(),y.double(),borders).flatten().tolist()
  output.append(dict(regime=row['regime'],n_context=row['n_context'],head_oracle_nll=float(np.mean(losses)),exact_gp_nll=row['gp_nll'],head_excess_nll=float(np.mean(losses)-row['gp_nll'])))
 return dict(status='COMPLETE',meaning='Analytic GP region masses in the identical fixed full-support Riemann family; no neural training; component shapes remain restricted; masses floored at1e-300 for numerical log',records=output)
if __name__=='__main__':
 path=ROOT/'_verify_l061_v2_results.json';r=fixed_head_diagnostic(json.loads(path.read_text()));r['evidence_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 (ROOT/'_analysis_l061_v2_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
