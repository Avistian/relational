"""A changed live routing implementation must change the probe's recorded identity."""
import json
from pathlib import Path
import torch
import relkit.trompt_l049 as tm
from _verify_trompt_l049 import probe

def check():
    original=tm.prompt_weights
    a=probe(seeds=(0,),epochs=1)
    def uniform_weights(fused,columns):
        return torch.ones((*fused.shape[:2],len(columns)),device=fused.device,dtype=fused.dtype)/len(columns)
    try:
        tm.prompt_weights=uniform_weights
        b=probe(seeds=(0,),epochs=1)
    finally:tm.prompt_weights=original
    fa=a['executable_identity']['functions'];fb=b['executable_identity']['functions']
    assert fa['prompt_weights']!=fb['prompt_weights']
    assert all(fa[k]==fb[k] for k in fa if k!='prompt_weights')
    assert 'model_sha256' not in b and 'canonical_reference_sha256' not in b
    return dict(status='PASS',scope='Actual one-epoch supplied-model probes; changing live prompt_weights changes its identity, retains unrelated function identities; no canonical hash masquerades as student identity')

if __name__=='__main__':
    r=check();Path(__file__).parents[1].joinpath('reviews/lesson-quality-audit-047-070/049-trompt-identity.json').write_text(json.dumps(r,indent=2));print(r)
