"""Fault probes must reject plausible implementations with the wrong semantics."""
import json
from pathlib import Path
import torch
from _check_l131 import check_summary,check_days,check_readout
from relkit.stack_l131 import activation_summary,relative_days,seed_readout
faults=[('absolute mean',check_summary,lambda x:{**activation_summary(x),'mean':float(x.abs().mean())}),
 ('first query clock',check_days,lambda s,t,b:(s[0]-t)/86400),
 ('missing future check',check_days,lambda s,t,b:(s[b]-t)/86400),
 ('all context nodes',check_readout,lambda x,b:x),
 ('detached roots',check_readout,lambda x,b:x[:b].detach().requires_grad_()),
 ('last roots',check_readout,lambda x,b:x[-b:])]
results={}
for name,check,fn in faults:
 try:check(fn)
 except (AssertionError,RuntimeError,TypeError,IndexError,ValueError):results[name]='REJECTED'
 else:raise AssertionError('Surviving mutant: '+name)
Path(__file__).with_name('_mutation_l131_results.json').write_text(json.dumps(dict(status='PASS',mutants=results),indent=2));print(results)
