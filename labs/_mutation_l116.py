"""Ensure live checks reject semantically different faults, not only missing code."""
import json
from pathlib import Path
from torch.nn import functional as F
from _check_l116 import check_loss,check_step,check_seed

def no_zero(m,o,x,a,y,i):
 m.train();loss=F.nll_loss(m(x,a)[i],y[i]);loss.backward();o.step();return loss.detach()
def no_mode(m,o,x,a,y,i):
 o.zero_grad();loss=F.nll_loss(m(x,a)[i],y[i]);loss.backward();o.step();return loss.detach()
def no_step(m,o,x,a,y,i):
 m.train();o.zero_grad();loss=F.nll_loss(m(x,a)[i],y[i]);loss.backward();return loss.detach()
mutants=[('all_node_labels',check_loss,lambda p,y,i:F.nll_loss(p,y)),('wrong_local_labels',check_loss,lambda p,y,i:F.nll_loss(p[i],y[:len(i)])),('no_zero_grad',check_step,no_zero),('no_train_mode',check_step,no_mode),('no_optimizer_step',check_step,no_step),('global_prefix',check_seed,lambda p,y,n,b:F.nll_loss(p[:b],y[:b])),('all_context_labels',check_seed,lambda p,y,n,b:F.nll_loss(p,y[n]))]
caught=[]
for name,check,fn in mutants:
 try:check(fn)
 except AssertionError:caught.append(name)
 else:raise AssertionError('Mutation escaped: '+name)
r={'status':'PASS','rejected_mutations':caught};Path(__file__).with_name('_mutation_l116_results.json').write_text(json.dumps(r,indent=2));print(r)
