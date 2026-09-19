"""Pin SubTab release and compare local NT-Xent outputs and gradients."""
import hashlib, importlib.util, json
from pathlib import Path
import requests
import torch
from relkit.contrastive_l072 import subtab_loss
ROOT=Path(__file__).resolve().parent
COMMIT='aa3ab1b97231fc37229ef3e55d98ae13bdbfb4fc'

def run():
    dest=ROOT/'sources/l072';dest.mkdir(parents=True,exist_ok=True)
    files=[]
    for name in ['utils/loss_functions.py','src/model.py','utils/model_utils.py','LICENSE']:
        url=f'https://raw.githubusercontent.com/AstraZeneca/SubTab/{COMMIT}/{name}'
        p=dest/name.replace('/','__')
        if not p.exists():
            r=requests.get(url,timeout=30);r.raise_for_status();p.write_bytes(r.content)
        files.append({'url':url,'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    spec=importlib.util.spec_from_file_location('subtab_upstream',dest/'utils__loss_functions.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    errors=[]
    for n in [2,5]:
        torch.manual_seed(n)
        a=torch.randn(n,7,requires_grad=True);b=torch.randn(n,7,requires_grad=True)
        upstream=mod.JointLoss({'batch_size':n,'tau':.7,'device':'cpu','cosine_similarity':True})
        ours=subtab_loss(a,b,.7);theirs=upstream.XNegloss(torch.cat([a,b]))
        go=torch.autograd.grad(ours,(a,b),retain_graph=True);gt=torch.autograd.grad(theirs,(a,b))
        error=max(float((u-v).abs().max()) for u,v in zip(go,gt))
        assert abs(float((ours-theirs).detach()))<1e-6 and error<1e-6
        errors.append({'batch':n,'value_error':abs(float((ours-theirs).detach())),'gradient_error':error})
    out={'papers':[{'url':'https://arxiv.org/html/2106.15147v2','scope':'SCARF Algorithm 1. Independent implementation; no author-code parity established.'},{'url':'https://arxiv.org/html/2110.04361v1','scope':'SubTab sections 2.1-2.3; released loss audit below.'}],
         'subtab_commit':COMMIT,'files':files,'loss_parity':errors,
         'limitations':'NT-Xent operator parity only. Compact model, noise, epoch count, datasets and probe protocol differ; no full-model/training parity.'}
    (ROOT/'_sources_l072.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['loss_parity']))
if __name__=='__main__':run()
