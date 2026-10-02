"""Freeze complete accounting inputs, not a sample of convenient successes."""
import hashlib,json,shutil,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l177';S=P/'sources/l177'
E.mkdir(exist_ok=True);S.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def freeze(p,b):
    if p.exists():assert p.read_bytes()==b,'Frozen input changed: '+str(p)
    else:p.write_bytes(b)
sources={
 'pricing.html':'https://modal.com/pricing',
 'rt-v1.html':'https://arxiv.org/html/2510.06377v1',
 'rdblearn-v1.html':'https://arxiv.org/html/2602.18495v1',
 'torch-memory.py':'https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/torch/cuda/memory.py',
}
for name,url in sources.items():
    if not (S/name).exists():
        with urllib.request.urlopen(url,timeout=45) as response:freeze(S/name,response.read())
for lesson in [175,176]:freeze(S/f'modal-l{lesson}.py',(P.parent/f'modal/l{lesson}_repro.py').read_bytes())
rates=dict(checked='2026-10-02',currency='USD',unit='per second',L4=.000222,physical_cpu=.0000131,memory_gib=.00000222,A100_40=.000583,A100_80=.000694,
           cpu_cores=2,host_gib=16,rt_gpu_count=8,rt_pretrain_hours=2,rt_finetune_hours=1.5,
           interpretation='Posted base rates; no credits, region premiums, invoice or overhead implied')
freeze(S/'rates.json',(json.dumps(rates,indent=2)+'\n').encode())
ledger=dict(checked='2026-10-02',sources=[dict(path=name,url=url,sha256=sha(S/name)) for name,url in sources.items()],rates_sha256=sha(S/'rates.json'))
freeze(S/'source-ledger.json',(json.dumps(ledger,indent=2)+'\n').encode())
paths=list(S.iterdir())
for l in [173,174,175,176]:
    d=P/f'evidence/l{l}'
    paths += [d/'local-budget.json',P/f'_budget_l{l}.py']
paths += list((P/'evidence/l173').glob('*/result.json'))+[P/'evidence/l173/report.json',P/'_run_l173.py',P/'relkit/multitask_l173.py']
paths += list((P/'evidence/l174/runs').glob('*/result.json'))+[P/'evidence/l174/runs/report.json',P/'_run_l174.py',P/'relkit/finetune_l174.py']
for name in ['cloud-budget.json','cost-summary.json','build-reservation.json','audit-2/cost.json','audit-3/cost.json','audit-2/failure.txt','audit-cloud.log','audit-cloud-2.log','audit-cloud-3.log','audit-cloud-4.log','verified-audit.json']:
    paths.append(P/'evidence/l175'/name)
paths += [P/'_dispatch_l175.py',P/'relkit/zero_shot_l175.py',P/'_run_l176.py',P/'relkit/few_shot_l176.py']
for name in ['budget.json','cost.json','cost-decision.json','input-manifest.json']:
    paths.append(P/'evidence/l176'/name)
for phase in ['pilot-1','remaining-1']:
    paths+=list((P/'evidence/l176'/phase).glob('*.json'))
# Verify against previously sealed author evidence wherever available.
inherited={}
for l in [173,174,175,176]:
    m=json.loads((P/f'evidence/l{l}/artifact-manifest.json').read_text())['files']
    inherited.update({k.removeprefix('labs/'):v for k,v in m.items() if k.startswith('labs/')})
verified=0
for p in paths:
    name=p.relative_to(P).as_posix()
    if name in inherited:assert sha(p)==inherited[name],name;verified+=1
manifest=dict(experiment='L177 Compute Feasibility Ledger',files={p.relative_to(P).as_posix():sha(p) for p in sorted(set(paths))},
              inherited_hashes_checked=verified,scope='Full accounting receipts; no checkpoint, prediction or model replay',cloud_usd=0)
freeze(E/'input-manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
print('Frozen',len(manifest['files']),'inputs;',verified,'match inherited seals')
