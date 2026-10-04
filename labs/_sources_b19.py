"""Archive pinned primary text/code and bounded artifact-discovery receipts."""
import hashlib,json,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent/'sources/b19';P.mkdir(parents=True,exist_ok=True)
T='1ce4cae6c12972227dea6247bac83234a2915748';D='72c30d48d26c01fba87b49efbc6f5d6ef67d2049'
urls={'paper.html':'https://arxiv.org/html/2606.30410v1','enterprise.html':'https://arxiv.org/html/2606.30452v1','tabdpt.html':'https://arxiv.org/html/2410.18164v3'}
for name,path in {'methods.py':'packages/tabarena/src/tabarena/contexts/beyondarena/methods.py','context.py':'packages/tabarena/src/tabarena/contexts/beyondarena/context.py','evaluation.py':'packages/tabarena/src/tabarena/evaluation/beyond_arena_eval.py','metadata.csv':'packages/tabarena/src/tabarena/benchmark/task/metadata/sources/data/BeyondArena_tasks_metadata.csv','leaderboard.py':'examples/beyondarena/run_generate_beyondarena_leaderboard.py','readme.md':'examples/beyondarena/README.md','method_metadata.py':'packages/tabarena/src/tabarena/models/_method_metadata.py','public_r2.py':'packages/tabarena/src/tabarena/models/_artifacts/downloader_public_r2.py','r2.py':'packages/tabarena/src/tabarena/models/_artifacts/downloader_r2.py','benchmark-log.md':'packages/tabflow_slurm/BENCHMARK_LOG.md'}.items():urls[name]=f'https://raw.githubusercontent.com/autogluon/tabarena/{T}/{path}'
for dataset in ['musk','sat11_hand_algo_runtime']:
 for regime,path in [('grouped',f'datasets/beyond_iid/grouped/{dataset}/{dataset}.ipynb'),('iid',f'datasets/beyond_iid/_ablations/{dataset}_as_iid/{dataset}.ipynb')]:urls[f'{dataset}-{regime}.ipynb']=f'https://raw.githubusercontent.com/TabArena/data-foundry/{D}/{path}'
for name,repo,sha in [('tabarena-tree','autogluon/tabarena',T),('data-foundry-tree','TabArena/data-foundry',D)]:urls[name+'.json']=f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'
urls['hf-tree.json']='https://huggingface.co/api/datasets/TabArena/BeyondArena/tree/main?recursive=true&limit=1000'
manifest=[]
for name,url in urls.items():
 path=P/name
 if not path.exists():
  with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'B19-source-audit'}),timeout=40) as r:data=r.read()
  path.write_bytes(data)
 manifest.append(dict(file=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(P/'manifest.json').write_text(json.dumps(dict(checked='2026-10-04',tabarena_commit=T,data_foundry_commit=D,files=manifest),indent=2)+'\n')
print('Archived',len(manifest),'sources')
