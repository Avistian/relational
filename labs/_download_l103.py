"""Download only final evidence/checkpoints; leave intermediate epochs on the volume."""
import concurrent.futures,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
selection=json.loads((P/'_run_selection_l103.json').read_text())
out=Path(sys.argv[1]) if len(sys.argv)>1 else P/'results/l103/gpu'
files=[(seed,name) for seed in selection['seeds'] for name in ['result.json','identity.json','predictions.npz','selected.pt','source_parity.json']]+[(0,'original-replay.json')]
def download(item):
 seed,name=item;dest=out/f'seed-{seed}'/name;dest.parent.mkdir(parents=True,exist_ok=True)
 remote=selection['directory_template'].format(seed=seed)+'/'+name
 result=subprocess.run([str(R/'.venv/bin/modal'),'volume','get',selection['volume'],remote,str(dest),'--force'],capture_output=True,text=True)
 if result.returncode:raise RuntimeError(f'{remote}: {result.stderr}')
 return str(dest)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for path in pool.map(download,files):print(path,flush=True)
