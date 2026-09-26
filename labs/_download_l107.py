"""Read completed evidence from Modal; never starts compute."""
import argparse,concurrent.futures
from pathlib import Path
import modal
P=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lane',default='wiki');p.add_argument('--run-id',default='');p.add_argument('--output',default=str(P/'evidence/l107'));p.add_argument('--digest',default='f31332e889f5d2578543a11ef666ea56311f802e15ea0778cbe4c26db344f447');a=p.parse_args();v=modal.Volume.from_name('l107-snapshot-evidence');prefix=f'{a.digest}/'+(f'reruns/{a.run_id}/' if a.run_id else '')+f'full/{a.lane}';entries=[x for x in v.iterdir('/'+prefix,recursive=True) if x.type.name=='FILE'];out=Path(a.output)/a.lane
 def fetch(entry):
  dest=out/Path(entry.path).relative_to(prefix);dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('wb') as f:
   for block in v.read_file(entry.path):f.write(block)
  return str(dest)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for result in ex.map(fetch,entries):print(result,flush=True)
