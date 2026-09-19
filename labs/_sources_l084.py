"""Refetch/verify pinned release sources and cross-check original GAT Cora data."""
import hashlib,json,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
LAB=Path(__file__).resolve().parent

def verify():
 m=json.loads((LAB/'_sources_l084.json').read_text());rev=m['revision'];checks=[]
 for record in m['files']:
  p=LAB/record['path'];p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():p.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/PetarV-/GAT/{rev}/'+record['upstream'],timeout=60).read())
  assert hashlib.sha256(p.read_bytes()).hexdigest()==record['sha256'],p
 def data_check(r):
  url=f'https://raw.githubusercontent.com/PetarV-/GAT/{rev}/data/'+Path(r['path']).name
  b=urllib.request.urlopen(url,timeout=60).read();digest=hashlib.sha256(b).hexdigest()
  assert digest==r['sha256'],url
  return {'file':Path(r['path']).name,'gat_url':url,'sha256':digest,'matches_tkipf_pinned_data':True}
 with ThreadPoolExecutor(max_workers=4) as pool:checks=list(pool.map(data_check,m['data']))
 r={'status':'PASS','source_files':len(m['files']),'files':checks}
 (LAB/'_data_identity_l084_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
if __name__=='__main__':verify()
