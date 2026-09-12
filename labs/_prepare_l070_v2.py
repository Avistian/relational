"""Stage the exact small public data snapshots needed to audit archived predictions.

No network fetch or cache overwrite. Byte identity matters because historical
artifacts hash the Parquet file, not only its decoded values.
"""
import hashlib,json,shutil
from pathlib import Path

def prepare_inputs(root):
 root=Path(root);folder=root/'data/l070-v2';manifest=json.loads((folder/'manifest.json').read_text());cache=root/'data/cache';cache.mkdir(parents=True,exist_ok=True);records=[]
 for name,item in manifest['datasets'].items():
  source=folder/item['filename'];target=cache/item['filename'];expected=item['sha256']
  if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Bundled snapshot checksum differs: '+name)
  if target.exists():
   if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Existing cache differs; preserve it and use a clean checkout/cache for L070: '+str(target))
   status='VERIFIED_EXISTING'
  else:
   # Exclusive write protects against overwriting a concurrently created cache.
   with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
   if hashlib.sha256(target.read_bytes()).hexdigest()!=expected:raise ValueError('Staged snapshot checksum differs: '+name)
   status='STAGED'
  records.append(dict(dataset=name,sha256=expected,status=status))
 return dict(status='PASS',files=records,manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

if __name__=='__main__':print(json.dumps(prepare_inputs(Path(__file__).parent),indent=2))
