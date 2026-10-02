"""Collect remote immutable evidence without rerunning compute."""
import argparse,json
from pathlib import Path
import modal
P=Path(__file__).resolve().parent;E=P/'evidence/l176'
p=argparse.ArgumentParser();p.add_argument('phase',choices=['pilot','remaining']);a=p.parse_args()
call=modal.FunctionCall.from_id(json.loads((E/(a.phase+'-call.json')).read_text())['call_id'])
receipt=call.get(timeout=0);volume=modal.Volume.from_name('l176-nested-support-evidence');folder=E/(a.phase+'-1');folder.mkdir(exist_ok=True)
files=['receipt.json','cost.json']+[r['filename'] for r in receipt['records']]+[r['filename'].replace('.npz','.json') for r in receipt['records']]
def collect(name):
    data=b''.join(volume.read_file('/'+a.phase+'-1/'+name));dest=folder/name
    if dest.exists():assert dest.read_bytes()==data
    else:dest.write_bytes(data)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(collect,files))
print('Collected',len(receipt['records']),'evaluations')
