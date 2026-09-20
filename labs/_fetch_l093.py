"""Download released OAG bytes and record provenance; never invent a historical hash."""
import argparse,json,urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'relkit'))
from hgt_l093 import file_sha256
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['NN','CS'],default='NN');parser.add_argument('--directory',type=Path,default=P/'data/l093');a=parser.parse_args()
m=json.loads((P/'_sources_l093.json').read_text());a.directory.mkdir(parents=True,exist_ok=True);out=a.directory/f'graph_{a.dataset}.pk'
url='https://drive.usercontent.google.com/download?id='+m['data_links'][a.dataset]+'&export=download&confirm=t'
if not out.exists():
 temporary=out.with_suffix('.pk.part');urllib.request.urlretrieve(url,temporary)
 if temporary.stat().st_size<1000000:raise RuntimeError('Download is too small; inspect response, not an OAG graph')
 temporary.replace(out)
h=file_sha256(out)
expected=m.get(a.dataset.lower()+'_data',{}).get('sha256')
if expected and h!=expected:raise ValueError('Released bytes changed: stop and audit the new snapshot')
record={'dataset':a.dataset,'filename':out.name,'bytes':out.stat().st_size,'sha256':h,'url':url,'identity_to_paper_snapshot':'NOT_ESTABLISHED'}
(out.parent/(out.stem+'-download.json')).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
