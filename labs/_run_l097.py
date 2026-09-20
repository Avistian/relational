"""Execute full predefined experiment from scratch and record provenance."""
import hashlib,importlib.metadata as metadata,json,platform
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"relkit"))
from negative_l097 import run_suite
P=Path(__file__).resolve().parent
result=run_suite(P/'data/l095/ml-100k.zip',P/'results/l097')
result['environment']={'python':platform.python_version(),'machine':platform.machine(),'device':'CPU','torch_threads':1,**{k:metadata.version(k) for k in ['numpy','torch','torch-geometric']}}
result['source_sha256']=hashlib.sha256((P/'relkit/negative_l097.py').read_bytes()).hexdigest()
(P/'_experiment_l097_results.json').write_text(json.dumps(result,indent=2)+'\n')
(P/'requirements-l097-observed.txt').write_text('# Python '+platform.python_version()+'\n'+''.join(f'{k}=={metadata.version(k)}\n' for k in ['numpy','torch','torch-geometric','nbformat','nbclient','nbconvert','matplotlib']))
print(json.dumps(result['summary'],indent=2))
