"""Run every declared database and save exact graph/SQL equality evidence."""
import hashlib,importlib.metadata as metadata,json,platform,sqlite3
from pathlib import Path
from relkit.schema_l096 import run_suite
P=Path(__file__).resolve().parent
result=run_suite()
result['environment']={'python':platform.python_version(),'sqlite':sqlite3.sqlite_version,**{k:metadata.version(k) for k in ['torch','torch-geometric']}}
result['source_sha256']=hashlib.sha256((P/'relkit/schema_l096.py').read_bytes()).hexdigest()
(P/'_experiment_l096_results.json').write_text(json.dumps(result,indent=2)+'\n')
(P/'requirements-l096-observed.txt').write_text('# Python '+platform.python_version()+'; SQLite '+sqlite3.sqlite_version+'\n'+''.join(f'{k}=={metadata.version(k)}\n' for k in ['numpy','torch','torch-geometric','nbformat','nbclient','nbconvert','matplotlib']))
print({'status':result['status'],'databases':result['databases'],'worked':result['records'][0]['query']})
