"""Execute the complete declared experiment and record its implementation/environment."""
import hashlib,importlib.metadata as metadata,json,platform,time
from pathlib import Path
from relkit.batching_l098 import run_suite
P=Path(__file__).resolve().parent
start=time.perf_counter();result=run_suite();result['seconds']=time.perf_counter()-start
result['source_sha256']=hashlib.sha256((P/'relkit/batching_l098.py').read_bytes()).hexdigest()
result['environment']={'python':platform.python_version(),**{k:metadata.version(k) for k in ['torch','torch-geometric','pyg-lib']}}
(P/'_experiment_l098_results.json').write_text(json.dumps(result,indent=2)+'\n')
(P/'requirements-l098-observed.txt').write_text('# Python '+platform.python_version()+'\n'+''.join(f'{k}=={metadata.version(k)}\n' for k in ['torch','torch-geometric','pyg-lib','numpy','nbformat','nbclient','nbconvert','matplotlib']))
print({'status':result['status'],'audits':len(result['audit']),'fits':len(result['runs']),'seconds':result['seconds']})
