"""Run only the author replay cell under a 1.5GiB address-space ceiling."""
import resource
resource.setrlimit(resource.RLIMIT_AS,(1536*1024**2,1536*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(15,15))
import os
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
import ast,json,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
P=Path(__file__).resolve().parent
nb=json.loads((P/'solutions/0132-identity-aware-message-passing.ipynb').read_text())
cell=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('# Author-reference predictions:'))
source=(P/'relkit/identity_l132.py').read_text();node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='mean_average_precision')
namespace={'np':np,'pd':pd,'Path':Path,'json':json,'report':{}};exec(ast.get_source_segment(source,node),namespace)
start=time.perf_counter();status='PASS'
try:
 if '--ipython' in sys.argv:
  from IPython.core.interactiveshell import InteractiveShell
  shell=InteractiveShell.instance();shell.user_ns.update(namespace)
  result=shell.run_cell(cell,store_history=False)
  if not result.success:status='FAIL'
 else:exec(compile(cell,'pilot-replay','exec'),namespace)
except BaseException as e:
 status='FAIL';print(type(e).__name__,str(e)[:300])
print(json.dumps(dict(status=status,seconds=time.perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,limit_mib=1536)))
sys.exit(0 if status=='PASS' else 1)
