import json,platform,importlib.metadata
from pathlib import Path
from relkit.flatten_b14 import run_diagnostic
P=Path(__file__).resolve().parent;E=P/'evidence/b14'
r=run_diagnostic();(E/'diagnostic.json').write_text(json.dumps(r,indent=2)+'\n')
(E/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','nbformat','nbclient','matplotlib']}),indent=2)+'\n')
print([(x['seed'],x['features'],x['backbone'],round(x['mse'],4)) for x in r['conditions']])
