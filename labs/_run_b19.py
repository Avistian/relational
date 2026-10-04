"""Execute every preregistered course condition and preserve original inputs."""
import json,platform,sys
from pathlib import Path
import numpy as np
from relkit.evidence_b19 import run_experiment
P=Path(__file__).resolve().parent
r=run_experiment();(P/'evidence/b19/diagnostic.json').write_text(json.dumps(r,indent=2)+'\n')
(P/'evidence/b19/environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),indent=2)+'\n')
print([(a['seed'],a['regime'],a['model'],round(a['brier'],6)) for a in r['arms']]);print(r['uncertainty'])
