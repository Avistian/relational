import json
from pathlib import Path
from relkit.forecast_b19b import run_experiment
r=run_experiment();p=Path(__file__).resolve().parent/'evidence/b19b/diagnostic.json';p.write_text(json.dumps(r,indent=2)+'\n');print(r['summary']);print('predictions',len(r['predictions']))
