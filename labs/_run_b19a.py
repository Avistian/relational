import json
from pathlib import Path
from relkit.scoring_b19a import run_experiment
P=Path(__file__).resolve().parent
report=run_experiment();(P/'evidence/b19a/diagnostic.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(report['summary'])
