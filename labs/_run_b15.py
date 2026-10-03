import json
from pathlib import Path
from relkit.labels_b15 import run_experiment
p=Path(__file__).resolve().parent/'evidence/b15/diagnostic.json'
p.write_text(json.dumps(run_experiment(),indent=2)+'\n')
print('Wrote complete 4 rule worlds, 8 column worlds and 16 interventions')
