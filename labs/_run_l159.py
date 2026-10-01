"""Run the approved local lane; no downloads, GPU, or cloud dispatch."""
import json,platform,time
from pathlib import Path
import torch
from relkit.foundation_preview_l159 import run_experiment,summarize
P=Path(__file__).resolve().parent
if __name__=='__main__':
    started=time.perf_counter();report=run_experiment()
    out=P/'evidence/l159';out.mkdir(parents=True,exist_ok=True)
    (out/'mechanism.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'mechanism.md').write_text(summarize(report)+'\n')
    (out/'runtime.json').write_text(json.dumps(dict(seconds=time.perf_counter()-started,python=platform.python_version(),torch=torch.__version__,cloud_spend_usd=0),indent=2)+'\n')
    print(summarize(report))
