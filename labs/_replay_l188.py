"""Replay all frozen query records; retain incomplete source coverage."""
import json
from pathlib import Path
from relkit.literature_l188 import replay,triage
P=Path(__file__).resolve().parent;E=P/'evidence/l188'
if __name__=='__main__':
    report=replay(E/'packet')
    (E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k!='papers'})
