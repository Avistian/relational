"""Execute the complete frozen course grid and separate receipt audit."""
import json,time
from pathlib import Path
from relkit.serving_l186 import CONFIG,run_grid
from _audit_l186 import audit_receipts
P=Path(__file__).resolve().parent;E=P/'evidence/l186'
assert CONFIG==json.loads((E/'config.json').read_text())
report=run_grid();report['receipt_replay']=audit_receipts(E,json.loads((E/'input-manifest.json').read_text()))
(E/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(report['status'],report['cells'],'cells;',report['requests'],'requests;',report['receipt_replay']['runs'],'batch receipts')
