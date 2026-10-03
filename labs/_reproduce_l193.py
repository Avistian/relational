"""Executable admission step for the complete named reproduction.

No paid dispatch is possible while the source gate is failed. This is a complete
preflight/ledger operator, not an implemented or validated GPU backend runner.
"""
import argparse,json,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--show-plan',action='store_true');args=parser.parse_args()
E=P/'evidence/l193';protocol=json.loads((E/'packet/protocol.json').read_text())
if args.show_plan:
 print(json.dumps(dict(protocol=protocol,validation_schedule=str(E/'validation-schedule.json'),selected_test_schedule=str(E/'test-schedule.json'),backend_execution='NOT_IMPLEMENTED_SOURCE_GATE'),indent=2));raise SystemExit(0)
subprocess.run([sys.executable,str(P/'_replay_l193.py')],check=True)
r=json.loads((E/'report.json').read_text())
print('STOP:',r['status'],'; 0/630 model evaluations; USD0 cloud/API.')
print('Continuation requires a reviewed source resolution, full temporal/data/checkpoint audit, exact normalizers, then a measured complete-search forecast and validated backend operator.')
raise SystemExit(2)
