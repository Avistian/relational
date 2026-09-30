"""Named historical-protocol lanes retained intact, separate from the course fit.

Use --audit for a safe preflight. Source tokens must pass temporal audits and
an explicit cost decision must authorize the complete RelGT search. This gate
cannot turn corrected K32 course scores into paper-table evidence.
"""
import argparse,json
from pathlib import Path
P=Path(__file__).resolve().parent

def preflight():
    a=json.loads((P/'evidence/l145/prepared/audit.json').read_text())
    d=json.loads((P/'evidence/l145/cost-decision.json').read_text())
    return dict(full_selected_reproduction='INCOMPLETE',temporal=a['temporal_status'],cost=d['decision'],
                historical_selection='NOT_ESTABLISHED',full_schedule='9 configurations x 100 epochs',
                canonical_rdl='labs/_run_l117.py: five seeds, ten full epochs each; NOT_RUN for L146')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--audit',action='store_true');parser.add_argument('--prior');parser.add_argument('--output');args=parser.parse_args()
    status=preflight();print(json.dumps(status,indent=2))
    if not args.audit:
        if status['temporal']!='PASS' or status['cost']!='PROCEED':
            raise SystemExit('STOP: complete source reproduction fails temporal/cost preflight; no training launched.')
        if not args.prior or not args.output:parser.error('--prior and --output required')
        from _full_l145 import full_search
        full_search(P/'evidence/l145/prepared',args.prior,P/'sources/l145',args.output)
