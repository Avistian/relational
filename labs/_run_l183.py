"""Local admission report for retained full protocols; no dispatch or bypass exists."""
import argparse,json
from pathlib import Path
from _audit_l183 import audit183
from relkit.pretraining_l183 import temporal_mask,keyed_mae,factorial_effect,transfer_gate
P=Path(__file__).resolve().parent
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--lane',choices=['relgt','griffin'],required=True);parser.add_argument('--run',action='store_true');args=parser.parse_args()
    E=P/'evidence/l183';r=audit183(E/'packet',json.loads((E/'input-manifest.json').read_text()),temporal_mask,keyed_mae,factorial_effect,transfer_gate)
    print(json.dumps(dict(lane=args.lane,status=r[args.lane+'_full_reproduction'],inherited_forecast_usd=r['inherited_cost_scenarios_usd'][args.lane],full_protocol='labs/l183-reproduction.md',paid_dispatches=0),indent=2))
    if args.run:raise SystemExit('STOP: approved audit scope only; inherited gates unresolved. No training dispatched.')
