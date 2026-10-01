"""Conservative complete-protocol scenario; early stopping is not assumed."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l164'
p=json.loads((E/'pilot.json').read_text());b=json.loads((E/'budget.json').read_text())
t=max(e['sample_seconds']+e['compute_seconds'] for e in p['events'] if e['phase']=='train')
v=sum(e['sample_seconds']+e['compute_seconds'] for e in p['events'] if e['phase']=='valid')
# Test was intentionally not executed. Scaling by row counts is an estimate;
# actual test sampling may differ, and release reevaluates it on improvement.
test=v*702/566
fits=[]
for size in [512,4096]:
    train=200*(size//256)*t;valid=100*v
    worst_test=101*test
    fits.append(dict(size=size,number_of_fits=10,training_seconds=train,validation_seconds=valid,test_seconds_estimated=worst_test,total_seconds=train+valid+worst_test))
seconds=sum(x['number_of_fits']*x['total_seconds'] for x in fits)
compute=seconds*b['rate_usd_per_second'];reserved=sum(x['upper_usd'] for x in b['reservations'])+b['overhead_reserve_usd']
r=dict(decision='STOP',status='INCOMPLETE_BUDGET_GATE',scenario='200epochs per fit, validation every2epochs, test at every improvement and once finally; 20fits; source16worker throughput unmeasured',fit_scenarios=fits,raw_compute_usd=compute,safety_factor=1.25,safety_adjusted_compute_usd=compute*1.25,existing_reserved_usd=reserved,untouched_reserve_usd=2,remaining_fit_allowance_usd=min(7,10-reserved-2),estimated_total_with_reserves_usd=reserved+2+compute*1.25,pilot_worker_body_usd=json.loads((E/'pilot-cost.json').read_text())['worker_body_seconds']*b['rate_usd_per_second'],billing_invoice='NOT_ITEMIZED',fresh_full_fits=0,test_predictions=0,reserved_seconds=196000,caveat='A conservative allowed-schedule scenario, not an observed full-run cost. Early stopping could reduce runtime; no assumption that all20fits stop early. Probe uses0loader workers; source16workers may improve throughput but prefetch memory has not been validated.')
assert r['safety_adjusted_compute_usd']>r['remaining_fit_allowance_usd']
(E/'cost-decision.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
