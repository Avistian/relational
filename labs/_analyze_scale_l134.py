"""Independent aggregate timings, query coverage, graph census and spend ledger."""
import hashlib,json,math
from pathlib import Path
P=Path(__file__).parent;s=json.loads((P/'evidence/l134/scale/scale.json').read_text())
assert s['query_contract']['status']=='PASS'
assert s['status']=='COMPLETE' and s['time_unit'].startswith('UNIX seconds')
assert sum(s['rows'].values())>=1_000_000
assert 2*sum(s['sql_forward_edges'].values())==s['directed_edges']
assert s['source_sha256']==hashlib.sha256((P/'_run_scale_l134.py').read_bytes()).hexdigest()
assert s['database']['sha256']=='5a97bf65a926529143e96f6413b2b0550ca55c01247f85156bfb999ee903e94e'
assert s['task']['historical_match'] and not s['database']['historical_match']
for c in s['configurations']:
 rows=c['batches'];q=sum(x['queries'] for x in rows)
 assert q==2048 and len(c['warmup'])==2
 assert sorted(i for x in rows for i in x['query_indices'])==list(range(q))
 t=math.fsum(x[k] for x in rows for k in ['sample_s','transfer_s','step_s'])
 assert math.isclose(q/t,c['summary']['queries_per_second'],rel_tol=1e-12)
 assert all(x['nodes']<=x['bound'] and x['peak_allocated_bytes']<=x['peak_reserved_bytes'] for x in rows)
 assert all(math.isfinite(x['loss']) for x in rows+c['warmup'])
b=json.loads((P/'_budget_l134.json').read_text());scale_reserved=sum(x['upper_bound_usd'] for x in b['scale_reservations'])
assert scale_reserved<=5 and b['maximum_worker_usd']+5+b['overhead_reserve_usd']<=10
r=dict(status='PASS',nodes=sum(s['rows'].values()),directed_edges=s['directed_edges'],configurations=len(s['configurations']),measured_queries=2048*len(s['configurations']),full_task_training='NOT_RUN',historical_database_identity='NOT_ESTABLISHED',scale_reserved_upper_bound_usd=scale_reserved,allocated_total_ceiling_usd=b['maximum_worker_usd']+5+b['overhead_reserve_usd'])
(P/'_scale_audit_l134_results.json').write_text(json.dumps(r,indent=2));print(r)
