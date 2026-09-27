"""Verify source pins, budget reservation, notebook-safe evidence and launch refusal."""
import hashlib,importlib.metadata,json,platform,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
manifest=json.loads((P/'_sources_l118.json').read_text())
for name,sha in manifest['files'].items():assert hashlib.sha256((P/'sources/l118'/name).read_bytes()).hexdigest()==sha,name
b=json.loads((P/'_budget_l121.json').read_text());p=json.loads((P/'_pilot_l121_results.json').read_text())
for name,sha in b['source_sha256'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==sha,name
assert b['pilot_reservation_usd']+b['overhead_retry_reserve_usd']<b['aggregate_cap_usd']
assert p['projected_five_fold_300_epoch_usd']+b['pilot_reservation_usd']+b['overhead_retry_reserve_usd']>b['aggregate_cap_usd']
r=subprocess.run([sys.executable,str(P/'_run_l121.py'),'--execute'],capture_output=True,text=True)
assert r.returncode!=0 and 'REFUSED' in r.stderr and not (P/'results/l121/full').exists()
neo=json.loads((P/'_neo4j_l121_results.json').read_text());assert neo['status']=='PASS' and neo['graphs']==32 and neo['empty_to_null_cells_corrected']==123
source=json.loads((P/'_source_check_l121_results.json').read_text());assert source['full_labeled_id_fold_identity']=='EXACT'
result={'status':'PASS','pinned_source_files':len(manifest['files']),'pilot_source_hashes':'EXACT','budget_refusal':'PASS, no full output created','five_fold_result':'NOT_RUN','historical_whole_paper':'NOT_ESTABLISHED','environment':{'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in ['torch','numpy','scikit-learn','nbformat','nbclient','matplotlib','playwright']}},'independent_extraction':'32 applicant exact multisets after explicit raw empty/null correction; not full population','learner':'PENDING_WRITTEN_DEFENSE'}
(P/'_audit_l121_results.json').write_text(json.dumps(result,indent=2));print(result)
