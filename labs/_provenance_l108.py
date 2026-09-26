"""Authenticate source, full data precision, frozen cohort, and completed artifacts."""
import hashlib,json,platform,sys
from pathlib import Path
import numpy as np
from relkit.tgat_l103 import load_wikipedia
from _run_l108 import sha,fingerprint
P=Path(__file__).resolve().parent
pins=json.loads((P/'_inputs_l108.json').read_text());source=json.loads((P/'_sources_l103.json').read_text())
assert sha(P/'relkit/tgat_l103.py')==pins['implementation_sha256']
source_count=0
for file,expected in source['files'].items():
 path=P/'sources/l103/original'/file
 assert path.exists(), 'Fetch pinned original source via labs/_fetch_l103.py first'
 assert sha(path)==expected,file;source_count+=1
n,e,d,a=load_wikipedia(P/'l103-cache');t=d['full']['t']
assert np.array_equal(t,t.astype(np.float32).astype(np.float64)), 'Float32 child timestamps lose temporal identity'
assert np.all(np.diff(t)>=0)
with np.load(P/'l103-cache/processed.npz') as cache:
 for field,item in pins['processed_arrays'].items():assert hashlib.sha256(cache[field].tobytes()).hexdigest()==item['sha256']
assert a['split_sha256']==pins['split_sha256'];inputs=0
for seed,files in pins['seeds'].items():
 for file,expected in files.items():assert sha(P/f'results/l103/gpu/seed-{seed}'/file)==expected;inputs+=1
analysis=json.loads((P/'_analysis_l108_results.json').read_text());assert analysis['fingerprint']==fingerprint()
for file,expected in analysis['artifacts'].items():assert sha(P/'evidence/l108'/file)==expected
r={'status':'PASS','pinned_original_source_files':source_count,'source_commit':source['commit'],'raw_sha256':a['raw_sha256'],'events':len(t),'timestamp_range':[float(t.min()),float(t.max())],'all_child_times_exact_in_float32':True,'processed_arrays':'MATCH','splits':'MATCH','cohort_files':inputs,'complete_output_artifacts':len(analysis['artifacts']),'fingerprint':fingerprint(),'historical_identity':'INCOMPARABLE','original_source_execution':'INHERITED_L103_SEED0_NOT_RERUN_L108','checks':{f:sha(P/f) for f in ['_check_l108.py','_audit_l108.py','_resume_l108.py','_collect_l108.py']}}
(P/'_provenance_l108_results.json').write_text(json.dumps(r,indent=2)+'\n');print(r)
