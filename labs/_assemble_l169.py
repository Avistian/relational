"""Complete234-run batch from authenticated per-run artifacts after RPC cancellations.
The receipt is explicitly author-assembled, never represented as an original worker receipt.
"""
import hashlib,json,shutil
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;E=P/'evidence/l169';out=E/'remaining-complete';out.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((E/'input-manifest.json').read_text());budget=json.loads((E/'budget.json').read_text())
for name,digest in budget['source_hashes'].items():assert sha(P.parent/name)==digest,name
expected={(d['database'],a,k,s) for d in manifest['experiments'] for a in manifest['arms'] for k in manifest['fresh_contexts'] for s in manifest['seeds'] if not(k==1024 and s==0)}
records={};origins={}
for path in sorted((E/'remaining-2').glob('*.json')):
 if path.name in ['cost.json','receipt.json']:continue
 r=json.loads(path.read_text());identity=(r['database'],r['arm'],r['context'],r['seed']);assert identity in expected and identity not in records
 assert sha(path.with_suffix('.npz'))==r['sha256'];records[identity]=r;origins[identity]='remaining-2'
assert len(records)==225
retry=json.loads((E/'tail-1/receipt.json').read_text());assert retry['input_manifest_sha256']==sha(E/'input-manifest.json');sentinels=0;sentinel_results=[]
for r in retry['records']:
 identity=(r['database'],r['arm'],r['context'],r['seed']);assert identity in expected
 assert sha(E/'tail-1'/r['filename'])==r['sha256']
 if identity in records:
  with np.load(E/'remaining-2'/r['filename']) as old,np.load(E/'tail-1'/r['filename']) as new:
   assert old.files==new.files
   for name in old.files:
    if name!='probability':np.testing.assert_array_equal(old[name],new[name])
   delta=float(np.max(np.abs(old['probability']-new['probability'])))
   sentinel_results.append(dict(database=r['database'],arm=r['arm'],max_probability_difference=delta,auc_difference=r['auc']-records[identity]['auc'],exact_probability_match=delta==0))
  sentinels+=1
 else:records[identity]=r;origins[identity]='tail-1'
assert set(records)==expected and sentinels==6
combined=[]
for identity,r in sorted(records.items()):
 phase=origins[identity];source=E/phase/r['filename'];dest=out/r['filename']
 if dest.exists():assert sha(dest)==sha(source)
 else:shutil.copyfile(source,dest)
 rr=dict(r,origin_phase=phase,origin_record_sha256=sha(source.with_suffix('.json')))
 combined.append(rr)
receipt=dict(receipt_kind='AUTHOR_ASSEMBLED_FROM_ORIGINAL_PER_RUN_RECEIPTS',records=combined,input_manifest_sha256=sha(E/'input-manifest.json'),
 original_complete_worker_receipt='NOT_AVAILABLE_FOR_CANCELLED_REMAINING_2',source_preserved=True,sentinel_runs=sentinels,sentinel_results=sentinel_results,exact_repeatability='PASS' if all(r['exact_probability_match'] for r in sentinel_results) else 'FAIL_TABICL',
 selection_rule='Retain all225complete attempt2 records; tail fills only missing identities; attempt1excluded and sentinel repeats never selected by score',
 composition={'remaining-2':225,'tail-1':9},explanation='Both long synchronous RPC waits were cancelled.225saved complete run files from remaining-2 plus9completed tail runs; six sentinel reruns have identical keys/supports/labels; TabICL probabilities differ slightly, reported without post-hoc tolerance. Repeats excluded from scores.')
path=out/'receipt.json'
if path.exists():assert json.loads(path.read_text())==receipt
else:path.write_text(json.dumps(receipt,indent=2)+'\n')
(E/'recovery-report.json').write_text(json.dumps({k:v for k,v in receipt.items() if k!='records'},indent=2)+'\n');print('Complete234-run assembled packet;',receipt['exact_repeatability']);print(sentinel_results)
