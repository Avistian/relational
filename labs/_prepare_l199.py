"""Freeze complete upstream evidence and exact replay sources without editing them."""
import hashlib,json,platform,subprocess,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;E=P/'evidence/l199';Q=E/'packet'
if (E/'input-manifest.json').exists():raise SystemExit('Already frozen; refusing to overwrite inputs')
Q.mkdir(parents=True,exist_ok=True);files={};origins={}
def put(data,name,origin):
    dest=Q/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    digest=hashlib.sha256(data).hexdigest();files[name]=digest;origins[name]=dict(path=origin,sha256=digest)
def local(path,name):put(path.read_bytes(),name,str(path.relative_to(P.parent)))
up=P/'evidence/l190';manifest=json.loads((up/'input-manifest.json').read_text())
for name,digest in manifest['files'].items():
    source=up/'packet'/name
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,name
    local(source,'evidence/l190/packet/'+name)
for name in ['input-manifest.json','report.json']:local(up/name,'evidence/l190/'+name)
for name in ['_replay_l190.py','_verify_l190.py','_test_l190.py','relkit/checkpoint_l190.py']:local(P/name,name)
for n in [194,197]:local(P/f'evidence/l{n}/report.json',f'receipts/l{n}.json')
policy=dict(candidate_origin='L190 author shortlist; L198 absent at freeze',weights=[1,2,3],ratings='Ordinal author choices, not measured probabilities or utilities',gates={k:'UNKNOWN' for k in ['data','baseline','design','budget']},gate_reasons=dict(data='No measured arrival-history dataset admitted',baseline='No healthy exact matched baseline configuration admitted for proposed task',design='Exact task, useful-effect threshold and dependence-aware analysis are not locked',budget='Full proposed experiment aggregate forecast NOT_ESTABLISHED'),recommendation='INVESTIGATE_AVAILABILITY_FIRST',learner_commitment='NOT_MADE',novelty='NOT_ESTABLISHED')
put((json.dumps(policy,indent=2)+'\n').encode(),'decision-policy.json','L199 approved authored policy')
url='https://www.cos.io/initiatives/prereg'
with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=25) as response:
    data=response.read();assert b'preregist' in data.lower();put(data,'sources/cos-preregistration.html',url)
(E/'input-manifest.json').write_text(json.dumps(dict(files=files,origins=origins),indent=2)+'\n')
protocol=dict(experiment='L199-DIRECTION-SELECTION-AUDIT',mode='Complete saved evidence replay and provisional author decision',inherited_target='RDB-PFN 2603.03805v5 Table 9, rel-f1/driver-dnf',arms=['RDBPFN','RDBPFN_single','TabICLv1.1'],seeds=list(range(10)),support=512,test_queries=702,predictions=21060,literature='All four frozen L188 searches, including incomplete traversals',weight_scenarios=27,rating_sensitivity='Each one-step feasible ordinal perturbation, one field at a time; equal weights',receipts_only=[194,197],new_training='NOT_RUN',cloud_usd=0,cap_seconds=1800,comparison='Full replay JSON must equal frozen L190 report exactly; independent AUROC tolerance 1e-12',selection='No new tuning; preserve original support draws, labels and released DFS',stop='INCOMPLETE at budget or integrity failure; no reduced grid',historical_features='NOT_ESTABLISHED',source_freeze='Complete inherited bytes plus contemporary COS guidance; no new literature search')
(E/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
(E/'environment.txt').write_text(platform.python_version()+'\n'+subprocess.check_output([str(P.parent/'.venv/bin/python'),'-m','pip','freeze'],text=True))
print('Frozen',len(files),'files')
