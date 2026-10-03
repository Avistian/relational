"""Freeze exact published source, extraction and declared eligibility policy once."""
import hashlib,json,shutil
from pathlib import Path
from _replay_l191 import parse_tables
P=Path(__file__).resolve().parent;S=P/'sources/l191';E=P/'evidence/l191';Q=E/'packet'
if Q.exists():raise SystemExit('Frozen packet already exists; refusing overwrite')
Q.mkdir(parents=True)
for p in S.iterdir():
 if p.is_file():shutil.copyfile(p,Q/p.name)
tables=parse_tables((Q/'paper-v1.html').read_text());(Q/'tables.json').write_text(json.dumps(tables,indent=2)+'\n')
policy={r['method']:dict(access='unverified',family='unverified',protocol='reported_same_table',reason='Outside bounded code-access audit; exclusion does not mean proprietary.') for t in tables for r in t['rows']}
for name,file,family in [('RDBLearn','rdblearn-readme.md','foundation'),('Griffin','griffin-readme.md','foundation'),('GraphSAGE','relbench-readme.md','supervised'),('LightGBM','relbench-readme.md','supervised'),('RelGNN','relgnn-readme.md','supervised'),('RelGT','relgt-readme.md','supervised')]:
 if not (Q/file).is_file() or (Q/file).stat().st_size<100:raise ValueError('Missing access source')
 policy[name]=dict(access='open_code',family=family,protocol='reported_same_table',source_file=file,reason='Open implementation inspected; exact historical run, weights, backend license and full reproducibility not certified.')
for name in ['KumoRFM-1','KumoRFM-2']:policy[name]=dict(access='service',family='foundation',protocol='reported_same_table',reason='Paper describes service access; live repository retrieval failed. Historical checkpoint/service pin not established.')
for name in ['Global Median','Entity Median']:policy[name]=dict(access='open_code',family='sanity',protocol='reported_same_table',reason='Sanity baselines retained in full tables; excluded from competitive model pool.')
(Q/'method-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
ledger=dict(as_of='2026-10-02',scope='Bounded primary-source/access check, not exhaustive leaderboard discovery',frozen='arXiv2604.12596v1 Tables3,4,7,8',updates=[
 dict(source='kumo-paper-current.html',finding='Abstract lists v1 dated 2026-04-14; no newer version shown in retrieved history.',admission='FROZEN_SOURCE'),
 dict(source='openrfm-v1.html',finding='OpenRFM is a June2026 paper absent from April Kumo tables. No values spliced into frozen pool.',admission='SEPARATE_PROTOCOL_REVIEW_REQUIRED'),
 dict(source='rdblearn-update.html',finding='July2026 RDBLearn v1.1 paper reports newer comparisons. Different source/version, retained separately.',admission='SEPARATE_PROTOCOL_REVIEW_REQUIRED'),
 dict(source='current-retrieval.json',finding='Kumo GitHub page/API returned404; Hugging Face data endpoint returned401. These failures do not prove the resources do not exist.',admission='INCOMPLETE_ACCESS_AUDIT')],current_global_sota='NOT_ESTABLISHED',fresh_inference='NOT_RUN',missing=['Pinned Kumo checkpoint or immutable service version','Historical data/query identities, context draws and full run configuration','Run-level predictions and seed uncertainty for published tables','Complete current leaderboard and protocol-matched OpenRFM comparison','Current service pricing and bounded all-task cost'])
(Q/'tracking-ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Q.iterdir()) if p.is_file()}
(E/'input-manifest.json').write_text(json.dumps(dict(experiment='L191 RelBench Published-Table Reconstruction',files=files),indent=2)+'\n')
print('Frozen',len(files),'files')
