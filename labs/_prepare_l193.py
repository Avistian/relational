"""Freeze all published task targets and the approved complete validation schedule."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from bs4 import BeautifulSoup
P=Path(__file__).resolve().parent;E=P/'evidence/l193';Q=E/'packet';S=P/'sources/l193'
def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')
def freeze(dst,src):
 dst.parent.mkdir(parents=True,exist_ok=True);b=src.read_bytes()
 if dst.exists():assert dst.read_bytes()==b,dst
 else:dst.write_bytes(b)
for name in ['paper.html','paper-configs.md','fastdfs-METADATA.txt','fastdfs-LICENSE']:
 freeze(S/name,P/'sources/l192'/name)
for directory in ['rdblearn','fastdfs']:
 for p in (P/'sources/l192'/directory).rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc':freeze(S/directory/p.relative_to(P/'sources/l192'/directory),p)
# Authenticate all RDBLearn source files against the named git object, not just inherited hashes.
commit='b5b03ebf8091547285a6e06cba53d2d1a40cb171';repo=Path('/tmp/l178-rdblearn');checked=[]
for p in (S/'rdblearn').rglob('*'):
 if p.is_file():
  relative=str(p.relative_to(S/'rdblearn'))
  assert p.read_bytes()==subprocess.check_output(['git','show',commit+':'+relative],cwd=repo)
  checked.append(relative)
try:
 b=subprocess.check_output(['git','show',commit+':LICENSE'],cwd=repo,stderr=subprocess.DEVNULL)
 (S/'rdblearn/LICENSE').write_bytes(b)
except subprocess.CalledProcessError:pass
soup=BeautifulSoup((S/'paper.html').read_text(),'html.parser');tasks=[];table_index=0
for table in soup.find_all('table'):
 rows=[[c.get_text(' ',strip=True) for c in row.find_all(['td','th'])] for row in table.find_all('tr')]
 header=next((r for r in rows if 'Dataset' in r and 'RDBLearn' in r),None)
 if header is None:continue
 table_index+=1;dataset='';group=['relbench_classification','relbench_regression','4dbinfer_classification'][table_index-1]
 for row in rows[rows.index(header)+1:]:
  if not row or len(row)!=len(header):continue
  dataset=row[0] or dataset;name=row[1].split(' (')[0].lower();metric='MAE' if table_index==2 else 'AUROC'
  tasks.append(dict(id=('relbench' if table_index<3 else '4dbinfer')+'/'+dataset.lower()+'/'+name,dataset=dataset.lower(),task=name,group=group,metric=metric,paper_score=float(row[header.index('RDBLearn')]),paper_baseline=float(row[header.index('AutoGluon w/o RDB')]),paper_table=table_index,normalizer=None))
assert len(tasks)==21 and [sum(t['group']==g for t in tasks) for g in ['relbench_classification','relbench_regression','4dbinfer_classification']]==[8,8,5]
protocol=dict(name='L193 RDBLearn Full 21-Task Validation Search',paper='https://arxiv.org/html/2602.18495v1',rdblearn_commit=commit,rdblearn_release='0.1.2',fastdfs_version='0.2.1',depths=[2,3,4],backends=['TabPFN-v2','TabPFN-v2.5','LimiX-16M'],seeds=[0,1,2],support_limit=10000,selection='Per task and seed: full nine-candidate validation search; listed order breaks ties; complete test once',seed_deviation='Three course repeatability seeds; historical seed identities NOT_ESTABLISHED',cap_usd=10,planned_stop_usd=8,reserve_usd=2,local_cap_seconds=3600,forecast='NOT_ESTABLISHED',normalization='Paper section5 mentions no-RDB naive baseline but does not identify its exact predictor or denominator. Table2 is labeled MAE. Do not substitute AutoGluon w/o RDB or target SD.',historical_environment='NOT_ESTABLISHED',whole_paper='NOT_RUN')
candidates=[f'd{d}-{b}' for d in protocol['depths'] for b in protocol['backends']]
schedule=[dict(task=t['id'],seed=seed,candidate=c,phase='validation',status='NOT_RUN') for t in tasks for seed in protocol['seeds'] for c in candidates]
write(Q/'tasks.json',tasks);write(Q/'protocol.json',protocol);write(E/'validation-schedule.json',schedule)
write(E/'test-schedule.json',[dict(task=t['id'],seed=seed,selected_candidate=None,status='NOT_RUN') for t in tasks for seed in protocol['seeds']])
write(S/'source-ledger.json',dict(commit=commit,git_object_authenticated_files=len(checked),paper_url=protocol['paper'],files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file() and p.name!='source-ledger.json' and '__pycache__' not in p.parts},historical_data_identity='NOT_ESTABLISHED',checkpoint_hashes='NOT_ESTABLISHED',fastdfs_origin='Copied frozen L192/L178 source; wheel hash recorded in L192 source ledger'))
print('Frozen 21 tasks,',len(schedule),'validation jobs, 63 selected-test slots;',len(checked),'git-authenticated source files')
