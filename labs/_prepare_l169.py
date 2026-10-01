"""Authenticate old evidence, pin readings, and prepare the approved full context sweep."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,shutil,signal,sys,urllib.request
from pathlib import Path
import numpy as np
from bs4 import BeautifulSoup
from relkit.scaling_l169 import sample_context
from _audit_l168 import audit168
from relkit.transfer_l167 import keyed_auc
from relkit.generalization_l168 import paired_gains,database_macro
signal.alarm(600)
P=Path(__file__).resolve().parent;E=P/'evidence/l169';S=P/'sources/l169';OUT=Path('/tmp/l169-input')
for d in [E,S,OUT]:d.mkdir(parents=True,exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
oldpins=json.loads((P/'evidence/l168/audit-manifest.json').read_text())
oldreport=audit168(P,oldpins,keyed_auc,paired_gains,database_macro)
assert oldreport==json.loads((P/'evidence/l168/report.json').read_text())
ledger=[]
for name,url,local in [('survey','https://arxiv.org/html/2506.16654v1',None),('rdbpfn','https://arxiv.org/html/2603.03805v5',P/'sources/l168/rdbpfn.html'),('pricing','https://modal.com/pricing',None)]:
 dest=S/(name+'.html')
 if not dest.exists():
  if local:shutil.copyfile(local,dest)
  else:
   with urllib.request.urlopen(url,timeout=60) as r:dest.write_bytes(r.read())
 ledger.append(dict(name=name,url=url,sha256=sha(dest),retrieved='2026-10-01',reused_from=str(local.relative_to(P)) if local else None))
(S/'source-ledger.json').write_text(json.dumps(dict(sources=ledger),indent=2)+'\n')
soup=BeautifulSoup((S/'rdbpfn.html').read_text(),'html.parser');targets={}
for table,context in zip(range(6,11),[64,128,256,512,1024]):
 section=soup.find(id=f'A6.T{table}');rows=section.find_all('tr');part2=False;found={}
 for row in rows:
  cells=[c.get_text(' ',strip=True) for c in row.find_all(['td','th'])]
  if 'Rel F1' in cells:part2=True;assert cells[1]=='Rel F1' and cells[6]=='Rel Trial'
  if part2 and cells and cells[0] in ['RDBPFN','RDBPFN_single_table','TabICLv1.1']:
   arm=cells[0].replace('RDBPFN_single_table','RDBPFN_single');found[arm]={'rel-f1':float(cells[1]),'rel-trial':float(cells[6])}
 assert len(found)==3;targets[str(context)]=found
(S/'paper-targets.json').write_text(json.dumps(targets,indent=2)+'\n')
sys.path.insert(0,str(P/'sources/l166/upstream/model_pretrain'))
from src.eval_utils import _stable_random_state,downsample_split
specs=[];schedule_checks=0
for spec in oldpins['experiments']:
 db=spec['database'];folder=P/spec['folder'];m=json.loads((folder/'input-manifest.json').read_text());data=np.load(folder/'prepared.npz')
 arrays={k:data[k] for k in data.files if k!='support'}
 for context in [64,128,256,512,1024]:
  schedule=[]
  for seed in range(10):
   token=spec['seed_key'].format(seed=seed);idx=sample_context(len(data['y_train']),context,token)
   x,y=downsample_split(data['X_train'],data['y_train'],context,_stable_random_state(token))
   np.testing.assert_array_equal(x,data['X_train'][idx]);np.testing.assert_array_equal(y,data['y_train'][idx])
   assert len(set(idx))==context and len(np.unique(y))==2
   if context==512:np.testing.assert_array_equal(idx,data['support'][seed])
   schedule.append(idx);schedule_checks+=1
  arrays['support_'+str(context)]=np.array(schedule)
 horizon=(60 if db=='rel-f1' else 365)*86400*10**9
 assert (data['train_keys'][:,1]+horizon<data['test_keys'][:,1].min()).all()
 np.savez_compressed(OUT/(db+'.npz'),**arrays);shutil.copyfile(OUT/(db+'.npz'),E/(db+'.npz'))
 specs.append(dict(database=db,task=spec['task'],seed_key=spec['seed_key'],test_rows=spec['test_rows'],train_rows=spec['train_rows'],features=spec['features'],original_folder=spec['folder'],original_phases=spec['phases']))
old=json.loads((P/'evidence/l168/input-manifest.json').read_text())
for name in ['RDBPFN.pt','RDBPFN_single.pt','tabicl-classifier-v1.1-0506.ckpt']:
 path=Path('/tmp/l168-input')/name;assert sha(path)==old['files'][name]['sha256'];shutil.copyfile(path,OUT/name)
manifest=dict(experiment='L169 RDB-PFN v5 Tables6–10 two-task context sweep',code_revision=old['code_revision'],data_revision=old['data_revision'],tabicl_revision=old['tabicl_revision'],
 contexts=[64,128,256,512,1024],fresh_contexts=[64,128,256,1024],seeds=list(range(10)),arms=['RDBPFN','RDBPFN_single','TabICLv1.1'],experiments=specs,
 files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.iterdir()) if p.name!='input-manifest.json'},paper_targets=targets,descriptive_tolerance=.02,
 source_files=old['source_files'],original_audit_manifest_sha256=sha(P/'evidence/l168/audit-manifest.json'),
 sampling='SOURCE_UNIFORM_WITHOUT_REPLACEMENT_PER_CONTEXT_NOT_NESTED',label_orientation='COMPLEMENT_OF_RAW_TASK_LABELS',
 original_dfs_regeneration='NOT_RUN',historical_identity='NOT_ESTABLISHED',historical_availability='NOT_ESTABLISHED',exact_target_schema_exclusion='NOT_ESTABLISHED')
for p in [E/'input-manifest.json',OUT/'input-manifest.json']:p.write_text(json.dumps(manifest,indent=2)+'\n')
audit=dict(status='PASS',authenticated_original_runs=60,original_predictions=45810,source_sampling_oracle_cases=schedule_checks,all_train_label_horizons_before_test=True,
 original_sources_verified=True,original_label_orientation='RETAINED_COMPLEMENT',fresh_runs_planned=240,reused_runs=60,source_packet_sha256=sha(E/'input-manifest.json'))
(E/'predispatch-audit.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
