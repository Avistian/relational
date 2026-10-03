"""Independent paper-table parser, Decimal arithmetic and hostile-input checks."""
import copy,hashlib,itertools,json,math,random,statistics,tempfile,shutil,subprocess,sys
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from _replay_l193 import replay193
from relkit.multitask_l193 import choose_config,summarize_task,aggregate_suite
P=Path(__file__).resolve().parent;E=P/'evidence/l193';Q=E/'packet';tasks=json.loads((Q/'tasks.json').read_text());manifest=json.loads((E/'input-manifest.json').read_text());report=json.loads((E/'report.json').read_text())
class Tables(HTMLParser):
 def __init__(self):super().__init__();self.tables=[];self.table=None;self.row=None;self.cell=None
 def handle_starttag(self,tag,attrs):
  if tag=='table':self.table=[]
  if tag=='tr' and self.table is not None:self.row=[]
  if tag in ['td','th'] and self.row is not None:self.cell=[]
 def handle_data(self,data):
  if self.cell is not None:self.cell.append(data)
 def handle_endtag(self,tag):
  if tag in ['td','th'] and self.cell is not None:self.row.append(' '.join(''.join(self.cell).split()));self.cell=None
  if tag=='tr' and self.row is not None:self.table.append(self.row);self.row=None
  if tag=='table' and self.table is not None:self.tables.append(self.table);self.table=None
parser=Tables();parser.feed((Q/'paper.html').read_text());found=[];groups=[]
for table in parser.tables:
 header=next((r for r in table if 'Dataset' in r and 'RDBLearn' in r),None)
 if header is None:continue
 vals=[];dataset='';index=len(groups)+1
 for r in table[table.index(header)+1:]:
  if len(r)!=len(header):continue
  dataset=r[0] or dataset;name=r[1].split(' (')[0].lower();v=Decimal(r[header.index('RDBLearn')]);vals.append(v)
  found.append((dataset.lower(),name,float(v),index))
 groups.append(vals)
assert found==[(t['dataset'],t['task'],t['paper_score'],t['paper_table']) for t in tasks]
assert list(map(len,groups))==[8,8,5]
assert float(sum(groups[0])/8)==report['published_reference']['relbench_classification']['mean_auc']
assert float(sum(groups[2])/5)==report['published_reference']['4dbinfer_classification']['mean_auc']
assert all(s['mean'] is None and s['sample_sd'] is None and s['completed_seeds']==0 for s in report['task_results'])
# Every possible missingness pattern for an 8-task classification suite.
coverage_checks=0
for bits in itertools.product([False,True],repeat=8):
 small=tasks[:8];summaries=[dict(task=t['id'],status='COMPLETE' if b else 'INCOMPLETE',mean=t['paper_score'] if b else None) for t,b in zip(small,bits)]
 result=aggregate_suite(small,summaries)['relbench_classification'];assert result['complete_tasks']==sum(bits)
 assert (result['mean'] is not None)==all(bits);coverage_checks+=1
rng=random.Random(193)
for i in range(100):
 vals=[rng.random() for _ in range(3)];task=tasks[0]
 runs=[dict(task=task['id'],seed=j,status='COMPLETE',score=v) for j,v in enumerate(vals)]
 r=summarize_task(task,runs,[0,1,2]);mean=sum(vals)/3;sd=math.sqrt(sum((v-mean)**2 for v in vals)/2)
 assert abs(r['mean']-mean)<1e-14 and abs(r['sample_sd']-sd)<1e-14
# Full replay must call the learner contracts, not print a hard-coded report.
correct=replay193(Q,manifest,choose_config,summarize_task,aggregate_suite);assert correct==report
wrong_functions=0
for a,b,c in [(lambda *x:'wrong',summarize_task,aggregate_suite),(choose_config,lambda *x:{'wrong':True},aggregate_suite),(choose_config,summarize_task,lambda *x:{})]:
 try:r=replay193(Q,manifest,a,b,c)
 except (ValueError,KeyError,TypeError):wrong_functions+=1
 else:assert r!=report;wrong_functions+=1
# Both broken bytes and semantically broken-but-rehashed evidence are rejected.
corruptions=0
for variant in ['bytes','missing_job','duplicate_run','fake_score','mapping','test_selected']:
 with tempfile.TemporaryDirectory() as tmp:
  dest=Path(tmp)/'packet';shutil.copytree(Q,dest);m=copy.deepcopy(manifest)
  name={'bytes':'tasks.json','missing_job':'validation-schedule.json','duplicate_run':'runs.json','fake_score':'runs.json','mapping':'preprocessing.json','test_selected':'test-schedule.json'}[variant]
  p=dest/name
  if variant=='bytes':p.write_bytes(p.read_bytes()+b' ')
  else:
   data=json.loads(p.read_text())
   if variant=='missing_job':data.pop()
   if variant=='duplicate_run':data[-1]=data[0]
   if variant=='fake_score':data[0].update(status='COMPLETE',score=.9)
   if variant=='mapping':data['observations'][0]['after'][0]=0
   if variant=='test_selected':data[0]['selected_candidate']='d4-TabPFN-v2'
   p.write_text(json.dumps(data));m['files'][name]=hashlib.sha256(p.read_bytes()).hexdigest()
  try:replay193(dest,m,choose_config,summarize_task,aggregate_suite)
  except ValueError:corruptions+=1
  else:raise AssertionError('Accepted '+variant)
result=subprocess.run([sys.executable,str(P/'_reproduce_l193.py')],capture_output=True,text=True);assert result.returncode==2 and 'STOP' in result.stdout
out=dict(status='PASS',paper_tasks_checked=21,independent_parser='stdlib HTMLParser versus preparation BeautifulSoup',reference_arithmetic='Decimal',coverage_patterns=coverage_checks,variance_oracle_cases=100,wrong_learner_functions_rejected=wrong_functions,corruptions_rejected=corruptions,dispatch_blocked=True,fresh_model_runs=0,regression_aggregate='NOT_ESTABLISHED')
(P/'_verify_l193_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out)
