"""Canonical L070 v2 package: manuscript, eight figures, six live tasks and fresh fits."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_foundation import markdown2html_mistune
ROOT=Path(__file__).parent;SLUG='0070-foundation-model-checkpoint';TITLE='Defend a foundation-model baseline'
TASKS=[
('binary_loss','def binary_loss(y,p):','Reconstruct probability loss','Return binary log loss with float64 endpoint clipping and reject invalid inputs.','A reversed class column or omitted row can hide behind a plausible saved score.','Check shape, range and class support before reducing row losses; probabilities mean P(y=1).'),
('choose_candidate','def choose_candidate(errors):','Freeze a validation choice','Return the first minimum index of a finite nonempty validation vector.','The real training loop uses this choice before any test prediction.','A tie is a protocol decision; NaN is not a losing candidate to silently ignore.'),
('complete_panel','def complete_panel(records,datasets,arms,seeds):','Require the complete declared panel','Validate the Cartesian design, then return seed means, sample SD, ranks and exploratory rank statistics.','Inferring the roster from observed rows can hide an entirely missing arm.','Compare explicit expected keys; average seeds inside each dataset before assigning average-tie ranks.'),
('dataset_bootstrap','def dataset_bootstrap(differences,repetitions=2000,seed=70):','Resample the paired dataset unit','Return mean, percentile95, units, dataset_means, repetitions, seed and scope from a dataset-by-seed gap matrix.','Fifteen downstream repetitions are not fifteen independent datasets.','Average within each row of the matrix, then resample those dataset means with one local RNG.'),
('lifecycle_cost','def lifecycle_cost(prepare_seconds,predict_seconds,batches):','Price a repeated workload','Return total seconds for one preparation and a declared count of identical query batches.','A fast fit can be overwhelmed by repeated context-conditioned inference.','Validate finite nonnegative inputs; the workload is a simple linear model, not a measured cache speedup.'),
('erase_feature','def erase_feature(train,query,column):','Run a frozen feature-loss intervention','Return a new floating-point query array with one column replaced by its training median.','The live experiment removes train-selected information without refitting or consulting test degradation.','Retain every other coordinate and the input arrays; integer inputs must not truncate a fractional median.')]
CAPTIONS={
'pipeline':'Illustrative row 17: train-only preprocessing, validation candidate selection, then frozen test prediction and scoring. The test label has no arrow back to selection.',
'architectures':'Connected operator comparison. Corrected TabM stores the fitted task in weights; axial PFNs retain feature groups; TabPFN-3 compresses to row embeddings before 24 ICL blocks. The actual TabICLv2 classifier uses 12 ICL blocks and a different head.',
'decoder':'Synthetic pre-log retrieval arithmetic, not final package probabilities: context labels [A,B,A] and weights [.2,.3,.5] produce [.7,.3,0]. Output cardinality does not provide an unseen label with context evidence.',
'selection':'Synthetic validation trace: [0,1] targets; candidate A probabilities [.2,.8], B [.1,.6]. The same loss operator selects A at .2231 versus .3081 before test access.',
'uncertainty':'Synthetic paired-bootstrap trace. Average the three seed gaps inside each dataset, then resample dataset means; a repeated dataset carries its paired evidence together.',
'results':'Actual corrected seven-arm hybrid panel: six archived arms plus freshly fitted corrected TabM. Dots retain three downstream seeds; bars are sample SD. Five dataset blocks support exploratory Friedman/Nemenyi summaries, not broad winner claims.',
'cost':'Synthetic lifecycle arithmetic: A=100+B seconds and B=10+4B seconds meet at 30 identical query batches. This is a workload illustration, not a measured serving or cache benchmark.',
'intervention':'Actual fresh fits on identical rows. Query feature with highest absolute training-target correlation is replaced by its training median; selected predictors are frozen. Positive loss gaps indicate damage. Dots are seeds; bars are sample SD.'}
CHECKS={
'binary_loss':"assert abs(binary_loss([0,1],[.2,.8])+np.log(.8))<1e-12\nassert np.isfinite(binary_loss([1],[0]))\ntry:binary_loss([0,1],[.1,np.nan]);raise AssertionError('Must reject nonfinite probabilities')\nexcept ValueError:pass",
'choose_candidate':"assert choose_candidate([.3,.2,.2])==1\ntry:choose_candidate([]);raise AssertionError('Must reject empty candidate set')\nexcept ValueError:pass",
'complete_panel':"rr=[dict(dataset=d,arm=a,seed=k,error=float(k)) for d in ['A','B','C'] for a in ['x','m','z'] for k in [0,1,2]]\nz=complete_panel(rr,['A','B','C'],['x','m','z'],[0,1,2]);assert z['mean_ranks']==dict(x=2.,m=2.,z=2.)\ntry:complete_panel(rr[:-1],['A','B','C'],['x','m','z'],[0,1,2]);raise AssertionError('Missing crossed cell accepted')\nexcept ValueError:pass",
'dataset_bootstrap':"z=dataset_bootstrap(np.array([[-.1,-.2,-.3],[.2,.3,.4]]),2000,70)\nassert z['units']==2 and abs(z['mean']-.05)<1e-12\nassert z==dataset_bootstrap(np.array([[-.1,-.2,-.3],[.2,.3,.4]]),2000,70)",
'lifecycle_cost':"assert lifecycle_cost(100,1,30)==lifecycle_cost(10,4,30)==130\ntry:lifecycle_cost(-1,2,3);raise AssertionError('Negative preparation cost accepted')\nexcept ValueError:pass",
'erase_feature':"q=np.array([[0,8],[9,10]]);before=q.copy();z=erase_feature(np.array([[2,0],[3,1]]),q,0)\nnp.testing.assert_array_equal(z,[[2.5,8],[2.5,10]]);np.testing.assert_array_equal(q,before)"}
def source(name,file='checkpoint_l070_v2.py'):
 s=(ROOT/'relkit'/file).read_text()
 for n in ast.parse(s).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name:return ast.get_source_segment(s,n)
 raise KeyError(name)
def parts():return re.split(r'(<!--figure:\w+-->|<!--results-table-->|<!--analysis-text-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def analysis():
 p=ROOT/'_analysis_l070_v2_results.json'
 return json.loads(p.read_text()) if p.exists() else dict(table_markdown='Author final fits are in progress; no unmeasured value shown.',interpretation='Final analysis pending.')
def figure(name,notebook=False):
 path='figures/l070/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1150 if name=='results' else 850
 if not (ROOT/path).exists():return 'Final measured figure is being prepared.'
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" tabindex="0" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+path+').'
 return '<figure id="l070-'+('pipeline' if name=='pipeline' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l070_v2_results.json','Measured evidence'),(p+'l070-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 070 · six live operations, fresh fitted baselines and a frozen intervention</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter. Default uses CPU and archived pretrained probabilities.</p></aside>'
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 070 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Author evidence](_verify_l070_v2_results.json) · [Reproduction](l070-reproduction.md)

Implement six evaluation operations and run **fresh corrected TabM-mini and XGBoost fits** on five real numeric tables, three downstream seeds and two candidates. Audit all 105 historical prediction sets, then produce a corrected seven-arm hybrid comparison and a fixed-model feature-erasure test. The historical TabM variant is retained in a separate diagnostic and is not the corrected model. The core skill is the comparison procedure; the corrected model and its trainer are fully visible PROVIDED code.

**Environment:** bounded CPU training, no pretrained checkpoint download on the default path. Colab bootstrap supplies code/data access; exact public data snapshots are bundled and verified before use. The entire recap below is self-contained. **Workflow:** attempt retrieval and prediction, implement TODO, diagnose CHECK, execute the live experiment, and defend EXIT. Figures and tables labeled author evidence are not your current kernel's results. Full original pretraining/benchmark reproduction is INCOMPARABLE/NOT_RUN.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:md(figure(m[1],True))
  elif p=='<!--results-table-->':md('### Author-reference evidence · separate from your live run\n\n'+analysis()['table_markdown'])
  elif p=='<!--analysis-text-->':md(analysis()['interpretation'])
  elif p.strip():md(p.replace('../labs/',''))
 md('## PROVIDED · Imports and immutable inputs\n\nThe original model outputs are input data for the audit. Actual fitting below uses the definitions in this notebook. A fresh timestamped output directory avoids overwriting previous experiments. No resume is promised.')
 code('''# PROVIDED — imports and independent output
import copy,hashlib,importlib.metadata,inspect,itertools,json,math,os,time,types,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata,friedmanchisquare,studentized_range,t
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from IPython.display import display
# Pinned import dependency of the official comparator, not the teaching model.
import importlib.util,subprocess
if importlib.util.find_spec("rtdl_num_embeddings") is None:
    subprocess.check_call([sys.executable,"-m","pip","install","rtdl_num_embeddings==0.0.12"])
for directory in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (directory/'relkit').is_dir():ROOT=directory.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start in labs')
from _prepare_l070_v2 import prepare_inputs
input_preparation=prepare_inputs(ROOT)
print("Exact public data snapshots",input_preparation["status"])
from relkit.data import load_tier_a,CACHE,SPECS
torch.set_num_threads(1)
output=Path(os.environ.get('L070_OUTPUT',str(ROOT/'data/cache/l070-student'/str(time.time_ns()))))
output.mkdir(parents=True,exist_ok=False)''')
 from relkit import checkpoint_l070_v2 as c
 code('# PROVIDED — declared design and prediction before intervention\n'+'\n'.join(k+'='+repr(getattr(c,k)) for k in ['DATASETS','ARCHIVE_ARMS','ARMS','PRESETS70','PROTOCOL70'])+'\nprint(PROTOCOL70["prediction"])')
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint:** {hint}')
  if name=='complete_panel':md('Return keys: `datasets,arms,seeds,values,means,sample_sd,dataset_ranks,mean_ranks,friedman_p,nemenyi_cd,unit`. Values has shape datasets×arms×seeds in the supplied order. Use `None` for sample SD with one seed, Friedman p when fewer than three datasets/models or every row is tied, and critical difference with one model. Use average ties, scipy Friedman on dataset mean columns, and CD=`studentized_range.ppf(.95,k,inf)*sqrt(k*(k+1)/(12*n))`. The exercise is correctly selecting and aggregating the units, not reimplementing a distribution quantile.')
  code('# TODO — '+name+'\n'+(source(name) if solution else sig+'\n    raise NotImplementedError("Implement the live operation")'))
  code('# CHECK — '+name+'\n'+CHECKS[name]+'\nprint("CHECK passed: '+name+'")')
 for title,names,why in [
 ('Correct initialization and ensemble operators',['kaiming_','sign_pm1','batchensemble_linear','packed_head','member_mean_loss','ensemble_predict'],'Row-vector weights have shape [input,output], so fan-in is the penultimate axis. Mini keeps only the first R; probabilities are averaged after per-member softmax.'),
 ('The full corrected numeric TabM',['BatchEnsembleLinear','PackedHead','TabM'],'These are the actual classes constructed by your experiment: shared mini backbone biases, no S adapters, eight independent heads. No model code is hidden behind an import.')]:
  md('## PROVIDED · '+title+'\n\n'+why);code('# PROVIDED — corrected model\n\n'+'\n\n'.join(source(n,'tabm_v2.py') for n in names))
 for title,names,why in [
 ('Data and archive alignment',['load_task','audit_archive'],'Load original rows, reproduce fixed partitions and class semantics, fit training imputation, then reconstruct every saved test loss. The old validation scalar trace cannot by itself reconstruct validation probabilities.'),
 ('Bind actual runtime definitions and weights',['stable_code','model_identity','kernel_identity'],'The graph traverses nested code objects, function defaults, closures, global helpers and actual model methods. Selected weights and runtime settings are separate identities; external libraries have recorded versions.'),
 ('Candidate fitting before test access',['fit_candidate'],'Read this loop line by line. Every candidate sees only train and validation; all epoch losses and final validation probabilities are retained. The live student binary_loss controls epoch selection.'),
 ('Paired fitting, intervention and complete summary',['run_experiment'],'The dynamic namespace calls your six functions. The erased feature is selected by training-only correlation, never by test degradation. The primary panel replaces the old TabM variant and keeps all six other archived arms.')]:
  md('## PROVIDED · '+title+'\n\n'+why);code('# PROVIDED — '+title+'\n\n'+'\n\n'.join(source(n) for n in names))
 md('## CHECK · Independent invariants and corrected-model source parity\n\nRun negative fixtures and compare the visible corrected model to the pinned original TabM under copied weights. Every parameter and input gradient is checked; this establishes an operator comparison, not reproduction of its trained benchmark.')
 code('''# CHECK — behavior and original implementation
from _check_l070_v2 import check as check_operations
from _source_check_l070_v2 import check as check_source
behavior_check=check_operations(globals())
source_check=check_source(globals())
print(behavior_check)
print(source_check['status'],source_check['max_logit_difference'])''')
 md('## CHECK · Changed hidden helpers and model methods invalidate identity\n\nA test changes a helper reached inside a generator expression and the actual XGBoost prediction method, then restores both. Fresh-process identity checks are also recorded by the author. None of these makes arbitrary edited code correct; they prevent evidence being silently attached to different code.')
 code('''# CHECK — the actual live graph
before=kernel_identity(globals(),ROOT)['sha256'];original=erase_feature
exec('def hidden70(x):return x+1\\ndef erase_feature(*a):return sum(hidden70(x) for x in [1,2])')
a=kernel_identity(globals(),ROOT)['sha256'];exec('def hidden70(x):return x+2')
assert kernel_identity(globals(),ROOT)['sha256']!=a
erase_feature=original
method=XGBClassifier.predict_proba
XGBClassifier.predict_proba=lambda self,x:np.zeros((len(x),2))
assert kernel_identity(globals(),ROOT)['sha256']!=before
XGBClassifier.predict_proba=method
assert kernel_identity(globals(),ROOT)['sha256']==before
print('Hidden-helper and actual model-method mutation checks pass')''')
 md('## RUN · Fit fresh baselines, then erase the prespecified feature\n\nDefault runs five real datasets × three seeds × two fitted arms × two candidates. The six archived comparison arms are audited, not refit. Predict a dataset on which redundant features might protect the model, then run. Partial results are saved after every fitted arm/seed; select a new output if interrupted.')
 code('''# PROVIDED — actual live measurement with per-case persistence
result=run_experiment(ROOT,PRESETS70['lab'],globals(),output=output/'partial-v2.json')
assert result['status']=='COMPLETE'
for r in result['records']:
    for candidate in r['candidates']:
        assert abs(binary_loss(r['validation_targets'],candidate['validation_predictions'])-candidate['validation_error'])<1e-12
    assert r['selected']==choose_candidate([c['validation_error'] for c in r['candidates']])
display(pd.DataFrame(result['summary']['means'],index=result['summary']['datasets'],columns=result['summary']['arms']).round(4))
display(pd.DataFrame([dict(dataset=r['dataset'],arm=r['arm'],seed=r['seed'],loss=r['error'],erased=r['intervention']['feature'],delta_loss=r['intervention']['delta_loss']) for r in result['records']]))''')
 md('## CHECK · Fresh probabilities, archived control and uncertainty\n\nCompare your actual predictions to the separately recorded author run under the same protocol, then inspect any deviation. A parity check never overwrites your code identity. Tolerance allows small platform arithmetic differences; a changed recipe requires new evidence and an explicit interpretation.')
 code('''# CHECK — independently saved matching reference
reference=json.loads((ROOT/'_verify_l070_v2_results.json').read_text())
maximum=0.
for row in result['records']:
    author=next(a for a in reference['records'] if (a['dataset'],a['arm'],a['seed'])==(row['dataset'],row['arm'],row['seed']))
    assert row['test_ids']==author['test_ids'] and row['targets']==author['targets']
    delta=float(np.max(abs(np.array(row['predictions'])-author['predictions'])));maximum=max(maximum,delta)
    assert delta<5e-5,(row['dataset'],row['arm'],row['seed'],delta)
print('Maximum author/current-kernel probability gap',maximum)
print('Dataset paired effects (negative favors model):')
display(pd.DataFrame(result['paired_effects']).T[['mean','percentile95','units']])''')
 md('## EXIT TICKET · A defensible deployment and research argument\n\nExplain an actual per-dataset result, corrected versus legacy identity, one failed or supported intervention prediction, the statistical unit and an honest cost limitation. Name a falsifiable relational follow-up and a result that would count against it. A minimum length requests an attempt; a tutor judges the reasoning.')
 interpretation='The corrected seven-arm comparison uses six archived procedures and freshly fitted TabM-mini-v2 on exactly matching test rows; the legacy mini model remains a separately named diagnostic. The pretrained seeds change downstream inference around fixed weights and do not represent independent pretraining histories. On phoneme, corrected mini mean loss .3623 exceeds XGBoost .2667 and TabPFN-3 .2382, whereas on WDBC mini .0758 beats XGBoost .1315. This is a concrete counterexample to a universal winner. Dataset-average loss ranks are descriptive across five small numeric binary tasks. The prespecified feature-erasure experiment freezes fitting and removes the training-most-correlated feature with a train-median fallback. Positive gaps support reliance on that information, while a negative gap can reflect redundancy or harmful fitting and must not be removed from the report. In kc1, mini erasure improves loss by .000151 and .000228 on seeds 1 and 2; the prespecified dataset-average loss increase still holds because seed 0 worsens by .001803. Costs include two candidate fits and validation for fitted arms, but archived wrapper fit/predict boundaries and shared CPU load prevent a controlled efficiency claim. I would test an event-order relational hypothesis by comparing identical legal histories, aggregate features and a frozen future window, then shuffling order while preserving counts. If a strong aggregate baseline matches the relational model within a declared practical margin at lower measured cost, that weakens the need for preserved order in this task. None of this reproduces full paper pretraining or benchmark results.'
 code('# TODO — write your evidence argument\ninterpretation='+repr(interpretation if solution else '')+'\nassert len(interpretation.split())>=80,"Interpret the measured result and its evidence boundary"')
 code('''# EXIT — bind the live result to its actual namespace
assert result['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
result['input_preparation']=input_preparation
result['behavior_check']=behavior_check;result['source_check']=source_check;result['interpretation']=interpretation
path=output/'exit-v2.json'
with path.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
print('EXIT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print(result['paper_reproduction'])''')
 md('## NEXT STEP · Same implementation, larger local optimization budget\n\nThe gate increases TabM to 128 epochs and XGBoost to 800 trees on the same frozen tasks. Archived pretrained inference remains fixed, so this measures a changed local tuning budget. It is NOT the original paper benchmark. For full checkpoint replay, use the original-package operators in the reproduction contract; for a benchmark claim, first implement the named dataset/split/metric/bagging protocol.')
 code('''# PROVIDED — OFF until you choose the longer local experiment
RUN_BROADER=False
if RUN_BROADER:
    broader=run_experiment(ROOT,PRESETS70['closer'],globals(),output=output/'closer-v2.json')
    print(broader['paper_reproduction'])
else:print('Broader current-kernel fits NOT_RUN. Full paper protocols NOT_RUN.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,cell in enumerate(nb.cells):cell.id=hashlib.sha256(f'{SLUG}:v2:{i}:{cell.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True)
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
   if old.metadata.get('execution_verification'):nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for t in soup.select('[id]'):t['id']=unquote(t['id'])
 for t in soup.select('[href],[src]'):
  key='href' if t.has_attr('href') else 'src';u=urlsplit(t[key])
  if not u.scheme and not u.netloc and u.path:t[key]='../'+t[key]
  elif key=='href' and t[key].startswith('#'):t[key]=unquote(t[key])
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'style':'max-width:100%;overflow-x:auto','role':'region','tabindex':'0','aria-label':'Scrollable evidence table'}))
 style=soup.new_tag('style');style.string='th,td{padding:.6rem .8rem!important;white-space:nowrap;word-break:normal;overflow-wrap:normal}';soup.head.append(style)
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'));soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 body=''
 for p in parts():
  match=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if match:
   name=match[1];body+=figure(name)
   if name in ['decoder','selection','cost']:body+='<div class="l070-widget" id="l070-'+name+'-viz"><p>Interactive controls require JavaScript. The static worked figure remains available.</p></div>'
  elif p=='<!--results-table-->':body+=markdown2html_mistune(analysis()['table_markdown'])
  elif p=='<!--analysis-text-->':body+=markdown2html_mistune(analysis()['interpretation'])
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 070 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l070-checkpoint'])+'</head><body class="l070"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l070-checkpoint'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0069-tabpfn-open-environment-failures.html">← Lesson 069</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 070</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=BeautifulSoup(head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 070</a></nav><h1>Baseline decision and evidence card</h1>'+launcher()+markdown2html_mistune((ROOT/'l070-reference.md').read_text())+figure('pipeline')+markdown2html_mistune(analysis()['table_markdown'])+'</article></body></html>','html.parser')
 for t in ref.find_all('table'):t.wrap(ref.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(str(ref))
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built complete L070 v2 package')
if __name__=='__main__':build_package()
