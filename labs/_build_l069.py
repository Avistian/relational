"""Canonical L069 evaluation lesson, six live operations and full v2 notebook."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_foundation import markdown2html_mistune
ROOT=Path(__file__).resolve().parent;SLUG='0069-tabpfn-open-environment-failures';TITLE='TabPFN in open environments: measure the broken contract'
TASKS=[
('novel_split','def novel_split(y,held_label,seed):','Construct a real emerging-class task','Return context, query and binary novel arrays; reserve every held-class row and an equal stratified known sample.','Class support changes only when no held-class label can enter context. Row IDs preserve the denominator.','Use a local seeded splitter; match sample count, retain query ordering, and handle a held class smaller than the number of known classes.'),
('novelty_scores','def novelty_scores(probabilities,lower=.4,upper=.6):','Separate a score from a decision','Return continuous and interval arrays from normalized class probabilities, including boundary values.','The released script feeds a binary interval decision to ROC/AP; recovering a continuous ranking changes the measured detector.','Reduce only the class axis. A three-class maximum can be below the interval lower bound.'),
('impute_features','def impute_features(context,query,columns):','Remove information with a frozen fallback','Return a fresh query matrix with selected columns replaced by their context means.','Test-derived means or mutated inputs change the intervention.','Check schema and column IDs; an empty column set should preserve every value.'),
('all_row_metrics','def all_row_metrics(y,probabilities,classes,epsilon=1e-12):','Score every required row','Return the counts, all-row/known accuracy and clipped losses specified in the function contract below.','Unsupported targets cannot silently disappear or be mapped to probability-column indices.','Build a semantic label-to-column map; zero target mass needs a declared finite diagnostic, and no-known-row subsets need None.'),
('dataset_summary','def dataset_summary(records):','Aggregate at the dataset unit','Return per-axis/dataset/arm/condition means, seed values and sample SD for available numeric metrics.','First average class interventions within each seed, then seeds; datasets remain distinct.','The provided contract names the metric fields. Use None for SD with one seed; missing AUC is not zero.'),
('shift_task','def shift_task(context_x,query_x,kind):','Change one probability law','Return source X/y and query X/y for iid, covariate and concept cases in the two-feature threshold family.','Same-X label reversal exposes what any frozen predictor cannot know.','Source rule is coordinate 0>0; covariate moves query coordinate 0 by 1.5 and preserves the rule; concept preserves query X and reverses the rule.')]
CAPTIONS={
'pipeline':'One query ID through context selection, full historical model, class mapping and all-row evaluation. The available labels stop at the context boundary. Evaluation labels appear only after probabilities exist.',
'novelty':'Synthetic six-row arithmetic used in the source check: continuous novelty ROC-AUC 8/9 and AP 11/12; the [.4,.6] confidence interval reduces both to .5. The most uncertain row is missed by the lower bound.',
'features':'Synthetic two-query trace: context sensor [2,4,6] supplies mean 4, erasing its distinction between query 1 and query 7. Context labels and the other column stay fixed.',
'shift':'Generated-law controls: covariate shift moves X and retains the threshold rule; concept reversal retains exactly the same X and reverses y. A distance based only on fixed f(X) cannot detect the latter.',
'objectives':'Synthetic 90:10 confusion matrix, always class 0: accuracy .90, balanced accuracy .50, minority F1=0 and macro F1=.4737. Identical predictions, different metric definitions.',
'feature_results':'Fresh full-model feature-loss curves for every actual removal level; connected means and sample SD summarize three split/mask seeds. Faint dots retain seed outcomes. Iris nominal 20% removes zero of four columns and coincides with 0%.',
'results':'Fresh full-row diagnostic, AUC detail axis .3–.9. Panels retain dataset units; points/error bars show three actual split/mask repetitions and sample SD, not independent-dataset confidence intervals. The single raw numeric v2 view and fixed XGBoost recipe do not reproduce the paper benchmark.'}
def source(name,file='openenv_l069_v2.py'):
 s=(ROOT/'relkit'/file).read_text()
 for n in ast.parse(s).body:
  if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name:return ast.get_source_segment(s,n)
 raise KeyError(name)
def parts():return re.split(r'(<!--figure:\w+-->|<!--results-table-->|<!--analysis-text-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def table_markdown():
 path=ROOT/'_analysis_l069_v2_results.json'
 if not path.exists():return 'Author evidence is being computed; no unmeasured numbers are displayed.'
 return json.loads(path.read_text())['table_markdown']
def analysis_text():
 path=ROOT/'_analysis_l069_v2_results.json';return json.loads(path.read_text())['interpretation'] if path.exists() else 'Full panel NOT_RUN in this intermediate build.'
def figure(name,notebook=False):
 path='figures/l069/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1050 if name in ['results','feature_results'] else 820
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" tabindex="0" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+path+').'
 return '<figure id="l069-'+('pipeline' if name=='pipeline' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l069_v2_results.json','Measured evidence'),(p+'l069-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 069 · six live operations, complete historical checkpoint and EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter. Built with TabPFN.</p></aside>'
CHECKS={
'novel_split':"y=np.repeat([0,1,2],20);z=novel_split(y,1,42)\nassert len(z['context'])==20 and len(z['query'])==40 and z['novel'].sum()==20\nassert not set(z['context'])&set(z['query']) and set(z['context'])|set(z['query'])==set(range(60))\nassert not np.any(y[z['context']]==1) and np.all(y[z['query'][:20]]==1)\nassert np.array_equal(z['query'],novel_split(y,1,42)['query']),'Fixed seed must preserve row identities'",
'novelty_scores':"p=np.array([[.5,.25,.25],[.34,.33,.33],[.8,.1,.1]])\nz=novelty_scores(p);np.testing.assert_allclose(z['continuous'],[.5,.66,.2]);np.testing.assert_array_equal(z['interval'],[1,0,0])\nassert novelty_scores(np.array([[.6,.4]]))['interval'][0]==1,'Interval endpoints are inclusive'",
'impute_features':"cx=np.array([[2.,5.],[4.,6.],[6.,7.]]);qx=np.array([[1.,10.],[9.,20.]]);before=qx.copy();z=impute_features(cx,qx,[0])\nnp.testing.assert_array_equal(z,[[4,10],[4,20]]);np.testing.assert_array_equal(qx,before)\nnp.testing.assert_array_equal(impute_features(cx,qx,[]),qx)",
'all_row_metrics':"z=all_row_metrics(np.array([10,20,30]),np.array([[.8,.2],[.4,.6],[.5,.5]]),[10,20])\nassert z['n']==3 and z['unsupported_n']==1 and abs(z['unsupported_fraction']-1/3)<1e-12\nassert abs(z['clipped_log_loss']-(-np.log(.8)-np.log(.6)-np.log(1e-12))/3)<1e-10\nassert z['known_accuracy']==1 and z['accuracy']==2/3\nassert all_row_metrics([30],[[.5,.5]],[10,20])['known_log_loss'] is None",
'dataset_summary':"r=[dict(axis='novelty',dataset=d,arm='v2',condition='held',seed=seed,auc=a) for d,seed,a in [('A',1,.2),('A',1,.4),('A',2,.7),('B',1,.9)]]\nz=dataset_summary(r);a=next(v for v in z if v['dataset']=='A');b=next(v for v in z if v['dataset']=='B')\nassert abs(a['auc_mean']-.5)<1e-12 and abs(a['auc_sd']-2**.5*.2)<1e-12\nassert len(z)==2 and b['auc_sd'] is None,'Average classes within seed first, do not pool datasets'",
'shift_task':"cx=np.array([[-1.,0],[1.,0]]);qx=np.array([[-.2,3],[.4,4]])\na=shift_task(cx,qx,'iid');b=shift_task(cx,qx,'concept');c=shift_task(cx,qx,'covariate')\nnp.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(b[3],1-a[3]);np.testing.assert_array_equal(a[1],b[1])\nnp.testing.assert_allclose(c[2][:,0],qx[:,0]+1.5);np.testing.assert_array_equal(c[3],[1,1]);np.testing.assert_array_equal(qx,[[-.2,3],[.4,4]])"}
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 069 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Author evidence](_verify_l069_v2_results.json) · [Reproduction](l069-reproduction.md)

**Built with TabPFN.** [Original model license](sources/foundation/v2-LICENSE). Implement six live operations for the paper's evaluation questions, then run complete historical v2 weights. This is a key-parts evaluation mirror with the full pretrained network provided visibly. Default: all CMC/Iris rows, split seed 42, both arms, all CMC held classes and generated IID/covariate/concept controls. The author-reference panel separately covers full CMC/red/white novelty and Iris/CMC/red feature tests with three seeds. CPU works; the ~29 MB immutable checkpoint and small released CSVs are downloaded or reused. No original pretraining, tuned seven-model benchmark or paid/cloud run is performed.

**Workflow:** retrieve and predict before reading the results; implement TODO; use CHECK to diagnose; run your actual namespace; explain the EXIT evidence. Teacher mastery is not inferred from successful execution. The complete recap below makes this notebook usable independently of the lesson.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:md(figure(m[1],True))
  elif p=='<!--results-table-->':md('### Author-reference evidence · not your current-kernel results\n\n'+table_markdown())
  elif p=='<!--analysis-text-->':md(analysis_text())
  elif p.strip():
   p=p.replace('../labs/','')
   p=re.sub(r'(?<![/\w])(006[0-9]-[^)]+\.html)',r'../lessons/\1',p)
   md(p)
 md('## PROVIDED · Your current kernel owns the computation\n\nThe definitions below are the actual implementation. Complete checkpoint inference and the experiment call the live student functions. Source checkers are independent comparators; they do not replace your definitions.')
 code('''# PROVIDED — imports and fresh output directory
import hashlib,importlib.metadata,inspect,itertools,json,math,os,time,types,urllib.request,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,average_precision_score
from xgboost import XGBClassifier
from IPython.display import display
for directory in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (directory/'relkit').is_dir():ROOT=directory.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start in labs')
torch.set_num_threads(1)
output=Path(os.environ.get('L069_OUTPUT',str(ROOT/'data/cache/l069-student'/str(time.time_ns()))))
output.mkdir(parents=True,exist_ok=False)''')
 from relkit import openenv_l069_v2 as c
 from relkit import tabpfn_l069_v2 as m
 code('# PROVIDED — pinned source, data, checkpoint and protocol\n'+'\n'.join(k+'='+repr(getattr(c,k)) for k in ['SOURCE_COMMIT','SOURCE_URL','DATA_SHA','PROTOCOL','PRESETS69'])+'\n'+'\n'.join(k+'='+repr(getattr(m,k)) for k in ['MODEL_CONFIG','CHECKPOINT_SHA','CHECKPOINT_URL','RECIPE']))
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint:** {hint}')
  if name=='all_row_metrics':md('Return keys: `n, unsupported_n, unsupported_fraction, accuracy, known_accuracy, clipped_log_loss, known_log_loss, unsupported_penalty, epsilon`. Known-only values are `None` when support is empty. Predictions use semantic `classes[argmax]`; label 30 is not probability-column 30.')
  if name=='dataset_summary':md('Group keys: `axis,dataset,arm,condition`. Preserve sorted `seeds`. Available metric fields: `accuracy,balanced_accuracy,macro_f1,auc,ap,interval_auc,interval_ap,clipped_log_loss,known_accuracy,unsupported_fraction,delta_accuracy,relative_accuracy_gap`. For each present metric return `<metric>_mean`, `<metric>_sd`, `<metric>_seed_values`, `<metric>_seed_ids` and `<metric>_n` (available seed count). Ignore absent/None measurements; never assign missing AUC zero.')
  code('# TODO — '+name+'\n'+(source(name) if solution else sig+'\n    raise NotImplementedError("Implement this live operation")'))
  code('# CHECK — '+name+'\n'+CHECKS[name]+'\nprint("CHECK passed: '+name+'")')
 model_chunks=[('Numeric feature values and missing flags',['group_features','impute_with_flags','soften_outliers','encode_groups'],'Context means/sample standard deviations define numeric scaling. Missing flags remain distinct channels. Active-group selection depends on all supplied rows, which is why query batching is explicit.'),('Targets and attention axes',['encode_targets','attention_mix','PackedAttention','row_attention','postnorm_update'],'Query targets contain a missing flag and context-derived placeholder. Feature attention mixes columns within a row; row attention receives context keys only. CPU SDPA flattens independent batch/group axes and preserves the original float32-rounded scale.'),('Every pretrained layer and the probability map',['V2Block','TabPFNv2','load_pretrained','ensure_checkpoint','predict_numeric'],'All twelve blocks and 81 checkpoint tensors load strictly. Feature groups and a target token share the blocks; only the query target token enters the head. One raw view at temperature .9 is deliberately simpler than the historical four-view wrapper.')]
 for title,names,why in model_chunks:
  md('## PROVIDED · '+title+'\n\n'+why);code('# PROVIDED — complete model\n\n'+'\n\n'.join(source(n,'tabpfn_l069_v2.py') for n in names))
 chunks=[('Data and metric adapters',['load_dataset','metric_record','align_schema'],'The collector checks original released CSV hashes and retains floating-point values. Metric adapters call your all_row_metrics. Added columns are explicitly ignored by training-name alignment; equality is a control for discarded information.'),('Bind evidence to real code and weights',['stable_code','model_digest','model_runtime_identity','kernel_identity'],'The identity follows nested generator code and actual model class methods, binds defaults and library versions, and fingerprints loaded weights/runtime settings. Hooks and per-instance forward replacements are rejected.'),('Run all four evaluation axes',['run_experiment'],'Follow context/query IDs through your six functions. The balanced and natural-prevalence tasks remain separately named. The same raw model and same checkpoint are used in every intervention; the real task seed and generated law are explicitly recorded.')]
 for title,names,why in chunks:
  md('## PROVIDED · '+title+'\n\n'+why);code('# PROVIDED — '+title+'\n\n'+'\n\n'.join(source(n) for n in names))
 md('## CHECK · Live operations against original source\n\nThis calls isolated source fragments for all 16 held-class partitions and all 15 Iris feature subsets, and compares complete current model logits, input gradients and 81 parameter gradients to original TabPFN 2.0.9. Current-source protocol findings are not historical-result certification.')
 code('''# CHECK — independent source and full model
from _paper_audit_l069_v2 import check as check_protocol
from _check_l064_v2 import check as check_original_model
source_check=check_protocol(globals())
model_check=check_original_model(globals(),save=False)
assert source_check['kernel_identity']==kernel_identity(globals(),ROOT)['sha256']
print('Source split/subset cases',source_check['split_label_cases'],source_check['feature_subset_cases'])
print('Original full model logit maximum error',model_check['full_float64_logit_max_delta'])''')
 md('## CHECK · Hidden helper changes must invalidate the run\n\nA generator expression contains its own code object. Fingerprinting only the outer function misses helper edits. The test temporarily changes an isolated operation, checks the dependency, then restores your actual shift function before prediction.')
 code('''# CHECK — live dependency traversal
saved_shift=shift_task
exec('def hidden69(x):return x+1\\ndef shift_task(*a):return sum(hidden69(x) for x in [1,2])')
first=kernel_identity(globals(),ROOT)['sha256']
exec('def hidden69(x):return x+2')
assert kernel_identity(globals(),ROOT)['sha256']!=first
shift_task=saved_shift
assert source_check['kernel_identity']==kernel_identity(globals(),ROOT)['sha256']
print('Nested helper detection and restore pass')''')
 md('## RUN · Your live full-checkpoint experiment\n\nPredict one non-failure and one failure before executing. Default uses complete CMC and Iris, including every held CMC label and every required test row. Full White Wine contexts in the broader panel are slower. Compare numerical outputs only on matching saved splits; code identities remain separate.')
 code('''# PROVIDED — fresh current-kernel predictions, not author JSON reanalysis
result=run_experiment(ROOT,PRESETS69['lab'],globals())
assert result['kernel_identity']['sha256']==source_check['kernel_identity']
display(pd.DataFrame(result['summary'])[['axis','dataset','arm','condition','accuracy_mean','auc_mean','auc_n','clipped_log_loss_mean']].fillna('—'))''')
 md('## CHECK · Paired predictions and all-row reconstruction\n\nUse the matching author rows as a numerical reference. Your own code identity is never overwritten with the module identity. An edited implementation can be scientifically useful, but must earn new evidence instead of claiming the reference run.')
 code('''# CHECK — exact task keys and numerical reference
reference=json.loads((ROOT/'_verify_l069_v2_results.json').read_text())
maximum=0.
for row in result['records']:
    key=(row['split_id'],row['arm'],row['condition'])
    author=next(a for a in reference['records'] if (a['split_id'],a['arm'],a['condition'])==key)
    assert row['classes']==author['classes']
    error=float(np.max(abs(np.array(row['probabilities'])-author['probabilities'])));maximum=max(maximum,error)
    assert error<5e-5,(key,error)
    split=next(s for s in result['splits'] if s['id']==row['split_id'])
    recalculated=all_row_metrics(split['targets'],row['probabilities'],row['classes'])
    assert abs(recalculated['clipped_log_loss']-row['clipped_log_loss'])<1e-10
print('Maximum probability difference',maximum)
print('Current identity',result['kernel_identity']['sha256'])
print('Author identity',reference['kernel_identity']['sha256'])''')
 md('## EXIT TICKET · A measured failure report\n\nExplain an actual failure and a negative control; compare interval versus continuous novelty, identify the unsupported-row denominator and one objective disagreement. Name the real unit of uncertainty and a next experiment that distinguishes explanations. The length check requests an attempt; a tutor assesses reasoning.')
 interpretation='The generated concept reversal defeats both unchanged models because context and query features remain identical while labels reverse. This negative control establishes evaluator behavior rather than a special flaw of TabPFN. The incremental schema policy returns exactly baseline input, proving that a new sensor is ignored rather than learned. Continuous confidence and the fixed interval measure different detectors, especially when three or more context classes allow maximum probability below 0.4. Balanced leave-one-class-out deliberately makes half of queries unsupported; all-row clipped loss therefore includes the explicit 27.631-nat penalty per unsupported row. Known-only performance must be shown beside coverage. Accuracy, balanced accuracy and macro F1 use different class weighting, so a favorable accuracy cannot establish minority recall. Three split seeds are repetitions inside each dataset. My next experiment would validate an abstention threshold using separate novel classes and measure risk and coverage on untouched natural-prevalence queries. The original tuned seven-model, four-axis benchmark remains unrun.'
 code('# TODO — written diagnosis\ninterpretation='+repr(interpretation if solution else '')+'\nassert len(interpretation.split())>=70,"Explain the measurements and remaining uncertainty"')
 code('''# EXIT — fresh evidence bound to actual definitions
assert result['kernel_identity']['sha256']==source_check['kernel_identity']==kernel_identity(globals(),ROOT)['sha256']
result['source_check']=source_check;result['model_check']=model_check;result['interpretation']=interpretation
path=output/'exit-v2.json'
with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print('EXIT',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('Paper result status:',result['paper_reproduction'])''')
 md('## NEXT STEP · Reproduce the full local panel with your code\n\nThe gate retains all rows and all three seeds in the declared CMC/red/white novelty panel and Iris/CMC/red feature panel. It uses these visible definitions, not a hidden library experiment. This CPU route is the measured closer experiment; it remains INCOMPARABLE to the original tuned benchmark. The reproduction document gives the original source route and precise reconciliation requirements before any historical-table claim.')
 code('''# PROVIDED — full local panel, optional and OFF by default
RUN_BROADER=False
if RUN_BROADER:
    assert kernel_identity(globals(),ROOT)['sha256']==source_check['kernel_identity']
    broader=run_experiment(ROOT,PRESETS69['closer'],globals())
    with (output/'closer-v2.json').open('x') as stream:json.dump(broader,stream,indent=2,allow_nan=False)
else:print('Broader current-kernel run NOT_RUN. Author full local evidence is separately labeled. Original paper benchmark NOT_RUN.')''')
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
   if name in ['novelty','features','shift','objectives']:body+='<div class="l069-widget" id="l069-'+name+'-viz"><p>Interactive controls require JavaScript. The static worked figure above remains available.</p></div>'
  elif p=='<!--results-table-->':body+=markdown2html_mistune(table_markdown())
  elif p=='<!--analysis-text-->':body+=markdown2html_mistune(analysis_text())
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 069 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l069-openenv'])+'</head><body class="l069"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l069-openenv'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0068-pfns-under-temporal-shift.html">← Lesson 068</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 069</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=BeautifulSoup(head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 069</a></nav><h1>Open environments: computation and evidence card</h1>'+launcher()+markdown2html_mistune((ROOT/'l069-reference.md').read_text())+figure('pipeline')+markdown2html_mistune(table_markdown())+'</article></body></html>','html.parser')
 for t in ref.find_all('table'):t.wrap(ref.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(str(ref))
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built complete L069 evaluation package')
if __name__=='__main__':build_package()
