"""Canonical scoped L060 authoring; no shared registry or delivery mutations."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _lesson_depth import enrich_html,enrich_notebook
from _check_l060_v2 import CHECKS
ROOT=Path(__file__).resolve().parent
SLUG='0060-broad-model-comparison';TITLE='A model comparison a skeptic can rerun'
TASKS=[
('validate_partitions','def validate_partitions(ids):','Validate the row boundary','Accept exactly train, val and test, each a nonempty one-dimensional integer ID array; reject duplicates within and across parts. Return True.','No later statistic can repair an overlapped or misaligned evaluation.','Inspect shape before concatenating; uniqueness must hold for the combined IDs.'),
('choose_validation','def choose_validation(errors):','Freeze a candidate','Return the first index of the smallest finite loss in a nonempty one-dimensional sequence.','The selector receives validation losses only; deterministic ties make replay inspectable.','Check all values before selecting an index.'),
('score_predictions','def score_predictions(y,p,regression):','Reconstruct an honest metric','Require aligned finite nonempty vectors and a Boolean task flag. For regression compute RMSE. Otherwise validate labels in {0,1}, probabilities in [0,1], clip with float64 machine epsilon, and compute binary log loss.','Wrong class convention or target units can produce plausible-looking scores.','Validate before clipping. Work out the probability assigned to each observed label.'),
('aggregate_panel','def aggregate_panel(records,datasets,arms,seeds):','Check the declared panel and rank its seed means','Reject duplicate, missing or extra dataset/arm/seed keys against the explicit roster. Average errors over seeds first, then use average-tie rankdata. Return datasets, seeds, arms, details, mean_ranks, dataset_ranks, friedman_p, nemenyi_cd and uncertainty. Each detail includes dataset, arm, mean, sd, seed_values, conditional_t95 and seconds. SD and interval are None for one seed.','A completely missing dataset disappears if the expected roster is inferred from successes. The experiment calls this function on both the full panel and one-seed smoke.','A MultiIndex product expresses expected coverage. See the formulas above for the interval and critical difference; scipy.stats supplies the distribution quantiles.'),
('paired_effect','def paired_effect(records,dataset,arm,baseline):','Retain paired seed differences','For one task and two distinct arms, require matching unique seed sets with at least two pairs. Return n, mean, sd, t95, differences and direction for arm minus baseline, sorted by seed.','Separate unpaired intervals discard covariance and can give the wrong impression of the gap.','Join by seed, subtract first, then compute ddof=1 spread and the t interval.')]
FIGURES={'protocol':('protocol-v2.png','Synthetic worked protocol: validation [.43,.39] selects candidate 1. Only then may test labels enter the evaluator. The selected neural epoch and the candidate index are different choices.'),
'aggregation':('aggregation-v2.png','Synthetic two-seed fixture: rank the mean losses, not the individual-seed ranks. The interactive slider in the lesson keeps this baseline visible.'),
'ranks':('ranks-v2.png','Fresh v2 dataset-level mean ranks and exploratory Nemenyi critical differences. Random and temporal panels have different task rosters; compare regimes on paired tasks.'),
'results':('scores-random-v2.png','Fresh v2 random panel. All three seed points and the mean are shown per task; axes retain each task’s raw error units. No paper result reproduction.'),
'comparison':('scores-temporal-v2.png','Fresh v2 temporal panel, with three seed points per arm. Sberbank uses RMSE of the released log(price_doc/full_sq) target. See the notebook for paired t intervals and all 70 task/arm summaries.')}

def piece(name,module='comparison_l060'):
 s=(ROOT/f'relkit/{module}.py').read_text()
 return next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)

def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())

def figure(name,notebook=False):
 f,caption=FIGURES[name];path='figures/l060/'+f
 if notebook:return f'![{caption}](data:image/png;base64,{base64.b64encode((ROOT/path).read_bytes()).decode()})\n\n{caption} [Open full-size figure]({path}).'
 return '<figure'+(' id="protocol-overview"' if name=='protocol' else '')+'><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computation figure"><img src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' Scroll horizontally on narrow screens. <a href="../labs/'+path+'">Open full-size figure</a>.</figcaption></figure>'

def launcher(prepared=False):
 prefix='../' if prepared else '../labs/'
 links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l060_v2_results.json','Measured evidence'),(prefix+'l060-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 060 · 5 live code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'"'+(' download' if title=='Download notebook' else '')+'>'+title+'</a>' for u,title in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter. Student functions are intentionally blank.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 060 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Measured evidence](_verify_l060_v2_results.json) · [Source inventory](_sources_l060_v2.json) · [Reproduction](l060-reproduction.md)

**Outcome:** five live audit/selection functions, a fresh five-arm smoke, a fully reconstructed 210-record author panel, and a defensible EXIT report. This is a protocol checkpoint: reused neural implementations remain visible and tree libraries are peripheral baselines. CPU-only core; no pretrained checkpoint. The full eleven-dataset run is author-reference evidence, not a result you have already obtained. Live Colab UI remains unverified. Read, predict, implement, audit, then explain.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:md(figure(m[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
The five TODO functions must remain in this kernel's namespace. The provided driver below calls them directly. We first isolate their behavior on small fixtures, then test them on real saved predictions, then fit fresh models. Source text in this notebook identifies your code; saved author file hashes identify the separate reference operator.''')
 code('''# PROVIDED — setup and CPU limits
import sys,copy,hashlib,json,time,math,itertools,inspect,functools,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata,friedmanchisquare,studentized_range,t
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier,XGBRegressor
from catboost import CatBoostClassifier,CatBoostRegressor
from IPython.display import display
for p in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (p/'relkit').is_dir():ROOT=p.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run the course bootstrap or start inside labs')
from relkit.data import load_tier_a,CACHE,SPECS
__file__=str(ROOT/'relkit/checkpoint_l060_v2.py')
torch.set_num_threads(1)
output=ROOT/'data/cache/l060-student';output.mkdir(parents=True,exist_ok=True)
print({name:importlib.metadata.version(name) for name in ['numpy','scipy','torch','scikit-learn','xgboost','catboost']})''')
 md("""## PROVIDED · Freeze the comparison ledger before reading results
This concrete ledger defines the author experiment you are about to audit. If a Lesson 059 EXIT exists locally, the cell prints its decisions so you can record discrepancies. Its absence does not block the checkpoint. The new smoke intentionally uses its separately declared reduced budget. Your final report must retain this ledger identity and label any post-result adjustment as a future hypothesis.""")
 code('''# PROVIDED — a concrete prior decision record, before fitting
frozen_ledger={'metric':'binary log loss; RMSE of released log target','candidate_procedures':'five numeric/one-hot arms; corrected TabM-mini','split_identity':'public60/61; TabReD split0; IDs in author artifact','tuning_budget':'two candidates,24 epochs,100 trees,seeds0/1/2','selection_rule':'first minimum candidate; last minimum epoch','freeze_point':'before selected predictor sees test rows','postponed_adjustment':'native categorical route and new windows require fresh evaluation'}
frozen_ledger_sha256=hashlib.sha256(json.dumps(frozen_ledger,sort_keys=True).encode()).hexdigest()
prior_path=ROOT/'data/cache/l059-student/exit.json'
prior_ledger=json.loads(prior_path.read_text()).get('ledger') if prior_path.exists() else None
print('L059 ledger:',prior_ledger if prior_ledger else 'Not present in this runtime; use your saved seven decisions')
print('Frozen L060 declaration:',frozen_ledger)
print('Declaration identity:',frozen_ledger_sha256)''')
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Implement the live comparison operation")'))
  code(CHECKS[name])
 md('''## PROVIDED · Validate every link in the author artifact
The following auditor is separate from fitting. It checks each row ID, shared target sequence, candidate choice and reconstructed metric, then calls your explicit-roster aggregator. A hash can identify an artifact; it cannot substitute for these semantic checks. Read why the target sequence must agree across both arms and seeds.''')
 code('# PROVIDED — artifact integrity\n'+piece('audit_records'))
 code('''# PROVIDED — run the whole evidence audit through your functions
reference=json.loads((ROOT/'_verify_l060_v2_results.json').read_text())
assert reference['status']=='COMPLETE', 'An incomplete run cannot support this checkpoint'
summaries=audit_records(reference)
for regime,summary in summaries.items():
    print(regime,summary['datasets'],'dataset blocks')
    display(pd.DataFrame(summary['details'])[['dataset','arm','mean','sd','conditional_t95','seconds']])
    display(pd.Series(summary['mean_ranks'],name='Mean rank'))
    print('Exploratory asymptotic Friedman p:',summary['friedman_p'],'; exploratory CD:',summary['nemenyi_cd'])
paired=[dict(dataset=d,arm=a,**paired_effect(reference['records'],d,a,'XGBoost')) for d in reference['datasets'] for a in reference['design']['arms'] if a!='XGBoost']
display(pd.DataFrame(paired)[['dataset','arm','mean','sd','t95']])
cost=pd.DataFrame(reference['records']).groupby('arm')[['fit_selection_seconds','predict_seconds']].sum()
display(cost)
print('Costs exclude common loading/encoding and are confounded by variable host load; no hardware-normalized speed claim.')''')
 md('''### CHECK · Evidence mutation and roster omission
Predict what happens if a saved target changes after the validation choice. Selection must remain identical, while the reported error can no longer match the original prediction artifact. Then remove a whole dataset; an explicit expected roster must reject that too. These probes check the saved evidence boundary, not a proof of semantic feature legality.''')
 code('''# CHECK — no test-driven choice; tampering must be detected
bad=copy.deepcopy(reference);bad['records'][0]['targets'][0]=1-bad['records'][0]['targets'][0]
assert choose_validation(bad['records'][0]['validation_errors'])==reference['records'][0]['selected']
for corrupted in [bad,dict(reference,records=[r for r in reference['records'] if r['dataset']!=next(iter(reference['datasets']))])]:
    try:audit_records(corrupted)
    except ValueError:pass
    else:raise AssertionError('Audit accepted changed targets or a missing declared task')
print('Target mutation and whole-task omission rejected')''')
 md('''## PROVIDED · Calibrate rank dispersion without a large-sample fiction
The exact temporal enumeration fixes the first task's labels because simultaneously renaming every method leaves the statistic unchanged. Enumerate all permutations in each remaining task: 120²=14,400 equally weighted arrangements for five methods. The larger panel uses 20,000 Monte Carlo arrangements with a plus-one correction. This tests exchangeability in the declared blocks, not representativeness of their sampling.''')
 code('# PROVIDED — conditional permutation calibration\n'+piece('rank_permutation','rank_audit_l060'))
 code('''# PROVIDED — your seed-first ranks determine the calibrated test
permutation={reg:rank_permutation(list(s['dataset_ranks'].values())) for reg,s in summaries.items()}
display(pd.DataFrame(permutation).T)
assert permutation['temporal']['draws']==14400
# Degenerate tie fixture has no dispersion to reject.
assert rank_permutation(np.full((3,3),2.))['p']==1.''')
 md('''## PROVIDED · Reused model mechanics, visible from input to prediction
The following chunks are copied from the corrected canonical teaching models. These are provided because this lesson tests evaluation rather than redoing the architecture exercises. Trace the dimensions and objective anyway: a wrong reusable model invalidates an otherwise clean comparison. The model names in the fresh driver below resolve to these local definitions.''')
 groups=[('tabm_v2','Input diversity, member loss and probability averaging',['kaiming_','sign_pm1','batchensemble_linear','packed_head','member_mean_loss','ensemble_predict']),('tabm_v2','Shared mini backbone and distinct heads',['MLP','BatchEnsembleLinear','PackedHead','TabM']),('realmlp','Training-derived robust coordinates and scaled linear map',['robust_parameters','smooth_clip','RobustSmooth','coslog4','ntp_linear']),('realmlp','RealMLP layers, feature scales and optimizer groups',['NTPLinear','RealMLPS'])]
 live_model_names=[]
 for module,title,names in groups:
  md('### PROVIDED · '+title)
  code('# PROVIDED — '+title+'\n\n'+'\n\n'.join(piece(n,module) for n in names));live_model_names+=names
 md('''## PROVIDED · Fitting is separated from test evaluation
First inspect the train-only encoder and splitter. Then follow fit_candidate: tree candidates return after fixed rounds; neural candidates fit all epochs and restore the best validation state. Its signature contains no test data. The runner calls your selector, predicts, and scores in that order. `fit_selection_seconds` includes every candidate and its validation work.''')
 for name in ['random_task','encode_train','fit_candidate']:
  code('# PROVIDED — '+name+'\n'+piece(name,'checkpoint_l060_v2'))
 code('# PROVIDED — declared budgets\nPRESETS='+repr(__import__('relkit.checkpoint_l060_v2',fromlist=['PRESETS']).PRESETS))
 code('# PROVIDED — complete fresh fitting driver; no existing output is overwritten\n'+piece('run_checkpoint','checkpoint_l060_v2'))
 identity_names=[v[0] for v in TASKS]+['audit_records','rank_permutation','random_task','encode_train','fit_candidate','run_checkpoint']+live_model_names
 code('# PROVIDED — capture actual kernel code; repository hashes are reference identities\nKERNEL_NAMES='+repr(identity_names)+'''
def capture_kernel_identity():
    sources={}
    for name in KERNEL_NAMES:
        obj=inspect.unwrap(globals()[name])
        if inspect.isclass(obj):
            sources[name]={method:inspect.getsource(value) for method,value in vars(obj).items() if inspect.isfunction(value)}
        else:sources[name]=inspect.getsource(obj)
    content={'sources':sources,'presets':copy.deepcopy(PRESETS),'capture_source':inspect.getsource(capture_kernel_identity)}
    content['sha256']=hashlib.sha256(json.dumps(content,sort_keys=True).encode()).hexdigest()
    return content

def attach_kernel_identity(result,path):
    result['repository_source_hashes_are_reference_only']=True
    result['source_hashes_scope']='Repository reference files; actual live notebook functions and class methods are recorded separately'
    result['kernel_identity']=capture_kernel_identity()
    Path(path).write_text(json.dumps(result,indent=2,allow_nan=False)+'\\n')
    return result
''')
 md('''## Run a fresh five-arm smoke through your live functions
This small execution check uses only diabetes, one seed, one candidate and six epochs. It cannot establish a comparative winner. We count the calls to all five functions across a full saved-panel audit, paired-effects calculation and fresh fitting. The student functions are restored afterward. The run refuses to reuse an existing result filename; each invocation gets a new name.''')
 code('''# PROVIDED — substantive live wiring check and fresh measurements
names=['validate_partitions','choose_validation','score_predictions','aggregate_panel','paired_effect']
originals={n:globals()[n] for n in names};calls={n:0 for n in names}
def counted(name,fn):
    @functools.wraps(fn)
    def wrapper(*args,**kwargs):calls[name]+=1;return fn(*args,**kwargs)
    return wrapper
for n,fn in originals.items():globals()[n]=counted(n,fn)
try:
    summaries=audit_records(reference)
    paired=[dict(dataset=d,arm=a,**paired_effect(reference['records'],d,a,'XGBoost')) for d in reference['datasets'] for a in reference['design']['arms'] if a!='XGBoost']
    fresh_path=output/f'smoke-{time.time_ns()}.json'
    with threadpool_limits(1):fresh=run_checkpoint('smoke',fresh_path)
    attach_kernel_identity(fresh,fresh_path)
finally:
    for n,fn in originals.items():globals()[n]=fn
assert all(v>0 for v in calls.values()), 'Every live function must reach actual analysis or fitting'
assert len(fresh['records'])==5 and fresh['status']=='COMPLETE'
print('Live call counts:',calls)
display(pd.DataFrame(fresh['records'])[['arm','error','selected','epoch']])''')
 md('''## EXIT TICKET · Write the claim you can defend
Save the declared protocol, reconstructed summaries, paired effects, calibrated tests, fresh smoke and live source text. Supply all seven ledger fields. If your Lesson 059 ledger differs, identify the difference and classify the new run correctly. In at least 120 words, discuss a concrete per-task gap, a paired random/temporal contrast, the cost definition, your preferred deployment baseline under a stated objective, and a fresh test that could reverse it. State that the paper results are INCOMPARABLE and why. Submit the artifact and completed notebook; written reasoning needs teacher review.''')
 verdict='On Ecom Offers, corrected TabM-mini minus XGBoost has random-split mean log-loss gap +0.04180, with conditional three-seed t95 [0.03109,0.05250]. The temporal gap is -0.11314, with t95 [-0.17868,-0.04760]. The gap changes by -0.15494, reversing the observed preference on that task. This is a hypothesis about differential regime sensitivity, not proof that temporal drift alone caused the reversal. I would prefer TabM-mini for this fixed temporal task if predictive loss were the only objective, while retaining a tree baseline for a fresh test with native categorical processing and measured serving costs. On temporal Homesite, the TabM-minus-XGBoost gap is instead +0.11602, illustrating why the Ecom result cannot support a universal neural winner. Our paired intervals vary only model seeds on fixed sampled rows and are not adjusted for many comparisons. Search timing sums all candidates and validation, excluding common encoding and loading; variable host contention prevents a hardware efficiency claim. A later untouched temporal window with full feature-availability checks, prespecified candidates and matched cost measurements could reverse this decision. TabArena and TabReD paper results remain INCOMPARABLE because their datasets, windows, tuning, ensembling, metrics and repetitions are not replicated.'
 ledger={'metric':'binary log loss; RMSE of released log regression target','candidate_procedures':'five restricted numeric/one-hot arms including corrected TabM-mini','split_identity':'public RNG60/61; TabReD split0; stored row IDs and hashes','tuning_budget':'two candidates,24 neural epochs,100 tree rounds; seeds0/1/2','selection_rule':'first minimum validation candidate; last minimum epoch','freeze_point':'before selected predictor sees test rows; no train+validation refit','postponed_adjustment':'native categorical and larger-budget experiments require a new declared evaluation'}
 code('# EXIT — your report and frozen decisions\nverdict = '+repr(verdict if solution else '')+'\nledger = '+repr(ledger if solution else {k:'' for k in ledger}))
 names=[v[0] for v in TASKS]+['audit_records','rank_permutation','random_task','encode_train','fit_candidate','run_checkpoint']+live_model_names
 code('''# EXIT — persist concrete, inspectable work
assert len(verdict.split())>=120, 'Write the interpretation before submitting'
required={'metric','candidate_procedures','split_identity','tuning_budget','selection_rule','freeze_point','postponed_adjustment'}
assert set(ledger)==required and all(ledger.values()), 'Complete every frozen decision'
live_names='''+repr(names)+'''
live_sources={}
for name in live_names:
    obj=globals()[name]
    if inspect.isclass(obj):
        live_sources[name]={method:inspect.getsource(value) for method,value in vars(obj).items() if inspect.isfunction(value)}
    else:live_sources[name]=inspect.getsource(obj)
artifact=dict(lesson=60,frozen_ledger=frozen_ledger,frozen_ledger_sha256=frozen_ledger_sha256,prior_ledger=prior_ledger,ledger=ledger,verdict=verdict,design=reference['design'],summaries=summaries,paired_effects=paired,permutation=permutation,fresh=fresh,calls=calls,live_sources=live_sources,
    live_sha256=hashlib.sha256(json.dumps(live_sources,sort_keys=True).encode()).hexdigest(),
    author_evidence_sha256=hashlib.sha256((ROOT/'_verify_l060_v2_results.json').read_bytes()).hexdigest(),
    paper_reproduction='INCOMPARABLE',timing_limit='Variable host load; common data loading and encoding excluded')
(output/'exit.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
print('Saved data/cache/l060-student/exit.json. Submit this and your completed notebook.')''')
 md('''## NEXT STEP · Refit the corrected panel at a declared budget
This is a runnable extension, with the same visible functions. `lab` repeats the 210-record panel; `closer` increases sample/epoch/tree budgets. Neither is TabArena's or TabReD's original protocol. The gate is off; full TabReD data may need downloading. Reruns choose fresh output names. For unattended CPU work use the separately supplied Modal wrapper described in the reproduction contract. No cloud run is implied by this notebook.''')
 code('''# PROVIDED — explicit optional execution gate after the required EXIT
RUN_LARGER_LOCAL=False
REPRO_PRESET='closer'
if RUN_LARGER_LOCAL:
    from _fetch_l055 import fetch
    fetch()
    larger_path=output/f'{REPRO_PRESET}-{time.time_ns()}.json'
    with threadpool_limits(1):larger=run_checkpoint(REPRO_PRESET,larger_path)
    attach_kernel_identity(larger,larger_path)
    print('Verified here: corrected local panel. Paper claim: cited. Scale-up: INCOMPARABLE to original protocols.')
else:print('Scale-up NOT_RUN in this kernel. Full paper result reproduction remains INCOMPARABLE.')''')
 nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}}),60)
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for previous,current in zip(before,after):current.outputs=previous.outputs;current.execution_count=previous.execution_count
   if 'execution_verification' in old.metadata:nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
 soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 from _foundation_config import QUIZ,PREDICT
 body=''
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   name=m[1];body+=('<div id="aggregation-viz" class="comparison-widget"><p>Interactive seed-loss control requires JavaScript. The worked table and notebook CHECK show the same operation.</p></div>' if name=='aggregation' else '')+figure(name)
   if name=='ranks' and (ROOT/'_analysis_l060_v2_results.json').exists():
    evidence=json.loads((ROOT/'_analysis_l060_v2_results.json').read_text())
    body+='<p>'+html.escape(evidence['interpretation'])+'</p>'
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 060 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l060-comparison'])+'</head><body class="l060"><article>'
 opening='<nav><a href="../index.html">Course</a> · <a href="0059-validation-set-overfitting.html">← Lesson 059</a> · <a href="0061-prior-data-fitted-networks.html">Lesson 061 →</a></nav><p class="mission-tag">Year 2 · Quarter 2 · Lesson 060</p><h1>'+TITLE+'</h1>'+launcher()+'<section id="retrieval"><h2>Retrieve before reading</h2><p>Which labels may choose a procedure? How many dataset blocks do three seeds create?</p><div id="warmup"></div></section><div id="prediction"></div>'
 cfg=dict(lesson=60,mode='checkpoint',answer=PREDICT[60][1],quiz=QUIZ[60])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-lesson','l060-comparison-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>')
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 060</a></nav><h1>Comparison protocol reference</h1>'+launcher()+figure('protocol')+markdown2html_mistune('''## The invariant evidence chain

1. Declare task versions, metric, rows, arms, seeds and candidate budgets.
2. Fit preprocessing on training rows and fit/choose using train/validation only.
3. Freeze candidate and epoch; score aligned test predictions afterward.
4. Reconstruct metrics, reject incomplete declared panels, average seeds then rank datasets.
5. Report paired raw effects with conditional intervals and costs with their timing scope.
6. Keep random/temporal panels separate and compare regimes on the same underlying tasks.

Binary log loss uses P(y=1), validates [0,1] before float64-epsilon clipping. Regression RMSE remains in the released target's units; Sberbank is log(price_doc/full_sq). Three seeds are not three datasets. Nemenyi CD uses the entire declared method pool. Small-panel asymptotic Friedman p-values are exploratory; exact conditional temporal method-label permutation is supplied.

## Five live operations

''')+markdown2html_mistune('\n'.join('- `'+task[1]+'` — '+task[3] for task in TASKS))+markdown2html_mistune('''

## Source and evidence boundary

The old 210-record run used incorrect historical TabM-mini. The v2 run uses corrected tabm_v2.py and new predictions, with full IDs, selection traces and dependency hashes. It remains a controlled numeric/one-hot small-recipe comparison. It omits CatBoost's native categorical route, TabM nonlinear embeddings, paper training budgets and nested benchmark protocols.

[Corrected evidence](../labs/_verify_l060_v2_results.json) · [Permutation and paired analysis](../labs/_analysis_l060_v2_results.json) · [Claim/source inventory](../labs/_sources_l060_v2.json) · [Exact regeneration and scale-up](../labs/l060-reproduction.md).

[TabArena v1 §§2–3,A–D](https://arxiv.org/html/2506.16791v1) · [TabReD v4 §5.4,C](https://arxiv.org/html/2406.19380v4) · [TabM v3 §3,D](https://arxiv.org/html/2410.24210v3) · [RealMLP v2 §3,A.2](https://arxiv.org/html/2407.04491v2).
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>');enrich_html(60)
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L060 package')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--keep-notebooks',action='store_true');a=p.parse_args();build_package(notebooks=not a.keep_notebooks)
