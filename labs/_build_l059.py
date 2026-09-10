"""Canonical scoped L059 v2 builder; does not modify shared delivery/gallery files."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _lesson_depth import enrich_html,enrich_notebook
from _check_l059_v2 import CHECKS
ROOT=Path(__file__).resolve().parent;SLUG='0059-validation-set-overfitting';TITLE='When validation becomes training'
TASKS=[
('choose_validation','def choose_validation(errors):','Freeze a choice using selection scores only','Return the first index of the minimum finite loss. Reject empty, nonfinite and non-vector inputs.','Deterministic ties make a replay meaningful; the API must not receive test targets.','Check shape and finiteness before choosing an index.'),
('loo_residuals','def loo_residuals(alpha, inverse_diagonal):','Recover actual deleted-row residuals','Return the paper §2.1 residual vector using the augmented inverse diagonal. Validate finite aligned nonempty vectors and strictly positive diagonal entries.','Training residuals score a model that saw the target. Deleted residuals test the fixed recipe without that row.','The Schur-complement derivation tells you which two quantities to combine; do not invert the diagonal again.'),
('select_krr','def select_krr(x, y, candidates):','Select by mean PRESS across candidates','Fit each declared configuration using krr_fit, compute mean squared live loo_residuals, choose with live choose_validation, and return (chosen_model, index, all_losses). Reject an empty list.','The final candidate search must use your deleted residuals; merely displaying a LOO example cannot validate a different training loop.','Keep model and score ordering aligned. Candidate values are regularization and eta.'),
('nested_predictions','def nested_predictions(x, y, folds, candidates):','Nest the entire selector','Return a length-n prediction array and per-fold trace. For each integer fold, call live select_krr on complement rows, predict held rows, and scatter in original order. Trace fields: fold, fit_ids, held_ids, selected, selection_mse. Validate target/fold lengths and at least two integer folds.','Excluding a target from coefficient fitting is insufficient if it already selected the hyperparameters.','Selection happens inside the loop, after slicing x and y. Original row IDs belong in the trace.'),
('paired_difference','def paired_difference(left, right):','Quantify the paired repeat-level gap','Return n, mean, sd (ddof=1), mc_se, and t95 for left−right. Reject nonfinite, non-vector, mismatched or fewer-than-two observations. Use scipy.stats.t.ppf.','A shared difficult dataset affects both protocols. Pairing retains that covariance; folds are averaged before they reach this function.','Subtract first, then compute the SD and standard error. Use n−1 degrees of freedom for the interval.')]

def piece(name,module='validation_audit_l059'):
 s=(ROOT/f'relkit/{module}.py').read_text()
 return next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)

def launcher(prepared=False):
 prefix='../' if prepared else '../labs/'
 links=[(f'https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb','Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l059_v2_results.json','Measured evidence'),(prefix+'l059-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 059 · 5 code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join(f'<a href="{u}"'+(' download' if t=='Download notebook' else '')+'>'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter. Student functions are intentionally blank.</p></aside>'

FIGURES={'mechanism':('null-minimum.png','Synthetic two-candidate enumeration. Average selected minima, not candidate expectations.'),
'results':('null-results-v2.png','Preserved historical 200-repeat null control and the exact independent-binomial expectation. No new training or paper result reproduction.'),
'architecture':('architecture-v2.png','End-to-end KRR: candidate fitting and PRESS selection above; frozen-model cross-kernel inference and sign head below. Queries supply features only.'),
'kernel':('kernel-v2.png','Paper equation instrument: ARD similarity, augmented KRR/intercept solve, deleted residuals and new-row predictions.'),
'protocol':('protocol-v2.png','Declared 64-row nested/external protocol. Same48-row outer models supply each matched fresh target. Four folds are averaged within a repetition.'),
'nested_results':('nested-results-v2.png','Fresh author runs:30 and1,000 independent synthetic repeats, with paired t95 intervals. The1,000 contain the first30. Selected LOO has a63/64 training-size difference; outer contrasts match48 rows.')}

def figure(name,notebook=False):
 f,cap=FIGURES[name];cap=prose_spacing(cap);path='figures/l059/'+f
 if notebook:return f'![{cap}](data:image/png;base64,{base64.b64encode((ROOT/path).read_bytes()).decode()})\n\n{cap} [Open full-size figure]({path}).'
 return '<figure'+(' id="protocol-overview"' if name=='protocol' else ' id="kernel-overview"' if name=='architecture' else '')+'><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable full-size computation"><img src="../labs/'+path+'" alt="'+html.escape(cap)+'"></div><figcaption>'+cap+' Scroll horizontally on narrow screens. <a href="../labs/'+path+'">Open full-size figure</a>.</figcaption></figure>'

def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())

def prose_spacing(s):
 # Only prose lines: never touch portable-image payloads or executable cells.
 return '\n'.join(line if 'data:image/' in line else re.sub(r'(?<![\w/])((?!(?:float|int|uint|complex|cawley)\d)[A-Za-z]{2,})(?=\d)',r'\1 ',line) for line in s.splitlines())

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(prose_spacing(s.strip())))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab059 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Evidence](_verify_l059_v2_results.json) · [Reproduction](l059-reproduction.md)

**Skill:** distinguish selecting a model from evaluating the complete selection procedure. Five live code TODOs, independent CHECKs and a written EXIT. The paper's KRR equations and synthetic generator are visible below; the experiments are a declared protocol extension, not a reproduced paper figure. CPU only, no external data/checkpoint downloads in the core experiment. The1,000-repeat precision track can take several minutes. Retrieval → input → implementation → measured evidence → teach-back. Author-reference figures are not outputs of this kernel.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:md(figure(m[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
Your functions drive both the preserved null replay and the fresh nested experiment. The provided kernel instrument is a direct implementation of the equations, not an imported substitute for the exercises. No preprocessing is refitted inside LOO; doing so would change the deletion identity. Read the source from input to prediction before solving the five small operations.''')
 code('''# PROVIDED — setup and versions
import sys,json,hashlib,inspect,functools,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.stats import t,binom
from threadpoolctl import threadpool_limits
from IPython.display import display
for p in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (p/'relkit').is_dir():ROOT=p.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run the Colab bootstrap or start in the course repository')
print({name:importlib.metadata.version(name) for name in ['numpy','scipy']})''')
 for name,title in [('mixture','Paper§3.1 exact Gaussian mixture'),('kernel','Paper§4 ARD kernel'),('krr_fit','PaperEq3 augmented system, including b'),('predict_krr','New-row prediction without query labels'),('candidate_grid','Frozen27-candidate finite grid')]:
  md('### PROVIDED · '+title)
  code('# PROVIDED — '+title+'\n'+piece(name))
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Implement this live protocol operation")'))
  code(CHECKS[name])
 md('''## Replay the preserved negative control with your selector
The historical stored seed is a repetition index. Replay uses RNG seed59+index. Each budget reruns the same prefix stream, so changing the budget leaves earlier candidates unchanged. Test predictions exist only after the selected index is frozen. The historical null operator is shown below exactly as measured; it calls your choose_validation.''')
 code('# PROVIDED — historical measured operator, unchanged\n'+piece('null_search','benchmark_core'))
 code('# PROVIDED — exact expectation and live paired-summary reanalysis\n'+piece('exact_null_minimum')+'\n\n'+piece('null_reanalysis'))
 code('''# PROVIDED — replay every saved repetition/budget row
historical=json.loads((ROOT/'_verify_l059_results.json').read_text())
for row in historical['rows']:
    got=null_search(row['candidates'],80,2000,59+row['seed'])
    assert all(got[k]==row[k] for k in got), 'Historical replay differs; inspect seed offset, ties and RNG streams'
null_summary=null_reanalysis(historical['rows'])
display(pd.DataFrame([{k:v for k,v in r.items() if k not in ['optimism','paired_optimism_increase']} for r in null_summary]))
print('256-minus1 paired optimism increase:',null_summary[-1]['paired_optimism_increase'])''')
 md('''## PROVIDED · The full nested experiment
Trace the two selection paths in this driver. The contaminated comparator intentionally chooses on the full64-row dataset. Both outer protocols refit coefficients on the same48-row complements. Fresh rows are generated after all indices freeze; each outer model receives the same4,096 queries. Four fold results are averaged within a repetition. Your nested_predictions function is the only internal selection path.

Predict both reported and matched fresh MSE. The external procedure may really predict better because its hyperparameter choice saw more data, while reporting an optimistic estimate. Do not assume the difference between the two reported scores equals the external bias.''')
 code('# PROVIDED — inspect the information boundary and sampling unit\n'+piece('experiment'))
 code('''# PROVIDED — count actual live calls, preserving the student implementations
names=['choose_validation','loo_residuals','select_krr','nested_predictions','paired_difference']
originals={name:globals()[name] for name in names};calls={name:0 for name in names}
def counted(name,fn):
    @functools.wraps(fn)
    def wrapper(*args,**kwargs):
        calls[name]+=1
        return fn(*args,**kwargs)
    return wrapper
for name,fn in originals.items():globals()[name]=counted(name,fn)
try:
    with threadpool_limits(1):trial=experiment(repetitions=30)
finally:
    for name,fn in originals.items():globals()[name]=fn
assert all(v>0 for v in calls.values()), 'Every TODO must drive the actual experiment'
print('Live experiment calls:',calls)
display(pd.DataFrame(trial['summary']).T)
display(pd.Series(trial['means'],name='Mean'))
author=json.loads((ROOT/'_verify_l059_v2_results.json').read_text())
for key in trial['means']:
    np.testing.assert_allclose(trial['means'][key],author['means'][key],rtol=1e-8,atol=1e-10)
print('Numerical agreement with declared author protocol; original paper results remain INCOMPARABLE')''')
 md('''## Diagnose a disagreement without changing the target
Find a repetition where a noisy estimate reverses the mean tendency; report its seed and both compared quantities alongside the full aggregate. A single counterexample is compatible with an expectation claim. Explain why the internal-label mutation CHECK is a correctness test while an interval excluding zero is an experimental outcome.

Do not compute a t interval over120 outer folds: their training sets overlap. The thirty differences are the units. The selected-LOO gap additionally compares63-row deleted fits with a64-row final fit; it is not a pure estimate of selection bias.''')
 code('''# PROVIDED — preserve per-repeat diagnostics and traceable failure cases
rows=pd.DataFrame(trial['records'])
display(rows[['seed','internal_mse','external_mse','internal_fresh_mse','external_fresh_mse']].head())
print('Internal-minus-external outer score is negative in',int((rows.internal_mse<rows.external_mse).sum()),'of30 repeats')
output=Path('data/cache/l059-student');output.mkdir(parents=True,exist_ok=True)
(output/'analysis.json').write_text(json.dumps(trial,indent=2,allow_nan=False))''')
 md('''## Required final track · 1,000 independent synthetic repetitions
The first30 repeats are retained, so this is a precision extension of the same experiment, not an independent replication. Commit a prediction about interval width and the sign of each mean gap before running. A matching repetition count does not recreate Figure2's iterative optimizer/four-fold inner score or Table8's thirteen real datasets. The grid and all decision rules remain fixed. No paid/cloud execution is needed.''')
 code('''# PROVIDED — run the same visible functions for the declared precision track
with threadpool_limits(1):closer=experiment(repetitions=1000)
assert closer['records'][:30]==trial['records'], 'Nested repetition prefix changed'
display(pd.DataFrame(closer['summary']).T)
(output/'closer.json').write_text(json.dumps(closer,indent=2,allow_nan=False))
print('1,000-repeat precision track RUN; paper figures/tables remain INCOMPARABLE')''')
 md('''## EXIT · a frozen decision and a measured argument
Write80–180 words identifying the selected object, information boundary, training sizes, one measured interval with its unit, and a falsifiable next experiment. Fill a lesson060 decision ledger: metric, candidate procedures, split rule/identity, tuning budget, selection rule, freeze point, and a postponed adjustment. Submit the artifact and completed notebook; the tutor judges reasoning, not the word count. This saves live source text and evidence identities; it makes no cross-process resume promise.''')
 verdict='The selected object is a kernel recipe chosen by minimum mean PRESS. Coefficient-level out-of-fold prediction is insufficient when the entire dataset already chose the hyperparameters. Internal outer predictions rerun selection on48 rows and refit on48; matched fresh evaluation uses the same models. The selected LOO comparison additionally changes training size from63 to64. The thirty-repeat external optimism is about.0693 with a paired t95 interval about[.0220,.1165], whose units are independent synthetic dataset draws, not overlapping folds or real tasks. I would freeze a smaller candidate grid using development evidence and evaluate it on a new synthetic seed block; if matched external optimism persists unchanged, fewer candidates alone did not solve the problem.' if solution else ''
 ledger={'metric':'binary log loss','candidate_procedures':['CatBoost fixed','RealMLP fixed'],'split_identity':'declare temporal cutoff and entity exclusion before data access','tuning_budget':'one fixed recipe per family','selection_rule':'inner mean log loss; first tie','freeze_point':'before viewing outer outcomes','postponed_adjustment':'feature-history window redesign after test inspection'} if solution else {}
 code('# TODO — written interpretation and adaptation ledger\nverdict='+repr(verdict)+'\nledger='+repr(ledger)+'''
assert len(verdict.split())>=80, 'Write the reasoning before submitting; code checks cannot grade it'
required={'metric','candidate_procedures','split_identity','tuning_budget','selection_rule','freeze_point','postponed_adjustment'}
assert set(ledger)==required and all(ledger.values()), 'Record every declared decision'
functions=['mixture','kernel','krr_fit','predict_krr','candidate_grid','choose_validation','loo_residuals','select_krr','nested_predictions','paired_difference','experiment','null_search','null_reanalysis','exact_null_minimum']
live_sources={name:inspect.getsource(globals()[name]) for name in functions}
artifact={'lesson':59,'analysis':trial,'precision':closer,'historical_null':null_summary,'calls':calls,
 'live_sources':live_sources,'verdict':verdict,'ledger':ledger,'paper_reproduction':'INCOMPARABLE',
 'source_manifest':json.loads((ROOT/'_sources_l059_v2.json').read_text()),
 'author_evidence_sha256':hashlib.sha256((ROOT/'_verify_l059_v2_results.json').read_bytes()).hexdigest()}
(output/'exit.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
print('Saved data/cache/l059-student/exit.json; submit this and the completed notebook.')''')
 nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}}),59)
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb');nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
 next(x for x in soup.find_all('h2') if x.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
 soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 from _foundation_config import QUIZ,PREDICT
 body=''
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   name=m[1];body+=('<div id="selection-viz" data-foundation-viz="selection"></div>' if name=='mechanism' else '<div id="nested-boundary" class="validation-widget"><p>Interactive label mutation requires JavaScript. The protocol table below and notebook CHECK explain the same boundary.</p></div>' if name=='protocol' else '')+figure(name)
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson059 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','foundation-course','lab-access','l059-validation'])+'</head><body class="l059"><article>'
 opening=f'<nav><a href="../index.html">Course</a> · <a href="0058-surveys-meta-benchmarks.html">← Lesson058</a> · <a href="0060-broad-model-comparison.html">Lesson060 →</a></nav><p class="mission-tag">Year2 · Quarter2 · Lesson059</p><h1>{TITLE}</h1>{launcher()}<section id="retrieval"><h2>Retrieve before reading</h2><p>Explain why OOF base predictions do not automatically protect a learned combiner. Name what must remain independent of its selection.</p><div id="warmup"></div></section><div id="prediction"></div>'
 cfg=dict(lesson=59,mode='selection',answer=PREDICT[59][1],quiz=QUIZ[59],selection=json.loads((ROOT/'_verify_l059_results.json').read_text())['summary'])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-viz','foundation-lesson','l059-validation-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>')
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson059</a></nav><h1>Selection and evaluation boundary</h1>'+launcher()+figure('protocol')+markdown2html_mistune('''## Operational reference

| Decision | Eligible labels | Evaluation target |
|---|---|---|
| Coefficient fitting | Declared fit rows | Fixed recipe |
| Inner LOO selection | Outer-development rows | Minimum PRESS/n; first tie |
| Outer evaluation | Held rows after choice freezes | Whole selection procedure at48-row fit size |
| Fresh matched evaluation | New rows after all choices | Same fitted outer models |
| Final refit | All64 development rows | Separate larger-sample procedure |

KRR Eq3 includes the unpenalized intercept and sum(α)=0. LOO uses the augmented inverse diagonal. Arbitrarily refitting preprocessing per deleted row would change that identity. ARD reduces to isotropic RBF when coordinate scales agree.

Thirty and1,000 synthetic dataset repetitions use the same27-candidate grid and four outer folds. The longer run retains the first30. Use paired repeat-level differences; do not treat four overlapping outer folds as independent samples. The selected-LOO gap has a63/64 training-size difference; outer optimism contrasts match48-row coefficient fits.

Original paper Figure2/fourfold inner iterative search and Table8/thirteen-task RBF benchmark are INCOMPARABLE. The old pure-noise result is preserved separately. Source-equation checks are not official GKM software parity.

[Primary full paper §§2–6](https://jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) · [Source inventory](../labs/_sources_l059_v2.json) · [Precision evidence](../labs/_verify_l059_closer_results.json)
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>')
 enrich_html(59)
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L059 package')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--keep-notebooks',action='store_true');a=p.parse_args();build_package(notebooks=not a.keep_notebooks)
