"""Canonical scoped L058 package builder. Never regenerates shared galleries."""
import ast,base64,hashlib,html,json,os,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _lesson_depth import enrich_html,enrich_notebook,authored
from _check_l058_v2 import CHECKS
ROOT=Path(__file__).resolve().parent;SLUG='0058-surveys-meta-benchmarks';TITLE='Read a benchmark without inheriting its bias'
TASKS=[
('parse_talent','def parse_talent(path):','Parse evidence without inventing observations','Return two aligned DataFrames (means, reported SDs). Preserve original row labels, reject duplicate IDs/headers and malformed nonmissing values. Recognize empty, dash, em dash, NA, N/A and nan+nan as missing. Strip Markdown emphasis and accept scientific notation and + or ± separators.','A numeric regular expression must distinguish an exponent sign from the separator; validate the whole cell. Missing means remain NaN.'),
('rank_matrix','def rank_matrix(scores, higher_is_better):','Orient and rank a complete panel','Return [datasets, methods] average-tie ranks starting at one, lower rank meaning better. Reject nonfinite, empty or non-2D inputs and require an explicit Boolean direction. This function must rank only the supplied columns.','scipy.stats.rankdata can handle ties; its axis must correspond to the model dimension. A monotone unit conversion should preserve ranks.'),
('dataset_bootstrap','def dataset_bootstrap(matrix, draws=2000, seed=58):','Recover paired task sensitivity','Return [2, methods] percentile bounds at .025 and .975 by averaging resampled whole dataset rows. Use a local default_rng(seed), validate finite nonempty 2D input and positive integer draws.','One draw uses a shared list of dataset indices for all columns. Do not generate or resample training seeds from reported SDs.'),
('select_tiny','def select_tiny(seen_ranks, size, trials=1000, seed=58):','Optimize the stated rank-fidelity objective','Propose trials subsets of size unique rows using default_rng(seed).choice(..., replace=False); sort their indices. Minimize MAE between subset and full-column means, keeping the first minimum. Return indices, first_indices, seen_mae and best_so_far (one incumbent loss per proposal). Validate finite nonempty 2D input and positive integer size/trials; size cannot exceed row count.','Track an incumbent solution. The first proposal provides a matched unselected baseline; a larger nested proposal budget cannot worsen its seen objective.'),
('method_holdout','def method_holdout(scores, higher_is_better, seen, size, trials=1000, seed=58):','Protect held-out method columns','Validate a complete score matrix and a unique nonempty integer seen-index list with a nonempty complement. Rank seen columns by themselves, call select_tiny, then rank unseen columns independently. Return seen, unseen, selection, and random/selected dictionaries each containing seen_mae and unseen_mae.','Only frozen selected task IDs may cross from selection to evaluation. Reusing all-model ranks after hiding columns would leak the held-out methods into selection.')]

def piece(name):
 s=(ROOT/'relkit/talent_audit_l058.py').read_text()
 return next(ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)

def launcher(prepared=False):
 prefix='../' if prepared else '../labs/'
 colab=f'https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb'
 links=[(colab,'Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l058_v2_results.json','Measured evidence'),(prefix+'l058-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 058 · 5 code tasks + EXIT</strong></p><nav class="lab-access-links">'+''.join(f'<a href="{url}"'+(' download' if text=='Download notebook' else '')+'>'+text+'</a>' for url,text in links)+'</nav><p>This preview is read-only. Complete the runnable notebook in Colab or Jupyter; the student functions are intentionally blank.</p></aside>'

def figure(name,notebook=False):
 captions={'mechanism':'Synthetic rank example: omitting fixed task blocks changes the mean. The interactive lesson shows all three task states.',
 'protocol':'Synthetic six-task computation: seen and held-out columns are ranked independently. A second candidate improves seen MAE from .50 to .00 and worsens held-out MAE from .00 to .50.',
 'tiny':'Fresh author-reference reanalysis of frozen scores. Each line pairs one random proposal with its seen-optimized subset; 15 overlapping rotation/search cases per task type. These are not independent model fits.',
 'results':'Fresh author-reference six-method mean ranks and 95% dataset percentile intervals (2,000 paired draws). These are not training-seed intervals or student-kernel outputs.'}
 file='rank-intervals' if name=='results' else name
 path=f'figures/l058/{file}.png';caption=captions[name]
 if notebook:return f'![{caption}](data:image/png;base64,{base64.b64encode((ROOT/path).read_bytes()).decode()})\n\n{caption} [Open full-size figure]({path}).'
 identity=' id="protocol-overview"' if name=='protocol' else ''
 return f'<figure{identity}><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable full-resolution figure"><img src="../labs/{path}" alt="{html.escape(caption)}"></div><figcaption>{caption} Scroll horizontally on narrow screens. <a href="../labs/{path}">Open full-size figure</a>.</figcaption></figure>'

def manuscript_parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 058 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Evidence]( _verify_l058_v2_results.json) · [Reproduction contract](l058-reproduction.md)

**Skill:** implement a benchmark reader and a tiny-benchmark selector with an enforceable information boundary. Five TODOs feed the real frozen-table experiment. PROVIDED = readable support, CHECK = diagnostic feedback, EXIT = code artifact plus written argument. No predictive model, raw training-data download or pretrained checkpoint is needed.

**Evidence:** real-dataset published results (Tier A evidence), 300 tasks in a six-model panel; 276 complete tasks in a second paper-like 13/12-method panel. Synthetic fixtures isolate arithmetic only. Source tables and hashes are committed. Default CPU audit uses 1,000 subset proposals; the required final track uses 10,000. Core reading 60–90 minutes; lab 60–90 minutes plus roughly tens of seconds of CPU analysis. Live Colab is unverified.

**Retrieve first:** explain why withholding model columns can play the same role that withholding test labels played in lesson 057. Write your prediction about seen and unseen rank fidelity before inspecting results.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in manuscript_parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:md(figure(m[1],True))
  elif part.strip():md(part)
 md('''## Start the live implementation
The five functions below are the evaluator, not disconnected exercises. The final provided audit calls load_tables → your parser; then your ranks/bootstrap; finally your method_holdout → your selector. Keep all definitions in this kernel. The source module is a teacher/reference implementation; it is not imported as a substitute for your TODOs.''')
 code('''# PROVIDED — setup and runtime versions
import os,sys,re,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from IPython.display import display
for p in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (p/'sources/l058').is_dir():ROOT=p.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run the bootstrap or start within the relational repository')
print({name:importlib.metadata.version(name) for name in ['numpy','pandas','scipy']})''')
 for i,(name,sig,title,goal,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** '+{'parse_talent':'A silent typo or duplicated task changes the denominator before any model is ranked.','rank_matrix':'Ranking raw losses in the wrong direction can invert the entire conclusion.','dataset_bootstrap':'Shared rows preserve pairing; separate column resampling invents a different experiment.','select_tiny':'This is the paper\'s subset-search mechanism, with an explicit objective and a reproducible baseline.','method_holdout':'A new method must not help choose the benchmark that will evaluate it.'}[name]+f'\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Complete this live benchmark operation")'))
  code(CHECKS[name])
 md('''## PROVIDED · Frozen data and the full audit loop
Read the loop before running. The six-method panel ranks all 300 rows. The tiny panel additionally uses historical TabPFN and therefore has four binary and twenty multiclass missing rows. Coverage is fixed to the common complete panel before the method rotations; this conditions the experiment on availability, an explicit limit even though scores remain hidden. We do not infer that these omissions are random.

The paper's family list is retained with the LR alias. Five fixed cyclic holdouts reserve three classification or two regression methods; some held-out memberships overlap. Each search seed uses the same candidate stream for a nested 1,000/10,000 comparison. For regression, two of the twelve methods are never held out in these five rotations; this is a declared sensitivity design, not exhaustive cross-validation. No individual model seed is recovered from the Markdown SDs.''')
 from relkit.talent_audit_l058 import ARMS,PAPER_POOL,FILES
 code('# PROVIDED — named pools and exact raw files\nARMS='+repr(ARMS)+'\nPAPER_POOL='+repr(PAPER_POOL)+'\nFILES='+repr(FILES))
 code('# PROVIDED — source hashes, parser invocation and aliases\n'+piece('load_tables'))
 code('# PROVIDED — live audit driver\n'+piece('audit'))
 md('''## Run the 300-task and 276-task analyses
Predict the smallest full-panel rank, whether CatBoost-minus-RealMLP's paired interval excludes zero, and whether every optimized subset improves held-out fidelity. Then run. The comparisons with author values check deterministic arithmetic on identical published means; they do not mean you trained the benchmark methods.''')
 code('''# PROVIDED — invoke your live functions; count calls without replacing them
call_counts={name:0 for name in ['parse_talent','rank_matrix','dataset_bootstrap','select_tiny','method_holdout']}
import functools
originals={name:globals()[name] for name in call_counts}
def counted(name,fn):
    @functools.wraps(fn)
    def wrapper(*args,**kwargs):
        call_counts[name]+=1
        return fn(*args,**kwargs)
    return wrapper
for name,fn in originals.items():globals()[name]=counted(name,fn)
try:trial=audit(ROOT,1000)
finally:
    for name,fn in originals.items():globals()[name]=fn
assert all(v>0 for v in call_counts.values()), 'Every TODO must drive the actual experiment'
print('Live calls:',call_counts)
display(pd.DataFrame(trial['coverage']).T)
display(pd.DataFrame(trial['tiny_coverage']).T[['released','complete','size','excluded']])
display(pd.DataFrame(trial['per_task_ranks']).assign(all_tasks=pd.Series(trial['mean_ranks'])))
display(pd.DataFrame(trial['paired_gaps']).T)
author=json.loads((ROOT/'_verify_l058_v2_results.json').read_text())
for key in ['coverage','tiny_coverage','mean_ranks','per_task_ranks','paired_gaps','tiny']:
    assert trial[key]==author[key], 'Author arithmetic differs at '+key+'; inspect protocol and live implementation'
print('PASS: exact same-release reanalysis, not paper training reproduction')''')
 code('''# PROVIDED — readable paired subset outcomes; preserve all cases
records=[]
for row in trial['tiny']:
    records.append(dict(task=row['task'],fold=row['fold'],seed=row['seed'],
       seen_change=row['selected']['seen_mae']-row['random']['seen_mae'],
       unseen_change=row['selected']['unseen_mae']-row['random']['unseen_mae']))
comparison=pd.DataFrame(records)
display(comparison.groupby('task')[['seen_change','unseen_change']].agg(['mean','min','max']))
print('Unseen fidelity worsens in',int((comparison.unseen_change>1e-12).sum()),'of',len(comparison),'overlapping cases (tolerance 1e-12)')
# Seen and unseen pools have different rank ranges: compare changes within each.
assert (comparison.seen_change<=1e-12).all()
Path('data/cache/l058-student').mkdir(parents=True,exist_ok=True)
Path('data/cache/l058-student/analysis.json').write_text(json.dumps(trial,indent=2,allow_nan=False))''')
 md('''## Interpretation and paper-to-code checkpoint
Explain why the lower seen MAE is guaranteed for this baseline but lower unseen MAE is not. Identify one failure case by its declared task/fold/seed without concealing the other cases. Explain the four/20 missing-task exclusions, why two rank-pool MAEs cannot be directly compared as a generalization gap, and why a paired percentile interval is not a training-seed interval.

| Paper / question | Visible computation | Checked limit |
|---|---|---|
| §4–5 released scores | parse_talent, rank_matrix | Rounded means; separate SDs; six-method pool |
| Dataset sensitivity extension | dataset_bootstrap on paired rank gaps | Published means conditional; not paper significance tests |
| §8.2 random subset search | select_tiny | Explicit mean-rank MAE, first-minimum ties |
| Evaluation on new methods | method_holdout | Re-rank seen pool before selection; unseen-score mutation CHECK |
| Table 7 | audit | 276 complete historical tasks, declared new rotations; INCOMPARABLE |

The upstream average-rank function was independently checked on complete numeric fixtures and real rows with the correct input orientation. It reads spreadsheets and mean-imputes missing ranks; those choices are not silently inherited. Exact selector parity is not claimed because a corresponding tiny-selector implementation was not located in the inspected pinned repository tree.''')
 md('''## Required final track · increase search to 10,000 candidates
This is affordable CPU work, so no cloud or checkpoint gate is needed. Predict the direction of seen and unseen changes before executing. The same functions and first 1,000 proposals are reused. The authors' random search also uses 10,000 candidates, but matching this budget does not restore their dataset revision, method splits or original selector. This run remains INCOMPARABLE to Table 7. It is a required fidelity improvement over the 1,000-candidate exercise, not optional extra reading.''')
 code('''# PROVIDED — run your live selector at the paper's stated random-search count
closer=audit(ROOT,10000)
assert all(b['selected']['seen_mae']<=a['selected']['seen_mae']+1e-12 for a,b in zip(trial['tiny'],closer['tiny']))
change=pd.DataFrame([dict(task=a['task'],fold=a['fold'],seed=a['seed'],
    seen=b['selected']['seen_mae']-a['selected']['seen_mae'],
    unseen=b['selected']['unseen_mae']-a['selected']['unseen_mae']) for a,b in zip(trial['tiny'],closer['tiny'])])
display(change.groupby('task')[['seen','unseen']].agg(['mean','min','max']))
Path('data/cache/l058-student/closer.json').write_text(json.dumps(closer,indent=2,allow_nan=False))
print('10,000-candidate track RUN; original TALENT Table 7 still INCOMPARABLE')''')
 md('''## EXIT · an evidence card and next experiment
Write a 60–150 word argument covering your source revision, two complete-panel denominators, pool/direction/ties, resampling unit, one actual unexpected case, and one falsifiable next experiment for lesson 060. Name a tree and neural baseline procedure with the deployment split. This text is judged by the tutor, not by word count. Attach your family map and completed notebook. Code completion alone does not establish mastery.''')
 verdict='This audit reconstructs the frozen historical six-method ranking on 300 tasks and examines 276 complete tasks for subset selection. Published means provide neither paired model-seed observations nor original data identity. The selector sees only re-ranked seen-method columns. Increasing proposals cannot worsen its seen objective, but held-out fidelity must be checked separately. I would retain CatBoost and RealMLP as candidate procedures and compare them on a prespecified temporal split with matched budgets before adding relational information. The original TALENT Table 7 remains incomparable.' if solution else ''
 code("# TODO — written evidence interpretation\nverdict="+repr(verdict)+'''
assert len(verdict.split())>=60, 'Write the evidence argument before submitting; the tutor grades its reasoning'
# Audit outputs and live definitions are saved; this is not a resume promise.
import inspect
live_sources={name:inspect.getsource(fn) for name,fn in originals.items()}
exit_artifact={'lesson':58,'analysis':trial,'closer':closer,'calls':call_counts,
 'live_todo_sources':live_sources,'verdict':verdict,'paper_reproduction':'INCOMPARABLE',
 'source_manifest':json.loads((ROOT/'_sources_l058.json').read_text())}
Path('data/cache/l058-student/exit.json').write_text(json.dumps(exit_artifact,indent=2,allow_nan=False))
print('Saved data/cache/l058-student/exit.json; submit it with your notebook and family map.')''')
 nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python','version':'3.12'}}),58)
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True);nbf.write(nb,path)
 return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
 first=next(x for x in soup.find_all('h2') if x.get_text().startswith('TODO 1'))
 first.insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'))
 soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'))
 (ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 from _foundation_config import QUIZ,PREDICT
 body=''
 for part in manuscript_parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:
   name=m[1];body+=('<div id="ranks-viz" data-foundation-viz="ranks"></div>' if name=='mechanism' else '<div id="tiny-selector" class="foundation-exhibit"></div>' if name=='protocol' else '')+figure(name)
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 058 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l058-benchmark'])+'</head><body class="l058"><article>'
 opening=f'<nav><a href="../index.html">Course</a> · <a href="0057-cross-family-ensembling.html">← Lesson 057</a> · <a href="0059-validation-set-overfitting.html">Lesson 059 →</a></nav><p class="mission-tag">Year 2 · Quarter 2 · Lesson 058</p><h1>{TITLE}</h1>{launcher()}<section id="retrieval"><h2>Retrieve before reading</h2><p>Why can withholding a model column protect an evaluation? Compare to withholding an outer-test target in lesson 057. Write a prediction before viewing the tiny-subset result.</p><div id="warmup"></div></section><div id="prediction"></div>'
 closing='<section id="lab"><h2>Run the companion lab</h2><p>Five live TODOs: parse mean/SD tables, orient and rank, bootstrap paired tasks, select a tiny subset, and enforce a held-out-method boundary. Complete the 10,000-candidate CPU track and submit your EXIT artifact.</p>'+launcher()+'<div id="teachback"></div></section>'
 cfg=dict(lesson=58,mode='ranks',answer=PREDICT[58][1],quiz=QUIZ[58])
 scripts='<script id="foundation-config" type="application/json">'+json.dumps(cfg)+'</script>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','foundation-viz','foundation-lesson','l058-benchmark-viz'])
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(head+opening+body+closing+'</article>'+scripts+'</body></html>')
 ref='<nav><a href="../lessons/'+SLUG+'.html">Lesson 058</a> · <a href="../labs/html/'+SLUG+'.html">Read-only lab preview</a></nav><h1>Benchmark evidence card</h1>'+launcher()+'<p>Evidence source → declared complete panel → metric direction → within-pool ranks → dataset aggregation → paired sensitivity → frozen decision → held-out-method evaluation.</p>'+figure('protocol')
 ref+=markdown2html_mistune('''## Operational reference

1. Preserve source revision, raw-file hashes, raw row labels and explicit model aliases. A hash verifies bytes, not the underlying dataset protocol.
2. Parse means and SDs separately. Reject malformed nonmissing cells; never replace missing results with zero.
3. Orient accuracy upward / RMSE downward, then assign average-tie ranks within the declared pool. Mean ranks give each retained dataset one vote.
4. Bootstrap paired rank gaps across whole datasets. The interval is percentile-based, conditional on published means; no raw-seed covariance is available.
5. Reserve method columns BEFORE ranking. Minimize mean-rank MAE over seen methods; evaluate frozen selected IDs using an independently ranked unseen pool.
6. Compare random and selected subsets within the same pool; pool sizes set rank scales. Method rotations and repeated search seeds are overlapping sensitivity cases.

Six-model panel: 300 tasks, 120/80/100 by type. Paper-like tiny panel: 276 complete tasks, 116/60/100; select17/9/15, totaling41. Original TALENT tiny benchmark:45; exact Table7 not regenerated. Neither observed rank nor survey taxonomy establishes a universal model family winner.

## Source reading route

[Borisov v3 §§III–IV and VII–VIII](https://arxiv.org/html/2110.01889v3) for taxonomy and its own five-task experiment; [TALENT v3 §§4–5 and8](https://arxiv.org/html/2407.00956v3) for protocol and two distinct tiny benchmarks; [2024 survey §§2–3](https://arxiv.org/html/2410.12034v1) for a further map, followed by the primary papers it cites.
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(head+ref+'</article></body></html>')
 enrich_html(58)
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L058 package')
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--keep-notebooks',action='store_true');args=p.parse_args();build_package(notebooks=not args.keep_notebooks)
