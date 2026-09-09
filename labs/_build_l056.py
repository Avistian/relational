"""Build the portable L056 audit lab; evaluator source remains visible and live."""
import ast, base64, os
from pathlib import Path
from urllib.parse import urlsplit
import nbformat as nbf
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_l049 import mdtext

ROOT=Path(__file__).resolve().parent
SLUG='0056-tabarena-benchmark-literacy'

def piece(file,name):
    source=(ROOT/file).read_text()
    return next(ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)

def build(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
    def figure(name,caption):
        md('!['+caption+'](data:image/png;base64,'+base64.b64encode((ROOT/'figures/l056'/f'{name}.png').read_bytes()).decode()+')\n\n'+caption)
    soup=BeautifulSoup((ROOT.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')
    def section(name):
        el=BeautifulSoup(str(soup.find('section',id=name)),'html.parser').find('section')
        for node in el.find_all(['figure','script']):node.decompose()
        for node in el.find_all('div',id=True):node.decompose()
        for a in el.find_all('a',href=True):
            u=urlsplit(a['href'])
            if not u.scheme and u.path:a['href']=os.path.relpath((ROOT.parent/'lessons'/u.path).resolve(),ROOT)+('#'+u.fragment if u.fragment else '')
        md(mdtext(el))
    def task(name,signature,text):
        md(text)
        code('# TODO — '+name+'\n'+(piece('relkit/leaderboard.py',name) if solution else signature+'\n    raise NotImplementedError("Implement this audit operation")'))
    md('''# Lab 056 · TabArena: living benchmark literacy

**Skill:** audit a benchmark claim by reconstructing its aggregation from individual released measurements. **Implementation scope:** evaluator key parts, no new model or training. Published-result reanalysis is not training reproduction.

[Lesson](../lessons/0056-tabarena-benchmark-literacy.html) · [Reference](../reference/tabarena-benchmark-audit.html) · [Contract](l056-reproduction.md)

**Before input:** explain from memory why random versus temporal splitting changes the target question, what validation may select, and why more folds are not more datasets.

**Contract:** four frozen `tabarena-2025-06-12` method artifacts, 51 real datasets, three regimes, 816 outer splits; 9,792 rows. We analyze published measurements from real datasets (Tier A evidence), not raw data or model weights. The two-dataset weighting example is synthetic mechanism isolation only. Artifacts are committed (~254 KB), SHA256-verified, and runnable offline after setup. Python dependencies follow `requirements-labs.txt`; runtime versions print below. Bootstrap seed=56, 2,000 dataset draws. No new model/split seeds are introduced; published folds are retained. Exact paper Figure 1: **INCOMPARABLE**.

PROVIDED = read/run; TODO = implement; CHECK = immediate feedback; EXIT = evidence plus your interpretation. Four TODO functions drive the actual all-dataset audit. Budget: 35–50 minutes learning, seconds of CPU computation; no GPU required.''')
    for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    md('''## Concept recap

A leaderboard entry is a fitted procedure: a family plus preprocessing, validation, tuning and possibly several levels of ensembling. `default` may still include eight-fold bagging. Selecting hyperparameters and weighting ensembles must use inner validation; outer test errors evaluate the selected procedure.

Align dataset, split, method and regime before comparing. Published `metric_error` is already lower-is-better: 1−AUC, log loss or RMSE. Errors on unrelated targets cannot be averaged in their original units. Rank within one dataset/split instead, then average splits within a dataset and datasets equally.

Worked micro-example: errors `[.4,.2,.2,.9]` yield ranks `[3,1.5,1.5,4]`. If one dataset ranks A/B as 1/2 once and another ranks them 2/1 three times, pooling splits gives A=1.75/B=1.25; equal dataset weighting gives 1.5/1.5. Repeating a task changed its influence, not its evidence population.

For uncertainty, compute a paired method difference for each dataset. Resample entire datasets, preserving the paired methods and all splits. Percentiles of repeated bootstrap means describe sensitivity to that dataset mix. They do not estimate future temporal drift; crossing zero does not prove equivalence.''')
    code('''# PROVIDED — setup, versions and location
import os,sys,json,hashlib,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
for p in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (p/'relkit').is_dir():
        ROOT=p.resolve();os.chdir(ROOT);sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start in the relational workspace')
ARMS=['CatBoost','LightGBM','RealMLP','TabM']
print({p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','pyarrow']})''')
    section('scope');section('procedure');figure('protocol','Illustrative inner/outer split boundary. Trace V rows into validation; outer-test labels remain withheld.')
    code('# PROVIDED — exact artifact hash checks and loading\n'+piece('_verify_l056.py','load_snapshot')+'\nrows=load_snapshot()\nprint(rows.shape)\ndisplay(rows.groupby(["arm","method_subtype"]).size().unstack())')
    task('aligned_errors','def aligned_errors(rows, arms):','''## TODO 1 · Refuse an invalid comparison
**Goal:** return an error matrix indexed by `(dataset,fold)` with columns in `arms` order. Select the requested arms, require unique nonempty arm names, reject duplicate keys, missing/nonfinite errors, any imputed row and conflicting metrics within a dataset. Require complete coverage across arms. Sort the index.

**Why:** silent deletion of a failed model or unmatched split can change the question. **Hint boundary:** validate the rows before reshaping; absent arm/split cells must raise `ValueError`, not disappear. Do not select by test performance.''')
    code('''# CHECK — coverage, failure and metric interventions
base=rows.loc[rows.method_subtype=='default'].copy()
wide=aligned_errors(base,ARMS)
assert wide.shape==(816,4)
assert set(wide.groupby(level='dataset').size())=={9,30}
for bad in (base.iloc[1:],pd.concat([base,base.iloc[:1]]),base.assign(imputed=True)):
    try:aligned_errors(bad,ARMS)
    except ValueError:pass
    else:raise AssertionError('Corrupted evaluation grid accepted')
mixed=base.copy();mixed.loc[mixed.index[0],'metric']='WRONG'
try:aligned_errors(mixed,ARMS)
except ValueError:pass
else:raise AssertionError('Conflicting metrics accepted')
print('PASS: 51 datasets, 816 splits; omissions, duplicates and metric/imputation corruption rejected')''')
    section('aggregation');figure('weighting','Synthetic fixed-rank example: three repetitions change pooled-split weights; equal dataset averages remain tied.')
    task('rank_errors','def rank_errors(errors):','''## TODO 2 · Rank errors, preserving ties
**Goal:** assign lower errors better ranks starting at 1. Ties receive the mean of their occupied ranks. Reject non-vector, empty and nonfinite inputs.

**Why:** input order must not reward one model when errors tie. **Hint boundary:** use comparisons/counts or a stable sorting procedure; do not call a ranking library inside your implementation.''')
    code('''# CHECK — independent reference and invariance to units
from scipy.stats import rankdata
np.testing.assert_array_equal(rank_errors([3,1,1,2]),[4,1.5,1.5,3])
for sample in wide.to_numpy()[::19]:
    np.testing.assert_array_equal(rank_errors(sample),rankdata(sample,method='average'))
    np.testing.assert_array_equal(rank_errors(sample),rank_errors(7*sample+12))
for sample in ([],[1,np.nan],[1,np.inf]):
    try:rank_errors(sample)
    except ValueError:pass
    else:raise AssertionError('Invalid rank input accepted')
print('PASS: ties and monotone unit changes preserve correct ranks')''')
    task('macro_ranks','def macro_ranks(split_ranks):','''## TODO 3 · Make each dataset one unit
**Goal:** return one mean-rank row per dataset from a DataFrame indexed by `(dataset,fold)`. Sort datasets; preserve method columns. Reject empty/nonfinite inputs.

**Why:** 30 outer splits should not give a small task 30/9 times another task’s influence. **Hint boundary:** aggregate within dataset first; the caller will average the returned dataset rows.''')
    code('''# CHECK — duplicate the whole split pattern of one task
split_ranks=pd.DataFrame([rank_errors(v) for v in wide.to_numpy()],index=wide.index,columns=ARMS)
balanced=macro_ranks(split_ranks)
name=balanced.index[0]
extra=split_ranks.loc[[name]].copy()
extra.index=pd.MultiIndex.from_arrays([extra.index.get_level_values('dataset'),extra.index.get_level_values('fold')+100],names=['dataset','fold'])
doubled=pd.concat([split_ranks,extra])
np.testing.assert_allclose(macro_ranks(doubled),balanced)
assert not np.allclose(doubled.mean(),split_ranks.mean())
display(pd.DataFrame({'equal_datasets':balanced.mean(),'pooled_splits':split_ranks.mean()}))
print('PASS: repeating a dataset pattern changes only the pooled summary')''')
    section('elo')
    code('''# PROVIDED — trace the Elo mapping, not an Elo fitting routine
gap=400
odds=10**(gap/400);probability=odds/(1+odds)
assert np.isclose(probability,10/11)
print('Expected benchmark win probability:',probability,'; not classification accuracy')''')
    task('bootstrap_gap','def bootstrap_gap(dataset_gaps, seed=56, n_boot=2000):','''## TODO 4 · Resample paired datasets
**Goal:** return `[mean_gap, lower_95, upper_95]` using the empirical 2.5/97.5 percentiles of `n_boot` bootstrap means. Use `np.random.default_rng(seed)` and resample D gaps with replacement per draw. Require a finite vector of at least two datasets and at least 100 draws.

**Why:** paired comparisons must travel together; outer folds overlap. **Hint boundary:** each input entry is already one dataset’s method difference. Bootstrap these entries, not individual split errors.''')
    code('''# CHECK — known constant and bounded reproducible resampling
np.testing.assert_allclose(bootstrap_gap(np.full(6,-.4)),[-.4,-.4,-.4])
g=np.array([-1.,0.,2.,1.])
actual=bootstrap_gap(g,seed=56)
np.testing.assert_array_equal(actual,bootstrap_gap(g,seed=56))
assert actual[0]==.5 and g.min()<=actual[1]<=actual[2]<=g.max()
print('PASS: paired dataset bootstrap is reproducible and bounded')''')
    md('''## PROVIDED · Complete evaluator
Read the remaining implementation. It calls **your four functions above** for every regime. Win rates are computed pairwise within the same split and then dataset-balanced. The supplementary Friedman/Nemenyi summary ranks mean errors per dataset, a different order of operations. The helper is inlined below; no packaged evaluator replaces your functions.''')
    code('# PROVIDED — complete live audit\n'+piece('relkit/leaderboard.py','summarize'))
    md('''## First run · one split per dataset
This is a Lite sensitivity view, not the full evidence. Predict whether a single split must preserve the all-split ordering. No rankings are forced to change. More repeated evaluations can stabilize estimates without adding new independent datasets.''')
    code('''# PROVIDED — Lite view uses the live functions
lite=summarize(rows.loc[rows.fold==0],ARMS)
display(pd.DataFrame({k:v['mean_ranks'] for k,v in lite.items()}))
assert all(v['datasets']==51 and v['outer_splits']==51 for v in lite.values())''')
    md('''## Required next step · all archived outer splits
No gating or cloud run is necessary: the full score artifact is small. Run the exact same functions on all 816 outer splits. This expands the analysis, not model training. The subsequent CHECK compares the numerical reconstruction, not a claim of fresh paper replication.''')
    code('''# PROVIDED — full published-result reanalysis
full=summarize(rows,ARMS)
display(pd.DataFrame({k:v['mean_ranks'] for k,v in full.items()}))
OUT=ROOT/'data/cache/l056';OUT.mkdir(parents=True,exist_ok=True)
(OUT/'student-audit.json').write_text(json.dumps(full,indent=2))
print('Saved',OUT/'student-audit.json')''')
    code('''# CHECK — audited reference, not a substituted experiment
reference=json.loads((ROOT/'_verify_l056_results.json').read_text())
for regime,s in full.items():
    assert s['datasets']==51 and s['outer_splits']==816
    for arm in ARMS:assert abs(s['mean_ranks'][arm]-reference['summary'][regime]['mean_ranks'][arm])<1e-12
    np.testing.assert_allclose(s['tabm_minus_catboost_rank_gap'],reference['summary'][regime]['tabm_minus_catboost_rank_gap'],atol=1e-12)
print('PASS: live implementation reproduces the recorded aggregate audit')''')
    section('evidence')
    for name,caption in [('ranks','Author-reference four-method ranks; same 51 datasets. Different from full-pool paper Elo.'),('uncertainty','Author-reference paired dataset bootstrap. Negative favors TabM; crossing zero does not establish equivalence.'),('critical-difference','Supplementary rank-of-mean-error analysis, separate from average split ranks; exploratory tests.')]:figure(name,caption)
    code('''# PROVIDED — one real dataset, uncertainty and all-split sensitivity
dataset=sorted(rows.dataset.unique())[0]
sample=rows.loc[(rows.dataset==dataset)&(rows.method_subtype=='tuned_ensemble')]
task_table=sample.groupby('arm').metric_error.agg(['mean','std','count']).reindex(ARMS)
print(dataset,'metric:',sample.metric.unique())
display(task_table)
print('SD is across published outer splits; it is not an independent-fold standard error.')
comparison=pd.DataFrame({'one_split':lite['tuned_ensemble']['mean_ranks'],'all_splits':full['tuned_ensemble']['mean_ranks']})
display(comparison)
print('TabM−CatBoost paired rank gap and interval:',full['tuned_ensemble']['tabm_minus_catboost_rank_gap'])''')
    md('''## EXIT TICKET · an audit you would sign
Include snapshot/hash identity, method pool, regimes, metric orientation and coverage; the three-regime table; one dataset’s mean/SD; naive-versus-balanced ranks; the paired ensemble interval; and one-split versus all-split sensitivity.

Write your own explanation: why does the full-pool paper ranking not have to equal these four-method mean ranks? What supports a stronger tabular baseline, and what temporal relational claim is still unanswered? Distinguish **released-score reanalysis / source primitive parity / exact table reproduction / fresh training**. State which are verified and which are not.

Paste the output and your explanation to the tutor. Running teacher solutions is not independent mastery evidence.''')
    code('''# EXIT — replace the placeholder with your own interpretation
interpretation="WRITE your defensible conclusion here"
exit_ticket={'snapshot':'tabarena-2025-06-12','source_manifest':json.loads((ROOT/'_sources_l056.json').read_text()),
    'rank_table':{k:v['mean_ranks'] for k,v in full.items()},'dataset':dataset,'dataset_errors':task_table.to_dict(),
    'naive_vs_balanced':{'balanced':balanced.mean().to_dict(),'naive':split_ranks.mean().to_dict()},
    'ensemble_gap':full['tuned_ensemble']['tabm_minus_catboost_rank_gap'],
    'lite_vs_full':comparison.to_dict(),'interpretation':interpretation,
    'reanalysis':'PASS','training':'NOT_RUN','paper_table':'INCOMPARABLE'}
(OUT/'exit.json').write_text(json.dumps(exit_ticket,indent=2))
print('Saved',OUT/'exit.json')
print('EXIT needs your written interpretation before submission.')''')
    md('''## Separate unrun track · full official leaderboard
For full-pool ratings, use the pinned official example and environment instructions in [the reproduction contract](l056-reproduction.md#next-reproduction-track-full-official-leaderboard). It is **NOT_RUN** here. Reconcile the exact paper version, roster, imputation, evaluator and bootstrap before claiming historical table recovery. Fresh training is a different, still unrun task; score-only files cannot verify it.

Ask the tutor about any failing CHECK or source discrepancy. The next integer-numbered unit, L057, builds cross-family ensembles.''')
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
    return nb

if __name__=='__main__':
    for solution in (False,True):
        path=(ROOT/'solutions' if solution else ROOT)/(SLUG+'.ipynb')
        path.parent.mkdir(parents=True,exist_ok=True);nbf.write(build(solution),path);print(path)
