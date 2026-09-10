"""Build L055 student + solution notebooks from canonical prose/code/figures."""
from _lesson_depth import enrich_notebook
import ast,base64,os
from pathlib import Path
from urllib.parse import urlsplit
import nbformat as nbf
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_l049 import mdtext
ROOT=Path(__file__).resolve().parent
SLUG='0055-tabred-temporal-splits'

def pieces(file,names):
    source=(ROOT/file).read_text();tree=ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)

def build(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
    def figure(name,caption):md('!['+caption+'](data:image/png;base64,'+base64.b64encode((ROOT/'figures/l055'/f'{name}.png').read_bytes()).decode()+')\n\n'+caption+' [Open full-size PNG](figures/l055/'+name+'.png).')
    soup=BeautifulSoup((ROOT.parent/'lessons'/f'{SLUG}.html').read_text(),'html.parser')
    def section(name):
        node=BeautifulSoup(str(soup.find('section',id=name)),'html.parser').find('section')
        for el in node.select('[data-lesson-depth]'):el.decompose()
        for el in node.find_all(['script','figure']):el.decompose()
        for el in node.find_all('div',id=True):
            if el.get('id') not in ['measured-results','paper-report-table']:el.decompose()
        for a in node.find_all('a',href=True):
            p=urlsplit(a['href'])
            if not p.scheme and p.path:a['href']=os.path.relpath((ROOT.parent/'lessons'/p.path).resolve(),ROOT)+('#'+p.fragment if p.fragment else '')
        md(mdtext(node))
    def task(name,signature,prompt,file='relkit/temporal.py'):
        md(prompt)
        code('# TODO — '+name+'\n'+(pieces(file,{name}) if solution else signature+'\n    raise NotImplementedError("Complete the live function")'))
    md('''# Lab 055 · TabReD: the temporal-split reality

**Skill:** audit and implement an evaluation protocol that matches future deployment. No new model is introduced. The load-bearing split, preprocessing, selection and rank operations are visible and editable. The corrected L054 MLP and TabM-mini implementations and their trainer are also visible below.

[Lesson](../lessons/0055-tabred-temporal-splits.html) · [Reference](../reference/tabred-temporal-splits.html) · [Protocol contract](l055-reproduction.md)

**Cold recall:** what does TabM average at inference, and why does a training-seed interval not describe new future periods? Write your answer before continuing.

**Contract:** public July 2026 TabReD release, Ecom Offers / Homesite Insurance / Sberbank Housing. Official archive SHA256 checks; released temporal window 0 versus a new random assignment of the exact same capped row pool; label-blind caps 1500/600/600; row-sampling seeds 550/551/552; model seeds 0/1/2. Numeric and binary columns only. Train-only median/standard scaling. Two candidates per arm; MLP/TabM-mini width64/depth2, k=8, 32 epochs; XGBoost 120 trees. Lower-is-better 1−AUROC or released-log-target RMSE. **INCOMPARABLE to Figure 2.** Allow several CPU minutes plus ~60 MB download. Use the smoke cell before the full run.

PROVIDED = read/run; TODO = implement; CHECK = immediate feedback; EXIT = submit evidence and interpretation. Paper-report figures reanalyze original author logs. Local figures show newly measured v2 reference evidence, not your current kernel results. The earlier historical operator and scores remain separately archived.''')
    for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
    code('''# PROVIDED — imports and workspace setup
import os,sys,copy,math,json,hashlib,time,importlib.metadata
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/relational-matplotlib')
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import rankdata,friedmanchisquare,studentized_range
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier,XGBRegressor
from threadpoolctl import threadpool_limits
from IPython.display import display
import matplotlib.pyplot as plt
torch.set_num_threads(1)
for p in (Path.cwd(),Path.cwd()/'labs',Path.cwd().parent):
    if (p/'relkit').is_dir():
        LABS=p.resolve();os.chdir(LABS);sys.path.insert(0,str(LABS));break
else: raise RuntimeError('Run the bootstrap or start inside the course workspace')
from _fetch_l055 import fetch,HASHES
from _check_release_l055 import verify_extracted
from _live_identity_l055 import live_identity
DATA_ROOT=fetch()
release_identity=verify_extracted(DATA_ROOT)
print({p:importlib.metadata.version(p) for p in ('numpy','torch','scipy','scikit-learn','xgboost')})
print('Official archive checksums passed:',list(HASHES))''')
    section('question');section('features');figure('feature-history','Synthetic as-of feature: day-12 spend is 50, count 2 and mean 25; a day-10 amount arriving day 14 is unavailable.');section('split');figure('splits','Illustration: identical twelve events and 7/3/2 counts, different memberships. T=train, V=validation, E=test.')
    task('chronological_split','def chronological_split(time, validation_start, test_start):','''### TODO 1 · Keep entire timestamp groups together
Return a dict of original row indices (`train`, `val`, `test`) for the half-open intervals in the lesson. Input may be unsorted. Reject non-increasing boundaries and empty partitions. Do not access labels. This exercise is a stricter alternative to the released index arrays; the experiment retains the released arrays and audits their boundary ties.''')
    code('''# CHECK — unsorted inputs and shared timestamps
t=np.array([4,1,3,2,3,6,5,7,8,9])
s=chronological_split(t,4,7)
assert set(s['train'])=={1,2,3,4}, 'Keep both events at time 3 in training'
assert set(s['val'])=={0,5,6} and set(s['test'])=={7,8,9}
assert max(t[s['train']]) < min(t[s['val']])
print('PASS: ties stay together, original indices preserved')''')
    task('paired_random_split','def paired_random_split(temporal_ids, seed):','''### TODO 2 · Construct the paper's paired intervention on the capped pool
**Goal:** concatenate the train/val/test row IDs, shuffle that single pool with the supplied seed, and split it back into the same three sizes. Return sorted IDs for each partition; reject empty or overlapping input.

**Why:** equal counts alone do not ensure the same row pool. This function is called by the actual loader before any labels are read for fitting. The original released temporal groups stay unchanged; only the derived random arm is reassigned.

**Hint:** preserve the three input lengths, use a local NumPy generator, and cut its permutation at cumulative lengths. Move IDs, never feature columns independently.''',file='relkit/temporal_v2.py')
    code('''# CHECK — identity, sizes, reproducibility and overlap rejection
parts={'train':np.array([2,4,7,8]),'val':np.array([10,12]),'test':np.array([15,17])}
paired=paired_random_split(parts,55)
np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))),np.sort(np.concatenate(list(paired.values()))))
assert {k:len(v) for k,v in parts.items()}=={k:len(v) for k,v in paired.items()}
assert all(np.array_equal(paired[k],paired_random_split(parts,55)[k]) for k in parts)
try: paired_random_split({'train':[1,2],'val':[2],'test':[3]},0)
except ValueError: pass
else: raise AssertionError('Reject repeated row IDs before shuffling')
print('PASS: identical pool and counts; memberships differ',paired)''')
    section('availability');figure('availability','Illustrative delay=3: fit at 8 uses 5 training labels; selection at 11 uses 1 validation label. Test labels are scored later.')
    code('''# CHECK / prediction — alter DELAY after predicting the eligible counts
DELAY=3
event=np.arange(1,11);label_available=event+DELAY
fit_eligible=(event<8)&(label_available<=8)
select_eligible=(event>=8)&(label_available<=11)
print('Training labels:',fit_eligible.sum(),'Validation labels:',select_eligible.sum())
if DELAY==3: assert (fit_eligible.sum(),select_eligible.sum())==(5,1)
assert not fit_eligible[event==6].item() if DELAY==3 else True''')
    section('selection');figure('selection','Illustration: validation selects B with .18 error; test error .25 remains the reported result even though A would score .20.')
    task('fit_preprocessor','def fit_preprocessor(train):','''### TODO 3 · Fit preprocessing on training only
Return `(median, mean, scale)` arrays, one entry per column. Treat nonfinite entries as missing; use zero for an entirely missing training column. Compute mean/std after imputation, replacing zero std by one. The PROVIDED transform and every fitted model below consume your returned state. Changing test values must not change it.''')
    code('# PROVIDED — apply frozen statistics\n'+pieces('relkit/temporal.py',{'apply_preprocessor'}))
    code('''# CHECK — held-out intervention and entirely missing training column
raw=np.array([[1.,np.nan],[3.,np.nan],[999.,9.]])
state=fit_preprocessor(raw[:2])
np.testing.assert_allclose(state[1],[2,0])
np.testing.assert_allclose(apply_preprocessor(raw[:2],state),[[-1,0],[1,0]])
changed=raw.copy();changed[2]=[-999999,888888]
for a,b in zip(state,fit_preprocessor(changed[:2])):np.testing.assert_allclose(a,b)
assert np.isfinite(apply_preprocessor(changed,state)).all()
print('PASS: held-out intervention cannot move fitted statistics')''')
    task('select_candidate','def select_candidate(validation_errors):','''### TODO 4 · Select before test
Return the index of the smallest validation error, resolving ties in favor of the first candidate. Reject empty/nonfinite input. No test argument is permitted. The neural trainer selects its checkpoint by validation; this function selects the candidate across its two hyperparameter settings.''')
    code('''# CHECK — test peeking gives the wrong selection
validation=[.21,.18];test=[.20,.25]
chosen=select_candidate(validation)
assert chosen==1 and test[chosen]==.25
assert select_candidate([.2,.2])==0
print('PASS: selected B; report .25 despite test temptation')''')
    section('protocol');figure('protocol','Current local protocol: cap temporal rows once, derive a same-pool random assignment, fit each training-only transform, select by validation, then score test.');section('paper')
    section('paper-analysis');figure('paper-margins','AUTHOR_REPORT_REANALYSIS: XGBoost advantage over MLP-PLR; positive favors XGBoost. Equal window means; 2879/2880 original reports, one missing Cooking seed.')
    md('''### PROVIDED · Existing model operations from L054
The protocol is the new mechanism here. For portability, inspect the complete model implementation below. `x` has shape [batch, features]; MLP produces [batch, outputs]. TabM broadcasts rows to [batch, k, features], applies shared blocks with member-specific input adapters and separate heads, and averages probabilities (classification) or values (regression). No label enters prediction. Numeric/binary inputs only; no embeddings. These are teaching baselines, not Figure 2's full model suite.''')
    code('# PROVIDED — member operations\n'+pieces('relkit/tabm_v2.py',{'kaiming_','sign_pm1','batchensemble_linear','packed_head','member_mean_loss','ensemble_predict'}))
    code('# PROVIDED — MLP and shared layers\n'+pieces('relkit/tabm_v2.py',{'MLP','BatchEnsembleLinear','PackedHead'}))
    code('# PROVIDED — TabM complete forward path\n'+pieces('relkit/tabm_v2.py',{'TabM'}))
    md('''### PROVIDED · Load released indices and audit before fitting
The first metadata column provides event ordering and is excluded from model features. The second Ecom metadata column is also excluded. Numeric and binary columns are concatenated; categorical columns are omitted explicitly. All models use identical rows within a protocol. Cap the temporal partitions once, then derive a random assignment with your live function. Its union hash and counts must match the temporal arm exactly; it is not the original released random array.''')
    code("TASKS=('ecom-offers','homesite-insurance','sberbank-housing')\nARMS=('MLP','TabM-mini','XGBoost')\n"+pieces('relkit/temporal_experiment_v2.py',{'load_release','metric'}))
    code('''# CHECK — actual release geometry, then compare your strict splitter
audit_rows=[]
for name in TASKS:
    d=DATA_ROOT/name/name
    ids={mode:{s:np.load(d/'splits'/mode/(s+'.npy')) for s in ('train','val','test')} for mode in ('random-0','sliding-window-0')}
    pools=[np.sort(np.concatenate(list(v.values()))) for v in ids.values()]
    np.testing.assert_array_equal(*pools)
    assert [len(v) for v in ids['random-0'].values()]==[len(v) for v in ids['sliding-window-0'].values()]
    t=np.load(d/'x_meta.npy')[:,0];given=ids['sliding-window-0']
    a=t[given['val']].min();b=t[given['test']].min()
    strict=chronological_split(t[pools[0]],a,b)
    _,_,_,audit=load_release(name,'temporal',root=DATA_ROOT)
    random_audit=load_release(name,'random',root=DATA_ROOT)[-1]
    assert audit['pool_hash']==random_audit['pool_hash'] and audit['sizes']==random_audit['sizes']
    audit_rows.append(dict(task=name,released_train=len(given['train']),strict_train=len(strict['train']),strict_release=audit['strict_time_boundaries']))
display(pd.DataFrame(audit_rows))
print('False means a shared boundary timestamp, not a reversed timeline.')''')
    md('''### PROVIDED · Fit weights and select checkpoints
Each neural candidate sees only train and validation. Fit regression scaling on training labels. Run a shuffled training minibatch loop; store the first best validation checkpoint. Restore it before returning a predictor. The next stage fits two candidates, calls your selector, then makes test predictions only for the selected candidate.''')
    code('# PROVIDED — complete neural trainer\n'+pieces('relkit/temporal_experiment_v2.py',{'fit_neural'}))
    code('# PROVIDED — validation-only candidate selection\n'+pieces('relkit/temporal_experiment_v2.py',{'evaluate_arm'}))
    code('# PROVIDED — shared experiment and dataset-unit summaries\n'+pieces('relkit/temporal_experiment_v2.py',{'run_suite','summarize'}))
    code('''# CHECK — prove the experiment calls YOUR live preprocessor and selector
# Small real-data run; these counters wrap the functions above, not relkit imports.
calls={'preprocess':0,'select':0,'paired':0}
saved_preprocessor=fit_preprocessor;saved_selector=select_candidate;saved_paired=paired_random_split
def paired_random_split(temporal_ids,seed):
    calls['paired']+=1
    return saved_paired(temporal_ids,seed)
def fit_preprocessor(train):
    calls['preprocess']+=1
    return saved_preprocessor(train)
def select_candidate(values):
    calls['select']+=1
    return saved_selector(values)
try:
    smoke=run_suite(names=('sberbank-housing',),seeds=(0,),train_cap=300,eval_cap=150,epochs=2,trees=10,root=DATA_ROOT)
finally:
    fit_preprocessor=saved_preprocessor;select_candidate=saved_selector;paired_random_split=saved_paired
assert calls=={'preprocess':2,'select':6,'paired':1}, calls
print('PASS: six selected fits used the live functions',calls)''')
    section('evidence');figure('results','Our corrected local errors: seed points and conditional 95% t intervals on fixed same-pool capped pair 0; INCOMPARABLE to Figure 2.')
    md('''### Predict before fitting
Write which margin you expect to change and why. Do not change the random seed to force a reversal. The cell below reruns the complete local protocol; your result is authoritative for your environment.''')
    code('''# PROVIDED — your current experiment (all three seeds)
result=run_suite(root=DATA_ROOT)
result['live_identity']=live_identity(globals())
result['scope']='live notebook run; operator name describes its template, not immutable code'
result['release_identity']=release_identity
result['summary']=summarize(result)
(DATA_ROOT/'student-v2-results.json').write_text(json.dumps(result,indent=1))
rows=[]
for name,task in result['tasks'].items():
    for strategy,records in task.items():
        for arm in ARMS:
            errors=np.array([v['error'] for v in records[0]['arms'][arm]])
            rows.append(dict(task=name,protocol=strategy,model=arm,mean=errors.mean(),sd=errors.std(ddof=1),ci95_halfwidth=4.30265273*errors.std(ddof=1)/np.sqrt(len(errors))))
table=pd.DataFrame(rows);display(table.round(5))
print('Mean ± sample SD; intervals condition on one split and three model seeds.')''')
    task('rank_change','def rank_change(random_errors, temporal_errors):','''### TODO 5 · Rank datasets before averaging
Input is one task's vector of seed-mean errors for each protocol, in ARMS order. Return temporal ranks minus random ranks (average ties). Lower error earns a better rank; positive output means worse relative placement. This is a change in relative placement, not an error difference.''')
    code('''# CHECK — direction, ties and actual results
np.testing.assert_array_equal(rank_change([.1,.2,.3],[.3,.2,.1]),[2,0,-2])
np.testing.assert_array_equal(rank_change([.1,.1,.3],[.1,.1,.3]),[0,0,0])
random_errors=np.array(result['summary']['random']['errors'])
time_errors=np.array(result['summary']['temporal']['errors'])
changes=np.array([rank_change(a,b) for a,b in zip(random_errors,time_errors)])
display(pd.DataFrame(changes,index=list(result['tasks']),columns=ARMS))
print('Mean-rank changes:',dict(zip(ARMS,changes.mean(0))))
for strategy,s in result['summary'].items():
    print(strategy,'Friedman p:',s['friedman_p'],'Nemenyi CD:',s['nemenyi_cd'])''')
    figure('ranks','Our corrected local mean ranks across THREE task units; Nemenyi critical difference. Training seeds are not additional datasets.')
    md('''### PROVIDED · Recompute the actual paper-study aggregates
The compact artifact contains values and SHA256 hashes extracted from pinned author reports. The extractor additionally checks each file against its committed Git blob. Rerun `python labs/_paper_reports_l055.py` to independently fetch and verify the original source reports. The code below performs the complete numerical aggregation locally; it does not train the paper models.

Compute a seed mean within each window, then an equal mean across three windows. Inspect the count beside each window and the matched-seed sensitivity before interpreting a margin. After the CHECK, inspect `pd.DataFrame(counts)` for all within-window means and seed SDs; change the displayed task filter to examine another dataset.''')
    code("# PROVIDED — visible author-report aggregation\nPAPER_ARMS={'mlp':'MLP','mlp-plr':'MLP-PLR','xgboost_':'XGBoost','tabr':'TabR-S'}\nPAPER_TASKS=('homesite-insurance','ecom-offers','homecredit-default','sberbank-housing','cooking-time','delivery-eta','maps-routing','weather')\n"+pieces('_paper_reports_l055.py',{'aggregate_reports'}))
    code('''# CHECK — original report completeness and your reanalysis
paper=json.loads((LABS/'_paper_l055_results.json').read_text())
assert paper['found']==2879 and len(paper['missing'])==1
paper_summary=aggregate_reports(paper['records'])
assert paper_summary==paper['summary']
matched=aggregate_reports(paper['records'],matched=True)
assert matched==paper['matched_seed_summary']
paper_table=pd.DataFrame([{k:v for k,v in r.items() if k!='windows'} for r in paper_summary])
display(paper_table[paper_table.task.isin(['ecom-offers','homecredit-default'])].round(6))
counts=[dict(task=r['task'],model=r['model'],protocol=r['protocol'],**w) for r in paper_summary for w in r['windows']]
display(pd.DataFrame(counts).query('n < 15'))
sensitivity=[dict(task=a['task'],model=a['model'],protocol=a['protocol'],metric=a['metric'],matched_minus_available=b['mean']-a['mean']) for a,b in zip(paper_summary,matched) if a['task']=='cooking-time']
display(pd.DataFrame(sensitivity).round(8))
print('Missing source report:',paper['missing'][0])
# Equal window weight is not pooled weight when counts differ.
windows=[[.1],[.3,.4],[.5,.6]]
assert np.isclose(np.mean([np.mean(v) for v in windows]),1/3)
assert np.isclose(np.mean(np.concatenate(windows)),.38)
print('PASS: 2879 source records reaggregated; equal-window and pooled means distinguished')''')
    section('diagnosis')
    code('''# CHECK / intervention — reconstruct a legal relational aggregate
# Synthetic data: keep event values fixed and move only their arrival times.
amount=np.array([20.,30.,70.]);event=np.array([4,9,10]);arrival=np.array([4,9,14]);query_time=12
legal=(event<query_time)&(arrival<=query_time)
assert (amount[legal].sum(),legal.sum(),amount[legal].mean())==(50.,2,25.)
arrival_earlier=arrival.copy();arrival_earlier[2]=11
legal_earlier=(event<query_time)&(arrival_earlier<=query_time)
assert amount[legal_earlier].sum()==120.
print('As-of spend:',amount[legal].sum(),'Earlier arrival:',amount[legal_earlier].sum())
print('Prediction prompt: would moving only event day 10 to day 9 legalize its day-14 arrival?')''')
    md('''### EXIT TICKET · Hand in an audit, not a winner story
1. Include your timestamp audit and explain the difference between shared boundary timestamps and unavailable labels.
2. Give one task's three-model random/time errors with sample SD; state whether its winner changes.
3. Quantify a margin change and explain why it is a protocol effect, not isolated causal drift evidence.
4. Name what seed intervals omit and one experiment that addresses it.
5. Use the original author reports to compute one XGBoost-minus-MLP-PLR advantage change, with metric direction, window counts and the missing-report policy.
6. Write separate **verified here / author-report reanalysis / new paper training** statements. The paper verdict remains **INCOMPARABLE**.

Paste this output and your explanation to the tutor for feedback. No automatic mastery credit is awarded for running the solution.''')
    code('''# EXIT — replace the text, retaining your measured table and audits
explanation={"protocol":"WRITE your row/information audit", "local":"WRITE your measured margin and uncertainty", "paper":"WRITE an author-report finding and missing-report policy"}
assert len(table)==18 and changes.shape==(3,3)
assert paper_summary==paper['summary']
exit_ticket=dict(audit=audit_rows,table=table.to_dict(orient='records'),rank_changes=changes.tolist(),paper_summary=paper_summary,paper_missing=paper['missing'],live_identity=result['live_identity'],release_identity=release_identity,explanation=explanation,verdict='INCOMPARABLE',interpretation_complete=all('WRITE' not in v for v in explanation.values()))
(DATA_ROOT/'exit-v2.json').write_text(json.dumps(exit_ticket,indent=1))
print('Saved',DATA_ROOT/'exit-v2.json','; interpretation complete:',exit_ticket['interpretation_complete'])''')
    md('''### Required next step · more temporal windows, same live code
After completing EXIT, run this larger comparison. It uses three released temporal windows with same-pool random reassignments, three model seeds, 6000/2000/2000 caps, 64 epochs and 300 trees. This probes split stability while staying within the same numeric/binary teaching implementation. It still differs from Figure 2 in models, feature policy, tuning, repetitions and unresolved original-data identity. **NOT_RUN until you run it.**

The cell deliberately calls the functions currently in this notebook. A CPU cloud alternative is `modal run --detach modal/l055_paper_repro.py --preset closer` from the repository root. The full paper experiment is a separate, unrun reconstruction, described in the reproduction contract.''')
    code('''# PROVIDED — gated scale-up; uses your live functions
RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    closer=run_suite(root=DATA_ROOT,split_ids=(0,1,2),train_cap=6000,eval_cap=2000,epochs=64,trees=300)
    closer['live_identity']=live_identity(globals())
    closer['scope']='live notebook scale-up'
    closer['release_identity']=release_identity
    closer['summary']=summarize(closer)
    closer['verdict']='INCOMPARABLE'
    (DATA_ROOT/'closer-v2-results.json').write_text(json.dumps(closer,indent=1))
    display(pd.DataFrame({s:v['mean_ranks'] for s,v in closer['summary'].items()},index=ARMS))
else:
    print('Scale-up NOT_RUN. Full paper Figure 2 NOT_RUN. Local evidence INCOMPARABLE.')''')
    if solution:
        for c in cells:
            if c.cell_type=='code' and c.source.startswith('# EXIT'):
                c.source=c.source.replace('WRITE your row/information audit','Same capped row pool and partition counts; released time ties disclosed. Per-feature arrival clocks are not verified.')
                c.source=c.source.replace('WRITE your measured margin and uncertainty','See measured table: Sberbank future error falls for all arms. Three-seed SD conditions on window 0 and does not measure new-period uncertainty.')
                c.source=c.source.replace('WRITE an author-report finding and missing-report policy','Author-report XGBoost advantage over MLP-PLR shrinks for Ecom but increases for Homecredit. Cooking random window 0 lacks XGBoost seed 1; retain available means and matched-seed sensitivity. No independent paper retraining.')
    nb=enrich_notebook(nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}),55)
    return nb

if __name__=='__main__':
    for solution in [False,True]:
        path=(ROOT/'solutions' if solution else ROOT)/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True)
        nbf.write(build(solution),path);print(path)
