"""Deterministic lesson, portable notebooks and reference; preserve executed outputs."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0106-temporal-link-prediction';TITLE='Temporal link prediction: test the candidates'
report=json.loads((P/'_analysis_l106_results.json').read_text());canonical=(P/'relkit/edgebank_l106.py').read_text()
CAP={'architecture':'Synthetic membership trace: score AX and BY from strictly earlier history, then update only with observed positives.','candidates':'Candidate populations differ even when the scorer and positive events stay fixed. Inductive pairs need not involve unseen nodes.','metrics':'Pairwise AUROC credits: one for a win, half for a tie, zero for a loss. The same two positives face different negatives.','results':'Measured full Wikipedia source replay. Points are means with population SD over five source iterations; crosses are rounded paper targets. Zero SD can mean an identical candidate stream, not certainty.'}
def results():
 paper_sd=json.loads((P/'_sources_l106.json').read_text())['paper_reported_sd']
 text='All **157,474 raw events** were authenticated. Released history: **'+str(report['history'])+'** events; test: **'+str(report['test'])+'** events; held-out nodes: **922**. Every test event was evaluated in all six conditions, five times.\n\n| Sampler | Memory | Replay AP | Paper AP | Replay AUROC | Paper AUROC | Numeric status |\n|---|---|---:|---:|---:|---:|---|\n'
 for strategy,r in report['conditions'].items():
  for mode,v in r['summary'].items():text+=f"| {strategy} | {mode} | {v['mean'][0]:.4f} ± {v['sd_ddof0'][0]:.4f} | {v['paper_target'][0]:.2f} ± {paper_sd[strategy][mode][0]:.3f} | {v['mean'][1]:.4f} ± {v['sd_ddof0'][1]:.4f} | {v['paper_target'][1]:.2f} ± {paper_sd[strategy][mode][1]:.3f} | {v['numeric_status']} |\n"
 text+='\nValues are mean ± population SD for our replay, alongside the paper’s reported SD. CLOSE compares means only; reported variability need not match. Every batch prediction matched the pinned original memory implementation; an independent metrics library agreed with the visible tie formulas.\n\n'
 for strategy,r in report['conditions'].items():text+=f"- **{strategy}:** {r['distinct_negative_arrays']} distinct candidate arrays across five iterations; positive/negative collisions per iteration: {[x['collisions'] for x in r['runs']]}.\n"
 pooled=sum(x['window']['pooled'][0] for x in report['conditions']['induc_nre']['runs'])/5
 batch=report['conditions']['induc_nre']['summary']['window']['mean'][0]
 text+=f"\n**Aggregation witness:** inductive/window pooled AP is {pooled:.4f}, versus batch-mean AP {batch:.4f}, from exactly the same predictions. Only the latter is the source's paper-comparison metric.\n"
 text+=f"\nRows in history at or after a batch's first timestamp, summed across batch starts: **{report['conditions']['rnd']['runs'][0]['nonpast_history_records_at_batch_starts']}** per iteration. This audit count is not a claim about feature ingestion times, which are unavailable."
 return text

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for key,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l106/{key}.png').read_bytes()).decode() if portable else f'../labs/figures/l106/{key}.svg'
  s=s.replace('[[FIG:'+key+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for key,id_,fallback in [('WIDGET','l106-candidates','**Portable intervention:** remembered negative count 0 / 1 / 2 gives AP 0.7500 / 0.5000 / 0.4167 and AUROC 0.7500 / 0.5000 / 0.2500. Predict these values before calculating.'),('PREDICT','l106-predict','**Predict before reading:** which memory is likely to produce more false positives on historical negatives?'),('TEACHBACK','l106-teachback','**Write the EXIT defense before opening the model answer.**')]:s=s.replace('[['+key+']]',fallback if portable else f'<div id="{id_}"></div>')
 fn=next(n for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef) and n.name=='memory_scores');snippet='\n'.join(canonical.splitlines()[fn.lineno-1:fn.end_lineno])
 s=s.replace('[[CODE:memory]]','Implement the membership computation in TODO 2 below.' if portable else '```python\n'+snippet+'\n```')
 if portable:
  s=s.replace('<div id="warmup"></div>','')
  s=re.sub(r'<details><summary>(.*?)</summary>',r'**\1**\n',s).replace('</details>','')
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,content,scripts=()):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/candidate-evaluation.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0105-continuous-time.html">Lesson 105</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 106</p><h1>'+title+'</h1></header>'+render(content).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),['retrieval-pool','retrieval-bank','predict','teachback','candidate-evaluation-viz','l106-lesson']))
checks={
'legal_history':"""np.testing.assert_array_equal(legal_history([1,2,3,3],[1,4,3,3],3),[True,False,False,False])
try: legal_history([2],[1],3)
except ValueError: pass
else: raise AssertionError('Reject availability before event time')
print('PASS: strict timestamp ties and late arrival')""",
'memory_scores':"""h=np.array([[1,10,1],[2,11,2],[1,10,3],[3,12,10]],float)
c=np.array([[1,10],[10,1],[2,11],[3,12],[4,13]])
np.testing.assert_array_equal(memory_scores(h,c),[1,0,1,1,0])
np.testing.assert_array_equal(memory_scores(h,c,'window'),[0,0,0,1,0])
np.testing.assert_array_equal(memory_scores(np.empty((0,3)),c),np.zeros(5))
print('PASS: directed identity, repetition, quantile window, empty memory')""",
'binary_metrics':"""from sklearn.metrics import average_precision_score,roc_auc_score
assert np.allclose(binary_metrics([1,0],[0,0]),[.75,.75])
assert np.allclose(binary_metrics([1,0],[1,1]),[5/12,.25])
rng=np.random.default_rng(106)
for _ in range(100):
    p=rng.integers(0,2,rng.integers(1,30));n=rng.integers(0,2,rng.integers(1,30))
    y=np.r_[np.ones(len(p)),np.zeros(len(n))];s=np.r_[p,n]
    assert np.allclose(binary_metrics(p,n),[average_precision_score(y,s),roc_auc_score(y,s)])
print('PASS: tie arithmetic and 100 independent metric comparisons')"""}
intent={'legal_history':'Return a boolean mask for history available strictly before the event query. Validate finite aligned clocks and availability ≥ event time. Why is equality permitted for one clock but not the other?', 'memory_scores':'Accept history [N,3], candidates [Q,2], and unlimited/window mode. Apply the released quantile retention rule, then return binary membership scores. Preserve direction; repeated events should not create multiple votes.', 'binary_metrics':'Return [AP, AUROC] for two nonempty binary score vectors. Group equal scores; validate inputs. Derive your answer from counts rather than calling sklearn in this function.'}
bootstrap="""# PROVIDED — standalone runtime; restart the kernel if pip upgrades an imported package.
import sys,subprocess,importlib.metadata
required={'numpy':'2.5.0','scikit-learn':'1.9.0'}
missing=[]
for package,version in required.items():
    try: current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: current=None
    if current!=version: missing.append(package+'=='+version)
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version.split()[0],required)
"""
# Self-contained authenticated source candidates avoid unpublished download URLs.
artifacts={x.name:{'sha256':report['artifacts'][x.name],'base64':base64.b64encode(x.read_bytes()).decode()} for x in sorted((P/'evidence/l106').glob('*.npz')) if x.name.startswith('negatives-') or x.name=='split.npz'}
extract="""# PROVIDED — candidate bytes are embedded, not computed by a hidden model.
# Their hashes tie this scoring replay to the recorded original-sampler execution.
import base64,io,json
FROZEN=__ARTIFACTS__
frozen={}
for name,item in FROZEN.items():
    data=base64.b64decode(item['base64'])
    assert hashlib.sha256(data).hexdigest()==item['sha256'], 'Candidate artifact changed'
    frozen[name]=np.load(io.BytesIO(data),allow_pickle=False)
RAW_PATH=Path('l106-data/wikipedia.csv')
events=load_events(RAW_PATH)
history_ids,test_ids,held_nodes,cuts=released_split(events)
np.testing.assert_array_equal(history_ids,frozen['split.npz']['history_ids'])
np.testing.assert_array_equal(test_ids,frozen['split.npz']['test_ids'])
np.testing.assert_array_equal(held_nodes,frozen['split.npz']['held_nodes'])
print('PASS: authenticated raw data and complete split; no processed data cache')
""".replace('__ARTIFACTS__',repr(artifacts))
fresh="""# PROVIDED — optional exact source-candidate regeneration (CPU, potentially several minutes).
# This executes all five original loops. It materializes ~8.2 million possible pairs.
# False = full scoring replay of authenticated recorded candidates (default).
# True = regenerate all candidates as well. Record the Python/NumPy environment.
REGENERATE_CANDIDATES=False
negative_sets={s:frozen['negatives-'+s+'.npz']['edges'] for s in ['rnd','hist_nre','induc_nre']}
if REGENERATE_CANDIDATES:
    u,v,t=events.T;u=u.astype(np.int64);v=v.astype(np.int64)
    val_end=t[t<=cuts[1]].max()
    for strategy in negative_sets:
        sampler=RandEdgeSampler(u,v,seed=2) if strategy=='rnd' else RandEdgeSampler_adversarial(u,v,t,val_end,strategy,seed=2)
        draws=[]
        for repeat in range(5):
            sampler.reset_random_state();batches=[]
            for start in range(0,len(test_ids),200):
                ids=test_ids[start:start+200]
                if strategy=='rnd':
                    a,b=sampler.sample(len(ids),u[ids],v[ids]);a=u[ids]
                else: a,b=sampler.sample(len(ids),u[ids],v[ids],t[ids[0]],t[ids[-1]])
                batches.append(np.column_stack([a,b]).astype(np.int64))
            draws.append(np.concatenate(batches))
        del sampler
        regenerated=np.asarray(draws)
        print(strategy,'exact candidate equality:',np.array_equal(regenerated,negative_sets[strategy]))
        negative_sets[strategy]=regenerated
"""
run="""# RUN — all test events, all six conditions, all five source iterations.
measured={};expected=__EXPECTED__
for strategy,negatives in negative_sets.items():
    rows,predictions=replay(events,history_ids,test_ids,negatives)
    measured[strategy]={}
    for mode in ['unlimited','window']:
        values=np.array([x[mode]['batch_mean'] for x in rows])
        measured[strategy][mode]=values.mean(0).tolist()
        if not REGENERATE_CANDIDATES:
            np.testing.assert_allclose(values,expected[strategy][mode],rtol=0,atol=1e-14)
        print(strategy,mode,'AP/AUROC',values.mean(0),'SD',values.std(0))
Path('l106-fresh.json').write_text(json.dumps(measured,indent=2))
print('PASS: full prediction replay using the live TODO functions')
""".replace('__EXPECTED__',repr({s:{m:[r[m]['batch_mean'] for r in x['runs']] for m in ['unlimited','window']} for s,x in report['conditions'].items()}))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 106 lab · '+TITLE+'\n\n**'+('Teacher solution' if solution else 'Student')+'** · Real Wikipedia event stream (Tier B). Complete source-candidate scoring replay. Three live implementation tasks; source generation is available below. Start with retrieval and keep the solution closed until you try. No learner mastery is inferred from author execution.'),nbf.v4.new_code_cell(bootstrap)]
 for part in re.split(r'(?=^## )',prose(True),flags=re.M):
  if part.strip():cells.append(nbf.v4.new_markdown_cell(part.strip()))
 cells.append(nbf.v4.new_markdown_cell('## Implement the live computation\n\nPROVIDED cells handle data and replay. TODO cells define functions used by the complete experiment. CHECK cells test behavior immediately. Predict their outputs before running.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  title,code=chunk.split('\n',1);code=code.strip();task=next((name for name in checks if 'def '+name+'(' in code),None)
  cells.append(nbf.v4.new_markdown_cell('### '+title+ ('\n\n**Goal and why:** '+intent[task] if task else '')))
  if task and not solution:
   fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef));code=code.splitlines()[fn.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")'
  cells.append(nbf.v4.new_code_cell(code))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+task),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## Authenticate inputs\n\nFirst raw download is about 560 MB. Existing bytes are always hashed. Embedded candidate artifacts contain all five runs and all test edges; they are original-sampler evidence, not predictions. Default notebook execution recomputes predictions and metrics from scratch. It does not claim fresh candidate generation.'),nbf.v4.new_code_cell(extract),nbf.v4.new_markdown_cell('## Original sampler implementation · MIT licensed\n\nThe following complete released sampler is retained visibly for auditing and optional regeneration. Source: DGB commit '+report['source_commit']+'. Pay attention to global versus instance random state and pair-set ordering.\n\n```text\n'+(P/'sources/l106/LICENSE').read_text()+'\n```'),nbf.v4.new_code_cell((P/'sources/l106/edge_sampler.py').read_text()),nbf.v4.new_code_cell(fresh),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit the three implementations, six fresh score rows and your written evaluation defense. Create one additional failing-case fixture (e.g. tied scores, direction reversal, or late arrival) and explain the bug it rejects.\n\n**Evidence boundary:** full candidate generation and original-code cross-check were run by the author via `_run_l106.py`; this notebook defaults to full scoring replay on authenticated embedded candidates. The optional regeneration switch is off unless you choose it. Historical/full-paper equivalence remains unestablished. Live Colab and deployment NOT_CHECKED. Learner status PENDING_WRITTEN_DEFENSE.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l106-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [x.source for x in a]==[x.source for x in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in nb.cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Define the experiment

A query is (source, destination, time). Declare legal history, eligible candidates, observation interval, negative sampling, update order and metric aggregation. A sampled candidate-discrimination task does not automatically evaluate a forecast over all pairs and a future horizon.

## Memory

EdgeBank∞: score 1 iff a directed pair is in supplied history. EdgeBank window: first retain a recent subset. The DGB `fixed` source uses the 0.85 timestamp quantile of history events, not the paper's fixed-duration description. Update only with observed positives after scoring.

## Candidates

| Sampler | Main pool | Key distinction |
|---|---|---|
| Random | Endpoint pairs | Sparse graphs often yield unseen pairs |
| Historical | Prior pairs absent in current interval | Tests obsolete memories |
| Inductive | Historical pool minus train/validation pairs | New pair is not necessarily new node |

Source fallback and collision behavior must be audited. Inductive negative sampling and withholding unseen nodes are separate mechanisms.

## Binary-score metrics

P/N = positive/negative counts; a/b = their counts scoring 1.

AP = (a/P) a/(a+b) + (1−a/P) P/(P+N), with zero first term if a+b=0.

AUROC = [a(N−b) + 0.5{ab + (P−a)(N−b)}]/PN.

Ties enter AP together and receive half credit in AUROC. Batch-mean and pooled metrics differ. Zero run SD can mean repeated candidates, not certainty.

## Evidence

Full selected Wikipedia released-code replay: two memories × three samplers × five iterations. Complete model prediction parity is separate from paper target closeness and historical identity. Other datasets/neural baselines NOT_RUN.

[Lesson 106](../lessons/0106-temporal-link-prediction.html) · [Protocol](../labs/l106-reproduction.md) · [Paper](https://arxiv.org/html/2207.10128v2)
'''
refdoc=document('Temporal link evaluation · reference',ref).replace('href="0105-continuous-time.html"','href="../lessons/0105-continuous-time.html"')
(R/'reference/temporal-link-evaluation.html').write_text(refdoc)
print('Built HTML, standalone notebooks and reference')
