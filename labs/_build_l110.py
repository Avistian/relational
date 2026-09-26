"""Generate aligned checkpoint lesson, reference and visible standalone notebooks."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0110-temporal-gnn-checkpoint';TITLE='Q3 checkpoint: a temporal GNN you can defend'
canonical=(P/'relkit/checkpoint_l110.py').read_text()
CAP={'architecture':'Complete Wikipedia TGN-attn path: 688-number old message → GRU state → strict history attention → score; current features enter only after scoring.','checkpoint':'Illustrative selected epoch 12 and stopping epoch 17: the release can combine selected weights/memory/clocks with stopping-epoch queued messages.','ties':'Worked timestamp stream: fixed size-two slicing crosses a time-2 boundary; extending the batch preserves strict-past memory.','results':'Author-reference full-data runs. Individual seeds and mean ± sample seed SD; published targets apply to the release arm only.'}
def results():
 path=P/'evidence/l110/summary.json'
 if not path.exists():return '**Fresh full-data runs: RUNNING.** No completed ten-seed mean is claimed.'
 r=json.loads(path.read_text());s='**Author-reference evidence, separate from your current notebook output.**\n\n| Protocol | Population | Batch AP mean | Seed SD | Pooled AP mean | Paper comparison |\n|---|---|---:|---:|---:|---|\n'
 for arm in ['release','clean']:
  for lane in ['all','new']:
   a=r['summary'][arm][lane];s+=f"| {arm} | {lane} | {a['mean_ap_percent']:.4f}% | {a['sample_sd_pp']:.4f} pp | {a['pooled_mean_ap_percent']:.4f}% | {a['numerical_verdict']} |\n"
 s+=f"\nExecution: **{r['status']}**; {len(r['paired'])} complete seed pairs. Independently reconstructed {r['event_evaluations']:,} positive event evaluations plus one sampled negative per event. The clean-arm changes are course evidence, not a second paper reproduction.\n\n"
 for lane in ['all','new']:
  a=r['paired_deltas'][lane];s+=f"Clean minus release, {lane}: batch AP {a['mean_batch_delta_pp']:+.4f} pp (paired seed SD {a['sd_batch_delta_pp']:.4f}); pooled AP {a['mean_pooled_delta_pp']:+.4f} pp. "
 s+=f"\n\nCompleted-call resource estimate including the pilot: **USD{r['completed_call_resource_usd']:.4f}**. Startup/build/storage overhead is not itemized; the aggregate plan remains USD10. Full-paper parity remains NOT_ESTABLISHED."
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 census=json.loads((P/'_protocol_l110_results.json').read_text())['counts']
 table='| Population | Events | Fixed-slice tied boundaries | Clean tied boundaries | Largest clean batch |\n|---|---:|---:|---:|---:|\n'
 for name,a in census.items():table+=f"| {name} | {a['events']:,} | {a['release_tied_boundaries']} | {a['clean_tied_boundaries']} | {a['largest_clean_batch']} |\n"
 s=s.replace('[[BOUNDARIES]]',table)
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l110/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l110/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for name in ['restore_checkpoint','strict_batches','legal_history']:
  node=next(n for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef) and n.name==name)
  s=s.replace('[[CODE:'+name+']]',('Implement `'+name+'` in the live TODO below.') if portable else '```python\n'+ast.get_source_segment(canonical,node)+'\n```')
 for tag,id_,fallback in [('WIDGET','l110-batches','**Predict:** partition the six-event stream at sizes 1, 2 and 3, with and without keeping ties together.'),('PREDICT','l110-predict','**Predict before results:** must consistent checkpoint restoration improve AP? Commit your reason before reading.'),('TEACHBACK','l110-teachback','**Teach back:** explain how legal adjacency can coexist with invalid memory, and distinguish consistency from accuracy.')]:s=s.replace('[['+tag+']]',fallback if portable else '<div id="'+id_+'"></div>')
 if portable:
  s=s.replace('<div id="warmup"></div>','').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,scripts=()):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/temporal-checkpoint.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0109-database-timestamp-contracts.html">Lesson 109</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 110</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),['retrieval-pool','retrieval-bank','predict','teachback','temporal-checkpoint-viz','l110-lesson']))
# Behavioral CHECK functions are visible too; they invoke live student functions.
tree=ast.parse((P/'_check_l110.py').read_text());checkdefs={n.name:ast.get_source_segment((P/'_check_l110.py').read_text(),n).replace('base.','') for n in tree.body if isinstance(n,ast.FunctionDef)}
checks={'restore_checkpoint':checkdefs['check_restore']+'\ncheck_restore(restore_checkpoint)\nprint("PASS: atomic restore recovers the next prediction")','strict_batches':checkdefs['check_batches']+'\ncheck_batches(strict_batches)\nprint("PASS: all events retained, no tied boundaries")','legal_history':checkdefs['check_legal']+'\ncheck_legal(legal_history)\nprint("PASS: both clocks and strict ties")'}
goals={'restore_checkpoint':'Restore selected weights and the complete cloned temporal snapshot. A state_dict-only solution must fail the next-prediction counterexample.','strict_batches':'Preserve every event once, in order; extend through a boundary timestamp group. Return chronological dictionaries with a soft size limit.','legal_history':'Return an elementwise boolean mask requiring both event and observed times strictly before the query.'}
bootstrap='''# @colab-bootstrap — standalone implementation, no course-package imports.
import sys,importlib.metadata
print('Python',sys.version.split()[0])
print({name:importlib.metadata.version(name) for name in ['torch','numpy','pandas','scikit-learn']})
'''
run='''# RUN — full real data for the split audit; a separate short training exercise.
torch.set_num_threads(1)
nodes,edges,data,split_audit=load_wikipedia(Path('l110-data'))
display(pd.DataFrame([split_audit['counts']]))
frontiers=[]
for name,events in data.items():
    if name=='full':continue
    groups=list(strict_batches(events))
    for previous,following in zip(groups,groups[1:]):
        assert legal_history(previous['t'],previous['t'],following['t'][0]).all()
    frontiers.append({'population':name,'events':len(events['t']),'groups':len(groups),'largest':max(len(g['t']) for g in groups)})
display(pd.DataFrame(frontiers))
print('Real arrival assumption: observed_time = event_time; actual ingestion NOT_RECORDED')
small={name:({k:v[:(400 if name=='train' else 200)] for k,v in events.items()} if name!='full' else events) for name,events in data.items()}
fresh=[]
for arm in ['release','clean']:
    fresh.append(run_training(nodes,edges,small,seed=19,epochs=2,output=Path('l110-teaching')/arm,arm=arm))
display(pd.DataFrame([{'arm':r['arm'],'selected_epoch':r['selected_epoch'],'AP':r['test']['ap'],'new_AP':r['new_test']['ap']} for r in fresh]))
Path('l110-fresh.json').write_text(json.dumps(fresh,indent=2))
print('MEASURED short teaching run; INCOMPARABLE to paper scores')
'''
gate='''# Full selected experiment; the default kernel run does not execute this gate.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    device='cuda' if torch.cuda.is_available() else 'cpu'
    full_records=[]
    for seed in range(10):
        for arm in ['release','clean']:
            full_records.append(run_training(nodes,edges,data,seed=seed,epochs=50,output=Path('l110-full')/arm,device=device,arm=arm))
    rows=[]
    for arm in ['release','clean']:
        for lane,key in [('all','test'),('new','new_test')]:
            scores=np.array([r[key]['ap']*100 for r in full_records if r['arm']==arm])
            rows.append({'arm':arm,'population':lane,'mean_AP_percent':scores.mean(),'seed_SD_pp':scores.std(ddof=1),'paper_verdict':('CLOSE' if abs(scores.mean()-PAPER_AP[lane])<=CLOSE_TOLERANCE_PP else 'OUTSIDE_TOLERANCE') if arm=='release' else 'COURSE_INTERVENTION'})
    display(pd.DataFrame(rows))
    print('Selected experiment COMPLETE; historical identity INCOMPARABLE; full paper NOT_ESTABLISHED')
else:
    print('Full training NOT_RUN in this kernel; full-data author evidence is separate.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 110 · '+TITLE+'\n\n'+('Solution' if solution else 'Student lab')+' · Three live TODOs, real-data CHECKs, visible model and trainer, written EXIT. Author evidence does not establish learner mastery.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and data\n\nThe full GPU author runs pin Python 3.12, torch 2.8.0, NumPy 2.2.6, pandas 2.3.2 and scikit-learn 1.7.1. Install the linked `requirements-l110-runtime.txt` in a separate environment to match those libraries. The local CPU notebook records its installed versions and is a separate teaching execution. If needed, `%pip install torch numpy pandas scikit-learn` supplies a convenient unpinned teaching runtime; restart after installation. First data access downloads about 534 MiB and verifies its SHA-256. Allow at least 2 GiB RAM. Live Colab is NOT_CHECKED. The long-run gate is off by default; it can take hours and should use an explicit external runtime/cost limit.')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nRead each PROVIDED block as the computation it implements. The original Apache-2.0 TGN source is pinned at `e38cdf85998c6ca077167610dc4e769a688efa95`. Only three functions are TODOs; your implementations are used in the actual checks and clean trainer.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((name for name in checks if 'def '+name+'(' in body),None)
  if not task:heading=re.sub(r'^Task \d+: ','PROVIDED · ',heading)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+goals[task] if task else '')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+task),nbf.v4.new_code_cell(checks[task])])
 cells.extend([nbf.v4.new_markdown_cell('## Execute your temporal contract\n\nThe frontier audit uses your batching and eligibility functions over all released populations. The two short fits use the actual full architecture and your clean checkpoint restore, with reduced training populations only for this teaching exercise.'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## Full reproduction gate\n\nRun all ten seeds in both arms using the complete dataset. This gate is optional for a learner session; the delivered author evidence is collected separately. Use a fresh output directory after modifying code. For bounded cloud runs, use the exact commands and aggregate budget in the reproduction contract.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT · submit and defend\n\nSubmit your three implementations, `l110-fresh.json`, one failed temporal counterexample and a 200–300 word explanation. Include the split, candidate pools, timestamp ties, selected state and branch reset. Explain which results are published-protocol evidence and which are a course intervention. A complete answer must state the unrecorded arrival-history assumption. Ask the teaching agent for feedback. **PENDING_WRITTEN_DEFENSE**.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l110-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Prediction contract

Declare question time, event time, observed time, tie convention, entity eligibility, candidate pool and label maturity. Strict event history here requires event < query AND observed < query. Wikipedia only supplies event times; observed=event is an assumption.

## TGN inference state

Parameters θ; node memory M; last-update clocks T; queued messages Q. Capture and clone all four together. Score with old history, then queue current positive events. Negative candidates never create messages. Restore each evaluation branch from the same selected state.

## Batch boundary

Keep a complete timestamp group inside one batch. The preceding maximum must be strictly below the next minimum. A strict sampler cannot fix a memory path that already consumed a tied event. The nominal batch size becomes a soft limit.

## Selection and resumption

Validation selects the checkpoint; test never selects. The clean protocol restores best validation state even at the epoch cap. Evaluation replay requires θ/M/T/Q. Mid-epoch optimization resume also requires optimizer, RNG and cursor; these artifacts do not claim that capability.

## Metrics and evidence

Release AP is an equal average of within-batch AP. Pooled AP ranks all predictions together. Changing batches can change both candidates and aggregation. Report seed SD on the fixed split; do not reinterpret it as dataset uncertainty. Numerical CLOSE does not establish historical identity. Clean policy scores are course evidence, not a second paper result.

## Minimum audit

Mutate current and future event features; earlier predictions must not change. Advance then restore; recover the next score exactly. Verify tie boundaries, branch independence, late arrivals, full split IDs and independently reconstructed metrics. A deliberately broken implementation must fail.

[Lesson 110](../lessons/0110-temporal-gnn-checkpoint.html) · [Reproduction contract](../labs/l110-reproduction.md) · [TGN paper](https://arxiv.org/abs/2006.10637v3) · [Released trainer](https://github.com/twitter-research/tgn/blob/e38cdf85998c6ca077167610dc4e769a688efa95/train_self_supervised.py)
'''
(R/'reference/temporal-gnn-checkpoint.html').write_text(document('Temporal GNN checkpoint · quick reference',ref));print('Built L110 lesson, reference, student and solution')
