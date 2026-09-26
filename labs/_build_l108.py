"""Build aligned lesson, portable student/solution notebooks, and reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0108-temporal-neighbor-sampling';TITLE='Temporal neighbor sampling: spend context carefully'
canonical=(P/'relkit/sampling_l108.py').read_text()
CAP={'interval':'Synthetic boundary calculation: two left searches retain both time-3 events and exclude the cutoff tie at 8.','pipeline':'The sampler returns aligned B-by-k identities and timestamps; frozen TGAT turns this context into positive and negative scores.','recursion':'The A–B edge at 3 passes cutoff 3 into B’s next query. A B–C edge at 6 is excluded on this path.','results':'Author experiment: same questions and negatives, four fixed-weight inference interventions, ten seeds. Points show seeds; bars show sample SD, not confidence intervals.'}
def results():
 b=json.loads((P/'_benchmark_l108_results.json').read_text());import numpy as np
 scalar=float(np.median(b['timing_seconds']['scalar']));fast=float(np.median(b['timing_seconds']['batched']))
 text=f"**Author-reference evidence:** these numbers are not output from your current notebook kernel.\n\n**Measured CPU sampler:** {b['queries']:,} queries; scalar median {scalar:.3f} s, batched median {fast:.3f} s, **{scalar/fast:.2f}× speedup**. Every sample matched. The index stores {b['index_array_bytes']:,} array bytes ({b['index_array_bytes']/2**20:.2f} MiB); construction took {b['construction_seconds']:.3f} s. Three warmed repetitions on the local ARM64 CPU, batch 200; random draws were prepared before timing.\n\n"
 file=P/'_analysis_l108_results.json'
 if not file.exists():return text+'**Full checkpoint evidence is running; no completed result claimed.**'
 r=json.loads(file.read_text())
 text+='**Released-evaluation replay:** all ten checkpoints; '
 text+='; '.join(f"{lane} batch-mean AP {v['mean']:.4f}% ± {v['sd']:.4f} pp seed SD (paper {v['target']:.2f}%, {v['status']})" for lane,v in r['release'].items())+'. '
 text+=f"Maximum absolute probability difference from archived evaluation: {r['release_prediction_max_error']:.3g}. This uses 23,620 all-event and 11,714 new-node positives per seed; the released batching omits the final event in each population.\n\n"
 text+='| Policy | All AP (%) | All Δ (pp) | All questions/s | New AP (%) | New Δ (pp) | Peak GPU MiB |\n|---|---:|---:|---:|---:|---:|---:|\n'
 for arm,v in r['summary'].items():
  a,n=v['all'],v['new'];fmt=lambda x:f"{x['mean']:.3f} ± {x['sd']:.3f}"
  text+=f"| {arm} | {fmt(a['ap'])} | {fmt(a['delta_pp'])} | {a['questions_per_second']['mean']:.0f} | {fmt(n['ap'])} | {fmt(n['delta_pp'])} | {a['cuda_peak_mib']['mean']:.1f} |\n"
 text+='\nAP and paired differences show mean ± sample SD over ten seeds. Throughput is mean positive questions/second, each with one negative. Peak GPU allocated bytes include the loaded model and tensors; CPU index-array bytes are separate. The GPU is an NVIDIA T4. Neither statistic measures whole-system peak resident memory.\n\n'
 text+=f"All {r['intervention_positive_questions']:,} intervention positive questions were evaluated, each with one negative; all recursive access audits found zero nonpast or expired sampled records. New-node and all-event counts overlap. Independent metric reconstruction passed."
 small=r['summary']['uniform5']['all'];base=r['summary']['uniform20']['all'];recent=r['summary']['recent20']['all']
 text+=f"\n\n**Read the trade-off:** reducing fanout from 20 to 5 delivered {small['questions_per_second']['mean']/base['questions_per_second']['mean']:.2f}× measured inference throughput, but lost {-small['delta_pp']['mean']:.2f} AP percentage points on all-event questions. Recent-20 gained {recent['delta_pp']['mean']:.2f} points. These weights were trained with fanout 20: changing context at inference can create a substantial distribution shift. This experiment does not isolate which internal attention behavior caused the loss. The 421/31 tree-slot ratio is not the measured speed ratio."
 return text

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for key,cap in CAP.items():
  path=P/f'figures/l108/{key}.png'
  if not path.exists():s=s.replace('[[FIG:'+key+']]','');continue
  src='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode() if portable else f'../labs/figures/l108/{key}.svg'
  s=s.replace('[[FIG:'+key+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for tag,id_,fallback in [('WIDGET','l108-sampler','**Portable intervention:** at cutoff 9 and width 5, [4,9) retains only time 8. Recent k=2 adds one padding slot.'),('PREDICT','l108-predict','**Predict first:** does lower fanout guarantee identical accuracy? Write why before examining the measurements.'),('TEACHBACK','l108-teachback','**Write your temporal batch defense before consulting the solution.**')]:s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
 for name in ['window_bounds','choose_positions']:
  node=next(n for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef) and n.name==name)
  code='\n'.join(canonical.splitlines()[node.lineno-1:node.end_lineno])
  s=s.replace('[[CODE:'+name+']]','Implement `'+name+'` in the live TODO below.' if portable else '```python\n'+code+'\n```')
 if portable:
  s=s.replace('<div id="warmup"></div>','').replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,scripts=()):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0107-snapshot-methods.html">Lesson 107</a></nav><header><p class="stream-kicker">Year 3 · Quarter 3 · Lesson 108</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in scripts)+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),['retrieval-pool','retrieval-bank','predict','teachback','temporal-sampling-viz','l108-lesson']))
checks={
 'window_bounds':"""assert window_bounds(np.array([1.,3.,3.,8.]),3,2)==(0,1), 'Cutoff ties must be excluded'
assert window_bounds(np.array([1.,3.,3.,8.]),8,5)==(1,3), 'Lower window boundary is included'
assert window_bounds(np.array([1.]),2)==(0,1), 'A singleton history is useful'
assert window_bounds(np.array([]),2)==(0,0)
print('PASS: half-open boundary, ties, empty and singleton')""",
 'choose_positions':"""assert choose_positions(3,5,'recent',np.zeros(5)).tolist()==[-1,-1,0,1,2]
assert choose_positions(2,4,'uniform',np.array([.1,.9,.1,.9])).tolist()==[0,1,0,1]
assert choose_positions(0,3,'uniform',np.zeros(3)).tolist()==[-1,-1,-1]
print('PASS: recent padding and uniform replacement')""",
 'expansion_size':"""assert expansion_size(0,20)==1
assert expansion_size(2,20)==421
assert expansion_size(2,5)==31
assert expansion_size(3,1)==4
print('PASS: neighbor-tree bound; extra source recursion is not counted')"""}
bootstrap="""# @colab-bootstrap — standalone CPU teaching runtime, separate from pinned GPU replay.
import sys,subprocess,importlib.metadata
required={'numpy':'2.5.0','torch':'2.13.0','pandas':'3.0.3','scikit-learn':'1.9.0'}
missing=[]
for package,version in required.items():
    try: current=importlib.metadata.version(package).split('+')[0]
    except importlib.metadata.PackageNotFoundError: current=None
    if current!=version: missing.append(package+'=='+version)
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version.split()[0], 'teaching pins',required)
"""
identity=json.loads((P/'_inputs_l108.json').read_text())
load="""# PROVIDED — raw bytes AND processed arrays must match the frozen experiment.
DATA_DIR=Path('l108-data')
nodes,edge_features,streams,data_audit=load_wikipedia(DATA_DIR)
EXPECTED=__PINS__
assert data_audit['split_sha256']==EXPECTED['split_sha256']
with np.load(DATA_DIR/'processed.npz') as arrays:
    for name,item in EXPECTED['processed_arrays'].items():
        assert hashlib.sha256(arrays[name].tobytes()).hexdigest()==item['sha256'], name
print('Authenticated',len(streams['full']['e']),'events;',len(nodes)-1,'nodes')
""".replace('__PINS__',repr({k:identity[k] for k in ['split_sha256','processed_arrays']}))
model_check="""# CHECK — the readable student's sampler and optimized sampler must drive identical predictions.
torch.set_num_threads(1);torch.manual_seed(108)
small={'u':np.array([1,2,2,1]),'v':np.array([2,3,4,4]),'t':np.array([2.,3.,6.,7.]),'e':np.arange(1,5)}
class StudentFinder(TemporalIndex):
    def get_temporal_neighbor(self,nodes,cutoffs,num_neighbors=20):
        return self.sample_scalar(nodes,cutoffs,num_neighbors,self.policy,self.window)
fast=TemporalIndex(small,5);student_finder=StudentFinder(small,5)
model=TGAT(fast,np.zeros((5,4),np.float32),np.random.default_rng(0).normal(size=(5,4)).astype(np.float32),dropout=0).eval()
args=(np.array([1,2]),np.array([3,4]),np.array([4,3]),np.array([8.,10.]),3)
with torch.no_grad():
    np.random.seed(108);expected=model.contrast(*args)
    model.ngh_finder=student_finder;np.random.seed(108);actual=model.contrast(*args)
for a,b in zip(expected,actual):torch.testing.assert_close(a,b,rtol=0,atol=0)
print('PASS: exact full-model prediction parity; synthetic mechanism check, not trained AP')
"""
benchmark="""# RUN — all Wikipedia source queries, equality first, then three warmed timing repeats.
# Both live TODO functions execute through sample_scalar; changed semantics must fail parity.
measured=benchmark_sampler(streams['full'],len(nodes),batch_size=200,repeats=3)
scalar=float(np.median(measured['timing_seconds']['scalar']))
batched=float(np.median(measured['timing_seconds']['batched']))
print('Same records:',measured['samples_exactly_equal'])
print(f'Scalar {scalar:.3f}s; batched {batched:.3f}s; speedup {scalar/batched:.2f}x')
print('Index array bytes:',measured['index_array_bytes'])
print('Expansion bounds:',{k:expansion_size(2,k) for k in [5,20]})
Path('l108-fresh.json').write_text(json.dumps(measured,indent=2))
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 108 · '+TITLE+'\n\n'+('Executed solution' if solution else 'Student lab')+' · Three live tasks. Default CPU run authenticates the full dataset and benchmarks every source query. The ten-checkpoint author GPU experiment is separate and explicitly identified below. No learner completion is inferred.'),nbf.v4.new_code_cell(bootstrap)]
 # Split narrative by headings to keep the notebook navigable.
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Implement and test the sampler\n\nThe first two tasks define the readable reference actually used below. The third predicts an upper bound on neighbor-tree slots. The optimized sampler is PROVIDED and must match your reference; it does not silently replace your answers.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in checks if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   signature=body.splitlines()[node.lineno-1]
   body=signature+'\n    """'+ast.get_docstring(node)+'"""\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+task),nbf.v4.new_code_cell(checks[task])])
 cells.append(nbf.v4.new_markdown_cell('## Complete TGAT model, preprocessing and trainer\n\nProvided for independent reading and reproduction. This is the unchanged L103 reconstruction of the pinned released attention variant. The code is visible in chunks, including selection and training; default notebook execution defines the trainer but does not train a new checkpoint. `release=True` preserves release behavior. Our new sampler is substituted only for the declared interventions.'))
 for chunk in re.split(r'^# %% ',(P/'relkit/tgat_l103.py').read_text(),flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);cells.extend([nbf.v4.new_markdown_cell('### PROVIDED · '+heading),nbf.v4.new_code_cell(body.strip())])
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · sampler-to-model handoff'),nbf.v4.new_code_cell(model_check),nbf.v4.new_markdown_cell('## Full-data execution\n\nThe first download is approximately 560 MB. Existing raw bytes and processed arrays are authenticated. Timing depends on your hardware; do not expect the author’s exact seconds. No GPU or hidden scoring model is needed for this full sampler experiment.'),nbf.v4.new_code_cell(load),nbf.v4.new_code_cell(benchmark)])
 cells.append(nbf.v4.new_markdown_cell('## Reproduce the complete checkpoint evaluation\n\nThe author executed all ten seeds in a separate pinned T4 runtime. Full commands, input retrieval, artifacts and budget are in [l108-reproduction.md](https://avistian.github.io/relational/labs/l108-reproduction.md). From a checkout with authenticated L103 checkpoints:\n\n```bash\n.venv/bin/modal run modal/l108_replay.py --pilot\n.venv/bin/modal run --detach modal/l108_replay.py\n.venv/bin/python labs/_collect_l108.py --download\n```\n\nThe default notebook does not execute paid GPU jobs. Existing checkpoints are 10 × approximately 10 MB and may be downloaded by an authorized operator from the recorded L103 volume. The visible `run_training` provides an independent fresh-training route; fresh training creates new evidence and is not part of this lesson’s executed scope. Historical/full-paper identity remains unestablished.'))
 cells.append(nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit the three TODO implementations, a new boundary fixture, your fresh full-data benchmark and a written two-hop/evidence defense. Explain why the window changes eligible history without shrinking this index, and why 421/31 is not a measured speed ratio. Ask the teaching agent to review it. Status: **PENDING_WRITTEN_DEFENSE**.'))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 for i,c in enumerate(cells):c.id=f'l108-{i:03d}'
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count;x.metadata=y.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Legal history

For `(node, cutoff)` and duration W, use `[cutoff − W, cutoff)`. Two left-sided binary searches delimit the interval. Real late arrivals also need availability checks. Node ID alone is not a temporal cache key.

## Bounded records

Uniform sampling uses replacement; repeated records are possible. Recent sampling keeps the last k eligible records and left-pads missing slots. Each slot preserves neighbor ID, event ID, and timestamp. Zero IDs mark padding.

## Recursive query

After traversing an edge at time s, TGAT queries the child at cutoff s. Returning to the root cutoff can reveal later events. Neighbor-tree slots sum to 1+k+…+kᴸ; actual TGAT also computes source recursion and three endpoint branches.

## Costs

Binary search O(log degree); gather O(k); sorting sampled times O(k log k). Compressed index O(E+N). A recency window does not shrink an offline retained index. Report construction, index bytes, sampler throughput and synchronized end-to-end inference separately.

## Evidence

An optimization preserves identical samples; a sampling policy changes context. Pair questions and negatives. Frozen-checkpoint inference interventions do not establish retrained accuracy. Source batch-mean AP differs from pooled intervention AP. Report sample SD separately from confidence intervals.

[Lesson 108](../lessons/0108-temporal-neighbor-sampling.html) · [Protocol](../labs/l108-reproduction.md) · [TGAT §3.4](https://arxiv.org/html/2002.07962v1#S3.SS4)
'''
(R/'reference/temporal-sampling.html').write_text(document('Temporal sampling · quick reference',ref))
print('Built lesson, reference and portable notebooks')
