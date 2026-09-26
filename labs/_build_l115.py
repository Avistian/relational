"""Deterministic lesson/notebook build from canonical implementation and source prose."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0115-graph-ml-design-patterns';TITLE='Graph ML design patterns: encoder → message passing → head'
summary=json.loads((P/'evidence/l115/summary.json').read_text());canonical=(P/'relkit/patterns_l115.py').read_text()
CAP={'architecture':'The exact arxiv GCN: identity encoder, two hidden GCN blocks, a third graph convolution in the head, train-label loss and validation-selected inference.','readouts':'Fixed synthetic vectors illustrate node selection, endpoint composition and within-graph pooling. These are alternative heads, not sequential layers.','boundaries':'Every module preserves row identity, type, edge direction, graph membership, visibility and supervision boundaries.','results':'Ten fresh full-data modular GCN fits, 500 epochs each. Dots are seeds; diamonds and whiskers are mean and sample seed SD; dashed lines are published means.'}
TASKS={'graph_forward':('check_compose','Compose the three live modules once each; preserve adjacency at both graph operations.'),'pair_features':('check_pairs','Use symmetric product or ordered concatenation according to the link direction contract.'),'graph_mean':('check_pool','Group by graph IDs, compute means and return zero for explicitly empty graphs.')}
checks_source=(P/'_check_l115.py').read_text();checks={n.name:ast.get_source_segment(checks_source,n) for n in ast.parse(checks_source).body if isinstance(n,ast.FunctionDef)}
def results():
 s='**Author-reference evidence: fresh training, separate from your kernel execution.**\n\n| Population | Mean accuracy | Sample seed SD | Paper mean | Verdict |\n|---|---:|---:|---:|---|\n'
 for pop in ['valid','test']:
  a=summary['summary'][pop];s+=f"| {pop} | {a['mean_percent']:.4f}% | {a['sample_sd_pp']:.4f} pp | {a['target_percent']:.2f}% | {a['verdict']} |\n"
 s+='\nThe descriptive tolerance was declared before training: absolute mean difference ≤0.5 percentage points. Variance is reported, not required to match. Ten-run seed SD measures training variation on this graph; it is not uncertainty over independent datasets. All **1,693,430 final node predictions** were independently scored and replayed through the original model, with zero class mismatches.\n\n'
 s+=f"The recorded pilot-plus-ten-fit worker-resource estimate is **USD{summary['successful_resource_usd']:.4f}**, excluding unitemized startup/build/storage overhead. The conservative worker ceiling is USD1.497312, with USD8.502688 reserved under the USD10 aggregate plan. [Full evidence](../labs/evidence/l115/summary.json).\n\n[[FIG:results]]"
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l115/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l115/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 warm='**Recall before reading:** What population enters the arxiv GCN forward pass? Which labels enter loss? What does validation select?'
 teach='**Teach back:** Explain why a shape-preserving refactor can still alter propagation, normalization, visibility or the unit of prediction.'
 s=s.replace('[[WARMUP]]',warm if portable else '<div id="l115-warmup"></div><noscript><p>'+warm+'</p></noscript>')
 s=s.replace('[[TEACHBACK]]',teach if portable else '<div id="l115-teachback"></div><noscript><p>'+teach+'</p></noscript>')
 static='Baseline: node 0 selects [1,2]; pair 0→1 gives product [3,10] or ordered concatenation [1,2,3,5]; graph IDs [0,1,0] give means [[4,6.5],[3,5]]. Reverse the pair and recompute before checking the notebook.'
 s=s.replace('[[EXPLORER]]',static if portable else '<div data-graph-patterns></div><noscript><p>'+static+'</p></noscript>')
 cutoff='At cutoff day 10, an event occurring on day 4 but available on day 11 must be excluded. Raising the cutoff to day 11 admits it, provided the event precedes the query under your declared time convention.'
 s=s.replace('[[CUTOFF]]',cutoff if portable else '<div id="l115-cutoff"></div><noscript><p>'+cutoff+'</p></noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=''.join('<script src="../assets/'+n+'.js"></script>' for n in ['retrieval-pool','retrieval-bank','teachback','rdl-stack-viz','graph-patterns','l115-lesson']) if interactive else ''
 styles=''.join('<link rel="stylesheet" href="../assets/'+n+'.css">' for n in ['lesson','event-snapshot','reproduction','rdl-stack-viz','graph-patterns'])
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0114-ogb-error-analysis.html">Lesson 114</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 115</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap · install missing packages only; print actual versions.
import importlib.util, importlib.metadata, subprocess, sys
required={'torch':'torch==2.8.0','numpy':'numpy==2.2.6','pandas':'pandas==2.3.2','ogb':'ogb==1.3.6','sklearn':'scikit-learn==1.7.1'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version)
print({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','ogb']})
'''
run=(P/'_teaching_l115.py').read_text().split('# NOTEBOOK_RUN\n',1)[1]
gate='''# Full named experiment: complete raw data and visible trainer, default OFF.
# Prefer GPU; arrange a runtime/spend cutoff before enabling this cell.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    import tempfile
    torch.set_num_threads(1)
    x,edge,y,split,audit=load_arxiv(Path('l115-data'))
    adj=normalized_adjacency(edge,len(y));device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(tempfile.mkdtemp(prefix='l115-full-'))
    records=[train_run(x,adj,y,split,seed,500,output/f'seed-{seed}',device) for seed in range(10)]
    for pop,target in TARGETS.items():
        values=np.array([r['scores'][pop]*100 for r in records])
        print(pop,'mean',values.mean(),'seed SD',values.std(ddof=1),'CLOSE' if abs(values.mean()-target)<=TOLERANCE_PP else 'OUTSIDE_TOLERANCE')
else:
    print('Full training NOT_RUN in this kernel; author ten-seed execution is separate evidence.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 115 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks. Default execution fits tiny synthetic examples; the named experiment is separately gated.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and source identity\n\nDefault execution needs no dataset download or repository imports. All implementation is visible below. The optional full run downloads an 80 MiB archive, verifies SHA-256 and runs all ten full schedules. Author runtime: Python 3.12, torch 2.8.0, NumPy 2.2.6, pandas 2.3.2, OGB 1.3.6, scikit-learn 1.7.1; independent original-source replay additionally uses PyG 2.6.1. Existing notebook package versions are printed, not silently represented as pinned. Live Colab NOT_CHECKED. [Protocol](https://avistian.github.io/relational/labs/l115-reproduction.md).')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nAdapted from the pinned MIT-licensed OGB GCN, with explicit modular boundaries. See [source and license](https://github.com/snap-stanford/ogb/tree/61e9784ca76edeaa6e259ba0f836099608ff0586). The same three functions you implement below are used by the RUN examples.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+TASKS[task][1] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task);body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=TASKS[task][0];cells.append(nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · exercise your live modules\n\nNode example trains the tiny modular GCN. Link and graph examples train readout heads on fixed synthetic embeddings. All fit and score the same examples: these verify executable composition, not generalization or benchmark parity.'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## Full named reproduction gate\n\nThe default does not launch full training. The complete author experiment uses bounded Modal workers described in the protocol, with a USD10 aggregate plan. The notebook switch itself does not enforce a cloud spending limit.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, the three-row design table, `l115-task-report.json`, and the five-point written defense. A green synthetic fit does not reproduce an OGB benchmark. Ask the teacher for feedback. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l115-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Before choosing a model

For each **prediction unit**, predict **target** using information available at **cutoff**. Record row IDs, type, graph membership, direction, availability and supervision.

| Unit | Readout | Shape | Boundary |
|---|---|---|---|
| Node | Select target vectors; node head may itself propagate | N×C, or selected seeds×C | Context nodes are not automatically supervised |
| Undirected link | Symmetric endpoint product + decoder | M×d → M scores | Swapping endpoints must preserve score |
| Directed link | Ordered endpoint concatenation + decoder | M×2d → M scores | Swapping endpoints may change score |
| Graph | Pool within batch IDs + graph head | B×d → B×C | No pooling or edges across independent graphs |

## Exact modular arxiv GCN

Identity encoder → GCN128→256/BN/ReLU/dropout.5 → GCN256→256/BN/ReLU/dropout.5 → GCN256→40/log-softmax. Every GCN uses S(HWᵀ)+b. There are THREE graph multiplications. Full-node BN; train-label mean NLL; Adam.01;500epochs; first best validation checkpoint with buffers;ten seeds. A Linear head or added learned encoder changes the experiment.

## Useful probes

Match outputs, gradients and an optimizer update under copied state and matched RNG. Check permutation equivariance for node representations and invariance for graph readouts. Reverse candidate endpoints. Duplicate all graph vectors to distinguish sum and mean. Insert a late-arriving record and ensure exclusion at earlier cutoffs. Change held-out labels and verify unchanged training loss.

## Evidence vocabulary

Operator parity is a controlled implementation comparison. Ten fresh full-data runs reproduce the selected executable experiment. CLOSE means within the declared mean-score tolerance, not historical identity. Synthetic fits establish execution only. Seed SD is not uncertainty over new datasets. Learner mastery requires a written defense.

[Lesson115](../lessons/0115-graph-ml-design-patterns.html) · [Protocol and exact commands](../labs/l115-reproduction.md) · [Measured evidence](../labs/evidence/l115/summary.json) · [Battaglia et al.](https://arxiv.org/abs/1806.01261) · [OGB source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py).
'''
(R/'reference/graph-ml-design-patterns.html').write_text(document('Graph ML design patterns · quick reference',ref))
print('Built lesson, reference, student and solution notebooks')
