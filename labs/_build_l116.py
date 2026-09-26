"""Build aligned lesson, portable notebooks and reference from one source."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0116-debug-gnn-training';TITLE='Debug GNN training: make the failure falsifiable'
summary=json.loads((P/'evidence/l116/summary.json').read_text());canonical=(P/'relkit/debug_l116.py').read_text()
CAP={'trace':'The complete GCN update: fixed graph and feature matrices, three convolutions, train-only loss, gradients, optimizer movement, and validation-selected state.','updates':'Measured full-data 80-epoch course intervention, seed 101: omitting only the optimizer step preserves gradients but eliminates every parameter update.','gradient':'Exact scalar-chain derivatives from autograd: changing multiplicative gain separates contraction from amplification. This is a mechanism illustration, not a trained GNN.','smoothing':'Fixed symmetric propagation on a three-node path. Raw vectors remain degree-shaped while degree-corrected variance vanishes. No weights or labels are trained.','sampling':'Local output row 0 belongs to global node 4. Supervise the two seed rows using mapped global labels; context labels stay outside the loss.','results':'Ten fresh full-data 500-epoch fits. Dots are seeds, diamonds and whiskers are means and sample seed SD, dashed lines are published means.'}
TASKS={'training_loss':('check_loss','Use only the authorized node IDs and return mean negative log-likelihood.'),'train_step':('check_step','Repair the complete update; use the live Task 1 function and return detached loss.'),'seed_loss':('check_seed','Use the first batch_size outputs and map seed rows through n_id to global labels.')}
checks_source=(P/'_check_l116.py').read_text();checks={n.name:ast.get_source_segment(checks_source,n) for n in ast.parse(checks_source).body if isinstance(n,ast.FunctionDef)}
def results():
 text='**Author-reference evidence: fresh training, separate from notebook execution.**\n\n| Population | Mean accuracy | Sample seed SD | Published mean | Verdict |\n|---|---:|---:|---:|---|\n'
 for pop in ['valid','test']:
  a=summary['summary'][pop];text+=f"| {pop} | {a['mean_percent']:.4f}% | {a['sample_sd_pp']:.4f} pp | {a['target_percent']:.2f}% | {a['verdict']} |\n"
 text+='\nAll ten selected checkpoints were independently rescored and replayed through the original released model on all 169,343 nodes: **1,693,430 predictions**, zero class mismatches. Source checks separately compare outputs, gradients and an optimizer update. The repaired loop preserves weights, gradients and BN buffers exactly against the same-runtime inline training order across three controlled updates.\n\n'
 text+=f"The recorded worker-resource estimate for the pilot, ten fits and two diagnostic runs is **USD{summary['successful_resource_usd']:.4f}**. Startup/build/storage overhead is not itemized; it is covered by the reserve. Twenty bounded worker reservations cap compute at USD2.49552 and leave USD7.50448 within the USD10 aggregate plan. Thirteen workers were used; no retries.\n\n"
 text+='> **Scope check.** The complete selected executable experiment is COMPLETE. CLOSE does not establish historical random-state/environment identity or whole-paper parity; both remain NOT_ESTABLISHED. Live Colab and deployment remain NOT_CHECKED. [Evidence summary](../labs/evidence/l116/summary.json).'
 return text

def paired():
 a=summary['paired_course_intervention'];text='| Course arm | Initial loss | Epoch-80 loss | Parameter movement |\n|---|---:|---:|---|\n'
 for arm in ['broken','repaired']:
  r=a[arm];movement='zero at every epoch' if arm=='broken' else 'positive at every epoch'
  text+=f"| {arm} | {r['first_loss']:.6f} | {r['last_loss']:.6f} | {movement} |\n"
 return text+'\nThe first loss and gradient norm match exactly between arms. Later states diverge because only one arm updates parameters. No test-score comparison selects this repair.'

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results()).replace('[[PAIRED_RESULTS]]',paired())
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l116/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l116/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 replacements={
 'WARMUP':('l116-warmup','Recall without notes: which node features may the transductive GCN use? Which labels enter loss? Which metric selects a checkpoint?'),
 'TEACHBACK':('l116-teachback','Teach back: why can validation change without parameter updates, and which two probes distinguish an absent step from label leakage?'),
 'DIAGNOSTIC_WIDGET':('update','Scalar probe: θ=1, L=θ²/2, learning rate 0.2. After four repaired steps θ=0.4096; omitting the step leaves θ=1. At learning rate zero both paths are frozen.'),
 'LEAK_WIDGET':('leak','Fixed probabilities [0.9,0.1], [0.2,0.8], [0.7,0.3], [0.4,0.6]. Flip only labels 2 and 3: train-ID loss stays 0.164252; all-label loss changes from 0.612192 to 0.299001.'),
 'SMOOTH_WIDGET':('l116-smoothing','At depth 1 on the self-looped path A—B—C, symmetric propagation maps [2,4,8] to approximately [2.6330,5.4158,5.6330]. Remove B—C to isolate C; compare symmetric and row-mean operators.')}
 for token,(id,text) in replacements.items():
  div='<div data-gnn-'+id+'></div>' if id in ['update','leak'] else '<div id="'+id+'"></div>'
  s=s.replace('[['+token+']]',text if portable else div+'<noscript><p>'+text+'</p></noscript>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/');s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 html=render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')
 scripts=''.join('<script src="../assets/'+n+'.js"></script>' for n in ['retrieval-pool','retrieval-bank','teachback','oversmoothing-viz','gnn-diagnostics','l116-lesson']) if interactive else ''
 styles=''.join('<link rel="stylesheet" href="../assets/'+n+'.css">' for n in ['lesson','event-snapshot','reproduction','oversmoothing-viz','gnn-diagnostics'])
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title>'+styles+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0115-graph-ml-design-patterns.html">Lesson 115</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 116</p><h1>'+title+'</h1></header>'+html+'</article>'+scripts+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# @colab-bootstrap · install missing packages only; report actual runtime.
import importlib.util, importlib.metadata, subprocess, sys
required={'torch':'torch==2.8.0','numpy':'numpy==2.2.6','pandas':'pandas==2.3.2','ogb':'ogb==1.3.6','sklearn':'scikit-learn==1.7.1','matplotlib':'matplotlib==3.11.0'}
missing=[package for module,package in required.items() if importlib.util.find_spec(module) is None]
if missing: subprocess.check_call([sys.executable,'-m','pip','install',*missing])
print('Python',sys.version)
print({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','ogb','matplotlib']})
'''
run=(P/'_teaching_l116.py').read_text().split('# NOTEBOOK_RUN\n',1)[1]
gate='''# Optional complete named experiment. No paid service is launched by default.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    import tempfile
    torch.set_num_threads(1)
    x,edge,y,split,audit=load_arxiv(Path('l116-data'))
    adj=normalized_adjacency(edge,len(y));device='cuda' if torch.cuda.is_available() else 'cpu'
    output=Path(tempfile.mkdtemp(prefix='l116-full-'))
    records=[train_run(x,adj,y,split,seed,500,output/f'seed-{seed}',device) for seed in range(10)]
    for pop,target in TARGETS.items():
        values=np.array([r['scores'][pop]*100 for r in records])
        print(pop,'mean',values.mean(),'sample seed SD',values.std(ddof=1),'CLOSE' if abs(values.mean()-target)<=TOLERANCE_PP else 'OUTSIDE_TOLERANCE')
else:
    print('Full training NOT_RUN in this kernel. The author experiment is separately recorded evidence.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 116 · '+TITLE+'\n\n'+('Teacher solution' if solution else 'Student lab')+' · Three live TODO/CHECK tasks. Default execution runs synthetic controlled diagnostics; full-data author measurements are separate.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and execution boundaries\n\nNo repository imports or dataset downloads are needed for the default lab. All model, training and diagnostic code is visible below. Full author runtime: Python 3.12, torch 2.8.0, PyG 2.6.1, NumPy 2.2.6, pandas 2.3.2, OGB 1.3.6, scikit-learn 1.7.1. The bootstrap installs missing packages and prints actual versions; existing installations are not claimed pinned. The optional full run downloads a hash-checked archive and executes all ten 500-epoch schedules. It has no notebook-enforced spending cap; the author Modal runner separately enforces bounded reservations. Live Colab NOT_CHECKED. [Protocol](https://avistian.github.io/relational/labs/l116-reproduction.md).')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nGCN architecture adapted from the pinned MIT-licensed [OGB source](https://github.com/snap-stanford/ogb/tree/61e9784ca76edeaa6e259ba0f836099608ff0586). The trainer calls your live functions. The intentionally faulty `missing_step` is supplied only for the course comparison.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+TASKS[task][1] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task);body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=TASKS[task][0];cells.append(nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · controlled diagnostics using your functions\n\nThe tiny two-component graph intentionally allows fitting all eight training examples. This is an optimizer diagnostic, not a generalization estimate. Further probes isolate scalar gradient gain, fixed-operator smoothing, seed indexing and label dependence.'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## Your measured repair curve\n\nThese curves come from this kernel and your live Task 2 function. They are the eight-node diagnostic, separate from the author full-data figures above.'),nbf.v4.new_code_cell('''import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for arm, color in [('broken', '#a5383d'), ('repaired', '#176c68')]:
    rows = report['update_traces'][arm]
    axes[0].plot([r['epoch'] for r in rows], [r['loss'] for r in rows], label=arm, color=color)
    axes[1].plot([r['epoch'] for r in rows], [r['update_l2'] for r in rows], label=arm, color=color)
axes[0].set_ylabel('Training loss'); axes[0].legend()
axes[1].set_ylabel('Parameter-change L2'); axes[1].set_xlabel('Epoch on the eight-node diagnostic')
fig.tight_layout(); fig.savefig('l116-my-repair.png', dpi=150); plt.show()
'''),nbf.v4.new_markdown_cell('## Full named reproduction gate\n\nDefault OFF. Use a GPU and explicit runtime/spend limits for the complete schedule. The author evidence uses separate bounded workers under the USD10 total budget.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nSubmit your three functions, `l116-task-report.json`, the repaired curve and the five-part written defense. Explain what your failing probe ruled out. Ask the teacher for feedback. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l116-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}});path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);aa=[c for c in cells if c.cell_type=='code'];bb=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in aa]==[c.source for c in bb]:
   nb.metadata=old.metadata
   for a,b in zip(aa,bb):a.outputs=b.outputs;a.execution_count=b.execution_count;a.metadata=b.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## One diagnosis loop

Reproduce → minimize → rank falsifiable causes → change one variable → retain a regression check. Curves motivate a probe; they do not uniquely identify a cause.

| Observation | Probe | Interpretation boundary |
|---|---|---|
| Flat training loss | Check loss, gradients and parameter movement on one fixed batch | Nonzero gradient + zero update narrows the update path; also check zero learning rate/frozen parameters |
| Huge/tiny gradients | Layerwise norms and a controlled depth comparison | Scalar gain a repeated k times has derivative aᵏ; a GNN need not share that exact rate |
| Similar node embeddings | Repeated fixed propagation; variance of H/√degree for symmetric normalization | Degree-shaped raw values can remain unequal after mixing; no universal accuracy conclusion |
| Suspicious sampled training | Map n_id[:batch_size]; supervise seed outputs only | Context may send messages without authorizing context labels |
| Implausible held-out score | Perturb unauthorized labels with fixed inputs/RNG; audit selection | Invariance only verifies the tested path, not every preprocessing/selection boundary |

## Correct training contract

Training mode → clear gradients → full forward → train-ID mean NLL → backward → optimizer step. Evaluate in eval mode with no gradient recording. Select first maximum validation accuracy; restore weights and BN buffers. Do not select on test. For the arxiv baseline, all graph/features are visible; held-out labels are not loss inputs.

## Useful measurements

Gradient norm: ‖∂L/∂θ‖₂. Update norm: ‖θafter−θbefore‖₂. These are different, especially with Adam. Zero weight movement does not imply fixed BN buffers. Layerwise gradient norms locate contraction/amplification. Output-row supervision is distinct from context-input gradients through graph messages.

## Selected reproduction

OGB v6 Table 6 GCN, complete ogbn-arxiv, ten seeds ×500epochs, hidden256, dropout.5, Adam.01, official split and evaluator. Targets valid73.00%,test71.74%; predeclared mean tolerance.5pp. Instrumentation adds observations, no clipping or schedule change. Author execution is separate from student mastery and historical identity.

[Lesson116](../lessons/0116-debug-gnn-training.html) · [Exact commands and deviations](../labs/l116-reproduction.md) · [Measured evidence](../labs/evidence/l116/summary.json) · [OGB source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py) · [Smoothing analysis](https://arxiv.org/abs/1801.07606) · [NeighborLoader convention](https://pytorch-geometric.readthedocs.io/en/2.6.1/tutorial/neighbor_loader.html).
'''
(R/'reference/debug-gnn-training.html').write_text(document('Debug GNN training · quick reference',ref))
print('Built lesson, reference, student and solution notebooks')
