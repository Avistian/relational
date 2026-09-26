"""Build L112 HTML, reference and standalone notebooks from canonical sources."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0112-ogb-gcn-reproduction';TITLE='OGB reproduction: defend a full GCN experiment'
canonical=(P/'relkit/ogb_l112.py').read_text()
CAP={'architecture':'Every layer and shape in the complete 110,120-parameter GCN. Training masks the labels, not the feature graph.','visibility':'Official year-based label split; all feature rows and citation edges are visible to the transductive baseline.','results':'Full-data author runs: individual seeds, mean and sample seed SD. The shaded target band is a descriptive predeclared tolerance.'}
TASKS={'normalized_adjacency':'Deduplicate undirected relations, add one self-loop, and return D^-1/2 A D^-1/2 as a sparse tensor.','training_loss':'Compute mean negative log-likelihood using only training labels.','selected_epoch':'Return the zero-based first maximizer of validation accuracy; reject an empty history.'}
checks_source=(P/'_check_l112.py').read_text();checks={n.name:ast.get_source_segment(checks_source,n) for n in ast.parse(checks_source).body if isinstance(n,ast.FunctionDef)}
check_names={'normalized_adjacency':'check_normalization','training_loss':'check_loss','selected_epoch':'check_selection'}
def results():
 f=P/'evidence/l112/summary.json'
 if not f.exists():return '**Full author experiment RUNNING.** No ten-seed mean is claimed yet.'
 r=json.loads(f.read_text());s='**Author-reference results, separate from your current notebook execution.**\n\n| Population | Mean accuracy | Sample seed SD | Published mean | Gap | Verdict |\n|---|---:|---:|---:|---:|---|\n'
 for k in ['valid','test']:
  a=r['summary'][k];s+=f"| {k} | {a['mean_percent']:.4f}% | {a['sample_sd_pp']:.4f} pp | {a['target_percent']:.2f}% | {a['gap_pp']:+.4f} pp | {a['verdict']} |\n"
 s+=f"\nCompleted **10 fresh runs × 500 epochs = 5,000 epochs**. Independent metrics cover all 169,343 node predictions per seed. Successful pilot plus full-run resource estimate: **USD{r['successful_resource_usd']:.4f}**; the failed pre-training checker attempt and startup/build/storage overhead are unitemized, covered by the conservative budget reserve. Historical identity and full-paper parity remain NOT_ESTABLISHED. [Machine-readable evidence](../labs/evidence/l112/summary.json)."
 return s

def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text().replace('[[RESULTS]]',results())
 for name,cap in CAP.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l112/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l112/{name}.svg'
  s=s.replace('[[FIG:'+name+']]',f'<figure class="stream-figure" tabindex="0"><img src="{src}" alt="{cap}"><figcaption>{cap}</figcaption></figure>')
 for name in TASKS:
  node=next(n for n in ast.parse(canonical).body if isinstance(n,ast.FunctionDef) and n.name==name)
  s=s.replace('[[CODE:'+name+']]',('Implement `'+name+'` in the live task below.') if portable else '```python\n'+ast.get_source_segment(canonical,node)+'\n```')
 s=s.replace('[[WIDGET]]','**Predict before revealing the worked example:** which epoch is selected, and which test score is reportable?' if portable else '<div class="repro-widget" data-checkpoint-selection data-valid="[72,73,72.8,73,72.9]" data-test="[71,71.6,72.2,71.8,72]"></div>')
 if portable:
  s=s.replace('](../','](https://avistian.github.io/relational/')
  s=re.sub(r'\]\((\d{4}-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
 return s

def document(title,body,interactive=False):
 return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/event-snapshot.css"><link rel="stylesheet" href="../assets/reproduction.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="../lessons/0110-temporal-gnn-checkpoint.html">Lesson 110</a></nav><header><p class="stream-kicker">Year 3 · Quarter 4 · Lesson 112</p><h1>'+title+'</h1></header>'+render(body).replace('<table>','<div class="stream-scroll" tabindex="0"><table>').replace('</table>','</table></div>')+'</article>'+('<script src="../assets/checkpoint-selection.js"></script>' if interactive else '')+'</body></html>'
(R/'lessons'/f'{S}.html').write_text(document(TITLE,prose(),True))
bootstrap='''# Install missing dependencies first, then restart the kernel if needed:
# %pip install torch==2.8.0 ogb==1.3.6 numpy==2.2.6 pandas==2.3.2 scikit-learn==1.7.1
import sys, importlib.metadata
print('Python',sys.version)
print({k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','ogb']})
'''
run='''# RUN: full official data census and two full-graph teaching epochs.
# This executes the live implementations you wrote above.
torch.set_num_threads(1)
x,edge,y,split,audit=load_arxiv(Path('l112-data'))
adj=normalized_adjacency(edge,len(y))
display(pd.DataFrame([{'split':k,'nodes':v,'years':audit['year_ranges'][k]} for k,v in audit['splits'].items()]))
print('Normalized adjacency entries:',adj.values().numel())
# Use a fresh output directory on every run; no silent reuse of old scores.
import tempfile
teaching_dir=Path(tempfile.mkdtemp(prefix='l112-teaching-'))
fresh=train_run(x,adj,y,split,seed=100,epochs=2,output=teaching_dir,device='cpu')
display(pd.DataFrame(fresh['history']))
Path('l112-fresh.json').write_text(json.dumps({k:v for k,v in fresh.items() if k!='history'},indent=2))
print('TEACHING_ONLY: full graph, two epochs; no published-target comparison.')
'''
gate='''# Complete selected reproduction; needs a GPU and an external cost/runtime cap.
# The delivered author ten-run evidence is separate from this kernel execution.
RUN_FULL_REPRODUCTION = False
if RUN_FULL_REPRODUCTION:
    device='cuda' if torch.cuda.is_available() else 'cpu'
    run_root=Path(tempfile.mkdtemp(prefix='l112-full-'))
    records=[train_run(x,adj,y,split,seed,500,run_root/f'seed-{seed}',device) for seed in range(10)]
    rows=[]
    for name,target in TARGETS.items():
        scores=np.array([r['scores'][name]*100 for r in records])
        rows.append({'split':name,'mean_percent':scores.mean(),'sample_sd_pp':scores.std(ddof=1),'gap_pp':scores.mean()-target,'verdict':'CLOSE' if abs(scores.mean()-target)<=TOLERANCE_PP else 'OUTSIDE_TOLERANCE'})
    display(pd.DataFrame(rows))
    print('Ten complete selected runs; historical identity and full-paper parity NOT_ESTABLISHED')
else:
    print('Full reproduction NOT_RUN in this kernel; consult the separate author evidence.')
'''
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# Lesson 112 · '+TITLE+'\n\n'+('Executed solution' if solution else 'Student lab')+' · Three live TODOs and independent CHECKs. Submission requires the written EXIT.'),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## Runtime and data\n\nThe GPU author runtime pins Python 3.12, torch 2.8.0, NumPy 2.2.6, pandas 2.3.2, OGB 1.3.6, and scikit-learn 1.7.1. Source parity also uses torch-geometric 2.6.1. The local CPU execution records its own installed versions. The archive is about 80 MiB and checked by SHA-256. Allow 4 GiB RAM for the notebook and more time on CPU. All model/training code is visible below. Live Colab is NOT_CHECKED. Full training is off by default; the default run performs two complete-graph epochs.')]
 for section in re.split(r'(?=^## )',prose(True),flags=re.M):
  if section.strip():cells.append(nbf.v4.new_markdown_cell(section))
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nArchitecture follows the MIT-licensed OGB release. [Original license](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/LICENSE). Each CHECK invokes your live function.'))
 for chunk in re.split(r'^# %% ',canonical,flags=re.M)[1:]:
  heading,body=chunk.split('\n',1);task=next((k for k in TASKS if 'def '+k+'(' in body),None)
  cells.append(nbf.v4.new_markdown_cell('### '+heading+('\n\n**Goal:** '+TASKS[task] if task else ' · PROVIDED')))
  if task and not solution:
   node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name==task)
   body=body.splitlines()[node.lineno-1]+'\n    raise NotImplementedError("TODO: '+task+'")\n'
  cells.append(nbf.v4.new_code_cell(body.strip()))
  if task:
   check=check_names[task];cells.append(nbf.v4.new_code_cell(checks[check]+'\n'+check+'('+task+')\nprint("PASS: '+task+'")'))
 cells.extend([nbf.v4.new_markdown_cell('## RUN · use your work on the actual graph'),nbf.v4.new_code_cell(run),nbf.v4.new_markdown_cell('## Full ten-seed gate\n\nThis gate uses the same live functions and complete dataset. A changed learner implementation can produce different results. Use the repository runner and protocol ledger for a pinned author-equivalent environment.'),nbf.v4.new_code_cell(gate),nbf.v4.new_markdown_cell('## EXIT submission\n\nSubmit your three functions, `l112-fresh.json`, and answers to the four lesson EXIT prompts. A passing CHECK is implementation evidence; the written defense demonstrates understanding. Explain visibility, normalized messages, complete selected state, and numerical versus historical reproduction. Ask the teaching agent for feedback. **PENDING_WRITTEN_DEFENSE**.')])
 for i,c in enumerate(cells):c.id=f'l112-{i:03d}'
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})
 path=P/('solutions' if solution else '')/f'{S}.ipynb'
 if solution and path.exists():
  old=nbf.read(path,as_version=4);a=[c for c in cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
  if [c.source for c in a]==[c.source for c in b]:
   nb.metadata=old.metadata
   for cc,dd in zip(a,b):cc.outputs=dd.outputs;cc.execution_count=dd.execution_count;cc.metadata=dd.metadata
 nbf.write(nb,path)
 if solution and all(c.execution_count is not None for c in cells if c.cell_type=='code'):
  html,_=HTMLExporter(template_name='lab').from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''## Experiment identity

Dataset/version/hash; official split IDs; visibility contract; graph transform; architecture; optimization; seeds; selection; evaluator; metric aggregation. Freeze these before looking at test results.

## GCN computation

Binary undirected adjacency with one self-loop. S = D^-1/2 (A+I) D^-1/2. Layer: S H W^T + b. ogbn-arxiv release: 128 → 256 → 256 → 40, BN/ReLU/dropout after first two layers. All graph features visible; train labels only in NLL.

## Selection

500 full-batch epochs, Adam lr 0.01. Select first maximum validation accuracy. Save parameters AND BN running buffers; reload before reporting. Full checkpoint replay needs both. This checkpoint does not support optimizer/RNG-exact mid-training resume.

## Ten-run report

Report every seed, mean and sample SD (denominator n−1). Fixed-graph seed SD is not uncertainty over datasets. Predeclare target tolerance; do not tune seeds or selection on test scores. L112 target 71.74%, tolerance 0.5 pp; historical identity is a separate question.

## Common failures

Duplicate reciprocal edges change weights. Masking features instead of labels changes the benchmark. Random splits change the question. Last-epoch or test-best selection changes the estimator. Missing BN buffers change the predictor. CLOSE is a numerical statement, not full-paper parity.

[Lesson 112](../lessons/0112-ogb-gcn-reproduction.html) · [Protocol and commands](../labs/l112-reproduction.md) · [OGB Table 6](https://arxiv.org/html/2005.00687v6#S4.SS3) · [Pinned source](https://github.com/snap-stanford/ogb/blob/61e9784ca76edeaa6e259ba0f836099608ff0586/examples/nodeproppred/arxiv/gnn.py)
'''
(R/'reference/ogb-reproduction.html').write_text(document('OGB reproduction · quick reference',ref));print('Built L112 lesson, reference and standalone notebooks')
