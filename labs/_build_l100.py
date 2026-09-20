"""Build coherent checkpoint lesson, standalone notebooks, reference and measured tables."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0100-heterogeneous-gnn-checkpoint';T='Heterogeneous GNN checkpoint: defend the whole pipeline'
r=json.loads((P/'_experiment_l100_results.json').read_text());assert r['status']=='COMPLETE'
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 captions={'ARCHITECTURE':('architecture','Trace one paper seed from full typed tables through native sampling, both message-passing layers and seed-only supervision. Full-graph evaluation follows frozen validation selection.'),'IDENTITY':('identity','Illustrative typed ID trace: author local row 1 is author 2; paper local row 1 is paper 4. The type selects the correct map.'),'RESULT_FIGURE':('results','Measured complete course run. Each point is one seed on the same graph/split; horizontal bars are means. These are not independent datasets.')}
 if portable:captions['WEIGHTING']=('weighting','Fixed arithmetic example: changing batch partition cannot change the five-seed mean. A shorter batch contributes proportionally less.')
 for key,(name,caption) in captions.items():
  src='data:image/png;base64,'+base64.b64encode((P/f'figures/l100/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l100/{name}.svg'
  s=s.replace('[['+key+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption} On a narrow screen, scroll the figure horizontally.</figcaption></figure>')
 if not portable:s=s.replace('[[WEIGHTING]]','<div id="batch-weighting"></div>')
 rows='| Arm | LR | Accuracy mean ± seed SD | Macro F1 mean | Parameters | Seconds per selected fit |\n|---|---:|---:|---:|---:|---:|\n'
 for arm in ['rgcn','hgt','hgt_uniform','mlp']:
  d=r['summary'][arm];x=next(x for x in r['selected'] if x['arm']==arm)
  rows+=f"| {arm} | {x['lr']} | {100*d['accuracy']['mean']:.2f}% ± {100*d['accuracy']['sample_sd']:.2f} pp | {d['macro_f1']['mean']:.4f} | {x['parameters']:,} | {d['seconds']['mean']:.2f} |\n"
 rows+='\n| Seed | HGT − R-GCN | HGT − uniform HGT | HGT − MLP |\n|---|---:|---:|---:|\n'
 for i in range(3):rows+=f'| {i} | '+' | '.join(f"{100*r['paired_accuracy_differences'][a][i]:+.2f} pp" for a in ['rgcn','hgt_uniform','mlp'])+' |\n'
 rows+='\n**Author-reference evidence:** all 24 fits, 960 epochs and 3,840 optimizer updates completed. HGT exceeds the uniform ablation by 1.70 percentage points on average here. This is conditional evidence under one fixed protocol, not a universal attention benefit. R-GCN predicts the majority class for all test papers in every selected run (49.54% accuracy); its failed learning trajectory limits the family comparison. The experiment is retained as specified.\n'
 s=s.replace('[[RESULTS]]',rows)
 for key,text in [('WARMUP','Retrieve the four questions below before checking your answers.'),('PREDICT','**Predict before reading:** if seed logits match, must sequential Adam updates match a full-batch update? Commit your answer first.'),('TEACHBACK','**Teach back:** explain the boundary between prediction parity, gradient parity, optimizer trajectories and historical reproduction.')]:s=s.replace('[['+key+']]',text if portable else f'<div id="l100-{key.lower()}"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 100 — '+T+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','hetero-checkpoint'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0099-rgcn-vs-hgt.html">Lesson 99</a> · <a href="../reference/heterogeneous-checkpoint.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 100</p><h1>'+T+'</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+'</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['batch-weighting','retrieval-pool','retrieval-bank','predict','teachback','l100-lesson'])+'</body></html>')
checks={
'global_edges':"batch=HeteroData()\nbatch['paper'].n_id=torch.tensor([17,4,83]);batch['author'].n_id=torch.tensor([9,2])\ne=('author','writes','paper');batch[e].edge_index=torch.tensor([[1,0],[0,2]])\nassert torch.equal(global_edges(batch,e),torch.tensor([[2,9],[17,83]])), 'Use the source and destination type maps separately'\nprint('PASS: typed local-to-global endpoint trace')",
'seed_loss':"batch=HeteroData();batch['paper'].y=torch.tensor([0,2,1]);batch['paper'].batch_size=2\nz=torch.tensor([[1.,0.,-1.],[0.,1.,2.],[8.,1.,0.]],requires_grad=True)\na=seed_loss(z,batch);a.backward()\nassert torch.allclose(a,F.cross_entropy(z[:2],torch.tensor([0,2])))\nassert torch.equal(z.grad[2:],torch.zeros_like(z.grad[2:])), 'Context output rows must have no direct supervised gradient'\nbatch['paper'].y[2]=0;assert torch.equal(seed_loss(z,batch),a)\nprint('PASS: only seed labels affect the objective')",
'batch_weight':"assert abs(batch_weight(3,5)*.5+batch_weight(2,5)*.75-.6)<1e-12\nassert abs(sum(batch_weight(b,804) for b in [256,256,256,36])-1)<1e-12\nfor args in [(0,5),(6,5),(2,0)]:\n    try:batch_weight(*args)\n    except ValueError:pass\n    else:raise AssertionError('Invalid batch size accepted')\nprint('PASS: unequal final batch and invalid inputs')"}
bootstrap="""# @colab-bootstrap — this notebook's computation is fully inline; no course clone needed.
# Install the pinned runtime from the instructions above in a fresh environment first.
import sys, torch, torch_geometric
import torch_geometric.typing as pyg_typing
assert pyg_typing.WITH_PYG_LIB, 'Native pyg-lib is required. Follow the pinned install recipe above; do not skip sampling checks.'
print('Runtime:',sys.version.split()[0],torch.__version__,torch_geometric.__version__)
"""
setup='''## Runtime setup — PROVIDED

The tested CPU environment is Python 3.12.3, torch 2.13.0+cpu, PyG 2.8.0.post1 and native pyg-lib 0.10.0 from commit `edc9e2a88d1c5d0953b5f69c98b8365597c6b699`. The executable model and trainer are inline; the notebook does not import the course repository. Native libraries must match your platform and torch ABI. Installation in a fresh environment and live Colab are **NOT_CHECKED**.

```bash
python -m pip install torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-geometric==2.8.0.post1 numpy==2.5.0 scipy==1.18.0 rdflib==7.1.4 pandas==3.0.3 dill==0.3.8 cmake==4.4.3 ninja==1.13.2
MAX_JOBS=2 python -m pip install --no-build-isolation 'git+https://github.com/pyg-team/pyg-lib.git@edc9e2a88d1c5d0953b5f69c98b8365597c6b699'
```

If your existing Colab torch is incompatible, use a fresh matching runtime; do not reinterpret a missing extension as a passing sampling test. The core downloads the checksum-verified 20 MB ACM archive. The AIFB appendix downloads a separate verified archive. CS requires its separate 8.1 GiB archive and adequate resources.
'''
source=(P/'relkit/checkpoint_l100.py').read_text();tree=ast.parse(source)
def source_cells(source,tasks=False):
 tree=ast.parse(source);cells=[]
 prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(prelude))
 for n in tree.body:
  if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
  name=n.name;code=ast.get_source_segment(source,n);task=tasks and name in checks
  cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if task else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(n) or 'Visible implementation used by the following trainer.')))
  if task and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement the contract above, then run CHECK")'
  cells.append(nbf.v4.new_code_cell(code,metadata={'task':name} if task else {}))
  if task:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+name),nbf.v4.new_code_cell(checks[name])])
 return cells
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nComplete real three-type ACM experiment with native mini-batching. Teacher measurements are reference evidence; your execution and written defense remain separate. PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_markdown_cell(setup),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip())
 cells.append(nbf.v4.new_markdown_cell('## Core lab · visible implementation\n\nRead one coherent function at a time. Tasks 1–3 feed the actual sampler audit and training pipeline. The model operators follow L099; the new integration begins at as_heterodata. PROVIDED does not mean unimportant: trace one seed through Model.forward before running the experiment.'))
 cells.extend(source_cells(source,True))
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · native real-graph integration\n\nFresh computation on 23 fixed training seeds, four arms, three batch sizes. All neighbors, float64, fixed weights, tolerance 2e-6. Do not substitute the frozen author results.'),nbf.v4.new_code_cell("torch.set_num_threads(2)\ntorch.use_deterministic_algorithms(True)\ng=load_graph('l100-data')\nreports=[audit_batches(g,arm,bs,seeds=g['train'][:23]) for arm in ARMS for bs in [1,7,23]]\nassert len(reports)==12\nprint('Largest logit/gradient gaps:',max(x['max_logit_gap'] for x in reports),max(x['max_gradient_gap'] for x in reports))"),nbf.v4.new_markdown_cell('## Full fresh course experiment\n\nAll 24 fits and 960 epochs run. Preserve the JSON and checkpoint directory. A few minutes on the author CPU; wall time depends on your runtime.'),nbf.v4.new_code_cell("fresh=run_suite('l100-data','l100-fresh.json')\nassert fresh['status']=='COMPLETE' and len(fresh['runs'])==24\nassert sum(t['updates'] for x in fresh['runs'] for t in x['trace'])==3840\nprint('Arm | accuracy mean | seed SD | macro F1 mean')\nfor arm,d in fresh['summary'].items():\n    print(f\"{arm:12} {100*d['accuracy']['mean']:7.3f}% {100*d['accuracy']['sample_sd']:6.3f}pp {d['macro_f1']['mean']:.4f}\")\nprint('Paired HGT differences:',fresh['paired_accuracy_differences'])\nAUTHOR_ACCURACY="+repr({k:v['accuracy']['mean'] for k,v in r['summary'].items()})+"\nprint('Gaps from frozen author reference:',{a:fresh['summary'][a]['accuracy']['mean']-AUTHOR_ACCURACY[a] for a in ARMS})"),nbf.v4.new_markdown_cell('## EXIT · your defense\n\nSubmit your three functions, fresh JSON/checkpoints and the five answers in lesson section 9. Include the attribution table and evidence gaps. The teacher scores five axes 0–2; pass requires ≥8/10 and no zero in correctness or reproduction defense. Execution alone leaves PENDING_WRITTEN_DEFENSE.')])
 cells.append(nbf.v4.new_markdown_cell('## Published-experiment appendix A · complete AIFB release port\n\nThis separate visible source reproduces the full ten-run release protocol: original 140/36 split, featureless width16 R-GCN, 50 updates, original Adam equation. It does not reuse the ACM wrapper. The following cells redefine some helper names; rerun the core definitions before repeating the ACM suite. The SHA-pinned manifest is embedded; historical Keras/Theano, RNG and graph-byte identity remain INCOMPARABLE. Source: [tkipf release](https://github.com/tkipf/relational-gcn/tree/4bec1341dd46b72bf482f7ed26c2dca4533577f6).'))
 cells.extend(source_cells((P/'relkit/rgcn_l091.py').read_text()))
 cells.append(nbf.v4.new_code_cell('AIFB_MANIFEST='+repr(json.loads((P/'_sources_l091.json').read_text()))+'\nRUN_AIFB = '+str(solution)+"\nif RUN_AIFB:\n    aifb=run_aifb('l100-aifb',AIFB_MANIFEST,seeds=range(10),bases=0)\n    Path('l100-aifb-fresh.json').write_text(json.dumps(aifb,indent=2))\n    print('AIFB mean / sample SD:',aifb['mean'],aifb['sample_sd'])\nelse:\n    print('AIFB execution gated off; full ten-run author replay is separate reference evidence.')"))
 cells.append(nbf.v4.new_markdown_cell('## Published-experiment appendix B · complete HGT CS port\n\nFull visible modern-release model, RTE, HGSampling-style sampling, field protocol, ranking metrics and trainer follow. This is NOT the core NeighborLoader/ACM experiment. Source: [pinned pyHGT](https://github.com/acbull/pyHGT/tree/85eaccd482bc1d1af56c2de297b6e3a88b96d5cd). The publication-era source differs (see reproduction contract). Full CS remains NOT_RUN. The gate is off even in the executed solution. No graph is silently replaced with NN. For downloaded bytes, use the author CS link in the contract and verify the exact SHA before loading.'))
 cells.extend(source_cells((P/'relkit/hgt_l093.py').read_text()))
 cells.append(nbf.v4.new_code_cell("RUN_HGT_CS = False\nCS_PATH = Path('graph_CS.pk')\nCS_SHA256 = 'bf054d93f45d385691f89902e7a021c53328c2ffefe50eb4d91c9dc51d05877f'\nif RUN_HGT_CS:\n    config=protocol_config('paper');config['device']='cuda' if torch.cuda.is_available() else 'cpu'\n    graph=load_oag(CS_PATH,CS_SHA256)\n    evidence={'status':'RUNNING','planned_runs':5,'runs':[],'paper_parity':'NOT_ESTABLISHED','historical_parity':'INCOMPARABLE','data_sha256':CS_SHA256}\n    for seed in range(5):\n        row,_=train_oag(graph,seed,config,'hgt',progress=True);evidence['runs'].append(row)\n        Path('l100-cs-fresh.json').write_text(json.dumps(evidence,indent=2))\n    evidence['status']='COMPLETE';Path('l100-cs-fresh.json').write_text(json.dumps(evidence,indent=2))\nelse:\n    config=protocol_config('paper')\n    assert [config[k] for k in ['width','layers','heads','epochs']]==[256,3,8,200]\n    print('HGT CS training NOT_RUN; complete visible port and full-setting config loaded.')"))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(cells):c.id=f'l100-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if solution:
  html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''# Heterogeneous checkpoint reference

[Lesson 100](../lessons/0100-heterogeneous-gnn-checkpoint.html) · [Full protocol](../labs/l100-reproduction.md)

| Contract | What to check |
|---|---|
| Identity | A node is (type, global row). Restore each edge endpoint using its own type's n_id. |
| Target boundary | First B seed rows receive cross-entropy; context labels never enter the loss. |
| Depth | Two incoming sample hops for two message-passing layers. |
| Full-neighbor parity | All dependencies, fixed weights, no stochastic dropout, node-wise normalization. |
| Gradient parity | Sum B/N times batch-mean gradients; no optimizer steps between batches. |
| Training trajectory | Sequential Adam updates change weights and optimizer state. Parity is not implied. |
| Finite fanout | Sampled nonlinear predictions need not be unbiased full-graph predictions. |
| Pairing | Separate sampling RNG; compare actual sample hashes and shared ablation weights. |
| Selection | Validation-only epoch/LR selection, frozen before complete test scoring. |
| Attribution | Family gap bundles operators. Uniform attention is a narrower intervention with residual confounds. |
| Reproduction | Full course execution, named released replay and original-paper parity are distinct. |

Worked mean: losses [.2,.8,.5,1.1,.4], batches3/2 → means .50/.75. Weighted mean .60; naive mean .625.

Paper anchors: [R-GCN Eq2](https://arxiv.org/abs/1703.06103v4), [HGT §3](https://arxiv.org/abs/2003.01332v1). Native API: [NeighborLoader](https://pytorch-geometric.readthedocs.io/en/latest/modules/loader.html#torch_geometric.loader.NeighborLoader).

Defense rubric: five axes ×0–2; pass ≥8/10, no zero in correctness or reproduction defense. PENDING_WRITTEN_DEFENSE until reviewed.
'''
(R/'reference/heterogeneous-checkpoint.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Heterogeneous checkpoint reference</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/hetero-checkpoint.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built lesson, standalone student/solution notebooks, prepared HTML and reference')
