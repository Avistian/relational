"""One prose/implementation source builds lesson, portable notebooks and reference."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0099-rgcn-vs-hgt';T='R-GCN vs HGT: what does a controlled comparison explain?'
result=json.loads((P/'_experiment_l099_results.json').read_text())
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 src='data:image/png;base64,'+base64.b64encode((P/'figures/l099/comparison.png').read_bytes()).decode() if portable else '../labs/figures/l099/comparison.svg'
 s=s.replace('[[ARCHITECTURE]]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="One typed ACM graph feeds R-GCN, HGT, uniform HGT and a feature-only MLP; common train and validation boundaries lead to frozen test scoring."></div><figcaption>Read left to right through the inputs, then down one branch to the prediction. Scroll horizontally on a narrow screen.</figcaption></figure>')
 s=s.replace('[[INTERVENTION]]','**Portable intervention:** reproduce the three scalar aggregations in the worked-trace cell below.' if portable else '<div id="architecture-comparison"></div>')
 table='| Arm | Selected LR | Test accuracy mean ± seed SD | Macro F1 mean | Parameters allocated / active | Seconds per selected fit |\n|---|---:|---:|---:|---:|---:|\n'
 for arm in ['rgcn','hgt','hgt_uniform','mlp']:
  d=result['summary'][arm];r=next(r for r in result['selected'] if r['arm']==arm)
  table+=f"| {arm} | {r['lr']} | {100*d['accuracy']['mean']:.2f}% ± {100*d['accuracy']['sample_sd']:.2f} pp | {d['macro_f1']['mean']:.4f} | {r['parameters']:,} / {r['active_parameters']:,} | {d['seconds']['mean']:.2f} |\n"
 paired='| Paired seed | HGT − R-GCN | HGT − uniform HGT | HGT − MLP |\n|---|---:|---:|---:|\n'
 for i in range(3):paired+=f"| {i} | "+' | '.join(f"{100*result['paired_accuracy_differences'][a][i]:+.2f} pp" for a in ['rgcn','hgt_uniform','mlp'])+' |\n'
 text='**Frozen author measurements: all 24 fits and all 1,440 epochs completed.** Accuracy SD is in percentage points, across three seeds. Runtime is measured CPU wall time for training, validation and final gradient audit; it is not a FLOP count or a separate timing benchmark.\n\n'+table+'\n'+paired+'\n**Observed attribution:** uniform HGT has the higher mean here. The HGT family gap over this R-GCN configuration does not establish a benefit from learned attention. The R-GCN and MLP curves reveal substantial optimization limitations within the fixed 60-epoch budget. No universal family ranking follows.\n'
 s=s.replace('[[RESULTS]]',text)
 for key,text in [('WARMUP','Retrieve the four questions below before reading the feedback.'),('PREDICT','**Predict first:** does replacing HGT attention with uniform weights make it R-GCN? Defend your answer using the trace.'),('TEACHBACK','**Teach back:** distinguish a family comparison from a within-family intervention and state one remaining confound.')]:
  s=s.replace('[['+key+']]',text if portable else '<div id="l099-'+key.lower()+'"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 099 — '+T+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','architecture-comparison'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0098-hetero-mini-batching.html">Lesson 98</a> · <a href="../reference/architecture-comparison-contract.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 099</p><h1>'+T+'</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+'</article>'+''.join('<script src="../assets/'+x+'.js"></script>' for x in ['architecture-comparison','retrieval-pool','retrieval-bank','predict','teachback','l099-lesson'])+'</body></html>')
checks={
'relation_mean':"x=torch.tensor([[2.],[8.],[-4.]],requires_grad=True)\ne=torch.tensor([[0,1,2],[0,0,1]])\ny=relation_mean(x,e,3)\nassert torch.equal(y,torch.tensor([[5.],[-4.],[0.]]))\ny.sum().backward();assert torch.equal(x.grad,torch.tensor([[.5],[.5],[1.]]))\nassert relation_mean(x,torch.empty((2,0),dtype=torch.long),0).shape==(0,1)\nprint('PASS: means, isolated receiver, empty relation, source gradients')",
'receiver_softmax':"s=torch.tensor([[1000.,1.],[1000.,2.],[2.,3.]],dtype=torch.double,requires_grad=True)\ndst=torch.tensor([0,0,2]);a=receiver_softmax(s,dst,4)\nreference=torch.cat([s[:2].softmax(0),torch.ones_like(s[2:])])\nassert torch.allclose(a,reference)\nga=torch.autograd.grad(a.square().sum(),s,retain_graph=True)[0]\ngb=torch.autograd.grad(reference.square().sum(),s)[0]\nassert torch.allclose(ga,gb)\nassert receiver_softmax(s[:0],dst[:0],4).shape==(0,2)\nprint('PASS: joint receiver normalization, stable scores, gradients, empty edges')",
'choose_config':"records=[{'arm':'hgt','lr':lr,'seed':seed,'best_val_ce':loss,'test_accuracy':test} for lr,seed,loss,test in [(.01,0,.4,.99),(.01,1,.1,.99),(.003,0,.2,.01),(.003,1,.2,.01)]]\nassert choose_config(records,'hgt')==.003\nfor r in records:r['test_accuracy']=1-r['test_accuracy']\nassert choose_config(records,'hgt')==.003\nprint('PASS: mean validation beats best-seed and test-score temptations')"}
source=(P/'relkit/compare_l099.py').read_text();tree=ast.parse(source)
bootstrap="""# Self-contained CPU notebook. No relkit/repository imports are needed.
# In a fresh kernel, install the pinned runtime before running the remaining cells:
# %pip install torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
# %pip install numpy==2.5.0 scipy==1.18.0
import sys
print('Python:',sys.version)
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nComplete 24-fit ACM course comparison; separate published-experiment tracks are linked in the contract. Learner status: PENDING_WRITTEN_DEFENSE. The standalone course run downloads 20 MB of checksum-verified data. Author CPU run takes a few minutes. Live Colab is NOT_CHECKED.'),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip())
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nAll data, operators, training, selection and scoring are inline. Complete the three TODO functions. The full suite does fresh computation; frozen author measurements are references only.'))
 prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(prelude))
 for node in tree.body:
  if not isinstance(node,(ast.FunctionDef,ast.ClassDef)):continue
  name=node.name;code=ast.get_source_segment(source,node)
  cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if name in checks else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(node) or '')))
  if name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement the contract above, then run its CHECK")'
  cells.append(nbf.v4.new_code_cell(code,metadata={'task':name} if name in checks else {}))
  if name in checks:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+name),nbf.v4.new_code_cell(checks[name])])
 cells.extend([nbf.v4.new_markdown_cell('## Worked trace: predict before execution\n\nExplain why changing the last score or duplicating one edge changes the output. This scalar trace is not a fitted-model comparison.'),nbf.v4.new_code_cell("messages=torch.tensor([2.,8.,-4.]);scores=torch.tensor([0.,math.log(3),math.log(2)])\nrgcn=(2+8)/2-4;uniform=messages.mean();hgt=(scores.softmax(0)*messages).sum()\nassert rgcn==1 and uniform==2 and torch.allclose(hgt,torch.tensor(3.))\nprint('R-GCN:',rgcn,'Uniform:',uniform.item(),'Attention:',hgt.item())"),nbf.v4.new_markdown_cell('## Full fresh experiment\n\nEvery fit runs all 60 epochs. The split, rates and seeds were frozen before test scoring. No cached scores are substituted. Preserve l099-fresh.json and its checkpoint folder.'),nbf.v4.new_code_cell("fresh=run_suite('l099-data','l099-fresh.json')\nassert fresh['status']=='COMPLETE' and len(fresh['runs'])==24\nprint('Selected learning rates:',fresh['selected_rates'])\nprint(json.dumps(fresh['summary'],indent=2))\nAUTHOR_ACCURACY="+repr({k:v['accuracy']['mean'] for k,v in result['summary'].items()})+"\nprint('Mean accuracy gaps from frozen author reference:',{a:fresh['summary'][a]['accuracy']['mean']-AUTHOR_ACCURACY[a] for a in ARMS})\nprint('Paired HGT differences:',fresh['paired_accuracy_differences'])"),nbf.v4.new_markdown_cell('## EXIT · PENDING_WRITTEN_DEFENSE\n\nSubmit your three functions, l099-fresh.json and an attribution table (observation / intervention / alternative explanation / justified conclusion). Answer the five defense questions in section 9. Explain why the full course experiment, named AIFB replay and unrun CS track have different evidence status. Ask the agent to challenge your claims.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(cells):c.id=f'l099-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if solution:
  html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
ref='''# Architecture comparison contract

[Lesson 99](../lessons/0099-rgcn-vs-hgt.html) · [Full reproduction protocol](../labs/l099-reproduction.md)

| Decision | Required record |
|---|---|
| Information | Same features, legal edges, typed identities and transductive/inductive access |
| Targets | Same train/validation/test IDs and label masks; remove target-defining conference edges |
| Computation | Depth, width, self paths, normalization, heads and sampling stated separately |
| R-GCN | Mean within relation, sum relation contributions, add learned self path |
| HGT | Typed Q/K/V, relation scores/messages, softmax over ALL incoming edges per receiver/head |
| Uniform HGT | No scorer; retain HGT message, output, residual and normalization route |
| Initialization | Match common HGT/ablation tensors; same seed does not match different architectures |
| Tuning | Two rates per arm, three seeds, 60 complete epochs; validation CE only |
| Checkpoint | Strict validation improvement; earliest tie; all training epochs still execute |
| Selection | Lowest mean validation CE across seeds, then freeze before test scoring |
| Capacity | Count allocated and supervised-path-active parameters; no equal-capacity claim |
| Compute | Measure seconds; equal updates are not equal wall time or FLOPs |
| Metrics | Accuracy and macro F1 on the same complete test set |
| Uncertainty | Paired seed differences; one graph/split is not independent datasets |
| Attribution | Observation, intervention, alternative explanation, narrow conclusion |
| Reproduction | New ACM comparison != named AIFB replay != full HGT CS experiment |

For messages 2, 8 (authors) and -4 (subject), R-GCN gives 1, uniform HGT gives 2, and weights (1,3,2)/6 give 3. These are scalar aggregation traces, not full model predictions.

Source anchors: [R-GCN Eq2–3](https://arxiv.org/abs/1703.06103v4) · [HGT §3](https://arxiv.org/abs/2003.01332v1).
'''
(R/'reference/architecture-comparison-contract.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Architecture comparison contract</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/architecture-comparison.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built lesson, two notebooks, prepared HTML and reference')
