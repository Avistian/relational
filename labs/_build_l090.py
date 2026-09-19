"""Build the checkpoint from canonical prose, live code, and measured artifacts."""
from _walkthrough_delivery import snapshot, finalize
snapshot(90)
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0090-gnn-checkpoint';TITLE='Q1 checkpoint: build and defend a GNN'
figs={'TRACE_FIG':('trace','Hand fixture: removing node 2 changes both available messages and augmented degrees.'),'ARCH_FIG':('architecture','GCN benchmark port and separate inductive extension: actual shapes, shared transforms, objectives and inference contexts.'),'RESULT_FIG':('results','Author measurements: 100 GCN initializations and three inductive runs; separate protocols, not a ranking.')}
paper=json.loads((LAB/'_paper_l090_results.json').read_text());ind=json.loads((LAB/'_inductive_l090_results.json').read_text())
def results():
 return f'''**Author-reference evidence, not the learner's current kernel output.**

| Experiment | Completed runs | Mean test accuracy ± sample SD | Status |
|---|---:|---:|---|
| Full GCN released-protocol port | {len(paper['runs'])} | {paper['mean']*100:.3f}% ± {paper['sample_sd']*100:.3f} pp | {paper['course_verdict']}: within frozen tolerance |
| Inductive Cora extension | {len(ind['runs'])} | {ind['mean']*100:.3f}% ± {ind['sample_sd']*100:.3f} pp | Executed; paper comparison INCOMPARABLE |

GCN standard error: **{paper['se']*100:.4f} percentage points**. Fresh CPU training took {paper['seconds']:.1f} seconds for the GCN lane and {ind['seconds']:.1f} seconds for the extension. Raw evidence: [GCN](../labs/_paper_l090_results.json), [inductive](../labs/_inductive_l090_results.json), [behavior checks](../labs/_verify_l090_results.json). A [fresh pinned environment](../labs/_clean_environment_l090_results.json) also completed both lanes; its 100 GCN scores and validation traces exactly matched the author run.'''
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text().replace('[[RESULTS]]',results())
 for tag,(name,caption) in figs.items():
  p=LAB/f'figures/l090/{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l090/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,mount,fallback in [('WARMUP','warmup','Write your three retrieval answers before continuing.'),('PREDICT','prediction','Predict: does changing node 2 affect the full-context message, the training-only message, or both?'),('BOUNDARY_WIDGET','boundary','Full-context message at x₂=20 is 10.314796; training-only remains 3. Explain before reading on.'),('TEACHBACK','teachback','Write a defense of the protocol, execution coverage, graph boundary and historical limitations.')]:
  s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{mount}"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 90 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','gnn-boundary-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0089-sampling-at-scale.html">Lesson 89</a> · <a href="../reference/gnn-checkpoint.html">Reference</a></nav><header><p>Year 3 · Quarter 1 · Lesson 090</p><h1>{TITLE}</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','gnn-boundary-viz','l090-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
checks={
'propagate':"s=normalized_support(sp.csr_matrix([[0,1,0],[1,0,1],[0,1,0]]))\nx=torch.tensor([[2.],[4.],[8.]])\nz=propagate(s,x,torch.ones(1,1))\nassert abs(float(z[1])-5.415816)<1e-5\ntorch.testing.assert_close(z,s.to_dense()@x)\nprint('CHECK: normalized sparse propagation agrees with independent dense arithmetic')",
'masked_objective':"z=torch.tensor([[2.,0.],[0.,2.],[1.,1.]],requires_grad=True);y=torch.tensor([0,1,0]);idx=torch.tensor([0]);w=torch.tensor([[2.]],requires_grad=True)\nl=masked_objective(z,y,idx,w);y[1:]=1-y[1:]\ntorch.testing.assert_close(l,masked_objective(z,y,idx,w));l.backward()\nassert torch.count_nonzero(z.grad[1:])==0\ntorch.testing.assert_close(w.grad,.0005*w)\nprint('CHECK: held-out labels cannot affect training loss; first-weight L2 gradient correct')",
'eligible_neighbors':"a=sp.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])\nns=eligible_neighbors(a,np.array([0,1]))\nassert [v.tolist() for v in ns]==[[1],[0],[]], 'Both endpoints must be eligible'\nprint('CHECK: excluded node cannot occur as a source or destination')",
'masked_mean':"v=torch.tensor([[[2.],[8.]],[[9.],[9.]]],requires_grad=True)\nz=masked_mean(v,torch.tensor([[True,True],[False,False]]))\ntorch.testing.assert_close(z,torch.tensor([[5.],[0.]]));z.sum().backward()\nassert torch.count_nonzero(v.grad[1])==0\nprint('CHECK: empty neighbor mean and gradient are zero')",
'verdict':"assert verdict(.815,100,False)=='INCOMPARABLE'\nassert verdict(.815,3,True)=='INCOMPLETE'\nassert verdict(.815,100,True)=='CLOSE'\nassert verdict(.9,100,True)=='FAIL'\nprint('CHECK: protocol and execution coverage precede numerical closeness')"}
descriptions={'propagate':'A · Propagate sparse messages. Transform features, then aggregate through the supplied support; preserve gradients.','masked_objective':'B · Restrict classification loss to selected nodes and add the released first-weight penalty.','eligible_neighbors':'C1 · Build the eligible graph. Remove every edge whose source or destination is unavailable.','masked_mean':'C2 · Average valid sampled messages; empty neighborhoods must give zero and no neighbor gradient.','verdict':'D · Return INCOMPARABLE, INCOMPLETE, CLOSE or FAIL in the correct priority order.'}
manifest=json.loads((LAB/'_sources_l078.json').read_text())
bootstrap="""# @colab-bootstrap — notebook is standalone; no repository model imports.
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','scipy==1.15.3','torch==2.8.0'])
import importlib.metadata as metadata
print({name:metadata.version(name) for name in ['numpy','scipy','torch']})
# Colab uses the portable pins; it is not claimed identical to the newer author environment.
"""
for solution in [False,True]:
 cells=[nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('# '+TITLE+'\n\n'+prose(True)),nbf.v4.new_markdown_cell('## PROVIDED · Pinned raw data manifest\nOnly the verified source pickles below are loaded. Missing data downloads from an immutable upstream revision. Original MIT attribution is retained in the reproduction contract.'),nbf.v4.new_code_cell('from pathlib import Path\nimport json\nLAB_ROOT=Path.cwd()/"l090-data"\nLAB_ROOT.mkdir(exist_ok=True)\nSOURCE_MANIFEST='+repr(manifest)+'\n(LAB_ROOT/"_sources_l078.json").write_text(json.dumps(SOURCE_MANIFEST))')]
 for filename in ['gcn_l082.py','checkpoint_l090.py']:
  src=(LAB/'relkit'/filename).read_text();lines=src.splitlines()
  cells.append(nbf.v4.new_markdown_cell('## PROVIDED · '+('Complete released-protocol GCN' if filename.startswith('gcn') else 'Complete inductive extension')+'\nDefinitions below are the implementation used by your training cells. Read each chunk before running it.'))
  for node in ast.parse(src).body:
   text='\n'.join(lines[node.lineno-1:node.end_lineno]);name=getattr(node,'name','')
   if isinstance(node,(ast.Import,ast.ImportFrom)) or isinstance(node,ast.Expr):
    if cells[-1].cell_type=='code' and cells[-1].metadata.get('imports'):cells[-1].source+='\n'+text
    else:cells.append(nbf.v4.new_code_cell(text,metadata={'imports':True}))
   elif name in checks:
    cells.append(nbf.v4.new_markdown_cell('### TODO · '+descriptions[name]))
    if not solution:
     signature=text[:text.index(':')+1]
     # These task signatures contain no type-annotation colons.
     text=signature+'\n    raise NotImplementedError("Implement this task before CHECK")'
    cells.append(nbf.v4.new_code_cell(text,metadata={'tags':['solution' if solution else 'todo'],'task':name}))
    cells.append(nbf.v4.new_code_cell('# CHECK\n'+checks[name],metadata={'tags':['check']}))
   else:
    cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+name.replace('_',' ')))
    cells.append(nbf.v4.new_code_cell(text))
 cells.append(nbf.v4.new_markdown_cell('## CHECK · Model-level isolation and live-function routing\nThis checks your actual model call, then runs one full-schedule GCN initialization and a three-epoch inductive diagnostic. These diagnostics do not satisfy the full benchmark gate.'))
 cells.append(nbf.v4.new_code_cell("""torch.set_num_threads(1)
assert GCN.forward.__globals__['propagate'] is propagate
assert train_cora.__globals__['masked_objective'] is masked_objective
assert SampledSAGE.sampled.__globals__['masked_mean'] is masked_mean
assert inductive_contexts.__globals__['eligible_neighbors'] is eligible_neighbors
a=sp.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])
ns=eligible_neighbors(a,np.array([0,1]));features=torch.randn(3,4)
m=SampledSAGE(4,2);roots=np.array([1])
before=m.sampled(features,roots,ns,np.random.default_rng(5))
features[2]=1e6
assert torch.allclose(before,m.sampled(features,roots,ns,np.random.default_rng(5)))
data=load_cora(LAB_ROOT)
smoke_gcn=train_cora(data,0)
smoke_inductive=train_inductive(data,0,epochs=3)
print({'gcn_smoke':smoke_gcn['test_accuracy'],'inductive_smoke':smoke_inductive['test_accuracy'],'coverage':'diagnostics only'})
"""))
 cells.append(nbf.v4.new_markdown_cell('## NEXT STEP · Full published experiment and inductive deliverable\nSet the switch to True after the short checks pass. This runs all 100 GCN initializations with the full schedule plus three complete inductive runs. CPU execution is sufficient (about four minutes in the author environment). No paid service is needed. Keep the frozen tolerance and record failures.'))
 cells.append(nbf.v4.new_code_cell("""RUN_FULL_REPRO = False
full_result=None
inductive_runs=[]
if RUN_FULL_REPRO:
    full_result=run_cora(100,LAB_ROOT)
    full_result['course_verdict']=verdict(full_result['mean'],len(full_result['runs']),True)
    full_result['historical_exact_parity']='INCOMPARABLE'
    inductive_runs=[train_inductive(data,seed,epochs=100) for seed in [0,1,2]]
    Path('l090-paper.json').write_text(json.dumps(full_result,indent=2))
    Path('l090-inductive.json').write_text(json.dumps(inductive_runs,indent=2))
    print({k:v for k,v in full_result.items() if k not in ['runs']})
"""))
 cells.append(nbf.v4.new_markdown_cell('## EXIT TICKET · Submit evidence and defend it\nWrite 150–250 words explaining the two graph-access contracts, a failed mutation, the seed distribution and the difference between course tolerance and historical parity. Attach the two raw result files. Leave the defense pending until you have written and reviewed it; author execution is not learner mastery.'))
 cells.append(nbf.v4.new_code_cell("""import platform,hashlib
execution_complete=full_result is not None and len(full_result['runs'])==100 and len(inductive_runs)==3
exit_ticket={'lesson':90,'execution_complete':execution_complete,
    'learner_mastery':'PENDING_WRITTEN_DEFENSE',
    'course_verdict':None if full_result is None else full_result['course_verdict'],
    'historical_parity':'INCOMPARABLE','gcn_smoke':smoke_gcn['test_accuracy'],
    'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
    'source_manifest_sha256':hashlib.sha256(json.dumps(SOURCE_MANIFEST,sort_keys=True).encode()).hexdigest()}
Path('l090-exit.json').write_text(json.dumps(exit_ticket,indent=2));print(exit_ticket)
"""))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=LAB/('solutions' if solution else '')/(SLUG+'.ipynb');path.parent.mkdir(exist_ok=True)
 if solution:
  for cell in nb.cells:
   if cell.cell_type=='code':cell.source=cell.source.replace('RUN_FULL_REPRO = False','RUN_FULL_REPRO = True')
  if path.exists():
   old=nbf.read(path,as_version=4)
   old_code={c.source:c for c in old.cells if c.cell_type=='code'}
   # Preserve execution evidence only if every code cell is unchanged.
   if all(c.source in old_code for c in nb.cells if c.cell_type=='code'):
    for cell in nb.cells:
     if cell.cell_type=='code':
      cell.outputs=old_code[cell.source].outputs
      cell.execution_count=old_code[cell.source].execution_count
 nbf.write(nb,path)
 if not solution:
  body,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(body)
print('Built lesson, student notebook, solution and portable preview')
finalize(90)
