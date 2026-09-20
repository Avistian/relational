"""Build HTML, portable student/solution notebooks and preview from canonical sources."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0091-r-gcn';TITLE='R-GCN: make the relation change the message'
manifest=json.loads((LAB/'_sources_l091.json').read_text())
figs={'TRACE_FIG':('trace','Per-relation means produce 10 − 8 + 1 = 3. Moving one edge changes both denominators and yields 12.'),'BASIS_FIG':('basis','Two learned bases reconstruct a relation matrix before applying its message transformation. Sharing differs from block sparsity.'),'ARCH_FIG':('architecture','Full featureless AIFB R-GCN: typed supports, two relation-specific layers, masked supervision and transductive prediction.'),'RESULT_FIG':('results','All ten full-schedule initialization results. The paper target and modern port have explicit historical differences.')}
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 p=json.loads((LAB/'_paper_l091_results.json').read_text());t=json.loads((LAB/'_teaching_l091_results.json').read_text())
 table=f'''**Executed author-reference evidence.**

| Experiment | Complete runs × epochs | Mean test accuracy | Sample SD |
|---|---:|---:|---:|
| AIFB unrestricted release port | 10 × 50 | {100*p['mean']:.2f}% | {100*p['sample_sd']:.2f} pp |
| Four-basis teaching extension | 3 × 50 | {100*t['mean']:.2f}% | {100*t['sample_sd']:.2f} pp |

The full port is **{100*(p['mean']-.9583):+.2f} percentage points** from the published95.83%. Every seed's correct count and all 36 predictions are retained in the raw results. The seed SD describes initialization variability on this fixed split, not uncertainty across datasets. The basis extension is separate and untuned.

The standalone solution repeated all thirteen runs from a fresh data download and exactly matched the author’s ten-run predictions and loss traces. A separate clean environment with the portable dependency pins also completed all thirteen runs and obtained the same target mean.

[Raw ten-run results](../labs/_paper_l091_results.json) · [Basis extension](../labs/_teaching_l091_results.json) · [Arithmetic and gradient checks](../labs/_verify_l091_results.json) · [Data and pruning audit](../labs/_audit_l091_results.json)'''.replace('published95','published 95').replace('either34','either 34').replace('or35','or 35').replace('of36','of 36')
 s=s.replace('[[RESULTS]]',table)
 for tag,(name,caption) in figs.items():
  p=LAB/f'figures/l091/{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l091/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id_,fallback in [('WARMUP','warmup','**Cold retrieval:** write the three answers below before continuing.'),('PREDICT','prediction','**Predict:** can changing only edge roles alter the typed or untyped aggregate? Commit your answer first.'),('TYPED_WIDGET','typed','**Intervention:** move A from buys to returns. Compute the two changed means before checking the answer.'),('TEACHBACK','teachback','**Teach-back:** explain role-specific weights, per-relation normalization, basis sharing and the AIFB evidence boundary in your own words.')]:
  s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 91 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','hetero-graph-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0090-gnn-checkpoint.html">Lesson 90</a> · <a href="../reference/r-gcn.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 091</p><h1>{TITLE}</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','hetero-graph-viz','l091-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
checks={
'compose_weights':"v=torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]],requires_grad=True)\na=torch.tensor([[2.,-1.]],requires_grad=True)\nw=compose_weights(v,a)\ntorch.testing.assert_close(w[0],torch.tensor([[2.,-1.],[-1.,2.]]))\ntorch.testing.assert_close(torch.tensor([[3.,1.]])@w[0],torch.tensor([[5.,-1.]]))\nw.square().sum().backward();assert v.grad.abs().sum()>0 and a.grad.abs().sum()>0\nprint('CHECK: basis arithmetic and gradients pass')",
'relation_sum':"a=sp.csr_matrix(([1.,1.],([0,0],[1,2])),shape=(3,3))\nb=sp.csr_matrix(([1.],([0],[2])),shape=(3,3))\ns=normalize_relations([a,b,sp.eye(3)])\nx=torch.tensor([[1.],[2.],[8.]])\nw=torch.tensor([[[2.]],[[-1.]],[[1.]]],requires_grad=True)\ny=relation_sum(s,x,w)\ntorch.testing.assert_close(y,torch.tensor([[3.],[2.],[8.]]))\ny.sum().backward();torch.testing.assert_close(w.grad.flatten(),torch.tensor([5.,8.,11.]))\nprint('CHECK: independent relation means, self path and gradients pass')",
'masked_loss':"z=torch.tensor([[2.,0.],[0.,2.],[1.,1.]],requires_grad=True)\ny=torch.tensor([0,1,0]);idx=torch.tensor([0]);loss=masked_loss(z,y,idx)\ny[1:]=1-y[1:];torch.testing.assert_close(loss,masked_loss(z,y,idx))\nloss.backward();assert torch.count_nonzero(z.grad[1:])==0\nprint('CHECK: held-out label intervention and gradient isolation pass')"}
desc={'compose_weights':'A · Reconstruct each relation matrix with a linear combination of bases. Do not normalize coefficients or activate the contributions.','relation_sum':'B · Apply each normalized support and its matching weight, then sum. The self support is already included.','masked_loss':'C · Average cross-entropy over training indices only. Held-out labels must not affect gradients.'}
bootstrap="""# @colab-bootstrap — all implementation definitions are visible below.
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','scipy==1.15.3','torch==2.8.0','rdflib==7.1.4'])
import importlib.metadata as metadata
print({name:metadata.version(name) for name in ['numpy','scipy','torch','rdflib']})
"""
source=(LAB/'relkit/rgcn_l091.py').read_text();lines=source.splitlines()
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\n'+prose(True)),nbf.v4.new_code_cell(bootstrap),nbf.v4.new_markdown_cell('## PROVIDED · Source/data identity\nArchive bytes and original split files are verified before parsing. Model code below is the same visible modern port used for author execution. Author-source attribution: tkipf/relational-gcn, MIT, revision '+manifest['source_revision']+'.'),nbf.v4.new_code_cell('SOURCE_MANIFEST='+repr(manifest))]
 for node in ast.parse(source).body:
  code='\n'.join(lines[node.lineno-1:node.end_lineno]);name=getattr(node,'name','')
  if name in checks:
   cells.append(nbf.v4.new_markdown_cell('### TODO · '+desc[name]))
   if not solution:code=code[:code.index(':')+1]+'\n    raise NotImplementedError("Complete this TODO before CHECK")'
   cells.append(nbf.v4.new_code_cell(code,metadata={'tags':['solution' if solution else 'todo'],'task':name}))
   cells.append(nbf.v4.new_code_cell('# CHECK\n'+checks[name],metadata={'tags':['check']}))
  else:
   if name:cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+name.replace('_',' ')+'\nRead the dimensions and information boundary before running this definition.'))
   cells.append(nbf.v4.new_code_cell(code))
 cells += [nbf.v4.new_markdown_cell('## CHECK · Real-data diagnostic and live-function routing\nThis three-update run verifies data loading and actual model composition. It is not the full target. The basis model must use your basis function and message sum. Do not claim paper reproduction from this cell.'),nbf.v4.new_code_cell("""torch.set_num_threads(1)
DATA_ROOT=Path.cwd()/'l091-data'
data=load_aifb(DATA_ROOT,SOURCE_MANIFEST)
assert RGCN.forward.__globals__['compose_weights'] is compose_weights
assert RGCN.forward.__globals__['relation_sum'] is relation_sum
assert train_aifb.__globals__['masked_loss'] is masked_loss
smoke=train_aifb(data,100,epochs=3,bases=4)
print({'data':{k:data[-1][k] for k in ['nodes','raw_relations','train','test']},'diagnostic_accuracy':smoke['test_accuracy'],'coverage':'three-update diagnostic only'})
"""),nbf.v4.new_markdown_cell('## RUN · Full target and separate basis extension\nEnable the switch after CHECKs pass. Ten seeds ×50 updates on the full graph match the released AIFB recipe. Three further runs with B=4 exercise basis sharing separately. Preserve every seed and raw prediction. Author machine runtime is roughly two minutes; other runtimes vary.'),nbf.v4.new_code_cell("""RUN_FULL_REPRO = False
paper_result=None
basis_result=None
if RUN_FULL_REPRO:
    paper_result=run_aifb(DATA_ROOT,SOURCE_MANIFEST,seeds=range(10),bases=0)
    basis_result=run_aifb(DATA_ROOT,SOURCE_MANIFEST,seeds=range(3),bases=4)
    Path('l091-paper.json').write_text(json.dumps(paper_result,indent=2))
    Path('l091-basis.json').write_text(json.dumps(basis_result,indent=2))
    print({'target_mean':paper_result['mean'],'sample_sd':paper_result['sample_sd'],
           'gap_pp':paper_result['gap_percentage_points'],'basis_mean':basis_result['mean'],
           'historical_exact_parity':'INCOMPARABLE'})
"""),nbf.v4.new_markdown_cell('## EXIT · Submit the evidence and your defense\nIn a new markdown cell, write150–250 words: derive the changed edge example, compare basis and block restrictions, explain why the target uses B=0, and distinguish full execution from exact historical identity. Attach both JSON files. Ask the teacher to challenge your evidence boundary.'),nbf.v4.new_code_cell("""exit_ticket={'lesson':91,'full_execution_complete':paper_result is not None and len(paper_result['runs'])==10 and basis_result is not None and len(basis_result['runs'])==3,
 'historical_exact_parity':'INCOMPARABLE','learner_mastery':'PENDING_WRITTEN_DEFENSE',
 'source_revision':SOURCE_MANIFEST['source_revision'],'data_sha256':SOURCE_MANIFEST['data']['sha256']}
Path('l091-exit.json').write_text(json.dumps(exit_ticket,indent=2));print(exit_ticket)
""")]
 if solution:
  for c in cells:
   if c.cell_type=='code':c.source=c.source.replace('RUN_FULL_REPRO = False','RUN_FULL_REPRO = True')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=LAB/('solutions' if solution else '')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(nb.cells):c.id=f'l091-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
print('Built lesson, inline student/solution notebooks and portable preview')
