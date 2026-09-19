"""Build lesson, portable notebooks and reference from canonical prose/source."""
import ast,base64,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from _figures_l081 import build as figures
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0081-mpnn-framework'
TITLE='MPNNs: message, aggregate, update, readout'
CAPTIONS={'trace':'Exact teaching trace; messages use old states and the isolated node receives a zero aggregate.','symmetry':'Move features, endpoints and graph IDs together. Node outputs permute; graph sums stay fixed.','architecture':'Complete sparse GG-NN reconstruction: bond matrices, shared recurrent updates and gated graph readout. This is not the edge-network/set2set variant.'}

def figure(name,portable=False):
    src=('data:image/png;base64,'+base64.b64encode((LAB/'figures/l081'/f'{name}.png').read_bytes()).decode()) if portable else '../labs/figures/l081/'+name+'.png'
    return f'<figure class="mpnn-figure"><small>Scroll horizontally on narrow screens to inspect the full figure.</small><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{CAPTIONS[name]}"></div><figcaption>{CAPTIONS[name]}</figcaption></figure>'


def prose(portable=False):
    s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
    for name in CAPTIONS:s=s.replace('[[FIG:'+name+']]',figure(name,portable))
    for name in ['warmup','predict','messages','reach','normalized','teachback']:
        fallback={'warmup':'**Cold recall:** write your answers in a markdown cell before reading further.','predict':'**Predict first:** compute B before reading the trace below.','messages':'**Intervention:** replace C=8 with C=20 in your live code.','reach':'**Predict:** after two rounds A changes from 3.75 to 5.25. Explain the route.','normalized':'**Predict:** why is the GCN weight not 1/degree(B)?','teachback':'**Write your teach-back before comparing with the reference.**'}[name]
        s=s.replace('[['+name.upper()+']]',fallback if portable else f'<div id="{name}"></div>')
    runs=[json.loads((LAB/'evidence/l081'/str(seed)/'result.json').read_text()) for seed in [81,82,83]]
    table='**Author-reference smoke evidence; not student output or paper-result reproduction.**\n\n| Split + initialization seed | Test MAE (Debye) | Updates | Status |\n|---|---:|---:|---|\n'
    for r in runs:table+=f'| {r["split_seed"]} | {r["test_mae_debye"]:.6f} | 40 | INCOMPARABLE |\n'
    import numpy as np
    values=[r['test_mae_debye'] for r in runs]
    table+=f'\nAcross these three smoke splits/runs: mean {np.mean(values):.6f}, sample SD {np.std(values,ddof=1):.6f} Debye. This is descriptive run variability, not dataset-level uncertainty.\n'
    s=s.replace('[[RESULTS]]',table)
    if portable:
        s=re.sub(r'\]\((00[0-9]{2}[^)]+)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
        s=s.replace('](../labs/','](https://avistian.github.io/relational/labs/').replace('](../reference/','](https://avistian.github.io/relational/reference/')
    return s

CHECKS={
'aggregate':"""m=torch.tensor([[2.],[8.],[8.]])
torch.testing.assert_close(aggregate(m,torch.tensor([1,1,1]),4,'mean'),torch.tensor([[0.],[6.],[0.],[0.]]))
assert aggregate(torch.empty(0,2),torch.empty(0,dtype=torch.long),3).shape==(3,2)
print('CHECK: multiple destinations, multiplicity and empty edges')""",
'mean_step':"""h=torch.tensor([[2.],[4.],[8.],[10.]])
e=torch.tensor([[0,1,1,2],[1,0,2,1]])
torch.testing.assert_close(mean_step(h,e),torch.tensor([[3.],[4.5],[6.],[5.]]))
p=torch.tensor([2,0,3,1]);inverse=torch.argsort(p)
torch.testing.assert_close(mean_step(h[p],inverse[e]),mean_step(h,e)[p])
print('CHECK: synchronous trace and node relabeling')""",
'graph_sum':"""torch.testing.assert_close(graph_sum(torch.tensor([[2.],[4.],[8.],[10.]]),torch.tensor([0,0,0,1])),torch.tensor([[14.],[10.]]))
print('CHECK: graph membership prevents cross-molecule pooling')"""}

DEMO="""# PROVIDED: recover the mean operator with the generic interface.
h=torch.tensor([[2.],[4.],[8.],[10.]])
e=torch.tensor([[0,1,1,2],[1,0,2,1]])
batch=torch.tensor([0,0,0,1])
generic=GenericMPNN(lambda hv,hw,edge:hw,lambda h,m:.5*h+.5*m,graph_sum,'mean')
node_out,graph_out=generic(h,e,None,batch)
torch.testing.assert_close(node_out,mean_step(h,e))
# GCN: add each self-loop once, then use symmetric degree normalization.
loops=torch.arange(len(h));ee=torch.cat([e,torch.stack([loops,loops])],1)
src,dst=ee;degree=torch.bincount(dst,minlength=len(h)).float()
weights=(degree[src]*degree[dst]).rsqrt()[:,None]
gcn=GenericMPNN(lambda hv,hw,w:hw*w,lambda h,m:m.relu(),graph_sum)
gcn_h,_=gcn(h,ee,weights,batch)
A=torch.zeros(4,4);A[dst,src]=1
S=degree.rsqrt()[:,None]*A*degree.rsqrt()[None,:]
torch.testing.assert_close(gcn_h,(S@h).relu())
# GAT one-head illustration W=1: softmax is grouped by DESTINATION.
score=torch.nn.functional.leaky_relu(h[dst,0]+h[src,0],negative_slope=.2)
alpha=torch.empty_like(score)
for v in range(len(h)):
    mask=dst==v;alpha[mask]=torch.softmax(score[mask],dim=0)
gat=GenericMPNN(lambda hv,hw,a:hw*a,lambda h,m:m,graph_sum)
gat_h,_=gat(h,ee,alpha[:,None],batch)
assert all(torch.isclose(alpha[dst==v].sum(),torch.tensor(1.)) for v in range(len(h)))
print('Mean nodes:',node_out.flatten().tolist(),'graph sums:',graph_out.flatten().tolist())
print('GCN B:',float(gcn_h[1]),'GAT B:',float(gat_h[1]))
"""
MODEL_CHECK="""# CHECK: the live student reducer participates in the molecular model.
torch.manual_seed(81);torch.set_num_threads(1)
x=torch.randn(4,13);edge_type=torch.tensor([0,0,2,2])
model=SparseGGNN(width=16,steps=3,readout_width=24)
calls=[];student_aggregate=aggregate
def observed_aggregate(*args,**kwargs):
    calls.append(1);return student_aggregate(*args,**kwargs)
aggregate=observed_aggregate
y=model(x,e,edge_type,batch)
aggregate=student_aggregate
assert len(calls)==7, 'Two message reductions per round plus the graph readout'
p=torch.tensor([2,0,3,1]);inverse=torch.argsort(p)
y_perm=model(x[p],inverse[e],edge_type,batch[p])
torch.testing.assert_close(y,y_perm,atol=1e-6,rtol=1e-5)
y.square().sum().backward()
assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
print('CHECK: live reducer, graph invariance, finite gradients')
"""
EXIT="""# EXIT: attach this artifact and your written explanation for feedback.
import json
from pathlib import Path
exit_artifact={'lesson':81,'mean_trace':node_out.flatten().tolist(),'graph_outputs':graph_out.flatten().tolist(),'relabeling_max_error':float((y-y_perm).abs().max().detach()),'paper_target_mae_debye':.394,'paper_reproduction':'NOT_RUN','explanation_required':['equivariance versus invariance','why edge remapping matters','three historical protocol gaps']}
Path('l081-exit.json').write_text(json.dumps(exit_artifact,indent=2))
print(json.dumps(exit_artifact,indent=2))
"""

def inline_cells(file,solution=False):
    source=file.read_text();cells=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant):continue
        if isinstance(node,ast.ImportFrom) and node.module and node.module.startswith('relkit.'):continue
        code=ast.get_source_segment(source,node)
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            name=node.name;cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if name in CHECKS else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(node) or 'Trace the inputs, intermediate tensors and returned values.')))
            if name in CHECKS and not solution:code=code.split('\n',1)[0]+'\n    # TODO: implement the contract above using the inputs.\n    raise NotImplementedError("'+name+'")'
        cells.append(nbf.v4.new_code_cell(code))
        if isinstance(node,ast.FunctionDef) and node.name in CHECKS:
            cells.append(nbf.v4.new_markdown_cell('### CHECK · predict the outcome first'))
            cells.append(nbf.v4.new_code_cell(CHECKS[node.name]))
    return cells


def build():
    from _walkthrough_delivery import snapshot, finalize
    snapshot(81)
    figures();s=prose()
    head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','message-passing-viz','mpnn-lesson'])+'</head><body><article>'
    head+=f'<nav><a href="../index.html">Course</a> · <a href="0080-year-2-exit-exam.html">Lesson 80</a></nav><header><p>Year 3 · Quarter 1 · Lesson 081</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/l081-reproduction.md">Reproduce</a></aside>'
    scripts=''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','message-passing-viz','teachback','l081-mpnn'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(s)+'</article>'+scripts+'</body></html>')
    for solution in [False,True]:
        cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nSelf-contained student lab: implement routing, verify symmetry, then inspect the full molecular reconstruction. Core exercises need PyTorch; the optional QM9 lane also uses RDKit and PyG.'),nbf.v4.new_code_cell("# @colab-bootstrap\nimport sys,subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch-geometric==2.6.1','rdkit==2025.9.6'])\nimport torch\nprint('Runtime torch:',torch.__version__)"),nbf.v4.new_markdown_cell(prose(True))]
        cells+=inline_cells(LAB/'relkit/mpnn_l081.py',solution)
        cells+=[nbf.v4.new_markdown_cell('## PROVIDED + CHECK · instantiate the framework\n\nThe GAT example is a one-head forward illustration with fixed scalar weights, not a trained GAT benchmark.'),nbf.v4.new_code_cell(DEMO),nbf.v4.new_code_cell(MODEL_CHECK),nbf.v4.new_markdown_cell('## EXIT · return an artifact, not a completion claim\n\nExplain your results in your own words.'),nbf.v4.new_code_cell(EXIT),nbf.v4.new_markdown_cell('## NEXT STEP · complete named-target reconstruction\n\nThe following loader, optimizer and selection code are visible and run your live model. Historical split/search details are missing. All presets remain INCOMPARABLE. Defaults below run a fresh short QM9 smoke; use a new output directory for each run. Download is about 44 MB. `paper-budget` requests 50 × 3 million updates and is not an interactive lesson task.')]
        cells+=inline_cells(LAB/'relkit/qm9_l081.py',solution)
        cells.append(nbf.v4.new_code_cell("import tempfile\noutput=tempfile.mkdtemp(prefix='l081-notebook-')\nresult=run_reconstruction('data/cache/l081',output,preset='smoke',seed=81)\nprint({k:result[k] for k in ['status','test_mae_debye','full_historical_reproduction']})"))
        cells.append(nbf.v4.new_code_cell("# Optional GPU reconstruction: never substitutes for historical protocol recovery.\nRUN_PAPER_REPRO=False\nif RUN_PAPER_REPRO:\n    assert torch.cuda.is_available(), 'Choose a GPU runtime for this search'\n    result=run_reconstruction('data/cache/l081',tempfile.mkdtemp(prefix='l081-closer-'),preset='closer',seed=81,device='cuda')"))
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        dest=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb';dest.parent.mkdir(exist_ok=True);nbf.write(nb,dest)
        if not solution:
            page,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(page)
    ref='''# MPNN contract card

| Stage | Inputs → outputs | Required check |
|---|---|---|
| Message | destination state, source state, edge → [E,D] | Edge direction |
| Aggregate | messages, destinations → [N,D] | Order independence; empty neighborhood |
| Update | old state, aggregate → new state | Synchronous old-state snapshot |
| Readout | final states, graph IDs → graph output | Invariance and graph isolation |

Node relabeling: `x_new=x[p]`, `edges_new=argsort(p)[edges]`, `batch_new=batch[p]`. Node predictions reorder; graph predictions stay fixed.

GCN: add self-loops once; sum messages weighted by `1/sqrt(degree_source*degree_destination)`. GAT: normalize attention within each destination neighborhood before summing.

Sparse GG-NN: bond matrices → two sums → bias-free recurrent gate → concatenate original features → gated per-node scalar → graph sum. Reset acts before recurrent multiplication in the released implementation.

Paper target: supplement Table 3 GG-NN μ, 3.94 × 0.1 = 0.394 Debye MAE. A matching scalar is insufficient: data, split, initialization, architecture, objective, schedule, search, selection and metric must align. Historical reader/trainer/search configurations are missing; local reconstruction remains INCOMPARABLE.

[Lesson](../lessons/0081-mpnn-framework.html) · [Protocol](../labs/l081-reproduction.md) · [Primary paper](https://proceedings.mlr.press/v70/gilmer17a.html)
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MPNN contract card</title><link rel="stylesheet" href="../assets/lesson.css"><article>'+render(ref)+'</article></html>')
    print('Built lesson, student, solution, prepared HTML and reference')
    finalize(81)
if __name__=='__main__':build()
