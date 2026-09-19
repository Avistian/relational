"""Build lesson, portable source-visible notebooks and compact reference."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0086-pyg-fundamentals';TITLE='PyG fundamentals: preserve the graph computation'
figdir=LAB/'figures/l086';figdir.mkdir(parents=True,exist_ok=True)
fig,ax=plt.subplots(figsize=(12,5),facecolor='#f7f5ef');ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
ax.text(.3,4.65,'ONE OPERATOR · TWO IMPLEMENTATIONS',fontsize=17,weight='bold',color='#153b4e')
boxes=[(.3,2.7,'INPUT','X [N,F] · W [F,C]\nedge_index [2,E]\nA—B—C; D isolated'),(4.35,2.7,'EDGE MESSAGES','Z = XW [N,C]\nZ_j [E+N,C]\nm_ji = Z_j / √(d_j d_i)'),(8.4,2.7,'NODE OUTPUT','sum at receiver i\nH [N,C]\nB = 5.415816 if W=1')]
for x,y,h,t in boxes:
 ax.add_patch(FancyBboxPatch((x,y),3.3,1.55,boxstyle='round,pad=0.12',facecolor='white',edgecolor='#438581',linewidth=2));ax.text(x+.12,y+1.22,h,fontsize=12,weight='bold',color='#23675e');ax.text(x+.12,y+.91,t,fontsize=11,va='top',linespacing=1.5)
for x in [3.7,7.75]:ax.annotate('',xy=(x+.5,3.4),xytext=(x,3.4),arrowprops={'arrowstyle':'->','lw':2,'color':'#d47736'})
ax.text(.4,2.08,'Dense oracle: S @ X @ W       ↔       PyG: transform → message → sum → update',fontsize=13,weight='bold',color='#153b4e')
ax.text(.4,1.35,'TRAINING EXTENSION    X → GCN(16) → ReLU → dropout(0.5) → GCN(7) → logits',fontsize=11,color='#153b4e')
ax.text(.4,.85,'Training mask: cross-entropy + Adam decay   |   Validation: choose state   |   Test: score once',fontsize=10)
ax.text(.4,.35,'Weights shared across nodes, separate across layers. Evaluation disables dropout. Full graph preserves degrees.',fontsize=10)
fig.savefig(figdir/'architecture.png',dpi=160,bbox_inches='tight');plt.close(fig)
caption='GCN computation trace and the distinct historical PyG training extension; shape C denotes output channels.'

def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 src='data:image/png;base64,'+base64.b64encode((figdir/'architecture.png').read_bytes()).decode() if portable else '../labs/figures/l086/architecture.png'
 s=s.replace('[[ARCH]]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 s=s.replace('[[VIZ]]','Predict B for C=20: the answer is 10.314796. The lesson page supplies an interactive control.' if portable else '<div id="normalized"></div>')
 p=LAB/'_paper_l086_results.json'
 if p.exists():
  r=json.loads(p.read_text());result=f"**Fresh author execution:** {len(r['runs'])} full-schedule runs; mean test accuracy **{100*r['mean']:.3f}%**, sample SD **{100*r['sample_sd']:.3f} percentage points**. These runs share one fixed graph and split; the SD measures initialization variation, not uncertainty across datasets. See [machine-readable results](../labs/_paper_l086_results.json)."
 else:result='Full author run is in progress. No final benchmark score claimed yet.'
 s=s.replace('[[RESULTS]]',result)
 if portable:
  s=s.replace('](0085-over-smoothing.html)','](https://avistian.github.io/relational/lessons/0085-over-smoothing.html)')
  s=re.sub(r'\]\(\.\./','](https://avistian.github.io/relational/',s)
 return s
checks={
'weighted_messages':"torch.testing.assert_close(weighted_messages(torch.tensor([[2.,4.],[3.,6.]]),torch.tensor([.5,2.])),torch.tensor([[1.,2.],[6.,12.]]))\nprint('CHECK: coefficient broadcasts across channels')",
'seed_loss':"b=Data(y=torch.tensor([1,0,1]));b.batch_size=1\nz=torch.tensor([[1.,2.],[3.,4.],[5.,6.]],requires_grad=True)\nl=seed_loss(z,b);l.backward();assert z.grad[1:].abs().sum()==0 and z.grad[0].abs().sum()>0\nb.y[1:]=1-b.y[1:];torch.testing.assert_close(l,seed_loss(z,b))\nprint('CHECK: context labels are excluded')",
'global_edges':"b=Data(edge_index=torch.tensor([[0,2],[1,0]]));b.n_id=torch.tensor([7,3,9])\nassert global_edges(b).tolist()==[[7,9],[3,7]]\nprint('CHECK: both endpoints remap')"}

def inline(source,solution,selected=None):
 cells=[]
 for node in ast.parse(source).body:
  if isinstance(node,ast.Expr):continue
  if selected is not None and isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in selected:continue
  code=ast.get_source_segment(source,node)
  if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
   cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in checks else '### PROVIDED · ')+node.name+'\n\n'+(ast.get_docstring(node) or 'Inspect the computation.')))
   if node.name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Complete '+node.name+'")'
  cells.append(nbf.v4.new_code_cell(code))
  if isinstance(node,ast.FunctionDef) and node.name in checks:cells.append(nbf.v4.new_code_cell(checks[node.name]))
 return cells
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','message-passing-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0085-over-smoothing.html">Lesson 85</a></nav><header><p>Year 3 · Quarter 1 · Lesson 086</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/solutions/{SLUG}.ipynb">Solution</a></aside>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+'</article><script src="../assets/message-passing-viz.js"></script><script src="../assets/l086-pyg.js"></script></body></html>')
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nThree live TODOs. The one-seed default runs the full training schedule; RUN_FULL=True executes all100 seeds. Author results in the prose are separate from your kernel.'),nbf.v4.new_markdown_cell('## Runtime\n\nUse the exact environment commands in [the reproduction guide](https://avistian.github.io/relational/labs/l086-reproduction.md). NeighborLoader requires pyg-lib or torch-sparse compiled for your installed torch. Do not silently replace the real loader with a mock.'),nbf.v4.new_code_cell("import torch, torch_geometric\nimport torch_geometric.typing as pyg_typing\nprint('torch',torch.__version__,'PyG',torch_geometric.__version__)\nassert pyg_typing.WITH_PYG_LIB or pyg_typing.WITH_TORCH_SPARSE, 'Install the compiled sampling backend; see reproduction guide'\ntorch.set_num_threads(1)")]
 cells += [nbf.v4.new_markdown_cell(p) for p in re.split(r'(?=^## )',prose(True),flags=re.M) if p.strip()]
 cells += [nbf.v4.new_markdown_cell('## Attribution\n\nHistorical reference files are archived at sources/l086; adapted under MIT:\n\n```text\n'+(LAB/'sources/l086/LICENSE').read_text()+'\n```')]
 cells+=inline((LAB/'relkit/pyg_l086.py').read_text(),solution)
 cells.append(nbf.v4.new_markdown_cell('## CHECK · independent dense operator and gradients\n\nPredict D’s output before running. It must retain its own transformed feature.'))
 cells.append(nbf.v4.new_code_cell("torch.manual_seed(86)\nd=toy_data();c=TraceGCN(1,2).double();x=d.x.double().requires_grad_()\na=torch.zeros(4,4,dtype=torch.float64);a[d.edge_index[1],d.edge_index[0]]=1\nt=a+torch.eye(4,dtype=torch.float64);q=t.sum(1).rsqrt();s=q[:,None]*t*q[None,:]\ny=c(x,d.edge_index);expected=s@x@c.weight\ntorch.testing.assert_close(y,expected,atol=1e-12,rtol=1e-12)\ng1=torch.autograd.grad(y.square().sum(),(x,c.weight),retain_graph=True)\ng2=torch.autograd.grad(expected.square().sum(),(x,c.weight))\nfor a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)\nprint('CHECK: independent outputs and gradients')\nsampling=loader_check();print(sampling)\nb=Batch.from_data_list([d,d]);torch.testing.assert_close(c(b.x.double(),b.edge_index),y.repeat(2,1))\nh=HeteroData();h['author'].x=torch.ones(2,3);h['paper'].x=torch.ones(3,4)\nh['author','writes','paper'].edge_index=torch.tensor([[0,1],[1,2]])\nassert h.validate(raise_on_error=True)"))
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · full Cora loader\n\nData manifest embedded; download only pinned bytes and verify SHA-256 before deserialization. No repository imports required.'))
 cells+=inline((LAB/'relkit/gcn_l082.py').read_text(),True,{'as_sparse','normalized_support','load_cora'})
 manifest=json.loads((LAB/'_sources_l078.json').read_text())
 cells.append(nbf.v4.new_code_cell('manifest='+repr(manifest)+"\nroot=Path.cwd()/'l086-cora-data';root.mkdir(exist_ok=True)\np=root/'_sources_l078.json'\nif p.exists():assert json.loads(p.read_text())==manifest\nelse:p.write_text(json.dumps(manifest,indent=2))\nloaded=load_cora(root);data=as_data(loaded)\nassert data.x.shape==(2708,1433)\nassert [int(data[k].sum()) for k in ['train_mask','val_mask','test_mask']]==[140,500,1000]\nc=TraceGCN(1433,16)\nwith torch.no_grad():\n actual=c(data.x,data.edge_index)\n expected=torch.sparse.mm(loaded[1],torch.sparse.mm(loaded[0],c.weight))\n torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)\nprint('CHECK: full Cora L082 operator parity')"))
 cells.append(nbf.v4.new_code_cell("RUN_FULL=False\nseeds=range(100) if RUN_FULL else range(1)\nruns=[]\nfor seed in seeds:\n row=train_citation(data,seed);runs.append(row);print(seed,row['epochs'],row['test_accuracy'])\nacc=np.array([r['test_accuracy'] for r in runs])\nresult={'status':'FULL_RECONSTRUCTION' if RUN_FULL else 'ONE_SEED_TEACHING_RUN','runs':runs,'mean':float(acc.mean()),'sample_sd':float(acc.std(ddof=1)) if len(acc)>1 else None,'sampling':sampling,'output_gradient_tolerance':1e-12,'historical_parity':'INCOMPARABLE'}\nPath('l086-exit.json').write_text(json.dumps(result,indent=2))\nprint(result['status'],result['mean'])"))
 cells.append(nbf.v4.new_markdown_cell('## EXIT · explain the boundary\n\nAdd your explanation of sampled-degree mismatch to l086-exit.json. Change a context label and show unchanged seed loss; change a context feature and explain why logits can change. Tomorrow, reconstruct the ID mapping without looking.'))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=LAB/('solutions/' if solution else '')/(SLUG+'.ipynb');nbf.write(nb,path)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
ref='''# PyG: graph computation contracts

| Object | Contract |
|---|---|
| Data.edge_index | [2,E], source row then destination row |
| MessagePassing | transform → lift x_j → message → reduce → update |
| GCN | add loops before degrees; sum symmetrically weighted messages |
| Batch | disjoint union; batch vector identifies original graph |
| NeighborLoader | seed nodes first; context nodes supply messages |
| n_id | local node position → original node ID |
| HeteroData | IDs belong to node types; relation stores index both types |

Use `batch.n_id[batch.edge_index]` to recover both endpoints. Use only the first `batch.batch_size` labels for seed supervision. Do not recompute sampled degrees and claim exact full-graph GCN equivalence.

For an implementation port compare outputs and input/weight gradients with identical parameters. For a paper reproduction also align data, preprocessing, architecture, optimization, selection, splits, seeds and metric.

[Lesson 86](../lessons/0086-pyg-fundamentals.html) · [Protocol](../labs/l086-reproduction.md) · [Primary paper](https://arxiv.org/abs/1903.02428)
'''
(ROOT/'reference/pyg-fundamentals.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PyG reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built L086 lesson, student, solution, reference and figure')
