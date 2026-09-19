"""Build L089 HTML, portable notebooks and prepared student preview from shared sources."""
from _walkthrough_delivery import snapshot, finalize
snapshot(89)
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0089-sampling-at-scale';TITLE='Sampling at scale: Cluster-GCN'
FIGS={'PARTITION_FIG':('partition','Synthetic path: the induced union restores edge 1–2; concatenating cut blocks would omit it.'),'SUPPORT_FIG':('support','Synthetic later-layer trace: changing the batch changes node 1’s enhanced message from 2.5 to 3.'),'ARCH_FIG':('architecture','Complete 2019 PPI release recipe: raw first-layer cache, four hidden stages, output logits and distinct evaluation contexts.'),'RESULT_FIG':('results','Fresh PPI teaching experiment: three seeds, ten passes and a hidden-state array estimate; not the paper target or peak-memory measurement.')}
def teaching():
 d=json.loads((LAB/'_teaching_l089_results.json').read_text());s='**Author-reference measurements, not your current kernel output.**\n\n| Arm | Test micro-F1, mean ± sample SD | Largest batch | Retained entries | Hidden-state proxy | Train time |\n|---|---|---|---|---|---|\n'
 for r in d['rows']:s+=f"| {r['mode']} | {r['mean']*100:.2f} ± {r['sample_sd']*100:.2f}% | {r['max_batch_nodes']:,} | {r['mean_edge_fraction']*100:.1f}% | {r['hidden_state_proxy_bytes']/2**20:.3f} MiB | {r['mean_train_seconds']:.2f} s |\n"
 return s+'\nSample SD describes three initialization/shuffle runs on one fixed split; it is not uncertainty across datasets. Times are CPU author measurements, averaged across seeds.'
def paper():
 d=json.loads((LAB/'_paper_l089_results.json').read_text())
 return '**Full paper run: '+d['status']+'.** '+d['reason']+' Historical result parity: **INCOMPARABLE**. Full-model one-update preflight and the completed teaching comparison are separate evidence.'
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 for tag,(name,caption) in FIGS.items():
  p=LAB/f'figures/l089/{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l089/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id,text in [('WARMUP','warmup','Write your three cold-retrieval answers before continuing.'),('PREDICT','prediction','Predict before revealing: how many edges belong in the induced union of C0 and C1?'),('PARTITION_WIDGET','cluster-sampling','Trace q=1,2,3 selected clusters on the six-node path; retained edge counts are1,3,5.'),('TEACHBACK','teachback','Explain the memory and gradient trade-off, including the cache and unequal update counts.')]:
  s=s.replace('[['+tag+']]',text if portable else f'<div id="{id}"></div>')
 s=s.replace('[[TEACHING]]',teaching()).replace('[[PAPER_STATUS]]',paper())
 if portable:
  s=s.replace('](0088-graph-classification.html)','](https://avistian.github.io/relational/lessons/0088-graph-classification.html)').replace('](0083-graphsage.html)','](https://avistian.github.io/relational/lessons/0083-graphsage.html)').replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 89 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','cluster-sampling'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0088-graph-classification.html">Lesson 88</a> · <a href="../reference/cluster-gcn.html">Reference</a></nav><header><p>Year 3 · Quarter 1 · Lesson 089</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a></aside>'
footer='</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','cluster-sampling-viz','l089-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
source=(LAB/'relkit/cluster_gcn_l089.py').read_text();nodes=ast.parse(source).body
checks={
'induced_batch':"a=sp.csr_matrix(np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]],dtype=np.float32))\nparts=[np.array([0,1]),np.array([2,3])]\nids,b=induced_batch(a,parts,[1,0])\nnp.testing.assert_array_equal(ids,[2,3,0,1]);assert b.nnz==6, 'Restore the crossing edge, in both directions'\nassert b[3,0]==1, 'Global edge1–2 must become local edge3–0'\nprint('CHECK: induced edges and ID order pass')",
'enhanced_support':"a=sp.csr_matrix(np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.float32))\ns=enhanced_support(a).toarray()\nnp.testing.assert_allclose(s[1],[1/3,2/3,1/3],rtol=1e-6)\nnp.testing.assert_allclose(enhanced_support(sp.csr_matrix((1,1))).toarray(),[[2.]])\nprint('CHECK: enhancement order and isolated node pass')",
'concat_message':"h=torch.tensor([[1.],[2.],[4.]],requires_grad=True)\ns=sparse_tensor(enhanced_support(sp.csr_matrix([[0,1,0],[1,0,1],[0,1,0]])))\ny=concat_message(h,s)\ntorch.testing.assert_close(y[1],torch.tensor([3.,2.]))\ny.sum().backward()\ntorch.testing.assert_close(h.grad,(s.to_dense().sum(0)+1)[:,None])\nprint('CHECK: neighbor/self order and gradients pass')"}
bootstrap="""# @colab-bootstrap · model, sampler, loader and trainer are fully inline below.
import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','torch==2.8.0','numpy==2.2.6','scipy==1.15.3','scikit-learn==1.7.1','pymetis==2025.2.2'])
import importlib.metadata as metadata
print({n:metadata.version(n) for n in ['torch','numpy','scipy','scikit-learn','pymetis']})
"""
teaching_code="""RUN_TEACHING = False
learner_results = []
if RUN_TEACHING:
    controlled = dict(data, a=data['a'].copy())
    controlled['a'].data[:] = 1  # same binary graph in all arms
    for mode,method,p,q in [('full','metis',1,1),('random','random',50,1),('cluster1','metis',50,1),('cluster5','metis',50,5)]:
        for seed in [1,2,3]:
            cfg=dict(preset('smoke'),epochs=10,num_parts=p,q=q,partition_method=method)
            record=train(controlled,cfg,seed,'cpu',output=f'l089-teaching/{mode}-seed{seed}',validate_every=10)
            record['mode']=mode
            learner_results.append(record)
    for mode in ['full','random','cluster1','cluster5']:
        scores=[r['test']['micro_f1'] for r in learner_results if r['mode']==mode]
        print(mode, 'mean', np.mean(scores), 'sample SD', np.std(scores,ddof=1))
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nTier B · full real PPI data. Three live TODOs; complete model/trainer inline. Default run is a two-epoch smoke demonstration. Full Table10 recipe is separately gated. Creating or executing this notebook does not establish learner mastery.'),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(s) for s in re.split(r'(?=^## )',prose(True),flags=re.M) if s.strip())
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED / TODO / CHECK · implementation\n\nRead the docstrings as contracts. Your three functions are used in live sampling and propagation. No hidden solution import replaces them. `train` uses `induced_batch` when q>1; the immediate smoke run uses q5 to exercise that path.'))
 imports='\n'.join(ast.get_source_segment(source,n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(imports+'\ntorch.set_num_threads(1)'))
 for n in nodes:
  if not isinstance(n,(ast.FunctionDef,ast.ClassDef)):continue
  code=ast.get_source_segment(source,n)
  if isinstance(n,ast.FunctionDef) and n.decorator_list:code='\n'.join('@'+ast.get_source_segment(source,d) for d in n.decorator_list)+'\n'+code
  if n.name in checks and not solution:code=code[:code.index('\n')]+'\n    '+repr(ast.get_docstring(n))+'\n    raise NotImplementedError("TODO: '+n.name+'")'
  cells.append(nbf.v4.new_markdown_cell(('TODO' if n.name in checks and not solution else 'PROVIDED')+' · `'+n.name+'`\n\n'+(ast.get_docstring(n) or '').split('\n\n')[0]))
  cells.append(nbf.v4.new_code_cell(code))
  if n.name in checks:cells.append(nbf.v4.new_code_cell(checks[n.name]))
 cells.append(nbf.v4.new_markdown_cell('## RUN · fresh real-data smoke\n\nThis loads the full hash-verified archive (about27MB compressed) and runs two width32 epochs. It tests all three TODOs but does not reproduce Table10. Cache uses original release loop weights.'))
 cells.append(nbf.v4.new_code_cell("data=load_ppi()\nsmoke=train(data,dict(preset('smoke'),q=5),seed=1,device='cpu')\nprint('Live smoke test micro-F1:',smoke['test']['micro_f1'])"))
 cells.append(nbf.v4.new_markdown_cell('## RUN · your controlled four-arm experiment\n\nFirst write predictions about edge retention and memory. Set `RUN_TEACHING=True` for all12 runs. The binary raw graph is held fixed; passes are matched but optimizer updates differ.'))
 cells.append(nbf.v4.new_code_cell(teaching_code))
 cells.append(nbf.v4.new_markdown_cell('## EXIT · runnable evidence and explanation\n\nComplete the three text fields with at least40 characters each. All12 completed run records are required. A smoke result alone does not pass.'))
 cells.append(nbf.v4.new_code_cell("explanations={'boundary_edge_and_gradient':'','memory_and_update_budget':'','historical_reproduction_gap':''}\nexpected={(m,s) for m in ['full','random','cluster1','cluster5'] for s in [1,2,3]}\ncoverage={(r['mode'],r['seed']) for r in learner_results if r['status']=='COMPLETE' and r['config']['epochs']==10}\nexit_artifact={'complete':coverage==expected and len(learner_results)==12 and all(len(v.strip())>=40 for v in explanations.values()),'results':learner_results,'explanations':explanations,'smoke':smoke['test']}\nPath('l089-exit.json').write_text(json.dumps(exit_artifact,indent=2))\nprint('EXIT complete:',exit_artifact['complete'])"))
 cells.append(nbf.v4.new_markdown_cell('## FULL REPRODUCTION · Table10 PPI\n\nThe full recipe is all400 epochs at width2048,5 layers. Choose a durable GPU runtime. Paper performance is cited, not established by running a smoke preset. The notebook preserves results under `l089-paper`; use a fresh directory to avoid overwriting evidence. Original TF1/METIS parity remains INCOMPARABLE.'))
 cells.append(nbf.v4.new_code_cell("RUN_PAPER_REPRO=False\nPAPER_PRESET='paper'\nif RUN_PAPER_REPRO:\n    device='cuda' if torch.cuda.is_available() else 'cpu'\n    out=Path('l089-paper')\n    if out.exists() and any(out.iterdir()): raise ValueError('Use a fresh output directory')\n    full_result=train(data,preset(PAPER_PRESET),seed=1,device=device,output=out)\n    print({'preset':PAPER_PRESET,'test':full_result['test'],'historical_parity':'INCOMPARABLE'})"))
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}})
 p=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb';nbf.write(nb,p)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
print('Built lesson, student, solution and portable preview')
finalize(89)
