"""Build L085 from markdown, canonical Python and measured evidence."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0085-over-smoothing';TITLE='Over-smoothing: when more neighbors erase distinctions'
CAPTIONS={'trace':'A three-node arithmetic trace: receiver B combines sender values with symmetric degree weights.','architecture':'Full Figure 2 setup: identity features, repeated GCN layers, two untrained output coordinates, labels only for plotting.','karate':'Seed 0, chosen in advance; all five depths on the complete karate graph, with identical axes.','mixing':'Fixed linear propagation and random nonlinear networks are separate experiments. Shading reports initialization variation.','depth':'Cora extension: all ten seeds at each depth; error bars are sample SD on one fixed split, not dataset uncertainty.'}
CHECKS={
'smooth':"s=torch.tensor([[.5,.5],[.5,.5]],dtype=torch.float64)\nh=torch.tensor([[0.],[4.]],dtype=torch.float64)\ntorch.testing.assert_close(smooth(s,h,0),h)\ntorch.testing.assert_close(smooth(s,h,2),torch.tensor([[2.],[2.]],dtype=torch.float64))\nassert h[0]==0, 'Do not mutate the input'\nprint('CHECK: simultaneous propagation and depth zero')",
'collapse_metrics':"h=torch.tensor([[1.,0.],[0.,1.]],dtype=torch.float64);d=torch.ones(2,dtype=torch.float64)\nm=collapse_metrics(h,d)\nassert abs(m['degree_variance']-.25)<1e-12\nassert abs(m['mean_cosine'])<1e-12\nassert collapse_metrics(torch.zeros_like(h),d)['mean_cosine'] is None\nq=torch.tensor([2.,3.,2.],dtype=torch.float64).sqrt()\nassert collapse_metrics(q[:,None],q.square())['degree_variance']<1e-20\nprint('CHECK: scale, degree correction, zero directions')",
'graph_layer':"s=torch.tensor([[1.,0.],[.5,.5]])\nh=torch.tensor([[2.,0.],[0.,4.]])\nw=torch.tensor([[3.],[2.]])\ntorch.testing.assert_close(graph_layer(s,h,w),torch.tensor([[6.],[7.]]))\nprint('CHECK: receiver routing and channel transform')"}

def prose(portable=False):
    text=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
    for name,caption in CAPTIONS.items():
        p=LAB/'figures/l085'/f'{name}.png'
        if not p.exists():text=text.replace('[[FIG:'+name+']]','Figure pending execution.');continue
        src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l085/{name}.png'
        text=text.replace('[[FIG:'+name+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
    for key,id in [('WARMUP','warmup'),('VIZ','smoothing'),('TEACHBACK','teachback')]:
        text=text.replace('[['+key+']]','Write your prediction or explanation before continuing. Use the lesson page for the interactive control.' if portable else f'<div id="{id}"></div>')
    p=LAB/'_depth_l085_results.json'
    if p.exists():
        rows=json.loads(p.read_text())['runs'];lines=['**Fresh author results, separate from your kernel:**','', '| Depth | Test accuracy, mean ± sample SD | Mean cosine |','|---|---|---|']
        import numpy as np
        for d in [1,2,4,8,16]:
            rr=[r for r in rows if r['depth']==d];a=np.array([r['test_accuracy'] for r in rr]);c=np.mean([r['mean_cosine'] for r in rr]);lines.append(f'| {d} | {100*a.mean():.2f}% ± {100*a.std(ddof=1):.2f} pp | {c:.3f} |')
        text=text.replace('[[RESULTS]]','\n'.join(lines))
    else:text=text.replace('[[RESULTS]]','Author Cora experiment is still running; no result claimed yet.')
    if portable:
        text=re.sub(r'\]\(\.\./','](https://avistian.github.io/relational/',text)
        text=text.replace('href="0082-gcn.html"','href="https://avistian.github.io/relational/lessons/0082-gcn.html"').replace('href="0084-gat.html"','href="https://avistian.github.io/relational/lessons/0084-gat.html"')
    return text

def inline(source,solution,selected=None):
    cells=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.Expr):continue
        if selected is not None and isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name not in selected:continue
        code=ast.get_source_segment(source,node)
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in CHECKS else '### PROVIDED · ')+node.name+'\n\n'+(ast.get_docstring(node) or 'Read the visible implementation.')))
            if node.name in CHECKS and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+node.name+' and pass the next CHECK")'
        cells.append(nbf.v4.new_code_cell(code))
        if isinstance(node,ast.FunctionDef) and node.name in CHECKS:cells.append(nbf.v4.new_code_cell(CHECKS[node.name]))
    return cells

def build(keep_execution=False):
    from _walkthrough_delivery import snapshot, finalize
    snapshot(85)
    head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','oversmoothing-viz'])+'</head><body><article>'
    head+=f'<nav><a href="../index.html">Course</a> · <a href="0084-gat.html">Lesson 84</a></nav><header><p>Year 3 · Quarter 1 · Lesson 085</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/solutions/{SLUG}.ipynb">Solution</a> · <a href="../labs/l085-reproduction.md">Reproduce</a></aside>'
    scripts=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','teachback','oversmoothing-viz','l085-over-smoothing'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+'</article>'+scripts+'</body></html>')
    graph=json.loads((LAB/'sources/l085/karate.json').read_text());manifest=json.loads((LAB/'_sources_l078.json').read_text())
    for solution in [False,True]:
        cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nStart with retrieval. Three live TODOs feed full, visible experiments. The named paper figure is untrained: training it would change the experiment. The default Cora extension runs one seed per depth; set RUN_FULL_CORA=True for the complete ten-seed author protocol.'),nbf.v4.new_code_cell("# @colab-bootstrap\nimport sys,subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy==2.5.0','scipy==1.18.0','torch==2.13.0','matplotlib==3.11.0'])")]
        cells += [nbf.v4.new_markdown_cell(p) for p in re.split(r'(?=^## )',prose(True),flags=re.M) if p.strip()]
        cells.append(nbf.v4.new_markdown_cell('## PROVIDED · attribution\n\nOperations follow liqimai/gcn revision 3b30a2d35ca2b144bf0f36337f233d407a7e2dd6. The separate Cora loader follows tkipf/gcn revision 39a4089fe72ad9f055ed6fdb9746abdcfebc4d81 (L082).\n\n```text\n'+(LAB/'sources/l085/LICENCE').read_text()+'\n```'))
        cells+=inline((LAB/'relkit/oversmoothing_l085.py').read_text(),solution)
        cells.append(nbf.v4.new_markdown_cell('## CHECK · the theorem and its boundary\n\nPredict: should a disconnected graph converge to one global value?'))
        cells.append(nbf.v4.new_code_cell("a=np.array([[0,1,0],[1,0,1],[0,1,0]],float)\ns,d=support(a);x=torch.tensor([[2.],[4.],[8.]],dtype=torch.float64)\nq=d.sqrt();q=q/q.norm()\nlimit=q[:,None]*(q@x)[None,:]\ntorch.testing.assert_close(smooth(s,x,200),limit)\nprint('Limit:',limit.flatten().tolist())\na=np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],float)\ns,d=support(a);x=torch.tensor([[0.],[0.],[2.],[2.]],dtype=torch.float64)\ntorch.testing.assert_close(smooth(s,x,100),x)\nassert collapse_metrics(x,d)['degree_variance']>0\nprint('CHECK: global collapse fails across disconnected components')"))
        cells.append(nbf.v4.new_markdown_cell('## Full named Figure 2 setup · 100 declared seeds\n\nThe graph bytes and labels are embedded for offline use. Labels are used only by the scatter plot. All depths and seeds are saved; do not cherry-pick the picture.'))
        cells.append(nbf.v4.new_code_cell('graph='+repr(graph)+"\ntorch.set_num_threads(1)\nassert len(graph['edges'])==78 and len(graph['labels'])==34\nresult=karate_experiment(graph,100)\nassert len(result['runs'])==500\nPath('l085-karate-inline.json').write_text(json.dumps(result,indent=2))\nprint(result['status'],result['historical_coordinate_parity'])"))
        cells.append(nbf.v4.new_code_cell("import matplotlib.pyplot as plt\nVIEW_SEED=0 # decide before looking at the scatter\nrows=[r for r in result['runs'] if r['seed']==VIEW_SEED]\nfig,axes=plt.subplots(1,5,figsize=(13,3),sharex=True,sharey=True)\nlim=max(abs(v) for r in rows for xy in r['coordinates'] for v in xy)*1.15\nfor ax,r in zip(axes,rows):\n    z=np.array(r['coordinates']);ax.scatter(z[:,0],z[:,1],c=graph['labels'],cmap='coolwarm',s=16)\n    ax.set(title=f\"Depth {r['depth']}\",xlim=(-lim,lim),ylim=(-lim,lim))\nplt.show()"))
        cells.append(nbf.v4.new_markdown_cell('## PROVIDED · complete pinned Cora loader\n\nOnly known SHA-256-verified upstream files are deserialized. The manifest is embedded. The 140/500/1000 train/validation/test labels are disjoint. Features and graph structure are transductively visible.'))
        cells+=inline((LAB/'relkit/gcn_l082.py').read_text(),True,{'as_sparse','normalized_support','load_cora'})
        cells.append(nbf.v4.new_code_cell('cora_manifest='+repr(manifest)+"\nDATA_ROOT=Path.cwd()/'l085-cora-data'\nDATA_ROOT.mkdir(exist_ok=True)\np=DATA_ROOT/'_sources_l078.json'\nif p.exists():assert json.loads(p.read_text())==cora_manifest, 'Existing provenance differs'\nelse:p.write_text(json.dumps(cora_manifest,indent=2))\ndata=load_cora(DATA_ROOT)\nassert data[0].shape==(2708,1433)\nassert [len(i) for i in data[3:]]==[140,500,1000]"))
        cells.append(nbf.v4.new_markdown_cell('## Extension · trained depth versus accuracy\n\nThis uses full data and the complete stopping schedule. One seed per depth is the default learning run; ten reproduce the author extension. This is not a published Li et al. table. The last hidden layer (output logits at depth1) supplies diagnostics.'))
        cells.append(nbf.v4.new_code_cell("RUN_FULL_CORA=False\nseed_count=10 if RUN_FULL_CORA else 1\ndepth_runs=[]\nfor depth in [1,2,4,8,16]:\n    for seed in range(seed_count):\n        row=train_depth(data,depth,seed);depth_runs.append(row)\n        print(depth,seed,row['test_accuracy'],row['mean_cosine'],row['rms'])\nPath('l085-cora-inline.json').write_text(json.dumps({'experiment':'LOCAL_EXTENSION','runs':depth_runs},indent=2))"))
        cells.append(nbf.v4.new_code_cell("fig,axes=plt.subplots(1,2,figsize=(10,3.5))\nfor ax,key in zip(axes,['test_accuracy','mean_cosine']):\n    for depth in [1,2,4,8,16]:\n        values=[r[key] for r in depth_runs if r['depth']==depth]\n        ax.scatter([depth]*len(values),values)\n    ax.set(xlabel='GCN depth',ylabel=key);ax.set_xscale('log',base=2)\nplt.show()"))
        cells.append(nbf.v4.new_markdown_cell('## EXIT · learner-produced evidence\n\nAttach both JSON files and your plots. Explain the degree-scaled limit, a disconnected counterexample, the role of RMS and zero rows, why the paper figure has no trainer, and which conclusion the Cora extension permits. Include one failing intervention and repair. Choose your definition of a practical collapse point before inspecting test scores; use validation evidence to propose a remedy. Ask the agent to assess your explanation.'))
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
        if solution and keep_execution:
            previous=nbf.read(path,4);old=[c for c in previous.cells if c.cell_type=='code'];new=[c for c in nb.cells if c.cell_type=='code']
            assert [c.source for c in old]==[c.source for c in new], 'Code changed: re-execute solution'
            for a,b in zip(old,new):b.outputs=a.outputs;b.execution_count=a.execution_count
        nbf.write(nb,path)
        if solution and keep_execution:
            import hashlib
            p=LAB/'_execution_l085_results.json';e=json.loads(p.read_text());e['notebook_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();e['prose_refresh_code_unchanged']=True;p.write_text(json.dumps(e,indent=2)+'\n')
        if not solution:
            page,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(page)
    card='''# Over-smoothing · diagnosis card

**Fixed symmetric propagation:** S = D̃⁻¹/²(A+I)D̃⁻¹/². On one connected component, q = √d̃ / ||√d̃|| and SᵏX → q(qᵀX). Raw rows need not match. H/√d̃ approaches a constant per component.

| Observation | Next check |
|---|---|
| Tiny variance | Inspect feature RMS; shrinking is not directional collapse |
| High cosine | Report zero rows; inspect task performance and components |
| Falling test accuracy | Check training accuracy, stopping and validation-only selection |
| More layers fail | Distinguish mixing from optimization and communication bottlenecks |

Figure 2 uses 34 karate nodes, 78 binary edges, one-hot features, untrained Glorot GCNs, hidden width16 and two output coordinates at depths1–5. The Cora training sweep is a separate extension.

[Return to lesson](../lessons/0085-over-smoothing.html) · [Primary source](https://arxiv.org/abs/1801.07606) · [Protocol](../labs/l085-reproduction.md)
'''
    (ROOT/'reference/over-smoothing.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Over-smoothing reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(card)+'</article></body></html>')
    finalize(85)

if __name__=='__main__':
    import sys
    build('--keep-execution' in sys.argv)
