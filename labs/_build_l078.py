"""Single-source L078 lesson/notebook builder, with complete visible Cora model/trainer."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _figures_l078 import build as figures
LAB=Path(__file__).resolve().parent;ROOT=LAB.parent;SLUG='0078-message-passing-preview';TITLE='Message passing: from a join to a GCN'
SOURCE=(LAB/'relkit/message_passing.py').read_text();TREE=ast.parse(SOURCE)
NODES={n.name:ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
IMPORTS='\n'.join(ast.get_source_segment(SOURCE,n) for n in TREE.body if isinstance(n,(ast.Import,ast.ImportFrom)))
CAPTIONS={'trace':'Synchronous hand fixture: old states, incoming messages, means and updated outputs. D has no incoming messages; its declared empty mean is zero.','normalization':'Same graph, different operator: symmetric GCN contributions to B, with W=1. Self-loop included; weights do not sum to one.','architecture':'Full Cora GCN: feature normalization, two graph layers and node prediction/loss. Dropout is active during training only.','results':'Author-reference measurement: 100 initializations on one fixed Cora split. Each dot is a measured port run, not a new dataset. Dashed line is the cited paper target.'}
CHECKS={
'mean_neighbors':"""x=np.array([[2.],[4.],[8.],[10.]])
edges=np.array([[0,1],[1,0],[1,2],[2,1]])
np.testing.assert_allclose(mean_neighbors(x,edges),[[4],[5],[4],[0]])
np.testing.assert_allclose(mean_neighbors(x,edges[::-1]),[[4],[5],[4],[0]])
np.testing.assert_allclose(mean_neighbors(x,[]),np.zeros_like(x))
print('CHECK: routing, empty neighbors and edge order pass')""",
'mean_step':"""np.testing.assert_allclose(mean_step(x,edges),[[3],[4.5],[6],[5]])
p=np.array([2,0,3,1]);inv=np.argsort(p)
np.testing.assert_allclose(mean_step(x[p],inv[edges]),mean_step(x,edges)[p])
changed=x.copy();changed[2]=20
assert mean_step(changed,edges)[0,0]==3
assert mean_step(mean_step(changed,edges),edges)[0,0]==5.25
print('CHECK: synchronous updates, relabeling and two-hop reach pass')""",
'gcn_support':"""a=np.zeros((4,4));a[edges[:,1],edges[:,0]]=1
s=gcn_support(a)
np.testing.assert_allclose(s[1],[1/np.sqrt(6),1/3,1/np.sqrt(6),0])
np.testing.assert_allclose((s@x).ravel(),[1+4/np.sqrt(6),10/np.sqrt(6)+4/3,4+4/np.sqrt(6),10])
assert not np.allclose(s.sum(1),1), 'Symmetric normalization is not row mean'
print('CHECK: self-loops, endpoint degrees and weighted sum pass')"""}
def manuscript(notebook=False):
    t=(ROOT/'lessons/content'/f'{SLUG}.md').read_text();r=json.loads((LAB/'_paper_l078_results.json').read_text())
    table=f"**Author-reference results — executed locally, not your current kernel output.**\n\n| Experiment | Seeds | Mean accuracy | Sample SD | Standard error |\n|---|---:|---:|---:|---:|\n| Full Cora release-protocol port | 100 | {100*r['mean']:.3f}% | {100*r['sample_sd']:.3f} percentage points | {100*r['se']:.3f} percentage points |\n| Paper Table 2 target | 100 | 81.5% | Not stated in this row | Not stated in this row |\n\nLocal minus paper: {100*(r['mean']-.815):+.3f} percentage points. No hyperparameters were changed after reading the test results. [All seed results](../labs/_paper_l078_results.json).\n\n<!--figure:results-->"
    t=t.replace('<!--results-->',table)
    for name,caption in CAPTIONS.items():
        src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l078'/f'{name}.png').read_bytes()).decode() if notebook else f'../labs/figures/l078/{name}.png'
        t=t.replace('<!--figure:'+name+'-->',f'<figure class="mp-figure"><div tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
    if notebook:
        t=re.sub(r'<div id="[^"]+"></div>','',t)
        t=re.sub(r'\((00\d[^)]*\.html)\)',r'(../lessons/\1)',t)
    return t

def notebook(solution=False):
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('# Lab 078 · '+TITLE+'\n\n**Core:** derive one-hop embeddings. **Research route:** reproduce the full Cora Table 2 GCN experiment with an explicit framework-deviation ledger. Read and implement before running the full experiment. All code is visible below.')
    cells.extend(nbf.from_dict(c) for c in bootstrap_cells())
    for part in re.split(r'(?=^## )',manuscript(True),flags=re.M):
        if part.strip():md(part)
    md('## PROVIDED · imports\n\nThe core functions below operate on arrays. Edges are source/destination pairs. Avoid in-place updates: every destination must read the same old snapshot.')
    code(IMPORTS)
    for name,task in [('mean_neighbors','Accumulate source states at each destination; divide by incoming count. Return zero for nodes without neighbors.'),('mean_step','Call your live mean_neighbors function, mix half self and half incoming mean, and apply ReLU without mutating the input.'),('gcn_support','Add identity to a loop-free undirected adjacency. Compute augmented degrees, then scale each entry by both inverse square-root degrees.')]:
        md('## TODO · '+name+'\n\n'+task+' Predict the fixture outputs on paper before executing CHECK.')
        code(NODES[name] if solution else NODES[name].split('\n')[0]+'\n    raise NotImplementedError("Implement '+name+'")')
        code('# CHECK — do not edit\n'+CHECKS[name])
    md('## EXIT · defend your calculation\n\nWrite: (1) B’s messages and update; (2) why an in-place loop is wrong; (3) how C can change A after two rounds; (4) why GCN weights are not an ordinary mean; (5) which availability boundaries a two-hop database model needs. Send your explanation and CHECK outputs for grading. This cell is a written task, not automatically graded.')
    md('## Research route · full Cora reproduction\n\nThe following complete implementation is separate from the teaching mean variant. Its sparse operator is independently checked against your dense normalization. No hidden GNN library is used. Run all 100 seeds to match the declared replication count. The original framework and unpublished seed sequence remain deviations. Allow several minutes on CPU.\n\n### PROVIDED · verified source manifest\n\nOnly pinned, hash-checked upstream data is deserialized. This embedded manifest lets the notebook fetch data without relying on an unpublished course checkout.')
    manifest=(LAB/'_sources_l078.json').read_text()
    code("LAB_ROOT=Path.cwd()\nif (LAB_ROOT/'labs').is_dir(): LAB_ROOT=LAB_ROOT/'labs'\nmanifest="+repr(manifest)+"\n(LAB_ROOT/'_sources_l078.json').write_text(manifest)\n")
    for name,why in [('as_sparse','Convert sparse arrays into coalesced sparse tensors.'),('load_cora','Load the pinned raw files, preserve test reordering, build the released fixed split, normalize rows and graph support.'),('GCN','Two bias-free layers with Glorot initialization; feature and hidden dropout only during training.'),('train_cora','The complete gradient loop, first-layer-only half-norm penalty, released stopping rule, and one final test evaluation.'),('run_cora','Load once; repeat independent initializations on the fixed split. Aggregate without selecting favorable seeds.')]:
        md('### PROVIDED · '+name+'\n\n'+why);code(NODES[name])
    md('### CHECK · dense vs sparse support and forward pass\n\nA forward oracle tests arithmetic, not training parity with TensorFlow. Hold graph, features and weights fixed; vary only the computational representation.')
    code("""torch.set_num_threads(1)
data=load_cora(LAB_ROOT)
xx,ss,yy,tr,va,te=data
np.testing.assert_allclose(as_sparse(sp.csr_matrix(gcn_support(a))).to_dense().numpy(),gcn_support(a),rtol=1e-6)
model=GCN(xx.shape[1],7);model.eval()
with torch.no_grad():
    oracle=ss.to_dense()@torch.relu(ss.to_dense()@xx.to_dense()@model.w0)@model.w1
    torch.testing.assert_close(model(xx,ss),oracle,atol=2e-6,rtol=1e-5)
assert (len(tr),len(va),len(te))==(140,500,1000)
assert torch.bincount(yy[tr]).tolist()==[20]*7
print('CHECK: dense/sparse arithmetic and published split pass')""")
    md('### PAPER RESULTS · execute the complete named experiment\n\nThis cell uses the model and trainer shown above, including your kernel’s current definitions. It does not load cached author scores. Record the mean, variability and deviations. Do not tune against the test target.')
    code("""results=run_cora(seeds=100,root=LAB_ROOT)
print('seed | epochs | test accuracy')
for r in results['runs']:print(f"{r['seed']:3d} | {r['epochs']:3d} | {100*r['test_accuracy']:.2f}%")
print(f"Mean {100*results['mean']:.3f}%; sample SD {100*results['sample_sd']:.3f} pp; SE {100*results['se']:.3f} pp")
print('Paper target: 81.5%. Framework and original-seed differences remain.')
(LAB_ROOT/'l078-student-results.json').write_text(json.dumps(results,indent=2)+'\\n')""")
    md('### Interpret before claiming reproduction\n\nCompare your mean with the author reference above; inspect per-seed dispersion. Explain what the fixed-split standard error does and does not measure. Distinguish full Cora execution, numerical forward verification, original TensorFlow parity (NOT_RUN), and the unrun QM9 experiment. For the next session, reproduce B’s calculations from memory.')
    for i,c in enumerate(cells):c.id=f'l078-{i:03d}'
    return nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},'language_info':{'name':'python'}})

def build():
    figures()
    head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 078 — '+TITLE+'</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/message-passing-viz.css"></head><body><article>'
    nav=f'<nav><a href="../index.html">Course</a> · <a href="0077-single-table-ceiling.html">Lesson 77</a></nav><header><p>Year 2 · Quarter 4 · Lesson 078</p><h1>{TITLE}</h1></header><aside><a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/{SLUG}.ipynb">Notebook</a> · <a href="../reference/{SLUG}.html">Reference card</a></aside>'
    body=render(manuscript()).replace('<table>','<div class="mp-table"><table>').replace('</table>','</table></div>')
    scripts=''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','message-passing-viz','l078-lesson'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+nav+body+'</article>'+scripts+'</body></html>')
    for solution in [False,True]:
        nb=notebook(solution);p=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
        if solution and p.exists():
            old=nbf.read(p,as_version=4);a=[c for c in nb.cells if c.cell_type=='code'];b=[c for c in old.cells if c.cell_type=='code']
            if [c.source for c in a]==[c.source for c in b]:
                for x,y in zip(a,b):x.outputs=y.outputs;x.execution_count=y.execution_count
        nbf.write(nb,p)
    preview,_=HTMLExporter().from_notebook_node(notebook());soup=BeautifulSoup(preview,'html.parser')
    for a in soup.find_all('a',href=True):
        if a['href'].startswith('../'):a['href']='../'+a['href']
    (LAB/'html'/f'{SLUG}.html').write_text(str(soup)+'\n')
    ref='''# Message passing · calculation card

**Route:** `(source,destination)` sends the source's old state. Every node reads the same old snapshot.

**Mean fixture:** incoming mean; empty mean zero; `h_new=ReLU(.5 h + .5 mean)`. A—B—C plus isolated D, states `[2,4,8,10]`: means `[4,5,4,0]`, new states `[3,4.5,6,5]`.

**MPNN:** construct messages → sum → update; invariant readout only for a graph-level target. Mean is a reduction variant, not the full molecular model.

**GCN:** add identity, compute augmented degrees, then `S[v,w]=(A+I)[v,w]/sqrt(d_v*d_w)`. Layer `ReLU(S H W)`. B with scalar W=1 gives `2/sqrt(6)+4/3+8/sqrt(6)=5.415816`; ordinary self-inclusive mean is 4.666667.

**Shapes:** H is N×d; W is d×k; S is N×N; output is N×k. Cora: 1433→16→7, two bias-free layers, node logits. Dropout at layer inputs is disabled at inference.

**Symmetry:** edge enumeration leaves reductions unchanged; node relabeling consistently reorders node outputs (equivariance). Mean loses multiplicity.

**Reach:** one local layer moves information one edge. Changing C to20 leaves A's first state3, but changes its second state from3.75 to5.25.

**Time:** filter nodes, edges, attributes and degree normalization by availability at every hop. A graph layer is not itself a temporal safety guarantee.

**Reproduction:** full Cora Table2 target81.5%; 100 seeds, fixed140/500/1000 labels, 200-epoch maximum. Release stopping compares current validation loss with previous10 mean and uses last weights. Modern PyTorch port and unknown original seeds prevent exact TensorFlow parity claims.

[Gilmer §2](https://proceedings.mlr.press/v70/gilmer17a.html) · [GCN paper](https://arxiv.org/html/1609.02907v4)
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text(head+render(ref+f'\n[Lesson](../lessons/{SLUG}.html) · [Lab](../labs/html/{SLUG}.html)')+'</article></body></html>')
    print('Built L078 lesson, both notebooks, preview, reference and portable figures')
if __name__=='__main__':build()
