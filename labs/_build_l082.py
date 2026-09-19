"""Build GCN lesson and fully inline, portable exercise/solution notebooks."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0082-gcn';TITLE='GCN: normalize, propagate, reproduce'
CAPTIONS={'trace':'Four-node arithmetic: loops, augmented degrees and each contribution to B.','architecture':'Complete Cora GCN: two layers, shared graph support, masked supervision, stopping and test inference.','results':'All 100 initialization scores on the same Cora split; variability is not across datasets.','embedding':'Seed-zero final hidden states projected to two principal components; descriptive visualization only.'}

def prose(portable=False):
    s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
    for name,caption in CAPTIONS.items():
        src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l082'/f'{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l082/{name}.png'
        s=s.replace('[[FIG:'+name+']]',f'<figure class="mpnn-figure"><small>Scroll horizontally on narrow screens to inspect the full computation.</small><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
    for name in ['warmup','predict','normalized','teachback']:
        s=s.replace('[['+name.upper()+']]', '**Pause and write your prediction before reading the explanation.**' if portable else '<div id="'+name+'"></div>')
    r=json.loads((LAB/'_paper_l082_results.json').read_text())
    s=s.replace('[[RESULTS]]',f'**Fresh author run:** {len(r["runs"])} initializations; mean **{100*r["mean"]:.3f}%**, sample SD **{100*r["sample_sd"]:.3f} percentage points**, standard error **{100*r["se"]:.3f} percentage points**. Difference from paper target: **{100*(r["mean"]-.815):+.3f} percentage points**. [Per-seed scores and complete validation traces](../labs/_paper_l082_results.json). These are measured port results, not original-framework parity.')
    if portable:
        s=s.replace('](0081-','](https://avistian.github.io/relational/lessons/0081-').replace('](../labs/','](https://avistian.github.io/relational/labs/').replace('](../reference/','](https://avistian.github.io/relational/reference/')
    return s

CHECKS={
'normalized_support':'''a=sp.csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]],dtype=float)
s=normalized_support(a)
expected=torch.tensor([[.5,6**-.5,0,0],[6**-.5,1/3,6**-.5,0],[0,6**-.5,.5,0],[0,0,0,1]],dtype=torch.float32)
torch.testing.assert_close(s.to_dense(),expected)
assert not torch.allclose(s.to_dense().sum(1),torch.ones(4))
p=np.array([2,0,3,1]);torch.testing.assert_close(normalized_support(a[p][:,p]).to_dense(),expected[p][:,p])
print('CHECK: exact coefficients, isolated node, non-mean weights, permutation')''',
'propagate':'''h=torch.tensor([[2.],[4.],[8.],[10.]]);w=torch.ones(1,1)
out=propagate(s,h,w)
torch.testing.assert_close(out,expected@h)
torch.testing.assert_close(propagate(s,h.to_sparse(),w),out)
torch.testing.assert_close(propagate(normalized_support(a[p][:,p]),h[p],w),out[p])
print('CHECK: sparse/dense states, equivariance; B =',float(out[1]))''',
'masked_objective':'''z=torch.tensor([[1.,0.],[0.,1.],[2.,0.]],requires_grad=True)
y=torch.tensor([0,1,1]);index=torch.tensor([0,1]);w0=torch.ones(2,2,requires_grad=True)
loss=masked_objective(z,y,index,w0);loss.backward()
assert torch.count_nonzero(z.grad[2])==0
torch.testing.assert_close(w0.grad,.0005*w0)
y2=y.clone();y2[2]=0
torch.testing.assert_close(loss,masked_objective(z,y2,index,w0))
print('CHECK: held-out labels excluded, first-layer L2 coefficient exact')'''}

def inline(solution):
    source=(LAB/'relkit/gcn_l082.py').read_text();cells=[]
    for node in ast.parse(source).body:
        if isinstance(node,ast.Expr):continue
        code=ast.get_source_segment(source,node)
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in CHECKS else '### PROVIDED · ')+node.name+'\n\n'+(ast.get_docstring(node) or 'Inspect the full implementation.')))
            if node.name in CHECKS and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+node.name+'; see the contract and CHECK")'
        cells.append(nbf.v4.new_code_cell(code))
        if isinstance(node,ast.FunctionDef) and node.name in CHECKS:cells.append(nbf.v4.new_code_cell(CHECKS[node.name]))
    return cells

def build(keep_execution=False):
    from _walkthrough_delivery import snapshot, finalize
    snapshot(82)
    head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{n}.css">' for n in ['lesson','message-passing-viz','mpnn-lesson'])+'</head><body><article>'
    head+=f'<nav><a href="../index.html">Course</a> · <a href="0081-mpnn-framework.html">Lesson 81</a></nav><header><p>Year 3 · Quarter 1 · Lesson 082</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/solutions/{SLUG}.ipynb">Solution</a> · <a href="../labs/l082-reproduction.md">Reproduce</a></aside>'
    scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','message-passing-viz','teachback','l082-gcn'])
    (ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+'</article>'+scripts+'</body></html>')
    manifest=json.loads((LAB/'_sources_l078.json').read_text())
    for solution in [False,True]:
        cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nThree live TODOs, full inline model/trainer and full Cora experiment. PyTorch port of the MIT-licensed tkipf/gcn release; data and original source hashes are embedded. No repository import is needed. Complete the short checks before launching 100 runs.'),nbf.v4.new_code_cell("# Colab setup: record your runtime; the exact author snapshot is linked in the contract.\nimport sys,subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy','scipy','torch'])"),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_code_cell("from pathlib import Path\nimport json\nROOT=Path.cwd()\nmanifest="+repr(manifest)+"\nmanifest_path=ROOT/'_sources_l078.json'\nif manifest_path.exists():\n    assert json.loads(manifest_path.read_text())==manifest, 'Existing provenance differs'\nelse:\n    manifest_path.write_text(json.dumps(manifest,indent=2))")]
        cells.append(nbf.v4.new_markdown_cell('## Source attribution and license\n\nPort adapted from tkipf/gcn revision `39a4089fe72ad9f055ed6fdb9746abdcfebc4d81`.\n\n```text\n'+(LAB/'sources/l078/LICENCE').read_text()+'\n```'))
        cells+=inline(solution)
        cells+=[nbf.v4.new_markdown_cell('## CHECK · full graph, masks and live functions\n\nData bytes are verified before unpickling. No test label enters the loss or stop rule.'),nbf.v4.new_code_cell("torch.set_num_threads(1)\ndata=load_cora(ROOT)\nx,s,y,train,valid,test=data\nassert x.shape==(2708,1433)\nassert [len(train),len(valid),len(test)]==[140,500,1000]\nassert len(set(train.tolist()+valid.tolist()+test.tolist()))==1640\nassert torch.bincount(y[train]).tolist()==[20]*7\nmodel=GCN(1433,7).eval()\nwith torch.no_grad():\n    oracle=s.to_dense()@torch.relu(s.to_dense()@x.to_dense()@model.w0)@model.w1\n    torch.testing.assert_close(model(x,s),oracle,atol=2e-6,rtol=1e-5)\nprint('CHECK: full graph and dense forward oracle')"),nbf.v4.new_markdown_cell('## One-run diagnostic\n\nThis proves that your functions participate in optimization; it does not reproduce the 100-run mean.'),nbf.v4.new_code_cell("diagnostic=train_cora(data,0)\nprint({k:v for k,v in diagnostic.items() if k!='validation_loss'})"),nbf.v4.new_markdown_cell('## Full named experiment · 100 initializations\n\nRun this cell after all checks pass. Do not tune the model on these test outcomes. Fresh training, maximum 200 epochs each; allow several minutes on CPU. The tolerance below was declared before interpreting this lesson’s run.'),nbf.v4.new_code_cell("result=run_cora(seeds=100,root=ROOT)\nPath('l082-student-results.json').write_text(json.dumps(result,indent=2))\nwithin_tolerance=abs(result['mean']-.815)<=.01\nprint('Mean:',result['mean'],'SD:',result['sample_sd'],'within declared ±1pp:',within_tolerance)\nassert len(result['runs'])==100"),nbf.v4.new_markdown_cell('## EXIT · attach evidence and explain\n\nWrite your own answers: why are coefficients not row means; which labels enter each phase; why does a close score not prove cross-framework identity; what changes if new edges arrive? Include a deliberately failing normalization check and its diagnosis.'),nbf.v4.new_code_cell("artifact={'lesson':82,'runs':len(result['runs']),'mean_accuracy':result['mean'],'sample_sd':result['sample_sd'],'paper_target':.815,'educational_tolerance_pp':1,'within_tolerance':within_tolerance,'source_revision':manifest['revision'],'framework':'PyTorch port','original_tensorflow_parity':'NOT_RUN','written_explanation':'ADD YOUR EXPLANATION IN A MARKDOWN CELL'}\nPath('l082-exit.json').write_text(json.dumps(artifact,indent=2))\nprint(json.dumps(artifact,indent=2))")]
        nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
        path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
        if solution and keep_execution:
            old=nbf.read(path,as_version=4)
            before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
            assert [c.source for c in before]==[c.source for c in after], 'Cannot preserve execution after code changes'
            for previous,current in zip(before,after):
                current.outputs=previous.outputs;current.execution_count=previous.execution_count
        nbf.write(nb,path)
        if solution and keep_execution:
            import hashlib
            evidence=LAB/'_execution_l082_results.json';record=json.loads(evidence.read_text())
            record['notebook_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            record['prose_refresh_code_unchanged']=True;evidence.write_text(json.dumps(record,indent=2)+'\n')
        if not solution:
            page,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(page)
    ref='''# GCN contract card

**S = D̃⁻¹ᐟ²(A+I)D̃⁻¹ᐟ²; H′ = σ(SHW).** Add self-loops once, then count augmented degrees. Sender and receiver both determine the coefficient. Rows need not sum to one. An isolated node has a self-message of weight one before channel mixing.

| Stage | Shape / rule |
|---|---|
| Inputs | Cora X: 2708×1433; row-normalized |
| Hidden | ReLU(S X W₀), width16 |
| Output | S H W₁, seven logits/node; no pooling |
| Training | Input/hidden dropout .5; mean CE on140 labels; .0005 × half squared norm W₀ |
| Validation | 500 nodes; after epoch index10, current regularized loss > previous-ten mean stops |
| Inference | Last weights, dropout off;1000 test nodes |
| Replication |100 initializations, fixed split; target81.5%; modern PyTorch port |

On A—B—C with [2,4,8], B=2/√6+4/3+8/√6=5.415816. Ordinary mean is4.666667. GCN is node-equivariant. Permute features AND both adjacency axes. Features and edges of unlabeled nodes can participate in transductive training; held-out labels cannot supervise gradients.

A close mean is not protocol identity. The port has different framework/RNG; original100 seeds are unavailable. SD is initialization variation on one graph, not across datasets. Original TensorFlow training remains NOT_RUN.

[Lesson](../lessons/0082-gcn.html) · [Lab](../labs/0082-gcn.ipynb) · [Protocol](../labs/l082-reproduction.md) · [Primary source](https://arxiv.org/html/1609.02907v4)
'''
    (ROOT/'reference'/f'{SLUG}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GCN contract card</title><link rel="stylesheet" href="../assets/lesson.css"><article>'+render(ref)+'</article></html>')
    print('Built L082 lesson, notebooks, preview and reference')
    finalize(82)
if __name__=='__main__':
    import sys
    build(keep_execution='--keep-execution' in sys.argv)
