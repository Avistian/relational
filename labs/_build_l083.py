"""Generate L083 from canonical prose/code, with portable figures and live student TODOs."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0083-graphsage';TITLE='GraphSAGE: sample, aggregate, generalize'
CAPTIONS={'boundary':'Training sees only training tissues; fixed preprocessing and weights transfer to unseen graphs.','trace':'Identity-weight illustration: distinct self and neighbor paths produce [2,4,3,2].','architecture':'Complete released mean encoder: outward support sampling and inward shared aggregation, normalization, head and objective.','results':'Measured full-data ten-epoch ports; each seed runs three rates. The paper target is not a protocol-parity claim.'}
CHECKS={
'eligible_neighbors':"adj=[np.array([1,2]),np.array([0]),np.array([0])]\nmask=np.array([True,True,False])\nassert [a.tolist() for a in eligible_neighbors(adj,mask)]==[[1],[0],[]], 'Filter both receiver and sender'\nprint('CHECK: held-out nodes cannot enter either side of a training edge')",
'sample_support':"table=torch.tensor([[1,1,1,1],[0,0,0,0],[3,3,3,3],[3,3,3,3]])\ns=sample_support(torch.tensor([0,2]),table,[2,3],torch.Generator().manual_seed(0))\nassert [len(v) for v in s]==[2,4,12]\nassert s[1].tolist()==[1,1,3,3] and s[2].tolist()==[0]*6+[3]*6\nprint('CHECK: outward expansion, node IDs, zero sentinel')",
'mean_concat':"h=torch.tensor([[2.,4.]])\nn=torch.tensor([[[1.,3.],[5.,1.]]]);w=torch.eye(2)\nz=mean_concat(h,n,w,w)\ntorch.testing.assert_close(z,torch.tensor([[2.,4.,3.,2.]]))\ntorch.testing.assert_close(mean_concat(h,n.flip(1),w,w),z)\nprint('CHECK: separate branches and permutation invariance')"}
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 for n,c in CAPTIONS.items():
  src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l083'/f'{n}.png').read_bytes()).decode() if portable else f'../labs/figures/l083/{n}.png'
  s=s.replace('[[FIG:'+n+']]',f'<figure class="mpnn-figure"><small>Scroll to inspect the full diagram on narrow screens.</small><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{c}"></div><figcaption>{c}</figcaption></figure>')
 for n in ['warmup','predict','mean','budget','teachback']:
  s=s.replace('[['+n.upper()+']]', '**Pause: write your prediction or teach-back before continuing.**' if portable else '<div id="'+n+'"></div>')
 r=json.loads((LAB/'_paper_l083_results.json').read_text())
 rows='\n'.join(f'| {v["seed"]} | {v["selected_lr"]} | {v["test"]["micro_f1"]:.5f} |' for v in r['runs'])
 s=s.replace('[[RESULTS]]',f'**Author reference, freshly executed:** {len(r["runs"])} seeds × three learning rates × ten epochs on the full archive. Mean test micro-F1 **{r["mean"]:.5f}**, sample SD **{r["sample_sd"]:.5f}**. This SD describes seed/search variation on the same split.\n\n| Seed | Validation-selected learning rate | Test micro-F1 |\n|---|---|---|\n'+rows+'\n\n[Full per-epoch traces and results](../labs/_paper_l083_results.json).')
 if portable:
  s=re.sub(r'\]\(\.\./(labs|reference)/',r'](https://avistian.github.io/relational/\1/',s)
  s=s.replace('](0082-gcn.html)','](https://avistian.github.io/relational/lessons/0082-gcn.html)')
 return s

def build():
 from _walkthrough_delivery import snapshot, finalize
 snapshot(83)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{n}.css">' for n in ['lesson','mpnn-lesson'])+'</head><body><article>'
 head+=f'<nav><a href="../index.html">Course</a> · <a href="0082-gcn.html">Lesson 82</a></nav><header><p>Year 3 · Quarter 1 · Lesson 083</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/l083-reproduction.md">Reproduce</a></aside>'
 scripts=''.join(f'<script src="../assets/{n}.js"></script>' for n in ['retrieval-pool','retrieval-bank','predict','teachback','l083-graphsage'])
 (ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+'</article>'+scripts+'</body></html>')
 source=(LAB/'relkit/graphsage_l083.py').read_text();manifest=json.loads((LAB/'_sources_l083.json').read_text())
 for solution in [False,True]:
  cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nFull inline GraphSAGE mean port and original full PPI archive. Three TODOs feed the actual training path. The one-epoch diagnostic and full ten-epoch search are separate experiments.'),nbf.v4.new_code_cell("# @colab-bootstrap\nimport sys,subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','numpy','torch','scikit-learn'])"),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## Source attribution\n\nAdapted from William L. Hamilton and Rex Ying, GraphSAGE (MIT), pinned revision '+manifest['revision']+'. Full license:\n\n```text\n'+(LAB/'sources/l083/LICENSE.txt').read_text()+'\n```')]
  for node in ast.parse(source).body:
   if isinstance(node,ast.Expr):continue
   code=ast.get_source_segment(source,node)
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in CHECKS else '### PROVIDED · ')+node.name+'\n\n'+(ast.get_docstring(node) or 'Inspect the implementation.')))
    if node.name in CHECKS and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+node.name+' from the contract and pass the CHECK")'
   cells.append(nbf.v4.new_code_cell(code))
   if isinstance(node,ast.FunctionDef) and node.name in CHECKS:cells.append(nbf.v4.new_code_cell(CHECKS[node.name]))
  cells += [nbf.v4.new_markdown_cell('## PROVIDED · immutable data identity\n\nThe archive downloads from Stanford only if absent; its SHA-256 is verified before loading. StandardScaler fits only training rows.'),nbf.v4.new_code_cell('manifest='+repr(manifest)+"\ntorch.set_num_threads(1)\ndata=load_ppi(Path.cwd()/'data/l083',manifest)\nassert data['x'].shape==(56945,50) and data['y'].shape==(56944,121)\nprint({k:int(v.sum()) for k,v in data['masks'].items()})"),nbf.v4.new_markdown_cell('## Teaching run · one full-data epoch, smaller width\n\nCheck that the implemented functions train. This is explicitly not the paper schedule.'),nbf.v4.new_code_cell("teaching_config=dict(PAPER_CONFIG,epochs=1,branch_dim=16,learning_rates=[.01])\nteaching=experiment(data,[123],teaching_config)\nprint('TEACHING ONLY',teaching['mean'])"),nbf.v4.new_markdown_cell('## Full named experiment · three seeds × three rates × ten epochs\n\nCPU is sufficient; author runs use one thread to avoid oversubscription. All data, full dimensions and full schedule. This is a modern release port with documented historical gaps. Read the contract before interpreting proximity to .598.'),nbf.v4.new_code_cell("RUN_FULL_REPRO = True\nif RUN_FULL_REPRO:\n    result=experiment(data,[123,124,125],copy.deepcopy(PAPER_CONFIG),'l083-student-results.json')\n    print('Full PPI port mean / sample SD:',result['mean'],result['sample_sd'])"),nbf.v4.new_markdown_cell('## EXIT TICKET\n\nAttach your full result JSON. Explain the split-access invariant, fanout reversal, dimensions, code-versus-pseudocode differences, and a remaining historical gap. Change a mean to a sum; predict and capture the failing CHECK, then restore it. Do not infer mastery from executing the solution.'),nbf.v4.new_code_cell("artifact={'lesson':83,'source_revision':manifest['revision'],'full_run':RUN_FULL_REPRO,'paper_target':.598,'historical_parity':'INCOMPARABLE','explanation':'ADD YOUR OWN MARKDOWN EXPLANATION'}\nif RUN_FULL_REPRO:artifact.update(mean=result['mean'],sample_sd=result['sample_sd'])\nPath('l083-exit.json').write_text(json.dumps(artifact,indent=2))\nprint(artifact)")]
  nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
  path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb';nbf.write(nb,path)
  if not solution:
   html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
 ref='''# GraphSAGE contract card

**Released mean block:** `[self @ W_self | mean(neighbors) @ W_neighbor]`. First layer ReLU; last layer identity; normalize the final embedding once. Branch width128 produces256 coordinates. Biased121-label head; sigmoid decisions, BCE averaged over labels and roots.

| Stage | Contract |
|---|---|
| Training graph | Both endpoints belong to training split |
| Preprocessing | Fit standardization on training nodes |
| Sampling | Fixed128-entry adjacency; outward10 then25 |
| Support sizes | B,10B,250B occurrences |
| Aggregation | Outside inward; first-layer weights shared across hops |
| Supervision | Root labels only; multilabel sigmoid loss |
| Selection | Full validation micro-F1 after10 epochs selects .01/.001/.0001 |
| Final score | Pooled micro-F1 across all test node-label pairs |
| Named paper target | Table1 supervised mean PPI .598 |
| Remaining gaps | PyTorch/TF, RNG, historical seeds/configuration; no exact parity claim |

**Diagnostic:** flip held-out labels/features. Training weights must stay identical. Permute a sampled neighbor list: mean must stay identical. Change the sampled multiset: output may change.

[Lesson](../lessons/0083-graphsage.html) · [Lab](../labs/0083-graphsage.ipynb) · [Protocol](../labs/l083-reproduction.md) · [Primary source](https://arxiv.org/html/1706.02216v4)
'''
 (ROOT/'reference'/f'{SLUG}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GraphSAGE contract card</title><link rel="stylesheet" href="../assets/lesson.css"><article>'+render(ref)+'</article></html>')
 print('Built lesson, reference, student and solution notebooks')
 finalize(83)
if __name__=='__main__':build()
