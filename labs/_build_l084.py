"""Build canonical HTML, portable notebooks, and reference from visible source."""
import ast,base64,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0084-gat';TITLE='GAT: learn which neighbors to weight'
CAPTIONS={'trace':'One receiver, three allowed senders: normalized scores multiply projected values before summation.','architecture':'Full Cora graph and features enter eight independent attention heads, then a class head; masks separate training, selection and testing.','attention':'Actual learned coefficients at receiver 0 from seed 0, hidden head 1; evaluation mode, self included.','results':'One hundred initializations on one fixed Cora split: seed variability is not uncertainty over datasets.'}
CHECKS={
'neighbor_softmax':"r=torch.tensor([0,0,1]);s=torch.tensor([1000.,1001.,-1000.])\na=neighbor_softmax(s,r,2)\ntorch.testing.assert_close(a,torch.tensor([.26894143,.7310586,1.]))\nprint('CHECK: stable, separate receiver denominators')",
'weighted_messages':"z=torch.tensor([[1.,0.],[0.,2.]])\nedge=torch.tensor([[0,0,1],[0,1,1]])\na=torch.tensor([.25,.75,1.])\ntorch.testing.assert_close(weighted_messages(z,edge,a),torch.tensor([[.25,1.5],[0.,2.]]))\nprint('CHECK: sender values sum into receivers, no extra mean')",
'merge_heads':"a=torch.tensor([[1.,2.]]);b=torch.tensor([[3.,6.]])\ntorch.testing.assert_close(merge_heads([a,b],True),torch.tensor([[1.,2.,3.,6.]]))\ntorch.testing.assert_close(merge_heads([a,b],False),torch.tensor([[2.,4.]]))\nprint('CHECK: concat changes width, averaging preserves classes')"}
def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 for name,caption in CAPTIONS.items():
  if f'[[FIG:{name}]]' not in s:continue
  src='data:image/png;base64,'+base64.b64encode((LAB/'figures/l084'/f'{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l084/{name}.png'
  s=s.replace(f'[[FIG:{name}]]',f'<figure class="mpnn-figure"><small>Scroll horizontally to inspect the diagram on narrow screens.</small><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 s=s.replace('[[ATTENTION]]','**On paper:** recompute after removing sender 2, then change its value to −2 (use the 0.2 negative slope).' if portable else '<div id="attention"></div>')
 p=LAB/'_paper_l084_results.json'
 if p.exists():
  r=json.loads(p.read_text());s=s.replace('[[RESULTS]]',f"**Fresh author execution:** {len(r['runs'])} complete runs; mean **{100*r['mean']:.3f}%**, sample SD **{100*r['sample_sd']:.3f} percentage points**. The paper reports 83.0 ± 0.7%. All runs use validation-only checkpoint selection. [Every epoch and seed](../labs/_paper_l084_results.json).\n\n"+('<img alt="Accuracy across all declared seeds" src="data:image/png;base64,'+base64.b64encode((LAB/'figures/l084/results.png').read_bytes()).decode()+'">' if portable else '<figure class="mpnn-figure"><small>Scroll horizontally to inspect the diagram on narrow screens.</small><div class="figure-scroll" tabindex="0"><img src="../labs/figures/l084/results.png" alt="Accuracy across all declared seeds"></div></figure>'))
 else:s=s.replace('[[RESULTS]]','**Execution pending:** full experiment results will be inserted after all runs finish.')
 weights=json.loads((LAB/'_attention_l084.json').read_text())['coefficients']
 s=s.replace('[[LEARNED]]',f"**Inspect the measurement:** this head's coefficients range from {min(weights):.4f} to {max(weights):.4f}, close to uniform {1/len(weights):.4f}. Having learned attention does not guarantee sharply differentiated routing. Compare this trained head with the deliberately stronger arithmetic example above.")
 if portable:
  s=re.sub(r'\]\(\.\./(labs|reference)/',r'](https://avistian.github.io/relational/\1/',s)
  s=s.replace('](0083-graphsage.html)','](https://avistian.github.io/relational/lessons/0083-graphsage.html)').replace('](0082-gcn.html)','](https://avistian.github.io/relational/lessons/0082-gcn.html)')
 return s

def build():
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/mpnn-lesson.css"><link rel="stylesheet" href="../assets/graph-attention.css"></head><body><article><nav><a href="../index.html">Course</a> · <a href="0083-graphsage.html">Lesson 83</a></nav><header><p>Year 3 · Quarter 1 · Lesson 084</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/l084-reproduction.md">Reproduce</a></aside>'
 (ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+'</article><script src="../assets/l084-gat.js"></script></body></html>')
 source=(LAB/'relkit/gat_l084.py').read_text();manifest=json.loads((LAB/'_sources_l084.json').read_text())
 for solution in [False,True]:
  cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nThree live TODOs feed a complete visible GAT trainer. Default execution performs a five-epoch teaching check and one full seed. Set RUN_100_SEEDS=True to run the full named 100-run Cora experiment; it can take tens of minutes or longer on CPU. Author results are separate from your fresh execution.'),nbf.v4.new_code_cell("# @colab-bootstrap\nimport sys,subprocess\nif 'google.colab' in sys.modules:\n    subprocess.check_call([sys.executable,'-m','pip','install','-q','torch==2.13.0','numpy==2.5.0','scipy==1.18.0','matplotlib==3.11.0'])"),nbf.v4.new_markdown_cell(prose(True)),nbf.v4.new_markdown_cell('## Attribution and release identity\n\nPort of PetarV-/GAT, pinned '+manifest['revision']+'. Original release is preserved in labs/sources/l084. Modern framework differences are in the reproduction contract.\n\n```text\n'+(LAB/'sources/l084/LICENSE').read_text()+'\n```')]
  for node in ast.parse(source).body:
   if isinstance(node,ast.Expr):continue
   code=ast.get_source_segment(source,node)
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
    cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in CHECKS else '### PROVIDED · ')+node.name+'\n\n'+(ast.get_docstring(node) or 'Read the complete implementation.')))
    if not solution and node.name in CHECKS:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+node.name+' and pass the CHECK")'
   cells.append(nbf.v4.new_code_cell(code))
   if isinstance(node,ast.FunctionDef) and node.name in CHECKS:cells.append(nbf.v4.new_code_cell(CHECKS[node.name]))
  cells.extend([
   nbf.v4.new_markdown_cell('## PROVIDED · data identity and transductive boundary\n\nPinned files are hashed before deserialization. Features are row-normalized; no fitted statistics cross the split. All features/edges are visible, while only the 140 training labels enter the objective.'),
   nbf.v4.new_code_cell('manifest='+repr(manifest)+"\ntorch.set_num_threads(1)\ndata=load_cora(Path.cwd(),manifest)\nassert data[0].shape==(2708,1433)\nassert data[1].shape==(2,13264)\nassert [len(t) for t in data[3:]]==[140,500,1000]\nprint('Cora verified: all features, fixed label masks')"),
   nbf.v4.new_markdown_cell('## Teaching lane · deliberately truncated\n\nFive epochs verifies connectivity of your implementation. It is not a benchmark score. Change an edge orientation and watch your CHECK fail before training.'),
   nbf.v4.new_code_cell("teaching=train_cora(data,999,max_epochs=5)\nprint('TEACHING ONLY',teaching['test_accuracy'])"),
   nbf.v4.new_markdown_cell('## One full seed · actual release stopping rule\n\nFull data, width and stopping schedule. A single seed is not the paper\'s 100-run statistic. The resulting first-head coefficients describe routing; they are not causal importance.'),
   nbf.v4.new_code_cell("one,model=train_cora(data,0,return_model=True)\nprint({k:v for k,v in one.items() if k!='trace'})\nwith torch.no_grad():\n    _,alpha=model.hidden[0](data[0],data[1],True)\nmask=data[1][0]==0\nattention={'receiver':0,'senders':data[1][1,mask].tolist(),'coefficients':alpha[mask].tolist()}\nassert abs(sum(attention['coefficients'])-1)<1e-6\nprint(attention)\nPath('l084-inline-seed0.json').write_text(json.dumps(one,indent=2))"),
   nbf.v4.new_code_cell("import matplotlib.pyplot as plt\nfig,ax=plt.subplots(figsize=(8,3))\nax.bar([str(v) for v in attention['senders']],attention['coefficients'])\nax.axhline(1/len(attention['senders']),color='orange',linestyle='--',label='Uniform')\nax.set(xlabel='Sender node ID',ylabel='Attention coefficient',title='Your trained model: receiver 0, hidden head 1')\nax.legend()\nplt.show()"),
   nbf.v4.new_markdown_cell('## Full named experiment · 100 fresh seeds\n\nSet the flag to True to complete the published repetition count. Seed 0 reuses the full fresh run immediately above, not an author checkpoint. No shortened schedule or model. Original-framework parity remains separate. Check environment and source hashes before comparing.'),
   nbf.v4.new_code_cell("RUN_100_SEEDS = False\nif RUN_100_SEEDS:\n    runs=[one]\n    for seed in range(1,100):\n        runs.append(train_cora(data,seed))\n        print(seed,runs[-1]['test_accuracy'])\n    result=summarize(runs)\n    Path('l084-student-results.json').write_text(json.dumps(result,indent=2))\n    print(result['mean'],result['sample_sd'])\nelse:\n    print('100-run notebook lane NOT_RUN; one full seed executed above')"),
   nbf.v4.new_markdown_cell('## EXIT · attach runnable evidence\n\nSubmit your JSON and a written explanation of receiver softmax, concat/mean, transductive feature access, the OR-reset/AND-save rule and why attention is not causal importance. Include a deliberately failing CHECK and its repair. Derive why changing the receiver cannot reverse two shared senders\' ranking for a fixed original-GAT head.'),
   nbf.v4.new_code_cell("artifact={'lesson':84,'source_revision':manifest['revision'],'one_seed_accuracy':one['test_accuracy'],'full_100_seeds':RUN_100_SEEDS,'historical_parity':'INCOMPARABLE','attention':attention,'explanation':'ADD YOUR OWN EXPLANATION'}\nPath('l084-exit.json').write_text(json.dumps(artifact,indent=2))\nprint(artifact)")])
  nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
  path=LAB/('solutions' if solution else '')/f'{SLUG}.ipynb'
  if solution and path.exists():
   previous=nbf.read(path,as_version=4)
   old=[c for c in previous.cells if c.cell_type=='code'];new=[c for c in nb.cells if c.cell_type=='code']
   if [c.source for c in old]==[c.source for c in new]:
    for before,after in zip(old,new):after.outputs=before.outputs;after.execution_count=before.execution_count
  nbf.write(nb,path)
  if not solution:
   html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
 ref='''# GAT contract card

| Object | Meaning |
|---|---|
| Edge `(i,j)` | Receiver i receives sender j |
| Score | LeakyReLU(q_i + k_j), slope .2; release includes biases |
| Neighbor softmax | Normalize only allowed senders of one receiver, one head |
| Message | alpha_ij times projected sender value |
| Hidden merge | Concatenate 8 heads × 8 features = 64 |
| Cora output | One head, 7 class logits; multiple output heads average |
| Dropout | Input, projected values, coefficients; .6, independent per head |
| Eval coefficients | Sum to 1 per receiver; training dropout breaks that sum |
| Mask | Citation neighbors plus exactly one self-loop |
| Training | Full graph/features, 140 labels; CE plus L2 on all parameters |
| Selection | 500 labels; OR-reset and AND-save, equality counts; patience100 |
| Test | 1000 nodes after checkpoint restore |
| Paper target | Table2 Cora 83.0 ± .7%, 100 runs |

**Interpretation:** weight is not contribution or causal importance. Compare alpha times values and specify an intervention. Original additive GAT cannot reverse shared sender rankings by changing only the receiver term.

**Evidence:** modern PyTorch release port; historical TensorFlow/RNG parity INCOMPARABLE. Read the deviation ledger before comparing scores.

[Lesson](../lessons/0084-gat.html) · [Lab](../labs/0084-gat.ipynb) · [Reproduction](../labs/l084-reproduction.md) · [Paper](https://arxiv.org/html/1710.10903v3)
'''
 (ROOT/'reference'/f'{SLUG}.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GAT contract card</title><link rel="stylesheet" href="../assets/lesson.css"><article>'+render(ref)+'</article></html>')
 print('Built L084 lesson, notebooks, reference')
if __name__=='__main__':build()
