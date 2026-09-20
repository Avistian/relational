"""One prose source and one visible implementation build lesson and portable notebooks."""
import ast,base64,hashlib,json,re
from pathlib import Path
from _walkthrough_delivery import snapshot, finalize
snapshot(97)
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
P=Path(__file__).resolve().parent;R=P.parent;S='0097-negative-sampling';T='Negative sampling: which missing links should teach the model?'
result=json.loads((P/'_experiment_l097_results.json').read_text())
def prose(portable=False):
 s=(R/'lessons/content'/f'{S}.md').read_text()
 src='data:image/png;base64,'+base64.b64encode((P/'figures/l097/pipeline.png').read_bytes()).decode() if portable else '../labs/figures/l097/pipeline.svg'
 s=s.replace('[[ARCHITECTURE]]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="Training-only observed edges and degrees feed a typed sampler, then shared user/item embeddings, pairwise loss, and frozen full-catalog evaluation."></div><figcaption>Sampling changes the pairs that produce gradients; evaluation changes the competitors used to measure the frozen scorer.</figcaption></figure>')
 s=s.replace('[[INTERVENTION]]','**Portable intervention:** calculate the three distributions in the intervention cell below.' if portable else '<div id="negative-intervention"></div>')
 table='| Training sampler | Full Recall@10 | Full NDCG@10 (fold SD) | Sampled NDCG@10 |\n|---|---:|---:|---:|\n'
 for arm in result['config']['arms']:
  m=result['summary'][arm];table+=f"| {arm} | {m['full_recall']['mean']:.5f} | {m['full_ndcg']['mean']:.5f} ({m['full_ndcg']['fold_sd']:.5f}) | {m['sampled_ndcg']['mean']:.5f} |\n"
 s=s.replace('[[RESULTS]]','**Frozen author measurements; your notebook recomputes all 45 fits.**\n\n'+table+'\nUnder this fixed budget, hard mining has the highest mean full-catalog NDCG; this includes extra candidate scoring. The sampled evaluation is much easier for every arm. These are course results, not BPR-paper scores.')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=re.sub(r'href="(00\d\d-[^"]+\.html)"',r'href="https://avistian.github.io/relational/lessons/\1"',s)
  s=s.replace('href="../','href="https://avistian.github.io/relational/').replace('](../','](https://avistian.github.io/relational/')
 return s
head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 097 — '+T+'</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','negative-sampling'])+'</head><body><article><nav><a href="../index.html">Course</a> · <a href="0096-multi-relational-data.html">Lesson 96</a> · <a href="../reference/negative-sampling-contract.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 097</p><h1>'+T+'</h1></header>'
(R/'lessons'/f'{S}.html').write_text(head+render(prose())+'</article><script src="../assets/negative-sampling-viz.js"></script><script src="../assets/l097-lesson.js"></script></body></html>')
source=(P/'relkit/negative_l097.py').read_text();tree=ast.parse(source)
checks={
 'proposal':"blocked=np.array([[True,True,False,False,False]])\ndegree=np.array([3,1,0,8,1])\nq=proposal(blocked,degree,'uniform')\nassert np.allclose(q,[[0,0,1/3,1/3,1/3]])\nqd=proposal(blocked,degree,'degree')\nassert qd[0,3]>qd[0,4]>qd[0,2]>0\ntry: proposal(np.ones((1,3),bool),np.ones(3),'uniform')\nexcept ValueError: pass\nelse: raise AssertionError('Empty pool must fail')\nprint('PASS: support, weights, zero-degree support, saturation')",
 'draw_negatives':"users=np.zeros(30000,dtype=int)\nj=draw_negatives(users,q,np.random.default_rng(97))\nassert not blocked[users,j].any()\nassert max(abs(np.bincount(j,minlength=5)[2:]/30000-1/3))<.02\nassert draw_negatives(np.array([0]),np.array([[1.,0.]]),np.random.default_rng(0)).item()==0\nassert np.array_equal(draw_negatives(users[:100],q,np.random.default_rng(7)),draw_negatives(users[:100],q,np.random.default_rng(7)))\nprint('PASS: conditional draws, typed identity, frequencies, seed replay')",
 'pairwise_loss':"sp=torch.tensor([0.,2.],requires_grad=True);sn=torch.tensor([0.,-1.],requires_grad=True)\nloss=pairwise_loss(sp,sn)\nassert abs(loss.item()-(np.log(2)+np.log1p(np.exp(-3)))/2)<1e-7\nloss.backward();assert (sp.grad<0).all() and (sn.grad>0).all()\nprint('PASS: stable objective and gradient signs')"}
bootstrap="""# @colab-bootstrap
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','torch==2.8.0'])
import importlib.metadata as metadata
print({k:metadata.version(k) for k in ['numpy','torch']})
"""
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+T+'\n\nTier B · complete ML-100K, five folds × three seeds × three samplers. CPU. Original archive is downloaded from GroupLens and hash-verified. No repository imports. PENDING_WRITTEN_DEFENSE.'),nbf.v4.new_code_cell(bootstrap)]
 cells.extend(nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=\n## )',prose(True)) if x.strip())
 cells.append(nbf.v4.new_markdown_cell('## Visible implementation\n\nThe full experiment runs your functions. Read the data boundary before the sampler; there is no hidden trainer.'))
 prelude='\n\n'.join(ast.get_source_segment(source,n) for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign)))
 cells.append(nbf.v4.new_code_cell(prelude))
 for node in tree.body:
  if not isinstance(node,ast.FunctionDef):continue
  code=ast.get_source_segment(source,node);name=node.name
  cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if name in checks else '### PROVIDED · ')+name+'\n\n'+(ast.get_docstring(node) or '')))
  if name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement the declared contract")'
  cells.append(nbf.v4.new_code_cell(code,metadata={'task':name} if name in checks else {}))
  if name in checks:cells.extend([nbf.v4.new_markdown_cell('### CHECK · '+name),nbf.v4.new_code_cell(checks[name])])
 cells.extend([nbf.v4.new_markdown_cell('## Predict → intervene → explain\n\nBefore execution, predict which probabilities move when item 3 becomes observed. Explain why item 2 may remain a legal negative even if it is positive in heldout data.'),nbf.v4.new_code_cell("blocked=np.array([[True,True,False,False,False]])\ndegree=np.array([3,1,0,8,1])\nprint('uniform',proposal(blocked,degree,'uniform'))\nprint('degree',proposal(blocked,degree,'degree'))\nblocked[0,3]=True\nprint('new observation',proposal(blocked,degree,'degree'))\nassert proposal(blocked,degree,'degree')[0,3]==0"),nbf.v4.new_markdown_cell('## Full reproduction of the declared course experiment\n\nEvery fit starts fresh. Data remain complete, including all five official partitions. This is not a reproduction of the historical BPR experiment. Author execution was approximately 50 CPU seconds; other devices/environments can differ.'),nbf.v4.new_code_cell("fresh=run_suite('l097-data/ml-100k.zip','l097-results')\nPath('l097-fresh.json').write_text(json.dumps(fresh,indent=2)+'\\n')\nassert len(fresh['runs'])==45\nassert all(r['known_positive_collisions']==0 for r in fresh['runs'])\nAUTHOR_SUMMARY="+repr(result['summary'])+"\ngaps={a:abs(fresh['summary'][a]['full_ndcg']['mean']-AUTHOR_SUMMARY[a]['full_ndcg']['mean']) for a in CONFIG['arms']}\nprint('Absolute full-NDCG differences from frozen author references:',gaps)\nprint(json.dumps(fresh['summary'],indent=2))\n# A different package/device can change optimization. Investigate differences; never relabel them as paper parity."),nbf.v4.new_markdown_cell('## EXIT · PENDING_WRITTEN_DEFENSE\n\nSubmit your functions, l097-fresh.json, and the five section 9 answers. Defend the candidate universe and training information boundary before comparing metrics. Ask the agent for feedback; execution alone is not mastery.')])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}});path=P/('solutions' if solution else '')/(S+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(cells):c.id=f'l097-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if solution:
  html,_=HTMLExporter().from_notebook_node(nb);(P/'html'/f'{S}.html').write_text(html)
reference='''# Negative sampling: the contract before the code

[Lesson 97](../lessons/0097-negative-sampling.html) · [Lab](../labs/0097-negative-sampling.ipynb) · [Full protocol](../labs/l097-reproduction.md)

| Object | Define explicitly |
|---|---|
| Typed relation | Source type, relation name, destination type; IDs are local |
| Positive event | Here: fitting rating ≥4 |
| Exclusion mask | Here: all fitting-rated items; test labels never enter fit |
| Candidate pool | Entire destination catalog minus the fitting mask |
| Proposal | Uniform; or (training like-degree+1)^0.75 renormalized per user |
| Hard mining | Highest detached score among four uniform replacement draws |
| Saturation | Empty candidate set raises an error; never retry indefinitely |
| Objective | Mean softplus(s_neg−s_pos), plus the declared L2 term |
| Evaluation | Full catalog minus fitting observations; fixed across training arms |
| Diagnostic | Same scores and test relevance, only 99 sampled distractors |
| Claim | Complete course experiment is not historical BPR paper reproduction |

## Trace from memory

At equal scores, pairwise loss is log(2). Its positive-score derivative is −1/2 before averaging. For a source with observed items {0,1} in a five-item catalog, uniform gives 1/3 to each of {2,3,4}. An unobserved true positive can still be sampled: unknown is not false. Typed user 0 → item 0 is legal when unobserved.

## Inspection checklist

Check destination type, blocked pairs, zero-degree support, replacement policy, RNG seed, empty pool, train-only degree statistics, and both directions of any held-out message edge. Keep the negative pool, encoder neighborhood and evaluation catalog distinct. A candidate sampler is not a leak detector.

## Interpretation

Compare samplers using fixed full-catalog metrics. Compare candidate sets only as an evaluation sensitivity diagnostic. Fold/seed variability is not dataset-level uncertainty. Importance correction p/q targets a different specified expectation and needs full support; hard mining changes with the scorer.

Primary sources: [BPR §4–5](https://arxiv.org/abs/1205.2618), [sampled-metric inconsistency](https://www.ijcai.org/proceedings/2021/0651.pdf), [PyG 2.6.1 implementation](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/utils/_negative_sampling.html), [GroupLens release README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt).
'''
(R/'reference/negative-sampling-contract.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Negative sampling contract</title><link rel="stylesheet" href="../assets/lesson.css"><link rel="stylesheet" href="../assets/mpnn-lesson.css"></head><body><article>'+render(reference)+'</article></body></html>')
print('Built lesson, student, solution, HTML, reference')

finalize(97, preview='solution')
