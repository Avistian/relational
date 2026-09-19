"""Build L087 from canonical prose, visible implementation and recorded evidence."""
from _walkthrough_delivery import snapshot, finalize
snapshot(87)
import ast,base64,json,re,importlib.metadata as md
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0087-link-prediction';TITLE='Link prediction: hide the edge, define the candidates'
figures={'CD_FIG':('cd','Equal-dataset mean ranks; the thick bar connects RA and AA, whose gap is below the Nemenyi critical difference.'),'SPLIT_FIG':('split','Synthetic neighbor-sum trace: a forbidden reverse edge changes C from 0 to 1.'),'ARCH':('architecture','Teaching GCN encoder, shared weights, dot-product arithmetic, loss and selection boundary.'),'RANK_FIG':('ranking','Synthetic tied-score example: average rank 2.5 and reciprocal rank .4.'),'SEAL_FIG':('seal','SEAL enclosing subgraph and DRNL labels for roots A,C; full classifier shown as a reading bridge, not executed.'),'RESULT_FIG':('results','Fresh author baseline reconstruction: ten split points per dataset, means and conditional t95% intervals; historical split identity unverified.')}
paper=json.loads((LAB/'_paper_l087_results.json').read_text());teaching=json.loads((LAB/'_teaching_l087_results.json').read_text())
# Published Table1 targets, percent mean +/- reported SD. Fresh candidates differ.
targets={'USAir':[(93.80,1.22),(95.06,1.03),(95.77,.92)],'NS':[(94.42,.95),(94.45,.93),(94.45,.93)],'PB':[(92.04,.35),(92.36,.34),(92.46,.37)],'Yeast':[(89.37,.61),(89.43,.62),(89.45,.62)],'Celegans':[(85.13,1.61),(86.95,1.40),(87.49,1.41)],'Power':[(58.80,.88),(58.79,.88),(58.79,.88)],'Router':[(56.43,.52),(56.43,.51),(56.43,.51)],'Ecoli':[(93.71,.39),(95.36,.34),(95.95,.35)]}
(LAB/'_targets_l087.json').write_text(json.dumps({'source':'https://proceedings.neurips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf','table':1,'methods':['CN','AA','RA'],'unit':'AUC percent; published mean and SD','targets':targets},indent=2)+'\n')
def results():
 s='| Dataset | CN: ours / paper | AA: ours / paper | RA: ours / paper |\n|---|---|---|---|\n'
 for name,row in paper['summary'].items():
  vals=[]
  for i,m in enumerate(['CN','AA','RA']):
   v=row[m];tar=targets[name][i];vals.append(f"{100*v['mean']:.2f} ± {100*v['sample_sd']:.2f} / {tar[0]:.2f} ± {tar[1]:.2f}")
  s+='| '+name+' | '+' | '.join(vals)+' |\n'
 st=paper['statistics'];s+='\nAll entries are AUC percentages, mean ± sample SD. Paper targets are transcribed from Table 1; closeness is descriptive, not an equivalence test.\n\n'
 s+=f"Across eight equally weighted datasets, CN/AA/RA mean ranks are {', '.join(f'{x:.3f}' for x in st['mean_ranks'])}. Friedman p={st['friedman_p']:.4g}; Nemenyi critical difference at .05 is {st['nemenyi_cd_05']:.3f}. Ranks are computed from each dataset’s mean AUC; seeds are not separate units. This three-method benchmark is not evidence about GNN superiority.\n"
 return s

def teaching_results():
 s='**Separate author teaching run: USAir GCN with 50 filtered negatives per query.**\n\n| Seed | Selected epoch | Test AUC | MRR | Hits@10 |\n|---|---|---|---|---|\n'
 for r in teaching['runs']:s+=f"| {r['seed']} | {r['selected_epoch']} | {r['test_auc']:.4f} | {r['ranking']['mrr']:.4f} | {r['ranking']['hits_at_k']:.4f} |\n"
 return s

def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text()
 for tag,(name,caption) in figures.items():
  src='data:image/png;base64,'+base64.b64encode((LAB/f'figures/l087/{name}.png').read_bytes()).decode() if portable else f'../labs/figures/l087/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 replacements={'WARMUP':('warmup','Write the three retrieval answers before continuing.'),'LEAK_WIDGET':('edge-leak','Predict and check: adding A → C changes H_C from 0 to 1; the safe baseline remains 0.'),'RANK_WIDGET':('edge-rank','Predict: raise the positive score from .5 to .7. Its rank changes from 2.5 to 1, with fixed negatives.'),'PREDICT':('prediction','Commit your prediction before reading the table. A rare common neighbor contributes more than a popular one under AA/RA.'),'TEACHBACK':('teachback','Explain why an edge-level test mask is insufficient, and why MRR needs an explicit candidate policy. Include both message directions and temporal availability.')}
 for tag,(id,text) in replacements.items():s=s.replace('[['+tag+']]',text if portable else f'<div id="{id}"></div>')
 s=s.replace('[[RESULTS]]',results()).replace('[[TEACHING]]',teaching_results())
 if portable:
  s=s.replace('](0086-pyg-fundamentals.html)','](https://avistian.github.io/relational/lessons/0086-pyg-fundamentals.html)');s=s.replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','mpnn-lesson','link-prediction-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0086-pyg-fundamentals.html">Lesson 86</a></nav><header><p>Year 3 · Quarter 1 · Lesson 087</p><h1>{TITLE}</h1></header><aside><a href="../labs/{SLUG}.ipynb">Download lab</a> · <a href="https://colab.research.google.com/github/Avistian/relational/blob/main/labs/{SLUG}.ipynb">Open in Colab</a> · <a href="../labs/html/{SLUG}.html">Read lab</a> · <a href="../labs/solutions/{SLUG}.ipynb">Solution</a></aside>'
footer='</article>'+''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','link-prediction-viz','l087-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
checks={
 'split_edges':"e=np.array([[0,1],[1,0],[0,2],[0,3],[1,2],[1,3],[2,3]])\na,b,c=split_edges(e,4,87,.2,.2)\nsets=[set(map(tuple,p)) for p in (a,b,c)]\nassert list(map(len,sets))==[4,1,1]\nassert all(not sets[i]&sets[j] for i in range(3) for j in range(i))\nassert len(set.union(*sets))==6\nprint('CHECK: reverse copies belong to the same canonical pair')",
 'edge_logits':"z=torch.tensor([[1.,2.],[3.,4.],[-1.,2.]],requires_grad=True)\ny=edge_logits(z,torch.tensor([[0,1],[0,2]]))\ntorch.testing.assert_close(y,torch.tensor([11.,3.]))\ny.sum().backward();torch.testing.assert_close(z.grad,torch.tensor([[2.,6.],[1.,2.],[1.,2.]]))\nprint('CHECK: correct per-pair scores and endpoint gradients')",
 'ranking_metrics':"r=ranking_metrics(np.array([.5,.8]),np.array([[.6,.5,.1],[.7,.2,.1]]),k=1)\nnp.testing.assert_allclose(r['ranks'],[2.5,1]);assert r['mrr']==.7 and r['hits_at_k']==.5\nprint('CHECK: tied rank and macro query aggregation')"}
source=(LAB/'relkit/link_l087.py').read_text();nodes=ast.parse(source).body
pins={n:md.version(n) for n in ['numpy','scipy','torch','scikit-learn','pandas','matplotlib','nbformat','nbclient','nbconvert','ipykernel']}
(LAB/'requirements-l087-runtime.txt').write_text('# Author environment: Python 3.12 CPU. Install torch CPU separately per guide.\n'+'\n'.join(f'{k}=={v}' for k,v in pins.items())+'\n')
bootstrap="""# @colab-bootstrap — inline lab, no checkout or private modules needed.
import sys, subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--index-url',
                          'https://download.pytorch.org/whl/cpu', 'torch==TORCHPIN'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', NUMPYPIN, SCIPYPIN, SKPIN])
import importlib.metadata as metadata
print({name: metadata.version(name) for name in ['numpy','scipy','torch','scikit-learn']})
""".replace('TORCHPIN',pins['torch']).replace('NUMPYPIN',repr('numpy=='+pins['numpy'])).replace('SCIPYPIN',repr('scipy=='+pins['scipy'])).replace('SKPIN',repr('scikit-learn=='+pins['scikit-learn']))
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nTier B: real static networks from the SEAL release. Three live TODOs; a full-size named baseline reconstruction is provided after EXIT. This is not SEAL classifier training. Author-reference evidence is labeled separately from your kernel output.'),nbf.v4.new_code_cell(bootstrap)]
 cells += [nbf.v4.new_markdown_cell(x) for x in re.split(r'(?=^## )',prose(True),flags=re.M) if x.strip()]
 cells.append(nbf.v4.new_markdown_cell('## PROVIDED · visible implementation\n\nRead each contract before running it. The code below is the canonical implementation used to generate the author results. The three TODO functions remain live in training/evaluation; no solution module is imported.'))
 imports='\n'.join(ast.get_source_segment(source,n) for n in nodes if isinstance(n,(ast.Import,ast.ImportFrom)))
 cells.append(nbf.v4.new_code_cell(imports+'\ntorch.set_num_threads(1)'))
 for node in nodes:
  if not isinstance(node,(ast.FunctionDef,ast.ClassDef)):continue
  code=ast.get_source_segment(source,node);doc=ast.get_docstring(node) or ''
  cells.append(nbf.v4.new_markdown_cell(('### TODO · ' if node.name in checks else '### PROVIDED · ')+node.name+'\n\n'+doc.replace('TODO: ','')))
  if node.name in checks and not solution:code=code.split('\n',1)[0]+'\n    raise NotImplementedError("Implement '+node.name+' using the contract and CHECK")'
  cells.append(nbf.v4.new_code_cell(code))
  if node.name in checks:cells.append(nbf.v4.new_code_cell(checks[node.name]))
 # The verification body resolves notebook globals, including student functions.
 vsrc=(LAB/'_verify_l087.py').read_text();vnode=next(n for n in ast.parse(vsrc).body if isinstance(n,ast.FunctionDef) and n.name=='verify')
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · adversarial graph contracts\n\nPredict the isolated node’s heuristic scores before running. This checks actual notebook functions, including your TODOs.'),nbf.v4.new_code_cell(ast.get_source_segment(vsrc,vnode)+'\nverification=verify()')])
 manifest=json.loads((LAB/'_sources_l087.json').read_text())
 cells.extend([nbf.v4.new_markdown_cell('## PROVIDED · pinned data and fresh training\n\nDownload only the pinned data, verify its SHA256, then train the teaching model. Seed 87 matches one author-reference run. The full graph is used; there is no node/edge cap.'),nbf.v4.new_code_cell('manifest='+repr(manifest)+"\ndata_root=Path.cwd()/'l087-source'\nfull_edges,n=load_graph('USAir',manifest,data_root)\nassert n==332 and len(full_edges)==2126\nrun,arrays=train_link(full_edges,n,seed=87)\nnp.savez_compressed('l087-teaching-artifact.npz',**arrays)\nprint({k:v for k,v in run.items() if k not in ['trace','ranking']})\nprint({k:v for k,v in run['ranking'].items() if k!='ranks'})")])
 cells.extend([nbf.v4.new_markdown_cell('## CHECK · change candidate count, keep model fixed\n\nPredict before running: taking the first 10 of the same 50 negatives cannot worsen a positive’s rank. This makes the evaluation easier without changing embeddings.'),nbf.v4.new_code_cell("small=ranking_metrics(arrays['positive_scores'],arrays['negative_scores'][:,:10],k=10)\nassert small['mrr']>=run['ranking']['mrr']\nassert small['hits_at_k']>=run['ranking']['hits_at_k']\nprint('50 negatives:',run['ranking']['mrr'],'10 negatives:',small['mrr'])")])
 cells.extend([nbf.v4.new_markdown_cell('## EXIT · artifact plus explanation\n\nAdd your written information boundary and candidate contract to this JSON. Include why it does not reproduce the SEAL classifier. Bring the JSON and NPZ to the agent for assessment; running the solution does not demonstrate mastery.'),nbf.v4.new_code_cell("exit_record={'verification':verification,'teaching':run,'candidate_intervention':small,'artifact_sha256':hashlib.sha256(Path('l087-teaching-artifact.npz').read_bytes()).hexdigest(),'explanation':'WRITE YOUR EXPLANATION HERE','paper_classifier':'NOT_RUN'}\nPath('l087-exit.json').write_text(json.dumps(exit_record,indent=2))\nprint('Wrote l087-exit.json and l087-teaching-artifact.npz')")])
 cells.extend([nbf.v4.new_markdown_cell('## NEXT STEP · full named paper experiment\n\nThis is the complete eight-network, ten-split CN/AA/RA baseline reconstruction, not a larger GCN teaching run. Expect roughly a minute on the author CPU. Python/MATLAB candidate permutations differ; native replay commands and unrun boundaries are in the reproduction guide.'),nbf.v4.new_code_cell("RUN_PAPER_REPRO="+str(solution)+"\nif RUN_PAPER_REPRO:\n    full=run_paper(manifest,data_root,Path.cwd()/'l087-paper-artifacts')\n    Path('l087-paper-results.json').write_text(json.dumps(full,indent=2))\n    display(full['summary'])\nelse:\n    print('Full paper baseline track NOT_RUN in this kernel; set RUN_PAPER_REPRO=True')")])
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 p=LAB/('solutions/' if solution else '')/(SLUG+'.ipynb')
 if solution and p.exists():
  previous=nbf.read(p,as_version=4)
  if [(c.cell_type,c.source) for c in previous.cells]==[(c.cell_type,c.source) for c in nb.cells]:nb=previous
 nbf.write(nb,p)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
ref='''# Link prediction: the protocol card

| Object | Required contract |
|---|---|
| Undirected split | canonical u<v pairs; split before adding reverse messages |
| Context graph | only declared visible edges; distinguish supervision pairs |
| Negative | candidate treated as absent; specify filtering and replacement |
| Dot decoder | one logit per pair; sum endpoint products; symmetric |
| Average tied rank | 1 + strictly higher + half the equal negative scores |
| MRR | average reciprocal rank over declared queries |
| Hits@k | fraction whose rank is at most k under the same tie convention |
| ROC AUC | positive versus negative wins, with half credit for ties |
| DRNL | roots 1; opposite-root-deleted distances; unreachable 0 |
| Temporal recommendation | history and candidate eligibility valid at query time |

Example: positive .5, negatives [.6,.5,.1] → rank2.5 → reciprocal .4.
Never compare sampled MRR with all-item MRR as if they were the same experiment.

Paper target here: Zhang & Chen2018 Table1 CN/AA/RA, 8 datasets ×10 splits. Complete Python protocol reconstruction executed; historical numerical identity INCOMPARABLE. Full SEAL classifier NOT_RUN.

[Lesson87](../lessons/0087-link-prediction.html) · [Lab](../labs/0087-link-prediction.ipynb) · [Reproduction guide](../labs/l087-reproduction.md) · [Primary paper](https://arxiv.org/abs/1802.09691)
'''
(ROOT/'reference/link-prediction.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Link prediction reference</title><link rel="stylesheet" href="../assets/lesson.css"></head><body><article>'+render(ref)+'</article></body></html>')
print('Built L087 lesson, student/solution notebooks, prepared HTML and reference')
finalize(87)
