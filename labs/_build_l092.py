"""Single-source HTML, portable notebooks and prepared preview for L092."""
import ast,base64,hashlib,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune as render
ROOT=Path(__file__).resolve().parents[1];LAB=ROOT/'labs';SLUG='0092-meta-paths';TITLE='Meta-paths: two attention decisions in HAN'
manifest=json.loads((LAB/'_sources_l092.json').read_text())
figs={'NODE_FIG':('neighbors','Receiver attention changes from [.25,.50,.25] to [1/7,2/7,4/7]; fixed sender values yield 3 then 3.857.'),'PATH_FIG':('paths','Two shared-author walks collapse into one PAP endpoint edge; multiplicity is discarded.'),'ARCH_FIG':('architecture','Full ACM HAN release port: two eight-head branches, semantic fusion, masked training and a separate KNN evaluation.'),'SEMANTIC_FIG':('semantic','Softmax of mean scores differs from mean per-node softmax: 0.731 versus 0.690 PAP weight.')}
def results():
 path=LAB/'_paper_l092_results.json'
 if not path.exists():return '**Author full run: RUNNING. No measured paper result is claimed yet.**'
 p=json.loads(path.read_text());r=p['runs'][0]
 s='**Author-reference evidence; not your current kernel output.** '+str(len(p['runs']))+' complete encoder fit; '+str(r['epochs_completed'])+' epochs executed; validation selected epoch '+str(r['selected_epoch'])+'. Direct test-head accuracy '+f"{100*r['test_accuracy']:.2f}%"+'.\n\n'
 s+='| KNN training fraction | Measured macro-F1 ± split SD | Paper macro | Measured micro-F1 ± split SD | Paper micro |\n|---|---:|---:|---:|---:|\n'
 for row in p['table3_acm']:
  s+=f"| {row['fraction']:.0%} | {100*row['macro_mean']:.2f} ± {100*row['macro_split_sd']:.2f}% | {100*row['paper_macro']:.2f}% | {100*row['micro_mean']:.2f} ± {100*row['micro_split_sd']:.2f}% | {100*row['paper_micro']:.2f}% |\n"
 beta=r['semantic_mean_all_nodes'];s+=f"\nMean released semantic weights over all nodes: **PAP {beta[0]:.4f}, PSP {beta[1]:.4f}**. The higher-weight path is **{'PAP' if beta[0]>beta[1] else 'PSP'}**. These are averaged node-specific weights, not the paper-global coefficient.\n\n"
 gates=[]
 for row in p['table3_acm']:
  inside=all(abs(row[key+'_mean']-row['paper_'+key])<=.02 for key in ['macro','micro'])
  gates.append(f"{row['fraction']:.0%}: "+('both within' if inside else 'at least one outside'))
 s+='Frozen ±2pp descriptive tolerance: '+ '; '.join(gates)+'. This does not change the INCOMPARABLE protocol verdict.\n\n'
 s+='Each SD is over ten dependent probe splits of one fixed embedding. There is no encoder-seed uncertainty estimate. The full run retains predictions and validation traces. Numerical proximity does not remove the protocol differences.\n\n[[RESULT_FIG]]'
 return s

def prose(portable=False):
 s=(ROOT/'lessons/content'/f'{SLUG}.md').read_text().replace('[[RESULTS]]',results())
 allfigs=dict(figs)
 if (LAB/'figures/l092/results.png').exists():allfigs['RESULT_FIG']=('results','Full-run KNN split scores versus Table3 targets. Probe variability is conditional on one embedding; historical protocol is incomparable.')
 for tag,(name,caption) in allfigs.items():
  p=LAB/f'figures/l092/{name}.png';src='data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode() if portable else f'../labs/figures/l092/{name}.png'
  s=s.replace('[['+tag+']]',f'<figure class="mpnn-figure"><div class="figure-scroll" tabindex="0"><img src="{src}" alt="{caption}"></div><figcaption>{caption}</figcaption></figure>')
 for tag,id_,fallback in [('WARMUP','warmup','**Cold retrieval:** answer the three prompts before reading onward.'),('PREDICT','prediction','**Predict before computing:** can another node change your mixture under each semantic rule?'),('SEMANTIC_WIDGET','semantic','**Intervene on the arithmetic:** replace node 1’s PAP score 0 with 4. Global node 0 weight becomes softmax([3,0]) ≈ .953; released node 0 weight stays .881.'),('TEACHBACK','teachback','**Teach-back:** write your explanation before comparing it with the equations above.')]:
  s=s.replace('[['+tag+']]',fallback if portable else f'<div id="{id_}"></div>')
 if portable:
  s=re.sub(r'\]\((00\d\d-[^)]+\.html)\)',r'](https://avistian.github.io/relational/lessons/\1)',s)
  s=s.replace('](../','](https://avistian.github.io/relational/')
 return s
head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 92 — {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{s}.css">' for s in ['lesson','mpnn-lesson','hetero-graph-viz','metapath-viz'])+'</head><body><article>'
head+=f'<nav><a href="../index.html">Course</a> · <a href="0091-r-gcn.html">Lesson 91</a> · <a href="../reference/meta-paths.html">Reference</a></nav><header><p>Year 3 · Quarter 2 · Lesson 092</p><h1>{TITLE}</h1></header>'
footer='</article>'+''.join(f'<script src="../assets/{s}.js"></script>' for s in ['retrieval-pool','retrieval-bank','predict','teachback','metapath-viz','l092-lesson'])+'</body></html>'
(ROOT/'lessons'/f'{SLUG}.html').write_text(head+render(prose())+footer)
checks={
'metapath_reachability':"b=sp.csr_matrix([[1,1],[1,1],[0,1]],dtype=np.float32)\na=metapath_reachability([b,b.T]);assert a.nnz==9 and a[0,1]==1\nc=sp.csr_matrix([[1,1],[1,1],[0,0]],dtype=np.float32)\nassert metapath_reachability([c,c.T]).nnz==4\nprint('CHECK: duplicate walks collapse; changed endpoint membership changes reachability')",
'neighbor_attention':"h=torch.tensor([[1.],[3.],[5.]],requires_grad=True)\ne=torch.tensor([[0,0,0,1,2],[0,1,2,1,2]])\nl=torch.zeros(3);r=torch.tensor([0.,np.log(2),0.],dtype=torch.float32,requires_grad=True)\no=neighbor_attention(h,e,l,r,0.,False)\ntorch.testing.assert_close(o,torch.tensor([[3.],[3.],[5.]]))\no.sum().backward();assert r.grad.abs().sum()>0\nprint('CHECK: receiver normalization, sender indexing, gradient flow')",
'semantic_fusion':"z=torch.tensor([[[2.,0.],[0.,0.]],[[0.,0.],[0.,0.]]],requires_grad=True)\nw=torch.eye(2,requires_grad=True);b=torch.zeros(2);q=torch.tensor([1.,0.],requires_grad=True)\nscores=torch.tanh(z)@q\ng,bg=semantic_fusion(z,w,b,q,'paper_global');n,bn=semantic_fusion(z,w,b,q,'release_node')\ntorch.testing.assert_close(bg,scores.mean(0).softmax(0).expand(2,-1))\ntorch.testing.assert_close(bn,scores.softmax(1));assert not torch.allclose(bg,bn)\ng.sum().backward();assert q.grad.abs().sum()>0 and w.grad.abs().sum()>0\nprint('CHECK: both semantic reduction orders and learned gradients')"}
desc={'metapath_reachability':('Compose typed relations','Multiply compatible incidence matrices, then return binary endpoint reachability. Preserve self when reachable; discard counts.'),'neighbor_attention':('Attend within a route','Compute stable receiver-row softmax, coefficient dropout, then weighted sender aggregation. Edges are sorted unique receiver/sender pairs; h already contains projected-value dropout.'),'semantic_fusion':('Combine route representations','Compute shared semantic scores, implement paper_global and release_node reduction orders, then return the fused embedding and node-by-path weights.')}
bootstrap="""# @colab-bootstrap — no hidden model imports; exact observed and portable versions are separate.
import sys,subprocess
if 'google.colab' in sys.modules:
    subprocess.check_call([sys.executable,'-m','pip','install','numpy==2.2.6','scipy==1.15.3','torch==2.8.0','scikit-learn==1.7.2'])
import importlib.metadata as metadata
print({name:metadata.version(name) for name in ['numpy','scipy','torch','scikit-learn']})
"""
annotations={
 'AttentionHead':'Paper §4.1, Eqs1–5; release utils/layers.py attn_head. Follow all three dropout sites and the per-receiver normalization.',
 'HAN':'Paper Figure2 and §4.2; release models/gat.py HeteGAT_multi. Keep separate path/head parameters and inspect the classifier handoff.',
 'load_acm':'Release ex_acm3025.py load_data_dblp plus the DGL mirror loader. Hashes and original split IDs determine this experiment.',
 'masked_loss':'Paper Eq10 and release BaseGAttN.masked_softmax_cross_entropy. Only these training indices supervise the encoder.',
 'knn_probe':'Paper §5.4/Table3 and release jhyexp.py my_KNN. The probe fits its own labels on frozen test-node embeddings.',
 'train_acm':'Paper §5.3 versus release ex_acm3025.py training loop. Trace L2 scope and the OR-reset / AND-save selection rule.',
 'summarize':'Paper Table3 ACM targets. Aggregate probe repetitions without pretending they are independent graph datasets.'}
source=(LAB/'relkit/han_l092.py').read_text();lines=source.splitlines()
for solution in [False,True]:
 cells=[nbf.v4.new_markdown_cell('# '+TITLE+'\n\nPROVIDED defines the complete mechanism. TODO cells are yours. Run CHECK immediately after each. EXIT requires your evidence and written defense. The full-run switch is off in the student copy; enable it only after the checks pass.'),nbf.v4.new_code_cell(bootstrap)]
 cells += [nbf.v4.new_markdown_cell(chunk) for chunk in re.split(r'(?=\n## )',prose(True)) if chunk.strip()]
 cells += [nbf.v4.new_markdown_cell('## PROVIDED · Provenance and runtime\nThe digest is checked before loading the public pickle. No normalization or split regeneration is applied. This is an independently written modern port of the pinned release; it is not the original TensorFlow runtime.'),nbf.v4.new_code_cell('SOURCE_MANIFEST='+repr(manifest))]
 for node in ast.parse(source).body:
  code='\n'.join(lines[node.lineno-1:node.end_lineno]);name=getattr(node,'name','')
  if name in checks:
   title,goal=desc[name];cells.append(nbf.v4.new_markdown_cell('### TODO · '+title+'\n**Goal:** '+goal+'\n**Why:** this function sits on the actual training path. Use the dimensions and worked example above; CHECK validates behavior.'))
   if not solution:code=code[:code.index(':')+1]+'\n    raise NotImplementedError("Complete this task before CHECK")'
   cells.append(nbf.v4.new_code_cell(code,metadata={'tags':['solution' if solution else 'todo'],'task':name}))
   cells.append(nbf.v4.new_code_cell('# CHECK\n'+checks[name],metadata={'tags':['check']}))
  else:
   if name:cells.append(nbf.v4.new_markdown_cell('### PROVIDED · '+name.replace('_',' ')+'\n'+annotations.get(name,'Trace the tensor dimensions, randomness, and label access in this part of the released computation.')))
   cells.append(nbf.v4.new_code_cell(code))
 cells += [nbf.v4.new_markdown_cell('## CHECK · Real ACM diagnostic and paired semantic intervention\nTwo full-graph updates test wiring, not paper performance. Keep weights fixed, then change only semantic reduction. This is an inference intervention, not a trained ablation.'),nbf.v4.new_code_cell("""torch.set_num_threads(2)
DATA_ROOT=Path.cwd()/'l092-data'
data=load_acm(DATA_ROOT,SOURCE_MANIFEST)
assert load_acm.__globals__['metapath_reachability'] is metapath_reachability
assert AttentionHead.forward.__globals__['neighbor_attention'] is neighbor_attention
assert HAN.forward.__globals__['semantic_fusion'] is semantic_fusion
smoke,model=train_acm(data,seed=99,epochs=2)
x,y,edges,train,val,test,meta=data
model.eval()
with torch.no_grad():
    original,z,beta=model(x,edges)
    model.mode='paper_global'
    changed,zg,bg=model(x,edges)
    assert not torch.allclose(original,changed)
    assert torch.allclose(bg[0],bg[-1])
print({'diagnostic_epochs':2,'max_logit_change_same_weights':float((changed-original).abs().max()),'data':meta})
"""),nbf.v4.new_markdown_cell('## RUN · Full released schedule and Table3 ACM probe\nEnable after the checks pass. One encoder fit, up to200 epochs with released early stopping, all40 probe evaluations. Seeds: encoder0, probe1000. The paper/code/mirror differences keep historical parity INCOMPARABLE. No saved-score substitution occurs.'),nbf.v4.new_code_cell("""RUN_FULL_REPRO = False
paper_result=None
if RUN_FULL_REPRO:
    full_run,full_model=train_acm(data,seed=0,epochs=200,mode='release_node',progress=True)
    paper_result=summarize([full_run],data[-1])
    Path('l092-paper.json').write_text(json.dumps(paper_result,indent=2))
    print(paper_result['table3_acm'])
    print({'semantic_mean':full_run['semantic_mean_all_nodes'],'historical_parity':'INCOMPARABLE'})
else:
    print('Full reproduction lane: NOT_RUN in this kernel')
"""),nbf.v4.new_markdown_cell('## EXIT · Evidence plus written defense\nWrite150–250 words in a new markdown cell. Derive both softmax axes, state which path receives greater mean weight, distinguish that average from paper-global attention, and explain why KNN repetitions are not independent model fits. Name at least three historical reproduction gaps. Ask the teacher to challenge your report.'),nbf.v4.new_code_cell("""exit_ticket={'lesson':92,'full_release_execution':paper_result['status'] if paper_result else 'NOT_RUN',
 'historical_parity':'INCOMPARABLE','full_paper_parity':'NOT_ESTABLISHED','learner_mastery':'PENDING_WRITTEN_DEFENSE',
 'source_revision':SOURCE_MANIFEST['source_revision'],'data_sha256':SOURCE_MANIFEST['data']['sha256']}
Path('l092-exit.json').write_text(json.dumps(exit_ticket,indent=2));print(exit_ticket)
""")]
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
 path=LAB/('solutions' if solution else '')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 for i,c in enumerate(nb.cells):c.id=f'l092-{i:03}-'+hashlib.sha256(c.source.encode()).hexdigest()[:8]
 nbf.validate(nb);nbf.write(nb,path)
 if not solution:
  html,_=HTMLExporter().from_notebook_node(nb);(LAB/'html'/f'{SLUG}.html').write_text(html)
