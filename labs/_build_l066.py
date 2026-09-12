"""Canonical original-checkpoint TabICL package, with five live model operations."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
ROOT=Path(__file__).resolve().parent;SLUG='0066-tabicl-column-row-attention';TITLE='Build TabICL: column memory, row identity and context prediction'
TASKS=[
('attention_mix','def attention_mix(q,k,v):','Mix values using scaled query–key scores','Return attention outputs with the same reader axes as q; normalize over keys and reject empty memory.','The head width controls score scaling; row count does not.','Trace the attention-only example. Preserve arbitrary leading batch/head axes.'),
('context_keys','def context_keys(k,v,n_context):','Enforce the dataset information boundary','Return the context prefix of keys and values while retaining all other axes; reject invalid boundaries.','Both context and query readers use this prefix in every ICL block.','The sequence axis is second from last after head splitting.'),
('rotary_pairs','def rotary_pairs(x,freqs):','Rotate adjacent head-coordinate pairs','Rotate every adjacent pair at its sequence position using the supplied frequency vector; preserve dtype and length.','Rotating halves or omitting CLS positions changes the pretrained score computation.','Construct the perpendicular pair and combine sine/cosine terms; do not rotate value vectors.'),
('inducing_memory','def inducing_memory(src,inducing,n_context,read_block,write_block):','Read context, then broadcast a learned memory','Expand inducing readers across batch dimensions, call the first complete block on context memory, then the second complete block for all cells.','Queries may read summaries but must never write them. Residual and FFN work belongs to each supplied block.','Each block accepts query input first and key/value input second.'),
('conditional_affine','def conditional_affine(values,hidden,weight_head,bias_head,weight_norm,bias_norm):','Apply the distribution-conditioned tokenizer','Project hidden states to separate W and B, apply their separate normalization modules, then combine with scalar values.','Omitting normalization keeps the shape but breaks copied-checkpoint parity.','values has a final singleton coordinate axis; the hidden vector has width 128.')]
CAPTIONS={'architecture':'Complete original February checkpoint: numeric context-only preprocessing, three column ISABs, three RoPE row blocks, twelve ICL blocks and the full head. Labels enter only at ICL. All weights are frozen.','inducing':'Synthetic one-head attention-only fixture with two inducing readers. Matrices expose both softmax operations; the query cannot write the memory. This omits full-block residuals/projections only for the arithmetic illustration.','rope':'Synthetic first adjacent pair with frequency one. The rotation preserves length while the relative-position dot product changes periodically; this is not a monotone distance-decay rule.','roles':'Synthetic C=3, Q=2 role mask for every ICL block, including all context self-keys and no query keys. Query features still supply queries and residual states.','cost':'Conceptual attention-score counts across all blocks/heads, with Q=100 and F=8 fixed. These are arithmetic counts, not GPU peak-memory estimates or measurements.','results':'Fresh original-checkpoint measurements on three complete small datasets and three split seeds. Connected points share fixed query IDs; other context prefixes are nested. These are individual runs, not confidence bands.','ranks':'Exploratory dataset-balanced loss ranks and measured stage CPU seconds. Seeds are averaged within each of three convenience datasets; the paper speedup and full benchmark remain untested.','symmetry':'Executed original-source fixed-weight intervention: context columns have identical marginals. Turning off RoPE nearly collapses swapped query representations. This is not a retrained ablation or reproduction of paper Figure 4.'}
def source():return (ROOT/'relkit/tabicl_l066_v2.py').read_text()
def piece(name):
 return next(ast.get_source_segment(source(),n) for n in ast.parse(source()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
def parts():return re.split(r'(<!--figure:\w+-->|<!--results-table-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def table_markdown():
 s=json.loads((ROOT/'_analysis_l066_v2_results.json').read_text())
 rows=['| Dataset | Context rows | Log loss mean ± SD | Accuracy mean ± SD |','|---|---:|---:|---:|']
 for r in s['summary']:rows.append(f"| {r['dataset']} | {r['context_rows']} | {r['loss_mean']:.4f} ± {r['loss_sd']:.4f} | {r['accuracy_mean']:.4f} ± {r['accuracy_sd']:.4f} |")
 return '\n'.join(rows)+'\n\nAuthor-reference results, three seeds per cell. SD is sample standard deviation, not a confidence interval.'
def figure(name,notebook=False):
 p='figures/l066/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1100 if name in ('inducing','results') else 850 if name in ('ranks','architecture') else 750
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/p).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+p+').'
 return '<figure id="l066-'+('architecture' if name=='architecture' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+p+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+p+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l066_v2_results.json','Measured evidence'),(p+'l066-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 066 · five live model tasks + original pretrained checkpoint experiment + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter.</p></aside>'
def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 066 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Author measurements](_verify_l066_v2_results.json) · [Reproduction](l066-reproduction.md)

**Skill:** implement and trace the complete three-stage pretrained model. **Five TODOs** drive the actual checkpoint forward and fresh context-size experiment. **CHECK** compares whole stages and preprocessing with the original source. **EXIT** binds your measured predictions to live code and weights. Read the complete architecture, then implement one operation at a time.

Default: all 768 diabetes rows, split seed 7, nested contexts 76/230/614 and 154 fixed queries. The author panel adds blood transfusion and WDBC, seeds 7/17/27. This is finite numeric one-view CPU inference, with a 108 MB original February checkpoint download; no synthetic pretraining occurs. Full TALENT benchmark and 500K memory experiment: NOT_RUN. Source license: [BSD 3-Clause](sources/l066-v2/LICENSE).''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:md(figure(m[1],True))
  elif p=='<!--results-table-->':md(table_markdown())
  elif p.strip():md(p.replace('../labs/',''))
 md('''## PROVIDED · Environment and scope
The model, preprocessing and experiment definitions below execute in this notebook's namespace. Your TODOs are global dependencies of these live model methods. The reference package runs separately and contributes only comparison outputs. Data loading, plotting and metric libraries are peripheral support; the pretrained forward pass is visible.''')
 code('''# PROVIDED — imports and bounded CPU runtime
import os,sys,hashlib,importlib.metadata,inspect,json,math,time,types,urllib.request
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss
from IPython.display import display
import pandas as pd
for directory in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (directory/'relkit').is_dir():ROOT=directory.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start Jupyter in labs')
torch.set_num_threads(1)
output=Path(os.environ.get('L066_OUTPUT',str(ROOT/'data/cache/l066-student')))
output.mkdir(parents=True,exist_ok=True)''')
 from relkit import tabicl_l066_v2 as core
 code('# PROVIDED — immutable checkpoint and declared numeric procedure\n'+'\n'.join(k+'='+repr(getattr(core,k)) for k in ['CHECKPOINT_SHA','CHECKPOINT_URL','CHECKPOINT_FILE','MODEL_CONFIG','PROTOCOL','PRESETS66']))
 checks={
 'attention_mix':"q=torch.tensor([[1.,0.],[0.,1.]])\nk=torch.tensor([[-1.,0.],[0.,1.],[2.,1.]])\na=attention_mix(q,k,k)\ntorch.testing.assert_close(a,torch.tensor([[1.3794,.9121],[.6044,.8022]]),atol=6e-5,rtol=0)\nassert attention_mix(q[None,None],k[None,None],k[None,None]).shape==(1,1,2,2)",
 'context_keys':"k=torch.arange(2*3*5*4.).reshape(2,3,5,4);v=k+1000\na,b=context_keys(k,v,3)\nassert torch.equal(a,k[:,:,:3]) and torch.equal(b,v[:,:,:3])\nassert a.shape==(2,3,3,4)",
 'rotary_pairs':"x=torch.tensor([[[1.,0.,1.,0.],[1.,0.,1.,0.],[1.,0.,1.,0.]]]);freq=torch.tensor([1.,.1])\na=rotary_pairs(x,freq)\ntorch.testing.assert_close(a[0,1],torch.tensor([math.cos(1),math.sin(1),math.cos(.1),math.sin(.1)]))\ntorch.testing.assert_close((a*a).sum(-1),(x*x).sum(-1))",
 'inducing_memory':"u=torch.tensor([[[-1.,0.],[0.,1.],[2.,1.],[1.,-1.]]]);ind=torch.eye(2);calls=[]\ndef block_probe(q,k):\n    calls.append((q.shape[-2],k.shape[-2]));return attention_mix(q,k,k)\na=inducing_memory(u,ind,3,block_probe,block_probe)\nassert calls==[(2,3),(4,2)], 'First memory has only context; second has inducing keys'\ntorch.testing.assert_close(a[0,-1],torch.tensor([1.0814,.8698]),atol=6e-5,rtol=0)\nv=u.clone();v[:,-1]=999\ntorch.testing.assert_close(a[:,:3],inducing_memory(v,ind,3,block_probe,block_probe)[:,:3])",
 'conditional_affine':"x=torch.tensor([[[2.],[3.]]]);h=torch.tensor([[[1.,-1.],[2.,0.]]]);w=lambda z:z*.5;b=lambda z:z+.1\nwn=lambda z:z+2;bn=lambda z:z-1\na=conditional_affine(x,h,w,b,wn,bn)\ntorch.testing.assert_close(a,x*wn(w(h))+bn(b(h)))\nassert a.shape==(1,2,2)"}
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Implement this live TabICL operation")'))
  code('# CHECK — '+name+'\n'+checks[name]+'\nprint("CHECK passed: '+name+'")')
 for title,names,why in [
 ('Complete attention blocks',['PackedAttention','AttentionBlock','InducedBlock'],'Follow the packed Q/K/V matrices into your attention_mix, context_keys and rotary_pairs. InducedBlock passes two complete pre-LN residual blocks into your inducing_memory.'),
 ('Column and row representations',['RotaryFrequencies','Encoder','SkipLinear','ColumnEmbedding','RowInteraction'],'Three column blocks lead to your normalized conditional_affine. Three row blocks prepend all four CLS tokens and concatenate their normalized outputs. The source sentinel -100 path is preserved; the wrapper supports finite numeric arrays.'),
 ('Dataset ICL and complete model',['ICLPredictor','TabICL'],'Only the ICL predictor encodes labels. Its twelve full blocks and complete native head run on the original checkpoint. Context keys exclude every query row. No packaged model replaces this code.'),
 ('Strict weights and wrapper',['ensure_checkpoint','load_pretrained','fit_numeric','transform_numeric','predict_numeric','load_dataset'],'Loading verifies the immutable checkpoint checksum and exact configuration, then strictly loads every state tensor. The wrapper fits its constant filter, standardizer and outlier bounds on context only.'),
 ('Live measurement identities',['model_digest','stable_value','model_runtime_identity','kernel_identity'],'The fingerprint follows global helpers in nested CodeType objects as well as direct calls, methods and function-valued defaults. The independent weight digest hashes actual loaded tensors, including fixed RoPE frequencies.'),
 ('The measured context experiment',['run_experiment'],'Read the split and label-blind permutation before executing. Your five functions are called by the live model stages. All rows are used in the largest context/query partition; query IDs stay fixed while context grows.')]:
  md('## PROVIDED · '+title+'\n\n'+why)
  for name in names:code('# PROVIDED — '+name+'\n'+piece(name))
 md('''## CHECK · Whole-stage source correspondence and query isolation
The checker prepares checksum-pinned reference dependencies in an isolated cache and runs the vendored 0.1.4 package in a subprocess. It compares your complete column embeddings, row representations and logits on three fixtures, then the one-view wrapper including a constant column and an extreme value. The reference's predictions never become your experiment's predictions.''')
 code('''# CHECK — bind source parity to this actual notebook namespace
from _check_l066_v2 import check as source_check
before=kernel_identity(globals(),ROOT)
parity=source_check(globals(),save=False)
assert parity['kernel_identity']['sha256']==before['sha256']==kernel_identity(globals(),ROOT)['sha256']
model=load_pretrained(ensure_checkpoint(ROOT))
assert model_digest(model)==parity['weights_sha256']
assert model_runtime_identity(model)==parity['runtime_sha256']
print('Whole model/source CHECK:',parity['status'])
print('Max stage absolute error:',max(x['max_abs'] for x in parity['errors']))''')
 md('''## CHECK · Changing hidden dependencies invalidates old evidence
A helper called inside a generator expression must change identity too. This deliberate temporary edit is restored before measurement. A successful stale-evidence rejection is different from a numerical source check.''')
 code('''# CHECK — recursive bytecode dependency, not only outer co_names
original_run=run_experiment
exec('def nested_probe(x): return x+1\\ndef run_experiment(*args,**kwargs): return sum(nested_probe(x) for x in [1,2])')
a=kernel_identity(globals(),ROOT)['sha256']
exec('def nested_probe(x): return x+2')
assert kernel_identity(globals(),ROOT)['sha256']!=a, 'Missed changed helper inside generator'
run_experiment=original_run
assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
weights=model_digest(model)
parameter=next(model.parameters());saved=parameter.detach().clone()
with torch.no_grad():parameter.add_(.01)
assert model_digest(model)!=weights, 'Missed changed checkpoint weights'
with torch.no_grad():parameter.copy_(saved)
assert model_digest(model)==weights
runtime=model_runtime_identity(model)
old_eps=model.row_interactor.out_ln.eps;model.row_interactor.out_ln.eps=.01
assert model_runtime_identity(model)!=runtime
model.row_interactor.out_ln.eps=old_eps
model.forward=lambda *args,**kwargs:None
try:
    model_runtime_identity(model)
except RuntimeError:print('Instance override correctly rejected')
else:raise AssertionError('Instance override escaped runtime binding')
del model.forward
assert model_runtime_identity(model)==runtime
print('Nested helper, weights, normalization setting and instance override checks passed.')''')
 md('''## Run · Predict, then execute your model
Write a prediction before running. The following loop uses all diabetes rows, one split, three nested contexts, the visible complete model and your live functions. The author-reference tables earlier describe a larger three-dataset panel. Your current run is a distinct measurement even when its probabilities agree with one author seed.''')
 code('''# PROVIDED — real current-kernel pretrained experiment
assert parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
result=run_experiment(ROOT,model,PRESETS66['lab'],globals())
assert result['kernel_identity']['sha256']==parity['kernel_identity']['sha256']
display(pd.DataFrame([{k:r[k] for k in ['dataset','seed','fraction','accuracy','log_loss','column_seconds','row_seconds','icl_seconds']} for r in result['records']]))
for r in result['records']:
    assert not set(r['context_ids'])&set(r['query_ids'])
print('Prediction arrays and original row IDs retained in result.')''')
 md('''## CHECK · Numerical correspondence to the module measurement
Module and notebook compilation can produce different bytecode identities. Keep both identities distinct. Compare probabilities for the same dataset/split/context membership instead of copying the author's identity into your live result. The live source-check identity must still equal your live measurement identity.''')
 code('''# CHECK — same numerical experiment, separately named implementation identities
author=json.loads((ROOT/'_verify_l066_v2_results.json').read_text())
for r in result['records']:
    a=next(a for a in author['records'] if (a['dataset'],a['seed'],a['fraction'])==(r['dataset'],r['seed'],r['fraction']))
    assert a['context_ids']==r['context_ids'] and a['query_ids']==r['query_ids']
    np.testing.assert_allclose(a['probabilities'],r['probabilities'],atol=2e-5,rtol=2e-5)
print('Live namespace identity:',result['kernel_identity']['sha256'])
print('Author module identity:',author['kernel_identity']['sha256'])
print('Live/reference probability correspondence passed.')''')
 md('''## EXIT TICKET · Submit a measured explanation
Explain the direction of one paired context change and whether accuracy and log loss agree. Name all three query-isolation boundaries and one wrapper caveat. Distinguish full-model source parity, this fresh inference measurement and original pretraining/TALENT/500K claims. Include one unexpected result and the next experiment it motivates. The length check requires an attempt; the tutor assesses your reasoning.''')
 verdict='On diabetes seed 7, growing context from 76 to 614 lowered measured log loss from 0.49448 to 0.44735 and raised accuracy from 0.75974 to 0.79221. The middle context also improved both here. The author WDBC panel surprised me: middle context lowered mean accuracy despite lowering loss. Additional context changes the wrapper statistics and column summaries as well as available labels. Column memories exclude queries, row attention stays within each row, and ICL keys are context-only. The none-view wrapper fits context statistics; the optional power fallback can couple queries. My next experiment would freeze preprocessing statistics while adding labels to separate those contributions. Copied-weight checks and this original-checkpoint run do not reproduce pretraining, the TALENT 32-view benchmark or 500K memory behavior.'
 code('# TODO — written interpretation, assessed by the tutor\ninterpretation='+repr(verdict if solution else '')+'\nassert len(interpretation.split())>=50, "Explain your measured result, boundaries and unrun claims"')
 code('''# EXIT — reject stale functions/weights, save fresh measurements without overwrite
assert result['kernel_identity']['sha256']==parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
assert result['weights_sha256']==parity['weights_sha256']==model_digest(model)
assert result['runtime_sha256']==parity['runtime_sha256']==model_runtime_identity(model)
assert parity['reference_worker_sha256']==hashlib.sha256((ROOT/'_reference_l066_v2.py').read_bytes()).hexdigest()
assert parity['source_manifest_sha256']==hashlib.sha256((ROOT/'_sources_l066_v2.json').read_bytes()).hexdigest()
result['source_check']=parity;result['interpretation']=interpretation
path=output/'exit-v2.json'
with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print('EXIT saved:',path)
print('Artifact SHA256:',hashlib.sha256(path.read_bytes()).hexdigest())
print('Paper reproduction:',result['paper_reproduction'])''')
 md('''## NEXT STEP · Broader local replication with this same implementation
Enable the gate to repeat the original model on all three complete datasets and all three seeds. This reproduces the declared local panel. The full 188-task 32-view paper benchmark, many-class hierarchy and 500K offloaded runtime are separate work and remain NOT_RUN. The supported closer operator is also available as `_verify_l066_v2.py --preset closer --output YOUR_FRESH_PATH.json`; the paper preset fails explicitly rather than relabeling a smaller run.''')
 code('''# PROVIDED — optional local/Colab run, OFF until you choose to run it
RUN_BROADER=False
if RUN_BROADER:
    assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
    larger=run_experiment(ROOT,model,PRESETS66['closer'],globals())
    with (output/f'closer-{time.time_ns()}.json').open('x') as stream:json.dump(larger,stream,indent=2,allow_nan=False)
else:print('Broader current-kernel run NOT_RUN; original pretraining and full paper benchmark NOT_RUN.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:v2:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
  elif key=='href' and tag[key].startswith('#'):tag[key]=unquote(tag[key])
 for table in soup.find_all('table'):table.wrap(soup.new_tag('div',attrs={'style':'max-width:100%;overflow-x:auto','role':'region','tabindex':'0','aria-label':'Scrollable evidence table'}))
 style=soup.new_tag('style');style.string='th,td{white-space:nowrap;word-break:normal;overflow-wrap:normal}';soup.head.append(style)
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'));soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 body=''
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:
   body+=figure(m[1])
   if m[1] in ('inducing','rope','cost'):body+='<div class="l066-widget" id="l066-'+m[1]+'-viz"><p>Interactive controls require JavaScript; the figure retains the baseline trace.</p></div>'
  elif p=='<!--results-table-->':body+=markdown2html_mistune(table_markdown())
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 066 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l066-tabicl'])+'</head><body class="l066"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l066-tabicl-viz'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0065-tabpfn-query-embeddings.html">← Lesson 065</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 066</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 066</a></nav><h1>TabICL: complete forward and evidence card</h1>'+launcher()+figure('architecture')+markdown2html_mistune('''
## Forward trace

1. Finite numeric context fits nonconstant feature filter, std+1e-6 scaling/clipping and two-pass outlier bounds. Optional second normalizer: none.
2. Shared scalar → 128 coordinates. Three ISABs; each has 128 inducing readers and two complete four-head pre-LN attention/FFN blocks.
3. First MAB reads context tokens only; second lets all cell tokens read inducing memories. Independent heads generate normalized W and B; E=xW+B.
4. Four CLS tokens prepend each row. Three eight-head blocks apply adjacent-pair Q/K RoPE, base 100000, head width 16. Normalize and concatenate four CLSs → 512.
5. Context adds Linear(one-hot(y)); query receives no label embedding. Twelve four-head pre-LN ICL blocks use only context keys and values.
6. Final LN → Linear 512→1024 → GELU → Linear 1024→10. Query rows, first K logits, temperature 0.9 softmax.

## Boundaries and costs

Column memory excludes queries; row attention stays within a row; ICL keys exclude queries. Own query features still enter queries/residuals. The checked none-view wrapper fits context only. Optional power-view fallback can couple query preprocessing; inspect the executed counterexample.

With N=C+Q, F features and m=128, conceptual scores: column 12Fm(C+N); row 24N(F+4)²; ICL 48NC. The final term includes C². Full eager score matrices are visible here; original FlashAttention/batching/offload are not implemented.

## Version and evidence

The active February v1-0208 checkpoint is the paper checkpoint, SHA f5bae1d31181a1bb4ab8e97d2a5e62a504e856f23b4ea62c7fcc2f8eec4995b6. Later v1.1-0506 results remain historical. Whole model parity does not reproduce pretraining or benchmark scores.

[Paper §§3–5](https://arxiv.org/html/2502.05564v1) · [Original sources/checkpoint pins](../labs/_sources_l066_v2.json) · [Full-stage checks](../labs/_check_l066_v2_results.json) · [Fresh predictions](../labs/_verify_l066_v2_results.json) · [Statistics](../labs/_analysis_l066_v2_results.json) · [RoPE intervention](../labs/_symmetry_l066_results.json) · [Wrapper coupling](../labs/_query_coupling_l066_results.json) · [Reproduction instructions](../labs/l066-reproduction.md).

Local full data: diabetes/blood transfusion/WDBC, seeds 7/17/27, 20% fixed queries, nested 12.5/37.5/100% context prefixes, one view. Original 64/16/20 and 32-view TALENT study, synthetic pretraining, many-class hierarchy and 500K offload test: NOT_RUN. The lab is INCOMPARABLE as a paper-result reproduction.
''')+markdown2html_mistune(table_markdown())
 reference=BeautifulSoup(ref+'</article></body></html>','html.parser')
 for table in reference.find_all('table'):table.wrap(reference.new_tag('div',attrs={'class':'table-scroll','role':'region','tabindex':'0','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(str(reference))
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built complete L066 original-checkpoint package')
if __name__=='__main__':build_package()
