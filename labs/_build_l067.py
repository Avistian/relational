"""Canonical complete L067 package: live LoCalPFN tasks, official checks and fresh CPU evidence."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _build_foundation import markdown2html_mistune
ROOT=Path(__file__).resolve().parent
SLUG='0067-local-pfn-retrieval-finetuning'
TITLE='LoCalPFN: retrieve a context, then adapt the inference rule'
TASKS=[
 ('neighbor_ids','def neighbor_ids(memory,queries,k,memory_ids=None,exclude_ids=None):','Eligible exact neighbors','Return Q×k memory indices with deterministic ties and identity exclusion.','A duplicate feature vector is not the same row; this function supplies both training episodes and inference contexts.','Compute squared distances, mask matching original IDs, then select in stable order.'),
 ('episode_indices','def episode_indices(neighbors,context_size,rng):','A context shared by several queries','Shuffle each retrieved neighborhood with the same drawn column permutation, then return context/query indices.','This is the paper’s load-bearing training approximation, not an approximate search engine.','Draw one permutation of neighborhood columns; every row belongs to exactly one side within its episode.'),
 ('local_normalize','def local_normalize(x,context_size,max_features=100):','Match the pretrained local input','Return padded B×(C+Q)×100 model inputs normalized from context rows only.','Wrong variance convention or reversed scaling order changes the full checkpoint’s input.','Pad, transpose to contiguous T×B×100, then use explicit masked sum/count and centered-square reductions. Normalize with epsilon 1e-6, clamp ±100, divide by F/100, transpose back; preserve the source float32 operation order.'),
 ('query_loss','def query_loss(logits,targets,classes):','Supervise only episode queries','Compute one mean cross entropy over all B×Q query targets on the fixed training class axis.','Context labels are inputs; query labels are targets. A context missing a class must not shrink the output.','Preserve K as the last dimension when flattening query positions.'),
 ('validation_choice','def validation_choice(scores):','Keep the earliest best validation state','Return the index of the first maximum finite validation AUC.','Step 0 protects the pretrained solution from a harmful update and ties must have a declared rule.','Validate finite nonempty input and choose using validation only.')]
CAPTIONS={
 'ranks':'Five-arm ranks formed after averaging split seeds within each of three datasets. The approximate global-all/random-k distinction is confined to this tiny panel; ties and only three dataset blocks limit generalization.',
 'architecture':'Complete historical v1 checkpoint inside LoCalPFN: 12 blocks, width 512, four heads, FFN 1024, context-only keys/values, query-only output. Local fine-tuning updates every neural stage; retrieval IDs remain discrete.',
 'normalization':'Illustrative one-coordinate trace: context [0,2,4], query 3, sample mean 2/std 2, active feature count F=4. Query ≈0.5 after local scaling becomes ≈12.5 after 100/F. The pictured active coordinate is unaffected by initial zero padding; the source pads before its contiguous masked reductions.',
 'episodes':'Illustrative shared neighborhood around anchor 0. Its query −2 would retrieve a different context at exact inference. Original anchor identity is excluded from its training neighborhood.',
 'cost':'Illustrative attention-score count, R=32/k=100/B=2: 323200 exact versus 23200 shared per head per layer. These are permitted-pair counts for rectangular attention, not original square-mask tensor sizes or measured timing.',
 'selection':'Illustrative validation histories. The ring identifies the selected state. Test labels are absent; a nonzero final update can be rejected in favor of step 0.',
 'results':'Fresh full-model author panel: all rows of three numeric datasets, three split seeds, five arms, dynamic paper k, 30 steps, 16 queries/context. Error bars are sample SD, not dataset-population uncertainty.'}
def source(name):
 for path in [ROOT/'relkit/localpfn_l067_v2.py',ROOT/'relkit/tabpfn_l062_v2.py']:
  s=path.read_text()
  for node in ast.parse(s).body:
   if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name:return ast.get_source_segment(s,node)
 raise KeyError(name)
def parts():return re.split(r'(<!--figure:\w+-->|<!--results-table-->|<!--analysis-text-->)',(ROOT.parent/'lessons/content'/(SLUG+'.md')).read_text())
def table_markdown():
 p=ROOT/'_analysis_l067_v2_results.json'
 if not p.exists():return 'Fresh full panel RUNNING; no result is inferred from an unfinished run.'
 s=json.loads(p.read_text());lines=['| Dataset | Arm | AUC mean ± SD | Log loss mean ± SD |','|---|---|---:|---:|']
 for r in s['summary']:lines.append(f"| {r['dataset']} | {r['arm']} | {r['auc_mean']:.4f} ± {r['auc_sd']:.4f} | {r['loss_mean']:.4f} ± {r['loss_sd']:.4f} |")
 return '\n'.join(lines)
def analysis_text():
 p=ROOT/'_analysis_l067_v2_results.json'
 return json.loads(p.read_text())['interpretation'] if p.exists() else 'Fresh full panel RUNNING; interpretation follows completed measurements.'
def figure(name,notebook=False):
 p='figures/l067/'+name+'-v2.png';caption=CAPTIONS[name];minimum=1100 if name=='results' else 850 if name=='architecture' else 750
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/p).read_bytes()).decode()+'" style="width:100%;min-width:'+str(minimum)+'px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+p+').'
 return '<figure id="l067-'+('architecture' if name=='architecture' else 'figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img style="min-width:'+str(minimum)+'px" src="../labs/'+p+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+p+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 p='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(p+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(p+'_verify_l067_v2_results.json','Measured evidence'),(p+'l067-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 067 · five live local adaptation tasks + complete pretrained model + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'">'+t+'</a>' for u,t in links)+'</nav><p>The preview is read-only. Run the notebook in Colab or Jupyter.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f"""# Lab 067 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Measured evidence](_verify_l067_v2_results.json) · [Reproduction](l067-reproduction.md)

**Skill:** implement five live functions that retrieve, construct, normalize, supervise and select local episodes around a complete pretrained TabPFN. **PROVIDED** cells expose the model and training loop. **TODO** cells contain your live blanks; **CHECK** validates their behavior against independent source; **EXIT** saves an actual measured interpretation.

Default CPU experiment: all 768 diabetes rows, stratified 80/10/10 split seed 7, paper dynamic k=247, two contexts with 16 queries each,30 updates, validation selection between 0 and 30. Two predeclared learning rates .01 and 1e-5 probe the paper/released-CLI discrepancy. Five inference arms share held-out rows. Full 25.8M historical model download≈103MB. The three-dataset author panel is larger than the default notebook. Original pretraining and 95-task 10-fold paper benchmark are NOT_RUN; this run is INCOMPARABLE to paper-result reproduction.""")
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for p in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',p)
  if m:
   if (ROOT/'figures/l067'/(m[1]+'-v2.png')).exists():md(figure(m[1],True))
  elif p=='<!--results-table-->':md(table_markdown())
  elif p=='<!--analysis-text-->':md(analysis_text())
  elif p.strip():md(p.replace('../labs/',''))
 md('## PROVIDED · Runtime and dependencies\n\nRead the definitions before running them. All model and local-algorithm definitions execute in this notebook’s namespace. The official PFN is used only for numerical comparison. Fresh CPU measurements can take several minutes; original checkpoint weights are reused, not retrained from the synthetic prior.')
 code("""# PROVIDED — numerical libraries and fresh output directory
import os,sys,copy,hashlib,inspect,importlib.metadata,json,math,time,types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss,roc_auc_score,accuracy_score
from sklearn.preprocessing import StandardScaler
from IPython.display import display
import pandas as pd
for directory in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (directory/'relkit').is_dir():ROOT=directory.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start Jupyter in labs')
torch.set_num_threads(1)
output=Path(os.environ.get('L067_OUTPUT',str(ROOT/'data/cache/l067-student')));output.mkdir(parents=True,exist_ok=True)""")
 from relkit import localpfn_l067_v2 as core
 code('# PROVIDED — checkpoint and explicit experiment contracts\n'+'\n'.join(k+'='+repr(getattr(core,k)) for k in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','PROTOCOL','PRESETS67']))
 checks={
 'neighbor_ids':"x=np.array([[0.,0.],[0.,0.],[2.,0.],[3.,0.]])\nassert neighbor_ids(x,x[:1],2,[10,11,12,13],[10]).tolist()==[[1,2]], 'Exclude identity, retain duplicate row11'\nassert neighbor_ids(x,x[:1],2).tolist()==[[0,1]], 'Stable tie order'",
 'episode_indices':"neighbors=np.array([[1,2,3,4,5],[5,4,3,2,1]])\nc,q=episode_indices(neighbors,3,np.random.default_rng(4))\nassert c.shape==(2,3) and q.shape==(2,2)\nassert all(not set(a)&set(b) for a,b in zip(c,q))\nassert all(set(a)|set(b)==set(n) for a,b,n in zip(c,q,neighbors))\nassert np.array_equal(np.argsort(np.concatenate([c,q],1)[0]),np.argsort(-np.concatenate([c,q],1)[1]))",
 'local_normalize':"x=torch.tensor([[[0.,1.],[2.,1.],[4.,1.],[3.,5.]]])\nz=local_normalize(x,3,max_features=4)\ntorch.testing.assert_close(z[0,:,0],torch.tensor([-2.,0.,2.,1.]),atol=2e-6,rtol=0)\nassert z.shape==(1,4,4) and z[0,-1,1]==200 and not z[...,2:].any()\ny=x.clone();y[0,-1,0]=999;torch.testing.assert_close(local_normalize(y,3,4)[:,:3],z[:,:3])",
 'query_loss':"z=torch.zeros(2,3,10,requires_grad=True);y=torch.tensor([[0,1,0],[1,1,0]])\nloss=query_loss(z,y,2);torch.testing.assert_close(loss,torch.tensor(math.log(2)))\nloss.backward();assert not z.grad[...,2:].any()\nassert torch.allclose(z.grad[0,0,:2],torch.tensor([-1/12,1/12]))",
 'validation_choice':"assert validation_choice([.7,.7,.6])==0\nassert validation_choice([.7,.8,.8])==1\ntry:validation_choice([float('nan')])\nexcept ValueError:pass\nelse:raise AssertionError('Reject nonfinite validation scores')"}
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(source(name) if solution else sig+'\n    raise NotImplementedError("Implement this live local adaptation operation")'))
  code('# CHECK — '+name+'\n'+checks[name]+'\nprint("CHECK passed: '+name+'")')
 for title,names,why in [
 ('The full original backbone',['attention_mix','postnorm_update','V1Block','TabPFNv1'],'These are the unchanged complete L062 model definitions. Follow context-only K/V into all 12 postnorm blocks and the full 10-logit head. Both feature and label encoders remain trainable.'),
 ('Checkpoint and real data',['load_pretrained','ensure_checkpoint','load_dataset'],'The immutable 103 MB checkpoint is SHA-checked and every model weight strictly loaded. Only the criterion buffer is excluded. Loaders preserve original row IDs in the later split.'),
 ('From retrieval IDs to logits',['episode_logits','predict_contexts','predict_shared','fit_geometry'],'Your local normalization feeds every arm. Exact inference batches distinct contexts with one query each; shared inference processes all queries against one context. Neither function accepts query targets.'),
 ('Runtime and code identity',['model_digest','stable_value','model_runtime_identity','kernel_identity'],'Code identity follows nested bytecode and function-valued defaults; weight and actual runtime identities are separate. Module-reference and notebook identities are not interchangeable.'),
 ('Complete paired adaptation experiment',['run_experiment'],'Read the train-only geometry, precomputed shared episodes, zero-step candidate, 30-step validation gate and final test scoring. Both learning rates start from fresh copies and receive the same episode sequence. The initial weights are restored before returning.')]:
  md('## PROVIDED · '+title+'\n\n'+why)
  for name in names:code('# PROVIDED — '+name+'\n'+source(name))
 md('## CHECK · Bind whole-model source correspondence to this namespace\n\nA local checker downloads the pinned official PFN source if absent and uses it only for comparison. It compares three complete pretrained forward/gradient fixtures, including a one-class context, and a double-precision full AdamW update. Passing does not certify the official sampler: the paper excludes anchors while the released code retains them.')
 code("""# CHECK — complete official model, gradients and optimizer correspondence
from _check_l067_v2 import check as source_check
before=kernel_identity(globals(),ROOT)
parity=source_check(globals(),save=False)
assert parity['kernel_identity']['sha256']==before['sha256']==kernel_identity(globals(),ROOT)['sha256']
model,_=load_pretrained(ensure_checkpoint(ROOT));model.eval()
assert model_digest(model)==parity['initial_weights_sha256']
assert model_runtime_identity(model)==parity['initial_runtime_sha256']
def certify_training_runtime(model):
    for module in model.modules():
        assert not module._backward_hooks and not module._backward_pre_hooks,'Backward hooks invalidate training evidence'
    for parameter in model.parameters():
        assert parameter.requires_grad,'This experiment requires all pretrained parameters to remain trainable'
        assert not parameter._backward_hooks,'Parameter gradient hooks invalidate training evidence'
    return model_runtime_identity(model)
certified_runtime=certify_training_runtime(model)
print('Source correspondence',parity['status'],'AdamW max parameter delta',parity['one_adamw_step_parameter_max_delta'])""")
 md('## CHECK · A changed nested helper must invalidate old evidence\n\nA generator’s body has its own bytecode. This temporary change tests that it is traversed, then restores the actual experiment. The weight and runtime probes are also restored before training.')
 code("""# CHECK — hidden dependency, function default, weight and runtime rejection
original_run=run_experiment
exec('def nested_probe(x): return x+1\\ndef run_experiment(*args,**kwargs): return sum(nested_probe(x) for x in [1,2])')
a=kernel_identity(globals(),ROOT)['sha256']
exec('def nested_probe(x): return x+2')
assert kernel_identity(globals(),ROOT)['sha256']!=a
run_experiment=original_run
assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
old_defaults=local_normalize.__defaults__;local_normalize.__defaults__=(99,)
assert kernel_identity(globals(),ROOT)['sha256']!=parity['kernel_identity']['sha256']
local_normalize.__defaults__=old_defaults
p=next(model.parameters());old=p.detach().clone();weights=model_digest(model)
with torch.no_grad():p.add_(.01)
assert model_digest(model)!=weights
with torch.no_grad():p.copy_(old)
old_eps=model.blocks[0].norm1.eps;model.blocks[0].norm1.eps=.01
assert model_runtime_identity(model)!=certified_runtime
model.blocks[0].norm1.eps=old_eps
p.requires_grad_(False)
try:certify_training_runtime(model)
except AssertionError:print('Frozen parameter rejected')
else:raise AssertionError('Missed frozen pretrained parameter')
p.requires_grad_(True)
handle=p.register_hook(lambda grad:grad)
try:certify_training_runtime(model)
except AssertionError:print('Parameter gradient hook rejected')
else:raise AssertionError('Missed gradient hook')
handle.remove()
assert certify_training_runtime(model)==certified_runtime
print('Nested helper, defaults, weights, normalization, trainable parameters and gradient hook checks passed.')""")
 md('## Run · Predict the validation choice, then measure it\n\nWrite your expected direction before executing. This is the full 768-row diabetes experiment with 30 gradient updates per learning rate. The result retains all neighbor/episode identities and both final and selected weight files. The model is restored to its initial state on return.')
 code("""# PROVIDED — fresh current-kernel experiment driven by your five TODOs
assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
assert certify_training_runtime(model)==certified_runtime
result=run_experiment(ROOT,model,PRESETS67['lab'],globals())
assert certify_training_runtime(model)==certified_runtime
assert model_digest(model)==result['initial_weights_sha256']==parity['initial_weights_sha256']
display(pd.DataFrame([dict(arm=arm,**{k:v[k] for k in ['auc','log_loss','accuracy']},selected_step=v.get('selected_step')) for arm,v in result['records'][0]['arms'].items()]))""")
 md('## CHECK · Same numerical experiment, separately named identities\n\nCompare your actual probabilities with the matching author seed, without copying its code identity. Source certification and your experiment must bind to the same live definitions.')
 code("""# CHECK — current predictions versus author reference, aligned by row IDs
reference=json.loads((ROOT/'_verify_l067_v2_results.json').read_text())
for row in result['records']:
    author=next(a for a in reference['records'] if (a['dataset'],a['seed'])==(row['dataset'],row['seed']))
    assert author['train_ids']==row['train_ids'] and author['test_ids']==row['test_ids']
    assert author['test_context_ids']==row['test_context_ids']
    for arm in row['arms']:
        np.testing.assert_allclose(row['arms'][arm]['probabilities'],author['arms'][arm]['probabilities'],atol=3e-5,rtol=3e-5)
print('Current live identity',result['kernel_identity']['sha256'])
print('Author module identity',reference['kernel_identity']['sha256'])
print('Numerical correspondence passed; identities remain separate.')""")
 md('## EXIT TICKET · A measured explanation\n\nExplain one selected-zero case, why full-model parameter updates can be rejected, how shared episodes differ from exact inference, and why the paper/CLI learning rates are separate arms. Discuss an AUC/log-loss disagreement from the author table and a concrete next experiment. Name original paper work still unrun. The tutor, not a word counter, assesses your explanation.')
 interpretation='On diabetes seed 7, both learning rates retain step 0 after 30 real updates: local frozen and both deployed adaptation arms have test AUC 0.81111. Validation selected the original state despite nonzero final parameter changes, so training was not skipped. In the author panel, blood transfusion seed 7 shows a metric disagreement: random-k raises AUC from global-all 0.83577 to 0.84942 while worsening log loss from 0.42333 to 0.45879. Better ranking need not mean better probability confidence. Shared training gives 16 queries one anchor neighborhood, whereas exact inference retrieves separately for every query. Excluding original anchor IDs preserves distinct duplicate rows. The paper specifies learning rate .01 and the released CLI defaults to 1e-5; these are predeclared arms, not test-selected rates. Only blood seed 7 selects a nonzero released-rate update, and its test AUC then falls from local frozen 0.83674 to 0.82505 despite a validation gain. My next experiment would increase query budget and evaluate more than two validation checkpoints while preserving test freeze. Original synthetic pretraining, the reported 95-task ten-fold benchmark, categorical effects and the GPU runtime study remain unrun; the appendix roster also needs reconciliation.'
 code('# TODO — written interpretation\ninterpretation='+repr(interpretation if solution else '')+'\nassert len(interpretation.split())>=60,"Explain the measured result and its limits"')
 code("""# EXIT — bind actual code, source checker, runtime and original weights; no overwrite
assert result['kernel_identity']['sha256']==parity['kernel_identity']['sha256']==kernel_identity(globals(),ROOT)['sha256']
assert result['initial_weights_sha256']==parity['initial_weights_sha256']==model_digest(model)
assert result['initial_runtime_sha256']==certified_runtime==certify_training_runtime(model)
assert parity['checker_sha256']==hashlib.sha256((ROOT/'_check_l067_v2.py').read_bytes()).hexdigest()
assert parity['source_manifest_sha256']==hashlib.sha256((ROOT/'_sources_l067_v2.json').read_bytes()).hexdigest()
result['source_check']=parity;result['interpretation']=interpretation
path=output/'exit-v2.json'
with path.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print('EXIT saved',path,'SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
print('Paper results',result['paper_reproduction'])""")
 md('## NEXT STEP · Reproduce the broader local panel\n\nThe gate below uses these live definitions on all three numeric datasets and three seeds. It reproduces the declared local panel, not the 95-task paper benchmark. The supported local CPU operator in the reproduction contract uses the same versioned implementation. No cloud job or live Colab run is implied by shipping a command.')
 code("""# PROVIDED — optional broader current-kernel run, OFF by default
RUN_BROADER=False
if RUN_BROADER:
    assert kernel_identity(globals(),ROOT)['sha256']==parity['kernel_identity']['sha256']
    assert certify_training_runtime(model)==certified_runtime
    broader=run_experiment(ROOT,model,PRESETS67['closer'],globals())
    assert certify_training_runtime(model)==certified_runtime
    with (output/f'closer-{time.time_ns()}.json').open('x') as stream:json.dump(broader,stream,indent=2,allow_nan=False)
else:print('Broader current-kernel run NOT_RUN; original paper benchmark NOT_RUN.')""")
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:v2:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for a,b in zip(before,after):b.outputs=a.outputs;b.execution_count=a.execution_count
   if old.metadata.get('execution_verification'):nb.metadata['execution_verification']=old.metadata['execution_verification']
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
   if (ROOT/'figures/l067'/(m[1]+'-v2.png')).exists():body+=figure(m[1])
   if m[1] in ('normalization','cost'):body+='<div class="l067-widget" id="l067-'+('geometry' if m[1]=='normalization' else 'cost')+'-viz"><p>Interactive controls require JavaScript; the static figure retains a worked trace.</p></div>'
  elif p=='<!--results-table-->':body+=markdown2html_mistune(table_markdown())
  elif p=='<!--analysis-text-->':body+=markdown2html_mistune(analysis_text())
  else:body+=markdown2html_mistune(p)
 head='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 067 · '+TITLE+'</title>'+''.join('<link rel="stylesheet" href="../assets/'+x+'.css">' for x in ['lesson','foundation-course','lab-access','l067-localpfn'])+'</head><body class="l067"><article>'
 scripts=''.join('<script src="../assets/'+x+'.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l067-localpfn-viz'])
 page=head+'<nav><a href="../index.html">Course</a> · <a href="0066-tabicl-column-row-attention.html">← Lesson 066</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 067</p><h1>'+TITLE+'</h1>'+launcher()+'<h2>Retrieve before reading</h2><div id="warmup"></div><div id="prediction"></div>'+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for t in soup.find_all('table'):t.wrap(soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 reference=head+'<nav><a href="../lessons/'+SLUG+'.html">Lesson 067</a></nav><h1>LoCalPFN: computation and evidence card</h1>'+launcher()+figure('architecture')+markdown2html_mistune((ROOT/'l067-reference.md').read_text())+markdown2html_mistune(table_markdown())+'</article></body></html>'
 ref=BeautifulSoup(reference,'html.parser')
 for t in ref.find_all('table'):t.wrap(ref.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable evidence table'}))
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(str(ref))
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built complete L067 local adaptation package')
if __name__=='__main__':build_package()
