"""Canonical scoped L064 package: full historical weights, live operators and evidence."""
import ast,base64,hashlib,html,json,re
from pathlib import Path
import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.filters.markdown import markdown2html_mistune
from bs4 import BeautifulSoup
from _colab import bootstrap_cells
from _check_l064_v2 import CHECKS
ROOT=Path(__file__).resolve().parent;SLUG='0064-tabpfn-v2';TITLE='Trace historical TabPFN v2 through a complete prediction'
TASKS=[
('group_features','def group_features(x,group_size=2):','Keep row identities while forming pairs','Pad only the feature axis and return B×N×ceil(F/g)×g. Validate the input rank and positive group size.','The same shape can conceal a scramble of row and feature identities.','The padding count is the distance to the next multiple of g.'),
('encode_targets','def encode_targets(y,total_rows):','Represent an unknown target honestly','Return B×N×2 target-rank and missing-flag inputs, given observed B×C labels only.','The released query token is not an all-zero embedding or the true held-out label.','Append NaNs; use impute_with_flags; compare each filled value with the distinct context class values.'),
('attention_mix','def attention_mix(q,k,v):','Turn context scores into a value mixture','Compute rectangular scaled dot-product attention over sender positions, allowing a single K/V head to broadcast.','The softmax axis determines which information is combined; a correct shape does not validate that conditional distribution.','For historical CPU parity the scale is torch.sqrt(torch.tensor(1.0/q.shape[-1])).to(q.device).'),
('row_attention','def row_attention(h,n,attention):','Implement both row routes','Transpose B×N×(G+1)×D into group sequences; compute context MHA and query first-head-KV MQA; return original layout.','Using ordinary MHA for queries is a different pretrained network.','Call attention(receiver,sender,first_kv=True) only for query receivers. Both calls use context senders.'),
('postnorm_update','def postnorm_update(h,update):','Put the residual inside normalization','Return non-affine LayerNorm of the incoming token plus its sublayer update, with epsilon 1e-5.','The actual block has three of these operations; moving a norm changes all later layer inputs.','Normalize the last D coordinates of each token, not context rows.')]
CAPTIONS={
'architecture':'Actual historical default classifier: 12 distinct 192-wide blocks, two-feature groups and a separate target token. Context and query receivers use different K/V-head routes. Every learned parameter is loaded; synthetic pretraining is outside this inference path.',
'encoding':'Synthetic two-feature trace: impute by context means, then compute sample z-scores. The missing context value has a zero normalized value and flag −2. Both varying channels give factor sqrt(2/2)=1.',
'attention':'Synthetic two-head comparison isolates K/V sharing. Query projections remain distinct; first-head reuse is applied only to query receivers. The fixed scores illustrate the value contribution; real key reuse also changes scores.',
'cost':'Derived receiver–sender pair counts per head and per layer. Projection costs, kernel memory optimization and cache construction are not represented by these counts.',
'boundary':'The all-row active-value count can change encoded context and existing queries before masked attention. The no-missingness control lets wrapper constant filtering remove the channel.',
'results':'Fresh author reference: full datasets, three overlapping stratified 50/50 splits, observed versus shuffled context labels. All paired measured losses are shown on one scale; these are not confidence intervals over new datasets.'}

def piece(name):
 source=(ROOT/'relkit/tabpfn_l064_v2.py').read_text();return next(ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==name)
def parts():return re.split(r'(<!--figure:\w+-->)',(ROOT.parent/'lessons/content'/f'{SLUG}.md').read_text())
def figure(name,notebook=False):
 path='figures/l064/'+name+'-v2.png';caption=CAPTIONS[name]
 if notebook:return '<div style="max-width:100%;overflow-x:auto" role="region" aria-label="Scrollable computational figure"><img alt="'+html.escape(caption)+'" src="data:image/png;base64,'+base64.b64encode((ROOT/path).read_bytes()).decode()+'" style="width:100%;min-width:720px;max-width:none;height:auto"></div>\n\n'+caption+' [Full-size figure]('+path+').'
 return '<figure id="'+('l064-architecture' if name=='architecture' else 'l064-figure-'+name)+'"><div class="figure-wide" tabindex="0" role="region" aria-label="Scrollable computational figure"><img src="../labs/'+path+'" alt="'+html.escape(caption)+'"></div><figcaption>'+caption+' <a href="../labs/'+path+'">Full-size figure</a>.</figcaption></figure>'
def launcher(prepared=False):
 prefix='../' if prepared else '../labs/';links=[('https://colab.research.google.com/github/Avistian/relational/blob/main/labs/'+SLUG+'.ipynb','Run in Colab'),(prefix+SLUG+'.ipynb','Download notebook'),('#lab-exercises' if prepared else '../labs/html/'+SLUG+'.html#lab-exercises','Jump to exercises'),(('../../' if prepared else '../')+'reference/'+SLUG+'.html','Reference'),(prefix+'_verify_l064_v2_results.json','Measured evidence'),(prefix+'l064-reproduction.md','Reproduction instructions')]
 if not prepared:links.insert(0,('../labs/html/'+SLUG+'.html','Read-only lab preview'))
 return '<aside class="lab-access" data-lab-launch><p><strong>Lab 064 · five live tasks + real pretrained inference + EXIT</strong></p><nav class="lab-access-links">'+''.join('<a href="'+u+'"'+(' download' if title=='Download notebook' else '')+'>'+title+'</a>' for u,title in links)+'</nav><p>The preview is read-only. Complete the runnable notebook in Colab or Jupyter.</p></aside>'

def build(solution=False):
 cells=[]
 def md(s):cells.append(nbf.v4.new_markdown_cell(s.strip()))
 def code(s):cells.append(nbf.v4.new_code_cell(s.strip()))
 md(f'''# Lab 064 · {TITLE}

[Lesson](../lessons/{SLUG}.html) · [Reference](../reference/{SLUG}.html) · [Author measurement](_verify_l064_v2_results.json) · [Source inventory](_sources_l064_v2.json) · [Reproduction contract](l064-reproduction.md)

**Skill:** trace actual pretrained v2 inference and diagnose the information boundary of its complete wrapper. Five live TODOs drive all 12 copied-weight layers. **Built with TabPFN** — [Prior Labs License](sources/foundation/v2-LICENSE). **Scope:** full default-classifier architecture; explicitly simplified one-view numeric recipe; separate original-default four-view connection check. This is CPU inference with frozen weights, not synthetic pretraining or a reproduction of the complete Nature benchmark. The fresh student run uses all 768 diabetes rows, a stratified 50/50 split (seed 7) and a shuffled-context-label control. Full author-reference results remain separate from your kernel. EXIT requires measured results, a counterexample, source checks and your interpretation.''')
 for c in bootstrap_cells():cells.append(nbf.v4.new_markdown_cell(c['source']) if c['cell_type']=='markdown' else nbf.v4.new_code_cell(c['source']))
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:md(figure(m[1],True))
  elif part.strip():md(part.replace('../labs/',''))
 md('''## Start the live implementation
The code below contains the full model and numeric inference recipe, split into coherent chunks. The class methods resolve their helper names in **this notebook**. Source checks construct your visible class, not a hidden completed model. The source worker runs the pinned historical package in an isolated subprocess because current sklearn is incompatible with that old release; it supplies comparison outputs only.''')
 code('''# PROVIDED — imports, local paths and reproducible CPU execution
import hashlib,importlib.metadata,inspect,json,math,time,functools,types,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from IPython.display import display
for path in [Path.cwd(),Path.cwd()/'labs',Path.cwd().parent]:
    if (path/'relkit').is_dir():ROOT=path.resolve();sys.path.insert(0,str(ROOT));break
else:raise RuntimeError('Run bootstrap or start Jupyter in labs')
output=ROOT/'data/cache/l064-student';output.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
VERSIONS={k:importlib.metadata.version(k) for k in ['torch','numpy','scikit-learn','scipy','pandas']}
print(VERSIONS)''')
 from relkit import tabpfn_l064_v2 as core
 code('# PROVIDED — immutable model identity and explicit recipe\n'+'\n'.join(name+'='+repr(getattr(core,name)) for name in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','RECIPE','PRESETS']))
 for i,(name,sig,title,goal,why,hint) in enumerate(TASKS,1):
  if i==2:
   md('''### PROVIDED · Missing-value arithmetic
impute_with_flags counts valid context values, replaces unknown entries by context means and retains a distinct missing flag. The same helper serves feature and target encoders. Read the feature path alongside the worked two-column trace above. The historical active-group count uses all supplied rows after normalization; reproducing it preserves the source counterexample.''')
   for item in ['impute_with_flags','soften_outliers','encode_groups']:code('# PROVIDED — '+item+'\n'+piece(item))
  if i==4:
   md('''### PROVIDED · Packed learned projections
Each of the six heads has its own query/key/value matrices. The packed parameter has shape 3×6×32×192. The first_kv flag slices only the key/value head axis to length one; broadcasting inside your attention_mix supplies it to all query heads. The output projection returns 192 coordinates. No original package model is called here.''')
   code('# PROVIDED — six learned projection heads\n'+piece('PackedAttention'))
  md(f'## TODO {i} · {title}\n\n**Goal:** {goal}\n\n**Why:** {why}\n\n**Hint boundary:** {hint}')
  code('# TODO — '+name+'\n'+(piece(name) if solution else sig+'\n    raise NotImplementedError("Complete this live v2 operation")'))
  code(CHECKS[name])
 md('''## PROVIDED · The complete repeated block and classifier
A feature-attention call receives B×N×(G+1)×192 and mixes the penultimate axis independently for every row. Your row_attention moves rows into that sequence axis and applies its two receiver routes. Three calls to your postnorm_update connect those operations and the GELU feed-forward map. TabPFNv2 then builds twelve distinct blocks, selects the final query target states and applies the full head. The target tokens come from your encode_targets, and grouping comes from your group_features.''')
 for item in ['V2Block','TabPFNv2']:code('# PROVIDED — '+item+'\n'+piece(item))
 md('''## PROVIDED · Load every released parameter, then connect the wrapper
The checkpoint SHA is checked before deserialization. Strict mapping loads all 81 tensors and rejects a shape/key mismatch. The wrapper performs the same context-based raw equality constant filter as the named source configuration; its treatment of partly missing constants is deliberately retained. Labels are encoded from context only. The experiment passes its actual model instance into every prediction and saves raw query logits for reconstruction.''')
 for item in ['load_pretrained','ensure_checkpoint','predict_numeric','load_dataset','run_experiment']:code('# PROVIDED — '+item+'\n'+piece(item))
 md('''## CHECK · Entire copied-weight computation against the original release
The checker creates your live model and compares complete outputs, input gradients and all 81 parameter gradients with the verified historical wheel. Five whole numeric-wrapper cases cover binary/multiclass data, constants, missingness and extremes. An actual default four-view wrapper supplies transformed tables; your model recomputes every network output including its outlier transform. That bridge borrows peripheral preprocessing and therefore is not a from-scratch reimplementation of every default transform. No worker output substitutes for your experiment's predictions.''')
 code('''# CHECK — validate the actual current notebook functions
from _check_l064_v2 import check as validate_live_v2
parity_before=capture_kernel_identity()['sha256']
parity=validate_live_v2(globals(),save=False)
parity_after=capture_kernel_identity()['sha256']
assert parity_before==parity_after==kernel_identity['sha256']
parity['kernel_sha256']=parity_after
print({k:v for k,v in parity.items() if k not in ['counterexample']})''')
 parity_cells=cells[-2:];del cells[-2:]  # Execute parity only after identity utilities exist.
 names=[n.name for n in ast.parse((ROOT/'relkit/tabpfn_l064_v2.py').read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
 md('''## PROVIDED · Record the current executable identity
The fingerprint covers semantic Python bytecode and nested code constants, every visible model method and helper, positional/keyword defaults, closure values, the configuration and recipe, versions, and the actual data-loader dependency source. It is not a hash of a file that may differ from your edited kernel. There is no resume path. We also hash all live model tensors so post-measurement weight edits cannot pass EXIT.''')
 code('''# PROVIDED — semantic executable content, including nested helper code
KERNEL_NAMES='''+repr(names)+'''
def stable_value(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if value is Ellipsis:return {'ellipsis':True}
    if inspect.isclass(value):return {'class_reference':value.__module__+'.'+value.__qualname__}
    if isinstance(value,(set,frozenset)):return {'set':sorted([stable_value(v) for v in value],key=repr)}
    if isinstance(value,bytes):return value.hex()
    if isinstance(value,(list,tuple)):return [stable_value(v) for v in value]
    if isinstance(value,dict):return {str(k):stable_value(v) for k,v in value.items()}
    if isinstance(value,types.CodeType):
        return {'bytecode':value.co_code.hex(),'constants':stable_value(value.co_consts),'names':value.co_names,'varnames':value.co_varnames,'freevars':value.co_freevars,'cellvars':value.co_cellvars,'argcount':value.co_argcount,'posonly':value.co_posonlyargcount,'kwonly':value.co_kwonlyargcount,'flags':value.co_flags}
    if inspect.isfunction(value):
        f=inspect.unwrap(value)
        return {'code':stable_value(f.__code__),'defaults':stable_value(f.__defaults__),'kwdefaults':stable_value(f.__kwdefaults__),'closure':stable_value([c.cell_contents for c in (f.__closure__ or [])])}
    raise TypeError('Unrecorded executable dependency: '+str(type(value)))
def capture_kernel_identity():
    operators={}
    for name in KERNEL_NAMES:
        obj=inspect.unwrap(globals()[name])
        operators[name]={k:stable_value(v) for k,v in vars(obj).items() if inspect.isfunction(v)} if inspect.isclass(obj) else stable_value(obj)
    from relkit.data import load_tier_a
    record={'operators':operators,'loader':stable_value(load_tier_a),'loader_file_sha256':hashlib.sha256(Path(inspect.getfile(load_tier_a)).read_bytes()).hexdigest(),'checker_sha256':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['_check_l064_v2.py','_reference_l064_v2.py']},'loader_globals':{k:stable_value(load_tier_a.__globals__[k]) for k in ['SPECS']},'loader_cache':str(load_tier_a.__globals__['CACHE']),'constants':{k:stable_value(globals()[k]) for k in ['CHECKPOINT_SHA','CHECKPOINT_URL','MODEL_CONFIG','RECIPE','PRESETS']},'versions':VERSIONS}
    record['sha256']=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return record
def model_digest(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():digest.update(name.encode());digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()
kernel_identity=capture_kernel_identity()
print('Live implementation',kernel_identity['sha256'])''')
 code('''# CHECK — helpers, model methods, defaults and configuration edits must invalidate identity
base=capture_kernel_identity()['sha256']
original_helper=impute_with_flags
def altered_helper(x,n):return original_helper(x,n)[0]*0,original_helper(x,n)[1]
impute_with_flags=altered_helper
assert capture_kernel_identity()['sha256']!=base
impute_with_flags=original_helper
original_method=V2Block.forward
def altered_forward(self,h,n):return original_method(self,h,n)*0
V2Block.forward=altered_forward
assert capture_kernel_identity()['sha256']!=base
V2Block.forward=original_method
original_defaults=group_features.__defaults__;group_features.__defaults__=(3,)
assert capture_kernel_identity()['sha256']!=base
group_features.__defaults__=original_defaults
original_kw=attention_mix.__kwdefaults__;attention_mix.__kwdefaults__={'probe':1}
assert capture_kernel_identity()['sha256']!=base
attention_mix.__kwdefaults__=original_kw
RECIPE['temperature']=1.
assert capture_kernel_identity()['sha256']!=base
RECIPE['temperature']=.9
assert capture_kernel_identity()['sha256']==base
print('Mutation detection passes; originals restored')''')
 cells.extend(parity_cells)
 code('''# CHECK — a post-parity helper edit must invalidate verification
original=impute_with_flags
impute_with_flags=altered_helper
try:
    assert parity['kernel_sha256']==capture_kernel_identity()['sha256']
except AssertionError:
    print('Stale source-check identity rejected as required')
else:
    raise AssertionError('Changed helper incorrectly reused earlier parity')
finally:
    impute_with_flags=original
assert parity['kernel_sha256']==capture_kernel_identity()['sha256']==kernel_identity['sha256']''')
 md('''## Predict, then measure a real context-label intervention
Write your expected direction before running: will shuffling context labels raise held-out log loss? What does a surprise mean? The split is fixed at seed 7, all 768 diabetes rows are included, and no test label is passed to the predictor. In both arms, context labels retain the same counts. Model loading and source comparison are excluded from prediction timers. This cell instruments your five functions and checks that each ran downstream during the real experiment.''')
 code('''# PROVIDED — actual measured inference through the five live tasks
call_counts={name:0 for name in ['group_features','encode_targets','attention_mix','row_attention','postnorm_update']}
def counted(name,fn):
    @functools.wraps(fn)
    def call(*args,**kwargs):call_counts[name]+=1;return fn(*args,**kwargs)
    return call
for name in call_counts:globals()[name]=counted(name,globals()[name])
model,checkpoint_config=load_pretrained(ensure_checkpoint(ROOT),TabPFNv2(**MODEL_CONFIG))
weights_before=model_digest(model);run_config=dict(PRESETS['lab'])
result=run_experiment(ROOT,model,config=run_config)
assert all(v>0 for v in call_counts.values()), 'A live TODO was bypassed'
assert capture_kernel_identity()['sha256']==kernel_identity['sha256']
display(pd.DataFrame([{k:r[k] for k in ['dataset','seed','condition','log_loss','auc','seconds']} for r in result['records']]))
print('Live calls',call_counts)''')
 md('''## CHECK · Reconstruct probabilities and log loss from saved logits
This is independent arithmetic on your current outputs. It checks that class truncation, temperature and the evaluation metric agree with the saved prediction arrays. Averaging a different object or scoring a shifted label axis can otherwise produce plausible numbers.''')
 code('''# CHECK — final probability and score arithmetic
from sklearn.metrics import log_loss
for record in result['records']:
    logits=np.asarray(record['trace']['logits'],dtype=np.float64)/record['trace']['temperature']
    values=np.exp(logits-logits.max(1,keepdims=True));reconstructed=values/values.sum(1,keepdims=True)
    assert np.allclose(reconstructed,record['probabilities'],atol=2e-7)
    assert abs(log_loss(record['targets'],np.asarray(record['probabilities']))-record['log_loss'])<2e-6
print('All measured probabilities and log losses reconstruct')''')
 md('''## Predict, then expose the complete-wrapper boundary
Recreate the partly missing constant-column fixture from the lesson through your own model. The raw wrapper retains that column; imputation makes it constant; appending a query with another value changes the all-row used-feature factor. Then replace the context NaN by 1 as a control. Do not infer the desired answer merely from whether the attention code has a mask.''')
 code('''# PROVIDED — live counterexample and its finite-constant control
cx=np.column_stack([np.linspace(-2,2,12),np.ones(12)]).astype('float32');cx[1,1]=np.nan;cy=(cx[:,0]>0).astype(int)
queries=np.array([[.3,1.],[0.,2.]],dtype='float32')
p_alone,_=predict_numeric(model,cx,cy,queries[:1]);p_joined,_=predict_numeric(model,cx,cy,queries)
query_delta=float((p_alone-p_joined[:1]).abs().max())
assert query_delta>1e-3, 'The historical uncached value-count boundary should be visible'
clean=cx.copy();clean[1,1]=1
control_a,_=predict_numeric(model,clean,cy,queries[:1]);control_b,_=predict_numeric(model,clean,cy,queries)
control_delta=float((control_a-control_b[:1]).abs().max());assert control_delta<3e-5
counterexample={'alone':p_alone.tolist(),'joined_first':p_joined[:1].tolist(),'max_delta':query_delta,'finite_constant_control_delta':control_delta,'context_x':np.where(np.isnan(cx),0,cx).tolist(),'context_missing_mask':np.isnan(cx).tolist(),'context_y':cy.tolist(),'queries':queries.tolist()}
print(counterexample)''')
 md('''## CHECK · Separate the fixed-token information graph
Perturb a later query only after token encoding. All earlier receivers should remain unchanged through every block, because row keys and values come only from context. This control is stronger than rereading a mask: it executes the learned layers. Separately, your observed/shuffled experiment checks that legal context-label information can be useful.''')
 code('''# CHECK — whole repeated stack with fixed initial tokens
with torch.no_grad():
    x=torch.from_numpy(np.concatenate([cx,queries]))[None];n=len(cy)
    groups=group_features(x);features=model.x_encoder(encode_groups(groups,n))
    # Positional values are fixed in both arms; zero is sufficient for this structural control.
    target=model.y_encoder(encode_targets(torch.tensor(cy,dtype=x.dtype)[None],x.shape[1]))
    h=torch.cat([features,target.unsqueeze(2)],2);changed=h.clone();changed[:,-1]+=torch.arange(192.)
    for block in model.blocks:h=block(h,n);changed=block(changed,n)
    fixed_token_delta=float((h[:,:-1]-changed[:,:-1]).abs().max())
assert fixed_token_delta<2e-5
print('Fixed encoded-token stack max earlier-row change:',fixed_token_delta)''')
 md('''## EXIT · Explain the computation and freeze the evidence
Write at least 55 words naming your actual two losses, both counterexample probabilities, the active-group-count cause and a precise paper-result limit. Save a unique run and the latest EXIT file. The gate rejects code/config/default/weight changes since measurement; rerunning only EXIT cannot repair stale evidence. Current author results do not satisfy this gate for your kernel.''')
 interpretation=("f'Observed-label log loss is {result[\"records\"][0][\"log_loss\"]:.4f}, while shuffled-context-label loss is {result[\"records\"][1][\"log_loss\"]:.4f}. The observed association helps on this measured split. The counterexample changes class-1 probability from {float(p_alone[0,1]):.6f} to {float(p_joined[0,1]):.6f}; a partly missing constant survives wrapper filtering and changes the all-row active-group normalization before attention. The finite-constant and fixed-token controls remain unchanged. My live five functions drive all twelve pretrained blocks. No synthetic pretraining or complete Nature benchmark was reproduced; local results are INCOMPARABLE.'" if solution else "''")
 code('''# EXIT — current code and weights must still match the measured computation
interpretation='''+interpretation+'''
assert len(interpretation.split())>=55, 'Explain your actual measurements, mechanism and limits'
assert capture_kernel_identity()['sha256']==kernel_identity['sha256'], 'Code/default/config dependency changed: rerun source checks and measurements'
assert parity['kernel_sha256']==capture_kernel_identity()['sha256'], 'Source checks belong to an earlier kernel: rerun parity and measurements'
assert model_digest(model)==weights_before, 'Model weights changed after inference: rerun'
assert all(v>0 for v in call_counts.values())
result.update(kernel_identity=capture_kernel_identity(),weights_sha256=weights_before,checkpoint_config=checkpoint_config,run_config=run_config,call_counts=dict(call_counts),parity=parity,counterexample=counterexample,fixed_token_stack_delta=fixed_token_delta,interpretation=interpretation)
run_path=output/f'run-{time.time_ns()}.json';run_path.write_text(json.dumps(result,indent=2)+chr(10))
exit_record={'status':'MEASURED_PRETRAINED_INFERENCE','paper_reproduction':'INCOMPARABLE','run_path':str(run_path),'run_sha256':hashlib.sha256(run_path.read_bytes()).hexdigest(),'kernel_sha256':kernel_identity['sha256'],'checkpoint_sha256':CHECKPOINT_SHA,'call_counts':call_counts,'query_probability_delta':query_delta,'interpretation':interpretation}
(output/'student-l064-v2-exit.json').write_text(json.dumps(exit_record,indent=2)+chr(10))
print(json.dumps(exit_record,indent=2))''')
 md('''## NEXT STEP · Broaden the same inference experiment
The gated closer preset uses full diabetes, blood transfusion and WDBC datasets with three stratified 50/50 split seeds. It uses the same live model and both label conditions. This is already the author-reference panel, but a new kernel must produce its own result. It is still a selected numeric subset with a one-view recipe, not the paper's 29-classification/28-regression protocol or four/eight-view defaults. Exact synthetic retraining is unavailable because the original generator was not released. Read the reproduction contract before attempting to reconstruct the full OpenML task roster; a preset name cannot certify equivalence.''')
 code('''# PROVIDED — optional broader local/Colab inference through this live model
RUN_PAPER_REPRO=False
if RUN_PAPER_REPRO:
    assert capture_kernel_identity()['sha256']==kernel_identity['sha256']
    larger=run_experiment(ROOT,model,config=PRESETS['closer'])
    larger['kernel_identity']=capture_kernel_identity()
    path=output/f'closer-{time.time_ns()}.json';path.write_text(json.dumps(larger,indent=2)+chr(10));print(path)
else:print('Broader current-kernel run NOT_RUN; original synthetic pretraining and full benchmark NOT_RUN.')''')
 nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python3','language':'python'},'language_info':{'name':'python','version':'3.12'}})
 for i,c in enumerate(nb.cells):c.id=hashlib.sha256(f'{SLUG}:v2:{i}:{c.cell_type}'.encode()).hexdigest()[:12]
 path=ROOT/('solutions' if solution else '.')/(SLUG+'.ipynb')
 if solution and path.exists():
  old=nbf.read(path,as_version=4);before=[c for c in old.cells if c.cell_type=='code'];after=[c for c in nb.cells if c.cell_type=='code']
  if [c.source for c in before]==[c.source for c in after]:
   for prev,cur in zip(before,after):cur.outputs=prev.outputs;cur.execution_count=prev.execution_count
   if 'execution_verification' in old.metadata:nb.metadata['execution_verification']=old.metadata['execution_verification']
 nbf.write(nb,path);return path

def render_preview():
 page,_=HTMLExporter().from_notebook_node(nbf.read(ROOT/(SLUG+'.ipynb'),as_version=4));soup=BeautifulSoup(page,'html.parser')
 from urllib.parse import unquote,urlsplit
 for tag in soup.select('[id]'):tag['id']=unquote(tag['id'])
 for tag in soup.select('[href],[src]'):
  key='href' if tag.has_attr('href') else 'src';u=urlsplit(tag[key])
  if not u.scheme and not u.netloc and u.path:tag[key]='../'+tag[key]
  elif key=='href' and tag[key].startswith('#'):tag[key]=unquote(tag[key])
 next(h for h in soup.find_all('h2') if h.get_text().startswith('TODO 1')).insert_before(soup.new_tag('span',id='lab-exercises'))
 soup.head.append(soup.new_tag('link',rel='stylesheet',href='../../assets/lab-access.css'));soup.body.insert(0,BeautifulSoup(launcher(True),'html.parser'));(ROOT/'html'/(SLUG+'.html')).write_text(str(soup))

def build_package(notebooks=True,render=True):
 body=''
 for part in parts():
  m=re.fullmatch(r'<!--figure:(\w+)-->',part)
  if m:body+=figure(m[1])+('<div id="l064-cost" class="l064-widget"><p>Interactive dimension controls require JavaScript. The figure above retains the baseline arithmetic.</p></div>' if m[1]=='cost' else '')
  else:body+=markdown2html_mistune(part)
 head=f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Lesson 064 · {TITLE}</title>'+''.join(f'<link rel="stylesheet" href="../assets/{x}.css">' for x in ['lesson','foundation-course','lab-access','l064-tabpfn'])+'</head><body class="l064"><article>'
 opening='<nav><a href="../index.html">Course</a> · <a href="0063-synthetic-scm-prior.html">← Lesson 063</a></nav><p class="mission-tag">Year 2 · Quarter 3 · Lesson 064</p><h1>'+TITLE+'</h1>'+launcher()+'<section id="retrieval"><h2>Retrieve before reading</h2><div id="warmup"></div></section><div id="prediction"></div>'
 scripts=''.join(f'<script src="../assets/{x}.js"></script>' for x in ['retrieval-pool','retrieval-bank','predict','teachback','l064-tabpfn-viz'])
 page=head+opening+body+'<section id="lab"><h2>Run the companion lab</h2>'+launcher()+'<div id="teachback"></div></section></article>'+scripts+'</body></html>'
 soup=BeautifulSoup(page,'html.parser')
 for table in soup.find_all('table'):wrapper=soup.new_tag('div',attrs={'class':'table-scroll','tabindex':'0','role':'region','aria-label':'Scrollable source-to-code evidence table'});table.wrap(wrapper)
 (ROOT.parent/'lessons'/(SLUG+'.html')).write_text(str(soup))
 ref=head+'<p><strong>Built with TabPFN</strong> · <a href="../labs/sources/foundation/v2-LICENSE">Prior Labs License</a></p><nav><a href="../lessons/'+SLUG+'.html">Lesson 064</a></nav><h1>Historical v2: shapes, boundaries and evidence</h1>'+launcher()+figure('architecture')+markdown2html_mistune('''
## Shape and computation card

F features → ceil(F/2) paired groups; add one target token. Hidden shape B×(C+Q)×(G+1)×192. Twelve blocks: within-row feature attention, context-only row attention, GELU 192→768→192 MLP; every sublayer adds its input before non-affine LayerNorm. Six 32-wide heads. Context receivers use all K/V heads; query receivers reuse first-head context K/V with six distinct Q projections. Query target readout → GELU head 192→768→10 → active classes → temperature 0.9 → softmax.

Feature encoder: two normalized values plus two flags →192, no bias. NaN flag−2. Group IDs: random48→learned192. Unknown target: rank of context-mean-filled label plus flag−2; not always zero. Default four-view classification averages probabilities after undoing class permutations. The live experiment deliberately uses a simple one-view numeric recipe.

## Information boundary

A correct attention mask guarantees isolation only with encoded tokens fixed. A partly missing constant context feature can survive raw constant filtering; after imputation, the historical all-row used count can change when an extra query varies that channel. This changes earlier encoded inputs. The finite-constant control and fixed-token stack control isolate the cause. Cached versus uncached parity is not assumed.

## Live operations

'''+ '\n'.join('- `'+t[1]+'` — '+t[3] for t in TASKS)+'''

## Reproduction ledger

Verified: full checkpoint forward/input/all 81 parameter gradients; five complete simple numeric-wrapper cases; actual default four-view network bridge; fresh full-data observed/shuffled inference on three datasets and three split seeds; a complete-wrapper query counterexample. The default bridge borrows peripheral release preprocessing. Not verified: original prior training, full paper benchmark, optimized caching parity, live Colab, deployment (tracked separately).

[New measured predictions](../labs/_verify_l064_v2_results.json) · [Source checks](../labs/_check_l064_v2_results.json) · [Independent release query-boundary audit](../labs/_query_coupling_l064_results.json) · [Sources](../labs/_sources_l064_v2.json) · [Reproduction contract](../labs/l064-reproduction.md).
''')
 (ROOT.parent/'reference'/(SLUG+'.html')).write_text(ref+'</article></body></html>')
 if notebooks:build(False);build(True)
 if render:render_preview()
 print('Built scoped L064 full historical-v2 package')
if __name__=='__main__':build_package()
